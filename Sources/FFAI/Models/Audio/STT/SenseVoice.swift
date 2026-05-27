// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SenseVoice — FunAudioLLM's non-autoregressive speech-understanding
// model (ASR + spoken-language-id + emotion + audio-event detection).
//
// Unlike Whisper, SenseVoice has NO autoregressive decoder: it is a
// SAN-M encoder followed by a single CTC projection. One forward pass
// over the audio produces frame-level token log-probabilities; a greedy
// CTC collapse turns those into a transcript.
//
//   waveform ──Kaldi FBANK──▶ ──LFR──▶ ──CMVN──▶ feature frames
//   [query prefix] + frames  ──SAN-M encoder──▶ ──CTC head──▶ log-probs
//   log-probs ──greedy CTC collapse──▶ transcript tokens
//
// A SAN-M block differs from a Whisper encoder block in two ways:
//   * one fused `linear_q_k_v` projection instead of three,
//   * an FSMN memory block — a depthwise (per-channel) 1-D conv over V,
//     added back as `memory + V` — which injects local context the way
//     a relative-position term would.
// Everything else (LayerNorm → MHA → residual, LayerNorm → ReLU MLP →
// residual) mirrors the shared `AudioEncoder` pattern.
//
// The front-end is Kaldi-style (per-frame mean removal, pre-emphasis,
// power-of-two FFT, HTK Mel, plain log) rather than Whisper's
// log-Mel + normalisation, so it does NOT route through
// `Ops.melSpectrogram`; it is a CPU path (`SenseVoiceFrontEnd`) — the
// FBANK cost is dwarfed by the 70-block encoder GEMMs.
//
// Like `WhisperModel`, this file does NOT route through `ModelRegistry`
// / `LanguageModel`; it is loaded via `SenseVoiceModel.load(...)` and
// exposes a `transcribe(...)` surface.

import Foundation
import Metal

// ─── Configuration ───────────────────────────────────────────────────

/// SenseVoice front-end (FBANK + LFR) hyper-parameters, decoded from the
/// `frontend_conf` block of `config.json`.
public struct SenseVoiceFrontEndConfig: Sendable {
    /// Sample rate the model expects (16 kHz).
    public let sampleRate: Int
    /// Number of Mel filterbank bins.
    public let nMels: Int
    /// STFT analysis-window length in milliseconds (25 ms → 400 samples).
    public let frameLengthMS: Int
    /// STFT hop in milliseconds (10 ms → 160 samples).
    public let frameShiftMS: Int
    /// Low-frame-rate stack width — `lfrM` consecutive FBANK frames are
    /// concatenated into one encoder frame.
    public let lfrM: Int
    /// Low-frame-rate stride — the stack advances `lfrN` frames at a time.
    public let lfrN: Int

    public init(
        sampleRate: Int = 16_000, nMels: Int = 80,
        frameLengthMS: Int = 25, frameShiftMS: Int = 10,
        lfrM: Int = 7, lfrN: Int = 6
    ) {
        self.sampleRate = sampleRate
        self.nMels = nMels
        self.frameLengthMS = frameLengthMS
        self.frameShiftMS = frameShiftMS
        self.lfrM = lfrM
        self.lfrN = lfrN
    }

    /// STFT window length in samples.
    public var winLength: Int { sampleRate * frameLengthMS / 1000 }
    /// STFT hop in samples.
    public var hopLength: Int { sampleRate * frameShiftMS / 1000 }
}

/// SenseVoice architecture hyper-parameters, decoded from `config.json`.
public struct SenseVoiceConfig: Sendable {
    /// CTC vocabulary size.
    public let vocab: Int
    /// Per-frame input feature dim after LFR (`nMels * lfrM`).
    public let inputSize: Int
    /// Encoder hidden dim (`output_size`).
    public let hidden: Int
    /// Encoder attention heads.
    public let heads: Int
    /// Feed-forward intermediate dim (`linear_units`).
    public let intermediate: Int
    /// Total SAN-M encoder blocks before the time-pooling stack
    /// (`num_blocks` — block 0 lives in `encoders0`, the rest in
    /// `encoders`).
    public let numBlocks: Int
    /// Time-pooling SAN-M blocks (`tp_blocks`).
    public let tpBlocks: Int
    /// FSMN depthwise-conv kernel size.
    public let fsmnKernel: Int
    /// FSMN shift — extra left padding so the memory block can look
    /// further into the past.
    public let fsmnShift: Int
    /// LayerNorm epsilon.
    public let layerNormEps: Float
    /// The front-end config that pairs with this variant.
    public let frontEnd: SenseVoiceFrontEndConfig

    public init(
        vocab: Int, inputSize: Int, hidden: Int, heads: Int,
        intermediate: Int, numBlocks: Int, tpBlocks: Int,
        fsmnKernel: Int, fsmnShift: Int,
        layerNormEps: Float = 1e-5,
        frontEnd: SenseVoiceFrontEndConfig
    ) {
        self.vocab = vocab
        self.inputSize = inputSize
        self.hidden = hidden
        self.heads = heads
        self.intermediate = intermediate
        self.numBlocks = numBlocks
        self.tpBlocks = tpBlocks
        self.fsmnKernel = fsmnKernel
        self.fsmnShift = fsmnShift
        self.layerNormEps = layerNormEps
        self.frontEnd = frontEnd
    }

    /// Encoder per-head dimension.
    public var headDim: Int { hidden / heads }

    /// Build a `SenseVoiceConfig` from a decoded `config.json`.
    /// SenseVoice nests its encoder hyper-parameters under
    /// `encoder_conf` and the front-end under `frontend_conf`.
    public static func from(_ config: ModelConfig) -> SenseVoiceConfig? {
        let enc = config.nested("encoder_conf")
        let front = config.nested("frontend_conf")
        guard let vocab = config.int("vocab_size") else { return nil }
        // Encoder fields fall back to the published SenseVoiceSmall
        // defaults when the config omits them.
        func encInt(_ key: String, _ fallback: Int) -> Int {
            (enc?[key] as? Int) ?? fallback
        }
        func frontInt(_ key: String, _ fallback: Int) -> Int {
            (front?[key] as? Int) ?? fallback
        }
        let frontEnd = SenseVoiceFrontEndConfig(
            sampleRate: frontInt("fs", 16_000),
            nMels: frontInt("n_mels", 80),
            frameLengthMS: frontInt("frame_length", 25),
            frameShiftMS: frontInt("frame_shift", 10),
            lfrM: frontInt("lfr_m", 7),
            lfrN: frontInt("lfr_n", 6))
        let hidden = encInt("output_size", 512)
        return SenseVoiceConfig(
            vocab: vocab,
            inputSize: config.int("input_size") ?? 560,
            hidden: hidden,
            heads: encInt("attention_heads", 4),
            intermediate: encInt("linear_units", 2048),
            numBlocks: encInt("num_blocks", 50),
            tpBlocks: encInt("tp_blocks", 20),
            fsmnKernel: encInt("kernel_size", 11),
            fsmnShift: encInt("sanm_shift", 0),
            frontEnd: frontEnd)
    }
}

// ─── SAN-M encoder block ─────────────────────────────────────────────

/// One SAN-M encoder block: pre-norm self-attention with an FSMN memory
/// block, then a pre-norm ReLU MLP — each with a residual add. Block 0
/// (`encoders0`) may have `inSize != size`, in which case the attention
/// sub-block does NOT add a residual (the input dim does not match).
public final class SenseVoiceEncoderLayer: Module {
    let norm1, norm2: LayerNorm
    /// Fused `[inSize → 3*hidden]` Q/K/V projection.
    let qkvProj: Linear
    /// Output projection `[hidden → hidden]`.
    let outProj: Linear
    /// FSMN depthwise conv weight `[hidden, 1, kernel]` — one
    /// `[kernel]` filter per channel, no bias.
    let fsmnWeight: Tensor
    let w1, w2: Linear

    let inSize, hidden, heads, headDim, fsmnKernel: Int
    /// Left / right FSMN padding (derived from kernel + shift).
    let fsmnLeftPad, fsmnRightPad: Int
    let scale: Float

    init(
        norm1: LayerNorm, norm2: LayerNorm, qkvProj: Linear,
        outProj: Linear, fsmnWeight: Tensor, w1: Linear, w2: Linear,
        inSize: Int, hidden: Int, heads: Int, fsmnKernel: Int,
        fsmnShift: Int
    ) {
        self.norm1 = norm1
        self.norm2 = norm2
        self.qkvProj = qkvProj
        self.outProj = outProj
        self.fsmnWeight = fsmnWeight
        self.w1 = w1
        self.w2 = w2
        self.inSize = inSize
        self.hidden = hidden
        self.heads = heads
        self.headDim = hidden / heads
        self.fsmnKernel = fsmnKernel
        // FSMN centres the kernel, then `sanm_shift` pushes the window
        // further into the past — matching FunASR's padding math.
        var left = (fsmnKernel - 1) / 2
        if fsmnShift > 0 { left += fsmnShift }
        self.fsmnLeftPad = left
        self.fsmnRightPad = fsmnKernel - 1 - left
        self.scale = 1.0 / Float(Double(hidden / heads).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in norm1.parameters() { out.append(("norm1.\(k)", v)) }
        for (k, v) in norm2.parameters() { out.append(("norm2.\(k)", v)) }
        for (k, v) in qkvProj.parameters() {
            out.append(("self_attn.linear_q_k_v.\(k)", v))
        }
        for (k, v) in outProj.parameters() {
            out.append(("self_attn.linear_out.\(k)", v))
        }
        out.append(("self_attn.fsmn_block.weight", fsmnWeight))
        for (k, v) in w1.parameters() { out.append(("feed_forward.w_1.\(k)", v)) }
        for (k, v) in w2.parameters() { out.append(("feed_forward.w_2.\(k)", v)) }
        return out
    }
}

// ─── SenseVoice model ────────────────────────────────────────────────

/// A loaded SenseVoice model. Holds the SAN-M encoder stack, the
/// query-prefix embedding table, the post-encoder norms and the CTC
/// projection.
public final class SenseVoiceModel: @unchecked Sendable {
    public let config: SenseVoiceConfig

    /// Query-prefix embedding `[16, inputSize]` — language / event /
    /// emotion / text-norm prompt tokens prepended to the FBANK frames.
    let queryEmbed: Tensor
    /// `encoders0` — the first SAN-M block (input dim = `inputSize`).
    let encoders0: [SenseVoiceEncoderLayer]
    /// `encoders` — the remaining `numBlocks - 1` SAN-M blocks.
    let encoders: [SenseVoiceEncoderLayer]
    /// Post-encoder LayerNorm (`after_norm`).
    let afterNorm: LayerNorm
    /// Time-pooling SAN-M blocks (`tp_encoders`).
    let tpEncoders: [SenseVoiceEncoderLayer]
    /// Post-time-pooling LayerNorm (`tp_norm`).
    let tpNorm: LayerNorm
    /// CTC projection `[hidden → vocab]`.
    let ctcProj: Linear
    /// CMVN mean / inverse-std vectors (`[inputSize]` each), or `nil`
    /// when the checkpoint ships no normalisation stats.
    let cmvnMean: [Float]?
    let cmvnInvStd: [Float]?
    let dtype: DType

    /// CTC blank token id — SenseVoice uses 0.
    public static let blankToken = 0
    /// Number of query-prefix rows the encoder prepends (language,
    /// event, emotion, text-norm). The first 4 output frames carry the
    /// rich-info predictions; the transcript starts at frame 4.
    public static let queryPrefixLength = 4

    public init(
        config: SenseVoiceConfig, queryEmbed: Tensor,
        encoders0: [SenseVoiceEncoderLayer],
        encoders: [SenseVoiceEncoderLayer], afterNorm: LayerNorm,
        tpEncoders: [SenseVoiceEncoderLayer], tpNorm: LayerNorm,
        ctcProj: Linear, cmvnMean: [Float]?, cmvnInvStd: [Float]?,
        dtype: DType
    ) {
        self.config = config
        self.queryEmbed = queryEmbed
        self.encoders0 = encoders0
        self.encoders = encoders
        self.afterNorm = afterNorm
        self.tpEncoders = tpEncoders
        self.tpNorm = tpNorm
        self.ctcProj = ctcProj
        self.cmvnMean = cmvnMean
        self.cmvnInvStd = cmvnInvStd
        self.dtype = dtype
    }

    // ─── Public surface ──────────────────────────────────────────────

    /// Run the full SenseVoice pipeline on a 16 kHz mono waveform and
    /// return the CTC frame log-probabilities `[nFrames, vocab]` — the
    /// frame-level distribution `transcribeTokens` collapses.
    public func ctcLogProbs(
        waveform: [Float], language: Int = 0,
        device: Device = .shared
    ) -> Tensor {
        // ── Front-end: FBANK → LFR → CMVN ──
        var feats = SenseVoiceFrontEnd.featureFrames(
            waveform: waveform, cfg: config.frontEnd)
        applyCMVN(&feats)
        let nFeatFrames = feats.count / config.inputSize

        // ── Prepend the query prefix ──
        // FunASR prepends [language, event, emotion] then inserts the
        // text-norm token directly before the FBANK frames:
        //   [lang, event, emo, textnorm, frame_0, frame_1, ...].
        let textNormToken = 15  // "woitn" — no inverse text-norm.
        let prefixIds = [language, 1, 2, textNormToken]
        var rows = [Float]()
        rows.reserveCapacity(
            (prefixIds.count + nFeatFrames)
                * config.inputSize)
        let queryVals = queryEmbed.toFloatArray()
        for id in prefixIds {
            let base = id * config.inputSize
            rows.append(contentsOf: queryVals[base ..< base + config.inputSize])
        }
        rows.append(contentsOf: feats)
        let nRows = prefixIds.count + nFeatFrames

        // ── SAN-M encoder ──
        let h = encode(rows: rows, nRows: nRows, device: device)

        // ── CTC head ──
        let cmd = device.makeCommandBuffer()
        let logits = projectRows(
            ctcProj, h, nRows: nRows,
            outDim: config.vocab, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return logSoftmaxRows(
            logits, nRows: nRows, rowSize: config.vocab,
            device: device)
    }

    /// Transcribe a waveform to CTC token ids. Runs the encoder, takes
    /// the argmax per frame, collapses repeats and drops the blank — the
    /// standard greedy CTC decode. The first `queryPrefixLength` frames
    /// (the rich-info predictions) are skipped.
    public func transcribeTokens(
        waveform: [Float], language: Int = 0,
        device: Device = .shared
    ) -> [Int] {
        let logProbs = ctcLogProbs(
            waveform: waveform, language: language,
            device: device)
        let nFrames = logProbs.shape[0]
        let V = config.vocab
        let vals = logProbs.toFloatArray()

        var collapsed: [Int] = []
        var previous = -1
        // Skip the query-prefix rows — only the transcript frames matter.
        let start = min(Self.queryPrefixLength, nFrames)
        for f in start ..< nFrames {
            var best = 0
            var bestVal = -Float.greatestFiniteMagnitude
            let base = f * V
            for v in 0 ..< V where vals[base + v] > bestVal {
                bestVal = vals[base + v]
                best = v
            }
            if best != previous {
                collapsed.append(best)
                previous = best
            }
        }
        return collapsed.filter { $0 != Self.blankToken }
    }

    // ─── Encoder internals ───────────────────────────────────────────

    /// Run the SAN-M encoder over `[nRows, inputSize]` row-major input
    /// (query prefix + FBANK frames). Returns `[nRows, hidden]`.
    private func encode(
        rows: [Float], nRows: Int,
        device: Device
    ) -> Tensor {
        let inputT = Tensor.empty(
            shape: [nRows, config.inputSize],
            dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(rows, into: inputT)

        // SenseVoice scales the input by sqrt(hidden) then adds a
        // sinusoidal position embedding before the first block.
        let cmd0 = device.makeCommandBuffer()
        let scale = Float(Double(config.hidden).squareRoot())
        var h = scaleRows(inputT, by: scale, on: cmd0)
        cmd0.commit()
        cmd0.waitUntilCompleted()
        h = addSinusoidalPositions(
            h, nRows: nRows,
            dim: config.inputSize, device: device)

        // encoders0 → encoders → after_norm → tp_encoders → tp_norm.
        for layer in encoders0 {
            h = forwardLayer(layer, h: h, nRows: nRows, device: device)
        }
        for layer in encoders {
            h = forwardLayer(layer, h: h, nRows: nRows, device: device)
        }
        h = normRows(afterNorm, h, nRows: nRows, device: device)
        for layer in tpEncoders {
            h = forwardLayer(layer, h: h, nRows: nRows, device: device)
        }
        return normRows(tpNorm, h, nRows: nRows, device: device)
    }

    /// One SAN-M block forward over the `[nRows, *]` frame sequence.
    private func forwardLayer(
        _ layer: SenseVoiceEncoderLayer, h hIn: Tensor,
        nRows: Int, device: Device
    ) -> Tensor {
        let H = config.hidden

        // ── Self-attention sub-block ──
        let cmd1 = device.makeCommandBuffer()
        let normed = Ops.layerNorm(
            hIn, weight: layer.norm1.weight,
            bias: layer.norm1.bias,
            eps: layer.norm1.eps,
            nRows: nRows, rowSize: layer.inSize,
            on: cmd1)
        // Fused Q/K/V projection → [nRows, 3*hidden].
        let qkv = projectRows(
            layer.qkvProj, normed, nRows: nRows,
            outDim: 3 * H, on: cmd1)
        cmd1.commit()
        cmd1.waitUntilCompleted()

        // Split the fused projection into Q / K / V row-major blocks.
        let qkvVals = qkv.toFloatArray()
        var qa = [Float](repeating: 0, count: nRows * H)
        var ka = [Float](repeating: 0, count: nRows * H)
        var va = [Float](repeating: 0, count: nRows * H)
        for r in 0 ..< nRows {
            let src = r * 3 * H
            let dst = r * H
            for c in 0 ..< H {
                qa[dst + c] = qkvVals[src + c]
                ka[dst + c] = qkvVals[src + H + c]
                va[dst + c] = qkvVals[src + 2 * H + c]
            }
        }

        // FSMN memory block — a depthwise conv over V, added back as
        // `memory + V`. Run before attention so it sees the raw V.
        let fsmnMemory = fsmn(layer, v: va, nRows: nRows)

        // Bidirectional multi-head attention over the V-derived context.
        let ctx = cpuAttention(
            qa: qa, ka: ka, va: va, nRows: nRows,
            heads: layer.heads, headDim: layer.headDim,
            scale: layer.scale, device: device)

        let cmd2 = device.makeCommandBuffer()
        var attnOut = projectRows(
            layer.outProj, ctx, nRows: nRows,
            outDim: H, on: cmd2)
        // attnOut += FSMN memory.
        let memT = Tensor.empty(
            shape: [nRows, H], dtype: dtype,
            device: device)
        AudioPreprocessing.copyFloats(fsmnMemory, into: memT)
        attnOut = Ops.add(attnOut, memT, on: cmd2)
        // Residual — block 0 of `encoders0` changes dim, so skip the
        // residual there (`inSize != hidden`).
        var h =
            layer.inSize == H
            ? Ops.add(hIn, attnOut, on: cmd2)
            : attnOut

        // ── ReLU MLP sub-block ──
        let mlpNormed = Ops.layerNorm(
            h, weight: layer.norm2.weight,
            bias: layer.norm2.bias,
            eps: layer.norm2.eps,
            nRows: nRows, rowSize: H, on: cmd2)
        let ff1 = projectRows(
            layer.w1, mlpNormed, nRows: nRows,
            outDim: layer.w1.weight.shape[0], on: cmd2)
        let act = Ops.relu(ff1, on: cmd2)
        let ff2 = projectRows(
            layer.w2, act, nRows: nRows,
            outDim: H, on: cmd2)
        h = Ops.add(h, ff2, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return h
    }

    /// FSMN memory block — a depthwise (per-channel) 1-D convolution
    /// over the value sequence with `memory + V` residual. `v` is
    /// row-major `[nRows, hidden]`; the conv weight is `[hidden, 1,
    /// kernel]`. Run on the CPU: it is one `[kernel]` filter per
    /// channel, far cheaper than the encoder GEMMs, and `Ops.audioConv1d`
    /// is a dense (non-grouped) conv so it cannot express it.
    private func fsmn(
        _ layer: SenseVoiceEncoderLayer, v: [Float],
        nRows: Int
    ) -> [Float] {
        let H = config.hidden
        let K = layer.fsmnKernel
        let weight = layer.fsmnWeight.toFloatArray()  // [H, 1, K]
        var out = [Float](repeating: 0, count: nRows * H)
        for c in 0 ..< H {
            let wBase = c * K
            for t in 0 ..< nRows {
                var acc: Float = 0
                // Output position t reads inputs centred by the left
                // padding — frame `t - leftPad + k` for tap k.
                for k in 0 ..< K {
                    let src = t - layer.fsmnLeftPad + k
                    if src >= 0 && src < nRows {
                        acc += weight[wBase + k] * v[src * H + c]
                    }
                }
                // `memory + V` residual.
                out[t * H + c] = acc + v[t * H + c]
            }
        }
        return out
    }

    /// CPU bidirectional multi-head attention. `qa` / `ka` / `va` are
    /// row-major `[nRows, heads*headDim]`. SenseVoice's encoder is
    /// non-causal — every frame attends to every other.
    ///
    /// Fans the `(head, query-row)` index space across CPU cores with
    /// `DispatchQueue.concurrentPerform`. Each iteration writes to a
    /// disjoint `[oBase, oBase + headDim)` output slice — race-free by
    /// construction. Mirrors the parallelization of
    /// `AudioEncoderLayer.cpuAttention`.
    private func cpuAttention(
        qa: [Float], ka: [Float], va: [Float],
        nRows: Int, heads: Int, headDim: Int,
        scale: Float, device: Device
    ) -> Tensor {
        let stride = heads * headDim
        var out = [Float](repeating: 0, count: nRows * stride)

        out.withUnsafeMutableBufferPointer { outBuf in
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            qa.withUnsafeBufferPointer { qPtr in
                ka.withUnsafeBufferPointer { kPtr in
                    va.withUnsafeBufferPointer { vPtr in
                        nonisolated(unsafe) let qb = qPtr.baseAddress!
                        nonisolated(unsafe) let kb = kPtr.baseAddress!
                        nonisolated(unsafe) let vb = vPtr.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: heads * nRows) { work in
                            let head = work / nRows
                            let i = work % nRows
                            let hOff = head * headDim
                            var scores = [Float](repeating: 0, count: nRows)
                            var maxScore = -Float.greatestFiniteMagnitude
                            let qBase = i * stride + hOff
                            for j in 0 ..< nRows {
                                var dot: Float = 0
                                let kBase = j * stride + hOff
                                for d in 0 ..< headDim {
                                    dot += qb[qBase + d] * kb[kBase + d]
                                }
                                let s = dot * scale
                                scores[j] = s
                                if s > maxScore { maxScore = s }
                            }
                            var sumExp: Float = 0
                            for j in 0 ..< nRows {
                                let e = exp(scores[j] - maxScore)
                                scores[j] = e
                                sumExp += e
                            }
                            let inv = sumExp > 0 ? 1 / sumExp : 0
                            let oBase = i * stride + hOff
                            for j in 0 ..< nRows {
                                let w = scores[j] * inv
                                let vBase = j * stride + hOff
                                for d in 0 ..< headDim {
                                    outPtr[oBase + d] += w * vb[vBase + d]
                                }
                            }
                        }
                    }
                }
            }
        }
        let result = Tensor.empty(
            shape: [nRows, stride], dtype: dtype,
            device: device)
        AudioPreprocessing.copyFloats(out, into: result)
        return result
    }

    // ─── Small helpers ───────────────────────────────────────────────

    /// Add the model's CMVN normalisation in place — `(feat + mean) *
    /// invStd` per feature dim, the FunASR convention. No-op when the
    /// checkpoint ships no stats.
    private func applyCMVN(_ feats: inout [Float]) {
        guard let mean = cmvnMean, let invStd = cmvnInvStd,
            mean.count == config.inputSize,
            invStd.count == config.inputSize
        else { return }
        let D = config.inputSize
        let nFrames = feats.count / D
        for f in 0 ..< nFrames {
            let base = f * D
            for d in 0 ..< D {
                feats[base + d] = (feats[base + d] + mean[d]) * invStd[d]
            }
        }
    }

    /// Apply a `Linear` to every row via `Ops.gemm` + broadcast bias.
    private func projectRows(
        _ linear: Linear, _ x: Tensor, nRows: Int,
        outDim: Int,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        let y = Ops.gemm(
            weight: linear.weight, input: x, nRows: nRows,
            on: cmd)
        guard let bias = linear.bias else { return y }
        return AudioEncoder.addRowBias(
            y, bias: bias, nRows: nRows,
            rowSize: outDim, on: cmd)
    }

    /// LayerNorm a `[nRows, hidden]` tensor on its own command buffer.
    private func normRows(
        _ ln: LayerNorm, _ x: Tensor, nRows: Int,
        device: Device
    ) -> Tensor {
        let cmd = device.makeCommandBuffer()
        let y = Ops.layerNorm(
            x, weight: ln.weight, bias: ln.bias,
            eps: ln.eps, nRows: nRows,
            rowSize: config.hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return y
    }

    /// Scale every element of a tensor by a scalar (CPU round-trip — the
    /// pre-encoder `sqrt(hidden)` input scale).
    private func scaleRows(
        _ x: Tensor, by factor: Float,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        let vals = x.toFloatArray().map { $0 * factor }
        let out = Tensor.empty(shape: x.shape, dtype: dtype)
        AudioPreprocessing.copyFloats(vals, into: out)
        return out
    }

    /// Add the SenseVoice sinusoidal position embedding to `[nRows, dim]`
    /// rows. FunASR's encoder builds the table at runtime: position
    /// index runs `1...nRows`, the `[sin | cos]` halves are concatenated.
    private func addSinusoidalPositions(
        _ x: Tensor, nRows: Int, dim: Int,
        device: Device
    ) -> Tensor {
        let half = max(dim / 2, 1)
        let logIncrement = log(10_000.0) / Double(max(half - 1, 1))
        var table = [Float](repeating: 0, count: nRows * dim)
        for p in 0 ..< nRows {
            // FunASR positions are 1-indexed.
            let pos = Double(p + 1)
            for i in 0 ..< half {
                let invFreq = exp(-logIncrement * Double(i))
                let angle = pos * invFreq
                table[p * dim + i] = Float(sin(angle))
                if half + i < dim {
                    table[p * dim + half + i] = Float(cos(angle))
                }
            }
        }
        let posT = Tensor.empty(
            shape: [nRows, dim], dtype: dtype,
            device: device)
        AudioPreprocessing.copyFloats(table, into: posT)
        let cmd = device.makeCommandBuffer()
        let out = Ops.add(x, posT, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return out
    }

    /// Row-wise log-softmax of a `[nRows, rowSize]` logit tensor — the
    /// CTC head emits log-probabilities. CPU path: `rowSize` is the
    /// vocab (~25 k) and `nRows` modest, cheap next to the encoder.
    private func logSoftmaxRows(
        _ logits: Tensor, nRows: Int, rowSize: Int,
        device: Device
    ) -> Tensor {
        let vals = logits.toFloatArray()
        var out = [Float](repeating: 0, count: nRows * rowSize)
        for r in 0 ..< nRows {
            let base = r * rowSize
            var maxV = -Float.greatestFiniteMagnitude
            for c in 0 ..< rowSize where vals[base + c] > maxV {
                maxV = vals[base + c]
            }
            var sumExp: Float = 0
            for c in 0 ..< rowSize { sumExp += exp(vals[base + c] - maxV) }
            let logSum = maxV + log(sumExp)
            for c in 0 ..< rowSize {
                out[base + c] = vals[base + c] - logSum
            }
        }
        let result = Tensor.empty(
            shape: [nRows, rowSize], dtype: dtype,
            device: device)
        AudioPreprocessing.copyFloats(out, into: result)
        return result
    }
}

// ─── Kaldi-style FBANK front-end ─────────────────────────────────────

/// SenseVoice's Kaldi-compatible FBANK front-end. Unlike Whisper's
/// log-Mel, this does per-frame mean removal, pre-emphasis, a
/// power-of-two FFT and a plain (un-normalised) log — so it is its own
/// CPU path rather than `Ops.melSpectrogram`. The output is the
/// `[nFrames, nMels * lfrM]` feature matrix the encoder consumes.
public enum SenseVoiceFrontEnd {

    /// Pre-emphasis coefficient — Kaldi's default.
    private static let preemph: Float = 0.97

    /// Compute the full feature matrix (FBANK → LFR) for a 16 kHz mono
    /// waveform. Returned row-major `[nFrames * (nMels * lfrM)]`.
    public static func featureFrames(
        waveform: [Float],
        cfg: SenseVoiceFrontEndConfig
    )
        -> [Float]
    {
        let fbank = kaldiFbank(waveform: waveform, cfg: cfg)
        return lowFrameRate(
            fbank, nMels: cfg.nMels,
            lfrM: cfg.lfrM, lfrN: cfg.lfrN)
    }

    /// Kaldi-style FBANK: `[nFrames, nMels]` row-major. Per the FunASR
    /// reference — frames are mean-removed, pre-emphasised, Hamming-
    /// windowed, FFT'd to a power spectrum, projected onto an HTK Mel
    /// bank and log-compressed.
    static func kaldiFbank(
        waveform: [Float],
        cfg: SenseVoiceFrontEndConfig
    ) -> [Float] {
        let winLength = cfg.winLength
        let hopLength = cfg.hopLength
        // FunASR scales [-1, 1] PCM up to int16 range before framing.
        var audio = waveform
        let peak = audio.map { abs($0) }.max() ?? 0
        if peak <= 1.0 {
            let g = Float(1 << 15)
            for i in audio.indices { audio[i] *= g }
        }
        guard audio.count >= winLength, winLength > 0, hopLength > 0 else {
            return []
        }
        let nFrames = 1 + (audio.count - winLength) / hopLength
        let fftLength = nextPowerOfTwo(winLength)
        let nFreq = fftLength / 2 + 1
        let window = hammingWindow(winLength)
        // FunASR's Kaldi FBANK uses an un-normalised HTK Mel bank;
        // `melFilterbank` applies Slaney triangle normalisation. The
        // difference is a fixed per-bin positive scale — the encoder's
        // LayerNorms and the learned `ctc_lo` projection absorb it, so
        // the greedy-CTC transcript stays coherent. A bit-exact HTK
        // bank is a later accuracy pass.
        let melBank = AudioPreprocessing.melFilterbank(
            AudioFrontEndConfig(
                sampleRate: cfg.sampleRate, nFFT: fftLength,
                hopLength: hopLength, nMels: cfg.nMels,
                fMin: 20.0))

        var out = [Float](repeating: 0, count: nFrames * cfg.nMels)
        var frame = [Float](repeating: 0, count: fftLength)
        for f in 0 ..< nFrames {
            let start = f * hopLength
            // Per-frame mean removal.
            var mean: Float = 0
            for i in 0 ..< winLength { mean += audio[start + i] }
            mean /= Float(winLength)
            for i in 0 ..< winLength { frame[i] = audio[start + i] - mean }
            // Pre-emphasis: y[t] = x[t] - 0.97 * x[t-1], y[0] keeps
            // x[0]*(1-0.97) as FunASR does.
            var prev = frame[0]
            frame[0] = prev - preemph * prev
            for i in 1 ..< winLength {
                let cur = frame[i]
                frame[i] = cur - preemph * prev
                prev = cur
            }
            // Hamming window.
            for i in 0 ..< winLength { frame[i] *= window[i] }
            // Zero-pad the tail to the FFT length.
            for i in winLength ..< fftLength { frame[i] = 0 }
            // Power spectrum, then Mel projection + log.
            let power = realFFTPower(
                frame, fftLength: fftLength,
                nFreq: nFreq)
            let oBase = f * cfg.nMels
            for m in 0 ..< cfg.nMels {
                var acc: Float = 0
                let wBase = m * nFreq
                for k in 0 ..< nFreq { acc += melBank[wBase + k] * power[k] }
                out[oBase + m] = log(max(acc, 1e-10))
            }
        }
        return out
    }

    /// Low-frame-rate stacking — concatenate `lfrM` consecutive FBANK
    /// frames into one encoder frame, advancing `lfrN` frames at a time.
    /// The first frame is left-padded by `(lfrM-1)/2` copies of frame 0;
    /// the tail is right-padded with the last frame. Output row-major
    /// `[lfrFrames * (nMels * lfrM)]`.
    static func lowFrameRate(
        _ fbank: [Float], nMels: Int,
        lfrM: Int, lfrN: Int
    ) -> [Float] {
        let nFrames = fbank.count / nMels
        guard nFrames > 0 else { return [] }
        let leftPad = max(0, (lfrM - 1) / 2)
        // Build a padded view: leftPad copies of frame 0, then the
        // FBANK, conceptually clamped on the right.
        func frame(_ idx: Int) -> ArraySlice<Float> {
            let clamped = min(max(idx - leftPad, 0), nFrames - 1)
            return fbank[clamped * nMels ..< (clamped + 1) * nMels]
        }
        // FunASR sizes the output by the unpadded frame count:
        // `ceil(nFrames / lfrN)`. The left padding only shifts the
        // window origin, it does not add output rows.
        let lfrFrames = Int(ceil(Double(nFrames) / Double(lfrN)))
        let rowSize = nMels * lfrM
        var out = [Float](repeating: 0, count: lfrFrames * rowSize)
        for r in 0 ..< lfrFrames {
            let dst = r * rowSize
            for m in 0 ..< lfrM {
                let slice = frame(r * lfrN + m)
                var offset = dst + m * nMels
                for v in slice {
                    out[offset] = v
                    offset += 1
                }
            }
        }
        return out
    }

    /// Periodic-style Hamming window of length `n` (FunASR uses the
    /// non-periodic `n-1` divisor for the Kaldi FBANK).
    static func hammingWindow(_ n: Int) -> [Float] {
        guard n > 1 else { return [Float](repeating: 1, count: max(n, 0)) }
        var w = [Float](repeating: 0, count: n)
        let denom = Double(n - 1)
        for i in 0 ..< n {
            w[i] = Float(
                0.54 - 0.46
                    * cos(2.0 * Double.pi * Double(i) / denom))
        }
        return w
    }

    /// Smallest power of two `>= value`.
    static func nextPowerOfTwo(_ value: Int) -> Int {
        guard value > 1 else { return max(value, 1) }
        var n = 1
        while n < value { n <<= 1 }
        return n
    }

    /// Real-FFT power spectrum of a `[fftLength]` frame, returning the
    /// `[nFreq]` non-redundant magnitudes squared. A direct radix-2
    /// Cooley-Tukey FFT — `fftLength` is small (512 for a 25 ms / 16 kHz
    /// window) so the O(n log n) CPU transform is negligible next to the
    /// encoder.
    static func realFFTPower(
        _ frame: [Float], fftLength: Int,
        nFreq: Int
    ) -> [Float] {
        var re = frame
        var im = [Float](repeating: 0, count: fftLength)
        fftInPlace(&re, &im)
        var power = [Float](repeating: 0, count: nFreq)
        for k in 0 ..< nFreq {
            power[k] = re[k] * re[k] + im[k] * im[k]
        }
        return power
    }

    /// In-place radix-2 Cooley-Tukey FFT. `re.count` must be a power of
    /// two. Decimation-in-time with a bit-reversal permutation.
    private static func fftInPlace(
        _ re: inout [Float],
        _ im: inout [Float]
    ) {
        let n = re.count
        guard n > 1 else { return }
        // Bit-reversal permutation.
        var j = 0
        for i in 1 ..< n {
            var bit = n >> 1
            while j & bit != 0 {
                j ^= bit
                bit >>= 1
            }
            j ^= bit
            if i < j {
                re.swapAt(i, j)
                im.swapAt(i, j)
            }
        }
        // Butterfly passes.
        var len = 2
        while len <= n {
            let ang = -2.0 * Double.pi / Double(len)
            let wRe = Float(cos(ang))
            let wIm = Float(sin(ang))
            var i = 0
            while i < n {
                var curRe: Float = 1
                var curIm: Float = 0
                for k in 0 ..< (len / 2) {
                    let a = i + k
                    let b = i + k + len / 2
                    let tRe = curRe * re[b] - curIm * im[b]
                    let tIm = curRe * im[b] + curIm * re[b]
                    re[b] = re[a] - tRe
                    im[b] = im[a] - tIm
                    re[a] += tRe
                    im[a] += tIm
                    let nextRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = nextRe
                }
                i += len
            }
            len <<= 1
        }
    }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension SenseVoiceModel {
    /// Recognised `model_type` strings for SenseVoice.
    public static let modelTypes: Set<String> = [
        "sensevoice",
        "sense_voice",
    ]
    public static let architectures: Set<String> = [
        "SenseVoiceSmall", "SenseVoice",
    ]

    /// Whether a decoded `config.json` describes a SenseVoice checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType,
            modelTypes.contains(mt.lowercased())
        {
            return true
        }
        if let arch = config.architecture,
            architectures.contains(arch)
        {
            return true
        }
        // SenseVoice configs nest `encoder_conf` with `tp_blocks` — a
        // distinctive marker no other audio family carries.
        if let enc = config.nested("encoder_conf"),
            enc["tp_blocks"] != nil
        {
            return true
        }
        return false
    }

    /// Load a SenseVoice checkpoint from a resolved snapshot directory.
    /// The directory must contain `config.json` + the safetensors
    /// shards; an optional `am.mvn` file carries the CMVN stats.
    public static func load(directory: URL, device: Device = .shared)
        throws -> SenseVoiceModel
    {
        let config = try ModelConfig.load(from: directory)
        guard let sc = SenseVoiceConfig.from(config) else {
            throw ModelError.unsupportedModelType(
                "config.json is not a SenseVoice config")
        }
        let bundle = try SafeTensorsBundle(
            directory: directory,
            device: device)
        let (mean, invStd) = loadCMVN(directory: directory, config: config)
        return try build(
            config: sc, bundle: bundle,
            cmvnMean: mean, cmvnInvStd: invStd)
    }

    /// Assemble a `SenseVoiceModel` from a decoded config + weight
    /// bundle. Factored out of `load` so tests can drive it directly.
    public static func build(
        config sc: SenseVoiceConfig,
        bundle: SafeTensorsBundle,
        cmvnMean: [Float]?, cmvnInvStd: [Float]?
    )
        throws -> SenseVoiceModel
    {
        // The first encoder weight sets the activation dtype.
        let probe =
            bundle.has("encoder.encoders0.0.norm1.weight")
            ? "encoder.encoders0.0.norm1.weight"
            : "encoder.after_norm.weight"
        let dtype = try bundle.tensor(named: probe).dtype

        func t(_ name: String) throws -> Tensor {
            try bundle.tensor(named: name)
        }
        func ln(_ base: String) throws -> LayerNorm {
            LayerNorm(
                weight: try t("\(base).weight"),
                bias: try t("\(base).bias"),
                eps: sc.layerNormEps)
        }
        func linear(_ base: String) throws -> Linear {
            let b = bundle.has("\(base).bias") ? try t("\(base).bias") : nil
            return Linear(weight: try t("\(base).weight"), bias: b)
        }
        // Build one SAN-M block. `inSize` differs from `hidden` only for
        // block 0 of `encoders0`.
        func block(_ base: String, inSize: Int) throws
            -> SenseVoiceEncoderLayer
        {
            SenseVoiceEncoderLayer(
                norm1: try ln("\(base).norm1"),
                norm2: try ln("\(base).norm2"),
                qkvProj: try linear("\(base).self_attn.linear_q_k_v"),
                outProj: try linear("\(base).self_attn.linear_out"),
                fsmnWeight: try t("\(base).self_attn.fsmn_block.weight"),
                w1: try linear("\(base).feed_forward.w_1"),
                w2: try linear("\(base).feed_forward.w_2"),
                inSize: inSize, hidden: sc.hidden, heads: sc.heads,
                fsmnKernel: sc.fsmnKernel, fsmnShift: sc.fsmnShift)
        }

        let encoders0 = [
            try block(
                "encoder.encoders0.0",
                inSize: sc.inputSize)
        ]
        var encoders: [SenseVoiceEncoderLayer] = []
        for i in 0 ..< max(sc.numBlocks - 1, 0) {
            encoders.append(
                try block(
                    "encoder.encoders.\(i)",
                    inSize: sc.hidden))
        }
        var tpEncoders: [SenseVoiceEncoderLayer] = []
        for i in 0 ..< sc.tpBlocks {
            tpEncoders.append(
                try block(
                    "encoder.tp_encoders.\(i)",
                    inSize: sc.hidden))
        }

        // CTC projection — some conversions prefix it `ctc.ctc_lo.`.
        let ctcBase =
            bundle.has("ctc_lo.weight")
            ? "ctc_lo" : "ctc.ctc_lo"
        return SenseVoiceModel(
            config: sc,
            queryEmbed: try t("embed.weight"),
            encoders0: encoders0,
            encoders: encoders,
            afterNorm: try ln("encoder.after_norm"),
            tpEncoders: tpEncoders,
            tpNorm: try ln("encoder.tp_norm"),
            ctcProj: try linear(ctcBase),
            cmvnMean: cmvnMean, cmvnInvStd: cmvnInvStd,
            dtype: dtype)
    }

    /// Load the CMVN mean / inverse-std stats. SenseVoice ships them in
    /// a Kaldi `am.mvn` text file; some conversions fold them into
    /// `config.json` as `cmvn_means` / `cmvn_istd` arrays instead.
    private static func loadCMVN(directory: URL, config: ModelConfig)
        -> (mean: [Float]?, invStd: [Float]?)
    {
        let mvnURL = directory.appendingPathComponent("am.mvn")
        if let text = try? String(contentsOf: mvnURL, encoding: .utf8),
            let parsed = parseAMMVN(text)
        {
            return (parsed.mean, parsed.invStd)
        }
        let mean = (config.raw["cmvn_means"] as? [Double])?
            .map { Float($0) }
        let invStd = (config.raw["cmvn_istd"] as? [Double])?
            .map { Float($0) }
        return (mean, invStd)
    }

    /// Parse a Kaldi `am.mvn` file — the `<AddShift>` block carries the
    /// mean offsets, `<Rescale>` carries the inverse-std scales.
    private static func parseAMMVN(_ text: String)
        -> (mean: [Float], invStd: [Float])?
    {
        func bracketed(after marker: String) -> [Float]? {
            guard let mRange = text.range(of: marker),
                let open = text.range(
                    of: "[",
                    range:
                        mRange.upperBound ..< text.endIndex),
                let close = text.range(
                    of: "]",
                    range:
                        open.upperBound ..< text.endIndex)
            else { return nil }
            return text[open.upperBound ..< close.lowerBound]
                .split(whereSeparator: {
                    $0 == " " || $0 == "\n"
                        || $0 == "\t"
                })
                .compactMap { Float($0) }
        }
        guard let mean = bracketed(after: "<AddShift>"),
            let invStd = bracketed(after: "<Rescale>"),
            !mean.isEmpty, mean.count == invStd.count
        else { return nil }
        return (mean, invStd)
    }
}
