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
// Whisper — OpenAI's encoder/decoder speech-to-text family
// (tiny → large-v3). All variants share one architecture: a
// Whisper-style `AudioEncoder` turns a log-Mel spectrogram into audio
// features, and a causal text decoder cross-attends to those features
// while autoregressively emitting transcript tokens.
//
//   waveform ──front-end──▶ log-Mel ──AudioEncoder──▶ audio features
//   tokens   ──decoder (self-attn + cross-attn to audio)──▶ next token
//
// Variants differ only in width / depth / n_mels — captured by
// `WhisperConfig`. The decoder self-attention runs through the
// head-dim-agnostic CPU attention core (Whisper head dims are 64 for
// most variants); the cross-attention is likewise a CPU core since the
// audio-feature K/V are recomputed per layer and there is no causal
// mask. Correctness-first; a fused decode SDPA path is a perf
// follow-up.
//
// This file deliberately does NOT route through `ModelRegistry` /
// `LanguageModel` — those describe a pure text-in/text-out decoder.
// Whisper is loaded via `Whisper.load(...)` and exposes a
// `transcribe(...)` surface instead.

import Foundation
import Metal
import Tokenizers

// ─── Configuration ───────────────────────────────────────────────────

/// Whisper architecture hyper-parameters, decoded from `config.json`.
public struct WhisperConfig: Sendable {
    /// Mel filterbank bins (80 for tiny→medium, 128 for large-v3).
    public let nMels: Int
    /// Encoder + decoder hidden dim (`d_model`).
    public let hidden: Int
    /// Encoder blocks.
    public let encoderLayers: Int
    /// Encoder attention heads.
    public let encoderHeads: Int
    /// Decoder blocks.
    public let decoderLayers: Int
    /// Decoder attention heads.
    public let decoderHeads: Int
    /// Feed-forward intermediate dim (`4 * d_model` for every Whisper).
    public let intermediate: Int
    /// Transcript vocabulary size.
    public let vocab: Int
    /// Maximum decoder context length.
    public let maxDecoderCtx: Int
    /// Maximum audio-context length (encoder positional rows).
    public let maxAudioCtx: Int

    public init(
        nMels: Int, hidden: Int, encoderLayers: Int, encoderHeads: Int,
        decoderLayers: Int, decoderHeads: Int, intermediate: Int,
        vocab: Int, maxDecoderCtx: Int = 448, maxAudioCtx: Int = 1500
    ) {
        self.nMels = nMels
        self.hidden = hidden
        self.encoderLayers = encoderLayers
        self.encoderHeads = encoderHeads
        self.decoderLayers = decoderLayers
        self.decoderHeads = decoderHeads
        self.intermediate = intermediate
        self.vocab = vocab
        self.maxDecoderCtx = maxDecoderCtx
        self.maxAudioCtx = maxAudioCtx
    }

    /// Decoder per-head dimension.
    public var decoderHeadDim: Int { hidden / decoderHeads }

    /// Build a `WhisperConfig` from a decoded `config.json`. Whisper
    /// HF configs name fields `d_model`, `encoder_layers`, etc.
    public static func from(_ config: ModelConfig) -> WhisperConfig? {
        guard let hidden = config.int("d_model"),
            let encLayers = config.int("encoder_layers"),
            let encHeads = config.int("encoder_attention_heads"),
            let decLayers = config.int("decoder_layers"),
            let decHeads = config.int("decoder_attention_heads"),
            let vocab = config.int("vocab_size")
        else { return nil }
        let nMels = config.int("num_mel_bins") ?? 80
        let intermediate = config.int("encoder_ffn_dim") ?? (4 * hidden)
        let maxDec = config.int("max_target_positions") ?? 448
        let maxAud = config.int("max_source_positions") ?? 1500
        return WhisperConfig(
            nMels: nMels, hidden: hidden,
            encoderLayers: encLayers, encoderHeads: encHeads,
            decoderLayers: decLayers, decoderHeads: decHeads,
            intermediate: intermediate, vocab: vocab,
            maxDecoderCtx: maxDec, maxAudioCtx: maxAud)
    }

    /// The front-end config that pairs with this Whisper variant.
    public var frontEnd: AudioFrontEndConfig {
        AudioFrontEndConfig(
            sampleRate: 16_000, nFFT: 400, hopLength: 160,
            nMels: nMels)
    }
}

// ─── Decoder block ───────────────────────────────────────────────────

/// One Whisper decoder block: pre-norm causal self-attention, then
/// pre-norm cross-attention to the audio features, then a pre-norm GELU
/// MLP — each with a residual add.
public final class WhisperDecoderLayer: Module {
    let selfAttnLayerNorm: LayerNorm
    let qProj, kProj, vProj, oProj: Linear
    let crossAttnLayerNorm: LayerNorm
    let crossQProj, crossKProj, crossVProj, crossOProj: Linear
    let finalLayerNorm: LayerNorm
    let fc1, fc2: Linear

    let hidden, nHeads, headDim, intermediate: Int
    let scale: Float

    init(
        selfAttnLayerNorm: LayerNorm,
        qProj: Linear, kProj: Linear, vProj: Linear, oProj: Linear,
        crossAttnLayerNorm: LayerNorm,
        crossQProj: Linear, crossKProj: Linear, crossVProj: Linear,
        crossOProj: Linear,
        finalLayerNorm: LayerNorm, fc1: Linear, fc2: Linear,
        hidden: Int, nHeads: Int, intermediate: Int
    ) {
        self.selfAttnLayerNorm = selfAttnLayerNorm
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.crossAttnLayerNorm = crossAttnLayerNorm
        self.crossQProj = crossQProj
        self.crossKProj = crossKProj
        self.crossVProj = crossVProj
        self.crossOProj = crossOProj
        self.finalLayerNorm = finalLayerNorm
        self.fc1 = fc1
        self.fc2 = fc2
        self.hidden = hidden
        self.nHeads = nHeads
        self.headDim = hidden / nHeads
        self.intermediate = intermediate
        self.scale = 1.0 / Float(Double(hidden / nHeads).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in selfAttnLayerNorm.parameters() {
            out.append(("self_attn_layer_norm.\(k)", v))
        }
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.out_proj.\(k)", v)) }
        for (k, v) in crossAttnLayerNorm.parameters() {
            out.append(("encoder_attn_layer_norm.\(k)", v))
        }
        for (k, v) in crossQProj.parameters() { out.append(("encoder_attn.q_proj.\(k)", v)) }
        for (k, v) in crossKProj.parameters() { out.append(("encoder_attn.k_proj.\(k)", v)) }
        for (k, v) in crossVProj.parameters() { out.append(("encoder_attn.v_proj.\(k)", v)) }
        for (k, v) in crossOProj.parameters() { out.append(("encoder_attn.out_proj.\(k)", v)) }
        for (k, v) in finalLayerNorm.parameters() {
            out.append(("final_layer_norm.\(k)", v))
        }
        for (k, v) in fc1.parameters() { out.append(("fc1.\(k)", v)) }
        for (k, v) in fc2.parameters() { out.append(("fc2.\(k)", v)) }
        return out
    }
}

// ─── Whisper model ───────────────────────────────────────────────────

/// A loaded Whisper STT model. Holds the audio encoder, the decoder
/// stack, the token embedding + decoder positional embedding, and the
/// projection back to the vocabulary (tied to the token embedding).
public final class WhisperModel: @unchecked Sendable {
    public let config: WhisperConfig
    public let encoder: AudioEncoder

    let tokenEmbedding: Tensor  // [vocab, hidden]
    let decoderPosEmbedding: Tensor  // [maxDecoderCtx, hidden]
    let decoderLayers: [WhisperDecoderLayer]
    let decoderLayerNorm: LayerNorm  // post-decoder LayerNorm
    let dtype: DType

    public init(
        config: WhisperConfig, encoder: AudioEncoder,
        tokenEmbedding: Tensor, decoderPosEmbedding: Tensor,
        decoderLayers: [WhisperDecoderLayer],
        decoderLayerNorm: LayerNorm, dtype: DType
    ) {
        self.config = config
        self.encoder = encoder
        self.tokenEmbedding = tokenEmbedding
        self.decoderPosEmbedding = decoderPosEmbedding
        self.decoderLayers = decoderLayers
        self.decoderLayerNorm = decoderLayerNorm
        self.dtype = dtype
    }

    /// Whisper's fixed analysis window: 30 s of 16 kHz audio. OpenAI's
    /// reference pads or trims every clip to exactly this length before
    /// the log-Mel front-end, so the encoder always sees a fixed
    /// `[nMels, 3000]` spectrogram → 1500 audio-context rows (the size of
    /// the baked positional table). whisper-tiny was trained only on
    /// 30 s-padded inputs; feeding a raw short clip starves the decoder's
    /// cross-attention of the positional structure it expects.
    public static let whisperWindowSamples = 30 * 16_000

    /// Run the audio encoder over a waveform, producing the
    /// `[nAudioCtx, hidden]` audio features the decoder cross-attends
    /// to. The waveform is resampled / framed by `AudioPreprocessing`.
    public func encodeAudio(waveform: [Float], device: Device = .shared)
        -> Tensor
    {
        // The mel_spectrogram kernel only emits f32 / f16; a bf16 model
        // gets the front-end run in f32, then the spectrogram cast to
        // the model's activation dtype before the conv stem.
        let melDtype: DType = dtype == .f16 ? .f16 : .f32
        // Pad / trim to Whisper's fixed 30 s window — the encoder's
        // positional table and the checkpoint's training regime both
        // assume it (OpenAI's `pad_or_trim`).
        let framed = WhisperModel.padOrTrim(
            waveform, to: WhisperModel.whisperWindowSamples)
        let cmd = device.makeCommandBuffer()
        // `whisperNormalize: true` — `logMelSpectrogram` commits + waits
        // on `cmd` itself (it normalises the kernel result on the CPU),
        // so the result is already CPU-synced; do NOT re-commit `cmd`.
        let melRaw = AudioPreprocessing.logMelSpectrogram(
            waveform: framed, cfg: config.frontEnd, dtype: melDtype,
            whisperNormalize: true, device: device, on: cmd)
        var mel = AudioPreprocessing.castTensor(
            melRaw, to: dtype,
            device: device)
        // The kernel's frame walk yields one extra frame versus
        // Whisper's reference (`torch.stft(center=True)` produces
        // `n_samples/hop + 1` columns, then drops the last as
        // `stft[..., :-1]`). After the stride-2 conv2 that surplus frame
        // would push `nAudioCtx` to `maxAudioCtx + 1` and overrun the
        // 1500-row positional table — so trim the log-Mel to exactly
        // `2 * maxAudioCtx` frames here.
        let maxMelFrames = 2 * config.maxAudioCtx
        if mel.shape[0] > maxMelFrames {
            mel = mel.slicedRows(start: 0, count: maxMelFrames)
        }
        return encoder.encode(mel: mel, melFrameMajor: true, device: device)
    }

    /// Pad (with trailing zeros) or trim a waveform to exactly `length`
    /// samples — OpenAI Whisper's `pad_or_trim`. A clip shorter than the
    /// 30 s window is zero-padded; a longer one is truncated to the
    /// first window. Trailing zeros become near-floor log-Mel frames,
    /// exactly the silence representation the decoder is trained on.
    public static func padOrTrim(_ waveform: [Float], to length: Int)
        -> [Float]
    {
        if waveform.count == length { return waveform }
        if waveform.count > length { return Array(waveform[0 ..< length]) }
        return waveform + [Float](repeating: 0, count: length - waveform.count)
    }

    /// One decoder step: embed `tokenIds` (the transcript so far),
    /// run the decoder stack cross-attending to `audioFeatures`, return
    /// the `[vocab]` logits for the next token. All-CPU attention cores
    /// keep the decode head-dim-agnostic and correct; Whisper transcript
    /// lengths are short so the O(L²) self-attention is cheap.
    public func decoderLogits(
        tokenIds: [Int], audioFeatures: Tensor,
        device: Device = .shared
    ) -> [Float] {
        let L = tokenIds.count
        precondition(
            L > 0 && L <= config.maxDecoderCtx,
            "WhisperModel.decoderLogits: bad token count \(L)")
        let H = config.hidden

        // Embed tokens + add positional embedding → [L, hidden].
        var h = embedTokens(tokenIds, device: device)

        let audio = audioFeatures.toFloatArray()
        let nAudioCtx = audioFeatures.shape[0]

        for layer in decoderLayers {
            h = decodeLayer(
                layer, h: h, L: L,
                audio: audio, nAudioCtx: nAudioCtx,
                device: device)
        }

        // Post-decoder LayerNorm, then project the LAST token's hidden
        // state to vocab logits via the tied token embedding.
        let cmdN = device.makeCommandBuffer()
        let normed = Ops.layerNorm(
            h, weight: decoderLayerNorm.weight,
            bias: decoderLayerNorm.bias,
            eps: decoderLayerNorm.eps,
            nRows: L, rowSize: H, on: cmdN)
        cmdN.commit()
        cmdN.waitUntilCompleted()
        // gemv requires a 1D input — drop the leading singleton row.
        let lastHidden = normed.slicedRows(start: L - 1, count: 1)
            .reshaped(to: [H])

        let cmdL = device.makeCommandBuffer()
        let logits = Ops.gemv(
            weight: tokenEmbedding, input: lastHidden,
            on: cmdL)
        cmdL.commit()
        cmdL.waitUntilCompleted()
        return logits.toFloatArray()
    }

    /// Greedy-decode a transcript. `initialTokens` is the Whisper
    /// prompt prefix (`<|startoftranscript|>` + language + task tokens);
    /// `eosToken` ends generation. Returns the generated token ids
    /// (excluding the prompt prefix).
    public func generateTranscript(
        audioFeatures: Tensor, initialTokens: [Int], eosToken: Int,
        maxTokens: Int = 224, device: Device = .shared
    ) -> [Int] {
        var tokens = initialTokens
        var generated: [Int] = []
        for _ in 0 ..< maxTokens {
            let logits = decoderLogits(
                tokenIds: tokens,
                audioFeatures: audioFeatures,
                device: device)
            var best = 0
            var bestVal = -Float.greatestFiniteMagnitude
            for (i, v) in logits.enumerated() where v > bestVal {
                bestVal = v
                best = i
            }
            if best == eosToken { break }
            tokens.append(best)
            generated.append(best)
            if tokens.count >= config.maxDecoderCtx { break }
        }
        return generated
    }

    // ─── Decoder internals ───────────────────────────────────────────

    /// Embed transcript tokens + add the decoder positional embedding.
    private func embedTokens(_ tokenIds: [Int], device: Device) -> Tensor {
        let H = config.hidden
        let L = tokenIds.count
        let idsT = Tensor.empty(shape: [L], dtype: .u32, device: device)
        idsT.copyIn(from: tokenIds.map { UInt32($0) })
        let cmd = device.makeCommandBuffer()
        let embed = Ops.gather(table: tokenEmbedding, tokenIds: idsT, on: cmd)
        let pos = decoderPosEmbedding.slicedRows(start: 0, count: L)
        let h = Ops.add(embed, pos, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        _ = H
        return h
    }

    /// One decoder block forward over the whole `[L, hidden]` token
    /// sequence: causal self-attention, cross-attention to audio, MLP.
    private func decodeLayer(
        _ layer: WhisperDecoderLayer, h hIn: Tensor,
        L: Int, audio: [Float], nAudioCtx: Int,
        device: Device
    ) -> Tensor {
        let H = config.hidden
        var h = hIn

        // ── Causal self-attention ──
        let cmd1 = device.makeCommandBuffer()
        let normed = Ops.layerNorm(
            h, weight: layer.selfAttnLayerNorm.weight,
            bias: layer.selfAttnLayerNorm.bias,
            eps: layer.selfAttnLayerNorm.eps,
            nRows: L, rowSize: H, on: cmd1)
        let q = project(layer.qProj, normed, nRows: L, on: cmd1)
        let k = project(layer.kProj, normed, nRows: L, on: cmd1)
        let v = project(layer.vProj, normed, nRows: L, on: cmd1)
        cmd1.commit()
        cmd1.waitUntilCompleted()

        let selfCtx = cpuAttention(
            qa: q.toFloatArray(), ka: k.toFloatArray(), va: v.toFloatArray(),
            nQuery: L, nKV: L, nHeads: layer.nHeads, headDim: layer.headDim,
            scale: layer.scale, causal: true, device: device)

        let cmd2 = device.makeCommandBuffer()
        let selfOut = project(layer.oProj, selfCtx, nRows: L, on: cmd2)
        h = Ops.add(h, selfOut, on: cmd2)

        // ── Cross-attention to audio features ──
        let crossNormed = Ops.layerNorm(
            h,
            weight: layer.crossAttnLayerNorm.weight,
            bias: layer.crossAttnLayerNorm.bias,
            eps: layer.crossAttnLayerNorm.eps,
            nRows: L, rowSize: H, on: cmd2)
        let crossQ = project(layer.crossQProj, crossNormed, nRows: L, on: cmd2)
        // K / V come from the audio features (recomputed per layer).
        let audioT = Tensor.empty(
            shape: [nAudioCtx, H], dtype: dtype,
            device: device)
        AudioPreprocessing.copyFloats(audio, into: audioT)
        let crossK = project(layer.crossKProj, audioT, nRows: nAudioCtx, on: cmd2)
        let crossV = project(layer.crossVProj, audioT, nRows: nAudioCtx, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        let crossCtx = cpuAttention(
            qa: crossQ.toFloatArray(), ka: crossK.toFloatArray(),
            va: crossV.toFloatArray(),
            nQuery: L, nKV: nAudioCtx, nHeads: layer.nHeads,
            headDim: layer.headDim, scale: layer.scale, causal: false,
            device: device)

        // ── MLP ──
        let cmd3 = device.makeCommandBuffer()
        let crossOut = project(layer.crossOProj, crossCtx, nRows: L, on: cmd3)
        h = Ops.add(h, crossOut, on: cmd3)
        let mlpNormed = Ops.layerNorm(
            h, weight: layer.finalLayerNorm.weight,
            bias: layer.finalLayerNorm.bias,
            eps: layer.finalLayerNorm.eps,
            nRows: L, rowSize: H, on: cmd3)
        let ff1 = project(layer.fc1, mlpNormed, nRows: L, on: cmd3)
        let act = Ops.gelu(ff1, on: cmd3)
        let ff2 = project(layer.fc2, act, nRows: L, on: cmd3)
        h = Ops.add(h, ff2, on: cmd3)
        cmd3.commit()
        cmd3.waitUntilCompleted()
        return h
    }

    /// Apply a `Linear` to every row via `Ops.gemm` + broadcast bias.
    private func project(
        _ linear: Linear, _ x: Tensor, nRows: Int,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        let outD = linear.weight.shape[0]
        let y = Ops.gemm(weight: linear.weight, input: x, nRows: nRows, on: cmd)
        guard let bias = linear.bias else { return y }
        return AudioEncoder.addRowBias(
            y, bias: bias, nRows: nRows,
            rowSize: outD, on: cmd)
    }

    /// CPU multi-head attention. `qa` is query-major `[nQuery, H]`,
    /// `ka`/`va` are key-major `[nKV, H]` (`H = nHeads*headDim`).
    /// `causal == true` masks `j > i`. Returns the context query-major.
    ///
    /// Fans the `(head, query-row)` index space across CPU cores with
    /// `DispatchQueue.concurrentPerform`. Each iteration writes to a
    /// disjoint `[oBase, oBase + headDim)` output slice — race-free by
    /// construction. Mirrors the parallelization of
    /// `AudioEncoderLayer.cpuAttention`.
    private func cpuAttention(
        qa: [Float], ka: [Float], va: [Float],
        nQuery: Int, nKV: Int, nHeads: Int,
        headDim: Int, scale: Float, causal: Bool,
        device: Device
    ) -> Tensor {
        let stride = nHeads * headDim
        var out = [Float](repeating: 0, count: nQuery * stride)

        out.withUnsafeMutableBufferPointer { outBuf in
            let outPtr = outBuf.baseAddress!
            qa.withUnsafeBufferPointer { qPtr in
                ka.withUnsafeBufferPointer { kPtr in
                    va.withUnsafeBufferPointer { vPtr in
                        let qb = qPtr.baseAddress!
                        let kb = kPtr.baseAddress!
                        let vb = vPtr.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: nHeads * nQuery) { work in
                            let head = work / nQuery
                            let i = work % nQuery
                            let hOff = head * headDim
                            let jMax = causal ? min(i, nKV - 1) : nKV - 1
                            var scores = [Float](repeating: 0, count: jMax + 1)
                            var maxScore = -Float.greatestFiniteMagnitude
                            let qBase = i * stride + hOff
                            for j in 0 ... jMax {
                                var dot: Float = 0
                                let kBase = j * stride + hOff
                                for d in 0 ..< headDim { dot += qb[qBase + d] * kb[kBase + d] }
                                let s = dot * scale
                                scores[j] = s
                                if s > maxScore { maxScore = s }
                            }
                            var sumExp: Float = 0
                            for j in 0 ... jMax {
                                let e = exp(scores[j] - maxScore)
                                scores[j] = e
                                sumExp += e
                            }
                            let inv = sumExp > 0 ? 1 / sumExp : 0
                            let oBase = i * stride + hOff
                            for j in 0 ... jMax {
                                let w = scores[j] * inv
                                let vBase = j * stride + hOff
                                for d in 0 ..< headDim { outPtr[oBase + d] += w * vb[vBase + d] }
                            }
                        }
                    }
                }
            }
        }
        let result = Tensor.empty(
            shape: [nQuery, stride], dtype: dtype,
            device: device)
        AudioPreprocessing.copyFloats(out, into: result)
        return result
    }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension WhisperModel {
    /// Recognised `model_type` / architecture strings for Whisper.
    public static let modelTypes: Set<String> = ["whisper"]
    public static let architectures: Set<String> = [
        "WhisperForConditionalGeneration", "WhisperModel",
    ]

    /// Whether a decoded `config.json` describes a Whisper checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        return false
    }

    /// Load a Whisper checkpoint from a resolved snapshot directory.
    /// The directory must contain `config.json` + the safetensors
    /// shards. HF Whisper weight keys are prefixed `model.encoder.` /
    /// `model.decoder.`; `proj_out.weight` (or the tied token embedding)
    /// is the vocabulary projection.
    public static func load(directory: URL, device: Device = .shared)
        throws -> WhisperModel
    {
        let config = try ModelConfig.load(from: directory)
        guard let wc = WhisperConfig.from(config) else {
            throw ModelError.unsupportedModelType(
                "config.json is not a Whisper config")
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return try build(config: wc, bundle: bundle)
    }

    /// Assemble a `WhisperModel` from a decoded config + a weight
    /// bundle. Factored out of `load` so tests can drive it directly.
    public static func build(
        config wc: WhisperConfig,
        bundle: SafeTensorsBundle
    ) throws -> WhisperModel {
        // The first weight tensor sets the activation dtype.
        let probeKey =
            bundle.has("model.encoder.conv1.weight")
            ? "model.encoder.conv1.weight" : "encoder.conv1.weight"
        let dtype = try bundle.tensor(named: probeKey).dtype
        // HF Whisper prefixes encoder/decoder weights with `model.`;
        // some conversions drop it. Detect once.
        let prefix = bundle.has("model.encoder.conv1.weight") ? "model." : ""

        func t(_ name: String) throws -> Tensor {
            try bundle.tensor(named: prefix + name)
        }
        func ln(_ base: String) throws -> LayerNorm {
            LayerNorm(
                weight: try t("\(base).weight"),
                bias: try t("\(base).bias"), eps: 1e-5)
        }
        func linear(_ base: String, hasBias: Bool = true) throws -> Linear {
            let w = try t("\(base).weight")
            let b =
                hasBias && bundle.has(prefix + "\(base).bias")
                ? try t("\(base).bias") : nil
            return Linear(weight: w, bias: b)
        }

        // ── Audio encoder ──
        var encLayers: [AudioEncoderLayer] = []
        for i in 0 ..< wc.encoderLayers {
            let base = "encoder.layers.\(i)"
            encLayers.append(
                AudioEncoderLayer(
                    layerNorm1: try ln("\(base).self_attn_layer_norm"),
                    qProj: try linear("\(base).self_attn.q_proj"),
                    kProj: try linear("\(base).self_attn.k_proj", hasBias: false),
                    vProj: try linear("\(base).self_attn.v_proj"),
                    oProj: try linear("\(base).self_attn.out_proj"),
                    layerNorm2: try ln("\(base).final_layer_norm"),
                    fc1: try linear("\(base).fc1"),
                    fc2: try linear("\(base).fc2"),
                    hidden: wc.hidden, nHeads: wc.encoderHeads,
                    intermediate: wc.intermediate))
        }
        let encoderConfig = AudioEncoderConfig(
            nMels: wc.nMels, hidden: wc.hidden, intermediate: wc.intermediate,
            nLayers: wc.encoderLayers, nHeads: wc.encoderHeads,
            maxAudioCtx: wc.maxAudioCtx, layerNormEps: 1e-5)
        let encoder = AudioEncoder(
            config: encoderConfig,
            conv1Weight: try t("encoder.conv1.weight"),
            conv1Bias: try t("encoder.conv1.bias"),
            conv2Weight: try t("encoder.conv2.weight"),
            conv2Bias: try t("encoder.conv2.bias"),
            positionEmbedding: try t("encoder.embed_positions.weight"),
            layers: encLayers,
            postLayerNorm: try ln("encoder.layer_norm"),
            dtype: dtype)

        // ── Decoder ──
        var decLayers: [WhisperDecoderLayer] = []
        for i in 0 ..< wc.decoderLayers {
            let base = "decoder.layers.\(i)"
            decLayers.append(
                WhisperDecoderLayer(
                    selfAttnLayerNorm: try ln("\(base).self_attn_layer_norm"),
                    qProj: try linear("\(base).self_attn.q_proj"),
                    kProj: try linear("\(base).self_attn.k_proj", hasBias: false),
                    vProj: try linear("\(base).self_attn.v_proj"),
                    oProj: try linear("\(base).self_attn.out_proj"),
                    crossAttnLayerNorm: try ln("\(base).encoder_attn_layer_norm"),
                    crossQProj: try linear("\(base).encoder_attn.q_proj"),
                    crossKProj: try linear("\(base).encoder_attn.k_proj", hasBias: false),
                    crossVProj: try linear("\(base).encoder_attn.v_proj"),
                    crossOProj: try linear("\(base).encoder_attn.out_proj"),
                    finalLayerNorm: try ln("\(base).final_layer_norm"),
                    fc1: try linear("\(base).fc1"),
                    fc2: try linear("\(base).fc2"),
                    hidden: wc.hidden, nHeads: wc.decoderHeads,
                    intermediate: wc.intermediate))
        }

        return WhisperModel(
            config: wc, encoder: encoder,
            tokenEmbedding: try t("decoder.embed_tokens.weight"),
            decoderPosEmbedding: try t("decoder.embed_positions.weight"),
            decoderLayers: decLayers,
            decoderLayerNorm: try ln("decoder.layer_norm"),
            dtype: dtype)
    }
}
