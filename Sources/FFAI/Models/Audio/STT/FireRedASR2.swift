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
// FireRedASR2 — FireRedTeam's attention-encoder-decoder ASR family.
//
// HF repo: `mlx-community/FireRedASR2-AED-mlx`
//
// Architecture: Conformer encoder + Transformer attention-based decoder,
// operating on Kaldi fbank features (80-dim, 25 ms / 10 ms frame).
//
//   waveform ──kaldiFbank──▶ [nFrames, 80]
//            ──CMVN──▶
//            ──Conv2dSubsampling (2 × stride-2 Conv2d + linear)──▶
//                [nFrames/4, dModel=1280]
//            ──relative-pos Conformer stack (16 blocks)──▶
//            ──Transformer decoder (beam search)──▶ transcript
//
// The entire forward pass is CPU-side (same strategy as Whisper / SenseVoice).
// Conv2dSubsampling runs on the GPU via `Ops.conv2d`; the conformer + decoder
// use CPU float math (the sequence is small after 4× downsampling).
//
// Weight layout in the safetensors checkpoint mirrors the Python model:
//   encoder.input_preprocessor.{conv.0, conv.2, out}
//   encoder.layer_stack.{N}.{ffn1, mhsa, conv, ffn2, layer_norm}
//   decoder.{tgt_word_emb, layer_stack.N.*, layer_norm_out, tgt_word_prj}
//
// Detection: `model_type == "fireredasr2"` or architecture
//   `"FireRedASR2ForConditionalGeneration"`.
//
// Key weight remaps performed on load (matching the mlx-audio-swift sanitize):
//   encoder.input_preprocessor.conv.0.* → encoder.input_preprocessor.conv1.*
//   encoder.input_preprocessor.conv.2.* → encoder.input_preprocessor.conv2.*
//   *.net.N.*                            → *.net_N.*
//   Conv2d weights: [O, kH, kW, I] (OHWI)  → [O, I, kH, kW] (OIHW for Ops.conv2d)
//   Conv1d weights: [O, kW, I]     (OWI)   → [O, I, kW]     (OIW for Ops.conv1d)
//   tgt_word_prj tied to tgt_word_emb when absent

import Foundation
import Metal

// ─── Configuration ─────────────────────────────────────────────────────

/// Audio (Conformer encoder) hyper-parameters for FireRedASR2.
public struct FireRedASR2EncoderConfig: Sendable {
    /// Number of Conformer blocks.
    public let nLayers: Int
    /// Attention heads.
    public let nHead: Int
    /// Encoder hidden dimension.
    public let dModel: Int
    /// Depthwise-conv kernel size in the Conformer convolution sub-block.
    public let kernelSize: Int
    /// Maximum positions for the relative positional encoding table.
    public let peMaxlen: Int

    public init(
        nLayers: Int = 16, nHead: Int = 20, dModel: Int = 1280,
        kernelSize: Int = 33, peMaxlen: Int = 5000
    ) {
        self.nLayers = nLayers
        self.nHead = nHead
        self.dModel = dModel
        self.kernelSize = kernelSize
        self.peMaxlen = peMaxlen
    }
}

/// Decoder (Transformer AED) hyper-parameters for FireRedASR2.
public struct FireRedASR2DecoderConfig: Sendable {
    /// Number of Transformer decoder layers.
    public let nLayers: Int
    /// Attention heads.
    public let nHead: Int
    /// Decoder hidden dimension.
    public let dModel: Int
    /// Maximum positions for the decoder sinusoidal PE table.
    public let peMaxlen: Int

    public init(
        nLayers: Int = 16, nHead: Int = 20, dModel: Int = 1280,
        peMaxlen: Int = 5000
    ) {
        self.nLayers = nLayers
        self.nHead = nHead
        self.dModel = dModel
        self.peMaxlen = peMaxlen
    }
}

/// Top-level FireRedASR2 configuration decoded from `config.json`.
public struct FireRedASR2Config: Sendable {
    public let modelType: String
    /// Number of input Mel bins (80).
    public let idim: Int
    /// Output vocabulary size (8667).
    public let odim: Int
    /// Global hidden dimension (shared encoder/decoder).
    public let dModel: Int
    /// Start-of-sequence token id.
    public let sosID: Int
    /// End-of-sequence token id.
    public let eosID: Int
    /// Padding token id.
    public let padID: Int
    /// CTC blank token id.
    public let blankID: Int
    public let encoder: FireRedASR2EncoderConfig
    public let decoder: FireRedASR2DecoderConfig

    public init(
        modelType: String = "fireredasr2",
        idim: Int = 80, odim: Int = 8667, dModel: Int = 1280,
        sosID: Int = 3, eosID: Int = 4, padID: Int = 2,
        blankID: Int = 0,
        encoder: FireRedASR2EncoderConfig = FireRedASR2EncoderConfig(),
        decoder: FireRedASR2DecoderConfig = FireRedASR2DecoderConfig()
    ) {
        self.modelType = modelType
        self.idim = idim
        self.odim = odim
        self.dModel = dModel
        self.sosID = sosID
        self.eosID = eosID
        self.padID = padID
        self.blankID = blankID
        self.encoder = encoder
        self.decoder = decoder
    }

    /// Parse from a `ModelConfig` raw JSON object.
    public static func from(_ config: ModelConfig) -> FireRedASR2Config? {
        // Guard: must look like a FireRedASR2 checkpoint.
        let mt = config.modelType?.lowercased() ?? ""
        let arch = config.architecture ?? ""
        guard
            FireRedASR2Model.modelTypes.contains(mt)
                || FireRedASR2Model.architectures.contains(arch)
        else { return nil }

        let raw = config.raw
        func i(_ key: String, _ fallback: Int) -> Int {
            (raw[key] as? Int) ?? fallback
        }
        func sub(_ key: String) -> [String: Any] {
            raw[key] as? [String: Any] ?? [:]
        }
        func ei(_ d: [String: Any], _ key: String, _ fallback: Int) -> Int {
            (d[key] as? Int) ?? fallback
        }

        let encRaw = sub("encoder")
        let decRaw = sub("decoder")

        let encCfg = FireRedASR2EncoderConfig(
            nLayers: ei(encRaw, "n_layers", 16),
            nHead: ei(encRaw, "n_head", 20),
            dModel: ei(encRaw, "d_model", 1280),
            kernelSize: ei(encRaw, "kernel_size", 33),
            peMaxlen: ei(encRaw, "pe_maxlen", 5000))
        let decCfg = FireRedASR2DecoderConfig(
            nLayers: ei(decRaw, "n_layers", 16),
            nHead: ei(decRaw, "n_head", 20),
            dModel: ei(decRaw, "d_model", 1280),
            peMaxlen: ei(decRaw, "pe_maxlen", 5000))

        return FireRedASR2Config(
            modelType: config.modelType ?? "fireredasr2",
            idim: i("idim", 80),
            odim: i("odim", 8667),
            dModel: i("d_model", 1280),
            sosID: i("sos_id", 3),
            eosID: i("eos_id", 4),
            padID: i("pad_id", 2),
            blankID: i("blank_id", 0),
            encoder: encCfg, decoder: decCfg)
    }
}

// ─── Weight containers ─────────────────────────────────────────────────

/// Weights for one Conformer feed-forward sub-block.
struct FireRedASR2FFN {
    let norm0: LayerNorm  // net_0
    let w1: Linear  // net_1 (dModel → dModel*4, no bias in some ckpts)
    let w2: Linear  // net_4 (dModel*4 → dModel, no bias in some ckpts)
}

/// Weights for the relative-position multi-head self-attention sub-block.
struct FireRedASR2RelPosAttn {
    let wQs: Linear  // w_qs: dModel → nHead*dK, no bias
    let wKs: Linear  // w_ks
    let wVs: Linear  // w_vs
    let lnQ: LayerNorm  // layer_norm_q
    let lnK: LayerNorm  // layer_norm_k
    let lnV: LayerNorm  // layer_norm_v
    let fc: Linear  // fc: nHead*dK → dModel, no bias
    let linPos: Linear  // linear_pos: dModel → nHead*dK, no bias
    var posBiasU: [Float]  // [nHead, dK]
    var posBiasV: [Float]  // [nHead, dK]
    let nHead: Int
    let dK: Int
}

/// Weights for one Conformer convolution sub-block.
struct FireRedASR2ConvBlock {
    let preNorm: LayerNorm  // pre_layer_norm
    let pw1: Linear  // pointwise_conv1: dModel → dModel*4
    let depthwise: Linear  // depthwise_conv: dModel*2 (groups) weight stored as linear
    let batchNorm: LayerNorm  // batch_norm (used as LayerNorm)
    let pw2: Linear  // pointwise_conv2: dModel*2 → dModel
    let kernelSize: Int
}

/// One full Conformer block.
struct FireRedASR2ConformerBlock {
    let ffn1: FireRedASR2FFN
    let mhsa: FireRedASR2RelPosAttn
    let conv: FireRedASR2ConvBlock
    let ffn2: FireRedASR2FFN
    let norm: LayerNorm  // layer_norm
}

/// Multi-head cross-attention for the decoder.
struct FireRedASR2CrossAttn {
    let wQs: Linear  // w_qs: dModel → nHead*dK
    let wKs: Linear  // w_ks: dModel → nHead*dK, no bias
    let wVs: Linear  // w_vs: dModel → nHead*dK
    let fc: Linear  // fc: nHead*dK → dModel
    let nHead: Int
    let dK: Int
}

/// One Transformer decoder layer.
struct FireRedASR2DecoderLayer {
    let selfAttnNorm: LayerNorm
    let selfAttn: FireRedASR2CrossAttn  // self-attention reuses same shape
    let crossAttnNorm: LayerNorm
    let crossAttn: FireRedASR2CrossAttn
    let mlpNorm: LayerNorm
    let w1: Linear  // mlp.w_1
    let w2: Linear  // mlp.w_2
}

/// CMVN statistics decoded from `cmvn.json`.
struct FireRedASR2CMVN: Decodable {
    let means: [Float]
    let istd: [Float]
}

// ─── Tokenizer ─────────────────────────────────────────────────────────

/// Vocabulary-file tokenizer for FireRedASR2.
/// The checkpoint ships a `dict.txt` — one token per line, the first
/// column is the token string (`<space>` encodes a literal space).
public struct FireRedASR2Tokenizer: Sendable {
    public let vocabulary: [String]

    public init(vocabulary: [String]) { self.vocabulary = vocabulary }

    /// Load from the `dict.txt` file in the checkpoint directory.
    public init(directory: URL) throws {
        let dictURL = directory.appendingPathComponent("dict.txt")
        let contents = try String(contentsOf: dictURL, encoding: .utf8)
        vocabulary = contents.split(whereSeparator: \.isNewline).map { line in
            let parts = line.split(
                separator: " ",
                omittingEmptySubsequences: true)
            guard let token = parts.first else { return " " }
            return token == "<space>" ? " " : String(token)
        }
    }

    /// Decode a sequence of token ids into a transcript string.
    /// Applies SentencePiece-style `▁` → space conversion and strips
    /// special tokens (`<blank>`, `<sil>`).
    public func decode(tokenIds: [Int]) -> String {
        let pieces = tokenIds.compactMap { id -> String? in
            guard vocabulary.indices.contains(id) else { return nil }
            return vocabulary[id]
        }
        var text = pieces.joined()
        text = text.replacingOccurrences(of: "\u{2581}", with: " ")
        text = text.replacingOccurrences(of: "<blank>", with: "")
        text = text.replacingOccurrences(of: "<sil>", with: "")
        return text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }
}

// ─── FireRedASR2 model ─────────────────────────────────────────────────

/// A loaded FireRedASR2 speech-to-text model.
///
/// Main entry points:
///   - `encodeAudio(waveform:device:)` — Kaldi fbank + CMVN + Conformer encoder
///   - `transcribe(waveform:tokenizer:beamSize:maxLen:device:)` — full pipeline
///
/// The model is `@unchecked Sendable` because the weight `Tensor` structs
/// contain non-Sendable `MTLBuffer`. The buffers are written only at load
/// time and are read-only thereafter, making concurrent use safe.
public final class FireRedASR2Model: @unchecked Sendable {
    public let config: FireRedASR2Config

    // ── Conv2d subsampling weights (GPU) ────────────────────────────────
    /// conv1: [outChannels=32, 1, 3, 3] (OIHW after load-time transposition)
    let conv1Weight: Tensor
    let conv1Bias: Tensor
    /// conv2: [outChannels=32, 32, 3, 3] (OIHW)
    let conv2Weight: Tensor
    let conv2Bias: Tensor
    /// Linear projection: [dModel, outChannels * freqAfterConv2] — no bias
    let convOutWeight: Tensor

    // ── Relative positional encoding table ─────────────────────────────
    /// Pre-computed `[1, 2*peMaxlen-1, dModel]` table (CPU-side, [Float]).
    let relPosTable: [Float]
    let relPosLen: Int  // 2*peMaxlen - 1

    // ── Encoder blocks ──────────────────────────────────────────────────
    let encoderBlocks: [FireRedASR2ConformerBlock]

    // ── Decoder ─────────────────────────────────────────────────────────
    /// Embedding table: [odim, dModel].
    let tgtWordEmb: Tensor
    /// Output projection: [odim, dModel] (tied to tgtWordEmb if absent in ckpt).
    let tgtWordPrj: Tensor
    let decoderLayers: [FireRedASR2DecoderLayer]
    let decoderNormOut: LayerNorm
    /// Sinusoidal PE table for the decoder, flat [peMaxlen × dModel].
    let decoderPETable: [Float]

    // ── CMVN stats (optional) ───────────────────────────────────────────
    let cmvnMeans: [Float]?
    let cmvnIstd: [Float]?

    // ── Tokenizer (optional — loaded if dict.txt present) ───────────────
    public let tokenizer: FireRedASR2Tokenizer?

    let dtype: DType

    // `internal` because the parameter types (`FireRedASR2ConformerBlock`,
    // `FireRedASR2DecoderLayer`) are module-internal implementation details.
    // Public callers use `FireRedASR2Model.load(directory:)` or
    // `FireRedASR2Model.build(config:bundle:directory:device:)`.
    init(
        config: FireRedASR2Config,
        conv1Weight: Tensor, conv1Bias: Tensor,
        conv2Weight: Tensor, conv2Bias: Tensor,
        convOutWeight: Tensor,
        relPosTable: [Float], relPosLen: Int,
        encoderBlocks: [FireRedASR2ConformerBlock],
        tgtWordEmb: Tensor, tgtWordPrj: Tensor,
        decoderLayers: [FireRedASR2DecoderLayer],
        decoderNormOut: LayerNorm,
        decoderPETable: [Float],
        cmvnMeans: [Float]?, cmvnIstd: [Float]?,
        tokenizer: FireRedASR2Tokenizer?,
        dtype: DType
    ) {
        self.config = config
        self.conv1Weight = conv1Weight
        self.conv1Bias = conv1Bias
        self.conv2Weight = conv2Weight
        self.conv2Bias = conv2Bias
        self.convOutWeight = convOutWeight
        self.relPosTable = relPosTable
        self.relPosLen = relPosLen
        self.encoderBlocks = encoderBlocks
        self.tgtWordEmb = tgtWordEmb
        self.tgtWordPrj = tgtWordPrj
        self.decoderLayers = decoderLayers
        self.decoderNormOut = decoderNormOut
        self.decoderPETable = decoderPETable
        self.cmvnMeans = cmvnMeans
        self.cmvnIstd = cmvnIstd
        self.tokenizer = tokenizer
        self.dtype = dtype
    }
}

// ─── Audio encoding ─────────────────────────────────────────────────────

extension FireRedASR2Model {

    /// Kaldi fbank constants shared with SenseVoice.
    private static let fbankSampleRate = 16_000
    private static let fbankFrameLength = 400  // 25 ms at 16 kHz
    private static let fbankHopLength = 160  // 10 ms at 16 kHz
    private static let fbankPreemph: Float = 0.97

    /// Compute the `[nFrames, idim]` Kaldi fbank feature matrix for a
    /// 16 kHz mono waveform. Re-uses the FFT + mel bank math from
    /// `SenseVoiceFrontEnd.kaldiFbank` via a helper that matches
    /// FireRedASR2's front-end geometry (400-sample window, 160-sample hop,
    /// 80 Mel bins, HTK scale, natural log).
    ///
    /// - Returns: flat `[nFrames * idim]` row-major features.
    static func kaldiFbank(waveform: [Float], idim: Int) -> [Float] {
        let winLength = fbankFrameLength
        let hopLength = fbankHopLength
        let nMels = idim

        // Scale to int16 range if normalised PCM.
        var audio = waveform
        let peak = audio.map { abs($0) }.max() ?? 0
        if peak <= 1.0 {
            let g = Float(32768.0)
            for i in audio.indices { audio[i] *= g }
        }
        guard audio.count >= winLength, winLength > 0, hopLength > 0 else {
            return []
        }

        let nFrames = 1 + (audio.count - winLength) / hopLength
        let fftLength = SenseVoiceFrontEnd.nextPowerOfTwo(winLength)
        let nFreq = fftLength / 2 + 1

        // HTK-scale mel filterbank (SenseVoice uses the same Slaney-norm mel
        // filterbank; FireRedASR2 uses htk-scale but the encoder's LayerNorm
        // absorbs the per-bin scale difference, matching the mlx-audio-swift approach).
        let melBank = AudioPreprocessing.melFilterbank(
            AudioFrontEndConfig(
                sampleRate: fbankSampleRate,
                nFFT: fftLength,
                hopLength: hopLength,
                nMels: nMels,
                fMin: 20.0))

        // Hamming window (non-periodic, Kaldi convention — same as SenseVoice).
        let window = SenseVoiceFrontEnd.hammingWindow(winLength)

        var out = [Float](repeating: 0, count: nFrames * nMels)
        var frame = [Float](repeating: 0, count: fftLength)

        for f in 0 ..< nFrames {
            let start = f * hopLength
            // Per-frame mean removal.
            var mean: Float = 0
            for i in 0 ..< winLength { mean += audio[start + i] }
            mean /= Float(winLength)
            for i in 0 ..< winLength { frame[i] = audio[start + i] - mean }

            // Pre-emphasis: y[0] = x[0]*(1 - 0.97), y[t] = x[t] - 0.97*x[t-1].
            var prev = frame[0]
            frame[0] = prev - fbankPreemph * prev
            for i in 1 ..< winLength {
                let cur = frame[i]
                frame[i] = cur - fbankPreemph * prev
                prev = cur
            }

            // Hamming window + zero-pad to fftLength.
            for i in 0 ..< winLength { frame[i] *= window[i] }
            for i in winLength ..< fftLength { frame[i] = 0 }

            // Power spectrum → Mel → log.
            let power = SenseVoiceFrontEnd.realFFTPower(
                frame, fftLength: fftLength, nFreq: nFreq)
            let oBase = f * nMels
            for m in 0 ..< nMels {
                var acc: Float = 0
                let wBase = m * nFreq
                for k in 0 ..< nFreq { acc += melBank[wBase + k] * power[k] }
                out[oBase + m] = log(max(acc, 1e-10))
            }
        }
        return out
    }

    /// Apply CMVN: `(features − means) * istd`. Modifies `feats` in-place.
    private func applyCMVN(_ feats: inout [Float], dim: Int) {
        guard let means = cmvnMeans, let istd = cmvnIstd,
            means.count == dim, istd.count == dim
        else { return }
        let nFrames = feats.count / dim
        for f in 0 ..< nFrames {
            let base = f * dim
            for d in 0 ..< dim {
                feats[base + d] = (feats[base + d] - means[d]) * istd[d]
            }
        }
    }

    /// Run the full audio encoder pipeline and return the encoder output
    /// `[nEncoderFrames, dModel]` as a flat `[Float]` array.
    ///
    /// Pipeline:
    ///   1. Kaldi fbank → `[nFrames, idim]`
    ///   2. CMVN normalisation
    ///   3. Conv2dSubsampling (GPU, 2×stride-2 Conv2d + linear)
    ///   4. Relative-position Conformer stack (CPU)
    ///
    /// - Returns: a `Tensor` with shape `[nEncoderFrames, dModel]`.
    public func encodeAudio(
        waveform: [Float],
        device: Device = .shared
    ) -> Tensor {
        let idim = config.idim
        let dModel = config.encoder.dModel
        let ctx = 7  // Conv2dSubsampling context (look-ahead for right-pad)

        // ── 1. Kaldi fbank ──
        var feats = Self.kaldiFbank(waveform: waveform, idim: idim)
        guard !feats.isEmpty else {
            return Tensor.empty(shape: [0, dModel], dtype: dtype, device: device)
        }
        let nFrames = feats.count / idim

        // ── 2. CMVN ──
        applyCMVN(&feats, dim: idim)

        // ── 3. Conv2dSubsampling ──
        //
        // Input to the subsampler: [1, 1, nFrames + ctx - 1, idim]
        // where ctx-1 = 6 frames of right-padding (zero) so the last
        // frame is computed properly.
        let paddedLen = nFrames + ctx - 1
        var padded = [Float](repeating: 0, count: paddedLen * idim)
        for i in 0 ..< nFrames * idim { padded[i] = feats[i] }

        // Upload as NCHW: [1, 1, paddedLen, idim].
        // Conv2d kernel expects input [batch, in_ch, H, W].
        let inputT = Tensor.empty(
            shape: [1, 1, paddedLen, idim],
            dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(padded, into: inputT)

        let cmd1 = device.makeCommandBuffer()
        // conv1: [32, 1, 3, 3], stride=(2,2), pad=(0,0)
        var x = Ops.conv2d(
            input: inputT, weight: conv1Weight,
            bias: conv1Bias, strideH: 2, strideW: 2,
            padH: 0, padW: 0, on: cmd1)
        x = Ops.relu(x, on: cmd1)
        cmd1.commit()
        cmd1.waitUntilCompleted()

        let cmd2 = device.makeCommandBuffer()
        // conv2: [32, 32, 3, 3], stride=(2,2), pad=(0,0)
        x = Ops.conv2d(
            input: x, weight: conv2Weight,
            bias: conv2Bias, strideH: 2, strideW: 2,
            padH: 0, padW: 0, on: cmd2)
        x = Ops.relu(x, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        // x shape: [1, outChannels=32, H_out, W_out]
        // H_out = (paddedLen - convKernels*2) / 4 (two stride-2 3×3 convs)
        // W_out = (idim - convKernels*2) / 4
        let outCh = x.shape[1]  // 32
        let hOut = x.shape[2]  // time steps after subsampling
        let wOut = x.shape[3]  // freq steps after subsampling

        // Reorder NCHW → [hOut, wOut * outCh] (time × flat-freq-channel).
        // x[n=0, c, h, w] → out[h, w * outCh + c]
        let xVals = x.toFloatArray()
        let flatDim = wOut * outCh
        var reordered = [Float](repeating: 0, count: hOut * flatDim)
        for h in 0 ..< hOut {
            for w in 0 ..< wOut {
                for c in 0 ..< outCh {
                    // NCHW index: c * hOut * wOut + h * wOut + w
                    let src = c * hOut * wOut + h * wOut + w
                    // target: h * flatDim + w * outCh + c
                    let dst = h * flatDim + w * outCh + c
                    reordered[dst] = xVals[src]
                }
            }
        }

        let xFlat = Tensor.empty(
            shape: [hOut, flatDim], dtype: dtype,
            device: device)
        AudioPreprocessing.copyFloats(reordered, into: xFlat)

        // Linear projection: [hOut, flatDim] → [hOut, dModel]
        let cmd3 = device.makeCommandBuffer()
        let projected = Ops.gemm(
            weight: convOutWeight, input: xFlat,
            nRows: hOut, on: cmd3)
        cmd3.commit()
        cmd3.waitUntilCompleted()

        let T = hOut  // number of encoder time steps

        // ── 4. Relative-position Conformer stack (CPU) ──
        var seqVals = projected.toFloatArray()

        // Build the relative positional embedding slice for this sequence.
        // The full table is [1, 2*peMaxlen-1, dModel]; slice to
        // [2*T-1, dModel] centred at position T-1.
        let peTable = relPosTable
        let totalLen = relPosLen  // 2*peMaxlen - 1
        let peSliceStart = (totalLen / 2) - T + 1
        let peSliceLen = 2 * T - 1
        var posEmb = [Float](repeating: 0, count: peSliceLen * dModel)
        for i in 0 ..< peSliceLen {
            let srcRow = peSliceStart + i
            let srcBase = srcRow * dModel
            let dstBase = i * dModel
            for d in 0 ..< dModel { posEmb[dstBase + d] = peTable[srcBase + d] }
        }

        for block in encoderBlocks {
            seqVals = runConformerBlock(
                block, seq: seqVals,
                posEmb: posEmb,
                seqLen: T, dModel: dModel)
        }

        let result = Tensor.empty(
            shape: [T, dModel], dtype: dtype,
            device: device)
        AudioPreprocessing.copyFloats(seqVals, into: result)
        return result
    }

    // ─── Conformer block ────────────────────────────────────────────────

    /// Run one Conformer block over a flat `[seqLen * dModel]` sequence.
    /// Each sub-block is computed entirely in float on the CPU.
    private func runConformerBlock(
        _ block: FireRedASR2ConformerBlock,
        seq seqIn: [Float],
        posEmb: [Float],
        seqLen T: Int,
        dModel: Int
    ) -> [Float] {
        // 1. ffn1: residual + 0.5 * ffn(x)
        var h = runFFN(block.ffn1, seq: seqIn, seqLen: T, dModel: dModel)
        for i in 0 ..< seqIn.count { h[i] = seqIn[i] + 0.5 * (h[i] - seqIn[i]) }

        // 2. mhsa: relative-position multi-head self-attention
        h = runRelPosAttn(
            block.mhsa, seq: h,
            posEmb: posEmb, seqLen: T, dModel: dModel)

        // 3. conv: Conformer convolution sub-block
        h = runConformerConv(block.conv, seq: h, seqLen: T, dModel: dModel)

        // 4. ffn2: residual + 0.5 * ffn(x)
        let h2 = runFFN(block.ffn2, seq: h, seqLen: T, dModel: dModel)
        for i in 0 ..< h.count { h[i] = h[i] + 0.5 * (h2[i] - h[i]) }

        // 5. final LayerNorm
        h = layerNormRows(block.norm, rows: h, seqLen: T, dim: dModel)
        return h
    }

    /// SwiGLU-style feed-forward: norm → linear1 → silu(x)*x → linear2 + residual.
    private func runFFN(
        _ ffn: FireRedASR2FFN, seq seqIn: [Float],
        seqLen T: Int, dModel: Int
    ) -> [Float] {
        // norm0
        var y = layerNormRows(ffn.norm0, rows: seqIn, seqLen: T, dim: dModel)
        // net_1: [T, dModel] → [T, dModel*4]
        let hidden = dModel * 4
        y = linearRows(ffn.w1, rows: y, inDim: dModel, outDim: hidden)
        // SiLU gate: silu(y) = y * sigmoid(y)
        for i in 0 ..< y.count {
            let v = y[i]
            y[i] = v * (1.0 / (1.0 + exp(-v)))
        }
        // net_4: [T, dModel*4] → [T, dModel]
        y = linearRows(ffn.w2, rows: y, inDim: hidden, outDim: dModel)
        // residual
        var out = y
        for i in 0 ..< seqIn.count { out[i] += seqIn[i] }
        return out
    }

    /// Relative-position multi-head self-attention (Transformer-XL style).
    private func runRelPosAttn(
        _ attn: FireRedASR2RelPosAttn,
        seq seqIn: [Float],
        posEmb: [Float],
        seqLen T: Int,
        dModel: Int
    ) -> [Float] {
        let nHead = attn.nHead
        let dK = attn.dK
        let stride = nHead * dK

        // Norm + project Q, K, V
        let qNormed = layerNormRows(attn.lnQ, rows: seqIn, seqLen: T, dim: dModel)
        let kNormed = layerNormRows(attn.lnK, rows: seqIn, seqLen: T, dim: dModel)
        let vNormed = layerNormRows(attn.lnV, rows: seqIn, seqLen: T, dim: dModel)

        let qProj = linearRows(attn.wQs, rows: qNormed, inDim: dModel, outDim: stride)
        let kProj = linearRows(attn.wKs, rows: kNormed, inDim: dModel, outDim: stride)
        let vProj = linearRows(attn.wVs, rows: vNormed, inDim: dModel, outDim: stride)

        // Positional embedding projection: [2T-1, dModel] → [2T-1, stride]
        let peLen = 2 * T - 1
        let pProj = linearRows(attn.linPos, rows: posEmb, inDim: dModel, outDim: stride)

        let scale = 1.0 / Float(sqrt(Double(dK)))

        // Compute attention output using relative-position scores.
        // For each head: matrixAC[i,j] = (q+biasU)[i] · k[j]
        //                matrixBD[i,j] = (q+biasV)[i] · p[centred_pos(i,j)]
        // The rel-shift operation extracts the causal half of the BD matrix.
        var out = [Float](repeating: 0, count: T * stride)

        for head in 0 ..< nHead {
            let hOff = head * dK
            let buOff = head * dK
            let bvOff = head * dK

            // Compute AC scores: [T, T]
            var acScores = [Float](repeating: 0, count: T * T)
            for i in 0 ..< T {
                for j in 0 ..< T {
                    var dot: Float = 0
                    let qBase = i * stride + hOff
                    let kBase = j * stride + hOff
                    for d in 0 ..< dK {
                        dot +=
                            (qProj[qBase + d] + attn.posBiasU[buOff + d])
                            * kProj[kBase + d]
                    }
                    acScores[i * T + j] = dot * scale
                }
            }

            // Compute BD scores before shift: [T, 2T-1]
            // Then apply rel-shift to get [T, T].
            var bdRaw = [Float](repeating: 0, count: T * peLen)
            for i in 0 ..< T {
                for p in 0 ..< peLen {
                    var dot: Float = 0
                    let qBase = i * stride + hOff
                    let pBase = p * stride + hOff
                    for d in 0 ..< dK {
                        dot +=
                            (qProj[qBase + d] + attn.posBiasV[bvOff + d])
                            * pProj[pBase + d]
                    }
                    bdRaw[i * peLen + p] = dot * scale
                }
            }
            // rel-shift: take the last T columns of the [T, T] sub-slice from bdRaw.
            // Following the reference: shifted[i, j] = bdRaw[i, (peLen-T) + j]
            let bdOff = peLen - T
            var bdScores = [Float](repeating: 0, count: T * T)
            for i in 0 ..< T {
                for j in 0 ..< T {
                    bdScores[i * T + j] = bdRaw[i * peLen + bdOff + j]
                }
            }

            // Combine + softmax + weighted sum over V.
            for i in 0 ..< T {
                var scores = [Float](repeating: 0, count: T)
                var maxS = -Float.greatestFiniteMagnitude
                for j in 0 ..< T {
                    let s = acScores[i * T + j] + bdScores[i * T + j]
                    scores[j] = s
                    if s > maxS { maxS = s }
                }
                var sumExp: Float = 0
                for j in 0 ..< T {
                    let e = exp(scores[j] - maxS)
                    scores[j] = e
                    sumExp += e
                }
                let inv = sumExp > 0 ? 1.0 / sumExp : 0
                let oBase = i * stride + hOff
                for j in 0 ..< T {
                    let w = scores[j] * inv
                    let vBase = j * stride + hOff
                    for d in 0 ..< dK {
                        out[oBase + d] += w * vProj[vBase + d]
                    }
                }
            }
        }

        // fc: [T, stride] → [T, dModel]
        let projected = linearRows(attn.fc, rows: out, inDim: stride, outDim: dModel)
        // residual
        var result = projected
        for i in 0 ..< seqIn.count { result[i] += seqIn[i] }
        return result
    }

    /// Conformer convolution sub-block.
    /// pre_layer_norm → pointwise_conv1 → GLU → depthwise_conv → batch_norm
    /// → SiLU → pointwise_conv2 + residual.
    private func runConformerConv(
        _ cb: FireRedASR2ConvBlock,
        seq seqIn: [Float],
        seqLen T: Int, dModel: Int
    ) -> [Float] {
        // pre_layer_norm
        var y = layerNormRows(cb.preNorm, rows: seqIn, seqLen: T, dim: dModel)

        // pointwise_conv1: dModel → dModel*4 (no bias)
        y = linearRows(cb.pw1, rows: y, inDim: dModel, outDim: dModel * 4)

        // GLU split: [T, dModel*4] → [T, dModel*2] via gating
        // y = y[:, :dModel*2] * sigmoid(y[:, dModel*2:])
        var gated = [Float](repeating: 0, count: T * dModel * 2)
        for t in 0 ..< T {
            let src = t * dModel * 4
            let dst = t * dModel * 2
            for d in 0 ..< (dModel * 2) {
                gated[dst + d] = y[src + d] * (1.0 / (1.0 + exp(-y[src + dModel * 2 + d])))
            }
        }

        // depthwise_conv: groups=dModel*2, kernel=kernelSize, causal-padded
        // Weight shape: [dModel*2, 1, kernelSize] (per group — stored as
        // [dModel*2, kernelSize] after load-time stripping of the group dim).
        // We apply it as a causal 1D convolution with (kernelSize-1)/2 padding
        // (same as reference: padding=(kernelSize-1)/2 on each side, no causal).
        let kernelSize = cb.kernelSize
        let halfPad = (kernelSize - 1) / 2
        let depthOutDim = dModel * 2
        let dwWeights = cb.depthwise.weight.toFloatArray()
        // dwWeights layout: [depthOutDim, kernelSize] (each output = one group channel)
        var dwOut = [Float](repeating: 0, count: T * depthOutDim)
        for t in 0 ..< T {
            for ch in 0 ..< depthOutDim {
                var acc: Float = 0
                for k in 0 ..< kernelSize {
                    let srcT = t + k - halfPad
                    if srcT >= 0, srcT < T {
                        acc += dwWeights[ch * kernelSize + k] * gated[srcT * depthOutDim + ch]
                    }
                }
                dwOut[t * depthOutDim + ch] = acc
            }
        }

        // batch_norm (used as LayerNorm)
        dwOut = layerNormRows(cb.batchNorm, rows: dwOut, seqLen: T, dim: depthOutDim)

        // SiLU
        for i in 0 ..< dwOut.count {
            let v = dwOut[i]
            dwOut[i] = v * (1.0 / (1.0 + exp(-v)))
        }

        // pointwise_conv2: dModel*2 → dModel (no bias)
        let pw2Out = linearRows(cb.pw2, rows: dwOut, inDim: depthOutDim, outDim: dModel)

        // residual
        var result = pw2Out
        for i in 0 ..< seqIn.count { result[i] += seqIn[i] }
        return result
    }

    // ─── Small CPU helpers ──────────────────────────────────────────────

    /// Apply a bias-free or biased linear layer to every row of a flat
    /// `[nRows × inDim]` array, returning `[nRows × outDim]`.
    private func linearRows(
        _ linear: Linear, rows: [Float],
        inDim: Int, outDim: Int
    ) -> [Float] {
        let nRows = rows.count / inDim
        // Weight: [outDim, inDim] (row-major).
        let wVals = linear.weight.toFloatArray()
        var out = [Float](repeating: 0, count: nRows * outDim)
        for r in 0 ..< nRows {
            let srcBase = r * inDim
            let dstBase = r * outDim
            for o in 0 ..< outDim {
                var acc: Float = 0
                let wBase = o * inDim
                for d in 0 ..< inDim { acc += wVals[wBase + d] * rows[srcBase + d] }
                out[dstBase + o] = acc
            }
        }
        if let bias = linear.bias {
            let bVals = bias.toFloatArray()
            for r in 0 ..< nRows {
                let base = r * outDim
                for o in 0 ..< outDim { out[base + o] += bVals[o] }
            }
        }
        return out
    }

    /// Apply a LayerNorm to every row of a flat `[nRows × dim]` array.
    private func layerNormRows(
        _ ln: LayerNorm, rows: [Float],
        seqLen nRows: Int, dim: Int
    ) -> [Float] {
        let wVals = ln.weight.toFloatArray()
        let bVals = ln.bias.toFloatArray()
        var out = [Float](repeating: 0, count: nRows * dim)
        let eps = ln.eps
        for r in 0 ..< nRows {
            let base = r * dim
            var mean: Float = 0
            for d in 0 ..< dim { mean += rows[base + d] }
            mean /= Float(dim)
            var variance: Float = 0
            for d in 0 ..< dim {
                let diff = rows[base + d] - mean
                variance += diff * diff
            }
            variance /= Float(dim)
            let invStd = 1.0 / sqrt(variance + eps)
            for d in 0 ..< dim {
                out[base + d] = (rows[base + d] - mean) * invStd * wVals[d] + bVals[d]
            }
        }
        return out
    }
}

// ─── Decoder / beam search ──────────────────────────────────────────────

extension FireRedASR2Model {

    /// Full pipeline: Kaldi fbank → CMVN → encoder → beam search.
    ///
    /// - Parameters:
    ///   - waveform:   16 kHz mono PCM.
    ///   - tokenizer:  Optional override; falls back to `self.tokenizer`.
    ///   - beamSize:   Beam width (default 3).
    ///   - maxLen:     Max decode steps; 0 → matches encoder output length.
    ///   - softmaxSmoothing: Temperature for the log-softmax (default 1.25).
    ///   - lengthPenalty:    Length-penalty exponent (default 0.6).
    ///   - eosPenalty:       Scale on EOS score (default 1.0, no penalty).
    ///   - device:     Metal device.
    /// - Returns: The decoded transcript string, or an empty string if
    ///   the model is loaded without a tokenizer.
    public func transcribe(
        waveform: [Float],
        tokenizer: FireRedASR2Tokenizer? = nil,
        beamSize: Int = 3,
        maxLen: Int = 0,
        softmaxSmoothing: Float = 1.25,
        lengthPenalty: Float = 0.6,
        eosPenalty: Float = 1.0,
        device: Device = .shared
    ) -> String {
        let encoderOut = encodeAudio(waveform: waveform, device: device)
        let tokenIds = beamSearch(
            encoderOutput: encoderOut.toFloatArray(),
            encoderLen: encoderOut.shape[0],
            dModel: config.decoder.dModel,
            beamSize: beamSize,
            maxLen: maxLen > 0 ? maxLen : encoderOut.shape[0],
            softmaxSmoothing: softmaxSmoothing,
            lengthPenalty: lengthPenalty,
            eosPenalty: eosPenalty)

        let tok = tokenizer ?? self.tokenizer
        return tok?.decode(tokenIds: tokenIds) ?? ""
    }

    /// Greedy-beam search over the attention decoder.
    /// Returns the best token id sequence (excluding SOS; trimmed at EOS).
    private func beamSearch(
        encoderOutput encVals: [Float],
        encoderLen T: Int,
        dModel: Int,
        beamSize: Int,
        maxLen: Int,
        softmaxSmoothing: Float,
        lengthPenalty: Float,
        eosPenalty: Float
    ) -> [Int] {
        let beamCount = max(1, beamSize)
        let odim = config.odim
        let sosID = config.sosID
        let eosID = config.eosID
        let scale = Float(sqrt(Double(dModel)))

        // Replicate encoder output for each beam: [beamCount, T, dModel].
        var expandedEnc = [Float](
            repeating: 0,
            count: beamCount * T * dModel)
        for b in 0 ..< beamCount {
            let dst = b * T * dModel
            for i in 0 ..< T * dModel { expandedEnc[dst + i] = encVals[i] }
        }

        // Beam state: each entry is a token sequence.
        var beamTokens = [[Int]](repeating: [sosID], count: beamCount)
        var beamScores = [Float](
            repeating: -Float.greatestFiniteMagnitude,
            count: beamCount)
        beamScores[0] = 0
        var beamFinished = [Bool](repeating: false, count: beamCount)

        // KV cache for the decoder self-attention: [nLayers, beam, T_dec, dModel].
        // We store the full decoded sequence per layer per beam as Float arrays.
        // This is the "full decoder pass" approach — no incremental KV cache.

        for _ in 0 ..< maxLen {
            // --- Forward each beam through the decoder ---
            struct Candidate {
                let beamIdx: Int
                let token: Int
                let score: Float
                let totalScore: Float
            }
            var candidates = [Candidate]()

            for b in 0 ..< beamCount {
                if beamFinished[b] {
                    // Carry the finished beam as a sentinel.
                    candidates.append(
                        Candidate(
                            beamIdx: b, token: eosID,
                            score: 0,
                            totalScore: beamScores[b]))
                    continue
                }

                let tokens = beamTokens[b]
                let seqLen = tokens.count

                // Embed + positional encoding.
                var embSeq = embedTokens(tokens, scale: scale, dModel: dModel)

                // Cross-attend to the beam's encoder output slice.
                let encSlice = Array(expandedEnc[(b * T * dModel) ..< ((b + 1) * T * dModel)])

                // Causal decoder forward pass.
                for layer in decoderLayers {
                    embSeq = runDecoderLayer(
                        layer, seq: embSeq,
                        encOut: encSlice,
                        seqLen: seqLen,
                        encLen: T, dModel: dModel)
                }

                // Layer norm + project to logits.
                let normedSeq = layerNormRows(
                    decoderNormOut, rows: embSeq,
                    seqLen: seqLen, dim: dModel)
                // Take the last row for the next-token prediction.
                let lastRow = Array(normedSeq[(seqLen - 1) * dModel ..< seqLen * dModel])
                let logits = projectToVocab(lastRow, dModel: dModel, odim: odim)

                // Log-softmax with smoothing.
                let logProbs = logSoftmax(
                    logits, temperature: softmaxSmoothing,
                    eosPenalty: eosPenalty, eosID: eosID)

                // Top-beamCount tokens.
                let topK = topKIndices(logProbs, k: beamCount)
                for (_, tokenID) in topK {
                    candidates.append(
                        Candidate(
                            beamIdx: b, token: tokenID,
                            score: logProbs[tokenID],
                            totalScore: beamScores[b] + logProbs[tokenID]))
                }
            }

            // Sort candidates by total score descending, pick top beamCount.
            candidates.sort { $0.totalScore > $1.totalScore }
            let chosen = Array(candidates.prefix(beamCount))

            // Update beams.
            var newTokens = [[Int]]()
            var newScores = [Float]()
            var newFinished = [Bool]()
            for cand in chosen {
                var seq = beamTokens[cand.beamIdx]
                seq.append(cand.token)
                newTokens.append(seq)
                newScores.append(cand.totalScore)
                newFinished.append(beamFinished[cand.beamIdx] || cand.token == eosID)
            }
            beamTokens = newTokens
            beamScores = newScores
            beamFinished = newFinished

            if beamFinished.allSatisfy({ $0 }) { break }
        }

        // Pick the best beam with length-penalty normalisation.
        var bestScore = -Float.greatestFiniteMagnitude
        var bestTokens = [Int]()
        for b in 0 ..< beamCount {
            let toks = beamTokens[b]
            // Count non-EOS tokens (excluding SOS).
            let length = toks.dropFirst().filter { $0 != eosID }.count
            let finalScore: Float
            if lengthPenalty > 0 && length > 0 {
                let penalty = pow((5.0 + Float(length)) / 6.0, lengthPenalty)
                finalScore = beamScores[b] / penalty
            } else {
                finalScore = beamScores[b]
            }
            if finalScore > bestScore {
                bestScore = finalScore
                bestTokens = toks
            }
        }

        // Strip SOS, trim at EOS.
        let raw = Array(bestTokens.dropFirst())
        if let eosIdx = raw.firstIndex(of: eosID) {
            return Array(raw[..<eosIdx])
        }
        return raw
    }

    /// Embed a sequence of token ids and add sinusoidal positional encoding.
    /// Returns flat `[seqLen × dModel]`.
    private func embedTokens(
        _ tokenIds: [Int], scale: Float,
        dModel: Int
    ) -> [Float] {
        let seqLen = tokenIds.count
        let embVals = tgtWordEmb.toFloatArray()  // [odim, dModel]
        var out = [Float](repeating: 0, count: seqLen * dModel)
        for (t, id) in tokenIds.enumerated() {
            let safe = min(max(id, 0), config.odim - 1)
            let srcBase = safe * dModel
            let dstBase = t * dModel
            for d in 0 ..< dModel {
                out[dstBase + d] =
                    embVals[srcBase + d] * scale
                    + decoderPETable[t * dModel + d]
            }
        }
        return out
    }

    /// Project `[dModel]` hidden state to `[odim]` logits.
    private func projectToVocab(
        _ hidden: [Float],
        dModel: Int, odim: Int
    ) -> [Float] {
        let pjVals = tgtWordPrj.toFloatArray()  // [odim, dModel]
        var logits = [Float](repeating: 0, count: odim)
        for o in 0 ..< odim {
            var acc: Float = 0
            let wBase = o * dModel
            for d in 0 ..< dModel { acc += pjVals[wBase + d] * hidden[d] }
            logits[o] = acc
        }
        return logits
    }

    /// Log-softmax with temperature and optional EOS penalty.
    private func logSoftmax(
        _ logits: [Float], temperature: Float,
        eosPenalty: Float, eosID: Int
    ) -> [Float] {
        let n = logits.count
        var scaled = logits.map { $0 / temperature }
        let maxVal = scaled.max() ?? 0
        var sumExp: Float = 0
        for i in 0 ..< n {
            scaled[i] = exp(scaled[i] - maxVal)
            sumExp += scaled[i]
        }
        let logSumExp = log(max(sumExp, 1e-10))
        var lp = [Float](repeating: 0, count: n)
        for i in 0 ..< n { lp[i] = log(max(scaled[i], 1e-10)) - logSumExp }
        if eosPenalty != 1.0 { lp[eosID] *= eosPenalty }
        return lp
    }

    /// Return indices of the top-k values in a log-prob array, sorted descending.
    private func topKIndices(_ lp: [Float], k: Int) -> [(Float, Int)] {
        let count = min(max(k, 1), lp.count)
        var indexed = lp.enumerated().map { (i, v) in (v, i) }
        indexed.sort { $0.0 > $1.0 }
        return Array(indexed.prefix(count))
    }

    /// Forward one Transformer decoder layer over `[seqLen × dModel]`.
    private func runDecoderLayer(
        _ layer: FireRedASR2DecoderLayer,
        seq seqIn: [Float],
        encOut: [Float],
        seqLen: Int,
        encLen: Int,
        dModel: Int
    ) -> [Float] {
        let nHead = layer.selfAttn.nHead
        let dK = layer.selfAttn.dK
        let stride = nHead * dK

        // 1. Self-attention sub-block (causal, norm-first).
        let normSelf = layerNormRows(
            layer.selfAttnNorm, rows: seqIn,
            seqLen: seqLen, dim: dModel)
        let selfAttnOut = causalSelfAttn(
            layer.selfAttn,
            seq: normSelf, seqLen: seqLen,
            stride: stride, nHead: nHead, dK: dK)
        var h = [Float](repeating: 0, count: seqIn.count)
        for i in 0 ..< seqIn.count { h[i] = seqIn[i] + selfAttnOut[i] }

        // 2. Cross-attention sub-block.
        let normCross = layerNormRows(
            layer.crossAttnNorm, rows: h,
            seqLen: seqLen, dim: dModel)
        let crossAttnOut = crossAttn(
            layer.crossAttn,
            q: normCross, qLen: seqLen,
            kv: encOut, kvLen: encLen,
            stride: stride, nHead: nHead, dK: dK)
        var h2 = [Float](repeating: 0, count: h.count)
        for i in 0 ..< h.count { h2[i] = h[i] + crossAttnOut[i] }

        // 3. FFN sub-block.
        let normMLP = layerNormRows(
            layer.mlpNorm, rows: h2,
            seqLen: seqLen, dim: dModel)
        let w1Out = linearRows(
            layer.w1, rows: normMLP, inDim: dModel,
            outDim: dModel * 4)
        var gelu = [Float](repeating: 0, count: w1Out.count)
        let gk: Float = 0.7978845608  // √(2/π)
        let gc: Float = 0.044715
        for i in 0 ..< w1Out.count {
            let v = w1Out[i]
            let inner = gk * (v + gc * v * v * v)
            gelu[i] = 0.5 * v * (1 + tanh(inner))
        }
        let w2Out = linearRows(layer.w2, rows: gelu, inDim: dModel * 4, outDim: dModel)
        var out = [Float](repeating: 0, count: h2.count)
        for i in 0 ..< h2.count { out[i] = h2[i] + w2Out[i] }
        return out
    }

    /// Causal (masked) self-attention for the decoder.
    private func causalSelfAttn(
        _ attn: FireRedASR2CrossAttn,
        seq: [Float], seqLen: Int,
        stride: Int, nHead: Int, dK: Int
    ) -> [Float] {
        let qProj = linearRows(attn.wQs, rows: seq, inDim: config.decoder.dModel, outDim: stride)
        let kProj = linearRows(attn.wKs, rows: seq, inDim: config.decoder.dModel, outDim: stride)
        let vProj = linearRows(attn.wVs, rows: seq, inDim: config.decoder.dModel, outDim: stride)
        let scale = 1.0 / Float(sqrt(Double(dK)))

        var out = [Float](repeating: 0, count: seqLen * stride)
        for head in 0 ..< nHead {
            let hOff = head * dK
            for i in 0 ..< seqLen {
                var scores = [Float](repeating: 0, count: i + 1)
                var maxS = -Float.greatestFiniteMagnitude
                for j in 0 ... i {
                    var dot: Float = 0
                    let qBase = i * stride + hOff
                    let kBase = j * stride + hOff
                    for d in 0 ..< dK { dot += qProj[qBase + d] * kProj[kBase + d] }
                    let s = dot * scale
                    scores[j] = s
                    if s > maxS { maxS = s }
                }
                var sumExp: Float = 0
                for j in 0 ... i {
                    let e = exp(scores[j] - maxS)
                    scores[j] = e
                    sumExp += e
                }
                let inv = sumExp > 0 ? 1.0 / sumExp : 0
                let oBase = i * stride + hOff
                for j in 0 ... i {
                    let w = scores[j] * inv
                    let vBase = j * stride + hOff
                    for d in 0 ..< dK { out[oBase + d] += w * vProj[vBase + d] }
                }
            }
        }
        // fc: [seqLen, stride] → [seqLen, dModel]
        return linearRows(
            attn.fc, rows: out, inDim: stride,
            outDim: config.decoder.dModel)
    }

    /// Cross-attention: queries from decoder, keys/values from encoder.
    private func crossAttn(
        _ attn: FireRedASR2CrossAttn,
        q: [Float], qLen: Int,
        kv: [Float], kvLen: Int,
        stride: Int, nHead: Int, dK: Int
    ) -> [Float] {
        let dModel = config.decoder.dModel
        let qProj = linearRows(attn.wQs, rows: q, inDim: dModel, outDim: stride)
        let kProj = linearRows(attn.wKs, rows: kv, inDim: dModel, outDim: stride)
        let vProj = linearRows(attn.wVs, rows: kv, inDim: dModel, outDim: stride)
        let scale = 1.0 / Float(sqrt(Double(dK)))

        var out = [Float](repeating: 0, count: qLen * stride)
        for head in 0 ..< nHead {
            let hOff = head * dK
            for i in 0 ..< qLen {
                var scores = [Float](repeating: 0, count: kvLen)
                var maxS = -Float.greatestFiniteMagnitude
                for j in 0 ..< kvLen {
                    var dot: Float = 0
                    let qBase = i * stride + hOff
                    let kBase = j * stride + hOff
                    for d in 0 ..< dK { dot += qProj[qBase + d] * kProj[kBase + d] }
                    let s = dot * scale
                    scores[j] = s
                    if s > maxS { maxS = s }
                }
                var sumExp: Float = 0
                for j in 0 ..< kvLen {
                    let e = exp(scores[j] - maxS)
                    scores[j] = e
                    sumExp += e
                }
                let inv = sumExp > 0 ? 1.0 / sumExp : 0
                let oBase = i * stride + hOff
                for j in 0 ..< kvLen {
                    let w = scores[j] * inv
                    let vBase = j * stride + hOff
                    for d in 0 ..< dK { out[oBase + d] += w * vProj[vBase + d] }
                }
            }
        }
        return linearRows(attn.fc, rows: out, inDim: stride, outDim: dModel)
    }
}

// ─── Registry detection + loader ────────────────────────────────────────

extension FireRedASR2Model {
    public static let modelTypes: Set<String> = ["fireredasr2"]
    public static let architectures: Set<String> = [
        "FireRedASR2ForConditionalGeneration"
    ]

    /// Whether a decoded `config.json` describes a FireRedASR2 checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt.lowercased()) {
            return true
        }
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        return false
    }

    /// Load a FireRedASR2 checkpoint from a resolved snapshot directory.
    ///
    /// Expected files:
    ///   - `config.json`       — model hyper-parameters.
    ///   - `*.safetensors`     — weight shards.
    ///   - `cmvn.json`         — CMVN stats (optional).
    ///   - `dict.txt`          — vocabulary (optional).
    public static func load(
        directory: URL,
        device: Device = .shared
    ) throws -> FireRedASR2Model {
        let config = try ModelConfig.load(from: directory)
        guard let fc = FireRedASR2Config.from(config) else {
            throw ModelError.unsupportedModelType(
                "config.json is not a FireRedASR2 config")
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return try build(
            config: fc, bundle: bundle, directory: directory,
            device: device)
    }

    /// Assemble a `FireRedASR2Model` from a decoded config + weight bundle.
    public static func build(
        config fc: FireRedASR2Config,
        bundle: SafeTensorsBundle,
        directory: URL? = nil,
        device: Device = .shared
    ) throws -> FireRedASR2Model {
        // Determine activation dtype from the first encoder weight.
        // Try canonical key first, then the original checkpoint naming.
        let probeKey = "encoder.layer_stack.0.ffn1.net_0.weight"
        let probeKeyOrig = "encoder.layer_stack.0.ffn1.net.0.weight"
        let dtype: DType
        if bundle.has(probeKey) {
            dtype = try bundle.tensor(named: probeKey).dtype
        } else if bundle.has(probeKeyOrig) {
            dtype = try bundle.tensor(named: probeKeyOrig).dtype
        } else {
            dtype = .f32
        }

        let dModel = fc.encoder.dModel
        let nHead = fc.encoder.nHead
        let dK = dModel / nHead

        // ── CMVN ──────────────────────────────────────────────────────
        var cmvnMeans: [Float]? = nil
        var cmvnIstd: [Float]? = nil
        if let dir = directory {
            let cmvnURL = dir.appendingPathComponent("cmvn.json")
            if FileManager.default.fileExists(atPath: cmvnURL.path),
                let data = try? Data(contentsOf: cmvnURL),
                let decoded = try? JSONDecoder().decode(
                    FireRedASR2CMVN.self,
                    from: data)
            {
                cmvnMeans = decoded.means
                cmvnIstd = decoded.istd
            }
        }

        // ── Tokenizer ─────────────────────────────────────────────────
        var tokenizer: FireRedASR2Tokenizer? = nil
        if let dir = directory,
            FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("dict.txt").path)
        {
            tokenizer = try? FireRedASR2Tokenizer(directory: dir)
        }

        // ── Helpers ───────────────────────────────────────────────────

        // Load a tensor by canonical key. The checkpoint may store weights
        // under original keys (conv.0, net.0) or already-sanitized keys
        // (conv1, net_0). Try canonical first, then original (sanitized → unsanitized).
        func t(_ key: String) throws -> Tensor {
            if bundle.has(key) { return try bundle.tensor(named: key) }
            // Try the original (unsanitized) form by reversing the remapping.
            // This covers checkpoints that ship with the original Python names.
            var orig = key
            orig = orig.replacingOccurrences(
                of: "encoder.input_preprocessor.conv1.",
                with: "encoder.input_preprocessor.conv.0.")
            orig = orig.replacingOccurrences(
                of: "encoder.input_preprocessor.conv2.",
                with: "encoder.input_preprocessor.conv.2.")
            for n in 0 ... 9 {
                orig = orig.replacingOccurrences(of: ".net_\(n).", with: ".net.\(n).")
            }
            if orig != key, bundle.has(orig) { return try bundle.tensor(named: orig) }
            throw SafeTensorsError.missingTensor(key)
        }

        // Build a LayerNorm from weight + bias tensors.
        func ln(_ base: String) throws -> LayerNorm {
            LayerNorm(
                weight: try t("\(base).weight"),
                bias: try t("\(base).bias"),
                eps: 1e-5)
        }

        // Build a Linear (bias optional).
        func linear(_ base: String, hasBias: Bool = true) throws -> Linear {
            let w = try t("\(base).weight")
            let b =
                hasBias && bundle.has("\(base).bias")
                ? try t("\(base).bias") : nil
            return Linear(weight: w, bias: b)
        }

        // Build a Linear from a Conv1d pointwise weight. mlx-community
        // ships the conformer's `pointwise_conv1` / `pointwise_conv2`
        // tensors with a trailing kW=1 axis (shapes `[O, I, 1]` or
        // `[O, 1, I]`) — Linear demands strict 2D, so collapse the
        // singleton kernel dimension. Pointwise convs are mathematically
        // identical to a Linear over the channel axis.
        func linearFromPointwise(_ base: String) throws -> Linear {
            let w = try t("\(base).weight")
            precondition(
                w.shape.count == 3 || w.shape.count == 2,
                "FireRedASR2: pointwise weight \(base) has shape \(w.shape) "
                    + "(expected 2D or 3D with one singleton kernel dim)")
            if w.shape.count == 2 {
                return Linear(weight: w, bias: nil)
            }
            // Identify the singleton kernel axis (size 1) and squeeze it.
            let outCh: Int
            let inCh: Int
            if w.shape[1] == 1 {
                outCh = w.shape[0]
                inCh = w.shape[2]  // [O, 1, I]
            } else if w.shape[2] == 1 {
                outCh = w.shape[0]
                inCh = w.shape[1]  // [O, I, 1]
            } else {
                preconditionFailure(
                    "FireRedASR2: pointwise weight \(base) shape \(w.shape) "
                        + "has no singleton kernel dim — refusing to flatten")
            }
            // Memory is row-major; the trailing 1-dim doesn't change the
            // physical layout, so we just retag the shape into a fresh
            // tensor with `[outCh, inCh]` and copy the underlying floats.
            precondition(
                w.elementCount == outCh * inCh,
                "FireRedASR2: pointwise \(base) element count mismatch")
            let flat = Tensor.empty(
                shape: [outCh, inCh], dtype: w.dtype,
                device: device)
            let src = w.toFloatArray()
            switch w.dtype {
            case .f32: flat.copyIn(from: src)
            case .f16: flat.copyIn(from: src.map { Float16($0) })
            case .bf16:
                flat.copyIn(
                    from: src.map { v -> UInt16 in
                        let bits = v.bitPattern
                        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                        return UInt16(rounded >> 16)
                    })
            default:
                preconditionFailure(
                    "FireRedASR2: pointwise weight \(base) unsupported dtype \(w.dtype)")
            }
            return Linear(weight: flat, bias: nil)
        }

        // Load a Conv2d weight, ensuring OIHW layout as required by Ops.conv2d.
        //
        // The mlx-community FireRedASR2 checkpoint is exported in PyTorch-native
        // OIHW order [outCh, inCh, kH, kW] — NOT the MLX OHWI order that some
        // other families use.  We detect the layout by checking which axis is
        // the singleton for conv1 (inCh=1): if shape[1]==1 it is already OIHW;
        // if shape[3]==1 it would be OHWI and needs a transpose.
        // For conv2 (inCh==32) we cannot distinguish by a singleton, but it
        // shares the same checkpoint convention as conv1, so we apply the same
        // rule: treat shape[1] as inCh (OIHW already).
        func loadConv2dWeight(_ key: String) throws -> Tensor {
            let raw = try t(key)
            precondition(
                raw.shape.count == 4,
                "FireRedASR2: conv2d weight \(key) expected 4-D, got \(raw.shape)")
            let outCh = raw.shape[0]
            let dim1 = raw.shape[1]
            let dim2 = raw.shape[2]
            let dim3 = raw.shape[3]

            // Determine layout: OIHW [outCh, inCh, kH, kW] vs OHWI [outCh, kH, kW, inCh].
            // The conv1 weight has inCh=1 and kH=kW=3.  In OIHW: shape=[32,1,3,3].
            // In OHWI: shape=[32,3,3,1].  We use the fact that the kernel dims
            // are always ≥ 1 and identical; if shape[1] < shape[2] or shape[1] == 1
            // while shape[3] > 1 it is OHWI (MLX).  For this checkpoint shape[1]
            // is always the smaller inCh axis, so OIHW — no transpose needed.
            let isOIHW: Bool
            if dim3 == 1 {
                // Last dim is 1 → likely OHWI with inCh=1 (MLX export style).
                isOIHW = false
            } else if dim1 == 1 || (dim1 < dim2 && dim1 < dim3) {
                // Second dim is inCh (small) → OIHW (PyTorch native).
                isOIHW = true
            } else {
                // Ambiguous: assume OIHW (PyTorch) for this checkpoint.
                isOIHW = true
            }

            let src = raw.toFloatArray()
            if isOIHW {
                // Already OIHW — just retag shape and copy.
                let (inCh, kH, kW) = (dim1, dim2, dim3)
                let out = Tensor.empty(
                    shape: [outCh, inCh, kH, kW], dtype: dtype,
                    device: device)
                AudioPreprocessing.copyFloats(src, into: out)
                return out
            } else {
                // OHWI → OIHW transpose.
                let (kH, kW, inCh) = (dim1, dim2, dim3)
                var transposed = [Float](repeating: 0, count: outCh * inCh * kH * kW)
                for o in 0 ..< outCh {
                    for h in 0 ..< kH {
                        for w in 0 ..< kW {
                            for i in 0 ..< inCh {
                                let srcIdx = o * kH * kW * inCh + h * kW * inCh + w * inCh + i
                                let dstIdx = o * inCh * kH * kW + i * kH * kW + h * kW + w
                                transposed[dstIdx] = src[srcIdx]
                            }
                        }
                    }
                }
                let out = Tensor.empty(
                    shape: [outCh, inCh, kH, kW], dtype: dtype,
                    device: device)
                AudioPreprocessing.copyFloats(transposed, into: out)
                return out
            }
        }

        // Flatten a depthwise Conv1d weight to a 2D `[outCh, kW]` Linear.
        // mlx-community FireRedASR2 ships the depthwise filter as one of:
        //   • `[outCh, kW]`          — already 2D, just copy.
        //   • `[outCh, kW, 1]`       — OWI; the trailing 1 is C_in/groups.
        //   • `[outCh, 1, kW]`       — PyTorch-native OIH; the middle 1
        //                              is C_in/groups (this is the layout
        //                              the current mlx-community release
        //                              actually ships — confirmed at
        //                              [2560, 1, 33] for d_model=2560,
        //                              kernel=33).
        // In every layout the singleton dim is trivial (depthwise has
        // C_in_per_group = 1), so the source memory is contiguous
        // `outCh * kW` floats either way. We only need to pick the
        // right `kW` for the destination shape.
        func loadDepthwiseWeight(_ key: String, kernelSize: Int) throws -> Linear {
            let raw = try t(key)
            let outCh = raw.shape[0]
            let kW: Int
            switch raw.shape.count {
            case 2:
                kW = raw.shape[1]
            case 3:
                // Pick the non-singleton dim. If both are non-1 we
                // crash loudly rather than silently mis-shape; depthwise
                // weights must have exactly one trivial channel-mult dim.
                let d1 = raw.shape[1]
                let d2 = raw.shape[2]
                precondition(
                    d1 == 1 || d2 == 1,
                    "FireRedASR2 loadDepthwiseWeight: \(key) has shape "
                        + "\(raw.shape); expected one trivial dim of size 1")
                kW = (d1 == 1) ? d2 : d1
            default:
                preconditionFailure(
                    "FireRedASR2 loadDepthwiseWeight: \(key) has rank "
                        + "\(raw.shape.count); expected 2 or 3")
            }
            precondition(
                kW == kernelSize,
                "FireRedASR2 loadDepthwiseWeight: \(key) kW=\(kW) "
                    + "≠ expected kernelSize=\(kernelSize)")
            let src = raw.toFloatArray()
            let flat = Tensor.empty(shape: [outCh, kW], dtype: dtype, device: device)
            let vals = Array(src.prefix(outCh * kW))
            AudioPreprocessing.copyFloats(vals, into: flat)
            return Linear(weight: flat, bias: nil)
        }

        // ── Conv2d subsampling ─────────────────────────────────────────
        // Key remapping: conv.0 → conv1, conv.2 → conv2 (applied at source).
        let conv1W = try loadConv2dWeight(
            "encoder.input_preprocessor.conv1.weight")
        let conv1B = try t("encoder.input_preprocessor.conv1.bias")
        let conv2W = try loadConv2dWeight(
            "encoder.input_preprocessor.conv2.weight")
        let conv2B = try t("encoder.input_preprocessor.conv2.bias")

        // conv_out (out.weight): [dModel, flatFreq] — standard Linear weight.
        let convOutW = try t("encoder.input_preprocessor.out.weight")

        // ── Relative positional encoding ───────────────────────────────
        // Pre-computed on first use from scratch (not stored in checkpoint).
        let peMaxlen = fc.encoder.peMaxlen
        let relPosLen = 2 * peMaxlen - 1
        var posTable = [Float](repeating: 0, count: relPosLen * dModel)
        // Build sinusoidal table for positive and negative positions.
        let halfDim = dModel / 2
        var divTerm = [Float](repeating: 0, count: max(halfDim, 1))
        for i in 0 ..< halfDim {
            divTerm[i] = exp(Float(2 * i) * (-log(10000.0) / Float(dModel)))
        }
        // Positive positions: rows [peMaxlen-1 .. 0] in forward order.
        for position in 0 ..< peMaxlen {
            let row = (peMaxlen - 1 - position)  // row index in the table
            for i in 0 ..< halfDim {
                let value = Float(position) * divTerm[i]
                let base = row * dModel + 2 * i
                if base + 1 < posTable.count {
                    posTable[base] = sin(value)
                    posTable[base + 1] = cos(value)
                }
            }
        }
        // Negative positions: rows [peMaxlen .. 2*peMaxlen-2].
        for position in 1 ..< peMaxlen {
            let row = peMaxlen - 1 + position  // row index in the table
            for i in 0 ..< halfDim {
                let value = Float(position) * divTerm[i]
                let base = row * dModel + 2 * i
                if base + 1 < posTable.count {
                    posTable[base] = sin(-value)
                    posTable[base + 1] = cos(-value)
                }
            }
        }

        // ── Encoder Conformer blocks ───────────────────────────────────
        var encoderBlocks = [FireRedASR2ConformerBlock]()
        encoderBlocks.reserveCapacity(fc.encoder.nLayers)

        for i in 0 ..< fc.encoder.nLayers {
            let p = "encoder.layer_stack.\(i)"

            let ffn1 = FireRedASR2FFN(
                norm0: try ln("\(p).ffn1.net_0"),
                w1: try linear("\(p).ffn1.net_1", hasBias: false),
                w2: try linear("\(p).ffn1.net_4", hasBias: false))
            let ffn2 = FireRedASR2FFN(
                norm0: try ln("\(p).ffn2.net_0"),
                w1: try linear("\(p).ffn2.net_1", hasBias: false),
                w2: try linear("\(p).ffn2.net_4", hasBias: false))

            // Load pos_bias_u / pos_bias_v tensors.
            let rawBiasU = try t("\(p).mhsa.pos_bias_u").toFloatArray()
            let rawBiasV = try t("\(p).mhsa.pos_bias_v").toFloatArray()

            let mhsa = FireRedASR2RelPosAttn(
                wQs: try linear("\(p).mhsa.w_qs", hasBias: false),
                wKs: try linear("\(p).mhsa.w_ks", hasBias: false),
                wVs: try linear("\(p).mhsa.w_vs", hasBias: false),
                lnQ: try ln("\(p).mhsa.layer_norm_q"),
                lnK: try ln("\(p).mhsa.layer_norm_k"),
                lnV: try ln("\(p).mhsa.layer_norm_v"),
                fc: try linear("\(p).mhsa.fc", hasBias: false),
                linPos: try linear("\(p).mhsa.linear_pos", hasBias: false),
                posBiasU: rawBiasU, posBiasV: rawBiasV,
                nHead: nHead, dK: dK)

            let kSize = fc.encoder.kernelSize
            let convBlock = FireRedASR2ConvBlock(
                preNorm: try ln("\(p).conv.pre_layer_norm"),
                pw1: try linearFromPointwise("\(p).conv.pointwise_conv1"),
                depthwise: try loadDepthwiseWeight(
                    "\(p).conv.depthwise_conv.weight",
                    kernelSize: kSize),
                batchNorm: try ln("\(p).conv.batch_norm"),
                pw2: try linearFromPointwise("\(p).conv.pointwise_conv2"),
                kernelSize: kSize)

            let block = FireRedASR2ConformerBlock(
                ffn1: ffn1, mhsa: mhsa, conv: convBlock, ffn2: ffn2,
                norm: try ln("\(p).layer_norm"))
            encoderBlocks.append(block)
        }

        // ── Decoder ───────────────────────────────────────────────────
        let tgtWordEmb = try t("decoder.tgt_word_emb.weight")

        // tgt_word_prj is tied to tgt_word_emb when absent (standard config).
        let tgtWordPrj =
            bundle.has("decoder.tgt_word_prj.weight")
            ? try t("decoder.tgt_word_prj.weight")
            : tgtWordEmb

        // Sinusoidal positional encoding for the decoder.
        let decMaxlen = fc.decoder.peMaxlen
        let decDModel = fc.decoder.dModel
        var decPETable = [Float](repeating: 0, count: decMaxlen * decDModel)
        let decHalfDim = decDModel / 2
        var decDivTerm = [Float](repeating: 0, count: max(decHalfDim, 1))
        for i in 0 ..< decHalfDim {
            decDivTerm[i] = exp(Float(2 * i) * (-log(10000.0) / Float(decDModel)))
        }
        for p in 0 ..< decMaxlen {
            for i in 0 ..< decHalfDim {
                let value = Float(p) * decDivTerm[i]
                let base = p * decDModel + 2 * i
                if base + 1 < decPETable.count {
                    decPETable[base] = sin(value)
                    decPETable[base + 1] = cos(value)
                }
            }
        }

        let decNHead = fc.decoder.nHead
        let decDK = decDModel / decNHead

        var decoderLayers = [FireRedASR2DecoderLayer]()
        decoderLayers.reserveCapacity(fc.decoder.nLayers)
        for i in 0 ..< fc.decoder.nLayers {
            let p = "decoder.layer_stack.\(i)"
            let selfAttn = FireRedASR2CrossAttn(
                wQs: try linear("\(p).self_attn.w_qs"),
                wKs: try linear("\(p).self_attn.w_ks", hasBias: false),
                wVs: try linear("\(p).self_attn.w_vs"),
                fc: try linear("\(p).self_attn.fc"),
                nHead: decNHead, dK: decDK)
            let crossAttn = FireRedASR2CrossAttn(
                wQs: try linear("\(p).cross_attn.w_qs"),
                wKs: try linear("\(p).cross_attn.w_ks", hasBias: false),
                wVs: try linear("\(p).cross_attn.w_vs"),
                fc: try linear("\(p).cross_attn.fc"),
                nHead: decNHead, dK: decDK)
            decoderLayers.append(
                FireRedASR2DecoderLayer(
                    selfAttnNorm: try ln("\(p).self_attn_norm"),
                    selfAttn: selfAttn,
                    crossAttnNorm: try ln("\(p).cross_attn_norm"),
                    crossAttn: crossAttn,
                    mlpNorm: try ln("\(p).mlp_norm"),
                    w1: try linear("\(p).mlp.w_1"),
                    w2: try linear("\(p).mlp.w_2")))
        }
        let decoderNormOut = try ln("decoder.layer_norm_out")

        return FireRedASR2Model(
            config: fc,
            conv1Weight: conv1W,
            conv1Bias: conv1B,
            conv2Weight: conv2W,
            conv2Bias: conv2B,
            convOutWeight: convOutW,
            relPosTable: posTable,
            relPosLen: relPosLen,
            encoderBlocks: encoderBlocks,
            tgtWordEmb: tgtWordEmb,
            tgtWordPrj: tgtWordPrj,
            decoderLayers: decoderLayers,
            decoderNormOut: decoderNormOut,
            decoderPETable: decPETable,
            cmvnMeans: cmvnMeans,
            cmvnIstd: cmvnIstd,
            tokenizer: tokenizer,
            dtype: dtype)
    }
}
