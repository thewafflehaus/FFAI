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
// Qwen3ASR — Qwen's speech-to-text family (Qwen3-ASR-0.6B, 1.7B).
//
// Architecture: a Conv2d-based audio encoder whose output is merged
// into the Qwen3 text-decoder embedding stream, followed by a causal
// autoregressive decoder that generates the transcript.
//
//   waveform  ──log-Mel──▶ [1, 1, nMels, nFrames]  (NCHW)
//             ──3×Conv2d (stride 2, kernel 3×3, pad 1)──▶ [1, dsCh, H, W]
//             ──reshape + linear (conv_out)──▶ [W, dModel]
//             ──sinusoidal PE──▶ ──N×BiAttn──▶ ──lnPost──▶ ──proj1/proj2──▶
//             audio features [nAudioTokens, outputDim]
//
//   <|im_start|>system<|im_end|><|im_start|>user<|audio_start|>
//   <|audio_pad|>×nAudioTokens <|audio_end|><|im_end|><|im_start|>assistant\n
//   ──embed_tokens, replacing audio_pad slots with audio features──▶
//   ──Qwen3 decoder (GQA + q/k_norm + RoPE)──▶ transcript tokens
//
// Audio encoder Conv2d weights in MLX checkpoints use OHWI layout
// `[out_ch, kH, kW, in_ch]`; transposed to OIHW for `Ops.conv2d` on load.
//
// Detection: `model_type == "qwen3_asr"` or architecture
// `"Qwen3ASRForConditionalGeneration"`.

import Foundation
import Metal
import Tokenizers

// ─── Configuration ───────────────────────────────────────────────────

/// Qwen3ASR audio-encoder hyper-parameters, decoded from the checkpoint's
/// `thinker_config.audio_config` block.
public struct Qwen3ASRAudioEncoderConfig: Sendable {
    /// Number of Mel filterbank bins.
    public let numMelBins: Int
    /// Encoder hidden dim (`d_model`).
    public let dModel: Int
    /// Number of transformer encoder blocks.
    public let encoderLayers: Int
    /// Encoder attention heads.
    public let encoderAttentionHeads: Int
    /// Encoder feed-forward intermediate dim.
    public let encoderFfnDim: Int
    /// Downsampled Conv2d output channels.
    public let downsampleHiddenSize: Int
    /// Maximum source positions (sinusoidal PE rows).
    public let maxSourcePositions: Int
    /// Sliding window size in frames for chunked encoding.
    public let nWindow: Int
    /// Output projection target dim (→ text hidden for merging).
    public let outputDim: Int

    public init(
        numMelBins: Int = 128,
        dModel: Int = 896,
        encoderLayers: Int = 18,
        encoderAttentionHeads: Int = 14,
        encoderFfnDim: Int = 3584,
        downsampleHiddenSize: Int = 480,
        maxSourcePositions: Int = 1500,
        nWindow: Int = 50,
        outputDim: Int = 1024
    ) {
        self.numMelBins = numMelBins
        self.dModel = dModel
        self.encoderLayers = encoderLayers
        self.encoderAttentionHeads = encoderAttentionHeads
        self.encoderFfnDim = encoderFfnDim
        self.downsampleHiddenSize = downsampleHiddenSize
        self.maxSourcePositions = maxSourcePositions
        self.nWindow = nWindow
        self.outputDim = outputDim
    }
}

/// Qwen3ASR top-level configuration, decoded from `config.json`.
/// The checkpoint nests audio + text hyper-parameters under `thinker_config`.
public struct Qwen3ASRConfig: Sendable {
    public let audioConfig: Qwen3ASRAudioEncoderConfig
    /// Qwen3 text-decoder hidden dim.
    public let textHidden: Int
    /// Qwen3 text-decoder layers.
    public let textLayers: Int
    /// Text-decoder attention heads.
    public let textHeads: Int
    /// Text-decoder KV heads (GQA).
    public let textKVHeads: Int
    /// Per-head dimension.
    public let headDim: Int
    /// Text-decoder feed-forward intermediate dim.
    public let textIntermediate: Int
    /// RMS norm epsilon.
    public let rmsNormEps: Float
    /// RoPE base frequency.
    public let ropeTheta: Float
    /// Maximum text-decoder context length.
    public let maxPositionEmbeddings: Int
    /// Vocabulary size.
    public let vocabSize: Int
    /// Whether lm_head shares weights with embed_tokens.
    public let tieWordEmbeddings: Bool
    /// Token id that marks audio padding positions (the slots to replace).
    public let audioTokenId: Int
    /// Token id for `<|audio_start|>`.
    public let audioStartTokenId: Int
    /// Token id for `<|audio_end|>`.
    public let audioEndTokenId: Int
    /// EOS token ids — generation stops when any is produced.
    public let eosTokenIds: [Int]
    /// Pad token id.
    public let padTokenId: Int
    /// Languages the checkpoint was trained on.
    public let supportLanguages: [String]

    public init(
        audioConfig: Qwen3ASRAudioEncoderConfig = Qwen3ASRAudioEncoderConfig(),
        textHidden: Int = 1024,
        textLayers: Int = 28,
        textHeads: Int = 16,
        textKVHeads: Int = 8,
        headDim: Int = 128,
        textIntermediate: Int = 3072,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000,
        maxPositionEmbeddings: Int = 65536,
        vocabSize: Int = 151936,
        tieWordEmbeddings: Bool = true,
        audioTokenId: Int = 151676,
        audioStartTokenId: Int = 151669,
        audioEndTokenId: Int = 151670,
        eosTokenIds: [Int] = [151643, 151645],
        padTokenId: Int = 151643,
        supportLanguages: [String] = []
    ) {
        self.audioConfig = audioConfig
        self.textHidden = textHidden
        self.textLayers = textLayers
        self.textHeads = textHeads
        self.textKVHeads = textKVHeads
        self.headDim = headDim
        self.textIntermediate = textIntermediate
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.vocabSize = vocabSize
        self.tieWordEmbeddings = tieWordEmbeddings
        self.audioTokenId = audioTokenId
        self.audioStartTokenId = audioStartTokenId
        self.audioEndTokenId = audioEndTokenId
        self.eosTokenIds = eosTokenIds
        self.padTokenId = padTokenId
        self.supportLanguages = supportLanguages
    }

    /// Build from a decoded `config.json`. Qwen3ASR nests the audio and
    /// text configs under `thinker_config`; also supports the flat layout
    /// used by some mlx-community conversions.
    public static func from(_ config: ModelConfig) -> Qwen3ASRConfig? {
        // Locate the thinker_config (or fall back to top-level).
        let thinker = config.nested("thinker_config")
        let audioRaw =
            (thinker?["audio_config"] as? [String: Any])
            ?? config.nested("audio_config")
        let textRaw =
            (thinker?["text_config"] as? [String: Any])
            ?? config.nested("text_config")

        guard audioRaw != nil || textRaw != nil else { return nil }

        // Helper accessors for the audio sub-dict.
        func ai(_ k: String) -> Int? {
            if let v = audioRaw?[k] as? Int { return v }
            if let v = audioRaw?[k] as? Double { return Int(v) }
            return nil
        }
        // Helper accessors for the text sub-dict.
        func ti(_ k: String) -> Int? {
            if let v = textRaw?[k] as? Int { return v }
            if let v = textRaw?[k] as? Double { return Int(v) }
            return nil
        }
        func tf(_ k: String) -> Float? {
            if let v = textRaw?[k] as? Double { return Float(v) }
            if let v = textRaw?[k] as? Int { return Float(v) }
            return nil
        }
        func tb(_ k: String) -> Bool? { textRaw?[k] as? Bool }

        let audioConfig = Qwen3ASRAudioEncoderConfig(
            numMelBins: ai("num_mel_bins") ?? 128,
            dModel: ai("d_model") ?? 896,
            encoderLayers: ai("encoder_layers") ?? ai("num_hidden_layers") ?? 18,
            encoderAttentionHeads: ai("encoder_attention_heads") ?? 14,
            encoderFfnDim: ai("encoder_ffn_dim") ?? 3584,
            downsampleHiddenSize: ai("downsample_hidden_size") ?? 480,
            maxSourcePositions: ai("max_source_positions") ?? 1500,
            nWindow: ai("n_window") ?? 50,
            outputDim: ai("output_dim") ?? (ti("hidden_size") ?? 1024)
        )

        // EOS token ids — generation_config ships [151643, 151645].
        let eosRaw = thinker?["eos_token_id"] ?? config.raw["eos_token_id"]
        var eosIds: [Int]
        if let arr = eosRaw as? [Int] {
            eosIds = arr
        } else if let single = eosRaw as? Int {
            eosIds = [single]
        } else {
            eosIds = [151643, 151645]
        }

        // Audio special token ids live in thinker_config.
        let audioTokenId = (thinker?["audio_token_id"] as? Int) ?? 151676
        let audioStartTokenId = (thinker?["audio_start_token_id"] as? Int) ?? 151669
        let audioEndTokenId = (thinker?["audio_end_token_id"] as? Int) ?? 151670

        let supportLanguages = (config.raw["support_languages"] as? [String]) ?? []

        return Qwen3ASRConfig(
            audioConfig: audioConfig,
            textHidden: ti("hidden_size") ?? 1024,
            textLayers: ti("num_hidden_layers") ?? 28,
            textHeads: ti("num_attention_heads") ?? 16,
            textKVHeads: ti("num_key_value_heads") ?? 8,
            headDim: ti("head_dim") ?? 128,
            textIntermediate: ti("intermediate_size") ?? 3072,
            rmsNormEps: tf("rms_norm_eps") ?? 1e-6,
            ropeTheta: tf("rope_theta") ?? 1_000_000,
            maxPositionEmbeddings: ti("max_position_embeddings") ?? 65536,
            vocabSize: ti("vocab_size") ?? 151936,
            tieWordEmbeddings: tb("tie_word_embeddings") ?? true,
            audioTokenId: audioTokenId,
            audioStartTokenId: audioStartTokenId,
            audioEndTokenId: audioEndTokenId,
            eosTokenIds: eosIds,
            padTokenId: (thinker?["pad_token_id"] as? Int)
                ?? (config.raw["pad_token_id"] as? Int)
                    ?? 151643,
            supportLanguages: supportLanguages
        )
    }
}

// ─── Audio encoder helpers ────────────────────────────────────────────

/// One Qwen3ASR audio encoder block.
/// Pre-norm bidirectional self-attention → pre-norm GELU MLP, each with
/// residual addition. All attention is CPU-side SDPA (no causal mask).
public final class Qwen3ASREncoderLayer: Module {
    let selfAttnNorm: LayerNorm
    let finalNorm: LayerNorm
    let qProj, kProj, vProj, outProj: Linear
    let fc1, fc2: Linear
    let hidden, nHeads, headDim: Int
    let scale: Float

    init(
        selfAttnNorm: LayerNorm, finalNorm: LayerNorm,
        qProj: Linear, kProj: Linear, vProj: Linear, outProj: Linear,
        fc1: Linear, fc2: Linear, hidden: Int, nHeads: Int
    ) {
        self.selfAttnNorm = selfAttnNorm
        self.finalNorm = finalNorm
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.outProj = outProj
        self.fc1 = fc1
        self.fc2 = fc2
        self.hidden = hidden
        self.nHeads = nHeads
        self.headDim = hidden / nHeads
        self.scale = 1.0 / Float(Double(hidden / nHeads).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in selfAttnNorm.parameters() {
            out.append(("self_attn_layer_norm.\(k)", v))
        }
        for (k, v) in finalNorm.parameters() {
            out.append(("final_layer_norm.\(k)", v))
        }
        for (k, v) in qProj.parameters() {
            out.append(("self_attn.q_proj.\(k)", v))
        }
        for (k, v) in kProj.parameters() {
            out.append(("self_attn.k_proj.\(k)", v))
        }
        for (k, v) in vProj.parameters() {
            out.append(("self_attn.v_proj.\(k)", v))
        }
        for (k, v) in outProj.parameters() {
            out.append(("self_attn.out_proj.\(k)", v))
        }
        for (k, v) in fc1.parameters() { out.append(("fc1.\(k)", v)) }
        for (k, v) in fc2.parameters() { out.append(("fc2.\(k)", v)) }
        return out
    }
}

// ─── Text decoder layer ───────────────────────────────────────────────

/// One Qwen3ASR text decoder layer (Qwen3 architecture with GQA and
/// per-head q/k RMSNorm).
public final class Qwen3ASRTextLayer: Module {
    let inputNorm: RMSNorm
    let postAttnNorm: RMSNorm
    let qProj, kProj, vProj, oProj: AnyLinear
    let qNorm, kNorm: RMSNorm
    let gateProj, upProj, downProj: AnyLinear

    public init(
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        qNorm: RMSNorm, kNorm: RMSNorm,
        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear
    ) {
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
    }

    public func parameters() -> [(String, Tensor)] { [] }
}

// ─── Qwen3ASR model ───────────────────────────────────────────────────

/// A loaded Qwen3ASR speech-to-text model.
///
/// The model is built from:
///   * A Conv2d audio encoder (3×stride-2 Conv2d → linear → transformer)
///   * A Qwen3 text decoder (GQA, q/k_norm, RoPE, tied embedding)
///
/// `transcribe(waveform:tokenizer:maxTokens:device:)` is the main entry
/// point. It runs the full pipeline end-to-end and returns the transcript.
public final class Qwen3ASRModel: @unchecked Sendable {
    public let config: Qwen3ASRConfig

    // Audio encoder weights — Conv2d frontend (OIHW for Ops.conv2d).
    let conv2d1Weight: Tensor  // [downsampleHidden, 1, 3, 3]
    let conv2d1Bias: Tensor
    let conv2d2Weight: Tensor  // [downsampleHidden, downsampleHidden, 3, 3]
    let conv2d2Bias: Tensor
    let conv2d3Weight: Tensor
    let conv2d3Bias: Tensor
    /// Linear projection [dModel, downsampleHidden × freqAfterConv], no bias.
    let convOutWeight: Tensor
    /// Sinusoidal positional embedding flattened [maxSourcePositions × dModel].
    let audioPositionalEmbedding: [Float]
    let audioEncoderLayers: [Qwen3ASREncoderLayer]
    let lnPost: LayerNorm
    /// proj1: [dModel, dModel] — GELU activation.
    let proj1Weight: Tensor
    let proj1Bias: Tensor
    /// proj2: [outputDim, dModel].
    let proj2Weight: Tensor
    let proj2Bias: Tensor

    // Text decoder weights (Qwen3 architecture).
    let embedTokens: AnyEmbedding
    let textLayers: [Qwen3ASRTextLayer]
    let textNorm: RMSNorm
    /// lm_head — may be tied to embed_tokens (including quantized variant).
    /// Stored as AnyLinear so dequantGemv fires correctly for quantized checkpoints.
    let lmHead: AnyLinear

    let dtype: DType

    public init(
        config: Qwen3ASRConfig,
        conv2d1Weight: Tensor, conv2d1Bias: Tensor,
        conv2d2Weight: Tensor, conv2d2Bias: Tensor,
        conv2d3Weight: Tensor, conv2d3Bias: Tensor,
        convOutWeight: Tensor,
        audioPositionalEmbedding: [Float],
        audioEncoderLayers: [Qwen3ASREncoderLayer],
        lnPost: LayerNorm,
        proj1Weight: Tensor, proj1Bias: Tensor,
        proj2Weight: Tensor, proj2Bias: Tensor,
        embedTokens: AnyEmbedding,
        textLayers: [Qwen3ASRTextLayer],
        textNorm: RMSNorm,
        lmHead: AnyLinear,
        dtype: DType
    ) {
        self.config = config
        self.conv2d1Weight = conv2d1Weight
        self.conv2d1Bias = conv2d1Bias
        self.conv2d2Weight = conv2d2Weight
        self.conv2d2Bias = conv2d2Bias
        self.conv2d3Weight = conv2d3Weight
        self.conv2d3Bias = conv2d3Bias
        self.convOutWeight = convOutWeight
        self.audioPositionalEmbedding = audioPositionalEmbedding
        self.audioEncoderLayers = audioEncoderLayers
        self.lnPost = lnPost
        self.proj1Weight = proj1Weight
        self.proj1Bias = proj1Bias
        self.proj2Weight = proj2Weight
        self.proj2Bias = proj2Bias
        self.embedTokens = embedTokens
        self.textLayers = textLayers
        self.textNorm = textNorm
        self.lmHead = lmHead
        self.dtype = dtype
    }

    // ─── Audio encoding ───────────────────────────────────────────────

    /// Encode a waveform into audio feature tokens `[nAudioTokens, outputDim]`.
    /// The result is ready to splice into the text embedding stream in place
    /// of `<|audio_pad|>` positions.
    public func encodeAudio(waveform: [Float], device: Device = .shared) -> Tensor {
        let ac = config.audioConfig
        let frontEnd = AudioFrontEndConfig(
            sampleRate: 16_000, nFFT: 400, hopLength: 160, nMels: ac.numMelBins)

        // ── Log-Mel spectrogram (Whisper-normalised) ──
        // `logMelSpectrogram` with `whisperNormalize: true` internally commits
        // and waits on `cmdMel` before returning (see `applyWhisperLogMelNorm`).
        // Do NOT commit again here; `castTensor` reads the already-completed result.
        let melDtype: DType = dtype == .f16 ? .f16 : .f32
        let cmdMel = device.makeCommandBuffer()
        let melRaw = AudioPreprocessing.logMelSpectrogram(
            waveform: waveform, cfg: frontEnd, dtype: melDtype,
            whisperNormalize: true, device: device, on: cmdMel)
        let melF = AudioPreprocessing.castTensor(melRaw, to: dtype, device: device)

        // melF shape: [nFrames, nMels]. Treat as a 2D image with a single
        // channel — reshape to [1, 1, nMels, nFrames] (NCHW).
        let nFrames = melF.shape[0]
        let nMels = melF.shape[1]
        let melNCHW = melF.reshaped(to: [1, 1, nMels, nFrames])

        // ── Conv2d frontend (3 × stride-2, kernel 3×3, padding 1) ──
        let cmd1 = device.makeCommandBuffer()
        var x = Ops.conv2d(
            input: melNCHW, weight: conv2d1Weight,
            bias: conv2d1Bias, strideH: 2, strideW: 2,
            padH: 1, padW: 1, on: cmd1)
        x = Ops.gelu(x, on: cmd1)
        cmd1.commit()
        cmd1.waitUntilCompleted()

        let cmd2 = device.makeCommandBuffer()
        x = Ops.conv2d(
            input: x, weight: conv2d2Weight, bias: conv2d2Bias,
            strideH: 2, strideW: 2, padH: 1, padW: 1, on: cmd2)
        x = Ops.gelu(x, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        let cmd3 = device.makeCommandBuffer()
        x = Ops.conv2d(
            input: x, weight: conv2d3Weight, bias: conv2d3Bias,
            strideH: 2, strideW: 2, padH: 1, padW: 1, on: cmd3)
        x = Ops.gelu(x, on: cmd3)
        cmd3.commit()
        cmd3.waitUntilCompleted()

        // x shape: [1, downsampleHidden, freqOut, timeOut]
        // Transpose NCHW → [timeOut, freqOut × channels] by iterating over
        // the GPU readback and reordering: x[n,c,h,w] → out[w, h*C + c].
        let C = x.shape[1]
        let H = x.shape[2]
        let W = x.shape[3]  // timeOut
        let flatDim = H * C  // freqOut * downsampleHidden

        let xVals = x.toFloatArray()
        // NCHW: x[c, h, w] = xVals[c*H*W + h*W + w] (batch dim 0 = singleton)
        var reordered = [Float](repeating: 0, count: W * flatDim)
        for w in 0 ..< W {
            for h in 0 ..< H {
                for c in 0 ..< C {
                    reordered[w * flatDim + h * C + c] = xVals[c * H * W + h * W + w]
                }
            }
        }
        let xFlat = Tensor.empty(shape: [W, flatDim], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(reordered, into: xFlat)

        // ── conv_out linear: [W, flatDim] → [W, dModel] ──
        let dModel = ac.dModel
        let cmd4 = device.makeCommandBuffer()
        var h = Ops.gemm(weight: convOutWeight, input: xFlat, nRows: W, on: cmd4)
        cmd4.commit()
        cmd4.waitUntilCompleted()

        // ── Add sinusoidal positional embedding for the W time steps ──
        let peCount = min(W, ac.maxSourcePositions) * dModel
        let peVals = Array(audioPositionalEmbedding.prefix(peCount))
        let peT = Tensor.empty(shape: [W, dModel], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(peVals, into: peT)
        let cmd5 = device.makeCommandBuffer()
        h = Ops.add(h, peT, on: cmd5)
        cmd5.commit()
        cmd5.waitUntilCompleted()

        // ── Transformer encoder blocks (bidirectional self-attention) ──
        // Executed CPU-side over [W * dModel] floats (same pattern as Whisper).
        var seqVals = h.toFloatArray()

        for layer in audioEncoderLayers {
            seqVals = runAudioEncoderLayer(
                layer, seq: seqVals, seqLen: W,
                hidden: dModel, device: device)
        }

        let seqT = Tensor.empty(shape: [W, dModel], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(seqVals, into: seqT)

        // ── Post layer-norm → proj1 (GELU) → proj2 ──
        let cmd6 = device.makeCommandBuffer()
        let normed = Ops.layerNorm(
            seqT, weight: lnPost.weight,
            bias: lnPost.bias, eps: lnPost.eps,
            nRows: W, rowSize: dModel, on: cmd6)
        let proj1Out = Ops.gemm(weight: proj1Weight, input: normed, nRows: W, on: cmd6)
        cmd6.commit()
        cmd6.waitUntilCompleted()

        // Add proj1 bias and apply GELU (CPU-side; W is small).
        let p1Vals = proj1Out.toFloatArray()
        let b1Vals = proj1Bias.toFloatArray()
        let p1Dim = proj1Weight.shape[0]  // output dim of proj1
        let gk: Float = 0.7978845608  // √(2/π)
        let gc: Float = 0.044715
        var p1Act = [Float](repeating: 0, count: W * p1Dim)
        for i in 0 ..< p1Act.count {
            let v = p1Vals[i] + b1Vals[i % p1Dim]
            // GELU approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
            let inner = gk * (v + gc * v * v * v)
            p1Act[i] = 0.5 * v * (1 + tanh(inner))
        }
        let p1T = Tensor.empty(shape: [W, p1Dim], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(p1Act, into: p1T)

        // proj2: [W, p1Dim] → [W, outputDim]
        let outputDim = ac.outputDim
        let cmd7 = device.makeCommandBuffer()
        let out2 = Ops.gemm(weight: proj2Weight, input: p1T, nRows: W, on: cmd7)
        cmd7.commit()
        cmd7.waitUntilCompleted()

        // Add proj2 bias.
        let o2Vals = out2.toFloatArray()
        let b2Vals = proj2Bias.toFloatArray()
        var finalVals = [Float](repeating: 0, count: W * outputDim)
        for i in 0 ..< finalVals.count {
            finalVals[i] = o2Vals[i] + b2Vals[i % outputDim]
        }
        let audioFeatures = Tensor.empty(
            shape: [W, outputDim], dtype: dtype,
            device: device)
        AudioPreprocessing.copyFloats(finalVals, into: audioFeatures)
        return audioFeatures  // [nAudioTokens, outputDim]
    }

    // ─── Encoder layer (CPU bidirectional attention) ──────────────────

    /// Run one audio encoder transformer layer over a full `[seqLen, hidden]`
    /// sequence stored flat as `[Float]`. Returns the updated sequence values.
    private func runAudioEncoderLayer(
        _ layer: Qwen3ASREncoderLayer,
        seq seqVals: [Float],
        seqLen: Int, hidden: Int,
        device: Device
    ) -> [Float] {
        // Upload, apply layer-norm, compute Q/K/V projections.
        let seqT = Tensor.empty(shape: [seqLen, hidden], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(seqVals, into: seqT)

        let cmd = device.makeCommandBuffer()
        let normed = Ops.layerNorm(
            seqT, weight: layer.selfAttnNorm.weight,
            bias: layer.selfAttnNorm.bias,
            eps: layer.selfAttnNorm.eps,
            nRows: seqLen, rowSize: hidden, on: cmd)
        let q = Ops.gemm(
            weight: layer.qProj.weight, input: normed,
            nRows: seqLen, on: cmd)
        let k = Ops.gemm(
            weight: layer.kProj.weight, input: normed,
            nRows: seqLen, on: cmd)
        let v = Ops.gemm(
            weight: layer.vProj.weight, input: normed,
            nRows: seqLen, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // Add Q/K/V biases if present (audio encoder uses biased Q/K/V).
        let qa = addRowBiasIfPresent(
            q, bias: layer.qProj.bias,
            nRows: seqLen, rowSize: hidden,
            device: device
        ).toFloatArray()
        let ka = addRowBiasIfPresent(
            k, bias: layer.kProj.bias,
            nRows: seqLen, rowSize: hidden,
            device: device
        ).toFloatArray()
        let va = addRowBiasIfPresent(
            v, bias: layer.vProj.bias,
            nRows: seqLen, rowSize: hidden,
            device: device
        ).toFloatArray()

        // CPU bidirectional multi-head attention (no causal mask).
        let attnCtx = cpuBidirectionalAttention(
            qa: qa, ka: ka, va: va,
            seqLen: seqLen, nHeads: layer.nHeads,
            headDim: layer.headDim, scale: layer.scale,
            device: device)

        // Output projection + residual.
        let cmd2 = device.makeCommandBuffer()
        let outProj = Ops.gemm(
            weight: layer.outProj.weight, input: attnCtx,
            nRows: seqLen, on: cmd2)
        let selfOut = addRowBiasIfPresent(
            outProj, bias: layer.outProj.bias,
            nRows: seqLen, rowSize: hidden,
            device: device)
        let h = Ops.add(seqT, selfOut, on: cmd2)

        // Pre-norm FFN (GELU MLP).
        let ffIntermediate = layer.fc1.weight.shape[0]
        let normed2 = Ops.layerNorm(
            h, weight: layer.finalNorm.weight,
            bias: layer.finalNorm.bias,
            eps: layer.finalNorm.eps,
            nRows: seqLen, rowSize: hidden, on: cmd2)
        let ff1 = Ops.gemm(
            weight: layer.fc1.weight, input: normed2,
            nRows: seqLen, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        // GELU on FFN output + bias (CPU-side for small intermediates).
        let ff1Vals = addRowBiasIfPresent(
            ff1, bias: layer.fc1.bias,
            nRows: seqLen, rowSize: ffIntermediate,
            device: device
        ).toFloatArray()
        let gk: Float = 0.7978845608
        let gc: Float = 0.044715
        var geluVals = [Float](repeating: 0, count: ff1Vals.count)
        for i in 0 ..< ff1Vals.count {
            let xv = ff1Vals[i]
            let inner = gk * (xv + gc * xv * xv * xv)
            geluVals[i] = 0.5 * xv * (1 + tanh(inner))
        }
        let geluT = Tensor.empty(
            shape: [seqLen, ffIntermediate], dtype: dtype,
            device: device)
        AudioPreprocessing.copyFloats(geluVals, into: geluT)

        let cmd3 = device.makeCommandBuffer()
        let ff2 = Ops.gemm(
            weight: layer.fc2.weight, input: geluT,
            nRows: seqLen, on: cmd3)
        let ff2Out = addRowBiasIfPresent(
            ff2, bias: layer.fc2.bias,
            nRows: seqLen, rowSize: hidden,
            device: device)
        let out = Ops.add(h, ff2Out, on: cmd3)
        cmd3.commit()
        cmd3.waitUntilCompleted()
        return out.toFloatArray()
    }

    /// CPU multi-head self-attention (bidirectional — no causal mask).
    /// Returns a `[seqLen, hidden]` Tensor.
    ///
    /// Fans the `(head, query-row)` index space across CPU cores with
    /// `DispatchQueue.concurrentPerform`. Each iteration writes to a
    /// disjoint `[oBase, oBase + headDim)` output slice — race-free by
    /// construction. Mirrors the parallelization of
    /// `AudioEncoderLayer.cpuAttention`.
    private func cpuBidirectionalAttention(
        qa: [Float], ka: [Float], va: [Float],
        seqLen: Int, nHeads: Int, headDim: Int, scale: Float,
        device: Device
    ) -> Tensor {
        let H = nHeads * headDim
        var out = [Float](repeating: 0, count: seqLen * H)

        out.withUnsafeMutableBufferPointer { outBuf in
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            qa.withUnsafeBufferPointer { qPtr in
                ka.withUnsafeBufferPointer { kPtr in
                    va.withUnsafeBufferPointer { vPtr in
                        nonisolated(unsafe) let qb = qPtr.baseAddress!
                        nonisolated(unsafe) let kb = kPtr.baseAddress!
                        nonisolated(unsafe) let vb = vPtr.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: nHeads * seqLen) { work in
                            let head = work / seqLen
                            let i = work % seqLen
                            let hOff = head * headDim
                            // Attention scores for query position i.
                            var scores = [Float](repeating: 0, count: seqLen)
                            var maxScore = -Float.greatestFiniteMagnitude
                            let qBase = i * H + hOff
                            for j in 0 ..< seqLen {
                                var dot: Float = 0
                                let kBase = j * H + hOff
                                for d in 0 ..< headDim { dot += qb[qBase + d] * kb[kBase + d] }
                                let s = dot * scale
                                scores[j] = s
                                if s > maxScore { maxScore = s }
                            }
                            var sumExp: Float = 0
                            for j in 0 ..< seqLen {
                                let e = exp(scores[j] - maxScore)
                                scores[j] = e
                                sumExp += e
                            }
                            let inv = sumExp > 0 ? 1 / sumExp : 0
                            let oBase = i * H + hOff
                            for j in 0 ..< seqLen {
                                let w = scores[j] * inv
                                let vBase = j * H + hOff
                                for d in 0 ..< headDim { outPtr[oBase + d] += w * vb[vBase + d] }
                            }
                        }
                    }
                }
            }
        }
        let result = Tensor.empty(shape: [seqLen, H], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(out, into: result)
        return result
    }

    /// Add a row-broadcast bias to a `[nRows, rowSize]` tensor if bias is
    /// non-nil. Returns the input unchanged when bias is nil.
    private func addRowBiasIfPresent(
        _ t: Tensor, bias: Tensor?, nRows: Int, rowSize: Int,
        device: Device
    ) -> Tensor {
        guard let bias = bias else { return t }
        let tVals = t.toFloatArray()
        let bVals = bias.toFloatArray()
        var out = [Float](repeating: 0, count: nRows * rowSize)
        for r in 0 ..< nRows {
            for c in 0 ..< rowSize {
                out[r * rowSize + c] = tVals[r * rowSize + c] + bVals[c]
            }
        }
        let result = Tensor.empty(shape: [nRows, rowSize], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(out, into: result)
        return result
    }

    // ─── Prompt construction ──────────────────────────────────────────

    /// Build the tokenized prompt that wraps audio tokens in the
    /// Qwen3ASR chat template. Returns the token id sequence.
    public func buildPrompt(
        numAudioTokens: Int,
        tokenizer: Tokenizer,
        language: String? = nil
    ) -> [Int] {
        let assistantPrefix: String
        if let lang = language {
            assistantPrefix = "language \(lang)<asr_text>"
        } else {
            assistantPrefix = ""
        }

        // The audio pad tokens will be replaced with audio features after
        // embedding. Their count must match the encoder output length.
        let audioPad = String(repeating: "<|audio_pad|>", count: numAudioTokens)
        let prompt =
            "<|im_start|>system\n<|im_end|>\n"
            + "<|im_start|>user\n<|audio_start|>"
            + audioPad
            + "<|audio_end|><|im_end|>\n"
            + "<|im_start|>assistant\n"
            + assistantPrefix

        return tokenizer.encode(text: prompt)
    }

    // ─── Autoregressive transcription ─────────────────────────────────

    /// Transcribe a 16 kHz mono waveform into a text string.
    ///
    /// - Parameters:
    ///   - waveform: 16 kHz mono PCM samples.
    ///   - tokenizer: Tokenizer loaded from the checkpoint directory.
    ///   - maxTokens: Maximum number of tokens to generate.
    ///   - language: Optional language hint (e.g. "English"). `nil` lets
    ///               the model detect the language automatically.
    ///   - device: Metal device for GPU dispatch.
    /// - Returns: The decoded transcript string.
    public func transcribe(
        waveform: [Float],
        tokenizer: Tokenizer,
        maxTokens: Int = 4096,
        language: String? = nil,
        device: Device = .shared
    ) -> String {
        // ── 1. Audio encoding ──
        let audioFeatures = encodeAudio(waveform: waveform, device: device)
        let nAudioTokens = audioFeatures.shape[0]

        // ── 2. Prompt tokenization ──
        let inputIds = buildPrompt(
            numAudioTokens: nAudioTokens,
            tokenizer: tokenizer,
            language: language)
        let promptLen = inputIds.count

        // ── 3. Embed tokens ──
        let idsTensor = Tensor.empty(shape: [promptLen], dtype: .u32, device: device)
        idsTensor.copyIn(from: inputIds.map { UInt32($0) })
        let cmd = device.makeCommandBuffer()
        let fullEmbeds = embedTokens(idsTensor, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── 4. Splice audio features into audio_pad positions ──
        let audioTokenId = Int32(config.audioTokenId)
        let embeds = spliceAudioFeatures(
            embeds: fullEmbeds, audioFeatures: audioFeatures,
            inputIds: inputIds.map { Int32($0) },
            audioTokenId: audioTokenId, device: device)

        // ── 5. Prefill KV cache token-by-token ──
        // We feed the prompt one token at a time through the seqLen=1 decode
        // path. This avoids batched GEMM over quantized weights — the single-
        // token path uses GPU dequantGemv which handles any weight dtype correctly.
        let nLayers = config.textLayers
        let nKVHeads = config.textKVHeads
        let hd = config.headDim
        let maxSeq = promptLen + maxTokens + 16

        let caches = (0 ..< nLayers).map { _ in
            KVCache(
                nKVHeads: nKVHeads, headDim: hd, contextLength: maxSeq,
                dtype: dtype, device: device)
        }

        // Feed every prompt position one at a time; retain only the last logits.
        var lastLogits: Tensor? = nil
        for pos in 0 ..< promptLen {
            // Slice the embedding row for this position → [1, H].
            let rowEmbed = embeds.slicedRows(start: pos, count: 1)
            lastLogits = forwardOneToken(embed: rowEmbed, caches: caches, device: device)
        }
        // promptLen is always ≥ 1 (the chat template has several fixed tokens).
        var logits: Tensor = lastLogits!

        // ── 6. Greedy autoregressive decode ──
        var generated: [Int] = []
        let eosIds = Set(config.eosTokenIds)

        for _ in 0 ..< maxTokens {
            // Pick the next token greedily.
            let logitVals = logits.toFloatArray()
            var best = 0
            var bestVal = -Float.greatestFiniteMagnitude
            for (i, v) in logitVals.enumerated() where v > bestVal {
                bestVal = v
                best = i
            }
            if eosIds.contains(best) { break }
            generated.append(best)

            // Backstop: if the last 24 tokens have ≤3 unique ids, stop to
            // prevent multi-GB KV cache growth on stuck greedy decoders.
            if generated.count >= 24, Set(generated.suffix(24)).count <= 3 {
                break
            }

            // Embed the new token and forward it.
            let nextIdT = Tensor.empty(shape: [1], dtype: .u32, device: device)
            nextIdT.copyIn(from: [UInt32(best)])
            let cmd2 = device.makeCommandBuffer()
            let nextEmbed = embedTokens(nextIdT, on: cmd2)
            cmd2.commit()
            cmd2.waitUntilCompleted()

            logits = forwardOneToken(embed: nextEmbed, caches: caches, device: device)
        }

        return tokenizer.decode(tokens: generated)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // ─── Text decoder forward pass ────────────────────────────────────

    /// Forward a single token embedding `[1, textHidden]` through all Qwen3
    /// text decoder layers with KV caching. Returns `[vocabSize]` logits.
    ///
    /// Always operates in seqLen=1 decode mode — this sidesteps the need for
    /// batched GEMM over quantized weights (which requires CPU dequantization
    /// and dtype alignment). The GPU `dequantGemv` path in `AnyLinear` handles
    /// any weight dtype correctly at seqLen=1.
    private func forwardOneToken(
        embed: Tensor,
        caches: [KVCache],
        device: Device
    ) -> Tensor {
        let H = config.textHidden
        // Ensure the input is [1, H] — callers may pass [H] for convenience.
        var h = embed.shape.count == 1 ? embed.reshaped(to: [1, H]) : embed

        let offset = caches[0].length

        for (i, layer) in textLayers.enumerated() {
            h = runTextLayer(
                layer, h: h,
                offset: offset, hidden: H,
                cache: caches[i], device: device)
        }

        // Post-decoder RMSNorm (1D) → lm_head → [vocabSize] logits.
        // lmHead is an AnyLinear so it dispatches gemv for dense weights or
        // dequantGemv for quantized tied embeddings — no raw Tensor gemv needed.
        let cmd = device.makeCommandBuffer()
        let hFlat = h.reshaped(to: [H])
        let normed = Ops.rmsNorm(
            hFlat, weight: textNorm.weight,
            eps: textNorm.eps, on: cmd)
        let logits = lmHead(normed, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return logits
    }

    /// One Qwen3 decoder layer with GQA, per-head q/k RMSNorm, and RoPE.
    /// Always seqLen=1 — uses `sdpaDecode` + `AnyLinear.callAsFunction`
    /// (GPU gemv / dequantGemv). No CPU weight dequantization needed.
    private func runTextLayer(
        _ layer: Qwen3ASRTextLayer,
        h hIn: Tensor,
        offset: Int, hidden: Int,
        cache: KVCache, device: Device
    ) -> Tensor {
        let nHeads = config.textHeads
        let nKVHeads = config.textKVHeads
        let hd = config.headDim
        let theta = config.ropeTheta
        let eps = config.rmsNormEps
        let scale = 1.0 / Float(Double(hd).squareRoot())

        let cmd1 = device.makeCommandBuffer()

        // ── Pre-norm (1D) + Q/K/V projections via GPU gemv/dequantGemv ──
        let hFlat = hIn.reshaped(to: [hidden])
        let normed = Ops.rmsNorm(
            hFlat, weight: layer.inputNorm.weight,
            eps: eps, on: cmd1)
        let q = layer.qProj(normed, on: cmd1)  // [nHeads * hd]
        let k = layer.kProj(normed, on: cmd1)  // [nKVHeads * hd]
        let v = layer.vProj(normed, on: cmd1)  // [nKVHeads * hd]
        cmd1.commit()
        cmd1.waitUntilCompleted()

        // ── Per-head Q/K RMSNorm (Qwen3 structural delta) ──
        // Treat each head's [hd] slice as a separate row.
        let cmd2 = device.makeCommandBuffer()
        let qNormed = Ops.rmsNormRows(
            q.reshaped(to: [nHeads, hd]),
            weight: layer.qNorm.weight,
            eps: eps, nRows: nHeads, rowSize: hd, on: cmd2)
        let kNormed = Ops.rmsNormRows(
            k.reshaped(to: [nKVHeads, hd]),
            weight: layer.kNorm.weight,
            eps: eps, nRows: nKVHeads, rowSize: hd, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        // ── RoPE (single position) ──
        let qFlat = qNormed.reshaped(to: [nHeads * hd])
        let kFlat = kNormed.reshaped(to: [nKVHeads * hd])
        let cmd3 = device.makeCommandBuffer()
        let qRot = Ops.rope(
            qFlat, position: offset, headDim: hd,
            thetaBase: theta, on: cmd3)
        let kRot = Ops.rope(
            kFlat, position: offset, headDim: hd,
            thetaBase: theta, on: cmd3)
        cmd3.commit()
        cmd3.waitUntilCompleted()

        // ── KV cache append + sdpaDecode ──
        let vShaped = v.reshaped(to: [nKVHeads, hd])
        let kShaped = kRot.reshaped(to: [nKVHeads, hd])
        let cmd4 = device.makeCommandBuffer()
        cache.appendOnGPU(kFlat: kShaped, vFlat: vShaped, on: cmd4)
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd4)
        let attnOut = Ops.sdpaDecode(
            q: qRot.reshaped(to: [nHeads, hd]),
            k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: hd,
            nKV: cache.length, kvStride: cache.capacity,
            scale: scale, on: cmd4)
        // attnOut: [nHeads, hd] → flatten to [nHeads * hd] for o_proj input.
        // Note: nHeads * hd may differ from hidden (e.g. 16*128=2048 vs 1024).
        let attnFlat = attnOut.reshaped(to: [nHeads * hd])

        // ── Output projection + residual ──
        // o_proj maps [nHeads * hd] → [hidden]; add residual on the [hidden] side.
        let oOut = layer.oProj(attnFlat, on: cmd4)
        let postAttn = Ops.add(hFlat, oOut, on: cmd4)

        // ── FFN (SiLU gated MLP) ──
        let normed2 = Ops.rmsNorm(
            postAttn, weight: layer.postAttnNorm.weight,
            eps: eps, on: cmd4)
        let gate = layer.gateProj(normed2, on: cmd4)
        let up = layer.upProj(normed2, on: cmd4)
        let gated = Ops.mul(Ops.silu(gate, on: cmd4), up, on: cmd4)
        let down = layer.downProj(gated, on: cmd4)
        let result = Ops.add(postAttn, down, on: cmd4)
        cmd4.commit()
        cmd4.waitUntilCompleted()

        // Return as [1, hidden] to match the caller's expected shape.
        return result.reshaped(to: [1, hidden])
    }

    // ─── Audio feature splice ─────────────────────────────────────────

    /// Replace `<|audio_pad|>` token embeddings in `embeds` with the
    /// audio feature rows from `audioFeatures`. Audio tokens are expected
    /// to be contiguous in the prompt (the standard Qwen3ASR template).
    ///
    /// The embedding dim is allowed to differ from outputDim only if a
    /// final linear projection maps them — in the standard checkpoint
    /// outputDim == textHidden so a direct copy works.
    private func spliceAudioFeatures(
        embeds: Tensor, audioFeatures: Tensor,
        inputIds: [Int32], audioTokenId: Int32,
        device: Device
    ) -> Tensor {
        let seqLen = embeds.shape[0]
        let hidden = embeds.shape[1]
        let nAudio = audioFeatures.shape[0]

        guard let firstAudio = inputIds.firstIndex(of: audioTokenId) else {
            return embeds
        }

        let embedVals = embeds.toFloatArray()
        let audioVals = audioFeatures.toFloatArray()
        var out = embedVals

        let replaceCount = min(nAudio, seqLen - firstAudio)
        for i in 0 ..< replaceCount {
            let dstRow = firstAudio + i
            for c in 0 ..< hidden {
                out[dstRow * hidden + c] = audioVals[i * hidden + c]
            }
        }

        let result = Tensor.empty(shape: [seqLen, hidden], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(out, into: result)
        return result
    }
}

// ─── Registry detection + loader ─────────────────────────────────────

extension Qwen3ASRModel {
    public static let modelTypes: Set<String> = ["qwen3_asr"]
    public static let architectures: Set<String> = [
        "Qwen3ASRForConditionalGeneration"
    ]

    /// Whether a decoded `config.json` describes a Qwen3ASR checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        return false
    }

    /// Load a Qwen3ASR checkpoint from a resolved snapshot directory.
    public static func load(directory: URL, device: Device = .shared)
        throws -> Qwen3ASRModel
    {
        let config = try ModelConfig.load(from: directory)
        guard let qc = Qwen3ASRConfig.from(config) else {
            throw ModelError.unsupportedModelType(
                "config.json is not a Qwen3ASR config")
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return try build(config: qc, bundle: bundle, rootConfig: config)
    }

    /// Assemble a `Qwen3ASRModel` from a decoded config + weight bundle.
    /// `rootConfig` is the top-level `ModelConfig` (used for quantization
    /// detection); pass `nil` to skip quantization (unquantized checkpoint).
    public static func build(
        config qc: Qwen3ASRConfig,
        bundle: SafeTensorsBundle,
        rootConfig: ModelConfig? = nil
    ) throws -> Qwen3ASRModel {
        let ac = qc.audioConfig

        // Detect the activation dtype from a probe tensor.
        let probeKey =
            bundle.has("audio_tower.conv2d1.weight")
            ? "audio_tower.conv2d1.weight"
            : "model.embed_tokens.weight"
        let dtype = try bundle.tensor(named: probeKey).dtype

        // ── Audio encoder ──
        // Conv2d weights in the checkpoint are [out_ch, kH, kW, in_ch] (MLX OHWI).
        // FFAI's Ops.conv2d expects [out_ch, in_ch, kH, kW] (OIHW).
        // Transpose on load: raw[o, h, w, i] → out[o, i, h, w].
        func loadConv2dWeight(_ key: String) throws -> Tensor {
            let raw = try bundle.tensor(named: key)
            let outCh = raw.shape[0]
            let kH = raw.shape[1]
            let kW = raw.shape[2]
            let inCh = raw.shape[3]
            let rawVals = raw.toFloatArray()
            var transposed = [Float](repeating: 0, count: outCh * inCh * kH * kW)
            for o in 0 ..< outCh {
                for h in 0 ..< kH {
                    for w in 0 ..< kW {
                        for i in 0 ..< inCh {
                            let src = o * kH * kW * inCh + h * kW * inCh + w * inCh + i
                            let dst = o * inCh * kH * kW + i * kH * kW + h * kW + w
                            transposed[dst] = rawVals[src]
                        }
                    }
                }
            }
            let out = Tensor.empty(
                shape: [outCh, inCh, kH, kW], dtype: dtype,
                device: .shared)
            AudioPreprocessing.copyFloats(transposed, into: out)
            return out
        }

        let conv2d1Weight = try loadConv2dWeight("audio_tower.conv2d1.weight")
        let conv2d1Bias = try bundle.tensor(named: "audio_tower.conv2d1.bias")
        let conv2d2Weight = try loadConv2dWeight("audio_tower.conv2d2.weight")
        let conv2d2Bias = try bundle.tensor(named: "audio_tower.conv2d2.bias")
        let conv2d3Weight = try loadConv2dWeight("audio_tower.conv2d3.weight")
        let conv2d3Bias = try bundle.tensor(named: "audio_tower.conv2d3.bias")

        // conv_out is a plain linear [dModel, flatFreq] — no transposing needed.
        let convOutWeight = try bundle.tensor(named: "audio_tower.conv_out.weight")

        // Sinusoidal positional embedding (computed, not stored in checkpoint).
        let audioPositionalEmbedding = AudioPreprocessing.sinusoidalPositions(
            length: ac.maxSourcePositions, dim: ac.dModel)

        // ── Encoder transformer layers ──
        func layerNorm(_ base: String) throws -> LayerNorm {
            LayerNorm(
                weight: try bundle.tensor(named: "\(base).weight"),
                bias: try bundle.tensor(named: "\(base).bias"),
                eps: 1e-5)
        }
        func linear(_ base: String, hasBias: Bool = true) throws -> Linear {
            let w = try bundle.tensor(named: "\(base).weight")
            let b =
                hasBias && bundle.has("\(base).bias")
                ? try bundle.tensor(named: "\(base).bias") : nil
            return Linear(weight: w, bias: b)
        }

        var audioLayers: [Qwen3ASREncoderLayer] = []
        audioLayers.reserveCapacity(ac.encoderLayers)
        for i in 0 ..< ac.encoderLayers {
            let p = "audio_tower.layers.\(i)"
            audioLayers.append(
                Qwen3ASREncoderLayer(
                    selfAttnNorm: try layerNorm("\(p).self_attn_layer_norm"),
                    finalNorm: try layerNorm("\(p).final_layer_norm"),
                    qProj: try linear("\(p).self_attn.q_proj"),
                    kProj: try linear("\(p).self_attn.k_proj"),
                    vProj: try linear("\(p).self_attn.v_proj"),
                    outProj: try linear("\(p).self_attn.out_proj"),
                    fc1: try linear("\(p).fc1"),
                    fc2: try linear("\(p).fc2"),
                    hidden: ac.dModel, nHeads: ac.encoderAttentionHeads))
        }

        // proj1 and proj2 may not exist in every conversion (some omit them
        // when dModel == outputDim). Use identity matrices as fallback.
        let proj1Weight: Tensor
        let proj1Bias: Tensor
        let proj2Weight: Tensor
        let proj2Bias: Tensor
        if bundle.has("audio_tower.proj1.weight") {
            proj1Weight = try bundle.tensor(named: "audio_tower.proj1.weight")
            proj1Bias = try bundle.tensor(named: "audio_tower.proj1.bias")
            proj2Weight = try bundle.tensor(named: "audio_tower.proj2.weight")
            proj2Bias = try bundle.tensor(named: "audio_tower.proj2.bias")
        } else {
            // Identity fallback: pass-through (dModel → dModel → outputDim).
            proj1Weight = makeIdentity(size: ac.dModel, dtype: dtype)
            proj1Bias = Tensor.empty(shape: [ac.dModel], dtype: dtype)
            proj1Bias.zero()
            proj2Weight = makeIdentity(size: ac.outputDim, dtype: dtype)
            proj2Bias = Tensor.empty(shape: [ac.outputDim], dtype: dtype)
            proj2Bias.zero()
        }

        // ── Text decoder ──
        // Read quantization from the root config.json (top-level `quantization`
        // block — standard for mlx-community 4-bit checkpoints).
        let quant = rootConfig?.quantization

        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: bundle,
            hidden: qc.textHidden, quantization: quant)

        var textLayers: [Qwen3ASRTextLayer] = []
        textLayers.reserveCapacity(qc.textLayers)
        for i in 0 ..< qc.textLayers {
            let p = "model.layers.\(i)"
            let inputNorm = RMSNorm(
                weight: try bundle.tensor(named: "\(p).input_layernorm.weight"),
                eps: qc.rmsNormEps)
            let postAttnNorm = RMSNorm(
                weight: try bundle.tensor(
                    named: "\(p).post_attention_layernorm.weight"),
                eps: qc.rmsNormEps)
            let qNorm = RMSNorm(
                weight: try bundle.tensor(named: "\(p).self_attn.q_norm.weight"),
                eps: qc.rmsNormEps)
            let kNorm = RMSNorm(
                weight: try bundle.tensor(named: "\(p).self_attn.k_norm.weight"),
                eps: qc.rmsNormEps)
            let qProj = try loadLinear(
                base: "\(p).self_attn.q_proj", in: bundle, quantization: quant)
            let kProj = try loadLinear(
                base: "\(p).self_attn.k_proj", in: bundle, quantization: quant)
            let vProj = try loadLinear(
                base: "\(p).self_attn.v_proj", in: bundle, quantization: quant)
            let oProj = try loadLinear(
                base: "\(p).self_attn.o_proj", in: bundle, quantization: quant)
            let gateProj = try loadLinear(
                base: "\(p).mlp.gate_proj", in: bundle, quantization: quant)
            let upProj = try loadLinear(
                base: "\(p).mlp.up_proj", in: bundle, quantization: quant)
            let downProj = try loadLinear(
                base: "\(p).mlp.down_proj", in: bundle, quantization: quant)
            textLayers.append(
                Qwen3ASRTextLayer(
                    inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    qNorm: qNorm, kNorm: kNorm,
                    gateProj: gateProj, upProj: upProj, downProj: downProj))
        }

        let textNorm = RMSNorm(
            weight: try bundle.tensor(named: "model.norm.weight"),
            eps: qc.rmsNormEps)

        // lm_head — tied to embed_tokens on all published Qwen3ASR checkpoints.
        // Honor an explicit untied lm_head.weight if present; otherwise mirror
        // the Qwen35 pattern: wrap quantized embeddings as QuantizedLinear so
        // dequantGemv fires correctly (dense embed.weight is packed uint32 in
        // quantized checkpoints — it cannot be used with a plain Ops.gemv).
        let lmHead: AnyLinear
        if !qc.tieWordEmbeddings, bundle.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: bundle, quantization: quant)
        } else if let q = quant, bundle.isQuantized("model.embed_tokens") {
            // Quantized tied embedding — reuse the embed triplet as a QuantizedLinear.
            let t = try bundle.quantizedTriplet("model.embed_tokens")
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            lmHead = AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: bits, groupSize: q.groupSize))
        } else {
            // Dense (unquantized) tied embedding — safe to use weight directly.
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        return Qwen3ASRModel(
            config: qc,
            conv2d1Weight: conv2d1Weight, conv2d1Bias: conv2d1Bias,
            conv2d2Weight: conv2d2Weight, conv2d2Bias: conv2d2Bias,
            conv2d3Weight: conv2d3Weight, conv2d3Bias: conv2d3Bias,
            convOutWeight: convOutWeight,
            audioPositionalEmbedding: audioPositionalEmbedding,
            audioEncoderLayers: audioLayers,
            lnPost: try layerNorm("audio_tower.ln_post"),
            proj1Weight: proj1Weight, proj1Bias: proj1Bias,
            proj2Weight: proj2Weight, proj2Bias: proj2Bias,
            embedTokens: embedTokens,
            textLayers: textLayers,
            textNorm: textNorm,
            lmHead: lmHead,
            dtype: dtype)
    }

    /// Build a `[size, size]` identity matrix in the given dtype (CPU-side).
    private static func makeIdentity(size: Int, dtype: DType) -> Tensor {
        var vals = [Float](repeating: 0, count: size * size)
        for i in 0 ..< size { vals[i * size + i] = 1.0 }
        let t = Tensor.empty(shape: [size, size], dtype: dtype)
        AudioPreprocessing.copyFloats(vals, into: t)
        return t
    }
}
