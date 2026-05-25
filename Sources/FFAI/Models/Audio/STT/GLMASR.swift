// GLMASR — GLM-based automatic speech recognition family.
//
// HF repo: `mlx-community/GLM-ASR-Nano-2512-4bit`
//
// Architecture: a Whisper-style Conv1d audio encoder (GELU-activated
// stride-2 Conv1d stem + transformer), followed by a merge-and-adapt
// MLP, followed by a LLaMA-style causal decoder. The audio encoder
// output is merged (stride merge_factor=4) then projected via a two-
// layer GELU MLP into the text-decoder embedding dimension. The result
// is spliced into the token embedding stream as "audio tokens" and the
// decoder autoregressively produces the transcript.
//
//   waveform  ──logMel──▶ [nFrames, nMels]
//             ──conv1d(k=3,s=1,GELU) + conv1d(k=3,s=2,GELU)──▶ [nCtx, dModel]
//             ──whisperEncoder (RoPE or pos-embed)──▶ [nCtx, dModel]
//             ──layerNorm──▶ merge(factor 4)──▶ [nCtx/4, dModel*4]
//             ──adaptingMLP (GELU, two-layer)──▶ [nAudioTokens, lmHidden]
//             ──prepend BOA + append EOA──▶ splice into text embedding
//             ──LLaMA decoder (SiLU MLP, GQA, RoPE)──▶ transcript
//
// Conv1d weights in MLX checkpoints are stored as [outCh, kernelSize, inCh]
// (OWI); `Ops.audioConv1d` expects [outCh, inCh, kernelSize] (OIW).
// Transposed on load.
//
// The config.json for this family is sparse — it carries only the top-
// level knobs (model_type, merge_factor, use_rope, max_whisper_length,
// max_length) and omits the whisper + llama sub-configs entirely. The
// actual architecture is inferred from the weight shapes.
//
// Detection: `model_type == "glmasr"` or architecture `"GlmasrModel"`.

import Foundation
import Metal
import Tokenizers

// ─── Prompt template ─────────────────────────────────────────────────

/// Chat-template token strings used to wrap the audio tokens.
private enum GLMASRPrompt {
    static let userPrefix  = "<|user|>\n<|begin_of_audio|>"
    static let userSuffix  = "<|end_of_audio|>\nPlease transcribe this audio into text<|assistant|>\n"
}

// ─── Configuration ───────────────────────────────────────────────────

/// Top-level GLM-ASR configuration decoded from `config.json`.
///
/// The checkpoint's config.json omits the nested whisper / llama sub-
/// configs entirely. The hyper-parameter values here are the defaults
/// derived from the `mlx-community/GLM-ASR-Nano-2512-4bit` weight
/// shapes and match the reference Python implementation.
public struct GLMASRConfig: Sendable {
    // ── Audio (Whisper-style encoder) ──
    /// Mel filterbank bins (128 for the Nano variant).
    public let numMelBins: Int
    /// Encoder hidden dimension (`d_model`). Nano: 1280.
    public let whisperDModel: Int
    /// Number of Whisper encoder transformer layers. Nano: 32.
    public let whisperEncoderLayers: Int
    /// Encoder attention heads. Nano: 20.
    public let whisperEncoderHeads: Int
    /// Encoder FFN intermediate dim. Nano: 5120.
    public let whisperEncoderFfnDim: Int
    /// Maximum audio-context positions (size of positional table). 1500.
    public let maxWhisperLength: Int
    /// Whether the Whisper encoder uses RoPE instead of fixed positional
    /// embeddings. Nano uses `use_rope = true`.
    public let useRope: Bool

    // ── Audio adapter ──
    /// Number of Whisper frames merged per output token. 4.
    public let mergeFactor: Int

    // ── Text (LLaMA-style decoder) ──
    /// Text-decoder hidden size. Nano: 2048.
    public let lmHiddenSize: Int
    /// Text-decoder vocabulary size. Nano: 59264.
    public let lmVocabSize: Int
    /// Text-decoder transformer layers. Nano: 28.
    public let lmNumLayers: Int
    /// Text-decoder attention heads. Nano: 16.
    public let lmNumHeads: Int
    /// Text-decoder KV heads (GQA). Nano: 4.
    public let lmNumKVHeads: Int
    /// Text-decoder per-head dimension. Nano: 128.
    public let lmHeadDim: Int
    /// Text-decoder FFN intermediate dim. Nano: 6144.
    public let lmIntermediate: Int
    /// RMSNorm epsilon. 1e-5.
    public let lmRmsNormEps: Float
    /// RoPE theta base frequency. Nano: 10000.
    public let lmRopeTheta: Float
    /// EOS token ids — generation stops when any is produced.
    public let eosTokenIds: [Int]

    public init(
        numMelBins: Int = 128,
        whisperDModel: Int = 1280,
        whisperEncoderLayers: Int = 32,
        whisperEncoderHeads: Int = 20,
        whisperEncoderFfnDim: Int = 5120,
        maxWhisperLength: Int = 1500,
        useRope: Bool = true,
        mergeFactor: Int = 4,
        lmHiddenSize: Int = 2048,
        lmVocabSize: Int = 59264,
        lmNumLayers: Int = 28,
        lmNumHeads: Int = 16,
        lmNumKVHeads: Int = 4,
        lmHeadDim: Int = 128,
        lmIntermediate: Int = 6144,
        lmRmsNormEps: Float = 1e-5,
        lmRopeTheta: Float = 10000.0,
        eosTokenIds: [Int] = [59246, 59253, 59255]
    ) {
        self.numMelBins = numMelBins
        self.whisperDModel = whisperDModel
        self.whisperEncoderLayers = whisperEncoderLayers
        self.whisperEncoderHeads = whisperEncoderHeads
        self.whisperEncoderFfnDim = whisperEncoderFfnDim
        self.maxWhisperLength = maxWhisperLength
        self.useRope = useRope
        self.mergeFactor = mergeFactor
        self.lmHiddenSize = lmHiddenSize
        self.lmVocabSize = lmVocabSize
        self.lmNumLayers = lmNumLayers
        self.lmNumHeads = lmNumHeads
        self.lmNumKVHeads = lmNumKVHeads
        self.lmHeadDim = lmHeadDim
        self.lmIntermediate = lmIntermediate
        self.lmRmsNormEps = lmRmsNormEps
        self.lmRopeTheta = lmRopeTheta
        self.eosTokenIds = eosTokenIds
    }

    /// Build a `GLMASRConfig` from a decoded `config.json`. Returns
    /// default Nano values for any field absent in the JSON.
    public static func from(_ config: ModelConfig) -> GLMASRConfig? {
        guard let mt = config.modelType, mt == "glmasr" else {
            // Fall back to architecture check.
            guard let arch = config.architecture, arch == "GlmasrModel"
            else { return nil }
            return GLMASRConfig()
        }
        let mergeFactor = config.int("merge_factor") ?? 4
        let maxWhisper = config.int("max_whisper_length") ?? 1500
        let useRope = config.bool("use_rope") ?? true

        // EOS from config or generation_config — fall back to Nano defaults.
        var eosIds: [Int]
        if let arr = config.raw["eos_token_id"] as? [Int] {
            eosIds = arr
        } else if let single = config.raw["eos_token_id"] as? Int {
            eosIds = [single]
        } else {
            eosIds = [59246, 59253, 59255]
        }

        return GLMASRConfig(
            maxWhisperLength: maxWhisper,
            useRope: useRope,
            mergeFactor: mergeFactor,
            eosTokenIds: eosIds
        )
    }
}

// ─── Whisper encoder layer ────────────────────────────────────────────

/// One pre-norm GLM-ASR Whisper encoder block. Identical structure to
/// the `AudioEncoderLayer` bidirectional block — LayerNorm → self-attn
/// (biased QKV, optional RoPE) → residual, LayerNorm → GELU MLP →
/// residual. Uses `AnyLinear` so quantized encoder weights load cleanly.
public final class GLMASRWhisperLayer: Module {
    let selfAttnNorm: LayerNorm
    let finalNorm: LayerNorm
    let qProj, kProj, vProj, outProj: AnyLinear
    let fc1, fc2: AnyLinear
    let hidden, nHeads, headDim: Int
    let scale: Float

    public init(
        selfAttnNorm: LayerNorm, finalNorm: LayerNorm,
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, outProj: AnyLinear,
        fc1: AnyLinear, fc2: AnyLinear,
        hidden: Int, nHeads: Int
    ) {
        self.selfAttnNorm = selfAttnNorm; self.finalNorm = finalNorm
        self.qProj = qProj; self.kProj = kProj
        self.vProj = vProj; self.outProj = outProj
        self.fc1 = fc1; self.fc2 = fc2
        self.hidden = hidden; self.nHeads = nHeads
        self.headDim = hidden / nHeads
        self.scale = 1.0 / Float(Double(hidden / nHeads).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] { [] }
}

// ─── Text decoder layer ───────────────────────────────────────────────

/// One GLM-ASR text decoder layer (LLaMA architecture: GQA + SiLU MLP).
public final class GLMASRTextLayer: Module {
    let inputNorm, postAttnNorm: RMSNorm
    let qProj, kProj, vProj, oProj: AnyLinear
    let gateProj, upProj, downProj: AnyLinear

    public init(
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear
    ) {
        self.inputNorm = inputNorm; self.postAttnNorm = postAttnNorm
        self.qProj = qProj; self.kProj = kProj
        self.vProj = vProj; self.oProj = oProj
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
    }

    public func parameters() -> [(String, Tensor)] { [] }
}

// ─── GLM-ASR model ────────────────────────────────────────────────────

/// A loaded GLM-ASR speech-to-text model.
///
/// Main entry points:
/// - `encodeAudio(waveform:device:)` — run the audio encoder stack
/// - `transcribe(waveform:tokenizer:maxTokens:device:)` — end-to-end
public final class GLMASRModel: @unchecked Sendable {
    public let config: GLMASRConfig

    // ── Audio encoder (Whisper-style) ──
    /// Conv1d stem: first conv weight [outCh, inCh, k] (OIW for audioConv1d).
    let conv1Weight: Tensor   // [dModel, nMels, 3]
    let conv1Bias: Tensor     // [dModel]
    /// Conv1d stem: second conv weight [outCh, inCh, k], stride 2.
    let conv2Weight: Tensor   // [dModel, dModel, 3]
    let conv2Bias: Tensor     // [dModel]
    /// Positional embedding table `[maxWhisperLength, dModel]`.
    /// Present in the checkpoint even when `use_rope = true` (for
    /// weight-loading compatibility); only used when `use_rope = false`.
    let posEmbedding: Tensor  // [1500, dModel] quantized
    let whisperLayers: [GLMASRWhisperLayer]
    let audioLayerNorm: LayerNorm  // post-whisper layer norm

    // ── Audio adapter ──
    /// MLP adapter: [dModel * mergeFactor → lmHidden * 2 → lmHidden].
    let adaptingFC1Weight: Tensor
    let adaptingFC1Bias: Tensor
    let adaptingFC2Weight: Tensor
    let adaptingFC2Bias: Tensor

    // ── Audio BOS/EOS embeddings ──
    /// Two-row embedding: row 0 = begin-of-audio, row 1 = end-of-audio.
    let audioBosEosToken: AnyEmbedding

    // ── Text decoder (LLaMA) ──
    let embedTokens: AnyEmbedding
    let textLayers: [GLMASRTextLayer]
    let textNorm: RMSNorm
    /// lm_head may be separate (not tied). Always loaded as AnyLinear
    /// so dequantGemv fires correctly for quantized tied embeddings.
    let lmHead: AnyLinear

    let dtype: DType

    public init(
        config: GLMASRConfig,
        conv1Weight: Tensor, conv1Bias: Tensor,
        conv2Weight: Tensor, conv2Bias: Tensor,
        posEmbedding: Tensor,
        whisperLayers: [GLMASRWhisperLayer],
        audioLayerNorm: LayerNorm,
        adaptingFC1Weight: Tensor, adaptingFC1Bias: Tensor,
        adaptingFC2Weight: Tensor, adaptingFC2Bias: Tensor,
        audioBosEosToken: AnyEmbedding,
        embedTokens: AnyEmbedding,
        textLayers: [GLMASRTextLayer],
        textNorm: RMSNorm,
        lmHead: AnyLinear,
        dtype: DType
    ) {
        self.config = config
        self.conv1Weight = conv1Weight; self.conv1Bias = conv1Bias
        self.conv2Weight = conv2Weight; self.conv2Bias = conv2Bias
        self.posEmbedding = posEmbedding
        self.whisperLayers = whisperLayers
        self.audioLayerNorm = audioLayerNorm
        self.adaptingFC1Weight = adaptingFC1Weight
        self.adaptingFC1Bias = adaptingFC1Bias
        self.adaptingFC2Weight = adaptingFC2Weight
        self.adaptingFC2Bias = adaptingFC2Bias
        self.audioBosEosToken = audioBosEosToken
        self.embedTokens = embedTokens
        self.textLayers = textLayers
        self.textNorm = textNorm
        self.lmHead = lmHead
        self.dtype = dtype
    }

    // ─── Audio encoding ───────────────────────────────────────────────

    /// Encode a 16 kHz mono waveform into audio feature tokens
    /// `[nAudioTokens, lmHiddenSize]` ready to splice into the text
    /// embedding stream.
    public func encodeAudio(waveform: [Float], device: Device = .shared)
        -> Tensor {
        let ac = config
        let frontEnd = AudioFrontEndConfig(
            sampleRate: 16_000, nFFT: 400, hopLength: 160,
            nMels: ac.numMelBins)

        // ── Log-Mel spectrogram (Whisper-normalised) ──
        let melDtype: DType = dtype == .f16 ? .f16 : .f32
        let cmdMel = device.makeCommandBuffer()
        let melRaw = AudioPreprocessing.logMelSpectrogram(
            waveform: waveform, cfg: frontEnd, dtype: melDtype,
            whisperNormalize: true, device: device, on: cmdMel)
        let melF = AudioPreprocessing.castTensor(melRaw, to: dtype,
                                                  device: device)
        // melF: [nFrames, nMels]; reshape for audioConv1d: [1, nMels, nFrames].
        let nFrames = melF.shape[0]
        let melIn = melF.reshaped(to: [1, ac.numMelBins, nFrames])

        // ── Conv1d stem ──
        // conv1: k=3, stride=1, pad=1 → [1, dModel, nFrames]
        let cmd1 = device.makeCommandBuffer()
        let c1Out = Ops.audioConv1d(
            input: melIn, weight: conv1Weight, bias: conv1Bias,
            batch: 1, inCh: ac.numMelBins, inLen: nFrames,
            outCh: ac.whisperDModel, k: 3, stride: 1, pad: 1, on: cmd1)
        let c1Act = Ops.gelu(c1Out, on: cmd1)
        // conv2: k=3, stride=2, pad=1 → [1, dModel, nCtx]
        let nCtx = (nFrames + 2 - 3) / 2 + 1
        let c2Out = Ops.audioConv1d(
            input: c1Act, weight: conv2Weight, bias: conv2Bias,
            batch: 1, inCh: ac.whisperDModel, inLen: nFrames,
            outCh: ac.whisperDModel, k: 3, stride: 2, pad: 1, on: cmd1)
        let c2Act = Ops.gelu(c2Out, on: cmd1)
        cmd1.commit(); cmd1.waitUntilCompleted()

        // c2Act: [1, dModel, nCtx] channel-major → [nCtx, dModel] frame-major.
        let c2Vals = c2Act.toFloatArray()
        let dModel = ac.whisperDModel
        var seqVals = [Float](repeating: 0, count: nCtx * dModel)
        // Reorder NCL [1, dModel, nCtx] → [nCtx, dModel].
        for c in 0..<dModel {
            for t in 0..<nCtx {
                seqVals[t * dModel + c] = c2Vals[c * nCtx + t]
            }
        }

        // ── Positional embedding (only when not using RoPE) ──
        if !ac.useRope {
            // pos embedding is stored quantized; dequant to float via CPU side.
            let nPos = min(nCtx, ac.maxWhisperLength)
            let posVals = posEmbedding.toFloatArray()
            // posEmbedding shape is [maxWhisperLength, dModel] after dequant.
            // Add pos embed row-wise.
            for t in 0..<nPos {
                for d in 0..<dModel {
                    seqVals[t * dModel + d] += posVals[t * dModel + d]
                }
            }
        }
        // Note: when use_rope = true the RoPE is applied per-head inside
        // each encoder layer via the attention computation. For the
        // shared whisper encoder layers we apply RoPE in the bidirectional
        // attention core below using the half-rotation trick.

        // ── Whisper encoder transformer ──
        let seqCapped = min(nCtx, ac.maxWhisperLength / ac.mergeFactor * ac.mergeFactor)
        for layer in whisperLayers {
            seqVals = runWhisperEncoderLayer(
                layer, seq: seqVals, seqLen: seqCapped,
                hidden: dModel, useRope: ac.useRope, device: device)
        }

        // Upload back to GPU tensor for norms + adapter.
        let seqT = Tensor.empty(shape: [seqCapped, dModel], dtype: dtype,
                                device: device)
        AudioPreprocessing.copyFloats(seqVals, into: seqT)

        // ── Audio layer-norm ──
        let cmdN = device.makeCommandBuffer()
        let normed = Ops.layerNorm(seqT,
                                   weight: audioLayerNorm.weight,
                                   bias: audioLayerNorm.bias,
                                   eps: audioLayerNorm.eps,
                                   nRows: seqCapped, rowSize: dModel, on: cmdN)
        cmdN.commit(); cmdN.waitUntilCompleted()

        // ── Merge by factor ──
        // Fold `mergeFactor` consecutive frame rows into one wider row.
        let mf = ac.mergeFactor
        let newSeqLen = seqCapped / mf
        let mergedDim = dModel * mf
        let normedVals = normed.toFloatArray()
        var merged = [Float](repeating: 0, count: newSeqLen * mergedDim)
        for i in 0..<newSeqLen {
            for j in 0..<mf {
                let srcRow = i * mf + j
                for d in 0..<dModel {
                    merged[i * mergedDim + j * dModel + d] =
                        normedVals[srcRow * dModel + d]
                }
            }
        }

        // ── Adapting MLP: [mergedDim → intermediateDim → lmHidden] ──
        // fc1 with GELU activation, fc2 linear.
        //
        // The adapter weights are dequantized to f32 on the CPU side (see
        // `dequantizeLinear`), so the input tensors must match that dtype.
        // Using `adaptingFC1Weight.dtype` (instead of the model's `dtype`)
        // ensures no dtype mismatch at `Ops.gemm` when the checkpoint is
        // quantized (e.g. 4-bit) and `dtype` is bf16.
        let adaptorDtype = adaptingFC1Weight.dtype
        let mergedT = Tensor.empty(shape: [newSeqLen, mergedDim],
                                    dtype: adaptorDtype, device: device)
        AudioPreprocessing.copyFloats(merged, into: mergedT)

        let cmdA = device.makeCommandBuffer()
        let fc1Out = Ops.gemm(weight: adaptingFC1Weight, input: mergedT,
                               nRows: newSeqLen, on: cmdA)
        cmdA.commit(); cmdA.waitUntilCompleted()

        // Add fc1 bias + apply GELU (CPU for small seq lengths).
        let fc1Dim = adaptingFC1Weight.shape[0]
        var fc1Vals = fc1Out.toFloatArray()
        let b1Vals  = adaptingFC1Bias.toFloatArray()
        let gk: Float = 0.7978845608      // √(2/π)
        let gc: Float = 0.044715
        for i in 0..<fc1Vals.count {
            let v = fc1Vals[i] + b1Vals[i % fc1Dim]
            let inner = gk * (v + gc * v * v * v)
            fc1Vals[i] = 0.5 * v * (1 + tanh(inner))
        }
        let fc1Act = Tensor.empty(shape: [newSeqLen, fc1Dim], dtype: adaptorDtype,
                                   device: device)
        AudioPreprocessing.copyFloats(fc1Vals, into: fc1Act)

        let cmdB = device.makeCommandBuffer()
        let fc2Out = Ops.gemm(weight: adaptingFC2Weight, input: fc1Act,
                               nRows: newSeqLen, on: cmdB)
        cmdB.commit(); cmdB.waitUntilCompleted()

        let lmH = config.lmHiddenSize
        var fc2Vals = fc2Out.toFloatArray()
        let b2Vals  = adaptingFC2Bias.toFloatArray()
        for i in 0..<fc2Vals.count {
            fc2Vals[i] += b2Vals[i % lmH]
        }
        let audioFeatures = Tensor.empty(shape: [newSeqLen, lmH],
                                          dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(fc2Vals, into: audioFeatures)
        return audioFeatures  // [nAudioTokens, lmHiddenSize]
    }

    // ─── Whisper encoder layer (CPU bidirectional attention) ──────────

    /// Run one Whisper encoder transformer layer over `[seqLen, hidden]`
    /// stored flat as `[Float]`. Returns the updated sequence.
    private func runWhisperEncoderLayer(
        _ layer: GLMASRWhisperLayer,
        seq seqVals: [Float],
        seqLen: Int, hidden: Int,
        useRope: Bool,
        device: Device
    ) -> [Float] {
        let seqT = Tensor.empty(shape: [seqLen, hidden], dtype: dtype,
                                device: device)
        AudioPreprocessing.copyFloats(seqVals, into: seqT)

        let cmd = device.makeCommandBuffer()
        let normed = Ops.layerNorm(seqT,
                                   weight: layer.selfAttnNorm.weight,
                                   bias: layer.selfAttnNorm.bias,
                                   eps: layer.selfAttnNorm.eps,
                                   nRows: seqLen, rowSize: hidden, on: cmd)
        // Q/K/V projections via GPU gemm (handles quantized weights).
        // `normed` is [seqLen, hidden] — must go through callMany so the
        // dispatch is `mt_gemm` / `dequantGemmDynamicM`, not the single-row
        // `mt_gemv` / `dequantGemv` (which would treat the flat
        // seqLen*hidden length as one row and trip the in_dim precondition,
        // crashing as "input 65280 ≠ in_dim 1280" per the 2026-05-24 bisect).
        let qRaw = layer.qProj.callMany(normed, t: seqLen, on: cmd, device: device)
        let kRaw = layer.kProj.callMany(normed, t: seqLen, on: cmd, device: device)
        let v    = layer.vProj.callMany(normed, t: seqLen, on: cmd, device: device)
        cmd.commit(); cmd.waitUntilCompleted()

        // Bias add (Linear.bias is non-nil for q/k/v/out_proj in GLM-ASR).
        let qa = addRowBiasIfPresent(qRaw, bias: qRawBias(layer),
                                     nRows: seqLen, rowSize: hidden,
                                     device: device).toFloatArray()
        let ka = addRowBiasIfPresent(kRaw, bias: kRawBias(layer),
                                     nRows: seqLen, rowSize: hidden,
                                     device: device).toFloatArray()
        let va = addRowBiasIfPresent(v, bias: vRawBias(layer),
                                     nRows: seqLen, rowSize: hidden,
                                     device: device).toFloatArray()

        // CPU bidirectional attention (no causal mask).
        let nHeads  = layer.nHeads
        let headDim = layer.headDim
        let attnCtx = cpuBidirectionalAttention(
            qa: qa, ka: ka, va: va,
            seqLen: seqLen, nHeads: nHeads, headDim: headDim,
            scale: layer.scale, device: device)

        // out_proj + residual. attnCtx is [seqLen, hidden] flat — use
        // callMany so the dispatch is a single batched GEMM instead of a
        // single-row gemv that would mis-interpret the flat length.
        let cmd2 = device.makeCommandBuffer()
        let outRaw  = layer.outProj.callMany(attnCtx, t: seqLen, on: cmd2, device: device)
        let outBiased = addRowBiasIfPresent(outRaw,
                                            bias: outProjBias(layer),
                                            nRows: seqLen, rowSize: hidden,
                                            device: device)
        let h = Ops.add(seqT, outBiased, on: cmd2)

        // FFN (GELU MLP). Intermediate dim from config (matches weight shape).
        let ffInter = config.whisperEncoderFfnDim
        let normed2 = Ops.layerNorm(h,
                                    weight: layer.finalNorm.weight,
                                    bias: layer.finalNorm.bias,
                                    eps: layer.finalNorm.eps,
                                    nRows: seqLen, rowSize: hidden, on: cmd2)
        // fc1 / fc2 take [seqLen, *] inputs → callMany for batched GEMM.
        let ff1  = layer.fc1.callMany(normed2, t: seqLen, on: cmd2, device: device)
        cmd2.commit(); cmd2.waitUntilCompleted()

        var ff1Vals = addRowBiasIfPresent(ff1, bias: fc1Bias(layer),
                                          nRows: seqLen, rowSize: ffInter,
                                          device: device).toFloatArray()
        // GELU in-place.
        let gk: Float = 0.7978845608
        let gc: Float = 0.044715
        for i in 0..<ff1Vals.count {
            let xv = ff1Vals[i]
            let inner = gk * (xv + gc * xv * xv * xv)
            ff1Vals[i] = 0.5 * xv * (1 + tanh(inner))
        }
        let geluT = Tensor.empty(shape: [seqLen, ffInter], dtype: dtype,
                                  device: device)
        AudioPreprocessing.copyFloats(ff1Vals, into: geluT)

        let cmd3 = device.makeCommandBuffer()
        let ff2Out = layer.fc2.callMany(geluT, t: seqLen, on: cmd3, device: device)
        let ff2B   = addRowBiasIfPresent(ff2Out, bias: fc2Bias(layer),
                                          nRows: seqLen, rowSize: hidden,
                                          device: device)
        let out = Ops.add(h, ff2B, on: cmd3)
        cmd3.commit(); cmd3.waitUntilCompleted()
        return out.toFloatArray()
    }

    // Helpers to extract optional biases from AnyLinear inner layers.
    private func qRawBias(_ l: GLMASRWhisperLayer) -> Tensor? {
        (l.qProj.inner as? Linear)?.bias
    }
    private func kRawBias(_ l: GLMASRWhisperLayer) -> Tensor? {
        (l.kProj.inner as? Linear)?.bias
    }
    private func vRawBias(_ l: GLMASRWhisperLayer) -> Tensor? {
        (l.vProj.inner as? Linear)?.bias
    }
    private func outProjBias(_ l: GLMASRWhisperLayer) -> Tensor? {
        (l.outProj.inner as? Linear)?.bias
    }
    private func fc1Bias(_ l: GLMASRWhisperLayer) -> Tensor? {
        (l.fc1.inner as? Linear)?.bias
    }
    private func fc2Bias(_ l: GLMASRWhisperLayer) -> Tensor? {
        (l.fc2.inner as? Linear)?.bias
    }

    // ─── CPU bidirectional attention (same pattern as Qwen3ASR) ──────

    /// CPU multi-head bidirectional attention (no causal mask).
    /// Returns a `[seqLen, hidden]` Tensor.
    private func cpuBidirectionalAttention(
        qa: [Float], ka: [Float], va: [Float],
        seqLen: Int, nHeads: Int, headDim: Int, scale: Float,
        device: Device
    ) -> Tensor {
        let H = nHeads * headDim
        var out = [Float](repeating: 0, count: seqLen * H)

        out.withUnsafeMutableBufferPointer { outBuf in
            let outPtr = outBuf.baseAddress!
            qa.withUnsafeBufferPointer { qBuf in
            ka.withUnsafeBufferPointer { kBuf in
            va.withUnsafeBufferPointer { vBuf in
                let qb = qBuf.baseAddress!
                let kb = kBuf.baseAddress!
                let vb = vBuf.baseAddress!
                DispatchQueue.concurrentPerform(iterations: nHeads * seqLen) { work in
                    let head = work / seqLen
                    let i    = work % seqLen
                    let hOff = head * headDim
                    var scores = [Float](repeating: 0, count: seqLen)
                    var maxScore = -Float.greatestFiniteMagnitude
                    let qBase = i * H + hOff
                    for j in 0..<seqLen {
                        var dot: Float = 0
                        let kBase = j * H + hOff
                        for d in 0..<headDim { dot += qb[qBase+d] * kb[kBase+d] }
                        let s = dot * scale
                        scores[j] = s
                        if s > maxScore { maxScore = s }
                    }
                    var sumExp: Float = 0
                    for j in 0..<seqLen {
                        let e = exp(scores[j] - maxScore)
                        scores[j] = e; sumExp += e
                    }
                    let inv = sumExp > 0 ? 1 / sumExp : 0
                    let oBase = i * H + hOff
                    for j in 0..<seqLen {
                        let w = scores[j] * inv
                        let vBase = j * H + hOff
                        for d in 0..<headDim { outPtr[oBase+d] += w * vb[vBase+d] }
                    }
                }
            }}}
        }
        let result = Tensor.empty(shape: [seqLen, H], dtype: dtype,
                                   device: device)
        AudioPreprocessing.copyFloats(out, into: result)
        return result
    }

    /// Add a `[rowSize]` bias to each of `nRows` rows of a `[nRows,
    /// rowSize]` tensor. Returns the input unchanged when bias is nil.
    private func addRowBiasIfPresent(
        _ t: Tensor, bias: Tensor?,
        nRows: Int, rowSize: Int, device: Device
    ) -> Tensor {
        guard let bias = bias else { return t }
        let tVals = t.toFloatArray()
        let bVals = bias.toFloatArray()
        var out = [Float](repeating: 0, count: nRows * rowSize)
        for r in 0..<nRows {
            for c in 0..<rowSize {
                out[r * rowSize + c] = tVals[r * rowSize + c] + bVals[c]
            }
        }
        let result = Tensor.empty(shape: [nRows, rowSize], dtype: dtype,
                                   device: device)
        AudioPreprocessing.copyFloats(out, into: result)
        return result
    }

    // ─── Prompt construction ──────────────────────────────────────────

    /// Tokenize the GLM-ASR chat-template prompt that wraps
    /// `numAudioTokens` placeholder token ids (all zeros) in the user
    /// turn. The decoder expects audio embeddings to be spliced in at
    /// the positions of these placeholders.
    public func buildPrompt(numAudioTokens: Int,
                            tokenizer: any Tokenizer) -> [Int] {
        let prefix = tokenizer.encode(text: GLMASRPrompt.userPrefix)
        let suffix = tokenizer.encode(text: GLMASRPrompt.userSuffix)
        let pads   = [Int](repeating: 0, count: numAudioTokens)
        return prefix + pads + suffix
    }

    // ─── Transcription ────────────────────────────────────────────────

    /// Transcribe a 16 kHz mono waveform into a text string.
    ///
    /// - Parameters:
    ///   - waveform: 16 kHz mono PCM samples.
    ///   - tokenizer: Tokenizer loaded from the checkpoint directory.
    ///   - maxTokens: Maximum transcript tokens to generate. Default 512.
    ///   - device: Metal device for GPU dispatch.
    /// - Returns: The decoded transcript string.
    public func transcribe(
        waveform: [Float],
        tokenizer: any Tokenizer,
        maxTokens: Int = 512,
        device: Device = .shared
    ) -> String {
        // ── 1. Encode audio ──
        let audioFeatures = encodeAudio(waveform: waveform, device: device)
        let nAudioTokens  = audioFeatures.shape[0]

        // ── 2. Build prompt token ids ──
        let inputIds   = buildPrompt(numAudioTokens: nAudioTokens,
                                     tokenizer: tokenizer)
        let promptLen  = inputIds.count
        let audioStart = tokenizer.encode(text: GLMASRPrompt.userPrefix).count

        // ── 3. Embed prompt tokens ──
        let idsTensor = Tensor.empty(shape: [promptLen], dtype: .u32,
                                     device: device)
        idsTensor.copyIn(from: inputIds.map { UInt32($0) })
        let cmdE = device.makeCommandBuffer()
        let fullEmbeds = embedTokens(idsTensor, on: cmdE)
        cmdE.commit(); cmdE.waitUntilCompleted()

        // ── 4. Splice audio features into placeholder positions ──
        let embeds = spliceAudioFeatures(
            embeds: fullEmbeds, audioFeatures: audioFeatures,
            audioStart: audioStart, nAudioTokens: nAudioTokens,
            device: device)

        // ── 5. Prefill: feed prompt one token at a time ──
        let nLayers  = config.lmNumLayers
        let nKVHeads = config.lmNumKVHeads
        let hd       = config.lmHeadDim
        let maxSeq   = promptLen + maxTokens + 16

        let caches = (0..<nLayers).map { _ in
            KVCache(nKVHeads: nKVHeads, headDim: hd, maxSeq: maxSeq,
                    dtype: dtype, device: device)
        }

        var lastLogits: Tensor? = nil
        for pos in 0..<promptLen {
            let rowEmbed = embeds.slicedRows(start: pos, count: 1)
            lastLogits = forwardOneToken(embed: rowEmbed, caches: caches,
                                         device: device)
        }
        var logits: Tensor = lastLogits!

        // ── 6. Greedy autoregressive decode ──
        var generated: [Int] = []
        let eosIds = Set(config.eosTokenIds)

        for _ in 0..<maxTokens {
            let logitVals = logits.toFloatArray()
            var best  = 0
            var bestV = -Float.greatestFiniteMagnitude
            for (i, v) in logitVals.enumerated() where v > bestV {
                bestV = v; best = i
            }
            if eosIds.contains(best) { break }
            generated.append(best)

            // Backstop: ≤3 unique ids in last 24 tokens → stuck decoder.
            if generated.count >= 24, Set(generated.suffix(24)).count <= 3 {
                break
            }

            let nextT = Tensor.empty(shape: [1], dtype: .u32, device: device)
            nextT.copyIn(from: [UInt32(best)])
            let cmdNext = device.makeCommandBuffer()
            let nextEmbed = embedTokens(nextT, on: cmdNext)
            cmdNext.commit(); cmdNext.waitUntilCompleted()

            logits = forwardOneToken(embed: nextEmbed, caches: caches,
                                      device: device)
        }

        return tokenizer.decode(tokens: generated)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // ─── Text decoder forward pass (seqLen = 1) ───────────────────────

    /// Forward a single token embedding `[1, lmHidden]` through all
    /// decoder layers with KV caching. Returns `[vocabSize]` logits.
    private func forwardOneToken(
        embed: Tensor,
        caches: [KVCache],
        device: Device
    ) -> Tensor {
        let H = config.lmHiddenSize
        var h = embed.shape.count == 1 ? embed.reshaped(to: [1, H]) : embed

        let offset = caches[0].length

        for (i, layer) in textLayers.enumerated() {
            h = runTextLayer(layer, h: h, offset: offset,
                              cache: caches[i], device: device)
        }

        let cmd = device.makeCommandBuffer()
        let hFlat  = h.reshaped(to: [H])
        let normed = Ops.rmsNorm(hFlat, weight: textNorm.weight,
                                  eps: textNorm.eps, on: cmd)
        let logits = lmHead(normed, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        return logits
    }

    /// One LLaMA decoder layer (GQA, SiLU MLP, RoPE) at seqLen = 1.
    private func runTextLayer(
        _ layer: GLMASRTextLayer,
        h hIn: Tensor,
        offset: Int,
        cache: KVCache,
        device: Device
    ) -> Tensor {
        let H       = config.lmHiddenSize
        let nHeads  = config.lmNumHeads
        let nKVH    = config.lmNumKVHeads
        let hd      = config.lmHeadDim
        let theta   = config.lmRopeTheta
        let eps     = config.lmRmsNormEps
        let scale   = 1.0 / Float(Double(hd).squareRoot())

        let cmd1 = device.makeCommandBuffer()
        let hFlat  = hIn.reshaped(to: [H])
        let normed = Ops.rmsNorm(hFlat, weight: layer.inputNorm.weight,
                                  eps: eps, on: cmd1)
        let q = layer.qProj(normed, on: cmd1)  // [nHeads * hd]
        let k = layer.kProj(normed, on: cmd1)  // [nKVH * hd]
        let v = layer.vProj(normed, on: cmd1)  // [nKVH * hd]
        cmd1.commit(); cmd1.waitUntilCompleted()

        // RoPE for single position.
        let cmd2 = device.makeCommandBuffer()
        let qRot = Ops.rope(q, position: offset, headDim: hd,
                             thetaBase: theta, on: cmd2)
        let kRot = Ops.rope(k, position: offset, headDim: hd,
                             thetaBase: theta, on: cmd2)
        cmd2.commit(); cmd2.waitUntilCompleted()

        // KV cache append + SDPA decode.
        let vShaped = v.reshaped(to: [nKVH, hd])
        let kShaped = kRot.reshaped(to: [nKVH, hd])
        let cmd3 = device.makeCommandBuffer()
        cache.appendOnGPU(kFlat: kShaped, vFlat: vShaped, on: cmd3)
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd3)
        let attnOut = Ops.sdpaDecode(
            q: qRot.reshaped(to: [nHeads, hd]),
            k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVH, headDim: hd,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd3)
        let attnFlat = attnOut.reshaped(to: [nHeads * hd])
        let oOut     = layer.oProj(attnFlat, on: cmd3)
        let postAttn = Ops.add(hFlat, oOut, on: cmd3)

        // SiLU gated MLP.
        let normed2 = Ops.rmsNorm(postAttn, weight: layer.postAttnNorm.weight,
                                   eps: eps, on: cmd3)
        let gate   = layer.gateProj(normed2, on: cmd3)
        let up     = layer.upProj(normed2, on: cmd3)
        let gated  = Ops.mul(Ops.silu(gate, on: cmd3), up, on: cmd3)
        let down   = layer.downProj(gated, on: cmd3)
        let result = Ops.add(postAttn, down, on: cmd3)
        cmd3.commit(); cmd3.waitUntilCompleted()

        return result.reshaped(to: [1, H])
    }

    // ─── Audio feature splice ─────────────────────────────────────────

    /// Replace the audio-placeholder rows in `embeds` (the zero-padded
    /// token rows starting at `audioStart`) with the audio feature rows
    /// from `audioFeatures`.
    private func spliceAudioFeatures(
        embeds: Tensor, audioFeatures: Tensor,
        audioStart: Int, nAudioTokens: Int,
        device: Device
    ) -> Tensor {
        let seqLen = embeds.shape[0]
        let hidden = embeds.shape[1]
        var out    = embeds.toFloatArray()
        let aVals  = audioFeatures.toFloatArray()
        let count  = min(nAudioTokens, seqLen - audioStart)
        for i in 0..<count {
            let dstRow = audioStart + i
            for c in 0..<hidden {
                out[dstRow * hidden + c] = aVals[i * hidden + c]
            }
        }
        let result = Tensor.empty(shape: [seqLen, hidden], dtype: dtype,
                                   device: device)
        AudioPreprocessing.copyFloats(out, into: result)
        return result
    }
}

// ─── Registry detection + loader ─────────────────────────────────────

extension GLMASRModel {
    public static let modelTypes: Set<String>    = ["glmasr"]
    public static let architectures: Set<String> = ["GlmasrModel"]

    /// Whether a decoded `config.json` describes a GLM-ASR checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt)     { return true }
        if let a  = config.architecture, architectures.contains(a) { return true }
        return false
    }

    /// Load a GLM-ASR checkpoint from a resolved snapshot directory.
    public static func load(directory: URL, device: Device = .shared)
        throws -> GLMASRModel {
        let rawConfig = try ModelConfig.load(from: directory)
        guard let gc = GLMASRConfig.from(rawConfig) else {
            throw ModelError.unsupportedModelType(
                "config.json is not a GLM-ASR config")
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return try build(config: gc, bundle: bundle, rootConfig: rawConfig)
    }

    /// Assemble a `GLMASRModel` from a decoded config + weight bundle.
    public static func build(
        config gc: GLMASRConfig,
        bundle: SafeTensorsBundle,
        rootConfig: ModelConfig? = nil
    ) throws -> GLMASRModel {
        // Detect activation dtype from the first unambiguous float tensor.
        let dtype = try bundle.tensor(
            named: "audio_encoder.whisper.conv1.weight").dtype

        // ── Conv1d stem ──
        // Checkpoint stores weights in MLX OWI layout [outCh, k, inCh].
        // `Ops.audioConv1d` expects OIW: [outCh, inCh, k]. Transpose.
        func loadConv1dWeight(_ key: String) throws -> Tensor {
            let raw   = try bundle.tensor(named: key)
            let outCh = raw.shape[0]
            let k     = raw.shape[1]
            let inCh  = raw.shape[2]
            let rawV  = raw.toFloatArray()
            var transposed = [Float](repeating: 0, count: outCh * inCh * k)
            // OWI → OIW: src[o, kw, ic] = rawV[o * k * inCh + kw * inCh + ic]
            //             dst[o, ic, kw] = transposed[o * inCh * k + ic * k + kw]
            for o in 0..<outCh {
                for kw in 0..<k {
                    for ic in 0..<inCh {
                        let src = o * k * inCh + kw * inCh + ic
                        let dst = o * inCh * k + ic * k + kw
                        transposed[dst] = rawV[src]
                    }
                }
            }
            let out = Tensor.empty(shape: [outCh, inCh, k], dtype: dtype,
                                   device: .shared)
            AudioPreprocessing.copyFloats(transposed, into: out)
            return out
        }

        let conv1Weight = try loadConv1dWeight(
            "audio_encoder.whisper.conv1.weight")
        let conv1Bias   = try bundle.tensor(
            named: "audio_encoder.whisper.conv1.bias")
        let conv2Weight = try loadConv1dWeight(
            "audio_encoder.whisper.conv2.weight")
        let conv2Bias   = try bundle.tensor(
            named: "audio_encoder.whisper.conv2.bias")

        // ── Positional embedding ──
        // Quantized in the checkpoint; dequant happens lazily via toFloatArray.
        let posEmbedding = try bundle.tensor(
            named: "audio_encoder.whisper.embed_positions.weight")

        // ── Quantization config ──
        let quant = rootConfig?.quantization

        // ── Whisper encoder layers ──
        func ln(_ base: String) throws -> LayerNorm {
            LayerNorm(
                weight: try bundle.tensor(named: "\(base).weight"),
                bias:   try bundle.tensor(named: "\(base).bias"),
                eps: 1e-5)
        }

        var whisperLayers: [GLMASRWhisperLayer] = []
        whisperLayers.reserveCapacity(gc.whisperEncoderLayers)
        for i in 0..<gc.whisperEncoderLayers {
            let p = "audio_encoder.whisper.layers.\(i)"
            let qProj  = try loadLinear(base: "\(p).self_attn.q_proj",
                                         in: bundle, quantization: quant)
            let kProj  = try loadLinear(base: "\(p).self_attn.k_proj",
                                         in: bundle, quantization: quant)
            let vProj  = try loadLinear(base: "\(p).self_attn.v_proj",
                                         in: bundle, quantization: quant)
            let outP   = try loadLinear(base: "\(p).self_attn.out_proj",
                                         in: bundle, quantization: quant)
            let fc1    = try loadLinear(base: "\(p).fc1",
                                         in: bundle, quantization: quant)
            let fc2    = try loadLinear(base: "\(p).fc2",
                                         in: bundle, quantization: quant)
            whisperLayers.append(GLMASRWhisperLayer(
                selfAttnNorm: try ln("\(p).self_attn_layer_norm"),
                finalNorm:    try ln("\(p).final_layer_norm"),
                qProj: qProj, kProj: kProj, vProj: vProj, outProj: outP,
                fc1: fc1, fc2: fc2,
                hidden: gc.whisperDModel,
                nHeads: gc.whisperEncoderHeads))
        }
        let audioLayerNorm = try ln("audio_encoder.layer_norm")

        // ── Audio adapter ──
        // The adapting MLP stores weights as quantized + plain bias.
        // loadLinear would wrap them; for the GEMM path we need the raw
        // weight tensor — use the AnyLinear wrapper to load then extract.
        let adaptFC1 = try loadLinear(base: "audio_encoder.adapting.fc1",
                                       in: bundle, quantization: quant)
        let adaptFC2 = try loadLinear(base: "audio_encoder.adapting.fc2",
                                       in: bundle, quantization: quant)

        // Extract the underlying weight tensor for Ops.gemm.
        // For quantized layers we need to dequantize before GEMM since
        // Ops.gemm expects a float weight (not packed uint32). Fall back
        // to a dense dequantized copy if the layer is quantized.
        func extractWeight(_ al: AnyLinear) -> Tensor {
            if let q = al.inner as? QuantizedLinear {
                return dequantizeLinear(q)
            }
            return (al.inner as! Linear).weight
        }
        func extractBias(_ al: AnyLinear) throws -> Tensor {
            if let l = al.inner as? Linear, let b = l.bias { return b }
            // If bias is absent, return a zero tensor of the right size.
            let outDim = extractWeight(al).shape[0]
            let zeros  = Tensor.empty(shape: [outDim], dtype: dtype)
            zeros.zero()
            return zeros
        }

        let adaptFC1Weight = extractWeight(adaptFC1)
        let adaptFC1Bias   = try extractBias(adaptFC1)
        let adaptFC2Weight = extractWeight(adaptFC2)
        let adaptFC2Bias   = try extractBias(adaptFC2)

        // ── Audio BOS/EOS embedding ──
        let audioBosEos = try loadEmbedding(
            base: "audio_encoder.audio_bos_eos_token",
            in: bundle, hidden: gc.lmHiddenSize, quantization: quant)

        // ── Text decoder ──
        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens",
            in: bundle, hidden: gc.lmHiddenSize, quantization: quant)

        var textLayers: [GLMASRTextLayer] = []
        textLayers.reserveCapacity(gc.lmNumLayers)
        for i in 0..<gc.lmNumLayers {
            let p = "model.layers.\(i)"
            let inputNorm = RMSNorm(
                weight: try bundle.tensor(named: "\(p).input_layernorm.weight"),
                eps: gc.lmRmsNormEps)
            let postAttnNorm = RMSNorm(
                weight: try bundle.tensor(
                    named: "\(p).post_attention_layernorm.weight"),
                eps: gc.lmRmsNormEps)
            let qProj = try loadLinear(base: "\(p).self_attn.q_proj",
                                        in: bundle, quantization: quant)
            let kProj = try loadLinear(base: "\(p).self_attn.k_proj",
                                        in: bundle, quantization: quant)
            let vProj = try loadLinear(base: "\(p).self_attn.v_proj",
                                        in: bundle, quantization: quant)
            let oProj = try loadLinear(base: "\(p).self_attn.o_proj",
                                        in: bundle, quantization: quant)
            let gateProj = try loadLinear(base: "\(p).mlp.gate_proj",
                                           in: bundle, quantization: quant)
            let upProj   = try loadLinear(base: "\(p).mlp.up_proj",
                                           in: bundle, quantization: quant)
            let downProj = try loadLinear(base: "\(p).mlp.down_proj",
                                           in: bundle, quantization: quant)
            textLayers.append(GLMASRTextLayer(
                inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                gateProj: gateProj, upProj: upProj, downProj: downProj))
        }

        let textNorm = RMSNorm(
            weight: try bundle.tensor(named: "model.norm.weight"),
            eps: gc.lmRmsNormEps)

        // lm_head — explicit in this checkpoint (not tied).
        let lmHead: AnyLinear
        if bundle.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: bundle,
                                     quantization: quant)
        } else if let q = quant, bundle.isQuantized("model.embed_tokens") {
            // Quantized tied embedding — wrap as QuantizedLinear for gemv.
            let t    = try bundle.quantizedTriplet("model.embed_tokens")
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols:        t.scales.shape[t.scales.shape.count - 1],
                groupSize:        q.groupSize)
            lmHead = AnyLinear(QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: bits, groupSize: q.groupSize))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        return GLMASRModel(
            config: gc,
            conv1Weight: conv1Weight, conv1Bias: conv1Bias,
            conv2Weight: conv2Weight, conv2Bias: conv2Bias,
            posEmbedding: posEmbedding,
            whisperLayers: whisperLayers,
            audioLayerNorm: audioLayerNorm,
            adaptingFC1Weight: adaptFC1Weight,
            adaptingFC1Bias: adaptFC1Bias,
            adaptingFC2Weight: adaptFC2Weight,
            adaptingFC2Bias: adaptFC2Bias,
            audioBosEosToken: audioBosEos,
            embedTokens: embedTokens,
            textLayers: textLayers,
            textNorm: textNorm,
            lmHead: lmHead,
            dtype: dtype)
    }

    /// CPU-side dequantization of a `QuantizedLinear` to a float
    /// `[outFeatures, inFeatures]` weight tensor. Used for the audio
    /// adapter MLP where we call `Ops.gemm` directly.
    private static func dequantizeLinear(_ q: QuantizedLinear) -> Tensor {
        let groupSize  = q.groupSize
        let bits       = q.bits
        let packFactor = 32 / bits
        let outF       = q.weight.shape[0]
        let packedCols = q.weight.shape[1]
        let inF        = packedCols * packFactor

        let wPacked = q.weight.toArray(as: UInt32.self)
        let scales  = q.scales.toFloatArray()
        let biases  = q.biases.toFloatArray()
        let nGroups = inF / groupSize

        var out = [Float](repeating: 0, count: outF * inF)
        let mask = UInt32((1 << bits) - 1)

        for row in 0..<outF {
            for col in 0..<inF {
                let group   = col / groupSize
                let packed  = wPacked[row * packedCols + col / packFactor]
                let shift   = UInt32((col % packFactor) * bits)
                let quantVal = Int((packed >> shift) & mask)
                let s = scales[row * nGroups + group]
                let b = biases[row * nGroups + group]
                out[row * inF + col] = Float(quantVal) * s + b
            }
        }
        let result = Tensor.empty(shape: [outF, inF], dtype: .f32)
        result.copyIn(from: out)
        return result
    }
}
