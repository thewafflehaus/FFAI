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
// CohereTranscribe — Cohere's speech-to-text family.
//
// HF repo: `mlx-community/c4ai-aya-expanse-transcribe-mlx`
//          (or any `cohere_transcribe` checkpoint)
//
// Architecture:
//   waveform  ──pre-emphasis──▶
//             ──STFT (nFFT=512, hop=160, win=400)──▶
//             ──power mel (128 bins, Slaney norm)──▶ [1, nMels, T]
//             ──ConvSubsampling (5 Conv2d layers, stride×8)──▶ [T', dModel]
//             ──RelPositionalEncoding──▶
//             ──N × ConformerLayer──▶ [T', dModel]
//             ──optional bridge_proj──▶ [T', decoderHidden]
//             ──TransformerDecoder (cross-attention, KV cache)──▶ transcript
//
// The entire encoder (ConvSubsampling + Conformer) runs CPU-side (same
// strategy as FireRedASR2). The decoder runs CPU-side with causal self-
// attention and cross-attention to the encoder output. CPU attention is
// parallelised across heads via DispatchQueue.concurrentPerform.
//
// ConvSubsampling layout:
//   conv0: Conv2d(in=1, out=convCh, k=3, s=2, pad=1) — standard
//   conv2: Conv2d(groups=convCh) depthwise 3×3, s=2, pad=1
//   conv3: Conv2d(1×1, pointwise)
//   conv5: Conv2d(groups=convCh) depthwise 3×3, s=2, pad=1
//   conv6: Conv2d(1×1, pointwise)
//   out:   Linear → dModel
//   (Indices 1, 4 are ReLU activations — no weights)
//
// Weight key normalisations applied on load (mirror mlx-audio-swift):
//   encoder.pre_encode.*  → encoder.subsampling.*
//   encoder_decoder_proj. → bridge_proj.
//   log_softmax.mlp.layer0.* → lm_head.*
//   transf_decoder.{embedding,_embedding}. → decoder.embedding.*
//   transf_decoder.{decoder,_decoder}. → decoder.core.*
//   QKV: split linear_{q,k,v} / {query,key,value}_net → merged qkv_proj
//   conv.{subsampling.conv.0,2,3,5,6} → conv.{conv0,conv2,conv3,conv5,conv6}
//
// Detection: `model_type == "cohere_transcribe"` or architecture
//   `"CohereTranscribeForConditionalGeneration"`.

import Foundation
import Metal
import Tokenizers

// ─── Configuration ────────────────────────────────────────────────────

/// CohereTranscribe audio encoder (Conformer) hyper-parameters.
public struct CohereTranscribeEncoderConfig: Sendable {
    /// Encoder model dimension.
    public let dModel: Int
    /// Feed-forward expansion factor (FFN hidden = dModel * ffExpansionFactor).
    public let ffExpansionFactor: Int
    /// Number of attention heads.
    public let nHeads: Int
    /// Depthwise conv kernel size in ConformerConvolution.
    public let convKernelSize: Int
    /// Number of Conformer encoder layers.
    public let nLayers: Int
    /// Max length for the relative positional encoding table.
    public let posEmbMaxLen: Int
    /// Conv channels in the ConvSubsampling stack.
    public let subsamplingConvChannels: Int
    /// Subsampling factor (total time reduction — product of Conv2d strides).
    public let subsamplingFactor: Int
    /// Input Mel bin count.
    public let featIn: Int

    public init(
        dModel: Int = 512,
        ffExpansionFactor: Int = 4,
        nHeads: Int = 8,
        convKernelSize: Int = 31,
        nLayers: Int = 18,
        posEmbMaxLen: Int = 5000,
        subsamplingConvChannels: Int = 256,
        subsamplingFactor: Int = 8,
        featIn: Int = 128
    ) {
        self.dModel = dModel
        self.ffExpansionFactor = ffExpansionFactor
        self.nHeads = nHeads
        self.convKernelSize = convKernelSize
        self.nLayers = nLayers
        self.posEmbMaxLen = posEmbMaxLen
        self.subsamplingConvChannels = subsamplingConvChannels
        self.subsamplingFactor = subsamplingFactor
        self.featIn = featIn
    }
}

/// CohereTranscribe text decoder (Transformer AED) hyper-parameters.
public struct CohereTranscribeDecoderConfig: Sendable {
    /// Decoder hidden dimension.
    public let hiddenSize: Int
    /// FFN inner dimension.
    public let innerSize: Int
    /// Number of attention heads.
    public let numAttentionHeads: Int
    /// Number of decoder layers.
    public let numLayers: Int
    /// Maximum sequence length for the fixed positional encoding.
    public let maxSequenceLength: Int

    public init(
        hiddenSize: Int = 512,
        innerSize: Int = 2048,
        numAttentionHeads: Int = 8,
        numLayers: Int = 6,
        maxSequenceLength: Int = 512
    ) {
        self.hiddenSize = hiddenSize
        self.innerSize = innerSize
        self.numAttentionHeads = numAttentionHeads
        self.numLayers = numLayers
        self.maxSequenceLength = maxSequenceLength
    }
}

/// Top-level CohereTranscribe configuration decoded from `config.json`.
public struct CohereTranscribeConfig: Sendable {
    public let vocabSize: Int
    /// Audio sample rate (typically 16000).
    public let sampleRate: Int
    /// Maximum audio clip duration in seconds.
    public let maxAudioClipS: Int
    public let encoder: CohereTranscribeEncoderConfig
    public let decoder: CohereTranscribeDecoderConfig

    public init(
        vocabSize: Int = 32000,
        sampleRate: Int = 16_000,
        maxAudioClipS: Int = 60,
        encoder: CohereTranscribeEncoderConfig = CohereTranscribeEncoderConfig(),
        decoder: CohereTranscribeDecoderConfig = CohereTranscribeDecoderConfig()
    ) {
        self.vocabSize = vocabSize
        self.sampleRate = sampleRate
        self.maxAudioClipS = maxAudioClipS
        self.encoder = encoder
        self.decoder = decoder
    }

    /// Decode from a `ModelConfig`. Returns nil when the config doesn't
    /// describe a CohereTranscribe checkpoint.
    public static func from(_ raw: ModelConfig) -> CohereTranscribeConfig? {
        guard CohereTranscribeModel.handles(raw) else { return nil }

        // Encoder sub-config lives under "encoder" key.
        let encRaw = raw.nested("encoder")
        func ei(_ k: String) -> Int? {
            if let v = encRaw?[k] as? Int { return v }
            if let v = encRaw?[k] as? Double { return Int(v) }
            return nil
        }

        // Decoder sub-config lives under "transf_decoder" → "config_dict".
        let decOuter = raw.nested("transf_decoder")
        let decRaw = decOuter?["config_dict"] as? [String: Any]
        func di(_ k: String) -> Int? {
            if let v = decRaw?[k] as? Int { return v }
            if let v = decRaw?[k] as? Double { return Int(v) }
            return nil
        }

        let enc = CohereTranscribeEncoderConfig(
            dModel: ei("d_model") ?? 512,
            ffExpansionFactor: ei("ff_expansion_factor") ?? 4,
            nHeads: ei("n_heads") ?? 8,
            convKernelSize: ei("conv_kernel_size") ?? 31,
            nLayers: ei("n_layers") ?? 18,
            posEmbMaxLen: ei("pos_emb_max_len") ?? 5000,
            subsamplingConvChannels: ei("subsampling_conv_channels") ?? 256,
            subsamplingFactor: ei("subsampling_factor") ?? 8,
            featIn: ei("feat_in") ?? 128
        )
        let dec = CohereTranscribeDecoderConfig(
            hiddenSize: di("hidden_size") ?? 512,
            innerSize: di("inner_size") ?? 2048,
            numAttentionHeads: di("num_attention_heads") ?? 8,
            numLayers: di("num_layers") ?? 6,
            maxSequenceLength: di("max_sequence_length") ?? 512
        )

        return CohereTranscribeConfig(
            vocabSize: raw.int("vocab_size") ?? 32000,
            sampleRate: raw.int("sample_rate") ?? 16_000,
            maxAudioClipS: raw.int("max_audio_clip_s") ?? 60,
            encoder: enc,
            decoder: dec
        )
    }
}

// ─── Weight holders ───────────────────────────────────────────────────

/// One Conformer encoder feed-forward sub-block weight holder.
final class CohereConformerFFN: Module {
    let linear1: Linear
    let linear2: Linear

    init(linear1: Linear, linear2: Linear) {
        self.linear1 = linear1
        self.linear2 = linear2
    }
    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in linear1.parameters() { out.append(("linear1.\(k)", v)) }
        for (k, v) in linear2.parameters() { out.append(("linear2.\(k)", v)) }
        return out
    }
}

/// One Conformer encoder self-attention weight holder (rel-pos MHA).
final class CohereConformerAttn: Module {
    /// Merged QKV weight: [3 * dModel, dModel] (Q then K then V stacked).
    let qkvProj: Linear
    let posProj: Linear  // bias=false
    let outProj: Linear
    let nHeads: Int
    let dK: Int
    /// Positional bias vectors [nHeads, dK] stored as flat [nHeads * dK].
    let posBiasU: [Float]  // bias_u
    let posBiasV: [Float]  // bias_v

    init(
        qkvProj: Linear, posProj: Linear, outProj: Linear,
        nHeads: Int, dK: Int, posBiasU: [Float], posBiasV: [Float]
    ) {
        self.qkvProj = qkvProj
        self.posProj = posProj
        self.outProj = outProj
        self.nHeads = nHeads
        self.dK = dK
        self.posBiasU = posBiasU
        self.posBiasV = posBiasV
    }
    public func parameters() -> [(String, Tensor)] { [] }
}

/// One Conformer encoder convolution sub-block weight holder.
/// Layout after load: pointwiseConv1 → GLU → depthwiseConv → batchNorm → SiLU → pointwiseConv2.
final class CohereConformerConv: Module {
    let pointwiseConv1: Linear  // dModel → dModel * 2 (GLU)
    /// Depthwise weights: [dModel, kernelSize] row-major (stripped from [dModel, kW, 1]).
    let depthwiseWeights: [Float]
    let kernelSize: Int
    let batchNormWeight: [Float]  // gamma
    let batchNormBias: [Float]  // beta
    let batchNormRunningMean: [Float]
    let batchNormRunningVar: [Float]
    let batchNormEps: Float
    let pointwiseConv2: Linear  // dModel → dModel

    init(
        pointwiseConv1: Linear,
        depthwiseWeights: [Float], kernelSize: Int,
        batchNormWeight: [Float], batchNormBias: [Float],
        batchNormRunningMean: [Float], batchNormRunningVar: [Float],
        batchNormEps: Float,
        pointwiseConv2: Linear
    ) {
        self.pointwiseConv1 = pointwiseConv1
        self.depthwiseWeights = depthwiseWeights
        self.kernelSize = kernelSize
        self.batchNormWeight = batchNormWeight
        self.batchNormBias = batchNormBias
        self.batchNormRunningMean = batchNormRunningMean
        self.batchNormRunningVar = batchNormRunningVar
        self.batchNormEps = batchNormEps
        self.pointwiseConv2 = pointwiseConv2
    }
    public func parameters() -> [(String, Tensor)] { [] }
}

/// One Conformer encoder block weight holder.
final class CohereConformerLayer: Module {
    let normFF1: LayerNorm
    let ff1: CohereConformerFFN
    let normSelfAttn: LayerNorm
    let selfAttn: CohereConformerAttn
    let normConv: LayerNorm
    let conv: CohereConformerConv
    let normFF2: LayerNorm
    let ff2: CohereConformerFFN
    let normOut: LayerNorm

    init(
        normFF1: LayerNorm, ff1: CohereConformerFFN,
        normSelfAttn: LayerNorm, selfAttn: CohereConformerAttn,
        normConv: LayerNorm, conv: CohereConformerConv,
        normFF2: LayerNorm, ff2: CohereConformerFFN,
        normOut: LayerNorm
    ) {
        self.normFF1 = normFF1
        self.ff1 = ff1
        self.normSelfAttn = normSelfAttn
        self.selfAttn = selfAttn
        self.normConv = normConv
        self.conv = conv
        self.normFF2 = normFF2
        self.ff2 = ff2
        self.normOut = normOut
    }
    public func parameters() -> [(String, Tensor)] { [] }
}

/// One Transformer decoder self-attention / cross-attention weight holder.
final class CohereDecoderAttn: Module {
    let qkvProj: Linear  // hiddenSize → 3 * hiddenSize
    let outProj: Linear
    let nHeads: Int
    let headDim: Int
    let scale: Float

    init(qkvProj: Linear, outProj: Linear, nHeads: Int, headDim: Int) {
        self.qkvProj = qkvProj
        self.outProj = outProj
        self.nHeads = nHeads
        self.headDim = headDim
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
    }
    public func parameters() -> [(String, Tensor)] { [] }
}

/// One Transformer decoder FFN weight holder.
final class CohereDecoderFFN: Module {
    let denseIn: Linear  // hiddenSize → innerSize
    let denseOut: Linear  // innerSize → hiddenSize

    init(denseIn: Linear, denseOut: Linear) {
        self.denseIn = denseIn
        self.denseOut = denseOut
    }
    public func parameters() -> [(String, Tensor)] { [] }
}

/// One Transformer decoder layer weight holder.
final class CohereDecoderLayer: Module {
    let norm1: LayerNorm
    let selfAttn: CohereDecoderAttn
    let norm2: LayerNorm
    let crossAttn: CohereDecoderAttn
    let norm3: LayerNorm
    let ffn: CohereDecoderFFN

    init(
        norm1: LayerNorm, selfAttn: CohereDecoderAttn,
        norm2: LayerNorm, crossAttn: CohereDecoderAttn,
        norm3: LayerNorm, ffn: CohereDecoderFFN
    ) {
        self.norm1 = norm1
        self.selfAttn = selfAttn
        self.norm2 = norm2
        self.crossAttn = crossAttn
        self.norm3 = norm3
        self.ffn = ffn
    }
    public func parameters() -> [(String, Tensor)] { [] }
}

// ─── Main model ───────────────────────────────────────────────────────

/// A loaded CohereTranscribe speech-to-text model.
///
/// Main entry points:
/// - `encodeAudio(waveform:device:)` — Conformer encoder stack
/// - `transcribe(waveform:language:maxTokens:device:)` — end-to-end STT
public final class CohereTranscribeModel: @unchecked Sendable {
    public let config: CohereTranscribeConfig

    // ── ConvSubsampling weights ──────────────────────────────────────
    /// conv0 weights [convCh, 1, 3, 3] standard Conv2d, stride 2, pad 1.
    let conv0Weight: [Float]
    let conv0Bias: [Float]
    let conv0OutCh: Int  // = subsamplingConvChannels
    /// conv2 depthwise [convCh, 3, 3] weights (group conv, groups=convCh).
    let conv2DwWeight: [Float]
    /// conv3 pointwise [convCh, convCh, 1, 1].
    let conv3Weight: [Float]
    let conv3Bias: [Float]
    /// conv5 depthwise [convCh, 3, 3].
    let conv5DwWeight: [Float]
    /// conv6 pointwise [convCh, convCh, 1, 1].
    let conv6Weight: [Float]
    let conv6Bias: [Float]
    /// out: Linear [dModel, convCh * (featIn / subsamplingFactor)].
    let subsamplingOut: Linear

    // ── Relative positional encoding table ──────────────────────────
    /// Pre-computed [2 * posEmbMaxLen - 1, dModel] sinusoidal table.
    let relPETable: [Float]

    // ── Conformer encoder layers ─────────────────────────────────────
    let encoderLayers: [CohereConformerLayer]

    // ── Bridge projection (optional) ────────────────────────────────
    /// `nil` when encoderDModel == decoderHiddenSize.
    let bridgeProj: Linear?

    // ── Decoder ──────────────────────────────────────────────────────
    /// Token embedding table [vocabSize, hiddenSize].
    let tokenEmbedding: Tensor
    /// Fixed positional encoding table [maxSequenceLength, hiddenSize].
    let decoderPETable: [Float]
    let decoderNormEmb: LayerNorm
    let decoderLayers: [CohereDecoderLayer]
    let decoderFinalNorm: LayerNorm
    /// lm_head: [vocabSize, hiddenSize].
    let lmHeadWeight: Tensor

    let dtype: DType

    /// Tokenizer — nil until `load(directory:)` wires it in.
    var tokenizer: CohereTranscribeTokenizer?

    init(
        config: CohereTranscribeConfig,
        conv0Weight: [Float], conv0Bias: [Float], conv0OutCh: Int,
        conv2DwWeight: [Float],
        conv3Weight: [Float], conv3Bias: [Float],
        conv5DwWeight: [Float],
        conv6Weight: [Float], conv6Bias: [Float],
        subsamplingOut: Linear,
        relPETable: [Float],
        encoderLayers: [CohereConformerLayer],
        bridgeProj: Linear?,
        tokenEmbedding: Tensor,
        decoderPETable: [Float],
        decoderNormEmb: LayerNorm,
        decoderLayers: [CohereDecoderLayer],
        decoderFinalNorm: LayerNorm,
        lmHeadWeight: Tensor,
        dtype: DType
    ) {
        self.config = config
        self.conv0Weight = conv0Weight
        self.conv0Bias = conv0Bias
        self.conv0OutCh = conv0OutCh
        self.conv2DwWeight = conv2DwWeight
        self.conv3Weight = conv3Weight
        self.conv3Bias = conv3Bias
        self.conv5DwWeight = conv5DwWeight
        self.conv6Weight = conv6Weight
        self.conv6Bias = conv6Bias
        self.subsamplingOut = subsamplingOut
        self.relPETable = relPETable
        self.encoderLayers = encoderLayers
        self.bridgeProj = bridgeProj
        self.tokenEmbedding = tokenEmbedding
        self.decoderPETable = decoderPETable
        self.decoderNormEmb = decoderNormEmb
        self.decoderLayers = decoderLayers
        self.decoderFinalNorm = decoderFinalNorm
        self.lmHeadWeight = lmHeadWeight
        self.dtype = dtype
    }

    // ─── Audio front-end ─────────────────────────────────────────────

    /// Compute a Slaney-normalised power-mel spectrogram + mean/var norm.
    ///
    /// Reference: `CohereTranscribeAudio.computeFeatures` from mlx-audio-swift.
    ///   1. Pre-emphasis (factor 0.97).
    ///   2. STFT (nFFT=512, win=400, hop=160, centred Hann, constant-pad).
    ///   3. Power spectrum → Mel (Slaney norm, fMin=0, fMax=Nyquist).
    ///   4. Log (offset 2^-24 to avoid log(0)).
    ///   5. Mean/std normalisation per clip.
    ///
    /// Returns a flat `[nFrames * featIn]` float array in [nMel, T] scan order.
    private func computeMelFeatures(waveform: [Float]) -> (melFlat: [Float], T: Int) {
        let ec = config.encoder
        let sr = config.sampleRate
        let featIn = ec.featIn
        let nFFT = 512
        let winLen = 400
        let hop = 160

        // ── Pre-emphasis ──
        var pcm = waveform
        if pcm.count > 1 {
            var emp = [Float](repeating: 0, count: pcm.count)
            emp[0] = pcm[0]
            for i in 1 ..< pcm.count {
                emp[i] = pcm[i] - 0.97 * pcm[i - 1]
            }
            pcm = emp
        }

        // ── Centred Hann window (length nFFT, with win_length padding) ──
        // Win is centred: left pad = (nFFT - winLen) / 2 zeros, then Hann, then right pad.
        let leftPad = (nFFT - winLen) / 2
        var window = [Float](repeating: 0, count: nFFT)
        for i in 0 ..< winLen {
            let w = Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(i) / Double(winLen)))
            window[leftPad + i] = w
        }

        // ── Constant-pad the signal by nFFT/2 on each side ──
        let pad = nFFT / 2
        var padded = [Float](repeating: 0, count: pcm.count + 2 * pad)
        for i in 0 ..< pcm.count { padded[pad + i] = pcm[i] }

        // ── STFT → power spectrum ──
        let nFrames: Int
        if padded.count < nFFT {
            nFrames = 0
        } else {
            nFrames = (padded.count - nFFT) / hop + 1
        }
        guard nFrames > 0 else { return ([], 0) }

        let nFreq = nFFT / 2 + 1
        var power = [Float](repeating: 0, count: nFrames * nFreq)
        // CPU STFT over frames.
        for f in 0 ..< nFrames {
            let start = f * hop
            // DFT over nFFT points with the window applied.
            for k in 0 ..< nFreq {
                let angle = -2.0 * Float.pi * Float(k) / Float(nFFT)
                var re: Float = 0
                var im: Float = 0
                for n in 0 ..< nFFT {
                    let x = padded[start + n] * window[n]
                    re += x * cos(angle * Float(n))
                    im += x * sin(angle * Float(n))
                }
                power[f * nFreq + k] = re * re + im * im
            }
        }

        // ── Mel filterbank (Slaney normalisation) ──
        // fMin=0, fMax=sr/2, norm="slaney", mel_scale="slaney"
        let fMaxHz = Double(sr) / 2.0
        // Slaney linear/log breakpoints.
        let fMin = 0.0
        let fBreak = 1000.0
        let logStep = log(6.4) / 27.0
        func hzToSlaney(_ hz: Double) -> Double {
            if hz < fBreak {
                return hz / 66.667
            } else {
                return 15.0 + log(hz / fBreak) / logStep
            }
        }
        func slaneyToHz(_ mel: Double) -> Double {
            if mel < 15.0 {
                return mel * 66.667
            } else {
                return fBreak * exp((mel - 15.0) * logStep)
            }
        }

        let melLo = hzToSlaney(fMin)
        let melHi = hzToSlaney(fMaxHz)
        // nMels+2 edge points.
        var edges = [Double](repeating: 0, count: featIn + 2)
        for i in 0 ..< (featIn + 2) {
            edges[i] = slaneyToHz(melLo + (melHi - melLo) * Double(i) / Double(featIn + 1))
        }
        // FFT bin centre frequencies.
        var fftFreqs = [Double](repeating: 0, count: nFreq)
        for k in 0 ..< nFreq { fftFreqs[k] = Double(k) * Double(sr) / Double(nFFT) }

        // [featIn, nFreq] filterbank with Slaney normalisation.
        var melFB = [Float](repeating: 0, count: featIn * nFreq)
        for m in 0 ..< featIn {
            let lo = edges[m]
            let ctr = edges[m + 1]
            let hi = edges[m + 2]
            // Slaney normalisation: enorm = 2 / (hi_hz - lo_hz).
            let enorm = 2.0 / (hi - lo)
            for k in 0 ..< nFreq {
                let f = fftFreqs[k]
                let lower = (f - lo) / max(ctr - lo, 1e-9)
                let upper = (hi - f) / max(hi - ctr, 1e-9)
                let tri = max(0.0, min(lower, upper))
                melFB[m * nFreq + k] = Float(tri * enorm)
            }
        }

        // ── Apply filterbank → log (offset 2^-24) ──
        // Output layout: [nFrames, featIn] (time-major).
        var mel = [Float](repeating: 0, count: nFrames * featIn)
        let logFloor = Float(pow(2.0, -24.0))
        for f in 0 ..< nFrames {
            for m in 0 ..< featIn {
                var acc: Float = 0
                let pBase = f * nFreq
                let fbBase = m * nFreq
                for k in 0 ..< nFreq { acc += power[pBase + k] * melFB[fbBase + k] }
                mel[f * featIn + m] = log(acc + logFloor)
            }
        }

        // ── Mean / variance normalisation per clip ──
        var mean: Float = 0
        for v in mel { mean += v }
        mean /= Float(mel.count)
        var variance: Float = 0
        for v in mel { variance += (v - mean) * (v - mean) }
        variance /= Float(mel.count)
        let invStd = 1.0 / sqrt(variance + 1e-5)
        for i in 0 ..< mel.count { mel[i] = (mel[i] - mean) * invStd }

        // ── Convert to [featIn, nFrames] (channel-first) for Conv2d input ──
        var melCF = [Float](repeating: 0, count: featIn * nFrames)
        for m in 0 ..< featIn {
            for f in 0 ..< nFrames {
                melCF[m * nFrames + f] = mel[f * featIn + m]
            }
        }
        return (melCF, nFrames)
    }

    // ─── ConvSubsampling (CPU) ────────────────────────────────────────

    /// CPU Conv2d (standard, no groups). Input: [H, W], weight: [outCh, inCh, kH, kW].
    /// Returns [outCh, outH, outW].
    private func cpuConv2dMultiCh(
        input: [Float], inCh: Int, inH: Int, inW: Int,
        weight: [Float], bias: [Float],
        outCh: Int, kH: Int, kW: Int,
        strideH: Int, strideW: Int, padH: Int, padW: Int
    ) -> (out: [Float], outH: Int, outW: Int) {
        let outH = (inH + 2 * padH - kH) / strideH + 1
        let outW = (inW + 2 * padW - kW) / strideW + 1
        var out = [Float](repeating: 0, count: outCh * outH * outW)
        for oc in 0 ..< outCh {
            let b = bias[oc]
            for oh in 0 ..< outH {
                for ow in 0 ..< outW {
                    var acc = b
                    for ic in 0 ..< inCh {
                        let wBase = ((oc * inCh + ic) * kH)
                        for kh in 0 ..< kH {
                            let ih = oh * strideH + kh - padH
                            if ih < 0 || ih >= inH { continue }
                            for kw in 0 ..< kW {
                                let iw = ow * strideW + kw - padW
                                if iw < 0 || iw >= inW { continue }
                                let wIdx = (wBase + kh) * kW + kw
                                let inIdx = ic * inH * inW + ih * inW + iw
                                acc += weight[wIdx] * input[inIdx]
                            }
                        }
                    }
                    out[oc * outH * outW + oh * outW + ow] = acc
                }
            }
        }
        return (out, outH, outW)
    }

    /// CPU depthwise 2D convolution. Each output channel convolves only its
    /// input channel (groups = nCh). weight: [nCh, kH, kW].
    private func cpuDepthwiseConv2d(
        input: [Float], nCh: Int, inH: Int, inW: Int,
        weight: [Float], kH: Int, kW: Int,
        strideH: Int, strideW: Int, padH: Int, padW: Int
    ) -> (out: [Float], outH: Int, outW: Int) {
        let outH = (inH + 2 * padH - kH) / strideH + 1
        let outW = (inW + 2 * padW - kW) / strideW + 1
        var out = [Float](repeating: 0, count: nCh * outH * outW)
        for ch in 0 ..< nCh {
            let wBase = ch * kH * kW
            for oh in 0 ..< outH {
                for ow in 0 ..< outW {
                    var acc: Float = 0
                    for kh in 0 ..< kH {
                        let ih = oh * strideH + kh - padH
                        if ih < 0 || ih >= inH { continue }
                        for kw in 0 ..< kW {
                            let iw = ow * strideW + kw - padW
                            if iw < 0 || iw >= inW { continue }
                            acc +=
                                weight[wBase + kh * kW + kw]
                                * input[ch * inH * inW + ih * inW + iw]
                        }
                    }
                    out[ch * outH * outW + oh * outW + ow] = acc
                }
            }
        }
        return (out, outH, outW)
    }

    /// CPU ReLU in-place.
    private func reluInPlace(_ x: inout [Float]) {
        for i in 0 ..< x.count { if x[i] < 0 { x[i] = 0 } }
    }

    /// Apply the 5-layer ConvSubsampling stack to `melCF` ([featIn, T]).
    /// Returns `[T', dModel]` as flat [T' × dModel].
    private func convSubsampling(
        melCF: [Float], featIn: Int, T: Int
    ) -> (out: [Float], outT: Int) {
        let convCh = conv0OutCh

        // ── conv0: standard Conv2d, in=1, out=convCh, k=3×3, s=2, pad=1 ──
        // Input shape: [1, featIn, T].
        var (h, hH, hW) = cpuConv2dMultiCh(
            input: melCF, inCh: 1, inH: featIn, inW: T,
            weight: conv0Weight, bias: conv0Bias,
            outCh: convCh, kH: 3, kW: 3,
            strideH: 2, strideW: 2, padH: 1, padW: 1)
        // ReLU after conv0.
        reluInPlace(&h)

        // ── conv2: depthwise 3×3, s=2, pad=1 ──
        let (h2, h2H, h2W) = cpuDepthwiseConv2d(
            input: h, nCh: convCh, inH: hH, inW: hW,
            weight: conv2DwWeight, kH: 3, kW: 3,
            strideH: 2, strideW: 2, padH: 1, padW: 1)

        // ── conv3: pointwise 1×1, standard Conv2d(convCh→convCh, k=1) ──
        var (h3, h3H, h3W) = cpuConv2dMultiCh(
            input: h2, inCh: convCh, inH: h2H, inW: h2W,
            weight: conv3Weight, bias: conv3Bias,
            outCh: convCh, kH: 1, kW: 1,
            strideH: 1, strideW: 1, padH: 0, padW: 0)
        reluInPlace(&h3)

        // ── conv5: depthwise 3×3, s=2, pad=1 ──
        let (h5, h5H, h5W) = cpuDepthwiseConv2d(
            input: h3, nCh: convCh, inH: h3H, inW: h3W,
            weight: conv5DwWeight, kH: 3, kW: 3,
            strideH: 2, strideW: 2, padH: 1, padW: 1)

        // ── conv6: pointwise 1×1 ──
        var (h6, h6H, h6W) = cpuConv2dMultiCh(
            input: h5, inCh: convCh, inH: h5H, inW: h5W,
            weight: conv6Weight, bias: conv6Bias,
            outCh: convCh, kH: 1, kW: 1,
            strideH: 1, strideW: 1, padH: 0, padW: 0)
        reluInPlace(&h6)

        // ── Reshape → linear ──
        // After subsampling: [convCh, h6H, h6W]. We want [h6W, convCh * h6H].
        // (time=width, freq=height in the NCHW layout).
        let outT = h6W
        let flatFreq = convCh * h6H
        var reshaped = [Float](repeating: 0, count: outT * flatFreq)
        for t in 0 ..< outT {
            for ch in 0 ..< convCh {
                for fq in 0 ..< h6H {
                    reshaped[t * flatFreq + ch * h6H + fq] =
                        h6[ch * h6H * h6W + fq * h6W + t]
                }
            }
        }

        // Linear: [outT, flatFreq] → [outT, dModel].
        let dModel = config.encoder.dModel
        let wVals = subsamplingOut.weight.toFloatArray()  // [dModel, flatFreq]
        let bVals = subsamplingOut.bias?.toFloatArray()
        var linear = [Float](repeating: 0, count: outT * dModel)
        for t in 0 ..< outT {
            for d in 0 ..< dModel {
                var acc: Float = bVals?[d] ?? 0
                let wBase = d * flatFreq
                let xBase = t * flatFreq
                for f in 0 ..< flatFreq { acc += wVals[wBase + f] * reshaped[xBase + f] }
                linear[t * dModel + d] = acc
            }
        }
        return (linear, outT)
    }

    // ─── Relative positional encoding slice ──────────────────────────

    /// Extract the centred `[2T-1, dModel]` slice from the pre-computed
    /// `[2*posEmbMaxLen-1, dModel]` rel-pos table for sequence length T.
    private func relPosSlice(T: Int, dModel: Int) -> [Float] {
        let totalLen = 2 * config.encoder.posEmbMaxLen - 1
        let peSliceLen = 2 * T - 1
        let peSliceStart = (totalLen / 2) - T + 1
        var posEmb = [Float](repeating: 0, count: peSliceLen * dModel)
        for i in 0 ..< peSliceLen {
            let srcRow = peSliceStart + i
            let srcBase = srcRow * dModel
            let dstBase = i * dModel
            for d in 0 ..< dModel { posEmb[dstBase + d] = relPETable[srcBase + d] }
        }
        return posEmb
    }

    // ─── Conformer encoder layer (CPU) ────────────────────────────────

    /// Run one Conformer block over flat `[T × dModel]`.
    private func runConformerLayer(
        _ layer: CohereConformerLayer,
        seq seqIn: [Float], posEmb: [Float],
        T: Int, dModel: Int
    ) -> [Float] {
        // 1. FF1: residual + 0.5 * dropout(silu(linear1(norm(x))))
        var h = applyConformerFFN(
            layer.ff1, norm: layer.normFF1,
            seq: seqIn, T: T, dModel: dModel)
        for i in 0 ..< seqIn.count { h[i] = seqIn[i] + 0.5 * (h[i] - seqIn[i]) }

        // 2. Self-attention with relative pos.
        h = applyRelPosMHA(
            layer.selfAttn, norm: layer.normSelfAttn,
            seq: h, posEmb: posEmb, T: T, dModel: dModel)

        // 3. Conformer convolution.
        h = applyConformerConv(
            layer.conv, norm: layer.normConv,
            seq: h, T: T, dModel: dModel)

        // 4. FF2: residual + 0.5 * silu(linear1(norm(x))) → linear2.
        let ff2Out = applyConformerFFN(
            layer.ff2, norm: layer.normFF2,
            seq: h, T: T, dModel: dModel)
        for i in 0 ..< h.count { h[i] = h[i] + 0.5 * (ff2Out[i] - h[i]) }

        // 5. Final norm.
        h = layerNormRows(layer.normOut, rows: h, T: T, dim: dModel)
        return h
    }

    /// Feed-forward sub-block: norm → SiLU(linear1) → linear2.
    private func applyConformerFFN(
        _ ffn: CohereConformerFFN, norm: LayerNorm,
        seq seqIn: [Float], T: Int, dModel: Int
    ) -> [Float] {
        let normed = layerNormRows(norm, rows: seqIn, T: T, dim: dModel)
        let dFF = dModel * config.encoder.ffExpansionFactor
        let w1 = ffn.linear1.weight.toFloatArray()
        let b1 = ffn.linear1.bias?.toFloatArray()
        var h = [Float](repeating: 0, count: T * dFF)
        for t in 0 ..< T {
            for d in 0 ..< dFF {
                var acc: Float = b1?[d] ?? 0
                let wBase = d * dModel
                let xBase = t * dModel
                for i in 0 ..< dModel { acc += w1[wBase + i] * normed[xBase + i] }
                // SiLU activation.
                h[t * dFF + d] = acc * (1.0 / (1.0 + exp(-acc)))
            }
        }
        let w2 = ffn.linear2.weight.toFloatArray()
        let b2 = ffn.linear2.bias?.toFloatArray()
        var out = [Float](repeating: 0, count: T * dModel)
        for t in 0 ..< T {
            for d in 0 ..< dModel {
                var acc: Float = b2?[d] ?? 0
                let wBase = d * dFF
                let xBase = t * dFF
                for i in 0 ..< dFF { acc += w2[wBase + i] * h[xBase + i] }
                out[t * dModel + d] = acc
            }
        }
        return out
    }

    /// Relative-position multi-head self-attention sub-block.
    /// Matches the `RelPositionMultiHeadAttention` reference (Transformer-XL).
    private func applyRelPosMHA(
        _ attn: CohereConformerAttn, norm: LayerNorm,
        seq seqIn: [Float], posEmb: [Float],
        T: Int, dModel: Int
    ) -> [Float] {
        let nH = attn.nHeads
        let dK = attn.dK
        let stride = nH * dK
        let peLen = 2 * T - 1

        let normed = layerNormRows(norm, rows: seqIn, T: T, dim: dModel)

        // QKV from merged qkv_proj [3*dModel, dModel].
        let qkvW = attn.qkvProj.weight.toFloatArray()
        let qkvB = attn.qkvProj.bias?.toFloatArray()
        var qkv = [Float](repeating: 0, count: T * 3 * stride)
        for t in 0 ..< T {
            for d in 0 ..< (3 * stride) {
                var acc: Float = qkvB?[d] ?? 0
                let wBase = d * dModel
                let xBase = t * dModel
                for i in 0 ..< dModel { acc += qkvW[wBase + i] * normed[xBase + i] }
                qkv[t * 3 * stride + d] = acc
            }
        }
        // Split QKV.
        var qParts = [Float](repeating: 0, count: T * stride)
        var kParts = [Float](repeating: 0, count: T * stride)
        var vParts = [Float](repeating: 0, count: T * stride)
        for t in 0 ..< T {
            let base = t * 3 * stride
            for d in 0 ..< stride {
                qParts[t * stride + d] = qkv[base + d]
                kParts[t * stride + d] = qkv[base + stride + d]
                vParts[t * stride + d] = qkv[base + 2 * stride + d]
            }
        }

        // Positional projection: [peLen, dModel] → [peLen, stride].
        let posW = attn.posProj.weight.toFloatArray()
        var pProj = [Float](repeating: 0, count: peLen * stride)
        for p in 0 ..< peLen {
            for d in 0 ..< stride {
                var acc: Float = 0
                let wBase = d * dModel
                let xBase = p * dModel
                for i in 0 ..< dModel { acc += posW[wBase + i] * posEmb[xBase + i] }
                pProj[p * stride + d] = acc
            }
        }

        let scale = 1.0 / Float(sqrt(Double(dK)))

        // Parallel attention over nH * T (head, query-row) pairs.
        var attnOut = [Float](repeating: 0, count: T * stride)
        attnOut.withUnsafeMutableBufferPointer { outBuf in
            // Safety: each `concurrentPerform` head writes to a disjoint
            // `[hOff ..< hOff + dK]` slice of every output row; the pointer
            // lifetimes strictly contain the parallel work.
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            qParts.withUnsafeBufferPointer { qBuf in
                kParts.withUnsafeBufferPointer { kBuf in
                    vParts.withUnsafeBufferPointer { vBuf in
                        pProj.withUnsafeBufferPointer { pBuf in
                            attn.posBiasU.withUnsafeBufferPointer { uBuf in
                                attn.posBiasV.withUnsafeBufferPointer { vbBuf in
                                    nonisolated(unsafe) let qb = qBuf.baseAddress!
                                    nonisolated(unsafe) let kb = kBuf.baseAddress!
                                    nonisolated(unsafe) let vb = vBuf.baseAddress!
                                    nonisolated(unsafe) let pb = pBuf.baseAddress!
                                    nonisolated(unsafe) let ub = uBuf.baseAddress!
                                    nonisolated(unsafe) let vbb = vbBuf.baseAddress!
                                    DispatchQueue.concurrentPerform(iterations: nH) { h in
                                        let hOff = h * dK
                                        var scores = [Float](repeating: 0, count: T * T)
                                        // AC: (Q + biasU) · K^T
                                        for i in 0 ..< T {
                                            for j in 0 ..< T {
                                                var dot: Float = 0
                                                let qBase = i * stride + hOff
                                                let kBase = j * stride + hOff
                                                for d in 0 ..< dK {
                                                    dot +=
                                                        (qb[qBase + d] + ub[hOff + d])
                                                        * kb[kBase + d]
                                                }
                                                scores[i * T + j] = dot * scale
                                            }
                                        }
                                        // BD: (Q + biasV) · P^T, then rel-shift to [T, T].
                                        var bdRaw = [Float](repeating: 0, count: T * peLen)
                                        for i in 0 ..< T {
                                            for p in 0 ..< peLen {
                                                var dot: Float = 0
                                                let qBase = i * stride + hOff
                                                let pBase = p * stride + hOff
                                                for d in 0 ..< dK {
                                                    dot +=
                                                        (qb[qBase + d] + vbb[hOff + d])
                                                        * pb[pBase + d]
                                                }
                                                bdRaw[i * peLen + p] = dot * scale
                                            }
                                        }
                                        // Rel-shift: pick last T columns of the [T, peLen] bdRaw.
                                        let bdOff = peLen - T
                                        for i in 0 ..< T {
                                            for j in 0 ..< T {
                                                scores[i * T + j] += bdRaw[i * peLen + bdOff + j]
                                            }
                                        }
                                        // Softmax + weighted V.
                                        for i in 0 ..< T {
                                            var maxS = -Float.greatestFiniteMagnitude
                                            for j in 0 ..< T {
                                                if scores[i * T + j] > maxS {
                                                    maxS = scores[i * T + j]
                                                }
                                            }
                                            var sumE: Float = 0
                                            for j in 0 ..< T {
                                                let e = exp(scores[i * T + j] - maxS)
                                                scores[i * T + j] = e
                                                sumE += e
                                            }
                                            let inv = sumE > 0 ? 1.0 / sumE : 0
                                            let oBase = i * stride + hOff
                                            for j in 0 ..< T {
                                                let w = scores[i * T + j] * inv
                                                let vBase = j * stride + hOff
                                                for d in 0 ..< dK {
                                                    outPtr[oBase + d] += w * vb[vBase + d]
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // out_proj + residual.
        let outW = attn.outProj.weight.toFloatArray()
        let outB = attn.outProj.bias?.toFloatArray()
        var proj = [Float](repeating: 0, count: T * dModel)
        for t in 0 ..< T {
            for d in 0 ..< dModel {
                var acc: Float = outB?[d] ?? 0
                let wBase = d * stride
                let xBase = t * stride
                for i in 0 ..< stride { acc += outW[wBase + i] * attnOut[xBase + i] }
                proj[t * dModel + d] = acc
            }
        }
        var result = proj
        for i in 0 ..< seqIn.count { result[i] += seqIn[i] }
        return result
    }

    /// Conformer convolution sub-block.
    /// norm → pointwiseConv1 → GLU → depthwiseConv → BatchNorm (inference) → SiLU → pointwiseConv2 + residual.
    private func applyConformerConv(
        _ cb: CohereConformerConv, norm: LayerNorm,
        seq seqIn: [Float], T: Int, dModel: Int
    ) -> [Float] {
        let normed = layerNormRows(norm, rows: seqIn, T: T, dim: dModel)

        // pointwiseConv1: dModel → dModel * 2.
        let pw1W = cb.pointwiseConv1.weight.toFloatArray()
        let pw1B = cb.pointwiseConv1.bias?.toFloatArray()
        let gluDim = dModel * 2
        var gluIn = [Float](repeating: 0, count: T * gluDim)
        for t in 0 ..< T {
            for d in 0 ..< gluDim {
                var acc: Float = pw1B?[d] ?? 0
                let wBase = d * dModel
                let xBase = t * dModel
                for i in 0 ..< dModel { acc += pw1W[wBase + i] * normed[xBase + i] }
                gluIn[t * gluDim + d] = acc
            }
        }
        // GLU: gate = sigmoid(upper half), output = lower half * gate.
        var gated = [Float](repeating: 0, count: T * dModel)
        for t in 0 ..< T {
            let base = t * gluDim
            for d in 0 ..< dModel {
                gated[t * dModel + d] =
                    gluIn[base + d]
                    * (1.0 / (1.0 + exp(-gluIn[base + dModel + d])))
            }
        }

        // Depthwise Conv1d: [T, dModel] with kernel size convKernelSize, same padding.
        let kSize = cb.kernelSize
        let halfPad = (kSize - 1) / 2
        var dwOut = [Float](repeating: 0, count: T * dModel)
        for t in 0 ..< T {
            for ch in 0 ..< dModel {
                var acc: Float = 0
                let wBase = ch * kSize
                for k in 0 ..< kSize {
                    let st = t + k - halfPad
                    if st >= 0, st < T {
                        acc += cb.depthwiseWeights[wBase + k] * gated[st * dModel + ch]
                    }
                }
                dwOut[t * dModel + ch] = acc
            }
        }

        // BatchNorm (inference mode): (x - running_mean) / sqrt(running_var + eps) * gamma + beta.
        let bnEps = cb.batchNormEps
        for t in 0 ..< T {
            let base = t * dModel
            for d in 0 ..< dModel {
                let v = dwOut[base + d]
                let normalised =
                    (v - cb.batchNormRunningMean[d])
                    / sqrt(cb.batchNormRunningVar[d] + bnEps)
                dwOut[base + d] = normalised * cb.batchNormWeight[d] + cb.batchNormBias[d]
            }
        }

        // SiLU.
        for i in 0 ..< dwOut.count {
            let v = dwOut[i]
            dwOut[i] = v * (1.0 / (1.0 + exp(-v)))
        }

        // pointwiseConv2: dModel → dModel.
        let pw2W = cb.pointwiseConv2.weight.toFloatArray()
        let pw2B = cb.pointwiseConv2.bias?.toFloatArray()
        var pw2Out = [Float](repeating: 0, count: T * dModel)
        for t in 0 ..< T {
            for d in 0 ..< dModel {
                var acc: Float = pw2B?[d] ?? 0
                let wBase = d * dModel
                let xBase = t * dModel
                for i in 0 ..< dModel { acc += pw2W[wBase + i] * dwOut[xBase + i] }
                pw2Out[t * dModel + d] = acc
            }
        }

        var result = pw2Out
        for i in 0 ..< seqIn.count { result[i] += seqIn[i] }
        return result
    }

    // ─── CPU helper: LayerNorm over rows ─────────────────────────────

    private func layerNormRows(
        _ ln: LayerNorm, rows: [Float], T: Int, dim: Int
    ) -> [Float] {
        let wVals = ln.weight.toFloatArray()
        let bVals = ln.bias.toFloatArray()
        var out = [Float](repeating: 0, count: T * dim)
        let eps = ln.eps
        for r in 0 ..< T {
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

    // ─── Audio encoding ───────────────────────────────────────────────

    /// Encode a 16 kHz mono waveform into encoder output `[T', encDim]`
    /// (after optional bridge projection, encDim = decoderHiddenSize).
    public func encodeAudio(waveform: [Float], device: Device = .shared) -> [Float] {
        let ec = config.encoder
        let dModel = ec.dModel

        // ── 1. Audio front-end: pre-emphasis + power mel + mean/std norm ──
        let (melCF, T) = computeMelFeatures(waveform: waveform)
        guard T > 0 else { return [] }

        // ── 2. ConvSubsampling ──
        let (subOut, outT) = convSubsampling(melCF: melCF, featIn: ec.featIn, T: T)
        guard outT > 0 else { return [] }

        // ── 3. Relative positional encoding slice ──
        let posEmb = relPosSlice(T: outT, dModel: dModel)

        // ── 4. Conformer encoder layers (all CPU) ──
        var encSeq = subOut
        for layer in encoderLayers {
            encSeq = runConformerLayer(
                layer, seq: encSeq, posEmb: posEmb,
                T: outT, dModel: dModel)
        }

        // ── 5. Bridge projection (optional) ──
        if let bp = bridgeProj {
            let decH = config.decoder.hiddenSize
            let bpW = bp.weight.toFloatArray()
            let bpB = bp.bias?.toFloatArray()
            var projected = [Float](repeating: 0, count: outT * decH)
            for t in 0 ..< outT {
                for d in 0 ..< decH {
                    var acc: Float = bpB?[d] ?? 0
                    let wBase = d * dModel
                    let xBase = t * dModel
                    for i in 0 ..< dModel { acc += bpW[wBase + i] * encSeq[xBase + i] }
                    projected[t * decH + d] = acc
                }
            }
            return projected
        }
        return encSeq
    }

    // ─── Decoder forward pass ─────────────────────────────────────────

    /// Run one Transformer decoder layer.
    /// `seqIn`: flat [S × hiddenSize] decoder sequence so far.
    /// `encOut`: flat [T' × hiddenSize] encoder output (bridge-projected).
    /// `encT`: number of encoder frames T'.
    private func runDecoderLayer(
        _ layer: CohereDecoderLayer,
        seqIn: [Float], S: Int,
        encOut: [Float], encT: Int,
        hiddenSize: Int
    ) -> [Float] {
        // ── 1. Self-attention (causal) ──
        let normSelf = layerNormRows(layer.norm1, rows: seqIn, T: S, dim: hiddenSize)
        let selfOut = causalSelfAttn(
            layer.selfAttn, seq: normSelf, S: S, hiddenSize: hiddenSize)
        var h = [Float](repeating: 0, count: seqIn.count)
        for i in 0 ..< seqIn.count { h[i] = seqIn[i] + selfOut[i] }

        // ── 2. Cross-attention to encoder output ──
        let normCross = layerNormRows(layer.norm2, rows: h, T: S, dim: hiddenSize)
        let crossOut = crossAttn(
            layer.crossAttn, q: normCross, S: S,
            kv: encOut, encT: encT, hiddenSize: hiddenSize)
        var h2 = [Float](repeating: 0, count: h.count)
        for i in 0 ..< h.count { h2[i] = h[i] + crossOut[i] }

        // ── 3. FFN (ReLU activation) ──
        let normMLP = layerNormRows(layer.norm3, rows: h2, T: S, dim: hiddenSize)
        let innerSize = config.decoder.innerSize
        let w1 = layer.ffn.denseIn.weight.toFloatArray()
        let b1 = layer.ffn.denseIn.bias?.toFloatArray()
        var ffnH = [Float](repeating: 0, count: S * innerSize)
        for t in 0 ..< S {
            for d in 0 ..< innerSize {
                var acc: Float = b1?[d] ?? 0
                let wBase = d * hiddenSize
                let xBase = t * hiddenSize
                for i in 0 ..< hiddenSize { acc += w1[wBase + i] * normMLP[xBase + i] }
                ffnH[t * innerSize + d] = acc > 0 ? acc : 0  // ReLU
            }
        }
        let w2 = layer.ffn.denseOut.weight.toFloatArray()
        let b2 = layer.ffn.denseOut.bias?.toFloatArray()
        var ffnOut = [Float](repeating: 0, count: S * hiddenSize)
        for t in 0 ..< S {
            for d in 0 ..< hiddenSize {
                var acc: Float = b2?[d] ?? 0
                let wBase = d * innerSize
                let xBase = t * innerSize
                for i in 0 ..< innerSize { acc += w2[wBase + i] * ffnH[xBase + i] }
                ffnOut[t * hiddenSize + d] = acc
            }
        }
        var out = [Float](repeating: 0, count: h2.count)
        for i in 0 ..< h2.count { out[i] = h2[i] + ffnOut[i] }
        return out
    }

    /// Causal (masked) self-attention for the decoder.
    /// Returns flat [S × hiddenSize].
    private func causalSelfAttn(
        _ attn: CohereDecoderAttn,
        seq: [Float], S: Int, hiddenSize: Int
    ) -> [Float] {
        let nH = attn.nHeads
        let hd = attn.headDim
        let stride = nH * hd

        let qkvW = attn.qkvProj.weight.toFloatArray()
        let qkvB = attn.qkvProj.bias?.toFloatArray()
        // Project: [S, hiddenSize] → [S, 3 * stride].
        var qkv = [Float](repeating: 0, count: S * 3 * stride)
        for t in 0 ..< S {
            for d in 0 ..< (3 * stride) {
                var acc: Float = qkvB?[d] ?? 0
                let wBase = d * hiddenSize
                let xBase = t * hiddenSize
                for i in 0 ..< hiddenSize { acc += qkvW[wBase + i] * seq[xBase + i] }
                qkv[t * 3 * stride + d] = acc
            }
        }
        var q = [Float](repeating: 0, count: S * stride)
        var k = [Float](repeating: 0, count: S * stride)
        var v = [Float](repeating: 0, count: S * stride)
        for t in 0 ..< S {
            let base = t * 3 * stride
            for d in 0 ..< stride {
                q[t * stride + d] = qkv[base + d]
                k[t * stride + d] = qkv[base + stride + d]
                v[t * stride + d] = qkv[base + 2 * stride + d]
            }
        }

        let scale = attn.scale
        var attnOut = [Float](repeating: 0, count: S * stride)
        attnOut.withUnsafeMutableBufferPointer { outBuf in
            // Safety: each `concurrentPerform` head writes to a disjoint
            // `[hOff ..< hOff + hd]` slice of every output row.
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            q.withUnsafeBufferPointer { qb in
                k.withUnsafeBufferPointer { kb in
                    v.withUnsafeBufferPointer { vb in
                        nonisolated(unsafe) let qPtr = qb.baseAddress!
                        nonisolated(unsafe) let kPtr = kb.baseAddress!
                        nonisolated(unsafe) let vPtr = vb.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: nH) { h in
                            let hOff = h * hd
                            for i in 0 ..< S {
                                var scores = [Float](repeating: 0, count: i + 1)
                                var maxS = -Float.greatestFiniteMagnitude
                                for j in 0 ... i {
                                    var dot: Float = 0
                                    let qBase = i * stride + hOff
                                    let kBase = j * stride + hOff
                                    for d in 0 ..< hd { dot += qPtr[qBase + d] * kPtr[kBase + d] }
                                    let s = dot * scale
                                    scores[j] = s
                                    if s > maxS { maxS = s }
                                }
                                var sumE: Float = 0
                                for j in 0 ... i {
                                    let e = exp(scores[j] - maxS)
                                    scores[j] = e
                                    sumE += e
                                }
                                let inv = sumE > 0 ? 1.0 / sumE : 0
                                let oBase = i * stride + hOff
                                for j in 0 ... i {
                                    let w = scores[j] * inv
                                    let vBase = j * stride + hOff
                                    for d in 0 ..< hd { outPtr[oBase + d] += w * vPtr[vBase + d] }
                                }
                            }
                        }
                    }
                }
            }
        }

        // out_proj.
        let outW = attn.outProj.weight.toFloatArray()
        let outB = attn.outProj.bias?.toFloatArray()
        var proj = [Float](repeating: 0, count: S * hiddenSize)
        for t in 0 ..< S {
            for d in 0 ..< hiddenSize {
                var acc: Float = outB?[d] ?? 0
                let wBase = d * stride
                let xBase = t * stride
                for i in 0 ..< stride { acc += outW[wBase + i] * attnOut[xBase + i] }
                proj[t * hiddenSize + d] = acc
            }
        }
        return proj
    }

    /// Cross-attention: decoder query [S × hiddenSize] attending to encoder [encT × hiddenSize].
    private func crossAttn(
        _ attn: CohereDecoderAttn,
        q: [Float], S: Int,
        kv: [Float], encT: Int,
        hiddenSize: Int
    ) -> [Float] {
        let nH = attn.nHeads
        let hd = attn.headDim
        let stride = nH * hd

        // Q from decoder, K/V from encoder via same qkv_proj.
        let qkvW = attn.qkvProj.weight.toFloatArray()
        let qkvB = attn.qkvProj.bias?.toFloatArray()

        // Project decoder query [S, hiddenSize] → [S, 3*stride], take only Q part.
        var qPart = [Float](repeating: 0, count: S * stride)
        for t in 0 ..< S {
            for d in 0 ..< stride {
                var acc: Float = qkvB?[d] ?? 0
                let wBase = d * hiddenSize
                let xBase = t * hiddenSize
                for i in 0 ..< hiddenSize { acc += qkvW[wBase + i] * q[xBase + i] }
                qPart[t * stride + d] = acc
            }
        }
        // Project encoder KV [encT, hiddenSize] → [encT, 3*stride], take K and V.
        var kvFull = [Float](repeating: 0, count: encT * 3 * stride)
        for t in 0 ..< encT {
            for d in 0 ..< (3 * stride) {
                var acc: Float = qkvB?[d] ?? 0
                let wBase = d * hiddenSize
                let xBase = t * hiddenSize
                for i in 0 ..< hiddenSize { acc += qkvW[wBase + i] * kv[xBase + i] }
                kvFull[t * 3 * stride + d] = acc
            }
        }
        var kPart = [Float](repeating: 0, count: encT * stride)
        var vPart = [Float](repeating: 0, count: encT * stride)
        for t in 0 ..< encT {
            let base = t * 3 * stride
            for d in 0 ..< stride {
                kPart[t * stride + d] = kvFull[base + stride + d]
                vPart[t * stride + d] = kvFull[base + 2 * stride + d]
            }
        }

        let scale = attn.scale
        var attnOut = [Float](repeating: 0, count: S * stride)
        attnOut.withUnsafeMutableBufferPointer { outBuf in
            // Safety: each `concurrentPerform` (h, i) pair writes to a disjoint
            // `[i*stride + hOff ..< i*stride + hOff + hd]` slice.
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            qPart.withUnsafeBufferPointer { qb in
                kPart.withUnsafeBufferPointer { kb in
                    vPart.withUnsafeBufferPointer { vb in
                        nonisolated(unsafe) let qPtr = qb.baseAddress!
                        nonisolated(unsafe) let kPtr = kb.baseAddress!
                        nonisolated(unsafe) let vPtr = vb.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: nH * S) { work in
                            let h = work / S
                            let i = work % S
                            let hOff = h * hd
                            var scores = [Float](repeating: 0, count: encT)
                            var maxS = -Float.greatestFiniteMagnitude
                            for j in 0 ..< encT {
                                var dot: Float = 0
                                let qBase = i * stride + hOff
                                let kBase = j * stride + hOff
                                for d in 0 ..< hd { dot += qPtr[qBase + d] * kPtr[kBase + d] }
                                let s = dot * scale
                                scores[j] = s
                                if s > maxS { maxS = s }
                            }
                            var sumE: Float = 0
                            for j in 0 ..< encT {
                                let e = exp(scores[j] - maxS)
                                scores[j] = e
                                sumE += e
                            }
                            let inv = sumE > 0 ? 1.0 / sumE : 0
                            let oBase = i * stride + hOff
                            for j in 0 ..< encT {
                                let w = scores[j] * inv
                                let vBase = j * stride + hOff
                                for d in 0 ..< hd { outPtr[oBase + d] += w * vPtr[vBase + d] }
                            }
                        }
                    }
                }
            }
        }

        // out_proj.
        let outW = attn.outProj.weight.toFloatArray()
        let outB = attn.outProj.bias?.toFloatArray()
        var proj = [Float](repeating: 0, count: S * hiddenSize)
        for t in 0 ..< S {
            for d in 0 ..< hiddenSize {
                var acc: Float = outB?[d] ?? 0
                let wBase = d * stride
                let xBase = t * stride
                for i in 0 ..< stride { acc += outW[wBase + i] * attnOut[xBase + i] }
                proj[t * hiddenSize + d] = acc
            }
        }
        return proj
    }

    // ─── Transcription ────────────────────────────────────────────────

    /// Transcribe a 16 kHz mono waveform to a text string.
    ///
    /// - Parameters:
    ///   - waveform:  16 kHz mono PCM samples.
    ///   - tokenizer: Tokenizer loaded from the checkpoint directory.
    ///   - language:  BCP-47 code (default "en").
    ///   - maxTokens: Maximum generated transcript tokens (default 512).
    ///   - device:    Metal device (unused in the CPU-only path; kept for API consistency).
    /// - Returns: The decoded transcript string.
    public func transcribe(
        waveform: [Float],
        tokenizer: CohereTranscribeTokenizer,
        language: String = "en",
        maxTokens: Int = 512,
        device: Device = .shared
    ) -> String {
        let dc = config.decoder
        let hiddenSize = dc.hiddenSize

        // ── 1. Encode audio ──
        let encOut = encodeAudio(waveform: waveform, device: device)
        guard !encOut.isEmpty else { return "" }
        let encT = encOut.count / hiddenSize

        // ── 2. Build prompt token ids ──
        let promptIds = tokenizer.buildPromptTokens(language: language)
        guard !promptIds.isEmpty else { return "" }
        let eosId = tokenizer.eosId

        // ── 3. Prefill: decoder forward with all prompt tokens ──
        var seqIds = promptIds
        var seqEmb = embedAndNorm(tokenIds: seqIds, hiddenSize: hiddenSize)
        for layer in decoderLayers {
            seqEmb = runDecoderLayer(
                layer, seqIn: seqEmb, S: seqIds.count,
                encOut: encOut, encT: encT,
                hiddenSize: hiddenSize)
        }
        // Get logits from the last position.
        var lastHidden = Array(seqEmb.suffix(hiddenSize))
        let finalNormW = decoderFinalNorm.weight.toFloatArray()
        let finalNormB = decoderFinalNorm.bias.toFloatArray()
        applyLayerNorm1D(
            &lastHidden, weight: finalNormW, bias: finalNormB,
            eps: decoderFinalNorm.eps)

        // ── 4. Greedy autoregressive decode ──
        var generated: [Int] = []
        for _ in 0 ..< maxTokens {
            let logits = projectToVocab(lastHidden, hiddenSize: hiddenSize)
            // Greedy: argmax.
            var bestId = 0
            var bestVal = -Float.greatestFiniteMagnitude
            for (i, v) in logits.enumerated() where v > bestVal {
                bestVal = v
                bestId = i
            }

            if bestId == eosId { break }
            generated.append(bestId)
            seqIds.append(bestId)

            // Backstop: repetition guard (≤ 3 distinct in last 24).
            if generated.count >= 24, Set(generated.suffix(24)).count <= 3 { break }

            // Next embed + decode.
            var nextEmb = embedAndNorm(tokenIds: seqIds, hiddenSize: hiddenSize)
            for layer in decoderLayers {
                nextEmb = runDecoderLayer(
                    layer, seqIn: nextEmb, S: seqIds.count,
                    encOut: encOut, encT: encT,
                    hiddenSize: hiddenSize)
            }
            lastHidden = Array(nextEmb.suffix(hiddenSize))
            applyLayerNorm1D(
                &lastHidden, weight: finalNormW, bias: finalNormB,
                eps: decoderFinalNorm.eps)
        }

        return tokenizer.decode(tokens: generated)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // ─── Embedding + norm helpers ─────────────────────────────────────

    /// Embed a token id sequence and add fixed positional encoding, then
    /// apply the embedding LayerNorm. Returns flat [S × hiddenSize].
    private func embedAndNorm(tokenIds: [Int], hiddenSize: Int) -> [Float] {
        let S = tokenIds.count
        let vocabSize = config.vocabSize
        let embVals = tokenEmbedding.toFloatArray()  // [vocabSize, hiddenSize]
        var out = [Float](repeating: 0, count: S * hiddenSize)
        for (t, id) in tokenIds.enumerated() {
            let safe = max(0, min(id, vocabSize - 1))
            let embBase = safe * hiddenSize
            let peBase = t * hiddenSize
            let dstBase = t * hiddenSize
            for d in 0 ..< hiddenSize {
                out[dstBase + d] = embVals[embBase + d] + decoderPETable[peBase + d]
            }
        }
        // Apply embedding LayerNorm.
        let wVals = decoderNormEmb.weight.toFloatArray()
        let bVals = decoderNormEmb.bias.toFloatArray()
        let eps = decoderNormEmb.eps
        for t in 0 ..< S {
            let base = t * hiddenSize
            var mean: Float = 0
            for d in 0 ..< hiddenSize { mean += out[base + d] }
            mean /= Float(hiddenSize)
            var variance: Float = 0
            for d in 0 ..< hiddenSize {
                let diff = out[base + d] - mean
                variance += diff * diff
            }
            variance /= Float(hiddenSize)
            let invStd = 1.0 / sqrt(variance + eps)
            for d in 0 ..< hiddenSize {
                out[base + d] = (out[base + d] - mean) * invStd * wVals[d] + bVals[d]
            }
        }
        return out
    }

    /// Apply layer norm in-place to a single [dim] vector.
    private func applyLayerNorm1D(
        _ x: inout [Float], weight: [Float], bias: [Float], eps: Float
    ) {
        let dim = x.count
        var mean: Float = 0
        for v in x { mean += v }
        mean /= Float(dim)
        var variance: Float = 0
        for v in x { variance += (v - mean) * (v - mean) }
        variance /= Float(dim)
        let invStd = 1.0 / sqrt(variance + eps)
        for d in 0 ..< dim {
            x[d] = (x[d] - mean) * invStd * weight[d] + bias[d]
        }
    }

    /// Project a [hiddenSize] hidden state to [vocabSize] logits.
    private func projectToVocab(_ hidden: [Float], hiddenSize: Int) -> [Float] {
        let vocabSize = config.vocabSize
        let lmW = lmHeadWeight.toFloatArray()  // [vocabSize, hiddenSize]
        var logits = [Float](repeating: 0, count: vocabSize)
        for o in 0 ..< vocabSize {
            var acc: Float = 0
            let wBase = o * hiddenSize
            for d in 0 ..< hiddenSize { acc += lmW[wBase + d] * hidden[d] }
            logits[o] = acc
        }
        return logits
    }
}

// ─── Tokenizer ────────────────────────────────────────────────────────

/// SentencePiece-based tokenizer for CohereTranscribe, with special
/// tokens read from `tokenizer_config.json`.
public final class CohereTranscribeTokenizer: @unchecked Sendable {
    private let inner: any Tokenizer
    private let specialTokenToId: [String: Int]
    private let specialIds: Set<Int>
    public let eosId: Int

    public init(
        tokenizer: any Tokenizer,
        specialTokenToId: [String: Int],
        specialIds: Set<Int>,
        eosId: Int
    ) {
        self.inner = tokenizer
        self.specialTokenToId = specialTokenToId
        self.specialIds = specialIds
        self.eosId = eosId
    }

    /// Load from a checkpoint directory. Reads `tokenizer.json` (or the
    /// standard transformers format) and `tokenizer_config.json` for the
    /// special-token mapping. Falls back to standard tokenizer loading.
    ///
    /// `async` because `AutoTokenizer.from(modelFolder:)` is async in
    /// swift-transformers 1.x.
    public static func load(from directory: URL) async throws
        -> CohereTranscribeTokenizer
    {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        let configData = try Data(contentsOf: configURL)
        guard
            let parsed = try JSONSerialization.jsonObject(with: configData)
                as? [String: Any],
            let addedDecoder = parsed["added_tokens_decoder"] as? [String: Any]
        else {
            throw CohereTranscribeError.tokenizerConfigMissing
        }

        var tokenToId: [String: Int] = [:]
        for (key, value) in addedDecoder {
            guard let id = Int(key),
                let dict = value as? [String: Any],
                let content = dict["content"] as? String
            else { continue }
            tokenToId[content] = id
        }

        let allSpecialIds = Set(tokenToId.values)
        let eosId = tokenToId["<|endoftext|>"] ?? tokenToId["</s>"] ?? 1

        // Load the transformers tokenizer (handles SPM, BPE, etc.).
        let tok = try await AutoTokenizer.from(modelFolder: directory)
        return CohereTranscribeTokenizer(
            tokenizer: tok,
            specialTokenToId: tokenToId,
            specialIds: allSpecialIds,
            eosId: eosId)
    }

    public func encode(text: String) -> [Int] {
        if let id = specialTokenToId[text] { return [id] }
        return inner.encode(text: text)
    }

    public func decode(tokens: [Int]) -> String {
        inner.decode(tokens: tokens.filter { !specialIds.contains($0) })
    }

    /// Build the decoder prompt token ids for language + settings.
    /// Mirror of `CohereTranscribeTokenizer.buildPromptTokens` in mlx-audio-swift.
    public func buildPromptTokens(
        language: String = "en",
        usePunctuation: Bool = true,
        useTimestamps: Bool = false
    ) -> [Int] {
        let langCode = mapLanguageCode(language)
        let tokens = [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            langCode,
            langCode,
            usePunctuation ? "<|pnc|>" : "<|nopnc|>",
            "<|noitn|>",
            useTimestamps ? "<|timestamp|>" : "<|notimestamp|>",
            "<|nodiarize|>",
        ]
        return tokens.compactMap { specialTokenToId[$0] }
    }

    private func mapLanguageCode(_ language: String) -> String {
        let lang = language.lowercased()
        let map: [String: String] = [
            "en": "<|en|>", "english": "<|en|>",
            "fr": "<|fr|>", "french": "<|fr|>",
            "de": "<|de|>", "german": "<|de|>",
            "es": "<|es|>", "spanish": "<|es|>",
            "it": "<|it|>", "italian": "<|it|>",
            "pt": "<|pt|>", "portuguese": "<|pt|>",
            "nl": "<|nl|>", "dutch": "<|nl|>",
            "pl": "<|pl|>", "polish": "<|pl|>",
            "el": "<|el|>", "greek": "<|el|>",
            "ar": "<|ar|>", "arabic": "<|ar|>",
            "ja": "<|ja|>", "japanese": "<|ja|>",
            "zh": "<|zh|>", "chinese": "<|zh|>",
            "vi": "<|vi|>", "vietnamese": "<|vi|>",
            "ko": "<|ko|>", "korean": "<|ko|>",
        ]
        return map[lang] ?? "<|en|>"
    }
}

// ─── Errors ────────────────────────────────────────────────────────────

public enum CohereTranscribeError: Error, CustomStringConvertible {
    case configNotFound
    case tokenizerConfigMissing
    case missingWeight(String)

    public var description: String {
        switch self {
        case .configNotFound:
            return "CohereTranscribe: config.json not found or not a CohereTranscribe checkpoint"
        case .tokenizerConfigMissing:
            return "CohereTranscribe: tokenizer_config.json missing or malformed"
        case .missingWeight(let key):
            return "CohereTranscribe: required weight '\(key)' not found in checkpoint"
        }
    }
}

// ─── Registry detection + loader ─────────────────────────────────────

extension CohereTranscribeModel {
    /// Accepted `model_type` values:
    /// - `cohere_transcribe` — the original CohereLabs upstream string.
    /// - `cohere_asr` — the renamed string mlx-community ships in its
    ///   2026-03 conversion (`mlx-community/cohere-transcribe-03-2026-mlx-8bit`
    ///   and successors). Same architecture; the rename happened on the
    ///   conversion side, not in the model code, so both load identically.
    public static let modelTypes: Set<String> = ["cohere_transcribe", "cohere_asr"]
    public static let architectures: Set<String> = [
        "CohereTranscribeForConditionalGeneration"
    ]

    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) { return true }
        return false
    }

    /// Load a CohereTranscribe checkpoint from a resolved snapshot directory.
    ///
    /// `async` because the tokenizer loader (`AutoTokenizer.from`) is async
    /// in swift-transformers 1.x.
    public static func load(directory: URL, device: Device = .shared)
        async throws -> CohereTranscribeModel
    {
        let rawConfig = try ModelConfig.load(from: directory)
        guard let ct = CohereTranscribeConfig.from(rawConfig) else {
            throw ModelError.unsupportedModelType(
                "config.json is not a CohereTranscribe config")
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        let model = try build(config: ct, bundle: bundle, rawConfig: rawConfig)
        model.tokenizer = try? await CohereTranscribeTokenizer.load(from: directory)
        return model
    }

    // ─── Weight loading ───────────────────────────────────────────────

    /// Assemble a `CohereTranscribeModel` from config + weight bundle.
    public static func build(
        config ct: CohereTranscribeConfig,
        bundle: SafeTensorsBundle,
        rawConfig: ModelConfig? = nil
    ) throws -> CohereTranscribeModel {
        let ec = ct.encoder
        let dc = ct.decoder
        let quant = rawConfig?.quantization

        // ── Detect dtype from first float tensor ──
        let dtype =
            (try? bundle.tensor(
                named: "encoder.subsampling.conv0.weight"))?.dtype
            ?? .f32

        // ─ Helper: load a dense-only linear (no quantization needed for these) ─
        func denseLinear(_ key: String, hasBias: Bool = true) throws -> Linear {
            let w = try bundle.tensor(named: "\(key).weight")
            let b =
                hasBias && bundle.has("\(key).bias")
                ? try bundle.tensor(named: "\(key).bias")
                : nil
            return Linear(weight: w, bias: b)
        }

        func ln(_ base: String) throws -> LayerNorm {
            LayerNorm(
                weight: try bundle.tensor(named: "\(base).weight"),
                bias: try bundle.tensor(named: "\(base).bias"),
                eps: 1e-5)
        }

        // ─── ConvSubsampling ─────────────────────────────────────────
        // Weights in the checkpoint use the normalised key names produced
        // by normalizeCohereWeightKeys: conv0, conv2, conv3, conv5, conv6.
        func loadConv2dFlat(_ key: String) throws -> [Float] {
            try bundle.tensor(named: key).toFloatArray()
        }

        let conv0W = try loadConv2dFlat("encoder.subsampling.conv0.weight")
        let conv0B = try loadConv2dFlat("encoder.subsampling.conv0.bias")
        let convCh = ec.subsamplingConvChannels

        // conv2 / conv5: depthwise. Weight shape [convCh, kH, kW, 1] (OHWI after
        // the key-normalisation step). We flatten to [convCh, kH, kW].
        func loadDepthwiseConv2d(_ key: String) throws -> [Float] {
            let raw = try bundle.tensor(named: key)
            // Shape may be [convCh, kH, kW, 1] or [convCh, 1, kH, kW] depending
            // on how normalizeCohereWeightKeys transposed it. Either way we want
            // the [convCh, 9] flat array.
            return raw.toFloatArray()
        }

        let conv2DwW = try loadDepthwiseConv2d("encoder.subsampling.conv2.weight")
        let conv3W = try loadConv2dFlat("encoder.subsampling.conv3.weight")
        let conv3B = try loadConv2dFlat("encoder.subsampling.conv3.bias")
        let conv5DwW = try loadDepthwiseConv2d("encoder.subsampling.conv5.weight")
        let conv6W = try loadConv2dFlat("encoder.subsampling.conv6.weight")
        let conv6B = try loadConv2dFlat("encoder.subsampling.conv6.bias")
        // out: Linear [dModel, convCh * (featIn / subsamplingFactor)].
        let subsampOut = try denseLinear("encoder.subsampling.out")

        // ─── Relative positional encoding table ──────────────────────
        // Build the [2*posEmbMaxLen-1, dModel] sinusoidal table on load.
        // Matches RelPositionalEncoding.createPE in the reference.
        let relPE = buildRelPETable(posEmbMaxLen: ec.posEmbMaxLen, dModel: ec.dModel)

        // ─── Conformer encoder layers ─────────────────────────────────
        var encLayers: [CohereConformerLayer] = []
        encLayers.reserveCapacity(ec.nLayers)
        for i in 0 ..< ec.nLayers {
            let p = "encoder.layers.\(i)"
            // FF1
            let ff1 = CohereConformerFFN(
                linear1: try denseLinear("\(p).feed_forward1.linear1"),
                linear2: try denseLinear("\(p).feed_forward1.linear2"))
            // Self-attention: weights may be pre-merged (qkv_proj) or
            // separate (linear_q, linear_k, linear_v) then merged on load.
            let selfAttn = try loadConformerAttn(
                bundle: bundle, base: "\(p).self_attn", quant: quant,
                nHeads: ec.nHeads, dModel: ec.dModel)
            // Conformer conv.
            let convBlock = try loadConformerConv(
                bundle: bundle, base: "\(p).conv", kernelSize: ec.convKernelSize,
                dModel: ec.dModel)
            // FF2
            let ff2 = CohereConformerFFN(
                linear1: try denseLinear("\(p).feed_forward2.linear1"),
                linear2: try denseLinear("\(p).feed_forward2.linear2"))

            encLayers.append(
                CohereConformerLayer(
                    normFF1: try ln("\(p).norm_feed_forward1"),
                    ff1: ff1,
                    normSelfAttn: try ln("\(p).norm_self_att"),
                    selfAttn: selfAttn,
                    normConv: try ln("\(p).norm_conv"),
                    conv: convBlock,
                    normFF2: try ln("\(p).norm_feed_forward2"),
                    ff2: ff2,
                    normOut: try ln("\(p).norm_out")))
        }

        // ─── Bridge projection ────────────────────────────────────────
        let bridgeProj: Linear?
        if ec.dModel != dc.hiddenSize, bundle.has("bridge_proj.weight") {
            bridgeProj = try denseLinear("bridge_proj")
        } else if bundle.has("bridge_proj.weight") {
            bridgeProj = try denseLinear("bridge_proj")
        } else {
            bridgeProj = nil
        }

        // ─── Decoder ─────────────────────────────────────────────────
        let tokenEmb = try bundle.tensor(named: "decoder.embedding.token_embedding.weight")
        let decPETable = buildFixedPETable(
            maxLen: dc.maxSequenceLength, hiddenSize: dc.hiddenSize)
        let decNormEmb = try ln("decoder.embedding.layer_norm")

        var decLayers: [CohereDecoderLayer] = []
        decLayers.reserveCapacity(dc.numLayers)
        for i in 0 ..< dc.numLayers {
            let p = "decoder.core.layers.\(i)"
            let norm1 = try ln("\(p).layer_norm_1")
            let norm2 = try ln("\(p).layer_norm_2")
            let norm3 = try ln("\(p).layer_norm_3")

            let firstAttn = try loadDecoderAttn(
                bundle: bundle,
                base: "\(p).first_sub_layer",
                nHeads: dc.numAttentionHeads,
                headDim: dc.hiddenSize / dc.numAttentionHeads)
            let secondAttn = try loadDecoderAttn(
                bundle: bundle,
                base: "\(p).second_sub_layer",
                nHeads: dc.numAttentionHeads,
                headDim: dc.hiddenSize / dc.numAttentionHeads)
            let ffn = CohereDecoderFFN(
                denseIn: try denseLinear("\(p).third_sub_layer.dense_in"),
                denseOut: try denseLinear("\(p).third_sub_layer.dense_out"))
            decLayers.append(
                CohereDecoderLayer(
                    norm1: norm1, selfAttn: firstAttn,
                    norm2: norm2, crossAttn: secondAttn,
                    norm3: norm3, ffn: ffn))
        }

        let decFinalNorm = try ln("decoder.core.final_layer_norm")
        let lmHeadW = try bundle.tensor(named: "lm_head.weight")

        return CohereTranscribeModel(
            config: ct,
            conv0Weight: conv0W, conv0Bias: conv0B, conv0OutCh: convCh,
            conv2DwWeight: conv2DwW,
            conv3Weight: conv3W, conv3Bias: conv3B,
            conv5DwWeight: conv5DwW,
            conv6Weight: conv6W, conv6Bias: conv6B,
            subsamplingOut: subsampOut,
            relPETable: relPE,
            encoderLayers: encLayers,
            bridgeProj: bridgeProj,
            tokenEmbedding: tokenEmb,
            decoderPETable: decPETable,
            decoderNormEmb: decNormEmb,
            decoderLayers: decLayers,
            decoderFinalNorm: decFinalNorm,
            lmHeadWeight: lmHeadW,
            dtype: dtype)
    }

    // ─── Load helpers ─────────────────────────────────────────────────

    /// Build the `[2*maxLen-1, dModel]` relative positional encoding table.
    /// Matches `RelPositionalEncoding.createPE` with positions
    /// `(maxLen-1) → -(maxLen-1)` (positive to negative).
    private static func buildRelPETable(posEmbMaxLen: Int, dModel: Int) -> [Float] {
        let totalLen = 2 * posEmbMaxLen - 1
        var table = [Float](repeating: 0, count: totalLen * dModel)
        let halfDim = dModel / 2
        let divTermDenom = Double(dModel)
        for p in 0 ..< totalLen {
            let pos = Double(posEmbMaxLen - 1 - p)  // (maxLen-1) down to -(maxLen-1)
            for i in 0 ..< halfDim {
                let divTerm = exp(-Double(i * 2) * log(10_000.0) / divTermDenom)
                let angle = pos * divTerm
                // sin at even index, cos at odd index.
                table[p * dModel + i * 2] = Float(sin(angle))
                table[p * dModel + i * 2 + 1] = Float(cos(angle))
            }
        }
        return table
    }

    /// Build a `[maxLen, hiddenSize]` fixed sinusoidal positional encoding
    /// (interleaved sin/cos, scaled by 1/sqrt(hiddenSize)).
    /// Matches `FixedPositionalEncoding` in the reference decoder.
    private static func buildFixedPETable(maxLen: Int, hiddenSize: Int) -> [Float] {
        let halfDim = hiddenSize / 2
        let scale = 1.0 / Float(sqrt(Double(hiddenSize)))
        var table = [Float](repeating: 0, count: maxLen * hiddenSize)
        for pos in 0 ..< maxLen {
            for i in 0 ..< halfDim {
                let divTerm = exp(-Double(i) * log(10_000.0) / Double(halfDim))
                let angle = Double(pos) * divTerm
                // Interleaved: [sin0, cos0, sin1, cos1, ...].
                table[pos * hiddenSize + i * 2] = Float(sin(angle)) * scale
                table[pos * hiddenSize + i * 2 + 1] = Float(cos(angle)) * scale
            }
        }
        return table
    }

    /// Load the Conformer self-attention weight block. The checkpoint may
    /// store QKV merged as `qkv_proj` (after normalizeCohereWeightKeys
    /// concatenates the split linear_q/k/v projections) or separately.
    private static func loadConformerAttn(
        bundle: SafeTensorsBundle, base: String, quant: ModelConfig.QuantizationConfig?,
        nHeads: Int, dModel: Int
    ) throws -> CohereConformerAttn {
        // After normalizeCohereWeightKeys, QKV are merged into qkv_proj.
        let qkvProj = try loadLinear(base: "\(base).qkv_proj", in: bundle, quantization: quant)
        let posProj = try loadLinear(base: "\(base).pos_proj", in: bundle, quantization: nil)
        let outProj = try loadLinear(base: "\(base).out_proj", in: bundle, quantization: quant)

        // pos_bias_u and pos_bias_v are 1D parameter tensors [nHeads * dK].
        let dK = dModel / nHeads
        let biasU: [Float]
        let biasV: [Float]
        if bundle.has("\(base).pos_bias_u") {
            biasU = try bundle.tensor(named: "\(base).pos_bias_u").toFloatArray()
            biasV = try bundle.tensor(named: "\(base).pos_bias_v").toFloatArray()
        } else {
            // Default to zeros if not present.
            biasU = [Float](repeating: 0, count: nHeads * dK)
            biasV = [Float](repeating: 0, count: nHeads * dK)
        }

        return CohereConformerAttn(
            qkvProj: qkvProj.inner as! Linear,  // dense-only for conformer attn
            posProj: posProj.inner as! Linear,
            outProj: outProj.inner as! Linear,
            nHeads: nHeads, dK: dK,
            posBiasU: biasU, posBiasV: biasV)
    }

    /// Load the Conformer convolution sub-block weights.
    private static func loadConformerConv(
        bundle: SafeTensorsBundle, base: String,
        kernelSize: Int, dModel: Int
    ) throws -> CohereConformerConv {
        let pw1 = try loadLinear(base: "\(base).pointwise_conv1", in: bundle, quantization: nil)
        let pw2 = try loadLinear(base: "\(base).pointwise_conv2", in: bundle, quantization: nil)

        // Depthwise conv: weight [dModel, kernelSize, 1] or [dModel, 1, kernelSize].
        let dwRaw = try bundle.tensor(named: "\(base).depthwise_conv.weight").toFloatArray()
        // Flatten to [dModel, kernelSize] — the trivial dimension (groups=1 per ch) is stripped.
        let dwWeight = Array(dwRaw.prefix(dModel * kernelSize))

        // BatchNorm parameters.
        let bnW = try bundle.tensor(named: "\(base).batch_norm.weight").toFloatArray()
        let bnB = try bundle.tensor(named: "\(base).batch_norm.bias").toFloatArray()
        let bnMean =
            bundle.has("\(base).batch_norm.running_mean")
            ? try bundle.tensor(named: "\(base).batch_norm.running_mean").toFloatArray()
            : [Float](repeating: 0, count: dModel)
        let bnVar =
            bundle.has("\(base).batch_norm.running_var")
            ? try bundle.tensor(named: "\(base).batch_norm.running_var").toFloatArray()
            : [Float](repeating: 1, count: dModel)

        return CohereConformerConv(
            pointwiseConv1: pw1.inner as! Linear,
            depthwiseWeights: dwWeight,
            kernelSize: kernelSize,
            batchNormWeight: bnW,
            batchNormBias: bnB,
            batchNormRunningMean: bnMean,
            batchNormRunningVar: bnVar,
            batchNormEps: 1e-5,
            pointwiseConv2: pw2.inner as! Linear)
    }

    /// Load a decoder attention block (self or cross).
    private static func loadDecoderAttn(
        bundle: SafeTensorsBundle, base: String,
        nHeads: Int, headDim: Int
    ) throws -> CohereDecoderAttn {
        // After normalizeCohereWeightKeys, QKV is merged into qkv_proj.
        let qkvW = try bundle.tensor(named: "\(base).qkv_proj.weight")
        let qkvB =
            bundle.has("\(base).qkv_proj.bias")
            ? try bundle.tensor(named: "\(base).qkv_proj.bias")
            : nil
        let outW = try bundle.tensor(named: "\(base).out_proj.weight")
        let outB =
            bundle.has("\(base).out_proj.bias")
            ? try bundle.tensor(named: "\(base).out_proj.bias")
            : nil
        return CohereDecoderAttn(
            qkvProj: Linear(weight: qkvW, bias: qkvB),
            outProj: Linear(weight: outW, bias: outB),
            nHeads: nHeads, headDim: headDim)
    }
}
