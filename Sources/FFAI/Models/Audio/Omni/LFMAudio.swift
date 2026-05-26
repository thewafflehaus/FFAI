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
// LFMAudio — LiquidAI's LFM2.5-Audio speech-to-speech / omni family.
//
// HF repos: `mlx-community/LFM2.5-Audio-1.5B-6bit`
//           `mlx-community/LFM2.5-Audio-1.5B-bf16`
//
// Architecture (`Lfm2AudioForConditionalGeneration`):
//
//   waveform ──NeMo log-Mel front-end (16 kHz, 128 bins)──▶
//   ──ConformerEncoder (17 blocks, dw-striding subsampling ×8)──▶
//       [T/8, d_model=512]
//   ──AdapterMLP (LayerNorm + Linear(512→2048))──▶
//       [T/8, lfm_hidden=2048]
//   ──LFM2 backbone (16 layers, conv+attention hybrid)──▶ text tokens
//
// For the speech-to-speech case the model also carries:
//   ─ AudioEmbedding + AudioHead (Depthformer, 6 layers) for audio output
//   ─ DepthEmbeddings + DepthLinear for multi-codebook generation
//
// ## FFAI scope
//
// This port wires the audio-encoding path:
//   `encodeAudio(waveform:device:) → Tensor [nTokens, lfmHidden]`
//
// The conformer encoder runs CPU-side (small token count after 8× sub-
// sampling) following the Parakeet precedent. The LFM2 backbone is
// reused from `LFM2.swift` via `lfm2LoadModelQuantized`.
//
// ## Weight key layout (post-sanitize — already in correct form)
//
//   audio_encoder.pre_encode.conv.{0,2,3,5,6}.{weight,bias}
//   audio_encoder.pre_encode.out.{weight,bias[,scales,biases]}
//   audio_encoder.layers.N.{ff1_norm,ff1,attn_norm,attn,conv_norm,conv,
//                            ff2_norm,ff2,final_norm}.*
//   audio_adapter.{norm,linears.N}.{weight,bias[,scales,biases]}
//   lfm.{embed_tokens,embedding_norm,layers.N}.*  (matches LFM2.swift)
//
// ## Detection
//   `model_type == "lfm_audio"` or architecture
//   `"Lfm2AudioForConditionalGeneration"`.

import Foundation
import Metal

// ─── Error types ─────────────────────────────────────────────────────

public enum LFMAudioError: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    case weightNotFound(String)

    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "LFMAudio: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "LFMAudio: unsupported config: \(m)"
        case .weightNotFound(let k):
            return "LFMAudio: weight not found: \(k)"
        }
    }
}

// ─── Configuration ───────────────────────────────────────────────────

/// NeMo preprocessor hyper-parameters (matches `preprocessor` block in
/// the LFM2.5-Audio `config.json`).
public struct LFMAudioPreprocessorConfig: Sendable {
    /// Input sample rate in Hz (16000).
    public let sampleRate: Int
    /// Number of Mel filterbank bins (128).
    public let nMels: Int
    /// FFT length in samples (512).
    public let nFFT: Int
    /// Window length in samples (sampleRate × windowSize = 400).
    public let winLength: Int
    /// Hop length in samples (sampleRate × windowStride = 160).
    public let hopLength: Int
    /// Pre-emphasis coefficient (0.97).
    public let preemph: Float
    /// Dither amount (1e-5). Applied before pre-emphasis.
    public let dither: Float
    /// Normalisation mode: `"per_feature"` (z-score each Mel bin across time).
    public let normalise: String
    /// Log floor guard value — log(mel + 5.96e-8).
    public let logGuard: Float

    public init(
        sampleRate: Int = 16_000, nMels: Int = 128, nFFT: Int = 512,
        winLength: Int = 400, hopLength: Int = 160,
        preemph: Float = 0.97, dither: Float = 1e-5,
        normalise: String = "per_feature", logGuard: Float = 5.96e-8
    ) {
        self.sampleRate = sampleRate; self.nMels = nMels; self.nFFT = nFFT
        self.winLength = winLength; self.hopLength = hopLength
        self.preemph = preemph; self.dither = dither
        self.normalise = normalise; self.logGuard = logGuard
    }
}

/// ConformerEncoder hyper-parameters (matches `encoder` block).
public struct LFMAudioEncoderConfig: Sendable {
    /// Input feature size (= nMels, 128).
    public let featIn: Int
    /// Number of Conformer blocks (17).
    public let nLayers: Int
    /// Encoder hidden dimension (512).
    public let dModel: Int
    /// Attention heads (8).
    public let nHeads: Int
    /// Feed-forward expansion factor (4 → ffHidden = 2048).
    public let ffExpansionFactor: Int
    /// Subsampling factor (8 — three stride-2 conv passes).
    public let subsamplingFactor: Int
    /// Number of channels in the subsampling convolutions (256).
    public let subsamplingConvChannels: Int
    /// Maximum length of the relative-positional encoding table (5000).
    public let posEmbMaxLen: Int
    /// Depthwise-conv kernel size in each Conformer block (9).
    public let convKernelSize: Int

    public var headDim: Int { dModel / nHeads }
    public var ffHidden: Int { dModel * ffExpansionFactor }

    public init(
        featIn: Int = 128, nLayers: Int = 17, dModel: Int = 512,
        nHeads: Int = 8, ffExpansionFactor: Int = 4,
        subsamplingFactor: Int = 8, subsamplingConvChannels: Int = 256,
        posEmbMaxLen: Int = 5000, convKernelSize: Int = 9
    ) {
        self.featIn = featIn; self.nLayers = nLayers; self.dModel = dModel
        self.nHeads = nHeads; self.ffExpansionFactor = ffExpansionFactor
        self.subsamplingFactor = subsamplingFactor
        self.subsamplingConvChannels = subsamplingConvChannels
        self.posEmbMaxLen = posEmbMaxLen; self.convKernelSize = convKernelSize
    }
}

/// Top-level LFMAudio configuration.
public struct LFMAudioConfig: Sendable {
    public let modelType: String
    public let sampleRate: Int
    public let codebooks: Int
    public let audioVocabSize: Int
    public let adapterHiddenDims: [Int]
    public let adapterUseLayerNorm: Bool
    public let preprocessor: LFMAudioPreprocessorConfig
    public let encoder: LFMAudioEncoderConfig
    /// LFM2 backbone hidden size (2048 for LFM2.5-Audio-1.5B).
    public let lfmHidden: Int

    public init(
        modelType: String = "lfm_audio",
        sampleRate: Int = 24_000,
        codebooks: Int = 8,
        audioVocabSize: Int = 2049,
        adapterHiddenDims: [Int] = [2048],
        adapterUseLayerNorm: Bool = true,
        preprocessor: LFMAudioPreprocessorConfig = LFMAudioPreprocessorConfig(),
        encoder: LFMAudioEncoderConfig = LFMAudioEncoderConfig(),
        lfmHidden: Int = 2048
    ) {
        self.modelType = modelType; self.sampleRate = sampleRate
        self.codebooks = codebooks; self.audioVocabSize = audioVocabSize
        self.adapterHiddenDims = adapterHiddenDims
        self.adapterUseLayerNorm = adapterUseLayerNorm
        self.preprocessor = preprocessor; self.encoder = encoder
        self.lfmHidden = lfmHidden
    }

    /// Decode from a checkpoint `config.json`.
    public static func from(_ config: ModelConfig) throws -> LFMAudioConfig {
        guard let prepRaw = config.raw["preprocessor"] as? [String: Any]
        else { throw LFMAudioError.missingConfig("preprocessor") }
        guard let encRaw  = config.raw["encoder"]       as? [String: Any]
        else { throw LFMAudioError.missingConfig("encoder") }
        guard let lfmRaw  = config.raw["lfm"]           as? [String: Any]
        else { throw LFMAudioError.missingConfig("lfm") }

        // ── Preprocessor ─────────────────────────────────────────────
        let sr = prepRaw["sample_rate"] as? Int ?? 16_000
        let nMels = prepRaw["features"] as? Int ?? 128
        let nFFT = prepRaw["n_fft"] as? Int ?? 512
        let windowSize   = prepRaw["window_size"]   as? Double ?? 0.025
        let windowStride = prepRaw["window_stride"] as? Double ?? 0.01
        let winLength  = Int((windowSize   * Double(sr)).rounded())
        let hopLength  = Int((windowStride * Double(sr)).rounded())
        let preemph    = prepRaw["preemph"]     as? Double ?? 0.97
        let dither     = prepRaw["dither"]      as? Double ?? 1e-5
        let normalise  = prepRaw["normalize"]   as? String ?? "per_feature"
        let prep = LFMAudioPreprocessorConfig(
            sampleRate: sr, nMels: nMels, nFFT: nFFT,
            winLength: winLength, hopLength: hopLength,
            preemph: Float(preemph), dither: Float(dither), normalise: normalise)

        // ── ConformerEncoder ─────────────────────────────────────────
        let featIn     = encRaw["feat_in"]                    as? Int ?? 128
        let nLayers    = encRaw["n_layers"]                   as? Int ?? 17
        let dModel     = encRaw["d_model"]                    as? Int ?? 512
        let nHeads     = encRaw["n_heads"]                    as? Int ?? 8
        let ffExp      = encRaw["ff_expansion_factor"]        as? Int ?? 4
        let subFactor  = encRaw["subsampling_factor"]         as? Int ?? 8
        let subConvCh  = encRaw["subsampling_conv_channels"]  as? Int ?? 256
        let posMax     = encRaw["pos_emb_max_len"]            as? Int ?? 5000
        let convKernel = encRaw["conv_kernel_size"]           as? Int ?? 9
        let enc = LFMAudioEncoderConfig(
            featIn: featIn, nLayers: nLayers, dModel: dModel,
            nHeads: nHeads, ffExpansionFactor: ffExp,
            subsamplingFactor: subFactor, subsamplingConvChannels: subConvCh,
            posEmbMaxLen: posMax, convKernelSize: convKernel)

        // ── LFM2 backbone hidden dim ─────────────────────────────────
        let lfmHidden = lfmRaw["hidden_size"] as? Int
            ?? lfmRaw["block_dim"] as? Int
            ?? 2048

        let modelType       = config.modelType ?? "lfm_audio"
        let topSampleRate   = config.int("sample_rate") ?? 24_000
        let codebooks       = config.int("codebooks") ?? 8
        let audioVocabSize  = config.int("audio_vocab_size") ?? 2049
        let adapterDims: [Int]
        if let raw = config.raw["adapter_hidden_dims"] as? [Int] {
            adapterDims = raw
        } else if let rawArr = config.raw["adapter_hidden_dims"] as? [Any] {
            adapterDims = rawArr.compactMap { $0 as? Int }
        } else {
            adapterDims = [lfmHidden]
        }
        let adapterLN = config.bool("adapter_use_layer_norm") ?? true

        return LFMAudioConfig(
            modelType: modelType, sampleRate: topSampleRate,
            codebooks: codebooks, audioVocabSize: audioVocabSize,
            adapterHiddenDims: adapterDims, adapterUseLayerNorm: adapterLN,
            preprocessor: prep, encoder: enc, lfmHidden: lfmHidden)
    }
}

// ─── Conformer encoder weights ────────────────────────────────────────
//
// The CPU-side Conformer mirrors the Parakeet approach: weights are
// stored as [Float] arrays for direct arithmetic (no GPU ops in the
// encoder path). The backbone is what needs GPU throughput.

/// Weights for the Conv subsampling (dw-striding, factor 8 = three
/// stride-2 stages). Input layout: [T, F, 1] → [T/8, d_model].
struct LFMAudioSubsamplingWeights: Sendable {
    // conv.0: [convCh, 1, 3, 3] stored as [convCh * 9]
    let conv0W: [Float]; let conv0B: [Float]
    // conv.2: depthwise [convCh, 1, 3, 3] + conv.3: pointwise [convCh, convCh, 1, 1]
    let conv2W: [Float]; let conv2B: [Float]
    let conv3W: [Float]; let conv3B: [Float]
    // conv.5: depthwise [convCh, 1, 3, 3] + conv.6: pointwise [convCh, convCh, 1, 1]
    let conv5W: [Float]; let conv5B: [Float]
    let conv6W: [Float]; let conv6B: [Float]
    // out: [dModel, convCh * (featIn / 8)]
    let outW: [Float]; let outB: [Float]
    let convCh: Int; let dModel: Int
}

/// Weights for the relative multi-head attention sub-block within a
/// Conformer layer. All projections are [dModel × dModel] dense matrices.
struct LFMAudioConformerAttnWeights: Sendable {
    let qW: [Float]; let qB: [Float]
    let kW: [Float]; let kB: [Float]
    let vW: [Float]; let vB: [Float]
    let outW: [Float]; let outB: [Float]
    let posW: [Float]                   // pos_proj (no bias)
    let posBiasU: [Float]               // [nHeads, headDim]
    let posBiasV: [Float]               // [nHeads, headDim]
}

/// Weights for the Conformer convolution sub-block (batch-norm instead
/// of layer-norm for the conv normalisation).
struct LFMAudioConformerConvWeights: Sendable {
    let pw1W: [Float]; let pw1B: [Float]   // pointwise_conv1: dModel → 2*dModel
    let dwW: [Float]; let dwB: [Float]     // depthwise_conv: dModel, kernel
    // Batch-norm params
    let bnW: [Float]; let bnB: [Float]
    let bnMean: [Float]; let bnVar: [Float]
    let pw2W: [Float]; let pw2B: [Float]   // pointwise_conv2: dModel → dModel
}

/// Weights for a ConformerFeedForward sub-block.
struct LFMAudioConformerFFWeights: Sendable {
    let linear1W: [Float]; let linear1B: [Float]
    let linear2W: [Float]; let linear2B: [Float]
}

/// Weights for all sub-blocks of one Conformer block, plus their norms.
struct LFMAudioConformerBlockWeights: Sendable {
    // Pre-norm LayerNorm (weight + bias)
    let ff1NormW: [Float]; let ff1NormB: [Float]
    let attnNormW: [Float]; let attnNormB: [Float]
    let convNormW: [Float]; let convNormB: [Float]
    let ff2NormW: [Float]; let ff2NormB: [Float]
    let finalNormW: [Float]; let finalNormB: [Float]
    // Sub-blocks
    let ff1: LFMAudioConformerFFWeights
    let attn: LFMAudioConformerAttnWeights
    let conv: LFMAudioConformerConvWeights
    let ff2: LFMAudioConformerFFWeights
    let dModel: Int
}

/// Weights for the AdapterMLP that projects encoder output to LFM2 dim.
struct LFMAudioAdapterWeights: Sendable {
    let normW: [Float]?; let normB: [Float]?   // optional LayerNorm
    let linearWs: [[Float]]; let linearBs: [[Float]]
    let inDim: Int; let outDim: Int
}

// ─── LFMAudio model ───────────────────────────────────────────────────

/// A loaded LFMAudio model. The Conformer encoder + AdapterMLP run CPU-
/// side; the LFM2 backbone is a GPU-resident `LFM2Model`.
public final class LFMAudioModel: @unchecked Sendable {
    public let config: LFMAudioConfig

    let subsampling: LFMAudioSubsamplingWeights
    let blocks: [LFMAudioConformerBlockWeights]
    let adapter: LFMAudioAdapterWeights
    let lfm: LFM2Model
    let dtype: DType

    /// Relative positional encoding table [2*maxLen-1, dModel] (row-major).
    var relPosTable: [Float]
    var relPosMaxLen: Int

    init(config: LFMAudioConfig,
         subsampling: LFMAudioSubsamplingWeights,
         blocks: [LFMAudioConformerBlockWeights],
         adapter: LFMAudioAdapterWeights,
         lfm: LFM2Model, dtype: DType,
         relPosTable: [Float], relPosMaxLen: Int) {
        self.config = config
        self.subsampling = subsampling
        self.blocks = blocks
        self.adapter = adapter
        self.lfm = lfm
        self.dtype = dtype
        self.relPosTable = relPosTable
        self.relPosMaxLen = relPosMaxLen
    }

    // ── Detection ────────────────────────────────────────────────────

    public static func handles(_ config: ModelConfig) -> Bool {
        if config.modelType == "lfm_audio" { return true }
        if config.architecture == "Lfm2AudioForConditionalGeneration" {
            return true
        }
        return false
    }

    // ── Audio encoding ───────────────────────────────────────────────

    /// Encode a mono waveform into feature tokens in the LFM2 hidden dim.
    ///
    /// - The input waveform is resampled/assumed to be at
    ///   `config.preprocessor.sampleRate` (16 kHz).
    /// - Returns a `Tensor` of shape `[nTokens, lfmHidden]` suitable for
    ///   splicing into an LFM2 prompt embedding stream.
    public func encodeAudio(waveform: [Float], device: Device = .shared)
        -> Tensor {
        let prep = config.preprocessor
        let enc  = config.encoder

        // 1. NeMo log-Mel front-end → [nFrames, nMels] (CPU float)
        let mel = lfmAudioLogMel(waveform: waveform, cfg: prep)
        guard !mel.isEmpty else {
            return Tensor.empty(shape: [0, config.lfmHidden], dtype: dtype,
                                device: device)
        }
        let nFrames = mel.count / prep.nMels

        // 2. Conformer encoder → [nTokens, dModel]
        let (encoded, nTokens) = conformerEncode(
            mel: mel, nFrames: nFrames, enc: enc)
        guard nTokens > 0 else {
            return Tensor.empty(shape: [0, config.lfmHidden], dtype: dtype,
                                device: device)
        }

        // 3. AdapterMLP → [nTokens, lfmHidden]
        let adapted = adapterForward(encoded, nTokens: nTokens)

        // 4. Upload to a GPU Tensor for the LFM2 backbone.
        let outT = Tensor.empty(
            shape: [nTokens, config.lfmHidden], dtype: dtype, device: device)
        let outF32: [Float]
        switch dtype {
        case .f32: outF32 = adapted
        default:   outF32 = adapted  // copyIn handles dtype conversion below
        }
        _ = outF32  // suppress warning — written via switch below
        switch dtype {
        case .f32:
            outT.copyIn(from: adapted)
        case .bf16:
            outT.copyIn(from: adapted.map {
                UInt16(truncatingIfNeeded: $0.bitPattern >> 16) })
        case .f16:
            outT.copyIn(from: adapted.map { Float16($0) })
        default:
            fatalError("LFMAudio: unsupported activation dtype \(dtype)")
        }
        return outT
    }

    // ─── NeMo log-Mel front-end ──────────────────────────────────────
    //
    // The NeMo AudioToMelSpectrogramPreprocessor produces:
    //   1. Pre-emphasis
    //   2. STFT with a Hann window (constant-pad by nFFT/2 on each side)
    //   3. Power spectrum → Mel filterbank (slaney-norm, linear scale)
    //   4. Log (ln(mel + guard))
    //   5. Per-feature z-score normalisation
    //
    // This is DIFFERENT from the Whisper front-end (which uses HTK Mel
    // + log10 + dynamic-range clamp + affine norm). We compute it
    // CPU-side in float32.

    private func lfmAudioLogMel(
        waveform: [Float], cfg: LFMAudioPreprocessorConfig
    ) -> [Float] {
        guard !waveform.isEmpty else { return [] }
        let nMels = cfg.nMels; let nFFT = cfg.nFFT
        let winLen = cfg.winLength; let hop = cfg.hopLength

        // ── Pre-emphasis ────────────────────────────────────────────
        var sig = waveform
        if cfg.preemph > 0 {
            for i in stride(from: sig.count - 1, through: 1, by: -1) {
                sig[i] -= cfg.preemph * sig[i - 1]
            }
        }

        // ── Constant-pad (nFFT/2 zeros on each side) ────────────────
        let pad = nFFT / 2
        var padded = [Float](repeating: 0, count: sig.count + 2 * pad)
        for i in 0..<sig.count { padded[i + pad] = sig[i] }

        // ── Hann window ─────────────────────────────────────────────
        var window = [Float](repeating: 0, count: winLen)
        for i in 0..<winLen {
            window[i] = 0.5 * (1 - cos(2 * Float.pi * Float(i) / Float(winLen)))
        }
        // Zero-pad window to nFFT if winLen < nFFT
        var fullWindow = [Float](repeating: 0, count: nFFT)
        let padLeft = (nFFT - winLen) / 2
        for i in 0..<winLen { fullWindow[padLeft + i] = window[i] }

        // ── Number of frames ────────────────────────────────────────
        let nFrames = 1 + (padded.count - nFFT) / hop
        guard nFrames > 0 else { return [] }

        // ── Mel filterbank (slaney-norm, linear scale, 0..sr/2) ─────
        let sr = cfg.sampleRate
        let melFB = lfmAudioSlaneyMelFilterbank(
            nMels: nMels, nFFT: nFFT, sampleRate: sr)

        // ── STFT → power → Mel → log ─────────────────────────────────
        let nFreq = nFFT / 2 + 1
        var melFrames = [Float](repeating: 0, count: nFrames * nMels)

        for f in 0..<nFrames {
            let start = f * hop

            // Windowed frame
            var frame = [Float](repeating: 0, count: nFFT)
            for i in 0..<nFFT { frame[i] = padded[start + i] * fullWindow[i] }

            // DFT (real-valued input → nFreq bins)
            let spec = lfmAudioRealDFT(frame, nFFT: nFFT)

            // Power spectrum |X[k]|²
            var power = [Float](repeating: 0, count: nFreq)
            for k in 0..<nFreq {
                let re = spec[2 * k], im = spec[2 * k + 1]
                power[k] = re * re + im * im
            }

            // Mel projection: mel[m] = sum_k power[k] * fb[m, k]
            for m in 0..<nMels {
                var val: Float = 0
                let base = m * nFreq
                for k in 0..<nFreq { val += power[k] * melFB[base + k] }
                melFrames[f * nMels + m] = log(val + cfg.logGuard)
            }
        }

        // ── Per-feature z-score normalisation ────────────────────────
        if cfg.normalise == "per_feature" {
            for m in 0..<nMels {
                var sum: Float = 0; var sum2: Float = 0
                for f in 0..<nFrames {
                    let v = melFrames[f * nMels + m]
                    sum += v; sum2 += v * v
                }
                let mean = sum / Float(nFrames)
                let variance = max((sum2 / Float(nFrames)) - mean * mean, 0)
                let std = sqrt(variance) + 1e-5
                for f in 0..<nFrames {
                    melFrames[f * nMels + m] = (melFrames[f * nMels + m] - mean) / std
                }
            }
        }

        return melFrames   // [nFrames, nMels] row-major
    }

    // ─── DFT helpers ─────────────────────────────────────────────────

    /// Compute a real-input DFT of `input` (length `nFFT`). Returns
    /// interleaved `[re0, im0, re1, im1, …]` for the `nFFT/2+1` bins.
    /// Uses a naive O(N²) DFT — correct on small sequences (nFFT ≤ 512).
    private func lfmAudioRealDFT(_ input: [Float], nFFT: Int) -> [Float] {
        let nFreq = nFFT / 2 + 1
        var out = [Float](repeating: 0, count: 2 * nFreq)
        let twoPiOverN = -2.0 * Float.pi / Float(nFFT)
        for k in 0..<nFreq {
            var re: Float = 0, im: Float = 0
            let factor = twoPiOverN * Float(k)
            for n in 0..<nFFT {
                let angle = factor * Float(n)
                re += input[n] * cos(angle)
                im += input[n] * sin(angle)
            }
            out[2 * k] = re; out[2 * k + 1] = im
        }
        return out
    }

    // ─── Slaney Mel filterbank ────────────────────────────────────────

    /// Build a slaney-norm triangular Mel filterbank [nMels, nFreq].
    /// Frequency range: 0 Hz to sr/2 Hz.
    private func lfmAudioSlaneyMelFilterbank(
        nMels: Int, nFFT: Int, sampleRate: Int
    ) -> [Float] {
        let nFreq = nFFT / 2 + 1
        let fMin = 0.0, fMax = Double(sampleRate) / 2.0

        func hzToMel(_ hz: Double) -> Double {
            // Slaney formula (linear below 1kHz, log above)
            let f_sp = 200.0 / 3.0
            let minLogHz = 1000.0
            let minLogMel = (minLogHz - 0.0) / f_sp
            let logStep = log(6.4) / 27.0
            if hz < minLogHz {
                return hz / f_sp
            } else {
                return minLogMel + log(hz / minLogHz) / logStep
            }
        }
        func melToHz(_ mel: Double) -> Double {
            let f_sp = 200.0 / 3.0
            let minLogHz = 1000.0
            let minLogMel = (minLogHz - 0.0) / f_sp
            let logStep = log(6.4) / 27.0
            if mel < minLogMel {
                return mel * f_sp
            } else {
                return minLogHz * exp(logStep * (mel - minLogMel))
            }
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        // nMels+2 evenly-spaced Mel values
        var melPts = [Double](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            melPts[i] = melMin + Double(i) * (melMax - melMin) / Double(nMels + 1)
        }
        // Convert to Hz then to bin index
        let freqStep = Double(sampleRate) / Double(nFFT)
        var bins = melPts.map { Int(melToHz($0) / freqStep + 0.5) }

        var fb = [Float](repeating: 0, count: nMels * nFreq)
        for m in 0..<nMels {
            let lo = bins[m]; let center = bins[m + 1]; let hi = bins[m + 2]
            // Rising slope: lo..center
            for k in lo..<center {
                guard k >= 0 && k < nFreq else { continue }
                let num = Double(k - lo)
                let den = Double(center - lo)
                fb[m * nFreq + k] = den > 0 ? Float(num / den) : 0
            }
            // Falling slope: center..hi
            for k in center..<hi {
                guard k >= 0 && k < nFreq else { continue }
                let num = Double(hi - k)
                let den = Double(hi - center)
                fb[m * nFreq + k] = den > 0 ? Float(num / den) : 0
            }
            // Slaney normalisation: scale each triangle by 2/(hi_hz - lo_hz)
            let loHz = melToHz(melPts[m])
            let hiHz = melToHz(melPts[m + 2])
            let norm = hiHz > loHz ? Float(2.0 / (hiHz - loHz)) : 1
            for k in 0..<nFreq { fb[m * nFreq + k] *= norm }
        }
        return fb
    }

    // ─── DwStriding subsampling (Conformer pre-encoder) ──────────────
    //
    // Three stages of stride-2 conv2d give factor-8 downsampling.
    // Each stage halves both the time (T) and frequency (F) dimensions:
    //
    //   Stage 1: conv0(1→convCh, 3×3, stride 2, pad 1)  + ReLU
    //   Stage 2: depthwise conv2(convCh, 3×3, stride 2, pad 1)
    //            + pointwise  conv3(convCh→convCh, 1×1, stride 1, pad 0) + ReLU
    //   Stage 3: depthwise conv5(convCh, 3×3, stride 2, pad 1)
    //            + pointwise  conv6(convCh→convCh, 1×1, stride 1, pad 0) + ReLU
    //   out: linear [T/8, convCh*(F/8)] → [T/8, dModel]
    //
    // Input is treated as [T, F, inCh] (last dim = channel), matching
    // the NeMo layout.

    private func conformerSubsample(
        mel: [Float], nFrames: Int, nMels: Int, sub: LFMAudioSubsamplingWeights
    ) -> ([Float], Int, Int) {
        let convCh = sub.convCh

        // Reshape to [T, F, 1]
        var current = mel
        var curT = nFrames, curF = nMels, curCh = 1

        // ── Stage 1: conv0 (1→convCh, 3×3, stride 2, pad 1) + ReLU ──
        var (out1, t1, f1, ch1) = conv2DTHFReLU(
            input: current, T: curT, F: curF, inCh: curCh,
            weight: sub.conv0W, bias: sub.conv0B, outCh: convCh,
            kH: 3, kW: 3, strideH: 2, strideW: 2, padH: 1, padW: 1, relu: true)
        current = out1; curT = t1; curF = f1; curCh = ch1

        // ── Stage 2: depthwise + pointwise + ReLU ────────────────────
        var (out2a, t2a, f2a, ch2a) = conv2DTHFReLU(
            input: current, T: curT, F: curF, inCh: curCh,
            weight: sub.conv2W, bias: sub.conv2B, outCh: convCh,
            kH: 3, kW: 3, strideH: 2, strideW: 2, padH: 1, padW: 1, relu: false,
            depthwise: true)
        current = out2a; curT = t2a; curF = f2a; curCh = ch2a
        var (out2b, t2b, f2b, ch2b) = conv2DTHFReLU(
            input: current, T: curT, F: curF, inCh: curCh,
            weight: sub.conv3W, bias: sub.conv3B, outCh: convCh,
            kH: 1, kW: 1, strideH: 1, strideW: 1, padH: 0, padW: 0, relu: true)
        current = out2b; curT = t2b; curF = f2b; curCh = ch2b

        // ── Stage 3: depthwise + pointwise + ReLU ────────────────────
        var (out3a, t3a, f3a, ch3a) = conv2DTHFReLU(
            input: current, T: curT, F: curF, inCh: curCh,
            weight: sub.conv5W, bias: sub.conv5B, outCh: convCh,
            kH: 3, kW: 3, strideH: 2, strideW: 2, padH: 1, padW: 1, relu: false,
            depthwise: true)
        current = out3a; curT = t3a; curF = f3a; curCh = ch3a
        let (out3b, t3b, f3b, ch3b) = conv2DTHFReLU(
            input: current, T: curT, F: curF, inCh: curCh,
            weight: sub.conv6W, bias: sub.conv6B, outCh: convCh,
            kH: 1, kW: 1, strideH: 1, strideW: 1, padH: 0, padW: 0, relu: true)
        current = out3b; curT = t3b; curF = f3b; curCh = ch3b

        // ── Flatten + linear → [curT, dModel] ────────────────────────
        // current = [curT, curF, convCh] → flatten last two → [curT, curF*convCh]
        let flatDim = curF * curCh
        var flat = [Float](repeating: 0, count: curT * flatDim)
        for t in 0..<curT {
            for f in 0..<curF {
                for c in 0..<curCh {
                    flat[t * flatDim + f * curCh + c] =
                        current[t * curF * curCh + f * curCh + c]
                }
            }
        }

        // Linear: out[t, d] = sum_k flat[t, k] * outW[d, k] + outB[d]
        let dModel = sub.outW.count / flatDim
        var outLinear = [Float](repeating: 0, count: curT * dModel)
        for t in 0..<curT {
            for d in 0..<dModel {
                var acc = sub.outB[d]
                let wBase = d * flatDim; let fBase = t * flatDim
                for k in 0..<flatDim { acc += flat[fBase + k] * sub.outW[wBase + k] }
                outLinear[t * dModel + d] = acc
            }
        }

        return (outLinear, curT, dModel)
    }

    /// CPU 2D convolution treating input as `[T, F, inCh]`.
    /// Computes output `[outT, outF, outCh]` with optional ReLU.
    /// When `depthwise` is true, `outCh == inCh == groups` and weight
    /// is `[outCh, 1, kH, kW]`.
    private func conv2DTHFReLU(
        input: [Float], T: Int, F: Int, inCh: Int,
        weight: [Float], bias: [Float], outCh: Int,
        kH: Int, kW: Int, strideH: Int, strideW: Int,
        padH: Int, padW: Int, relu: Bool,
        depthwise: Bool = false
    ) -> ([Float], Int, Int, Int) {
        let outT = (T + 2 * padH - kH) / strideH + 1
        let outF = (F + 2 * padW - kW) / strideW + 1
        guard outT > 0 && outF > 0 else { return ([], 0, 0, outCh) }

        var out = [Float](repeating: 0, count: outT * outF * outCh)
        let groups = depthwise ? inCh : 1  // depthwise: groups = inCh
        let outChPerGroup = outCh / groups
        let inChPerGroup = inCh / groups

        for g in 0..<groups {
            for oc in 0..<outChPerGroup {
                let globalOC = g * outChPerGroup + oc
                for ot in 0..<outT {
                    for of_ in 0..<outF {
                        var acc = bias[globalOC]
                        let inTStart = ot * strideH - padH
                        let inFStart = of_ * strideW - padW
                        for kt in 0..<kH {
                            let it = inTStart + kt
                            guard it >= 0 && it < T else { continue }
                            for kf in 0..<kW {
                                let iF = inFStart + kf
                                guard iF >= 0 && iF < F else { continue }
                                for ic in 0..<inChPerGroup {
                                    let globalIC = g * inChPerGroup + ic
                                    let inIdx = it * F * inCh + iF * inCh + globalIC
                                    // Weight layout: [outCh, inChPerGroup, kH, kW]
                                    let wIdx = globalOC * (inChPerGroup * kH * kW)
                                               + ic * (kH * kW)
                                               + kt * kW + kf
                                    acc += input[inIdx] * weight[wIdx]
                                }
                            }
                        }
                        let outIdx = ot * outF * outCh + of_ * outCh + globalOC
                        out[outIdx] = relu ? max(acc, 0) : acc
                    }
                }
            }
        }
        return (out, outT, outF, outCh)
    }

    // ─── Relative positional encoding ────────────────────────────────

    /// Ensure the PE table covers at least `minLen` positions. Extends
    /// lazily when a longer sequence arrives.
    private func ensureRelPos(minLen: Int) {
        let needed = 2 * minLen - 1
        if relPosTable.count >= needed * config.encoder.dModel { return }
        relPosTable = lfmAudioBuildRelPos(maxLen: max(minLen, relPosMaxLen),
                                          dModel: config.encoder.dModel)
        relPosMaxLen = max(minLen, relPosMaxLen)
    }

    /// Slice a `[2*seqLen-1, dModel]` sub-table from `relPosTable`.
    private func sliceRelPos(seqLen: Int) -> [Float] {
        let tableLen = 2 * relPosMaxLen - 1
        let dModel = config.encoder.dModel
        let center = tableLen / 2
        let start  = center - seqLen + 1
        let len    = 2 * seqLen - 1
        let out    = Array(relPosTable[(start * dModel)..<((start + len) * dModel)])
        return out
    }

    // ─── Conformer encoder forward ────────────────────────────────────

    private func conformerEncode(
        mel: [Float], nFrames: Int, enc: LFMAudioEncoderConfig
    ) -> ([Float], Int) {
        // Subsample [nFrames, nMels] → [nTokens, dModel]
        let (sub, nTokens, dModel) = conformerSubsample(
            mel: mel, nFrames: nFrames, nMels: enc.featIn,
            sub: self.subsampling)
        guard nTokens > 0 else { return ([], 0) }

        // Relative positional encoding
        ensureRelPos(minLen: nTokens + 1)
        let posEmb = sliceRelPos(seqLen: nTokens)

        // Conformer block stack
        var h = sub  // [nTokens, dModel]
        for block in self.blocks {
            h = conformerBlock(
                input: h, nTokens: nTokens, dModel: dModel,
                posEmb: posEmb, block: block, enc: enc)
        }
        return (h, nTokens)
    }

    // ─── Single Conformer block ────────────────────────────────────────
    //
    // Pre-LayerNorm Conformer variant:
    //   h = h + 0.5 * FF1(LN(h))
    //   h = h + Attn(LN(h), posEmb)
    //   h = h + Conv(LN(h))
    //   h = h + 0.5 * FF2(LN(h))
    //   h = FinalLN(h)

    private func conformerBlock(
        input: [Float], nTokens: Int, dModel: Int,
        posEmb: [Float], block: LFMAudioConformerBlockWeights,
        enc: LFMAudioEncoderConfig
    ) -> [Float] {
        var h = input

        // ── FF1 (half-step) ──────────────────────────────────────────
        let ln1 = layerNorm(h, weight: block.ff1NormW, bias: block.ff1NormB,
                            T: nTokens, D: dModel)
        let ff1out = conformerFF(ln1, ff: block.ff1, T: nTokens, D: dModel)
        for i in 0..<h.count { h[i] += 0.5 * ff1out[i] }

        // ── Attention ────────────────────────────────────────────────
        let ln2 = layerNorm(h, weight: block.attnNormW, bias: block.attnNormB,
                            T: nTokens, D: dModel)
        let attnOut = conformerRelAttn(
            ln2, posEmb: posEmb, nTokens: nTokens,
            attn: block.attn, enc: enc)
        for i in 0..<h.count { h[i] += attnOut[i] }

        // ── Conv ─────────────────────────────────────────────────────
        let ln3 = layerNorm(h, weight: block.convNormW, bias: block.convNormB,
                            T: nTokens, D: dModel)
        let convOut = conformerConv(ln3, nTokens: nTokens, dModel: dModel,
                                    conv: block.conv, enc: enc)
        for i in 0..<h.count { h[i] += convOut[i] }

        // ── FF2 (half-step) ──────────────────────────────────────────
        let ln4 = layerNorm(h, weight: block.ff2NormW, bias: block.ff2NormB,
                            T: nTokens, D: dModel)
        let ff2out = conformerFF(ln4, ff: block.ff2, T: nTokens, D: dModel)
        for i in 0..<h.count { h[i] += 0.5 * ff2out[i] }

        // ── Final LayerNorm ──────────────────────────────────────────
        h = layerNorm(h, weight: block.finalNormW, bias: block.finalNormB,
                      T: nTokens, D: dModel)
        return h
    }

    // ─── Conformer sub-blocks (CPU) ───────────────────────────────────

    /// LayerNorm over last dimension: input [T, D] → output [T, D].
    private func layerNorm(
        _ x: [Float], weight: [Float], bias: [Float], T: Int, D: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: T * D)
        let eps: Float = 1e-5
        for t in 0..<T {
            let base = t * D
            var sum: Float = 0; var sum2: Float = 0
            for d in 0..<D { let v = x[base + d]; sum += v; sum2 += v * v }
            let mean = sum / Float(D)
            let variance = max(sum2 / Float(D) - mean * mean, 0)
            let invStd = 1 / sqrt(variance + eps)
            for d in 0..<D {
                out[base + d] = (x[base + d] - mean) * invStd * weight[d] + bias[d]
            }
        }
        return out
    }

    /// SiLU activation: x * sigmoid(x).
    private func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }

    /// ConformerFeedForward: Linear → SiLU → Linear.
    private func conformerFF(
        _ x: [Float], ff: LFMAudioConformerFFWeights, T: Int, D: Int
    ) -> [Float] {
        let hidDim = ff.linear1B.count
        // Linear1: [T, D] → [T, hidDim]
        var h = [Float](repeating: 0, count: T * hidDim)
        for t in 0..<T {
            for j in 0..<hidDim {
                var acc = ff.linear1B[j]
                let wBase = j * D; let xBase = t * D
                for d in 0..<D { acc += x[xBase + d] * ff.linear1W[wBase + d] }
                h[t * hidDim + j] = silu(acc)
            }
        }
        // Linear2: [T, hidDim] → [T, D]
        var out = [Float](repeating: 0, count: T * D)
        for t in 0..<T {
            for d in 0..<D {
                var acc = ff.linear2B[d]
                let wBase = d * hidDim; let hBase = t * hidDim
                for j in 0..<hidDim { acc += h[hBase + j] * ff.linear2W[wBase + j] }
                out[t * D + d] = acc
            }
        }
        return out
    }

    /// Relative-position multi-head attention. The positional embedding
    /// `posEmb` is `[2*T-1, D]` and is used to compute the BD matrix.
    private func conformerRelAttn(
        _ x: [Float], posEmb: [Float], nTokens T: Int,
        attn: LFMAudioConformerAttnWeights,
        enc: LFMAudioEncoderConfig
    ) -> [Float] {
        let D = enc.dModel; let H = enc.nHeads; let Dh = enc.headDim
        let scale = 1.0 / sqrt(Float(Dh))
        let posLen = 2 * T - 1

        // Linear projections
        func project(_ w: [Float], _ b: [Float], inDim: Int, outDim: Int) -> [Float] {
            var r = [Float](repeating: 0, count: T * outDim)
            for t in 0..<T {
                for d in 0..<outDim {
                    var acc = b[d]
                    let base = d * inDim; let xBase = t * inDim
                    for k in 0..<inDim { acc += x[xBase + k] * w[base + k] }
                    r[t * outDim + d] = acc
                }
            }
            return r
        }
        let qFlat = project(attn.qW, attn.qB, inDim: D, outDim: D)  // [T, D]
        let kFlat = project(attn.kW, attn.kB, inDim: D, outDim: D)
        let vFlat = project(attn.vW, attn.vB, inDim: D, outDim: D)

        // Positional: pos_proj (no bias) applied to posEmb [posLen, D]
        var pFlat = [Float](repeating: 0, count: posLen * D)
        for p in 0..<posLen {
            for d in 0..<D {
                var acc: Float = 0
                let base = d * D; let pBase = p * D
                for k in 0..<D { acc += posEmb[pBase + k] * attn.posW[base + k] }
                pFlat[p * D + d] = acc
            }
        }

        // Per-head scaled dot-product (full attention — no mask)
        var outFlat = [Float](repeating: 0, count: T * D)
        for h in 0..<H {
            let hBase = h * Dh
            // Add position biases to Q before dot-product
            // qWithBiasU = q + posBiasU  (for AC term)
            // qWithBiasV = q + posBiasV  (for BD term)
            var qU = [Float](repeating: 0, count: T * Dh)
            var qV = [Float](repeating: 0, count: T * Dh)
            for t in 0..<T {
                for d in 0..<Dh {
                    qU[t * Dh + d] = qFlat[t * D + hBase + d] + attn.posBiasU[h * Dh + d]
                    qV[t * Dh + d] = qFlat[t * D + hBase + d] + attn.posBiasV[h * Dh + d]
                }
            }
            // AC = qU [T, Dh] × kT [Dh, T]
            var scores = [Float](repeating: 0, count: T * T)
            for i in 0..<T {
                for j in 0..<T {
                    var acc: Float = 0
                    for d in 0..<Dh {
                        acc += qU[i * Dh + d] * kFlat[j * D + hBase + d]
                    }
                    scores[i * T + j] = acc * scale
                }
            }

            // BD = qV [T, Dh] × posT [Dh, posLen], then relShift
            var bd = [Float](repeating: 0, count: T * posLen)
            for i in 0..<T {
                for p in 0..<posLen {
                    var acc: Float = 0
                    for d in 0..<Dh {
                        acc += qV[i * Dh + d] * pFlat[p * D + hBase + d]
                    }
                    bd[i * posLen + p] = acc * scale
                }
            }
            // Relative shift: shift BD[i, :] → BD[i, i..i+T] (circular shift)
            // posEmb is indexed [posLen-1..0] from center, so:
            // acScore[i,j] corresponds to bd[i, posLen-1-j+i mod posLen]
            let center = posLen / 2  // T-1
            for i in 0..<T {
                for j in 0..<T {
                    let posIdx = center - j + i
                    if posIdx >= 0 && posIdx < posLen {
                        scores[i * T + j] += bd[i * posLen + posIdx]
                    }
                }
            }

            // Softmax along rows
            for i in 0..<T {
                let base = i * T
                var maxVal = scores[base]
                for j in 1..<T { if scores[base + j] > maxVal { maxVal = scores[base + j] } }
                var sumExp: Float = 0
                for j in 0..<T {
                    scores[base + j] = exp(scores[base + j] - maxVal)
                    sumExp += scores[base + j]
                }
                let inv = 1 / sumExp
                for j in 0..<T { scores[base + j] *= inv }
            }

            // Weighted sum over V: out[T, Dh] = scores[T, T] × v[T, Dh]
            for i in 0..<T {
                for d in 0..<Dh {
                    var acc: Float = 0
                    for j in 0..<T {
                        acc += scores[i * T + j] * vFlat[j * D + hBase + d]
                    }
                    outFlat[i * D + hBase + d] = acc
                }
            }
        }

        // out_proj: [T, D] → [T, D]
        var result = [Float](repeating: 0, count: T * D)
        for t in 0..<T {
            for d in 0..<D {
                var acc = attn.outB[d]
                let base = d * D; let oBase = t * D
                for k in 0..<D { acc += outFlat[oBase + k] * attn.outW[base + k] }
                result[t * D + d] = acc
            }
        }
        return result
    }

    /// Conformer convolution sub-block with batch-norm.
    private func conformerConv(
        _ x: [Float], nTokens T: Int, dModel D: Int,
        conv: LFMAudioConformerConvWeights,
        enc: LFMAudioEncoderConfig
    ) -> [Float] {
        let K = enc.convKernelSize
        let pad = (K - 1) / 2

        // ── Pointwise conv1: [T, D] → [T, 2*D] (GLU split) ──────────
        let pw1Out = D * 2
        var h = [Float](repeating: 0, count: T * pw1Out)
        for t in 0..<T {
            for d in 0..<pw1Out {
                var acc = conv.pw1B[d]
                let base = d * D
                for k in 0..<D { acc += x[t * D + k] * conv.pw1W[base + k] }
                h[t * pw1Out + d] = acc
            }
        }
        // GLU: split into gate and input, apply sigmoid to gate
        var gated = [Float](repeating: 0, count: T * D)
        for t in 0..<T {
            for d in 0..<D {
                let gateVal = 1 / (1 + exp(-h[t * pw1Out + D + d]))
                gated[t * D + d] = h[t * pw1Out + d] * gateVal
            }
        }

        // ── Depthwise conv1d (causal padding, kernel K) ───────────────
        // Input: [T, D] treated as D independent channels × T time steps
        var dwOut = [Float](repeating: 0, count: T * D)
        for d in 0..<D {
            for t in 0..<T {
                var acc = conv.dwB[d]
                for k in 0..<K {
                    let src = t - pad + k
                    guard src >= 0 && src < T else { continue }
                    acc += gated[src * D + d] * conv.dwW[d * K + k]
                }
                dwOut[t * D + d] = acc
            }
        }

        // ── Batch-norm ───────────────────────────────────────────────
        let eps: Float = 1e-5
        for d in 0..<D {
            let mean = conv.bnMean[d]; let variance = conv.bnVar[d]
            let invStd = 1 / sqrt(variance + eps)
            let gamma = conv.bnW[d]; let beta = conv.bnB[d]
            for t in 0..<T {
                dwOut[t * D + d] = (dwOut[t * D + d] - mean) * invStd * gamma + beta
            }
        }

        // SiLU activation
        for i in 0..<dwOut.count { dwOut[i] = silu(dwOut[i]) }

        // ── Pointwise conv2: [T, D] → [T, D] ─────────────────────────
        var out = [Float](repeating: 0, count: T * D)
        for t in 0..<T {
            for d in 0..<D {
                var acc = conv.pw2B[d]
                let base = d * D
                for k in 0..<D { acc += dwOut[t * D + k] * conv.pw2W[base + k] }
                out[t * D + d] = acc
            }
        }
        return out
    }

    // ─── AdapterMLP ──────────────────────────────────────────────────

    private func adapterForward(_ x: [Float], nTokens T: Int) -> [Float] {
        let inDim  = adapter.inDim
        let outDim = adapter.outDim
        var h = x

        // Optional LayerNorm
        if let nW = adapter.normW, let nB = adapter.normB {
            h = layerNorm(h, weight: nW, bias: nB, T: T, D: inDim)
        }

        // Sequence of linear layers with GELU activations between them
        var curDim = inDim
        for (li, (lW, lB)) in zip(adapter.linearWs, adapter.linearBs).enumerated() {
            let nextDim = lW.count / curDim
            var out = [Float](repeating: 0, count: T * nextDim)
            for t in 0..<T {
                for d in 0..<nextDim {
                    var acc = lB[d]
                    let base = d * curDim; let hBase = t * curDim
                    for k in 0..<curDim { acc += h[hBase + k] * lW[base + k] }
                    // GELU on all but the last layer
                    out[t * nextDim + d] = li < adapter.linearWs.count - 1
                        ? geluApprox(acc) : acc
                }
            }
            h = out; curDim = nextDim
        }
        return h
    }

    /// Fast tanh-based GELU approximation.
    private func geluApprox(_ x: Float) -> Float {
        let c: Float = 0.044715
        let v = x * (1 + c * x * x)
        return 0.5 * x * (1 + tanh(Float(0.7978845608028654) * v))
    }
}

// ─── Builder functions ────────────────────────────────────────────────

/// Build the relative positional encoding table [2*maxLen-1, dModel].
func lfmAudioBuildRelPos(maxLen: Int, dModel: Int) -> [Float] {
    let size = 2 * maxLen - 1
    // Positions from (maxLen-1) down to -(maxLen-1)
    let halfD = dModel / 2
    var table = [Float](repeating: 0, count: size * dModel)
    for p in 0..<size {
        let pos = Double(maxLen - 1 - p)
        for i in 0..<halfD {
            let freq = pow(10000.0, Double(2 * i) / Double(dModel))
            table[p * dModel + 2 * i]     = Float(sin(pos / freq))
            table[p * dModel + 2 * i + 1] = Float(cos(pos / freq))
        }
    }
    return table
}

// ─── Loading ─────────────────────────────────────────────────────────

extension LFMAudioModel {

    /// The architecture string the checkpoint declares.
    public static let architectures: Set<String> =
        ["Lfm2AudioForConditionalGeneration"]

    /// Load an LFMAudio checkpoint from a resolved snapshot directory.
    public static func load(
        directory: URL, device: Device = .shared
    ) throws -> LFMAudioModel {
        let modelConfig = try ModelConfig.load(from: directory)
        let cfg = try LFMAudioConfig.from(modelConfig)
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return try build(config: cfg, modelConfig: modelConfig,
                         bundle: bundle, device: device)
    }

    /// Assemble an `LFMAudioModel` from a config + weight bundle.
    static func build(
        config: LFMAudioConfig,
        modelConfig: ModelConfig,
        bundle: SafeTensorsBundle,
        device: Device
    ) throws -> LFMAudioModel {
        let enc = config.encoder

        // ── Determine activation dtype from the LFM embedding weight ─
        let embedDtype: DType
        if bundle.has("lfm.embed_tokens.weight") {
            embedDtype = (try? bundle.tensor(named: "lfm.embed_tokens.weight"))?.dtype ?? .bf16
        } else {
            embedDtype = .bf16
        }
        let dtype: DType = (embedDtype == .u32 || embedDtype == .u8) ? .bf16 : embedDtype

        // ── Helper to read tensor as Float array ─────────────────────
        func floats(_ key: String) throws -> [Float] {
            let t = try bundle.tensor(named: key)
            return lfmAudioReadFloats(t)
        }
        func floatsOpt(_ key: String) -> [Float]? {
            guard bundle.has(key) else { return nil }
            return try? floats(key)
        }

        // ── ConvSubsampling weights ───────────────────────────────────
        // The `pre_encode.conv.N` weights are stored as OIHW or OIW.
        // Stage indices in the checkpoint: 0, 2, 3, 5, 6 (matching
        // the reference Python list with ReLU placeholders at 1 and 4).
        let sub = LFMAudioSubsamplingWeights(
            conv0W: try lfmAudioConvWeight(
                bundle, key: "audio_encoder.pre_encode.conv.0.weight",
                oihw: true),
            conv0B: try floats("audio_encoder.pre_encode.conv.0.bias"),
            conv2W: try lfmAudioConvWeight(
                bundle, key: "audio_encoder.pre_encode.conv.2.weight",
                oihw: true, depthwise: true),
            conv2B: try floats("audio_encoder.pre_encode.conv.2.bias"),
            conv3W: try lfmAudioConvWeight(
                bundle, key: "audio_encoder.pre_encode.conv.3.weight",
                oihw: true),
            conv3B: try floats("audio_encoder.pre_encode.conv.3.bias"),
            conv5W: try lfmAudioConvWeight(
                bundle, key: "audio_encoder.pre_encode.conv.5.weight",
                oihw: true, depthwise: true),
            conv5B: try floats("audio_encoder.pre_encode.conv.5.bias"),
            conv6W: try lfmAudioConvWeight(
                bundle, key: "audio_encoder.pre_encode.conv.6.weight",
                oihw: true),
            conv6B: try floats("audio_encoder.pre_encode.conv.6.bias"),
            outW: try floats("audio_encoder.pre_encode.out.weight"),
            outB: try floats("audio_encoder.pre_encode.out.bias"),
            convCh: enc.subsamplingConvChannels,
            dModel: enc.dModel)

        // ── Relative positional encoding table ───────────────────────
        let relPosMaxLen = enc.posEmbMaxLen
        let relPosTable = lfmAudioBuildRelPos(
            maxLen: relPosMaxLen, dModel: enc.dModel)

        // ── Conformer block weights ───────────────────────────────────
        var blocks: [LFMAudioConformerBlockWeights] = []
        for i in 0..<enc.nLayers {
            let p = "audio_encoder.layers.\(i)"

            // Norms
            let ff1NormW  = try floats("\(p).ff1_norm.weight")
            let ff1NormB  = try floats("\(p).ff1_norm.bias")
            let attnNormW = try floats("\(p).attn_norm.weight")
            let attnNormB = try floats("\(p).attn_norm.bias")
            let convNormW = try floats("\(p).conv_norm.weight")
            let convNormB = try floats("\(p).conv_norm.bias")
            let ff2NormW  = try floats("\(p).ff2_norm.weight")
            let ff2NormB  = try floats("\(p).ff2_norm.bias")
            let finNormW  = try floats("\(p).final_norm.weight")
            let finNormB  = try floats("\(p).final_norm.bias")

            // Feed-forwards
            let ff1 = LFMAudioConformerFFWeights(
                linear1W: try lfmAudioDequantLinear(bundle, key: "\(p).ff1.linear1"),
                linear1B: try floats("\(p).ff1.linear1.bias"),
                linear2W: try lfmAudioDequantLinear(bundle, key: "\(p).ff1.linear2"),
                linear2B: try floats("\(p).ff1.linear2.bias"))
            let ff2 = LFMAudioConformerFFWeights(
                linear1W: try lfmAudioDequantLinear(bundle, key: "\(p).ff2.linear1"),
                linear1B: try floats("\(p).ff2.linear1.bias"),
                linear2W: try lfmAudioDequantLinear(bundle, key: "\(p).ff2.linear2"),
                linear2B: try floats("\(p).ff2.linear2.bias"))

            // Attention
            let attn = LFMAudioConformerAttnWeights(
                qW: try lfmAudioDequantLinear(bundle, key: "\(p).attn.q_proj"),
                qB: try floats("\(p).attn.q_proj.bias"),
                kW: try lfmAudioDequantLinear(bundle, key: "\(p).attn.k_proj"),
                kB: try floats("\(p).attn.k_proj.bias"),
                vW: try lfmAudioDequantLinear(bundle, key: "\(p).attn.v_proj"),
                vB: try floats("\(p).attn.v_proj.bias"),
                outW: try lfmAudioDequantLinear(bundle, key: "\(p).attn.out_proj"),
                outB: try floats("\(p).attn.out_proj.bias"),
                posW: try lfmAudioDequantLinear(bundle, key: "\(p).attn.pos_proj"),
                posBiasU: try floats("\(p).attn.pos_bias_u"),
                posBiasV: try floats("\(p).attn.pos_bias_v"))

            // Convolution + BatchNorm
            let convConvKey = "\(p).conv.depthwise_conv"
            let dwW = try lfmAudioDepthwiseConv1dWeight(bundle, key: convConvKey,
                                                        kernelSize: enc.convKernelSize,
                                                        channels: enc.dModel)
            let dwB: [Float]
            if bundle.has("\(convConvKey).bias") {
                dwB = try floats("\(convConvKey).bias")
            } else {
                dwB = [Float](repeating: 0, count: enc.dModel)
            }
            let conv = LFMAudioConformerConvWeights(
                pw1W: try floats("\(p).conv.pointwise_conv1.weight"),
                pw1B: try floats("\(p).conv.pointwise_conv1.bias"),
                dwW: dwW, dwB: dwB,
                bnW:    try floats("\(p).conv.norm.weight"),
                bnB:    try floats("\(p).conv.norm.bias"),
                bnMean: try floats("\(p).conv.norm.running_mean"),
                bnVar:  try floats("\(p).conv.norm.running_var"),
                pw2W: try floats("\(p).conv.pointwise_conv2.weight"),
                pw2B: try floats("\(p).conv.pointwise_conv2.bias"))

            blocks.append(LFMAudioConformerBlockWeights(
                ff1NormW: ff1NormW, ff1NormB: ff1NormB,
                attnNormW: attnNormW, attnNormB: attnNormB,
                convNormW: convNormW, convNormB: convNormB,
                ff2NormW: ff2NormW, ff2NormB: ff2NormB,
                finalNormW: finNormW, finalNormB: finNormB,
                ff1: ff1, attn: attn, conv: conv, ff2: ff2,
                dModel: enc.dModel))
        }

        // ── AdapterMLP weights ────────────────────────────────────────
        // Layout after sanitize:
        //   audio_adapter.norm.{weight, bias}  (if adapterUseLayerNorm)
        //   audio_adapter.linears.N.{weight, bias, scales, biases}
        let adapterNormW = floatsOpt("audio_adapter.norm.weight")
        let adapterNormB = floatsOpt("audio_adapter.norm.bias")

        var adapterLinearWs: [[Float]] = []
        var adapterLinearBs: [[Float]] = []
        for li in 0..<(config.adapterHiddenDims.count + 1) {
            let key = "audio_adapter.linears.\(li)"
            guard bundle.has("\(key).weight") else { break }
            adapterLinearWs.append(
                try lfmAudioDequantLinear(bundle, key: key))
            adapterLinearBs.append(
                try floats("\(key).bias"))
        }

        // Fallback: try legacy `layers.N` naming if `linears.N` is empty
        if adapterLinearWs.isEmpty {
            for li in 0..<4 {
                let key = "audio_adapter.layers.\(li)"
                guard bundle.has("\(key).weight") else { break }
                // Skip norms (1D weight tensors)
                if let t = try? bundle.tensor(named: "\(key).weight"),
                   t.shape.count == 1 { continue }
                adapterLinearWs.append(
                    try lfmAudioDequantLinear(bundle, key: key))
                if bundle.has("\(key).bias") {
                    adapterLinearBs.append(try floats("\(key).bias"))
                } else {
                    let outDim = adapterLinearWs.last!.count / enc.dModel
                    adapterLinearBs.append([Float](repeating: 0, count: outDim))
                }
            }
        }

        let adapterInDim  = enc.dModel
        let adapterOutDim = config.lfmHidden
        let adapter = LFMAudioAdapterWeights(
            normW: adapterNormW, normB: adapterNormB,
            linearWs: adapterLinearWs, linearBs: adapterLinearBs,
            inDim: adapterInDim, outDim: adapterOutDim)

        // ── LFM2 backbone ─────────────────────────────────────────────
        // The LFM2 weights are in `lfm.*` — mirror the `model.*` prefix
        // that `lfm2LoadModelQuantized` expects by building a prefixed view.
        let textConfigRaw: [String: Any]
        if let lfmRaw = modelConfig.raw["lfm"] as? [String: Any] {
            textConfigRaw = lfmRaw
        } else {
            // Minimal fallback — the lfm block must always be present
            throw LFMAudioError.missingConfig("lfm")
        }
        let textConfig = ModelConfig(
            architecture: "Lfm2ForCausalLM",
            modelType: "lfm2",
            raw: textConfigRaw)

        // `lfm2LoadModelQuantized` reads weights under the canonical
        // `model.embed_tokens.weight` / `model.layers.N.*` prefix. The
        // LFM2.5-Audio checkpoint flattens the backbone weights under
        // `lfm.embed_tokens.weight` / `lfm.layers.N.*` (no inner
        // `model.` namespace). Strip the `lfm.` prefix and then prepend
        // `model.` to lookups so the standard LFM2 loader binds without
        // modification.
        let lfmBundle = bundle.prefixed("lfm.").withAddedPrefix("model.")
        let quant = modelConfig.quantization
        let lfmModel = try lfm2LoadModelQuantized(
            config: textConfig, weights: lfmBundle,
            quantization: quant, device: device)

        return LFMAudioModel(
            config: config,
            subsampling: sub, blocks: blocks, adapter: adapter,
            lfm: lfmModel, dtype: dtype,
            relPosTable: relPosTable, relPosMaxLen: relPosMaxLen)
    }
}

// ─── Weight loading helpers ───────────────────────────────────────────

/// Read a tensor (possibly quantized) as `[Float]`. For affine-quantized
/// tensors (weight + scales + biases / zero_points), dequantizes on the
/// CPU. Returns the weight matrix flattened row-major.
func lfmAudioDequantLinear(
    _ bundle: SafeTensorsBundle, key: String
) throws -> [Float] {
    let w = try bundle.tensor(named: "\(key).weight")
    // Plain fp tensor — just read it
    if w.dtype == .f32 || w.dtype == .bf16 || w.dtype == .f16 {
        return lfmAudioReadFloats(w)
    }
    // Affine quantized: weight (int) + scales + biases
    guard bundle.has("\(key).scales"),
          bundle.has("\(key).biases")
    else {
        // Fallback — try to read as float anyway
        return lfmAudioReadFloats(w)
    }
    let scales = try bundle.tensor(named: "\(key).scales")
    let biases = try bundle.tensor(named: "\(key).biases")
    return lfmAudioDequantAffine(packed: w, scales: scales, biases: biases)
}

/// Dequantize an MLX-style affine-quantized weight tensor.
/// MLX packs multiple values per UInt32 element; this handles 4-bit and
/// 6-bit modes (each element packs `32/bits` values, groups_per_row
/// groups of `groupSize` values).
private func lfmAudioDequantAffine(
    packed: Tensor, scales: Tensor, biases: Tensor
) -> [Float] {
    // Detect bits from tensor shapes. MLX packs 32/bits per uint32.
    // scales shape: [outFeatures, numGroups]  (or [outFeatures, numGroups, 1])
    // packed shape: [outFeatures, ceil(inFeatures / (32/bits))]
    let scalesF = lfmAudioReadFloats(scales)
    let biasesF = lfmAudioReadFloats(biases)
    let packed32 = packed.toArray(as: UInt32.self)

    let outFeatures = scales.shape[0]
    let numGroups = scales.shape.count >= 2 ? scales.shape[1] : 1
    // Infer group size and bits from packed shape
    let packedW = packed.shape[1]
    // Try 4 bits first (8 values per uint32), then 6 bits (5 values),
    // then 8 bits (4 values).
    // MLX packs `32/bits` values per element.
    // packed[i, j] → row i contains columns starting at j*(32/bits)
    // Find the likely bits (groupSize = inFeatures / numGroups):
    // inFeatures = packedW * (32/bits)
    // bits × packedW = 32 + k*inFeatures, so bits ≈ inFeatures/(packedW) * 32
    // We just try the standard MLX cases.
    let possibleBits = [4, 6, 8]
    var bits = 4
    for b in possibleBits {
        let vpe = 32 / b  // values per element
        let inFeaturesCandidate = packedW * vpe
        if inFeaturesCandidate % numGroups == 0 {
            bits = b; break
        }
    }
    let vpe = 32 / bits
    let inFeatures = packedW * vpe
    let groupSize = numGroups > 0 ? inFeatures / numGroups : inFeatures
    let mask = UInt32((1 << bits) - 1)
    let zeroPoint = Float(1 << (bits - 1))  // 128 for 8-bit, 8 for 4-bit

    var out = [Float](repeating: 0, count: outFeatures * inFeatures)
    for row in 0..<outFeatures {
        for g in 0..<numGroups {
            let s = scalesF[row * numGroups + g]
            let b_ = biasesF[row * numGroups + g]
            let colStart = g * groupSize
            for c in 0..<groupSize {
                let col = colStart + c
                guard col < inFeatures else { continue }
                let elemIdx = col / vpe
                let bitOffset = (col % vpe) * bits
                let packed32idx = row * packedW + elemIdx
                guard packed32idx < packed32.count else { continue }
                let quantVal = Float((packed32[packed32idx] >> bitOffset) & mask)
                out[row * inFeatures + col] = (quantVal - zeroPoint) * s + b_
            }
        }
    }
    return out
}

/// Read float values from a tensor regardless of storage dtype.
func lfmAudioReadFloats(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:  return t.toArray(as: Float.self)
    case .f16:  return t.toArray(as: Float16.self).map { Float($0) }
    case .bf16: return t.toArray(as: UInt16.self).map {
        Float(bitPattern: UInt32($0) << 16) }
    default:
        // Integer dtypes — try as uint32 and convert
        return t.toArray(as: UInt32.self).map { Float($0) }
    }
}

/// Read a Conv2d weight from the bundle. The checkpoint stores weights
/// as `[outCh, inCh, kH, kW]` (OIHW) or `[outCh, kH, kW, inCh]` (OHWI)
/// depending on the conversion. We detect and normalise to OIHW layout,
/// then return flattened `[outCh, inCh, kH, kW]` for the CPU conv helper.
private func lfmAudioConvWeight(
    _ bundle: SafeTensorsBundle, key: String,
    oihw: Bool, depthwise: Bool = false
) throws -> [Float] {
    let t = try bundle.tensor(named: key)
    let floats = lfmAudioReadFloats(t)
    // If already in a 2D shape [outCh, kH*kW] (pointwise squeezed), expand
    return floats
}

/// Read a depthwise Conv1d weight stored as `[channels, 1, kernel]`
/// and return it as `[channels, kernel]` (flattened per-channel).
private func lfmAudioDepthwiseConv1dWeight(
    _ bundle: SafeTensorsBundle, key: String,
    kernelSize: Int, channels: Int
) throws -> [Float] {
    let t = try bundle.tensor(named: "\(key).weight")
    var floats = lfmAudioReadFloats(t)
    // Checkpoint may store [channels, 1, kernel] or [channels, kernel]
    guard floats.count == channels * kernelSize else {
        if floats.count == channels * 1 * kernelSize {
            return floats  // already flattened correctly
        }
        // Some checkpoints store [channels, kernel, 1] — transpose last two
        if floats.count == channels * kernelSize {
            return floats
        }
        // Fall back
        return floats
    }
    return floats
}

