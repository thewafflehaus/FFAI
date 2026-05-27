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
// Parakeet — NVIDIA's Conformer-TDT speech-to-text family.
//
// Architecture overview (NeMo EncDecRNNTBPEModel with TDT decoding):
//
//   waveform ──NeMo STFT Mel front-end──▶ [T, nMels]
//   ──DwStridingSubsampling──▶  [T/8, d_model]
//   ──24 × ConformerBlock (rel_pos attention + conv + FF)──▶
//   ──Prediction Network (Embedding + StackedLSTM)──▶
//   ──Joint Network (enc proj + pred proj + ReLU + output proj)──▶
//   ──Greedy TDT decoding──▶ tokens ──vocabulary──▶ text
//
// This file handles both V2 (vocab 1024) and V3 (vocab 8192) checkpoints
// transparently — they share the same encoder architecture, differing
// only in vocabulary size.
//
// All computation runs CPU-side via [Float] arrays. The Conformer stack,
// LSTM prediction network, and joint network are implemented in pure
// Swift with Accelerate/vDSP where available. Weights are loaded from
// safetensors shards via SafeTensorsBundle; the Tensor type is used
// exclusively for storage, and `.toArray(as: Float.self)` materialises
// values for CPU arithmetic.
//
// Detection marker:  config.json must contain a top-level
// `"model_defaults"` block with `"tdt_durations"` **and**
// an `"encoder"` block with `"feat_in"` — a signature no other
// FFAI-registered model family carries.

import Accelerate
import Foundation
import Metal

// ─── Configuration ───────────────────────────────────────────────────

/// NeMo AudioToMelSpectrogramPreprocessor hyper-parameters.
/// The STFT parameters differ from the Whisper front-end:
/// sample_rate 16 kHz, window 25 ms (400 samples), stride 10 ms (160 samples),
/// 128 Mel bins, 512-point FFT, per-feature normalisation.
public struct ParakeetPreprocessorConfig: Sendable {
    public let sampleRate: Int
    public let nMels: Int  // "features"
    public let nFFT: Int
    public let winLength: Int  // derived: window_size * sample_rate
    public let hopLength: Int  // derived: window_stride * sample_rate
    public let preemph: Float
    public let logZeroGuardValue: Float
    /// "per_feature" means z-score each Mel bin across time; else global.
    public let normalise: String
}

/// Conformer encoder hyper-parameters.
public struct ParakeetEncoderConfig: Sendable {
    public let featIn: Int
    public let nLayers: Int
    public let dModel: Int
    public let nHeads: Int
    public let ffExpansionFactor: Int
    public let subsamplingFactor: Int
    public let subsamplingConvChannels: Int
    public let posEmbMaxLen: Int
    public let convKernelSize: Int
    public let useBias: Bool

    /// Per-head dimension (dModel / nHeads).
    public var headDim: Int { dModel / nHeads }
    /// Feed-forward hidden size.
    public var ffHidden: Int { dModel * ffExpansionFactor }
}

/// Prediction network (RNNT decoder) hyper-parameters.
public struct ParakeetPredNetConfig: Sendable {
    public let predHidden: Int
    public let predRnnLayers: Int
    public let rnnHiddenSize: Int
    public let blankAsPad: Bool
    public let vocabSize: Int  // embedding table size (vocab + 1 when blankAsPad)
}

/// Joint network hyper-parameters.
public struct ParakeetJointConfig: Sendable {
    public let encoderHidden: Int
    public let predHidden: Int
    public let jointHidden: Int
    public let numClasses: Int  // vocab + 1
    public let numExtraOutputs: Int  // TDT duration outputs (5 for 0-4)
}

/// Resolved Parakeet configuration. Both V2 and V3 decode into this struct.
public struct ParakeetConfig: Sendable {
    public let preprocessor: ParakeetPreprocessorConfig
    public let encoder: ParakeetEncoderConfig
    public let predNet: ParakeetPredNetConfig
    public let joint: ParakeetJointConfig
    public let vocabulary: [String]
    public let tdtDurations: [Int]
    public let maxSymbolsPerStep: Int?

    /// The blank token index is always `vocabulary.count` (one past the end).
    public var blankTokenId: Int { vocabulary.count }
}

// ─── Config parsing ──────────────────────────────────────────────────

public enum ParakeetConfigError: Error, LocalizedError {
    case missingPreprocessor
    case missingEncoder
    case missingDecoder
    case missingJoint
    case missingModelDefaults
    case missingTDTDurations
    case missingVocabulary

    public var errorDescription: String? {
        switch self {
        case .missingPreprocessor: return "Parakeet config: missing 'preprocessor' block"
        case .missingEncoder: return "Parakeet config: missing 'encoder' block"
        case .missingDecoder: return "Parakeet config: missing 'decoder' block"
        case .missingJoint: return "Parakeet config: missing 'joint' block"
        case .missingModelDefaults: return "Parakeet config: missing 'model_defaults' block"
        case .missingTDTDurations:
            return "Parakeet config: missing 'tdt_durations' in model_defaults"
        case .missingVocabulary: return "Parakeet config: missing 'vocabulary' in joint block"
        }
    }
}

extension ParakeetConfig {
    /// Decode a resolved checkpoint `config.json` into a `ParakeetConfig`.
    /// Both V2 and V3 are handled by the same parser — they differ only
    /// in vocab_size (1024 vs 8192) and the vocabulary list.
    public static func from(_ config: ModelConfig) throws -> ParakeetConfig {
        guard let prep = config.raw["preprocessor"] as? [String: Any]
        else { throw ParakeetConfigError.missingPreprocessor }
        guard let enc = config.raw["encoder"] as? [String: Any]
        else { throw ParakeetConfigError.missingEncoder }
        guard let dec = config.raw["decoder"] as? [String: Any]
        else { throw ParakeetConfigError.missingDecoder }
        guard let jnt = config.raw["joint"] as? [String: Any]
        else { throw ParakeetConfigError.missingJoint }
        guard let modelDefaults = config.raw["model_defaults"] as? [String: Any]
        else { throw ParakeetConfigError.missingModelDefaults }
        guard let tdtDurations = modelDefaults["tdt_durations"] as? [Int]
        else { throw ParakeetConfigError.missingTDTDurations }

        // ── Preprocessor ────────────────────────────────────────────
        let sr = prep["sample_rate"] as? Int ?? 16_000
        let nMels = prep["features"] as? Int ?? 128
        let nFFT = prep["n_fft"] as? Int ?? 512
        let windowSize = prep["window_size"] as? Double ?? 0.025
        let windowStride = prep["window_stride"] as? Double ?? 0.01
        let winLength = Int((windowSize * Double(sr)).rounded())
        let hopLength = Int((windowStride * Double(sr)).rounded())
        let preemph = Float(prep["preemph"] as? Double ?? 0.97)
        let logGuard = Float(prep["log_zero_guard_value"] as? Double ?? pow(2.0, -24.0))
        let normalise = prep["normalize"] as? String ?? "per_feature"

        // ── Encoder ─────────────────────────────────────────────────
        let featIn = enc["feat_in"] as? Int ?? nMels
        let nLayers = enc["n_layers"] as? Int ?? 24
        let dModel = enc["d_model"] as? Int ?? 1024
        let nHeads = enc["n_heads"] as? Int ?? 8
        let ffExpansion = enc["ff_expansion_factor"] as? Int ?? 4
        let subFactor = enc["subsampling_factor"] as? Int ?? 8
        let subConvCh = enc["subsampling_conv_channels"] as? Int ?? 256
        let posEmbMaxLen = enc["pos_emb_max_len"] as? Int ?? 5000
        let convKernel = enc["conv_kernel_size"] as? Int ?? 9
        let useBias = enc["use_bias"] as? Bool ?? false

        // ── Prediction network (decoder in NeMo parlance) ────────────
        let prednet = dec["prednet"] as? [String: Any] ?? [:]
        let predHidden = prednet["pred_hidden"] as? Int ?? 640
        let predRnnLayers = prednet["pred_rnn_layers"] as? Int ?? 2
        let rnnHiddenSize = prednet["rnn_hidden_size"] as? Int ?? predHidden
        let blankAsPad = dec["blank_as_pad"] as? Bool ?? true
        let vocabSize = dec["vocab_size"] as? Int ?? 1024

        // ── Joint network ────────────────────────────────────────────
        guard let vocabulary = jnt["vocabulary"] as? [String]
        else { throw ParakeetConfigError.missingVocabulary }
        let jointnetRaw = jnt["jointnet"] as? [String: Any] ?? [:]
        let jointHidden = jointnetRaw["joint_hidden"] as? Int ?? predHidden
        let numClasses = jnt["num_classes"] as? Int ?? vocabSize
        let numExtraOutputs = jnt["num_extra_outputs"] as? Int ?? 0
        let encoderHidden = jointnetRaw["encoder_hidden"] as? Int ?? dModel
        let predHiddenJ = jointnetRaw["pred_hidden"] as? Int ?? predHidden

        // ── Greedy config ────────────────────────────────────────────
        let decodingRaw = config.raw["decoding"] as? [String: Any]
        let greedyRaw = decodingRaw?["greedy"] as? [String: Any]
        let maxSymbols = greedyRaw?["max_symbols"] as? Int

        return ParakeetConfig(
            preprocessor: ParakeetPreprocessorConfig(
                sampleRate: sr,
                nMels: nMels,
                nFFT: nFFT,
                winLength: winLength,
                hopLength: hopLength,
                preemph: preemph,
                logZeroGuardValue: logGuard,
                normalise: normalise
            ),
            encoder: ParakeetEncoderConfig(
                featIn: featIn,
                nLayers: nLayers,
                dModel: dModel,
                nHeads: nHeads,
                ffExpansionFactor: ffExpansion,
                subsamplingFactor: subFactor,
                subsamplingConvChannels: subConvCh,
                posEmbMaxLen: posEmbMaxLen,
                convKernelSize: convKernel,
                useBias: useBias
            ),
            predNet: ParakeetPredNetConfig(
                predHidden: predHidden,
                predRnnLayers: predRnnLayers,
                rnnHiddenSize: rnnHiddenSize,
                blankAsPad: blankAsPad,
                vocabSize: blankAsPad ? vocabSize + 1 : vocabSize
            ),
            joint: ParakeetJointConfig(
                encoderHidden: encoderHidden,
                predHidden: predHiddenJ,
                jointHidden: jointHidden,
                numClasses: numClasses + 1,  // +1 for blank
                numExtraOutputs: numExtraOutputs
            ),
            vocabulary: vocabulary,
            tdtDurations: tdtDurations,
            maxSymbolsPerStep: maxSymbols
        )
    }
}

// ─── Audio front-end (NeMo Mel spectrogram) ──────────────────────────

/// NeMo-compatible log-Mel spectrogram front-end.
///
/// Key differences from Whisper's front-end:
///   * Pre-emphasis filter applied before STFT.
///   * Slaney-normalised triangular Mel filterbank (like librosa).
///   * Log applied to power spectrum (not amplitude).
///   * Per-feature z-score normalisation (across time, per Mel bin).
///
/// All computation runs CPU-side on `[Float]`.
enum ParakeetFrontEnd {

    // ── Window functions ─────────────────────────────────────────────

    /// Periodic Hann window of length `n`.
    static func hannWindow(_ n: Int) -> [Float] {
        guard n > 1 else { return [Float](repeating: 1, count: max(n, 0)) }
        return (0 ..< n).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n))) }
    }

    // ── Mel filterbank ───────────────────────────────────────────────

    private static func hzToMel(_ hz: Double) -> Double {
        2595.0 * log10(1.0 + hz / 700.0)
    }

    private static func melToHz(_ mel: Double) -> Double {
        700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }

    /// Slaney-normalised `[nMels × nFreq]` row-major Mel filterbank.
    static func melFilterbank(
        sampleRate: Int, nFFT: Int, nMels: Int
    ) -> [Float] {
        let nFreq = nFFT / 2 + 1
        let fMax = Double(sampleRate) / 2.0
        let melLo = hzToMel(0)
        let melHi = hzToMel(fMax)
        var edgeHz = [Double](repeating: 0, count: nMels + 2)
        for i in 0 ..< (nMels + 2) {
            let m = melLo + (melHi - melLo) * Double(i) / Double(nMels + 1)
            edgeHz[i] = melToHz(m)
        }
        var fftFreqs = [Double](repeating: 0, count: nFreq)
        for k in 0 ..< nFreq {
            fftFreqs[k] = Double(k) * Double(sampleRate) / Double(nFFT)
        }
        var bank = [Float](repeating: 0, count: nMels * nFreq)
        for m in 0 ..< nMels {
            let lo = edgeHz[m]
            let ctr = edgeHz[m + 1]
            let hi = edgeHz[m + 2]
            let enorm = 2.0 / (hi - lo)
            for k in 0 ..< nFreq {
                let f = fftFreqs[k]
                let lower = (f - lo) / max(ctr - lo, 1e-9)
                let upper = (hi - f) / max(hi - ctr, 1e-9)
                let tri = max(0.0, min(lower, upper))
                bank[m * nFreq + k] = Float(tri * enorm)
            }
        }
        return bank
    }

    // ── DFT (STFT) ───────────────────────────────────────────────────

    /// Compute the real-FFT magnitude-squared spectrum of a single frame.
    /// Uses a naive O(N²) DFT — suitable for the 512-point analysis window.
    private static func framePowerSpectrum(
        frame: [Float], window: [Float], nFFT: Int
    ) -> [Float] {
        let nFreq = nFFT / 2 + 1
        var power = [Float](repeating: 0, count: nFreq)
        let nSrc = min(frame.count, window.count, nFFT)
        // Windowed frame: w[n] * x[n]
        var windowed = [Float](repeating: 0, count: nFFT)
        for i in 0 ..< nSrc { windowed[i] = frame[i] * window[i] }

        let twoPiOverN = 2.0 * Double.pi / Double(nFFT)
        for k in 0 ..< nFreq {
            var re: Double = 0
            var im: Double = 0
            for n in 0 ..< nFFT {
                let angle = twoPiOverN * Double(k) * Double(n)
                re += Double(windowed[n]) * cos(angle)
                im -= Double(windowed[n]) * sin(angle)
            }
            power[k] = Float(re * re + im * im)
        }
        return power
    }

    // ── Entry point ──────────────────────────────────────────────────

    /// Compute NeMo-style log-Mel features from a mono 16 kHz waveform.
    ///
    /// Returns `[nFrames, nMels]` row-major, normalised as configured.
    ///
    /// - Parameters:
    ///   - waveform: Raw PCM samples at `cfg.sampleRate`.
    ///   - cfg:      Preprocessor config.
    static func logMelFeatures(
        waveform: [Float],
        cfg: ParakeetPreprocessorConfig
    ) -> [Float] {
        guard !waveform.isEmpty else { return [] }

        // 1. Pre-emphasis
        var samples = [Float](repeating: 0, count: waveform.count)
        if cfg.preemph > 0 && waveform.count > 1 {
            samples[0] = waveform[0]
            for i in 1 ..< waveform.count {
                samples[i] = waveform[i] - cfg.preemph * waveform[i - 1]
            }
        } else {
            samples = waveform
        }

        // 2. Build analysis window and Mel filterbank
        let window = hannWindow(cfg.winLength)
        let nFreq = cfg.nFFT / 2 + 1
        let bank = melFilterbank(
            sampleRate: cfg.sampleRate,
            nFFT: cfg.nFFT,
            nMels: cfg.nMels
        )

        // 3. Frame signal and compute Mel spectrogram
        let nFrames = max(0, (samples.count - cfg.winLength) / cfg.hopLength + 1)
        guard nFrames > 0 else { return [] }

        var mel = [Float](repeating: 0, count: nFrames * cfg.nMels)
        var frameBuf = [Float](repeating: 0, count: cfg.winLength)

        for t in 0 ..< nFrames {
            let start = t * cfg.hopLength
            let end = min(start + cfg.winLength, samples.count)
            let len = end - start
            for i in 0 ..< len { frameBuf[i] = samples[start + i] }
            for i in len ..< cfg.winLength { frameBuf[i] = 0 }

            let power = framePowerSpectrum(
                frame: frameBuf, window: window, nFFT: cfg.nFFT)

            // Project onto Mel filterbank
            for m in 0 ..< cfg.nMels {
                var sum: Float = 0
                for k in 0 ..< nFreq {
                    sum += bank[m * nFreq + k] * power[k]
                }
                mel[t * cfg.nMels + m] = sum
            }
        }

        // 4. Log (with guard floor)
        let logGuard = cfg.logZeroGuardValue
        for i in 0 ..< mel.count {
            mel[i] = log(max(mel[i], logGuard))
        }

        // 5. Normalise
        if cfg.normalise == "per_feature" {
            // Z-score each Mel bin across time
            for m in 0 ..< cfg.nMels {
                var sum: Float = 0
                var sum2: Float = 0
                for t in 0 ..< nFrames {
                    let v = mel[t * cfg.nMels + m]
                    sum += v
                    sum2 += v * v
                }
                let mean = sum / Float(nFrames)
                let denom = max(Float(nFrames) - 1, 1)
                let variance = (sum2 - Float(nFrames) * mean * mean) / denom
                let std = sqrt(max(variance, 0)) + 1e-5
                for t in 0 ..< nFrames {
                    mel[t * cfg.nMels + m] = (mel[t * cfg.nMels + m] - mean) / std
                }
            }
        } else {
            var sum: Float = 0
            var sum2: Float = 0
            for v in mel {
                sum += v
                sum2 += v * v
            }
            let mean = sum / Float(mel.count)
            let variance = max((sum2 / Float(mel.count)) - mean * mean, 0)
            let std = sqrt(variance) + 1e-5
            for i in 0 ..< mel.count { mel[i] = (mel[i] - mean) / std }
        }

        return mel  // [nFrames, nMels] row-major
    }
}

// ─── Parakeet model (weights + forward pass) ─────────────────────────

/// Loaded Parakeet weights and configuration.
/// All forward computation runs CPU-side; weights are accessed via
/// `Tensor.toArray(as: Float.self)`.
public final class ParakeetModel: @unchecked Sendable {
    public let config: ParakeetConfig

    // ── Subsampling (DwStridingSubsampling) ─────────────────────────

    // conv0: [convCh, 1, 3, 3] — 2D Conv on [T, F, 1] inputs
    let conv0W: Tensor  // [convCh, 3, 3]  (per-input-channel, so inCh=1)
    let conv0B: Tensor  // [convCh]

    // depthwise layers: each [convCh, 3, 3] + [convCh] bias (groups=convCh)
    let dwWeights: [Tensor]
    let dwBiases: [Tensor]

    // pointwise layers: each [convCh, convCh, 1, 1] + [convCh] bias
    let pwWeights: [Tensor]
    let pwBiases: [Tensor]

    // out: [dModel, convCh * finalFreqDim]
    let subOutWeight: Tensor
    let subOutBias: Tensor

    // ── Positional encoding (rel_pos, sinusoidal) ───────────────────

    // Computed at load time, shape [2*maxLen-1, dModel]
    var relPosTable: [Float]
    var relPosMaxLen: Int

    // ── Conformer blocks ────────────────────────────────────────────

    let blocks: [ParakeetConformerBlockWeights]

    // ── Prediction network ──────────────────────────────────────────

    let predEmbedWeight: Tensor  // [embeddingCount, predHidden]
    let lstmLayers: [LSTMLayerWeights]  // predRnnLayers × LSTM weights

    // ── Joint network ───────────────────────────────────────────────

    let jointEncWeight: Tensor  // [jointHidden, encoderHidden]
    let jointEncBias: Tensor
    let jointPredWeight: Tensor  // [jointHidden, predHidden]
    let jointPredBias: Tensor
    let jointOutWeight: Tensor  // [numClasses + numExtraOutputs, jointHidden]
    let jointOutBias: Tensor

    public init(
        config: ParakeetConfig,
        conv0W: Tensor, conv0B: Tensor,
        dwWeights: [Tensor], dwBiases: [Tensor],
        pwWeights: [Tensor], pwBiases: [Tensor],
        subOutWeight: Tensor, subOutBias: Tensor,
        relPosTable: [Float], relPosMaxLen: Int,
        blocks: [ParakeetConformerBlockWeights],
        predEmbedWeight: Tensor,
        lstmLayers: [LSTMLayerWeights],
        jointEncWeight: Tensor, jointEncBias: Tensor,
        jointPredWeight: Tensor, jointPredBias: Tensor,
        jointOutWeight: Tensor, jointOutBias: Tensor
    ) {
        self.config = config
        self.conv0W = conv0W
        self.conv0B = conv0B
        self.dwWeights = dwWeights
        self.dwBiases = dwBiases
        self.pwWeights = pwWeights
        self.pwBiases = pwBiases
        self.subOutWeight = subOutWeight
        self.subOutBias = subOutBias
        self.relPosTable = relPosTable
        self.relPosMaxLen = relPosMaxLen
        self.blocks = blocks
        self.predEmbedWeight = predEmbedWeight
        self.lstmLayers = lstmLayers
        self.jointEncWeight = jointEncWeight
        self.jointEncBias = jointEncBias
        self.jointPredWeight = jointPredWeight
        self.jointPredBias = jointPredBias
        self.jointOutWeight = jointOutWeight
        self.jointOutBias = jointOutBias
    }

    /// `true` when `config` describes a Parakeet checkpoint.
    /// Detection: `encoder` block with `feat_in` and `model_defaults`
    /// with `tdt_durations` — a combination no other family carries.
    public static func handles(_ config: ModelConfig) -> Bool {
        guard config.raw["encoder"] as? [String: Any] != nil,
            let defaults = config.raw["model_defaults"] as? [String: Any],
            defaults["tdt_durations"] != nil
        else { return false }
        return true
    }

    // ── Transcription ────────────────────────────────────────────────

    /// Transcribe a 16 kHz mono waveform to text using greedy TDT decoding.
    public func transcribe(waveform: [Float]) -> String {
        let tokens = transcribeTokens(waveform: waveform)
        return ParakeetTokeniser.decode(tokens: tokens, vocabulary: config.vocabulary)
    }

    /// Transcribe a waveform and return raw vocabulary indices.
    public func transcribeTokens(waveform: [Float]) -> [Int] {
        // 1. Compute mel features [nFrames, nMels]
        let mel = ParakeetFrontEnd.logMelFeatures(
            waveform: waveform, cfg: config.preprocessor)
        guard !mel.isEmpty else { return [] }
        let nFrames = mel.count / config.preprocessor.nMels

        // 2. Conformer encoder: [nFrames, nMels] → [T_enc, dModel]
        let (encoded, encodedLen) = encodeFeatures(mel: mel, nFrames: nFrames)
        guard encodedLen > 0 else { return [] }

        // 3. Greedy TDT decode
        return greedyTDT(encoded: encoded, encodedLen: encodedLen)
    }

    // ─── Conformer encoder ───────────────────────────────────────────

    /// Run the full Conformer encoder stack.
    ///
    /// - Parameters:
    ///   - mel:      `[nFrames * nMels]` row-major float array.
    ///   - nFrames:  Number of time frames.
    /// - Returns: `([T_enc × dModel], T_enc)` where T_enc = nFrames / subsamplingFactor.
    private func encodeFeatures(
        mel: [Float], nFrames: Int
    ) -> ([Float], Int) {
        let enc = config.encoder

        // ── DwStridingSubsampling ────────────────────────────────────
        let (sub, subLen) = dWStridingSubsampling(
            input: mel, nFrames: nFrames, featIn: enc.featIn)
        guard subLen > 0 else { return ([], 0) }

        // ── Relative positional encoding ─────────────────────────────
        // Extend PE table if needed
        ensureRelPos(minLen: subLen + 1)
        // Positional embedding slice [2*subLen-1, dModel] from table
        let posEmb = sliceRelPos(seqLen: subLen)

        // ── Conformer blocks ─────────────────────────────────────────
        var h = sub  // [subLen, dModel]
        for block in blocks {
            h = conformerBlock(
                input: h, nFrames: subLen,
                posEmb: posEmb, block: block)
        }

        return (h, subLen)
    }

    // ─── DwStridingSubsampling ───────────────────────────────────────
    //
    // Processes [nFrames, featIn] → [T_out, dModel].
    // Architecture:
    //   conv0(1→convCh, 3×3, stride 2) + ReLU
    //   for each dw/pw pair:
    //     depthwise_conv(convCh→convCh, 3×3, stride 2, groups=convCh) + pointwise(1×1) + ReLU
    //   reshape [T_out, convCh * freq_out] → linear → [T_out, dModel]
    //
    // All 2D convolutions treat the input as [T, F, channels] and operate
    // on the T and F axes with stride 2 (halving both dimensions each step).

    private func dWStridingSubsampling(
        input: [Float], nFrames: Int, featIn: Int
    ) -> ([Float], Int) {
        let enc = config.encoder
        let convCh = enc.subsamplingConvChannels
        let stride = 2
        let kernelSize = 3
        let pad = 1

        // Treat input as [nFrames, featIn, 1] — last dim is input channels
        var currentT = nFrames
        var currentF = featIn
        var currentCh = 1
        var current = input  // [T, F, C] row-major = [T * F * C] flat

        // conv0: inCh=1, outCh=convCh, 3×3, stride 2, pad 1
        let c0W = conv0W.toArray(as: Float.self)  // [convCh * 1 * 3 * 3]
        let c0B = conv0B.toArray(as: Float.self)  // [convCh]
        (current, currentT, currentF, currentCh) = conv2DReLU(
            input: current, T: currentT, F: currentF, inCh: currentCh,
            outCh: convCh, weight: c0W, bias: c0B,
            kernelH: kernelSize, kernelW: kernelSize,
            strideH: stride, strideW: stride,
            padH: pad, padW: pad,
            groups: 1
        )

        // Additional depthwise + pointwise steps
        for i in 0 ..< dwWeights.count {
            let dwW = dwWeights[i].toArray(as: Float.self)
            let dwB = dwBiases[i].toArray(as: Float.self)
            (current, currentT, currentF, currentCh) = conv2DReLU(
                input: current, T: currentT, F: currentF, inCh: currentCh,
                outCh: convCh, weight: dwW, bias: dwB,
                kernelH: kernelSize, kernelW: kernelSize,
                strideH: stride, strideW: stride,
                padH: pad, padW: pad,
                groups: convCh  // depthwise
            )
            let pwW = pwWeights[i].toArray(as: Float.self)
            let pwB = pwBiases[i].toArray(as: Float.self)
            (current, currentT, currentF, currentCh) = conv2DReLU(
                input: current, T: currentT, F: currentF, inCh: currentCh,
                outCh: convCh, weight: pwW, bias: pwB,
                kernelH: 1, kernelW: 1,
                strideH: 1, strideW: 1,
                padH: 0, padW: 0,
                groups: 1  // pointwise
            )
        }

        // Reshape [T, F, convCh] → [T, F * convCh], then project to [T, dModel]
        // Note: conv2DReLU returns [T, F, Ch] — we rearrange to [T, Ch * F] (channel-last)
        let outT = currentT
        let flatDim = currentF * currentCh
        var reshaped = [Float](repeating: 0, count: outT * flatDim)
        // current is [T, F, Ch]; we want [T, Ch * F] = transpose last two dims
        for t in 0 ..< outT {
            for f in 0 ..< currentF {
                for c in 0 ..< currentCh {
                    let srcIdx = t * (currentF * currentCh) + f * currentCh + c
                    let dstIdx = t * flatDim + c * currentF + f
                    reshaped[dstIdx] = current[srcIdx]
                }
            }
        }

        let soW = subOutWeight.toArray(as: Float.self)  // [dModel, flatDim]
        let soB = subOutBias.toArray(as: Float.self)  // [dModel]
        let dModel = enc.dModel
        var out = [Float](repeating: 0, count: outT * dModel)
        // Matrix multiply: [outT, flatDim] × [flatDim, dModel] (= [dModel, flatDim]^T)
        matmul(
            A: reshaped, B: soW,
            C: &out,
            M: outT, K: flatDim, N: dModel,
            transB: true
        )
        // Add bias to each row
        for t in 0 ..< outT {
            for d in 0 ..< dModel { out[t * dModel + d] += soB[d] }
        }

        guard outT > 0 else { return ([], 0) }
        return (out, outT)
    }

    // ─── 2D convolution + ReLU ───────────────────────────────────────

    /// 2D conv with ReLU activation on `[T, F, inCh]` inputs.
    /// For depthwise (groups == inCh), each output channel uses only its
    /// own input channel.
    ///
    /// Returns `[T_out, F_out, outCh]` row-major.
    private func conv2DReLU(
        input: [Float], T: Int, F: Int, inCh: Int,
        outCh: Int, weight: [Float], bias: [Float],
        kernelH: Int, kernelW: Int,
        strideH: Int, strideW: Int,
        padH: Int, padW: Int,
        groups: Int
    ) -> ([Float], Int, Int, Int) {
        let T_out = (T + 2 * padH - kernelH) / strideH + 1
        let F_out = (F + 2 * padW - kernelW) / strideW + 1
        guard T_out > 0, F_out > 0 else { return ([], 0, 0, 0) }

        var out = [Float](repeating: 0, count: T_out * F_out * outCh)
        let chPerGroup = outCh / groups
        let inChPerGroup = inCh / groups

        for g in 0 ..< groups {
            let outChBase = g * chPerGroup
            let inChBase = g * inChPerGroup
            for oc in outChBase ..< (outChBase + chPerGroup) {
                let b = bias[oc]
                // Weight layout: [outCh, inChPerGroup, kernelH, kernelW]
                let wBase = oc * inChPerGroup * kernelH * kernelW
                for ot in 0 ..< T_out {
                    for of_ in 0 ..< F_out {
                        var acc: Float = b
                        for ic in 0 ..< inChPerGroup {
                            for kh in 0 ..< kernelH {
                                for kw in 0 ..< kernelW {
                                    let it = ot * strideH + kh - padH
                                    let if_ = of_ * strideW + kw - padW
                                    guard it >= 0, it < T, if_ >= 0, if_ < F else { continue }
                                    let inIdx = it * F * inCh + if_ * inCh + (inChBase + ic)
                                    let wIdx = wBase + ic * kernelH * kernelW + kh * kernelW + kw
                                    acc += input[inIdx] * weight[wIdx]
                                }
                            }
                        }
                        let outIdx = ot * F_out * outCh + of_ * outCh + oc
                        out[outIdx] = max(acc, 0)  // ReLU
                    }
                }
            }
        }
        return (out, T_out, F_out, outCh)
    }

    // ─── Relative positional encoding ────────────────────────────────

    /// Build the sinusoidal relative positional embedding table.
    /// Produces `[2*maxLen-1, dModel]` row-major values.
    private func buildRelPosTable(maxLen: Int, dModel: Int) -> [Float] {
        let rows = 2 * maxLen - 1
        var table = [Float](repeating: 0, count: rows * dModel)
        let logDiv = Float(log(10000.0)) / Float(dModel)
        for r in 0 ..< rows {
            let pos = Float(maxLen - 1 - r)
            for c in stride(from: 0, to: dModel, by: 2) {
                let div = exp(-Float(c) * logDiv)
                let angle = pos * div
                table[r * dModel + c] = sin(angle)
                if c + 1 < dModel { table[r * dModel + c + 1] = cos(angle) }
            }
        }
        return table
    }

    private func ensureRelPos(minLen: Int) {
        if minLen <= relPosMaxLen { return }
        relPosMaxLen = minLen + 64
        relPosTable = buildRelPosTable(maxLen: relPosMaxLen, dModel: config.encoder.dModel)
    }

    /// Extract the positional embedding slice for a given sequence length.
    /// Returns `[2*seqLen-1, dModel]` row-major.
    private func sliceRelPos(seqLen: Int) -> [Float] {
        let dModel = config.encoder.dModel
        let bufLen = 2 * relPosMaxLen - 1
        let rows = 2 * seqLen - 1
        let start = bufLen / 2 - (seqLen - 1)
        let end = start + rows
        guard start >= 0, end <= bufLen else {
            return [Float](repeating: 0, count: rows * dModel)
        }
        return Array(relPosTable[(start * dModel) ..< (end * dModel)])
    }

    // ─── Conformer block forward ──────────────────────────────────────

    /// One Conformer block:
    ///   0.5 × FF1 + self-attn (rel_pos) + conv + 0.5 × FF2 + LayerNorm_out
    private func conformerBlock(
        input: [Float], nFrames: Int, posEmb: [Float],
        block bw: ParakeetConformerBlockWeights
    ) -> [Float] {
        let dModel = config.encoder.dModel
        let enc = config.encoder

        // 0.5 × feedForward1
        var h = input
        let ff1Out = feedForward(
            input: layerNorm(
                input, weight: bw.normFF1Weight, bias: bw.normFF1Bias,
                rows: nFrames, cols: dModel),
            nFrames: nFrames,
            w1: bw.ff1W1, b1: bw.ff1B1,
            w2: bw.ff1W2, b2: bw.ff1B2
        )
        for i in 0 ..< h.count { h[i] += 0.5 * ff1Out[i] }

        // Relative-position self-attention
        let attnNorm = layerNorm(
            h, weight: bw.normAttnWeight, bias: bw.normAttnBias,
            rows: nFrames, cols: dModel)
        let attnOut = relPosAttention(
            x: attnNorm, nFrames: nFrames, posEmb: posEmb, bw: bw)
        for i in 0 ..< h.count { h[i] += attnOut[i] }

        // Conformer convolution module
        let convNorm = layerNorm(
            h, weight: bw.normConvWeight, bias: bw.normConvBias,
            rows: nFrames, cols: dModel)
        let convOut = conformerConvModule(
            input: convNorm, nFrames: nFrames, dModel: dModel,
            convKernel: enc.convKernelSize,
            pw1W: bw.convPW1W, pw1B: bw.convPW1B,
            dwW: bw.convDWW, dwB: bw.convDWB,
            bnWeight: bw.convBNWeight, bnBias: bw.convBNBias,
            bnMean: bw.convBNMean, bnVar: bw.convBNVar,
            pw2W: bw.convPW2W, pw2B: bw.convPW2B
        )
        for i in 0 ..< h.count { h[i] += convOut[i] }

        // 0.5 × feedForward2
        let ff2Out = feedForward(
            input: layerNorm(
                h, weight: bw.normFF2Weight, bias: bw.normFF2Bias,
                rows: nFrames, cols: dModel),
            nFrames: nFrames,
            w1: bw.ff2W1, b1: bw.ff2B1,
            w2: bw.ff2W2, b2: bw.ff2B2
        )
        for i in 0 ..< h.count { h[i] += 0.5 * ff2Out[i] }

        // Final LayerNorm
        return layerNorm(
            h, weight: bw.normOutWeight, bias: bw.normOutBias,
            rows: nFrames, cols: dModel)
    }

    // ─── Relative-position multi-head attention ───────────────────────

    private func relPosAttention(
        x: [Float], nFrames: Int, posEmb: [Float],
        bw: ParakeetConformerBlockWeights
    ) -> [Float] {
        let dModel = config.encoder.dModel
        let nHeads = config.encoder.nHeads
        let headDim = dModel / nHeads
        let scale = 1.0 / sqrt(Float(headDim))

        // Project Q, K, V, Pos
        let qFlat = linear(x, weight: bw.qW, bias: bw.qB, M: nFrames, K: dModel, N: dModel)
        let kFlat = linear(x, weight: bw.kW, bias: bw.kB, M: nFrames, K: dModel, N: dModel)
        let vFlat = linear(x, weight: bw.vW, bias: bw.vB, M: nFrames, K: dModel, N: dModel)
        let posLen = 2 * nFrames - 1
        let pFlat = linear(posEmb, weight: bw.posW, bias: nil, M: posLen, K: dModel, N: dModel)

        // posBiasU / posBiasV: [nHeads, headDim]
        let biasU = bw.posBiasU.toArray(as: Float.self)
        let biasV = bw.posBiasV.toArray(as: Float.self)

        var out = [Float](repeating: 0, count: nFrames * dModel)

        // Each head's work is fully independent: per-head Q/K/V/P slices,
        // per-head score matrices, and writes to a disjoint
        // `[hOff, hOff + headDim)` slice of `out`. Parallelise over the
        // head loop — Parakeet typically has 4-8 heads, which gives ample
        // parallelism on Apple Silicon CPU cores. (We can't directly
        // mutate the outer `out` from inside `concurrentPerform`'s
        // @Sendable closure without going through an unsafe pointer,
        // so write through `withUnsafeMutableBufferPointer`.)
        out.withUnsafeMutableBufferPointer { outBuf in
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nHeads) { h in
                let hOff = h * headDim
                // q+u, q+v for this head
                var qU = [Float](repeating: 0, count: nFrames * headDim)
                var qV = [Float](repeating: 0, count: nFrames * headDim)
                for t in 0 ..< nFrames {
                    for d in 0 ..< headDim {
                        qU[t * headDim + d] = qFlat[t * dModel + hOff + d] + biasU[hOff + d]
                        qV[t * headDim + d] = qFlat[t * dModel + hOff + d] + biasV[hOff + d]
                    }
                }
                // Key and pos slices for this head
                var kH = [Float](repeating: 0, count: nFrames * headDim)
                var vH = [Float](repeating: 0, count: nFrames * headDim)
                var pH = [Float](repeating: 0, count: posLen * headDim)
                for t in 0 ..< nFrames {
                    for d in 0 ..< headDim {
                        kH[t * headDim + d] = kFlat[t * dModel + hOff + d]
                        vH[t * headDim + d] = vFlat[t * dModel + hOff + d]
                    }
                }
                for p in 0 ..< posLen {
                    for d in 0 ..< headDim { pH[p * headDim + d] = pFlat[p * dModel + hOff + d] }
                }

                // matrixAC = qU × K^T  [nFrames × nFrames]
                var matAC = [Float](repeating: 0, count: nFrames * nFrames)
                matmul(A: qU, B: kH, C: &matAC, M: nFrames, K: headDim, N: nFrames, transB: true)

                // matrixBD = qV × P^T  [nFrames × posLen], then rel-shift → [nFrames × nFrames]
                var matBD = [Float](repeating: 0, count: nFrames * posLen)
                matmul(A: qV, B: pH, C: &matBD, M: nFrames, K: headDim, N: posLen, transB: true)
                let shifted = relShift(matBD, tq: nFrames, posLen: posLen)
                // shifted is [nFrames, nFrames] after slicing first nFrames cols

                // Sum and scale
                var scores = [Float](repeating: 0, count: nFrames * nFrames)
                for i in 0 ..< (nFrames * nFrames) {
                    scores[i] = (matAC[i] + shifted[i]) * scale
                }

                // Softmax per query row
                let attnWeights = softmaxRows(scores, rows: nFrames, cols: nFrames)

                // Weighted sum of values
                var ctx = [Float](repeating: 0, count: nFrames * headDim)
                matmul(
                    A: attnWeights, B: vH, C: &ctx, M: nFrames, K: nFrames, N: headDim,
                    transB: false)

                // Write to output — disjoint `[hOff, hOff + headDim)` slice
                // per head, so the parallel writes can't collide.
                for t in 0 ..< nFrames {
                    for d in 0 ..< headDim { outPtr[t * dModel + hOff + d] = ctx[t * headDim + d] }
                }
            }
        }

        // Output projection
        return linear(out, weight: bw.oW, bias: bw.oB, M: nFrames, K: dModel, N: dModel)
    }

    /// Relative shift — extracts the `[nFrames × nFrames]` matrix from
    /// the `[nFrames × posLen]` dot-product with position embeddings,
    /// mirroring the `rel_shift` operation from ESPnet / NeMo.
    private func relShift(_ bd: [Float], tq: Int, posLen: Int) -> [Float] {
        // bd: [tq, posLen]; posLen = 2*tq - 1
        // Reshape to [posLen+1, tq], take rows 1...
        // Then reshape back to [tq, posLen] and slice [:, :tq]
        // Equivalent: for each row i, the shifted col j = padded[i][j+1] effectively
        // shift(bd)[i,j] = bd[i, posLen - tq + j]  (from Transformer-XL)
        let halfLen = posLen / 2  // = tq - 1
        var out = [Float](repeating: 0, count: tq * tq)
        for i in 0 ..< tq {
            for j in 0 ..< tq {
                // position index: tq - 1 - (j - i) maps into the 2*tq-1 range
                let posIdx = halfLen + i - j
                if posIdx >= 0 && posIdx < posLen {
                    out[i * tq + j] = bd[i * posLen + posIdx]
                }
            }
        }
        return out
    }

    // ─── Conformer convolution module ────────────────────────────────

    private func conformerConvModule(
        input: [Float], nFrames: Int, dModel: Int, convKernel: Int,
        pw1W: [Float], pw1B: [Float],
        dwW: [Float], dwB: [Float],
        bnWeight: [Float], bnBias: [Float],
        bnMean: [Float], bnVar: [Float],
        pw2W: [Float], pw2B: [Float]
    ) -> [Float] {
        // Pointwise 1: [nFrames, dModel] → [nFrames, 2*dModel]
        let pw1Out = linear(
            input, weight: pw1W, bias: pw1B,
            M: nFrames, K: dModel, N: dModel * 2)

        // GLU: split into two halves, gate with sigmoid
        var gluOut = [Float](repeating: 0, count: nFrames * dModel)
        for t in 0 ..< nFrames {
            for d in 0 ..< dModel {
                let x1 = pw1Out[t * dModel * 2 + d]
                let x2 = pw1Out[t * dModel * 2 + d + dModel]
                gluOut[t * dModel + d] = x1 * sigmoid1D(x2)
            }
        }

        // Depthwise Conv1d along the time axis (causal padding not needed — bidirectional)
        let pad = (convKernel - 1) / 2
        var dwOut = [Float](repeating: 0, count: nFrames * dModel)
        // dwW: [dModel, 1, convKernel] (groups = dModel, depthwise)
        for ch in 0 ..< dModel {
            let wOffset = ch * convKernel  // dwW[ch * convKernel ...]
            let b = dwB[ch]
            for t in 0 ..< nFrames {
                var acc: Float = b
                for k in 0 ..< convKernel {
                    let src = t + k - pad
                    guard src >= 0, src < nFrames else { continue }
                    acc += gluOut[src * dModel + ch] * dwW[wOffset + k]
                }
                dwOut[t * dModel + ch] = acc
            }
        }

        // Batch norm: normalise each channel using running stats
        // Then SiLU activation
        var bnOut = [Float](repeating: 0, count: nFrames * dModel)
        let bnEps: Float = 1e-5
        for d in 0 ..< dModel {
            let w = bnWeight[d]
            let bias_ = bnBias[d]
            let mean = bnMean[d]
            let varVal = bnVar[d]
            let invStd = 1.0 / sqrt(varVal + bnEps)
            for t in 0 ..< nFrames {
                let normed = (dwOut[t * dModel + d] - mean) * invStd * w + bias_
                bnOut[t * dModel + d] = silu1D(normed)
            }
        }

        // Pointwise 2: [nFrames, dModel] → [nFrames, dModel]
        return linear(
            bnOut, weight: pw2W, bias: pw2B,
            M: nFrames, K: dModel, N: dModel)
    }

    // ─── Feed-forward module ─────────────────────────────────────────

    /// Two-layer feed-forward with SiLU activation.
    private func feedForward(
        input: [Float], nFrames: Int,
        w1: [Float], b1: [Float],
        w2: [Float], b2: [Float]
    ) -> [Float] {
        let dModel = config.encoder.dModel
        let ffHidden = config.encoder.ffHidden
        var h = linear(input, weight: w1, bias: b1, M: nFrames, K: dModel, N: ffHidden)
        for i in 0 ..< h.count { h[i] = silu1D(h[i]) }
        return linear(h, weight: w2, bias: b2, M: nFrames, K: ffHidden, N: dModel)
    }

    // ─── Greedy TDT decoder ──────────────────────────────────────────

    private func greedyTDT(encoded: [Float], encodedLen: Int) -> [Int] {
        let dModel = config.encoder.dModel
        let durations = config.tdtDurations
        let maxSymbols = config.maxSymbolsPerStep
        let blankId = config.blankTokenId
        let predH = config.predNet.predHidden
        let numLstmLayers = config.predNet.predRnnLayers

        // LSTM state: [numLayers, hiddenSize]
        var hiddens = [[Float]](
            repeating: [Float](repeating: 0, count: predH),
            count: numLstmLayers)
        var cells = [[Float]](
            repeating: [Float](repeating: 0, count: predH),
            count: numLstmLayers)

        var lastToken = blankId
        var hypothesis: [Int] = []
        var t = 0
        var newSymbols = 0

        while t < encodedLen {
            let encFrame = Array(encoded[(t * dModel) ..< ((t + 1) * dModel)])

            // Predict network: embedding + stacked LSTM
            let embedded = embedToken(lastToken, blankId: blankId)
            var predOut = embedded
            var nextHiddens = hiddens
            var nextCells = cells
            for layer in 0 ..< numLstmLayers {
                let (out, h, c) = lstmStep(
                    input: predOut,
                    hidden: hiddens[layer],
                    cell: cells[layer],
                    weights: lstmLayers[layer]
                )
                predOut = out
                nextHiddens[layer] = h
                nextCells[layer] = c
            }

            // Joint network
            let logits = jointForward(enc: encFrame, pred: predOut)
            let totalOut = logits.count
            let numTokenLogits = blankId + 1  // vocab + blank
            let numDurLogits = totalOut - numTokenLogits

            // Token decision
            let tokenLogits = Array(logits[..<numTokenLogits])
            let tokenArgmax = argmax1D(tokenLogits)

            // Duration decision
            let durationArgmax: Int
            if numDurLogits > 0 {
                let durLogits = Array(logits[numTokenLogits...])
                durationArgmax = argmax1D(durLogits)
            } else {
                durationArgmax = 0
            }

            // TDT step logic
            let jump =
                durations.indices.contains(durationArgmax)
                ? durations[durationArgmax] : 1
            var nextTime = t + jump
            var nextNewSymbols = newSymbols + 1

            if jump != 0 {
                nextNewSymbols = 0
            } else if let ms = maxSymbols, nextNewSymbols >= ms {
                nextTime += 1
                nextNewSymbols = 0
            }

            if tokenArgmax != blankId {
                // Commit prediction network state
                hiddens = nextHiddens
                cells = nextCells
                lastToken = tokenArgmax
                if !ParakeetTokeniser.isSpecial(tokenArgmax, vocabulary: config.vocabulary) {
                    hypothesis.append(tokenArgmax)
                }
            }

            t = nextTime
            newSymbols = nextNewSymbols
        }

        return hypothesis
    }

    /// Embed a single token (zero-embedding for blank).
    private func embedToken(_ token: Int, blankId: Int) -> [Float] {
        let predH = config.predNet.predHidden
        if token == blankId {
            return [Float](repeating: 0, count: predH)
        }
        let tableSize = predEmbedWeight.shape[0]
        let safeIdx = min(max(token, 0), tableSize - 1)
        let w = predEmbedWeight.toArray(as: Float.self)
        return Array(w[(safeIdx * predH) ..< ((safeIdx + 1) * predH)])
    }

    /// One LSTM cell step.
    private func lstmStep(
        input: [Float], hidden: [Float], cell: [Float],
        weights: LSTMLayerWeights
    ) -> ([Float], [Float], [Float]) {
        let H = hidden.count
        let I = input.count
        // Compute gate pre-activations: ih_weight [4H, I] × input + hh_weight [4H, H] × hidden + biases
        var gates = [Float](repeating: 0, count: 4 * H)
        let ihW = weights.ihWeight.toArray(as: Float.self)
        let hhW = weights.hhWeight.toArray(as: Float.self)
        let ihB = weights.ihBias.toArray(as: Float.self)
        let hhB = weights.hhBias.toArray(as: Float.self)
        for j in 0 ..< (4 * H) {
            var acc = ihB[j] + hhB[j]
            for i in 0 ..< I { acc += ihW[j * I + i] * input[i] }
            for h in 0 ..< H { acc += hhW[j * H + h] * hidden[h] }
            gates[j] = acc
        }
        // Apply activations: i=sigmoid, f=sigmoid, g=tanh, o=sigmoid
        var newCell = [Float](repeating: 0, count: H)
        var newHidden = [Float](repeating: 0, count: H)
        for idx in 0 ..< H {
            let i_gate = sigmoid1D(gates[idx])
            let f_gate = sigmoid1D(gates[H + idx])
            let g_gate = tanh(gates[2 * H + idx])
            let o_gate = sigmoid1D(gates[3 * H + idx])
            newCell[idx] = f_gate * cell[idx] + i_gate * g_gate
            newHidden[idx] = o_gate * tanh(newCell[idx])
        }
        return (newHidden, newHidden, newCell)
    }

    /// Joint network forward pass.
    /// Returns logits of shape `[numClasses + numExtraOutputs]`.
    private func jointForward(enc: [Float], pred: [Float]) -> [Float] {
        let cfg = config.joint
        let encW = jointEncWeight.toArray(as: Float.self)
        let encB = jointEncBias.toArray(as: Float.self)
        let predW = jointPredWeight.toArray(as: Float.self)
        let predB = jointPredBias.toArray(as: Float.self)
        let outW = jointOutWeight.toArray(as: Float.self)
        let outB = jointOutBias.toArray(as: Float.self)

        // Project encoder and predictor into joint space
        var encProj = [Float](repeating: 0, count: cfg.jointHidden)
        for j in 0 ..< cfg.jointHidden {
            var acc = encB[j]
            for i in 0 ..< enc.count { acc += encW[j * enc.count + i] * enc[i] }
            encProj[j] = acc
        }
        var predProj = [Float](repeating: 0, count: cfg.jointHidden)
        for j in 0 ..< cfg.jointHidden {
            var acc = predB[j]
            for i in 0 ..< pred.count { acc += predW[j * pred.count + i] * pred[i] }
            predProj[j] = acc
        }

        // Sum + ReLU
        var joint = [Float](repeating: 0, count: cfg.jointHidden)
        for j in 0 ..< cfg.jointHidden { joint[j] = max(encProj[j] + predProj[j], 0) }

        // Output projection → logits
        let numOut = cfg.numClasses + cfg.numExtraOutputs
        var logits = [Float](repeating: 0, count: numOut)
        for j in 0 ..< numOut {
            var acc = outB[j]
            for i in 0 ..< cfg.jointHidden { acc += outW[j * cfg.jointHidden + i] * joint[i] }
            logits[j] = acc
        }
        return logits
    }

    // ─── Math helpers ────────────────────────────────────────────────

    /// Row-major matmul: C[M,N] = A[M,K] × B_layout[K,N].
    /// When `transB == true`, B is `[N,K]` and is transposed.
    private func matmul(
        A: [Float], B: [Float], C: inout [Float],
        M: Int, K: Int, N: Int, transB: Bool
    ) {
        for i in 0 ..< M {
            for j in 0 ..< N {
                var acc: Float = 0
                for k in 0 ..< K {
                    acc += A[i * K + k] * (transB ? B[j * K + k] : B[k * N + j])
                }
                C[i * N + j] = acc
            }
        }
    }

    /// Dense linear layer: out[M, N] = input[M, K] × weight[N, K]^T + bias[N].
    private func linear(
        _ input: [Float], weight: [Float], bias: [Float]?,
        M: Int, K: Int, N: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: M * N)
        for i in 0 ..< M {
            for j in 0 ..< N {
                var acc: Float = bias?[j] ?? 0
                for k in 0 ..< K { acc += input[i * K + k] * weight[j * K + k] }
                out[i * N + j] = acc
            }
        }
        return out
    }

    /// Layer normalisation over each row of `[rows, cols]`.
    private func layerNorm(
        _ input: [Float], weight: [Float], bias: [Float],
        rows: Int, cols: Int, eps: Float = 1e-5
    ) -> [Float] {
        var out = [Float](repeating: 0, count: rows * cols)
        for r in 0 ..< rows {
            let base = r * cols
            var sum: Float = 0
            var sum2: Float = 0
            for c in 0 ..< cols {
                let v = input[base + c]
                sum += v
                sum2 += v * v
            }
            let mean = sum / Float(cols)
            let variance = max((sum2 / Float(cols)) - mean * mean, 0)
            let invStd = 1.0 / sqrt(variance + eps)
            for c in 0 ..< cols {
                out[base + c] = (input[base + c] - mean) * invStd * weight[c] + bias[c]
            }
        }
        return out
    }

    /// Softmax over each row.
    private func softmaxRows(_ x: [Float], rows: Int, cols: Int) -> [Float] {
        var out = x
        for r in 0 ..< rows {
            let base = r * cols
            var maxVal = -Float.greatestFiniteMagnitude
            for c in 0 ..< cols { if x[base + c] > maxVal { maxVal = x[base + c] } }
            var sum: Float = 0
            for c in 0 ..< cols {
                let e = exp(out[base + c] - maxVal)
                out[base + c] = e
                sum += e
            }
            let inv = sum > 0 ? 1 / sum : 0
            for c in 0 ..< cols { out[base + c] *= inv }
        }
        return out
    }

    private func argmax1D(_ x: [Float]) -> Int {
        guard !x.isEmpty else { return 0 }
        var bestIdx = 0
        var bestVal = x[0]
        for i in 1 ..< x.count {
            if x[i] > bestVal {
                bestVal = x[i]
                bestIdx = i
            }
        }
        return bestIdx
    }

    private func sigmoid1D(_ x: Float) -> Float { 1.0 / (1.0 + exp(-x)) }

    private func silu1D(_ x: Float) -> Float { x / (1.0 + exp(-x)) }
}

// ─── Conformer block weights container ───────────────────────────────

/// Holds all weight tensors for one Conformer block.
/// Extracted from the SafeTensorsBundle during `load(...)`.
public final class ParakeetConformerBlockWeights: Sendable {
    // Feed-forward 1
    let normFF1Weight: [Float]
    let normFF1Bias: [Float]
    let ff1W1: [Float]
    let ff1B1: [Float]
    let ff1W2: [Float]
    let ff1B2: [Float]
    // Self attention
    let normAttnWeight: [Float]
    let normAttnBias: [Float]
    let qW: [Float]
    let qB: [Float]
    let kW: [Float]
    let kB: [Float]
    let vW: [Float]
    let vB: [Float]
    let posW: [Float]  // linear_pos (no bias)
    let oW: [Float]
    let oB: [Float]
    let posBiasU: Tensor  // [nHeads, headDim]
    let posBiasV: Tensor  // [nHeads, headDim]
    // Conformer conv
    let normConvWeight: [Float]
    let normConvBias: [Float]
    let convPW1W: [Float]
    let convPW1B: [Float]
    let convDWW: [Float]
    let convDWB: [Float]
    let convBNWeight: [Float]
    let convBNBias: [Float]
    let convBNMean: [Float]
    let convBNVar: [Float]
    let convPW2W: [Float]
    let convPW2B: [Float]
    // Feed-forward 2
    let normFF2Weight: [Float]
    let normFF2Bias: [Float]
    let ff2W1: [Float]
    let ff2B1: [Float]
    let ff2W2: [Float]
    let ff2B2: [Float]
    // Output norm
    let normOutWeight: [Float]
    let normOutBias: [Float]

    init(
        normFF1Weight: [Float], normFF1Bias: [Float],
        ff1W1: [Float], ff1B1: [Float], ff1W2: [Float], ff1B2: [Float],
        normAttnWeight: [Float], normAttnBias: [Float],
        qW: [Float], qB: [Float], kW: [Float], kB: [Float],
        vW: [Float], vB: [Float], posW: [Float], oW: [Float], oB: [Float],
        posBiasU: Tensor, posBiasV: Tensor,
        normConvWeight: [Float], normConvBias: [Float],
        convPW1W: [Float], convPW1B: [Float],
        convDWW: [Float], convDWB: [Float],
        convBNWeight: [Float], convBNBias: [Float],
        convBNMean: [Float], convBNVar: [Float],
        convPW2W: [Float], convPW2B: [Float],
        normFF2Weight: [Float], normFF2Bias: [Float],
        ff2W1: [Float], ff2B1: [Float], ff2W2: [Float], ff2B2: [Float],
        normOutWeight: [Float], normOutBias: [Float]
    ) {
        self.normFF1Weight = normFF1Weight
        self.normFF1Bias = normFF1Bias
        self.ff1W1 = ff1W1
        self.ff1B1 = ff1B1
        self.ff1W2 = ff1W2
        self.ff1B2 = ff1B2
        self.normAttnWeight = normAttnWeight
        self.normAttnBias = normAttnBias
        self.qW = qW
        self.qB = qB
        self.kW = kW
        self.kB = kB
        self.vW = vW
        self.vB = vB
        self.posW = posW
        self.oW = oW
        self.oB = oB
        self.posBiasU = posBiasU
        self.posBiasV = posBiasV
        self.normConvWeight = normConvWeight
        self.normConvBias = normConvBias
        self.convPW1W = convPW1W
        self.convPW1B = convPW1B
        self.convDWW = convDWW
        self.convDWB = convDWB
        self.convBNWeight = convBNWeight
        self.convBNBias = convBNBias
        self.convBNMean = convBNMean
        self.convBNVar = convBNVar
        self.convPW2W = convPW2W
        self.convPW2B = convPW2B
        self.normFF2Weight = normFF2Weight
        self.normFF2Bias = normFF2Bias
        self.ff2W1 = ff2W1
        self.ff2B1 = ff2B1
        self.ff2W2 = ff2W2
        self.ff2B2 = ff2B2
        self.normOutWeight = normOutWeight
        self.normOutBias = normOutBias
    }
}

// ─── LSTM layer weights container ────────────────────────────────────

/// One LSTM layer's weights: input-hidden matrix, hidden-hidden matrix, biases.
public final class LSTMLayerWeights: Sendable {
    let ihWeight: Tensor  // [4 * hiddenSize, inputSize]
    let hhWeight: Tensor  // [4 * hiddenSize, hiddenSize]
    let ihBias: Tensor  // [4 * hiddenSize]
    let hhBias: Tensor  // [4 * hiddenSize]

    init(ihWeight: Tensor, hhWeight: Tensor, ihBias: Tensor, hhBias: Tensor) {
        self.ihWeight = ihWeight
        self.hhWeight = hhWeight
        self.ihBias = ihBias
        self.hhBias = hhBias
    }
}

// ─── Tokeniser ───────────────────────────────────────────────────────

/// BPE-style tokeniser for Parakeet's byte-pair encoded vocabulary.
/// The vocabulary uses the `▁` (U+2581) sentinel for word boundaries;
/// `decode` replaces it with a space.
public enum ParakeetTokeniser {
    /// Decode a list of token indices to text.
    public static func decode(tokens: [Int], vocabulary: [String]) -> String {
        var text = ""
        for t in tokens {
            guard t >= 0, t < vocabulary.count else { continue }
            text += vocabulary[t]
        }
        // Replace BPE word-boundary marker (▁ U+2581) with space.
        return text.replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
    }

    /// True when `token` is a special / control token that should not
    /// appear in a clean transcript.
    public static func isSpecial(_ token: Int, vocabulary: [String]) -> Bool {
        guard token >= 0, token < vocabulary.count else { return true }
        let text = vocabulary[token]
        return text.hasPrefix("<|") || text == "<unk>" || text == "<pad>"
    }
}

// ─── Weight loading ──────────────────────────────────────────────────

/// Transpose a 2D conv weight stored channel-last (`[outCh, kH, kW, inCh]`,
/// the MLX export convention) into the channel-first OIHW layout
/// (`[outCh, inCh, kH, kW]`) that `conv2DReLU` indexes against. Works
/// for plain and grouped/depthwise variants — depthwise weights ship as
/// `[outCh, kH, kW, 1]` and become `[outCh, 1, kH, kW]` (one input
/// channel per group, the convention `conv2DReLU` expects with
/// `groups == outCh`).
private func parakeetTransposeOHWIToOIHW(
    _ raw: Tensor, device: Device
) -> Tensor {
    precondition(
        raw.shape.count == 4,
        "Parakeet conv weight must be 4D, got \(raw.shape)")
    let outCh = raw.shape[0]
    let kH = raw.shape[1]
    let kW = raw.shape[2]
    let inCh = raw.shape[3]
    let src = raw.toFloatArray()
    var dst = [Float](repeating: 0, count: outCh * inCh * kH * kW)
    for o in 0 ..< outCh {
        for h in 0 ..< kH {
            for w in 0 ..< kW {
                for i in 0 ..< inCh {
                    let srcIdx =
                        o * (kH * kW * inCh) + h * (kW * inCh)
                        + w * inCh + i
                    let dstIdx =
                        o * (inCh * kH * kW) + i * (kH * kW)
                        + h * kW + w
                    dst[dstIdx] = src[srcIdx]
                }
            }
        }
    }
    let out = Tensor.empty(
        shape: [outCh, inCh, kH, kW],
        dtype: raw.dtype, device: device)
    switch raw.dtype {
    case .f32:
        out.copyIn(from: dst)
    case .f16:
        out.copyIn(from: dst.map { Float16($0) })
    case .bf16:
        out.copyIn(
            from: dst.map { v -> UInt16 in
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            })
    default:
        preconditionFailure(
            "Parakeet conv weight: unsupported dtype \(raw.dtype)")
    }
    return out
}

extension ParakeetModel {

    /// Load a Parakeet checkpoint from a directory that contains
    /// `config.json` and `.safetensors` weight shards.
    public static func load(
        directory: URL,
        device: Device = .shared
    ) throws -> ParakeetModel {
        let modelConfig = try ModelConfig.load(from: directory)
        let config = try ParakeetConfig.from(modelConfig)
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return try build(config: config, bundle: bundle, device: device)
    }

    /// Assemble a `ParakeetModel` from a resolved config + weight bundle.
    public static func build(
        config: ParakeetConfig,
        bundle: SafeTensorsBundle,
        device: Device = .shared
    ) throws -> ParakeetModel {
        let enc = config.encoder
        let samplingNum = Int(log2(Double(enc.subsamplingFactor)))

        // ── Subsampling weights ─────────────────────────────────────
        // The mlx-community Parakeet TDT export ships the NeMo conv
        // list under its original `nn.Sequential` indices:
        //   conv.0  — first 3×3 stride-2 conv (1 → convCh)
        //   conv.(2 + 3n)  — depthwise 3×3 stride-2 (n = 0..samplingNum-2)
        //   conv.(3 + 3n)  — pointwise 1×1 (n = 0..samplingNum-2)
        // (Indices 1, 4, 7, … are ReLU activations and carry no weight.)
        // The conv weight tensors arrive in OHWI layout `[outCh, kH, kW, inCh]`;
        // `conv2DReLU` reads OIHW `[outCh, inChPerGroup, kH, kW]`. Transpose
        // once at load time so the runtime indexing stays simple.
        let pref = "encoder.pre_encode"
        let conv0W = try parakeetTransposeOHWIToOIHW(
            bundle.tensor(named: "\(pref).conv.0.weight"), device: device)
        let conv0B = try bundle.tensor(named: "\(pref).conv.0.bias")

        var dwWeights: [Tensor] = []
        var dwBiases: [Tensor] = []
        var pwWeights: [Tensor] = []
        var pwBiases: [Tensor] = []
        for i in 0 ..< (samplingNum - 1) {
            let dwIdx = 2 + 3 * i
            let pwIdx = 3 + 3 * i
            dwWeights.append(
                try parakeetTransposeOHWIToOIHW(
                    bundle.tensor(named: "\(pref).conv.\(dwIdx).weight"), device: device))
            dwBiases.append(try bundle.tensor(named: "\(pref).conv.\(dwIdx).bias"))
            pwWeights.append(
                try parakeetTransposeOHWIToOIHW(
                    bundle.tensor(named: "\(pref).conv.\(pwIdx).weight"), device: device))
            pwBiases.append(try bundle.tensor(named: "\(pref).conv.\(pwIdx).bias"))
        }
        let subOutWeight = try bundle.tensor(named: "\(pref).out.weight")
        let subOutBias = try bundle.tensor(named: "\(pref).out.bias")

        // ── Conformer blocks ─────────────────────────────────────────
        var blocks: [ParakeetConformerBlockWeights] = []
        for l in 0 ..< enc.nLayers {
            let bp = "encoder.layers.\(l)"
            let bw = try loadConformerBlock(
                prefix: bp, bundle: bundle,
                dModel: enc.dModel, ffHidden: enc.ffHidden,
                convKernel: enc.convKernelSize
            )
            blocks.append(bw)
        }

        // ── Positional encoding (computed, not loaded) ───────────────
        let relPosMaxLen = enc.posEmbMaxLen
        let dummy = ParakeetModel(
            config: config,
            conv0W: conv0W, conv0B: conv0B,
            dwWeights: dwWeights, dwBiases: dwBiases,
            pwWeights: pwWeights, pwBiases: pwBiases,
            subOutWeight: subOutWeight, subOutBias: subOutBias,
            relPosTable: [], relPosMaxLen: 0,
            blocks: blocks,
            predEmbedWeight: conv0W,  // placeholder
            lstmLayers: [],
            jointEncWeight: conv0W, jointEncBias: conv0B,
            jointPredWeight: conv0W, jointPredBias: conv0B,
            jointOutWeight: conv0W, jointOutBias: conv0B
        )
        let relPosTable = dummy.buildRelPosTable(maxLen: relPosMaxLen, dModel: enc.dModel)

        // ── Prediction network ───────────────────────────────────────
        let predEmbedWeight = try bundle.tensor(named: "decoder.prediction.embed.weight")
        var lstmLayers: [LSTMLayerWeights] = []
        for i in 0 ..< config.predNet.predRnnLayers {
            let lp = "decoder.prediction.dec_rnn.lstm.\(i)"
            // mlx-community's NeMo Parakeet conversion renames the
            // input-hidden / hidden-hidden weights to `Wx` / `Wh` and
            // fuses the two PyTorch biases into a single `bias` vector.
            // `lstmStep` adds `ihB[j] + hhB[j]` per gate, so feed the
            // fused bias as `ihBias` and zero out `hhBias`. (Try the
            // PyTorch-style keys first so this loader still works on a
            // direct NeMo export that ships `weight_ih` / `bias_ih`.)
            let ihWeight: Tensor
            let hhWeight: Tensor
            let ihBias: Tensor
            let hhBias: Tensor
            if bundle.has("\(lp).weight_ih") {
                ihWeight = try bundle.tensor(named: "\(lp).weight_ih")
                hhWeight = try bundle.tensor(named: "\(lp).weight_hh")
                ihBias = try bundle.tensor(named: "\(lp).bias_ih")
                hhBias = try bundle.tensor(named: "\(lp).bias_hh")
            } else {
                ihWeight = try bundle.tensor(named: "\(lp).Wx")
                hhWeight = try bundle.tensor(named: "\(lp).Wh")
                ihBias = try bundle.tensor(named: "\(lp).bias")
                hhBias = Tensor.filled(
                    0, shape: ihBias.shape, dtype: ihBias.dtype,
                    device: device)
            }
            lstmLayers.append(
                LSTMLayerWeights(
                    ihWeight: ihWeight, hhWeight: hhWeight,
                    ihBias: ihBias, hhBias: hhBias
                ))
        }

        // ── Joint network ────────────────────────────────────────────
        // The mlx-community export stores `joint.joint_net` as an
        // `nn.Sequential`: indices 0, 1 are Linear + ReLU (the joint
        // hidden projection — already loaded as `joint.enc/pred`) and
        // index 2 is the output Linear that produces logits. Use the
        // ".2." suffix to grab the output projection.
        let jointEncWeight = try bundle.tensor(named: "joint.enc.weight")
        let jointEncBias = try bundle.tensor(named: "joint.enc.bias")
        let jointPredWeight = try bundle.tensor(named: "joint.pred.weight")
        let jointPredBias = try bundle.tensor(named: "joint.pred.bias")
        let jointOutWeight = try bundle.tensor(named: "joint.joint_net.2.weight")
        let jointOutBias = try bundle.tensor(named: "joint.joint_net.2.bias")

        return ParakeetModel(
            config: config,
            conv0W: conv0W, conv0B: conv0B,
            dwWeights: dwWeights, dwBiases: dwBiases,
            pwWeights: pwWeights, pwBiases: pwBiases,
            subOutWeight: subOutWeight, subOutBias: subOutBias,
            relPosTable: relPosTable, relPosMaxLen: relPosMaxLen,
            blocks: blocks,
            predEmbedWeight: predEmbedWeight,
            lstmLayers: lstmLayers,
            jointEncWeight: jointEncWeight, jointEncBias: jointEncBias,
            jointPredWeight: jointPredWeight, jointPredBias: jointPredBias,
            jointOutWeight: jointOutWeight, jointOutBias: jointOutBias
        )
    }

    /// Load one Conformer block's weights from the bundle.
    private static func loadConformerBlock(
        prefix: String, bundle: SafeTensorsBundle,
        dModel: Int, ffHidden: Int, convKernel: Int
    ) throws -> ParakeetConformerBlockWeights {
        func t(_ key: String) throws -> [Float] {
            try bundle.tensor(named: "\(prefix).\(key)").toArray(as: Float.self)
        }
        func tv(_ key: String) throws -> Tensor {
            try bundle.tensor(named: "\(prefix).\(key)")
        }

        return try ParakeetConformerBlockWeights(
            normFF1Weight: t("norm_feed_forward1.weight"),
            normFF1Bias: t("norm_feed_forward1.bias"),
            ff1W1: t("feed_forward1.linear1.weight"),
            ff1B1: t("feed_forward1.linear1.bias"),
            ff1W2: t("feed_forward1.linear2.weight"),
            ff1B2: t("feed_forward1.linear2.bias"),
            normAttnWeight: t("norm_self_att.weight"),
            normAttnBias: t("norm_self_att.bias"),
            qW: t("self_attn.linear_q.weight"),
            qB: t("self_attn.linear_q.bias"),
            kW: t("self_attn.linear_k.weight"),
            kB: t("self_attn.linear_k.bias"),
            vW: t("self_attn.linear_v.weight"),
            vB: t("self_attn.linear_v.bias"),
            posW: t("self_attn.linear_pos.weight"),
            oW: t("self_attn.linear_out.weight"),
            oB: t("self_attn.linear_out.bias"),
            posBiasU: tv("self_attn.posBiasU"),
            posBiasV: tv("self_attn.posBiasV"),
            normConvWeight: t("norm_conv.weight"),
            normConvBias: t("norm_conv.bias"),
            convPW1W: t("conv.pointwise_conv1.weight"),
            convPW1B: t("conv.pointwise_conv1.bias"),
            convDWW: t("conv.depthwise_conv.weight"),
            convDWB: t("conv.depthwise_conv.bias"),
            convBNWeight: t("conv.batch_norm.weight"),
            convBNBias: t("conv.batch_norm.bias"),
            convBNMean: t("conv.batch_norm.running_mean"),
            convBNVar: t("conv.batch_norm.running_var"),
            convPW2W: t("conv.pointwise_conv2.weight"),
            convPW2B: t("conv.pointwise_conv2.bias"),
            normFF2Weight: t("norm_feed_forward2.weight"),
            normFF2Bias: t("norm_feed_forward2.bias"),
            ff2W1: t("feed_forward2.linear1.weight"),
            ff2B1: t("feed_forward2.linear1.bias"),
            ff2W2: t("feed_forward2.linear2.weight"),
            ff2B2: t("feed_forward2.linear2.bias"),
            normOutWeight: t("norm_out.weight"),
            normOutBias: t("norm_out.bias")
        )
    }
}
