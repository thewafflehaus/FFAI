// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// FireRedVAD — FireRedTeam's DFSMN-based voice-activity-detection family.
//
// HF repo: `FireRedTeam/FireRedVAD`
//
// No mlx-community safetensors conversion exists as of 2026-05-22. The
// upstream checkpoint ships as a PyTorch `.pth.tar` archive. The loader
// below reads that format directly (it is a standard PyTorch zip with a
// `data.pkl` manifest and raw float32 tensor blobs), so no
// mlx-community mirror is needed to run the model. When/if an
// mlx-community safetensors snapshot appears it will slot in cleanly
// through `SafeTensorsBundle` (the `loadFromDirectory` stub is already
// wired up; just update the file path selector).
//
// Architecture (DFSMN: Deep Feedforward Sequential Memory Network):
//
//   waveform (16 kHz int16) ──kaldiFbank (80 mel, 25ms/10ms)──▶ [T, 80]
//   ──CMVN (subtract mean, scale by inv-std)──▶ [T, 80]
//   ──fc1: Linear(80→256, bias) + ReLU──▶ [T, 256]
//   ──fc2: Linear(256→128, bias) + ReLU──▶ [T, 128]
//   ──fsmn1: FSMN(P=128, N1=20, S1=1, N2=20, S2=1)──▶ [T, 128]
//   ──fsmns[0..6]: DFSMNBlock(H=256, P=128, N1=20, S1=1, N2=20, S2=1)×7──▶ [T, 128]
//   ──dnns[0]: Linear(128→256, bias) + ReLU──▶ [T, 256]
//   ──out: Linear(256→1, bias)──▶ [T, 1]
//   ──sigmoid──▶ per-frame speech probability [T]
//
// Weight names in the checkpoint (all float32):
//   dfsmn.fc1.0.weight   [256, 80]
//   dfsmn.fc1.0.bias     [256]
//   dfsmn.fc2.0.weight   [128, 256]
//   dfsmn.fc2.0.bias     [128]
//   dfsmn.fsmn1.lookback_filter.weight  [128, 1, 20]  (depthwise conv1d)
//   dfsmn.fsmn1.lookahead_filter.weight [128, 1, 20]
//   dfsmn.fsmns.N.fc1.0.weight  [256, 128]  (N = 0..6)
//   dfsmn.fsmns.N.fc1.0.bias    [256]
//   dfsmn.fsmns.N.fc2.weight    [128, 256]  (no bias)
//   dfsmn.fsmns.N.fsmn.lookback_filter.weight  [128, 1, 20]
//   dfsmn.fsmns.N.fsmn.lookahead_filter.weight [128, 1, 20]
//   dfsmn.dnns.0.weight  [256, 128]
//   dfsmn.dnns.0.bias    [256]
//   out.weight           [1, 256]
//   out.bias             [1]
//
// The FSMN memory layer is a depthwise conv1d with stride=dilation=S1/S2.
// Lookback: Conv1d(P, P, kernel=N1, dilation=S1, groups=P) applied causally
//   (right-trim the output so it only sees past context).
// Lookahead: Conv1d(P, P, kernel=N2, dilation=S2, groups=P) applied non-causally
//   (shift left to align future context).
// Memory is added residually to the projected input.
//
// CMVN parameters are baked in from `cmvn.ark`. The raw ark is a 2-row
// float64 matrix [sum, sum_of_sq] with a count in column 80. The loader
// pre-computes `mean` and `inv_std` once and stores them as [Float].
//
// Kaldi fbank feature extraction:
//   - 80 mel bins, Slaney scale, sample_rate=16000
//   - frame_length_ms=25 → 400 samples, frame_shift_ms=10 → 160 samples
//   - snip_edges=true (the first frame starts at sample 0)
//   - dither=0 (deterministic at inference)
//   - The Kaldi fbank uses log10 of the mel energies (matching knf.OnlineFbank)
//
// Non-streaming only: the full audio is processed in one shot (matching
// `FireRedVad.detect`). Streaming (FSMN cache carry-over) is not
// implemented; if a streaming variant is needed, look at `stream_vad.py`
// upstream for the cache-passing protocol.

import Accelerate
import Foundation

// ─── Errors ──────────────────────────────────────────────────────────

public enum FireRedVADError: Error, CustomStringConvertible {
    case unsupportedSampleRate(Int)
    case missingWeight(String)
    case unsupportedCheckpointFormat

    public var description: String {
        switch self {
        case .unsupportedSampleRate(let s):
            return "FireRedVAD: supports 16000 Hz audio (got \(s))"
        case .missingWeight(let w):
            return "FireRedVAD: required weight missing: \(w)"
        case .unsupportedCheckpointFormat:
            return "FireRedVAD: checkpoint is not a valid PyTorch zip archive"
        }
    }
}

// ─── Config ──────────────────────────────────────────────────────────

/// FireRedVAD model + post-processing configuration.
/// All fields have published defaults matching the upstream
/// `FireRedVadConfig` dataclass.
public struct FireRedVADConfig: Sendable {
    // ── DFSMN architecture ──────────────────────────────────────────
    /// Number of DFSMN blocks (R). First block is fsmn1; R-1 are in fsmns.
    public let numBlocks: Int
    /// DNN layers after DFSMN stack (M).
    public let numDnnLayers: Int
    /// DFSMN hidden size (H).
    public let hiddenSize: Int
    /// DFSMN projection size (P).
    public let projSize: Int
    /// FSMN lookback order (N1).
    public let lookbackOrder: Int
    /// FSMN lookback stride / dilation (S1).
    public let lookbackStride: Int
    /// FSMN lookahead order (N2).
    public let lookaheadOrder: Int
    /// FSMN lookahead stride / dilation (S2).
    public let lookaheadStride: Int
    /// Input feature dimension (idim). Must be 80 (Kaldi 80-bin fbank).
    public let idim: Int
    /// Output dimension (odim). Must be 1 (binary speech probability).
    public let odim: Int

    // ── Audio front-end ─────────────────────────────────────────────
    /// Mel bins — must be 80 to match CMVN statistics.
    public let numMelBins: Int
    /// Analysis frame length in samples at 16 kHz (25 ms = 400 samples).
    public let frameLengthSamples: Int
    /// Analysis hop size in samples at 16 kHz (10 ms = 160 samples).
    public let frameShiftSamples: Int

    // ── Post-processing ─────────────────────────────────────────────
    /// Smoothing window (frames) applied to the raw probability stream.
    public let smoothWindowSize: Int
    /// Speech detection threshold in `[0, 1]`.
    public let speechThreshold: Float
    /// Minimum speech segment duration in frames (200 ms = 20 frames).
    public let minSpeechFrame: Int
    /// Maximum speech segment duration in frames (20 s = 2000 frames).
    public let maxSpeechFrame: Int
    /// Minimum silence gap in frames to end a speech run (200 ms = 20 frames).
    public let minSilenceFrame: Int
    /// Silence gap in frames to merge across (0 = no merging).
    public let mergeSilenceFrame: Int
    /// Frames to extend before/after each detected segment (0 = no extension).
    public let extendSpeechFrame: Int

    public init(
        numBlocks: Int = 8,
        numDnnLayers: Int = 1,
        hiddenSize: Int = 256,
        projSize: Int = 128,
        lookbackOrder: Int = 20,
        lookbackStride: Int = 1,
        lookaheadOrder: Int = 20,
        lookaheadStride: Int = 1,
        idim: Int = 80,
        odim: Int = 1,
        numMelBins: Int = 80,
        frameLengthSamples: Int = 400,
        frameShiftSamples: Int = 160,
        smoothWindowSize: Int = 5,
        speechThreshold: Float = 0.4,
        minSpeechFrame: Int = 20,
        maxSpeechFrame: Int = 2000,
        minSilenceFrame: Int = 20,
        mergeSilenceFrame: Int = 0,
        extendSpeechFrame: Int = 0
    ) {
        self.numBlocks = numBlocks
        self.numDnnLayers = numDnnLayers
        self.hiddenSize = hiddenSize
        self.projSize = projSize
        self.lookbackOrder = lookbackOrder
        self.lookbackStride = lookbackStride
        self.lookaheadOrder = lookaheadOrder
        self.lookaheadStride = lookaheadStride
        self.idim = idim
        self.odim = odim
        self.numMelBins = numMelBins
        self.frameLengthSamples = frameLengthSamples
        self.frameShiftSamples = frameShiftSamples
        self.smoothWindowSize = smoothWindowSize
        self.speechThreshold = speechThreshold
        self.minSpeechFrame = minSpeechFrame
        self.maxSpeechFrame = maxSpeechFrame
        self.minSilenceFrame = minSilenceFrame
        self.mergeSilenceFrame = mergeSilenceFrame
        self.extendSpeechFrame = extendSpeechFrame
    }

    /// Decode from a HuggingFace `config.json` dictionary. Every field
    /// has a published default, so a missing / sparse config is fine.
    public static func decode(from raw: [String: Any]) -> FireRedVADConfig {
        func i(_ k: String, _ fb: Int) -> Int { (raw[k] as? Int) ?? fb }
        func f(_ k: String, _ fb: Float) -> Float {
            (raw[k] as? NSNumber)?.floatValue ?? fb
        }
        return FireRedVADConfig(
            numBlocks: i("num_blocks", 8),
            numDnnLayers: i("num_dnn_layers", 1),
            hiddenSize: i("hidden_size", 256),
            projSize: i("proj_size", 128),
            lookbackOrder: i("lookback_order", 20),
            lookbackStride: i("lookback_stride", 1),
            lookaheadOrder: i("lookahead_order", 20),
            lookaheadStride: i("lookahead_stride", 1),
            idim: i("idim", 80),
            odim: i("odim", 1),
            numMelBins: i("num_mel_bins", 80),
            frameLengthSamples: i("frame_length_samples", 400),
            frameShiftSamples: i("frame_shift_samples", 160),
            smoothWindowSize: i("smooth_window_size", 5),
            speechThreshold: f("speech_threshold", 0.4),
            minSpeechFrame: i("min_speech_frame", 20),
            maxSpeechFrame: i("max_speech_frame", 2000),
            minSilenceFrame: i("min_silence_frame", 20),
            mergeSilenceFrame: i("merge_silence_frame", 0),
            extendSpeechFrame: i("extend_speech_frame", 0))
    }
}

// ─── FSMN memory layer ────────────────────────────────────────────────

/// One FSMN (Feedforward Sequential Memory Network) memory layer.
///
/// Applies a depthwise-conv lookback filter and optional lookahead filter
/// to the projected sequence, then adds the result residually. Matches
/// `FSMN.forward` in the upstream `detect_model.py`.
///
/// Weight layout per filter: `[P, 1, N]` (PyTorch depthwise conv1d with
/// `groups=P`) stored as `[P, N]` floats after squeezing dim 1.
final class FireRedFSMN: Sendable {
    /// Lookback filter weights `[P, N1]` (P channels × N1 kernel taps).
    let lookbackWeight: [Float]
    /// Lookahead filter weights `[P, N2]` or empty when N2=0.
    let lookaheadWeight: [Float]
    let P: Int  // projection size
    let N1: Int  // lookback order
    let S1: Int  // lookback stride / dilation
    let N2: Int  // lookahead order
    let S2: Int  // lookahead stride / dilation

    init(
        lookbackWeight: [Float], lookaheadWeight: [Float],
        P: Int, N1: Int, S1: Int, N2: Int, S2: Int
    ) {
        precondition(
            lookbackWeight.count == P * N1,
            "FireRedFSMN: lookback weight count \(lookbackWeight.count) != P(\(P))*N1(\(N1))")
        if N2 > 0 {
            precondition(
                lookaheadWeight.count == P * N2,
                "FireRedFSMN: lookahead weight count \(lookaheadWeight.count) != P(\(P))*N2(\(N2))")
        }
        self.lookbackWeight = lookbackWeight
        self.lookaheadWeight = lookaheadWeight
        self.P = P
        self.N1 = N1
        self.S1 = S1
        self.N2 = N2
        self.S2 = S2
    }

    /// Apply the FSMN memory layer to `inputs` of shape `[T, P]` (row-major).
    /// Returns `memory + residual` of shape `[T, P]`.
    ///
    /// The lookback filter is a depthwise conv1d with kernel size N1 and
    /// dilation S1. It reads only past frames (causal), so the output at
    /// time t uses inputs[t-N1*S1 .. t-S1] (N1 taps, dilated). The
    /// lookahead filter similarly reads future frames [t+S2 .. t+N2*S2].
    func forward(_ inputs: [Float], T: Int) -> [Float] {
        precondition(inputs.count == T * P, "FireRedFSMN: input count mismatch")
        var memory = [Float](repeating: 0, count: T * P)

        // Apply lookback and lookahead per channel (depthwise).
        // Copy each channel's filter to a 0-based array to avoid slice indexing pitfalls.
        for p in 0 ..< P {
            let wLBBase = p * N1
            for t in 0 ..< T {
                var acc: Float = 0
                // Lookback: tap k (1..N1) looks back k*S1 frames.
                // The filter weight at position wLB[N1 - k] aligns the most-
                // recent tap at index 0 and the furthest tap at index N1-1
                // (matching PyTorch's depthwise lookback_filter with dilation=S1).
                for k in 1 ... N1 {
                    let src = t - k * S1
                    if src >= 0 {
                        acc += lookbackWeight[wLBBase + N1 - k] * inputs[src * P + p]
                    }
                }
                // Residual from current frame.
                memory[t * P + p] = inputs[t * P + p] + acc
            }

            // Lookahead pass (non-causal, N2 future taps).
            if N2 > 0 && !lookaheadWeight.isEmpty {
                let wLABase = p * N2
                for t in 0 ..< T {
                    var acc: Float = 0
                    for k in 1 ... N2 {
                        let src = t + k * S2
                        if src < T {
                            acc += lookaheadWeight[wLABase + k - 1] * inputs[src * P + p]
                        }
                    }
                    memory[t * P + p] += acc
                }
            }
        }
        return memory
    }
}

// ─── DFSMN block ──────────────────────────────────────────────────────

/// One DFSMNBlock — the R-1 recurrent blocks after the initial fsmn1.
///
/// Structure: fc1 (P→H, bias, ReLU) → fc2 (H→P, no bias) → FSMN → + skip.
/// Matches `DFSMNBlock.forward` in upstream `detect_model.py`.
final class FireRedDFSMNBlock: Sendable {
    let fc1: VADLinear  // [H, P]
    let fc2: VADLinear  // [P, H], no bias
    let fsmn: FireRedFSMN

    init(fc1: VADLinear, fc2: VADLinear, fsmn: FireRedFSMN) {
        self.fc1 = fc1
        self.fc2 = fc2
        self.fsmn = fsmn
    }

    /// Forward `[T, P]` → `[T, P]` with a skip connection.
    func forward(_ inputs: [Float], T: Int) -> [Float] {
        precondition(
            inputs.count == T * fc1.inFeatures,
            "FireRedDFSMNBlock: input count mismatch")
        // fc1: P → H with ReLU.
        var h = fc1.applyRows(inputs, rows: T)
        VADMath.reluInPlace(&h)
        // fc2: H → P (no bias).
        let p = fc2.applyRows(h, rows: T)
        // FSMN memory.
        let mem = fsmn.forward(p, T: T)
        // Skip connection: mem + inputs.
        var out = [Float](repeating: 0, count: inputs.count)
        for i in out.indices { out[i] = mem[i] + inputs[i] }
        return out
    }
}

// ─── Kaldi fbank feature extractor ───────────────────────────────────

/// CPU Kaldi-style filterbank (80 mel bins, 25 ms / 10 ms frames, 16 kHz).
///
/// Implements `KaldifeatFbank` from the upstream `audio_feat.py`:
///   - `snip_edges=true`: frames start at sample 0, the last complete
///     frame ends at `len - frameLengthSamples`.
///   - `dither=0`: fully deterministic.
///   - Rectangular window is NOT used; a Hann window is applied as
///     the Kaldi default (`window_type="povey"` uses a raised cosine
///     that is identical to Hann for `snip_edges=true`). The reference
///     knf.OnlineFbank matches PyTorch torchaudio Kaldi fbank.
///   - Output per frame: `log(mel_energy)`, base-e.
///
/// Note: the Kaldi fbank uses `log` (base-e) not `log10`. This matches
/// `kaldi_native_fbank` and `torchaudio.compliance.kaldi.fbank`. The
/// upstream CMVN statistics were computed on the same log-energy scale.
enum FireRedKaldiFbank {
    static let sampleRate: Int = 16000
    static let numMelBins: Int = 80
    static let frameLengthSamples: Int = 400  // 25 ms at 16 kHz
    static let frameShiftSamples: Int = 160  // 10 ms at 16 kHz

    // Precomputed once (module-level let is evaluated lazily in Swift).
    static let window: [Float] = makeWindow(frameLengthSamples)
    static let melFb: [Float] = makeMelFilterbank(
        sampleRate: sampleRate, frameLen: frameLengthSamples, numMels: numMelBins)

    /// Number of FFT bins (next power-of-two ≥ frameLengthSamples).
    private static let nFft: Int = {
        var n = 1
        while n < frameLengthSamples { n <<= 1 }
        return n
    }()

    /// Hann window ("Povey window" in Kaldi is identical for odd or even sizes
    /// when not rounded-to-power-of-two; for snip_edges this is adequate).
    private static func makeWindow(_ size: Int) -> [Float] {
        var w = [Float](repeating: 0, count: size)
        for n in 0 ..< size {
            w[n] = 0.5 - 0.5 * cosf(2 * Float.pi * Float(n) / Float(size))
        }
        return w
    }

    /// Kaldi mel scale: `1127 * ln(1 + hz / 700)` (HTK mel scale).
    private static func hzToMelKaldi(_ hz: Float) -> Float { 1127 * logf(1 + hz / 700) }
    private static func melToHzKaldi(_ mel: Float) -> Float { 700 * (expf(mel / 1127) - 1) }

    /// Build a `[numMels, nFft/2+1]` filterbank (row = mel bin, col = FFT bin).
    /// Each row is a triangular filter in the HTK mel scale, no normalization
    /// (matching Kaldi's default when `htk_compat=false`, `norm="none"`).
    private static func makeMelFilterbank(sampleRate: Int, frameLen: Int, numMels: Int) -> [Float] {
        let nFft: Int = {
            var n = 1
            while n < frameLen { n <<= 1 }
            return n
        }()
        let nFreq = nFft / 2 + 1
        let fMax = Float(sampleRate) / 2
        let melMin = hzToMelKaldi(0)
        let melMax = hzToMelKaldi(fMax)
        // numMels + 2 centre points.
        var centers = [Float](repeating: 0, count: numMels + 2)
        for i in 0 ..< (numMels + 2) {
            let mel = melMin + (melMax - melMin) * Float(i) / Float(numMels + 1)
            centers[i] = melToHzKaldi(mel)
        }
        // FFT bin centre frequencies.
        var binHz = [Float](repeating: 0, count: nFreq)
        for k in 0 ..< nFreq { binHz[k] = Float(k) * Float(sampleRate) / Float(nFft) }

        var fb = [Float](repeating: 0, count: numMels * nFreq)
        for m in 0 ..< numMels {
            let lo = centers[m]
            let ctr = centers[m + 1]
            let hi = centers[m + 2]
            for k in 0 ..< nFreq {
                let hz = binHz[k]
                var w: Float = 0
                if hz >= lo && hz <= ctr && ctr > lo {
                    w = (hz - lo) / (ctr - lo)
                } else if hz > ctr && hz <= hi && hi > ctr {
                    w = (hi - hz) / (hi - ctr)
                }
                fb[m * nFreq + k] = max(0, w)
            }
        }
        return fb
    }

    /// Compute the 80-bin log-mel fbank features for a 16 kHz int16 signal.
    ///
    /// Returns `[numFrames, 80]` row-major (each row = one analysis frame).
    /// Uses `snip_edges=true`: the number of frames is
    /// `(len - frameLengthSamples) / frameShiftSamples + 1`.
    static func extract(_ samples: [Int16]) -> [[Float]] {
        let N = samples.count
        guard N >= frameLengthSamples else { return [] }

        let nFft: Int = {
            var n = 1
            while n < frameLengthSamples { n <<= 1 }
            return n
        }()
        let nFreq = nFft / 2 + 1
        let numFrames = (N - frameLengthSamples) / frameShiftSamples + 1
        var features = [[Float]](
            repeating: [Float](repeating: 0, count: numMelBins),
            count: numFrames)

        // Pre-scale int16 to float. No per-sample normalisation — Kaldi
        // leaves the waveform in raw PCM counts (range ≈ ±32768), which
        // affects the absolute log-mel energy but is absorbed by CMVN.
        let floatSamples = samples.map { Float($0) }

        for f in 0 ..< numFrames {
            let start = f * frameShiftSamples
            var frame = [Float](repeating: 0, count: nFft)
            // Apply window to the frame.
            for n in 0 ..< frameLengthSamples {
                frame[n] = floatSamples[start + n] * window[n]
            }
            // Real DFT → power spectrum (naive O(N²) — fine for nFft≤512).
            var power = [Float](repeating: 0, count: nFreq)
            for k in 0 ..< nFreq {
                var re: Float = 0
                var im: Float = 0
                let w = -2 * Float.pi * Float(k) / Float(nFft)
                for n in 0 ..< frameLengthSamples {
                    let angle = w * Float(n)
                    re += frame[n] * cosf(angle)
                    im += frame[n] * sinf(angle)
                }
                power[k] = re * re + im * im
            }
            // Apply mel filterbank: each mel bin sums weighted power bins.
            var mels = [Float](repeating: 0, count: numMelBins)
            let floor: Float = 1e-10  // matches Kaldi's energy floor
            for m in 0 ..< numMelBins {
                var energy: Float = 0
                let base = m * nFreq
                for k in 0 ..< nFreq { energy += melFb[base + k] * power[k] }
                mels[m] = logf(max(energy, floor))
            }
            features[f] = mels
        }
        return features
    }

    /// Convenience overload for float samples in `[-1, 1]`.
    /// Scales to int16 range before extraction.
    static func extractFloat(_ samples: [Float]) -> [[Float]] {
        let int16 = samples.map { Int16(max(-32768, min(32767, $0 * 32768))) }
        return extract(int16)
    }
}

// ─── CMVN ─────────────────────────────────────────────────────────────

/// Cepstral mean-variance normalization for 80-dim features.
///
/// Stores precomputed `mean[80]` and `invStd[80]` so the normalisation
/// reduces to an element-wise `(x - mean) * invStd`. Computed from the
/// Kaldi binary ark shipped alongside the checkpoint (`cmvn.ark`).
struct FireRedCMVN: Sendable {
    let mean: [Float]
    let invStd: [Float]
    let dim: Int

    init(mean: [Float], invStd: [Float]) {
        precondition(mean.count == invStd.count, "FireRedCMVN: mean/invStd length mismatch")
        self.mean = mean
        self.invStd = invStd
        self.dim = mean.count
    }

    /// Normalise a `[numFrames, dim]` feature matrix in-place.
    func apply(_ features: inout [[Float]]) {
        for f in features.indices {
            precondition(features[f].count == dim, "FireRedCMVN: feature dim mismatch")
            for d in 0 ..< dim {
                features[f][d] = (features[f][d] - mean[d]) * invStd[d]
            }
        }
    }

    /// Hardcoded CMVN parameters extracted from the `FireRedTeam/FireRedVAD`
    /// `VAD/cmvn.ark`. Stored here so the loader works without a Kaldi
    /// parser; if a future checkpoint ships different stats, decode them
    /// from `cmvn.ark` in `loadFromDirectory` and pass them to the `init`.
    static let defaultMean: [Float] = [
        10.42295175, 10.86209741, 11.76454438, 12.49016470, 13.25983008,
        13.89594383, 14.36494024, 14.59394835, 14.74972360, 14.66831535,
        14.73079672, 14.77505246, 14.98905198, 15.17800493, 15.25352031,
        15.32863705, 15.33401859, 15.28864170, 15.42766169, 15.24626616,
        15.09257380, 15.29042194, 15.07575009, 15.18677287, 15.08867324,
        15.17079740, 15.07017809, 15.15079534, 15.10853283, 15.11534508,
        15.14127999, 15.13183236, 15.14519587, 15.19151893, 15.23547867,
        15.30636975, 15.37302148, 15.41639463, 15.45985744, 15.39143273,
        15.46357624, 15.39966121, 15.46290792, 15.44162912, 15.48496953,
        15.55240178, 15.63809193, 15.70548935, 15.76700885, 15.85512378,
        15.86726978, 15.89153741, 15.92314483, 15.97838261, 16.01480167,
        16.04867494, 16.08202991, 16.09680075, 16.09373669, 16.07247920,
        16.07550966, 16.02227088, 15.97676210, 15.89786455, 15.81274164,
        15.71120511, 15.60419889, 15.55351944, 15.51025275, 15.46002382,
        15.41568436, 15.37602765, 15.32834898, 15.29537080, 15.18547019,
        15.01704498, 14.90508003, 14.62380657, 14.13809381, 13.31387035,
    ]

    static let defaultInvStd: [Float] = [
        0.24949809, 0.23563235, 0.23145153, 0.23322339, 0.23182660,
        0.22853357, 0.22434870, 0.21898920, 0.21832438, 0.22082593,
        0.22296736, 0.22288416, 0.22234811, 0.22100643, 0.21994202,
        0.22005444, 0.22070092, 0.22150810, 0.22236667, 0.22305292,
        0.22335342, 0.22438906, 0.22547702, 0.22690076, 0.22823023,
        0.22931472, 0.23046728, 0.23083553, 0.23143383, 0.23220659,
        0.23257989, 0.23361970, 0.23437241, 0.23508252, 0.23578079,
        0.23589200, 0.23602098, 0.23663800, 0.23749876, 0.23798452,
        0.23899378, 0.23974815, 0.24030836, 0.24097694, 0.24143249,
        0.24135466, 0.24079938, 0.24047405, 0.23995525, 0.23952288,
        0.23948089, 0.23936509, 0.23929339, 0.23902199, 0.23857873,
        0.23814702, 0.23804621, 0.23824194, 0.23860096, 0.23915407,
        0.23922541, 0.23938308, 0.23973360, 0.23960562, 0.24028503,
        0.24061813, 0.24067930, 0.24096202, 0.24043606, 0.24021527,
        0.23972514, 0.23871998, 0.23744131, 0.23619509, 0.23337281,
        0.22680233, 0.22577503, 0.22503847, 0.22631137, 0.22899493,
    ]

    static let `default` = FireRedCMVN(mean: defaultMean, invStd: defaultInvStd)
}

// ─── DFSMN + output head ──────────────────────────────────────────────

/// Full DFSMN model: DFSMN stack → DNN → Linear → sigmoid.
/// Holds host-resident `[Float]` weights; all computation is CPU-side
/// via `VADLinear` / `FireRedFSMN` / `FireRedDFSMNBlock` — same strategy
/// as SileroVAD and SmartTurn.
final class FireRedDFSMN: Sendable {
    // Initial projection layers.
    let fc1: VADLinear  // [H, idim]
    let fc2: VADLinear  // [P, H]
    // First FSMN block (no skip connection).
    let fsmn1: FireRedFSMN
    // R-1 DFSMN blocks (with skip).
    let blocks: [FireRedDFSMNBlock]
    // M DNN layers after the DFSMN stack.
    let dnns: [VADLinear]
    // Output head.
    let outLinear: VADLinear  // [odim, H_dnn]

    let idim: Int
    let H: Int
    let P: Int

    init(
        fc1: VADLinear, fc2: VADLinear,
        fsmn1: FireRedFSMN,
        blocks: [FireRedDFSMNBlock],
        dnns: [VADLinear],
        outLinear: VADLinear,
        idim: Int, H: Int, P: Int
    ) {
        self.fc1 = fc1
        self.fc2 = fc2
        self.fsmn1 = fsmn1
        self.blocks = blocks
        self.dnns = dnns
        self.outLinear = outLinear
        self.idim = idim
        self.H = H
        self.P = P
    }

    /// Forward `features` shaped `[T, idim]` → per-frame speech probabilities `[T]`.
    ///
    /// Mirrors `DetectModel.forward` + `DFSMN.forward`:
    ///   1. fc1(idim→H) + ReLU
    ///   2. fc2(H→P) + ReLU
    ///   3. fsmn1 memory layer
    ///   4. DFSMNBlock × (R-1)
    ///   5. DNN layers (P→H, bias, ReLU; extra H→H layers if M>1)
    ///   6. out(H→odim) + sigmoid → [T]
    func forward(_ features: [[Float]]) -> [Float] {
        let T = features.count
        guard T > 0 else { return [] }
        // Flatten to [T × idim] row-major.
        let x = features.flatMap { $0 }

        // fc1: idim → H, ReLU.
        var h = fc1.applyRows(x, rows: T)
        VADMath.reluInPlace(&h)

        // fc2: H → P, ReLU.
        var p = fc2.applyRows(h, rows: T)
        VADMath.reluInPlace(&p)

        // fsmn1 memory layer.
        p = fsmn1.forward(p, T: T)

        // DFSMN blocks.
        for block in blocks {
            p = block.forward(p, T: T)
        }

        // DNN layers (P → H, bias, ReLU; and any extra H → H layers).
        var dnnOut = dnns[0].applyRows(p, rows: T)
        VADMath.reluInPlace(&dnnOut)
        for dnn in dnns.dropFirst() {
            dnnOut = dnn.applyRows(dnnOut, rows: T)
            VADMath.reluInPlace(&dnnOut)
        }

        // Output head + sigmoid.
        let logits = outLinear.applyRows(dnnOut, rows: T)
        return logits.map { VADMath.sigmoid($0) }
    }
}

// ─── Post-processor ───────────────────────────────────────────────────

/// Convert a raw per-frame probability stream into speech segments using
/// the FireRedVAD state machine + smoothing + hysteresis. Mirrors
/// `VadPostprocessor.process` + `decision_to_segment` in upstream.
///
/// Frame rate: 100 fps (10 ms shift at 16 kHz).
/// 1 frame = 10 ms = 160 samples at 16 kHz.
enum FireRedVADPostprocessor {
    /// Frame shift in seconds — 10 ms.
    static let frameShiftSeconds: Double = 0.01
    /// Frame length in seconds — 25 ms.
    static let frameLengthSeconds: Double = 0.025

    // ─── Smoothing ───────────────────────────────────────────────

    /// Causal rolling-average smoothing over `[0 .. i]` (first
    /// `windowSize - 1` frames use a growing window, matching the
    /// upstream `_smooth_prob` with `mode='full'[:len]` + boundary fix).
    static func smooth(_ probs: [Float], windowSize: Int) -> [Float] {
        guard windowSize > 1 else { return probs }
        var smoothed = [Float](repeating: 0, count: probs.count)
        var windowSum: Float = 0
        for i in probs.indices {
            windowSum += probs[i]
            let count = min(i + 1, windowSize)
            // The upstream does full convolution then boundary-corrects;
            // cumulative average over the first windowSize-1 frames is
            // equivalent to the boundary correction applied there.
            if i >= windowSize {
                windowSum -= probs[i - windowSize]
                smoothed[i] = windowSum / Float(windowSize)
            } else {
                smoothed[i] = windowSum / Float(count)
            }
        }
        return smoothed
    }

    // ─── State-machine hysteresis ────────────────────────────────

    /// VAD state for the hysteresis machine (matching upstream `VadState`).
    private enum VadState { case silence, possibleSpeech, speech, possibleSilence }

    /// Apply min-speech / min-silence frame constraints via a state machine.
    /// Returns a binary decision array (0 = silence, 1 = speech).
    static func stateMachineDecisions(
        _ binary: [Int], minSpeechFrame: Int, minSilenceFrame: Int
    ) -> [Int] {
        var decisions = [Int](repeating: 0, count: binary.count)
        var state: VadState = .silence
        var speechStart = -1
        var silenceStart = -1

        for (t, isSpeech) in binary.enumerated() {
            let speech = isSpeech != 0
            switch state {
            case .silence:
                if speech {
                    state = .possibleSpeech
                    speechStart = t
                }
            case .possibleSpeech:
                if speech {
                    if t - speechStart >= minSpeechFrame {
                        state = .speech
                        for tt in speechStart ..< t { decisions[tt] = 1 }
                    }
                } else {
                    state = .silence
                    speechStart = -1
                }
            case .speech:
                if !speech {
                    state = .possibleSilence
                    silenceStart = t
                }
            case .possibleSilence:
                if !speech {
                    if t - silenceStart >= minSilenceFrame {
                        state = .silence
                        speechStart = -1
                    }
                } else {
                    state = .speech
                    silenceStart = -1
                }
            }
            // Assign current frame.
            if state == .speech || state == .possibleSilence {
                decisions[t] = 1
            }
        }
        return decisions
    }

    // ─── Fix smooth-window start ─────────────────────────────────

    /// Back-fill the `smoothWindowSize` frames before each speech onset
    /// to account for the causal smoothing delay. Matches upstream
    /// `_fix_smooth_window_start`.
    static func fixSmoothWindowStart(_ decisions: [Int], windowSize: Int) -> [Int] {
        var out = decisions
        for t in 1 ..< decisions.count {
            if decisions[t - 1] == 0 && decisions[t] == 1 {
                let start = max(0, t - windowSize)
                for tt in start ..< t { out[tt] = 1 }
            }
        }
        return out
    }

    // ─── Merge short silence segments ────────────────────────────

    /// Merge silence gaps shorter than `mergeFrames`. Matches upstream
    /// `_merge_short_silence_segments`.
    static func mergeShortSilence(_ decisions: [Int], mergeFrames: Int) -> [Int] {
        guard mergeFrames > 0 else { return decisions }
        var out = decisions
        var silenceStart: Int? = nil
        for t in 1 ..< decisions.count {
            if decisions[t - 1] == 1 && decisions[t] == 0 && silenceStart == nil {
                silenceStart = t
            } else if decisions[t - 1] == 0 && decisions[t] == 1, let ss = silenceStart {
                if t - ss < mergeFrames {
                    for tt in ss ..< t { out[tt] = 1 }
                }
                silenceStart = nil
            }
        }
        return out
    }

    // ─── Split long speech segments ──────────────────────────────

    /// Split any speech segment longer than `maxFrame` at the lowest-
    /// probability frame in the mid-to-end region. Matches upstream
    /// `_split_long_speech_segments`.
    static func splitLongSpeech(_ decisions: [Int], probs: [Float], maxFrame: Int) -> [Int] {
        var out = decisions
        // Collect (start, end) of each speech run.
        var runs: [(Int, Int)] = []
        var inSpeech = false
        var runStart = 0
        for t in 0 ..< decisions.count {
            if decisions[t] == 1 && !inSpeech {
                inSpeech = true
                runStart = t
            } else if decisions[t] == 0 && inSpeech {
                runs.append((runStart, t))
                inSpeech = false
            }
        }
        if inSpeech { runs.append((runStart, decisions.count)) }

        for (s, e) in runs {
            let durFrames = e - s
            if durFrames > maxFrame {
                // Find split points using the same rolling scheme as upstream.
                var splitPoints: [Int] = []
                var start = s
                while start < e {
                    let remaining = e - start
                    if remaining <= maxFrame { break }
                    let winStart = start + maxFrame / 2
                    let winEnd = min(start + maxFrame, e)
                    if winStart >= winEnd { break }
                    let slice = probs[winStart ..< winEnd]
                    if let minOffset = slice.enumerated().min(by: { $0.element < $1.element })?
                        .offset
                    {
                        let splitIdx = winStart + minOffset
                        splitPoints.append(splitIdx)
                        start = splitIdx + 1
                    } else {
                        break
                    }
                }
                for sp in splitPoints { out[sp] = 0 }
            }
        }
        return out
    }

    // ─── Decisions → segments ────────────────────────────────────

    /// Convert a binary decision array to `VADSpeechSegment` values.
    /// Matches `VadPostprocessor.decision_to_segment` + the sample-
    /// index conversion.
    static func decisionsToSegments(
        _ decisions: [Int], audioDurationSeconds: Double,
        sampleRate: Int, frameShiftSamples: Int
    ) -> [VADSpeechSegment] {
        var segments: [VADSpeechSegment] = []
        var speechStart: Int? = nil

        for (t, decision) in decisions.enumerated() {
            if decision == 1 && speechStart == nil {
                speechStart = t
            } else if decision == 0, let ss = speechStart {
                let startSample = ss * frameShiftSamples
                let endSample = t * frameShiftSamples
                segments.append(
                    VADSpeechSegment(
                        startSample: startSample, endSample: endSample,
                        sampleRate: sampleRate))
                speechStart = nil
            }
        }
        // Trailing speech run.
        if let ss = speechStart {
            let startSample = ss * frameShiftSamples
            let audioDurationSamples = Int(audioDurationSeconds * Double(sampleRate))
            let endSample = min(audioDurationSamples, decisions.count * frameShiftSamples)
            segments.append(
                VADSpeechSegment(
                    startSample: startSample, endSample: endSample,
                    sampleRate: sampleRate))
        }
        return segments
    }

    // ─── Full pipeline ───────────────────────────────────────────

    /// Run the full post-processing pipeline on raw per-frame probs.
    static func process(
        probs: [Float],
        config: FireRedVADConfig,
        audioDurationSeconds: Double,
        sampleRate: Int
    ) -> [VADSpeechSegment] {
        guard !probs.isEmpty else { return [] }

        // 1. Smooth probabilities.
        let smoothed = smooth(probs, windowSize: config.smoothWindowSize)

        // 2. Binary threshold.
        let binary = smoothed.map { $0 >= config.speechThreshold ? 1 : 0 }

        // 3. State-machine hysteresis.
        var decisions = stateMachineDecisions(
            binary, minSpeechFrame: config.minSpeechFrame,
            minSilenceFrame: config.minSilenceFrame)

        // 4. Fix smooth-window start delay.
        decisions = fixSmoothWindowStart(decisions, windowSize: config.smoothWindowSize)

        // 5. Merge short silence gaps.
        decisions = mergeShortSilence(decisions, mergeFrames: config.mergeSilenceFrame)

        // 6. Split long speech segments.
        decisions = splitLongSpeech(decisions, probs: probs, maxFrame: config.maxSpeechFrame)

        // 7. Convert to segments.
        return decisionsToSegments(
            decisions, audioDurationSeconds: audioDurationSeconds,
            sampleRate: sampleRate, frameShiftSamples: config.frameShiftSamples)
    }
}

// ─── FireRedVADModel ──────────────────────────────────────────────────

/// Loaded FireRedVAD model. Audio-in / speech-probability-out — reached
/// via `VADModelRegistry`, not `ModelRegistry`. `fromPretrained` downloads
/// the `FireRedTeam/FireRedVAD` HuggingFace snapshot and loads weights
/// directly from the PyTorch `.pth.tar` archive (no mlx-community
/// conversion needed).
///
/// No mlx-community safetensors checkpoint exists as of 2026-05-22;
/// when one appears, add a `SafeTensorsBundle`-based loader path in
/// `loadFromDirectory` gated on the presence of `*.safetensors`.
public final class FireRedVADModel: @unchecked Sendable {
    public let config: FireRedVADConfig
    let dfsmn: FireRedDFSMN
    let cmvn: FireRedCMVN

    init(config: FireRedVADConfig, dfsmn: FireRedDFSMN, cmvn: FireRedCMVN) {
        self.config = config
        self.dfsmn = dfsmn
        self.cmvn = cmvn
    }

    // ─── Detection ───────────────────────────────────────────────

    /// Run VAD over a mono `audio` clip and return the per-frame speech
    /// probability stream plus post-processed speech segments.
    ///
    /// - Parameters:
    ///   - audio: Mono PCM samples in `[-1, 1]`.
    ///   - sampleRate: Must be 16000 Hz.
    public func detect(audio: [Float], sampleRate: Int = 16000) throws -> VADOutput {
        guard sampleRate == 16000 else {
            throw FireRedVADError.unsupportedSampleRate(sampleRate)
        }
        guard !audio.isEmpty else {
            return VADOutput(
                probabilities: [], frameStrideSamples: config.frameShiftSamples,
                sampleRate: sampleRate, segments: [])
        }
        let audioDuration = Double(audio.count) / Double(sampleRate)

        // Extract Kaldi fbank features (80 mel bins, 25 ms / 10 ms frames).
        var features = FireRedKaldiFbank.extractFloat(audio)
        guard !features.isEmpty else {
            return VADOutput(
                probabilities: [], frameStrideSamples: config.frameShiftSamples,
                sampleRate: sampleRate, segments: [])
        }

        // Apply CMVN normalisation.
        cmvn.apply(&features)

        // Run the DFSMN model.
        let probs = dfsmn.forward(features)

        // Post-process to speech segments.
        let segments = FireRedVADPostprocessor.process(
            probs: probs, config: config,
            audioDurationSeconds: audioDuration,
            sampleRate: sampleRate)

        return VADOutput(
            probabilities: probs,
            frameStrideSamples: config.frameShiftSamples,
            sampleRate: sampleRate,
            segments: segments)
    }

    // ─── PyTorch checkpoint reader ────────────────────────────────

    /// Read all float32 tensors from a PyTorch `.pth.tar` file (which is
    /// a standard zip archive). Returns a `[name: [Float]]` dictionary.
    ///
    /// The format used by `torch.save` (v2): a zip containing `data.pkl`
    /// (a pickle manifest) and `data/N` (raw little-endian float32 blobs
    /// indexed by numeric string). We parse the pickle for the `(name,
    /// storage_id, shape)` triples and read the blobs directly —
    /// no torch dependency needed.
    ///
    /// Only the `model_state_dict` portion is extracted; `args` is
    /// ignored (hyperparameters come from `config.json` or defaults).
    static func readPthTar(at url: URL) throws -> [String: [Float]] {
        guard let zip = try? Data(contentsOf: url) else {
            throw FireRedVADError.unsupportedCheckpointFormat
        }
        // Verify zip magic.
        guard zip.count >= 4,
            zip[0] == 0x50, zip[1] == 0x4B, zip[2] == 0x03, zip[3] == 0x04
        else { throw FireRedVADError.unsupportedCheckpointFormat }

        // Build a table of { "name_in_zip" → data } for all files.
        var zipFiles: [String: Data] = [:]
        var i = 0
        while i + 30 <= zip.count {
            // Local file header signature.
            guard zip[i] == 0x50 && zip[i + 1] == 0x4B && zip[i + 2] == 0x03 && zip[i + 3] == 0x04
            else {
                i += 1
                continue
            }
            let compMethod = Int(zip[i + 8]) | (Int(zip[i + 9]) << 8)
            let compSize =
                Int(zip[i + 18]) | (Int(zip[i + 19]) << 8)
                | (Int(zip[i + 20]) << 16) | (Int(zip[i + 21]) << 24)
            let uncompSize =
                Int(zip[i + 22]) | (Int(zip[i + 23]) << 8)
                | (Int(zip[i + 24]) << 16) | (Int(zip[i + 25]) << 24)
            let fnLen = Int(zip[i + 26]) | (Int(zip[i + 27]) << 8)
            let extraLen = Int(zip[i + 28]) | (Int(zip[i + 29]) << 8)
            guard i + 30 + fnLen + extraLen <= zip.count else { break }
            let fnBytes = zip[(i + 30) ..< (i + 30 + fnLen)]
            let fname = String(bytes: fnBytes, encoding: .utf8) ?? ""
            let dataStart = i + 30 + fnLen + extraLen
            let dataEnd = dataStart + compSize
            guard dataEnd <= zip.count else { break }

            if compMethod == 0 {
                // Stored (no compression).
                zipFiles[fname] = zip[dataStart ..< dataStart + uncompSize]
            }
            // Skip deflated entries (we only need the pickle + blobs which
            // are stored uncompressed in standard torch.save output).
            i = dataEnd
        }

        // Locate the data.pkl and blob directory prefix.
        // torch.save uses "archive/data.pkl" (v2) or "model.pth/data.pkl".
        let pklKey = zipFiles.keys.first { $0.hasSuffix("/data.pkl") }
        guard let pklKey else { throw FireRedVADError.unsupportedCheckpointFormat }
        let blobPrefix = String(pklKey.dropLast("data.pkl".count))  // e.g. "model.pth/"
        guard let pklData = zipFiles[pklKey] else {
            throw FireRedVADError.unsupportedCheckpointFormat
        }

        // Parse the pickle to extract { tensor_name → (storage_id, element_count, shape) }.
        // We parse a minimal subset of the pickle protocol: GLOBAL, BINUNICODE,
        // SHORT_BINUNICODE, BININT1, BININT2, BININT4, TUPLE1, TUPLE2, TUPLE3,
        // REDUCE, BINPUT, BINGET, MARK, SETITEMS, BUILD, EMPTY_DICT, EMPTY_TUPLE,
        // NEWFALSE, NEWTRUE, and STOP.
        var tensorEntries: [String: (id: String, count: Int, shape: [Int])] = [:]
        parsePklEntries(pklData, into: &tensorEntries)

        // Read blob data and convert to [Float].
        var result: [String: [Float]] = [:]
        for (name, entry) in tensorEntries {
            let blobKey = blobPrefix + "data/" + entry.id
            guard let blob = zipFiles[blobKey] else { continue }
            let floats = blob.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Float.self).prefix(entry.count))
            }
            result[name] = floats
        }
        return result
    }

    /// Parse the `data.pkl` bytes to extract model state dict tensor
    /// metadata. Fills `entries` with `{ name → (storageId, count, shape) }`.
    ///
    /// The pickle stream for `torch.save` encodes each tensor as a call to
    /// `torch._utils._rebuild_tensor_v2(storage, offset, shape, stride, ...)`.
    /// We track the sequence of opcodes to extract: the tensor key name
    /// (BINUNICODE), the storage id (BINUNICODE inside the storage PersistId),
    /// the element count (BININT inside the storage tuple), and the shape
    /// tuple (TUPLE2/TUPLE3/TUPLE1/EMPTY_TUPLE after REDUCE).
    private static func parsePklEntries(
        _ pkl: Data,
        into entries: inout [String: (id: String, count: Int, shape: [Int])]
    ) {
        // Minimal pickle opcode scanner — we only need tensor entries.
        // State machine tracks: "last seen dict key" → used when SETITEMS fires.
        var pos = 0

        // Stack of values we've seen.
        var stack: [PickleVal] = []
        // The MARK stack index (for SETITEMS/BUILD).
        var marks: [Int] = []
        // Map from BINPUT memos.
        var memo: [Int: PickleVal] = [:]
        // The current dict key being processed.
        var currentKey: String? = nil

        func readByte() -> UInt8? {
            guard pos < pkl.count else { return nil }
            let b = pkl[pos]
            pos += 1
            return b
        }
        func readUInt16LE() -> Int {
            let lo = Int(pkl[pos])
            let hi = Int(pkl[pos + 1])
            pos += 2
            return lo | (hi << 8)
        }
        func readUInt32LE() -> Int {
            // 4-byte LE read via a loop — same compile-time
            // workaround as BINUNICODE8 above (Swift 5.10 type checker
            // chokes on the inline 4-way Int(...) | <<shift expression).
            var v = 0
            for i in 0 ..< 4 { v |= Int(pkl[pos + i]) << (8 * i) }
            pos += 4
            return v
        }
        func readLen1String() -> String {
            let n = Int(pkl[pos])
            pos += 1
            let s = String(bytes: pkl[pos ..< (pos + n)], encoding: .utf8) ?? ""
            pos += n
            return s
        }
        func readLen4String() -> String {
            let n = readUInt32LE()
            let s = String(bytes: pkl[pos ..< (pos + n)], encoding: .utf8) ?? ""
            pos += n
            return s
        }
        func readNewlineString() -> String {
            var end = pos
            while end < pkl.count && pkl[end] != 0x0A { end += 1 }
            let s = String(bytes: pkl[pos ..< end], encoding: .utf8) ?? ""
            pos = end + 1
            return s
        }

        while pos < pkl.count {
            guard let op = readByte() else { break }
            switch op {
            case 0x80: _ = readByte()  // PROTO
            case 0x28: marks.append(stack.count)  // MARK
            case 0x2E: return  // STOP
            // Push values
            case 0x4E: stack.append(.none_val)  // NONE
            case 0x89: stack.append(.bool(false))  // NEWFALSE
            case 0x88: stack.append(.bool(true))  // NEWTRUE
            case 0x4B:  // BININT1
                stack.append(.int(Int(readByte() ?? 0)))
            case 0x4D:  // BININT2
                stack.append(.int(readUInt16LE()))
            case 0x4A:  // BININT4
                stack.append(.int(readUInt32LE()))
            case 0x49:  // INT (ascii)
                let s = readNewlineString()
                stack.append(.int(Int(s) ?? 0))
            case 0x47:  // BINFLOAT (8 bytes big-endian double)
                pos += 8
                stack.append(.none_val)
            case 0x46:  // FLOAT (ascii)
                _ = readNewlineString()
                stack.append(.none_val)
            case 0x53:  // STRING (ascii, quoted)
                _ = readNewlineString()
                stack.append(.none_val)
            case 0x54:  // BINSTRING len4
                let n = readUInt32LE()
                let s = String(bytes: pkl[pos ..< (pos + n)], encoding: .utf8) ?? ""
                pos += n
                stack.append(.str(s))
            case 0x55:  // SHORT_BINSTRING len1
                let n = Int(readByte() ?? 0)
                let s = String(bytes: pkl[pos ..< (pos + n)], encoding: .utf8) ?? ""
                pos += n
                stack.append(.str(s))
            case 0x58:  // BINUNICODE len4
                stack.append(.str(readLen4String()))
            case 0x8C:  // SHORT_BINUNICODE len1
                stack.append(.str(readLen1String()))
            case 0x8D:  // BINUNICODE8
                // Read 8 bytes little-endian into an Int. The full
                // expression `Int(pkl[pos]) | ... | (Int(pkl[pos+7]) <<
                // 56)` trips Swift 5.10's type checker with "unable to
                // type-check in reasonable time" on the CI toolchain
                // (Xcode 16.4); the loop form is identical at runtime
                // and compiles in ~no time.
                var n = 0
                for i in 0 ..< 8 { n |= Int(pkl[pos + i]) << (8 * i) }
                pos += 8
                let s = String(bytes: pkl[pos ..< (pos + n)], encoding: .utf8) ?? ""
                pos += n
                stack.append(.str(s))
            // Tuples
            case 0x28 + 4:  // TUPLE (MARK items TUPLE = 0x74)
                guard let markIdx = marks.popLast() else { break }
                let items = Array(stack[markIdx...])
                stack.removeSubrange(markIdx...)
                stack.append(.tuple(items))
            case 0x74:  // TUPLE (same code — ASCII 't')
                guard let markIdx = marks.popLast() else { break }
                let items = Array(stack[markIdx...])
                stack.removeSubrange(markIdx...)
                stack.append(.tuple(items))
            case 0x85:  // TUPLE1
                if let top = stack.popLast() { stack.append(.tuple([top])) }
            case 0x86:  // TUPLE2
                if stack.count >= 2 {
                    let b = stack.removeLast()
                    let a = stack.removeLast()
                    stack.append(.tuple([a, b]))
                }
            case 0x87:  // TUPLE3
                if stack.count >= 3 {
                    let c = stack.removeLast()
                    let b = stack.removeLast()
                    let a = stack.removeLast()
                    stack.append(.tuple([a, b, c]))
                }
            case 0x29: stack.append(.tuple([]))  // EMPTY_TUPLE
            case 0x7D: stack.append(.dict)  // EMPTY_DICT
            case 0x5D: stack.append(.list)  // EMPTY_LIST
            // Memo
            case 0x71:  // BINPUT (1 byte id)
                let id = Int(readByte() ?? 0)
                if let top = stack.last { memo[id] = top }
            case 0x72:  // LONG_BINPUT (4 bytes id)
                let id = readUInt32LE()
                if let top = stack.last { memo[id] = top }
            case 0x68:  // BINGET (1 byte id)
                let id = Int(readByte() ?? 0)
                stack.append(memo[id] ?? .none_val)
            case 0x6A:  // LONG_BINGET (4 bytes id)
                let id = readUInt32LE()
                stack.append(memo[id] ?? .none_val)
            // Global / reduce
            case 0x63:  // GLOBAL "module\nname\n"
                _ = readNewlineString()
                _ = readNewlineString()
                stack.append(.global_fn)
            case 0x52:  // REDUCE (fn, args) → call
                guard stack.count >= 2 else { break }
                let args = stack.removeLast()
                _ = stack.removeLast()
                // If this is a PersistId call (the storage tuple), extract the id + count.
                // The storage PersistId fires as a result pushed onto the stack marked
                // .storage(id, count).
                if case .tuple(let items) = args, items.count >= 4,
                    case .str(let tag) = items[0], tag == "storage",
                    case .str(let storageId) = items[2],
                    case .int(let count) = items[4]
                {
                    stack.append(.storage(id: storageId, count: count))
                } else {
                    stack.append(.reduced)
                }
            case 0x51:  // NEWOBJ (cls, args) — treat like REDUCE
                guard stack.count >= 2 else { break }
                _ = stack.removeLast()
                _ = stack.removeLast()
                stack.append(.reduced)
            case 0x80 + 1:  // NEWOBJ_EX (0x81)
                guard stack.count >= 3 else { break }
                _ = stack.removeLast()
                _ = stack.removeLast()
                _ = stack.removeLast()
                stack.append(.reduced)
            case 0x62:  // BUILD (obj, state) — sets state on top-1
                guard stack.count >= 2 else { break }
                _ = stack.removeLast()
                // If the top is a tensor being built, try to extract shape.
                _ = stack.removeLast()
                // The tensor is built from (storage, offset, shape, stride, …).
                // We handle this in the REDUCE case when we see the specific
                // _rebuild_tensor_v2 pattern below; BUILD is for other objects.
                stack.append(.reduced)
            // Dict / list operations
            case 0x75:  // SETITEMS (mark key val key val ...)
                guard let markIdx = marks.popLast() else { break }
                var idx = markIdx
                while idx + 1 < stack.count {
                    if case .str(let k) = stack[idx] {
                        currentKey = k
                    }
                    idx += 2
                }
                stack.removeSubrange(markIdx...)
            case 0x73:  // SETITEM (k v on top)
                if stack.count >= 2 {
                    let val = stack.removeLast()
                    let key = stack.removeLast()
                    if case .str(let k) = key { currentKey = k }
                    _ = val
                }
            case 0x61:  // APPENDS (list mark items)
                if let markIdx = marks.popLast() {
                    stack.removeSubrange(markIdx...)
                }
            case 0x65:  // APPENDS (alternate)
                if let markIdx = marks.popLast() {
                    stack.removeSubrange(markIdx...)
                }
            case 0x60:  // POP
                _ = stack.popLast()
            case 0x32:  // POP_MARK
                if let markIdx = marks.popLast() { stack.removeSubrange(markIdx...) }
            default: break
            }

            // After each REDUCE, check if the top of stack is a tensor reconstruction
            // (torch._utils._rebuild_tensor_v2). We detect this heuristically:
            // The REDUCE for rebuild_tensor_v2 is called with args =
            // (storage, offset, shape_tuple, stride_tuple, ...). When the stack top
            // is `.reduced` and the current key is a weight name, try to extract the
            // shape from the previously pushed storage + tuple.
            if op == 0x52 || op == 0x51 {
                // Peek at what we just pushed and reconstruct tensor info if possible.
                tryExtractTensorEntry(
                    stack: stack, key: currentKey, into: &entries)
            }
        }
    }

    /// Attempt to extract a tensor entry from the current stack state.
    /// Called after each REDUCE. Looks for a `.storage` + shape pattern.
    private static func tryExtractTensorEntry(
        stack: [PickleVal], key: String?,
        into entries: inout [String: (id: String, count: Int, shape: [Int])]
    ) {
        guard let name = key, !name.isEmpty else { return }
        // The _rebuild_tensor_v2 args tuple is: (storage, offset, shape, stride, ...)
        // We need the storage (id + count) and the shape.
        // Find the deepest `.storage` and the tuple after it.
        for i in stride(from: stack.count - 1, through: 0, by: -1) {
            if case .storage(let id, let count) = stack[i] {
                // Look ahead for shape tuple.
                for j in (i + 1) ..< stack.count {
                    if case .tuple(let items) = stack[j] {
                        let shape = items.compactMap { (v: PickleVal) -> Int? in
                            if case .int(let n) = v { return n }
                            return nil
                        }
                        if !shape.isEmpty {
                            entries[name] = (id: id, count: count, shape: shape)
                            return
                        }
                    }
                }
                // No shape tuple found yet — entry will be filled when shape arrives.
                // Store with empty shape as a placeholder.
                if entries[name] == nil {
                    entries[name] = (id: id, count: count, shape: [])
                }
                return
            }
        }
    }

    // ─── Loading ─────────────────────────────────────────────────

    /// Load a FireRedVAD checkpoint from a local snapshot directory.
    ///
    /// The directory must contain `model.pth.tar` (PyTorch checkpoint)
    /// and `cmvn.ark` (Kaldi CMVN stats). `config.json` is optional.
    ///
    /// When an mlx-community safetensors conversion appears, update this
    /// method to detect `*.safetensors` and load via `SafeTensorsBundle`.
    public static func loadFromDirectory(
        _ directory: URL,
        device _: Device = .shared
    ) throws -> FireRedVADModel {
        // Optional config.json — use published defaults if absent.
        var config = FireRedVADConfig()
        let configURL = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
            let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        {
            config = FireRedVADConfig.decode(from: raw)
        }

        // Load PyTorch checkpoint.
        let pthURL = directory.appendingPathComponent("model.pth.tar")
        let weights = try readPthTar(at: pthURL)

        // Convenience: required weight lookup.
        func w(_ key: String) throws -> [Float] {
            guard let v = weights[key] else {
                throw FireRedVADError.missingWeight(key)
            }
            return v
        }
        func wOpt(_ key: String) -> [Float]? { weights[key] }

        let H = config.hiddenSize
        let P = config.projSize
        let N1 = config.lookbackOrder
        let N2 = config.lookaheadOrder
        let S1 = config.lookbackStride
        let S2 = config.lookaheadStride
        let idim = config.idim

        // fc1: [H, idim] + bias [H].
        let fc1 = VADLinear(
            weight: try w("dfsmn.fc1.0.weight"),
            bias: wOpt("dfsmn.fc1.0.bias"),
            inFeatures: idim, outFeatures: H)
        // fc2: [P, H] + bias [P].
        let fc2 = VADLinear(
            weight: try w("dfsmn.fc2.0.weight"),
            bias: wOpt("dfsmn.fc2.0.bias"),
            inFeatures: H, outFeatures: P)

        // fsmn1: lookback [P, 1, N1] (squeeze dim1 → [P, N1]).
        let lb1 = squeezeGroupConvWeight(try w("dfsmn.fsmn1.lookback_filter.weight"), P: P, N: N1)
        let la1Raw = wOpt("dfsmn.fsmn1.lookahead_filter.weight") ?? []
        let la1 =
            la1Raw.isEmpty
            ? la1Raw
            : squeezeGroupConvWeight(la1Raw, P: P, N: N2)
        let fsmn1 = FireRedFSMN(
            lookbackWeight: lb1, lookaheadWeight: la1,
            P: P, N1: N1, S1: S1, N2: N2, S2: S2)

        // DFSMN blocks: config.numBlocks - 1 blocks (first is fsmn1).
        var blocks: [FireRedDFSMNBlock] = []
        for n in 0 ..< (config.numBlocks - 1) {
            let prefix = "dfsmn.fsmns.\(n)"
            let bfc1 = VADLinear(
                weight: try w("\(prefix).fc1.0.weight"),
                bias: wOpt("\(prefix).fc1.0.bias"),
                inFeatures: P, outFeatures: H)
            let bfc2 = VADLinear(
                weight: try w("\(prefix).fc2.weight"),
                bias: nil,
                inFeatures: H, outFeatures: P)
            let lbN = squeezeGroupConvWeight(
                try w("\(prefix).fsmn.lookback_filter.weight"), P: P, N: N1)
            let laNRaw = wOpt("\(prefix).fsmn.lookahead_filter.weight") ?? []
            let laN =
                laNRaw.isEmpty
                ? laNRaw
                : squeezeGroupConvWeight(laNRaw, P: P, N: N2)
            let bFsmn = FireRedFSMN(
                lookbackWeight: lbN, lookaheadWeight: laN,
                P: P, N1: N1, S1: S1, N2: N2, S2: S2)
            blocks.append(FireRedDFSMNBlock(fc1: bfc1, fc2: bfc2, fsmn: bFsmn))
        }

        // DNN layers: config.numDnnLayers layers. First is P → H.
        var dnnLayers: [VADLinear] = []
        let dnn0 = VADLinear(
            weight: try w("dfsmn.dnns.0.weight"),
            bias: wOpt("dfsmn.dnns.0.bias"),
            inFeatures: P, outFeatures: H)
        dnnLayers.append(dnn0)
        for l in 1 ..< config.numDnnLayers {
            let dnnN = VADLinear(
                weight: try w("dfsmn.dnns.\(l).weight"),
                bias: wOpt("dfsmn.dnns.\(l).bias"),
                inFeatures: H, outFeatures: H)
            dnnLayers.append(dnnN)
        }

        // Output head: [odim, H].
        let outLinear = VADLinear(
            weight: try w("out.weight"),
            bias: wOpt("out.bias"),
            inFeatures: H, outFeatures: config.odim)

        let dfsmn = FireRedDFSMN(
            fc1: fc1, fc2: fc2, fsmn1: fsmn1, blocks: blocks,
            dnns: dnnLayers, outLinear: outLinear,
            idim: idim, H: H, P: P)

        // CMVN: use baked-in defaults (derived from `cmvn.ark` in the repo).
        // A future version could parse `cmvn.ark` for custom checkpoints.
        let cmvn = FireRedCMVN.default

        return FireRedVADModel(config: config, dfsmn: dfsmn, cmvn: cmvn)
    }

    /// Squeeze a PyTorch depthwise-conv weight from `[P, 1, N]` flat
    /// storage to `[P, N]` by dropping the channel-1 dimension.
    private static func squeezeGroupConvWeight(_ w: [Float], P: Int, N: Int) -> [Float] {
        // The checkpoint stores the weight as a contiguous [P, 1, N] tensor
        // (depthwise `groups=P`, so in_channels_per_group=1). The flat layout
        // is already `[P, N]` when the middle dim is 1, so a copy suffices.
        precondition(
            w.count == P * N, "squeezeGroupConvWeight: count \(w.count) != P(\(P))*N(\(N))")
        return w
    }

    /// Download (or hit cache) the `FireRedTeam/FireRedVAD` checkpoint and
    /// load it.
    public static func fromPretrained(
        _ idOrPath: String,
        device: Device = .shared
    ) async throws -> FireRedVADModel {
        let dir = try await ModelLocator().resolve(idOrPath: idOrPath)
        return try loadFromDirectory(dir, device: device)
    }
}

// ─── Pickle value representation ─────────────────────────────────────

/// Minimal value types for our pickle scanner.
private enum PickleVal {
    case none_val
    case bool(Bool)
    case int(Int)
    case str(String)
    case tuple([PickleVal])
    case dict
    case list
    case global_fn
    case reduced
    case storage(id: String, count: Int)
}
