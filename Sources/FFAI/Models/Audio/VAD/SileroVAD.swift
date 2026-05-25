// SileroVAD — the standard streaming voice-activity-detection model.
//
// SileroVAD ships two near-identical branches (16 kHz and 8 kHz); each
// branch is a small CNN+LSTM stack that turns a fixed-size audio chunk
// into a single speech probability. The full forward over a clip slides
// the chunk window across the audio and emits one probability per
// chunk:
//
//   reflect-pad → stft_conv (learned STFT) → magnitude
//     → conv1 (ReLU) → conv2 (ReLU, /2) → conv3 (ReLU, /2) → conv4 (ReLU)
//     → LSTM(128) → ReLU → final_conv (1x1) → sigmoid → mean
//
// Chunk size is 512 samples @ 16 kHz (256 @ 8 kHz); a 64-sample context
// (32 @ 8 kHz) carries over between chunks. The model is ~1.5M params,
// so the forward runs entirely on the CPU via `VADCompute` — see that
// file's header for why a CPU path is the right call here.
//
// Checkpoint layout (`mlx-community/silero-vad`): weights are prefixed
// `vad_16k.` / `vad_8k.`; we remap those to `branch16k.` / `branch8k.`
// and drop any `val_*` validation tensors, matching the reference
// `sanitize` in mlx-audio-swift.

import Foundation

// ─── Errors ──────────────────────────────────────────────────────────

public enum SileroVADError: Error, CustomStringConvertible {
    case unsupportedSampleRate(Int)
    case missingWeight(String)
    case missingConfig(String)

    public var description: String {
        switch self {
        case .unsupportedSampleRate(let s):
            return "SileroVAD: supports 8000 Hz and 16000 Hz audio (got \(s))"
        case .missingWeight(let w):
            return "SileroVAD: required weight missing: \(w)"
        case .missingConfig(let f):
            return "SileroVAD: required config field missing: \(f)"
        }
    }
}

// ─── Branch config ───────────────────────────────────────────────────

/// Per-sample-rate geometry for one SileroVAD branch.
public struct SileroVADBranchConfig: Sendable {
    public let sampleRate: Int
    public let filterLength: Int
    public let hopLength: Int
    public let pad: Int
    public let cutoff: Int
    public let contextSize: Int
    public let chunkSize: Int

    public init(sampleRate: Int, filterLength: Int, hopLength: Int,
                pad: Int, cutoff: Int, contextSize: Int, chunkSize: Int) {
        self.sampleRate = sampleRate
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.pad = pad
        self.cutoff = cutoff
        self.contextSize = contextSize
        self.chunkSize = chunkSize
    }

    /// Published 16 kHz defaults.
    public static let default16k = SileroVADBranchConfig(
        sampleRate: 16000, filterLength: 256, hopLength: 128,
        pad: 64, cutoff: 129, contextSize: 64, chunkSize: 512)

    /// Published 8 kHz defaults.
    public static let default8k = SileroVADBranchConfig(
        sampleRate: 8000, filterLength: 128, hopLength: 64,
        pad: 32, cutoff: 65, contextSize: 32, chunkSize: 256)
}

// ─── Top-level config ────────────────────────────────────────────────

/// SileroVAD model + post-processing configuration.
public struct SileroVADConfig: Sendable {
    public let threshold: Float
    public let minSpeechDurationMs: Int
    public let minSilenceDurationMs: Int
    public let speechPadMs: Int
    public let branch16k: SileroVADBranchConfig
    public let branch8k: SileroVADBranchConfig

    public init(threshold: Float = 0.5,
                minSpeechDurationMs: Int = 250,
                minSilenceDurationMs: Int = 100,
                speechPadMs: Int = 30,
                branch16k: SileroVADBranchConfig = .default16k,
                branch8k: SileroVADBranchConfig = .default8k) {
        self.threshold = threshold
        self.minSpeechDurationMs = minSpeechDurationMs
        self.minSilenceDurationMs = minSilenceDurationMs
        self.speechPadMs = speechPadMs
        self.branch16k = branch16k
        self.branch8k = branch8k
    }

    /// Decode from a HuggingFace `config.json` dictionary. Every field
    /// has a published default, so a missing / sparse config still
    /// yields a usable model.
    public static func decode(from raw: [String: Any]) -> SileroVADConfig {
        func branch(_ key: String, _ fallback: SileroVADBranchConfig) -> SileroVADBranchConfig {
            guard let b = raw[key] as? [String: Any] else { return fallback }
            return SileroVADBranchConfig(
                sampleRate: (b["sample_rate"] as? Int) ?? fallback.sampleRate,
                filterLength: (b["filter_length"] as? Int) ?? fallback.filterLength,
                hopLength: (b["hop_length"] as? Int) ?? fallback.hopLength,
                pad: (b["pad"] as? Int) ?? fallback.pad,
                cutoff: (b["cutoff"] as? Int) ?? fallback.cutoff,
                contextSize: (b["context_size"] as? Int) ?? fallback.contextSize,
                chunkSize: (b["chunk_size"] as? Int) ?? fallback.chunkSize)
        }
        let thr = (raw["threshold"] as? NSNumber)?.floatValue ?? 0.5
        return SileroVADConfig(
            threshold: thr,
            minSpeechDurationMs: (raw["min_speech_duration_ms"] as? Int) ?? 250,
            minSilenceDurationMs: (raw["min_silence_duration_ms"] as? Int) ?? 100,
            speechPadMs: (raw["speech_pad_ms"] as? Int) ?? 30,
            branch16k: branch("branch_16k", .default16k),
            branch8k: branch("branch_8k", .default8k))
    }
}

// ─── Branch (one sample-rate stack) ──────────────────────────────────

/// One SileroVAD CNN+LSTM branch. Holds the host-resident weights and
/// runs the per-chunk forward.
final class SileroVADBranch: Sendable {
    let config: SileroVADBranchConfig
    let stftConv: VADConv1d
    let conv1, conv2, conv3, conv4: VADConv1d
    let lstm: VADLSTM
    let finalConv: VADConv1d

    /// - Parameters:
    ///   - config: branch geometry.
    ///   - prefix: weight-name prefix (`branch16k` / `branch8k`).
    ///   - lookup: resolves a branch-relative weight name to a Tensor,
    ///     or returns nil if absent.
    init(config: SileroVADBranchConfig, prefix: String,
         lookup: (String) -> Tensor?) throws {
        self.config = config

        func tensor(_ name: String) throws -> Tensor {
            let key = "\(prefix).\(name)"
            guard let tensor = lookup(key) else {
                throw SileroVADError.missingWeight(key)
            }
            return tensor
        }
        func floats(_ name: String) throws -> [Float] {
            try tensor(name).toFloatArray()
        }
        func floatsOpt(_ name: String) -> [Float]? {
            lookup("\(prefix).\(name)")?.toFloatArray()
        }

        // Conv weight layout fix. The `mlx-community/silero-vad` checkpoint
        // was exported from an MLX model, so every conv weight ships in
        // MLX's `[outC, K, inC]` layout. `VADConv1d` (a PyTorch-style CPU
        // conv) expects `[outC, inC, K]`. For `stft_conv` (inC=1) and
        // `final_conv` (K=1) the two layouts are bit-identical, but the
        // feature convs conv1..conv4 have both inC>1 and K>1 — loading
        // them raw silently scrambles the kernel and collapses the speech
        // probability to ~0. Transpose `[outC, K, inC] → [outC, inC, K]`
        // here, keyed off the stored tensor shape so a future PyTorch-
        // layout checkpoint still loads correctly.
        func convWeight(_ name: String, outC: Int, inC: Int, k: Int) throws -> [Float] {
            let t = try tensor(name)
            let raw = t.toFloatArray()
            precondition(raw.count == outC * inC * k,
                         "SileroVAD: \(prefix).\(name) count \(raw.count) != \(outC)*\(inC)*\(k)")
            // Stored MLX layout [outC, K, inC] → transpose to [outC, inC, K].
            if t.shape.count == 3, t.shape[1] == k, t.shape[2] == inC, inC != k {
                var out = [Float](repeating: 0, count: raw.count)
                for o in 0..<outC {
                    for kk in 0..<k {
                        for ic in 0..<inC {
                            // src [o, kk, ic] → dst [o, ic, kk]
                            out[(o * inC + ic) * k + kk] = raw[(o * k + kk) * inC + ic]
                        }
                    }
                }
                return out
            }
            return raw
        }

        // stft_conv: learned STFT — Conv1d(1 → cutoff*2, K=filterLength,
        // stride=hopLength), no bias. inC=1 so layout is unambiguous.
        self.stftConv = VADConv1d(
            weight: try floats("stft_conv.weight"), bias: nil,
            inChannels: 1, outChannels: config.cutoff * 2,
            kernelSize: config.filterLength, stride: config.hopLength)

        // conv1..conv4: feature CNN over the magnitude spectrogram.
        self.conv1 = VADConv1d(
            weight: try convWeight("conv1.weight", outC: 128, inC: config.cutoff, k: 3),
            bias: floatsOpt("conv1.bias"),
            inChannels: config.cutoff, outChannels: 128, kernelSize: 3, padding: 1)
        self.conv2 = VADConv1d(
            weight: try convWeight("conv2.weight", outC: 64, inC: 128, k: 3),
            bias: floatsOpt("conv2.bias"),
            inChannels: 128, outChannels: 64, kernelSize: 3, stride: 2, padding: 1)
        self.conv3 = VADConv1d(
            weight: try convWeight("conv3.weight", outC: 64, inC: 64, k: 3),
            bias: floatsOpt("conv3.bias"),
            inChannels: 64, outChannels: 64, kernelSize: 3, stride: 2, padding: 1)
        self.conv4 = VADConv1d(
            weight: try convWeight("conv4.weight", outC: 128, inC: 64, k: 3),
            bias: floatsOpt("conv4.bias"),
            inChannels: 64, outChannels: 128, kernelSize: 3, padding: 1)

        // LSTM(128 → 128). PyTorch packs weights as `weight_ih_l0` /
        // `weight_hh_l0`; mlx-audio-swift's LSTM uses `Wx` / `Wh`.
        let lstmIH = floatsOpt("lstm.Wx") ?? floatsOpt("lstm.weight_ih_l0")
        let lstmHH = floatsOpt("lstm.Wh") ?? floatsOpt("lstm.weight_hh_l0")
        guard let lstmIH, let lstmHH else {
            throw SileroVADError.missingWeight("\(prefix).lstm.{Wx,Wh}")
        }
        // Bias: PyTorch splits into `bias_ih` + `bias_hh`; the MLX export
        // ships a single fused `lstm.bias` of length `4*hidden`. The two
        // are summed at runtime, so a single fused bias goes into `biasIH`
        // with `biasHH` left nil — it is then applied exactly once. The
        // earlier loader only looked for `bias_ih*` keys, so the fused
        // `lstm.bias` was dropped and the LSTM ran bias-free.
        let lstmBiasIH = floatsOpt("lstm.bias_ih") ?? floatsOpt("lstm.bias_ih_l0")
            ?? floatsOpt("lstm.bias")
        let lstmBiasHH = floatsOpt("lstm.bias_hh") ?? floatsOpt("lstm.bias_hh_l0")
        self.lstm = VADLSTM(
            weightIH: lstmIH, weightHH: lstmHH,
            biasIH: lstmBiasIH, biasHH: lstmBiasHH,
            inputSize: 128, hiddenSize: 128)

        // final_conv: 1x1 conv to a single channel.
        self.finalConv = VADConv1d(
            weight: try floats("final_conv.weight"), bias: floatsOpt("final_conv.bias"),
            inChannels: 128, outChannels: 1, kernelSize: 1)
    }

    /// Forward one context+chunk window → (speech probability, final
    /// LSTM hidden, final LSTM cell). `window` is `[contextSize +
    /// chunkSize]` mono samples.
    func forward(window: [Float],
                 lstmHidden: [Float]?, lstmCell: [Float]?)
        -> (prob: Float, hidden: [Float], cell: [Float])
    {
        // reflect-pad on the right by `pad` samples.
        let padded = reflectPadRight(window, pad: config.pad)

        // stft_conv → split real / imag → magnitude.
        let (stft, stftLen) = stftConv.apply(padded, inLength: padded.count)
        // stft layout: [cutoff*2, stftLen]. First `cutoff` channels are
        // real, next `cutoff` are imaginary.
        var mag = [Float](repeating: 0, count: config.cutoff * stftLen)
        for c in 0..<config.cutoff {
            let reBase = c * stftLen
            let imBase = (config.cutoff + c) * stftLen
            for t in 0..<stftLen {
                let re = stft[reBase + t]
                let im = stft[imBase + t]
                mag[reBase + t] = (re * re + im * im).squareRoot()
            }
        }

        // conv1..conv4 with ReLU.
        var (h, hLen) = conv1.apply(mag, inLength: stftLen)
        VADMath.reluInPlace(&h)
        (h, hLen) = conv2.apply(h, inLength: hLen)
        VADMath.reluInPlace(&h)
        (h, hLen) = conv3.apply(h, inLength: hLen)
        VADMath.reluInPlace(&h)
        (h, hLen) = conv4.apply(h, inLength: hLen)
        VADMath.reluInPlace(&h)

        // LSTM expects [seqLen, features]; conv output is [features=128,
        // seqLen=hLen] channel-major. Transpose to time-major.
        var seq = [Float](repeating: 0, count: hLen * 128)
        for f in 0..<128 {
            for t in 0..<hLen {
                seq[t * 128 + f] = h[f * hLen + t]
            }
        }
        let (hiddenSeq, finalH, finalC) = lstm.run(
            seq, seqLen: hLen, initialHidden: lstmHidden, initialCell: lstmCell)

        // ReLU(hiddenSeq) → back to channel-major for final_conv.
        var post = hiddenSeq
        VADMath.reluInPlace(&post)
        var chMajor = [Float](repeating: 0, count: 128 * hLen)
        for t in 0..<hLen {
            for f in 0..<128 {
                chMajor[f * hLen + t] = post[t * 128 + f]
            }
        }
        let (fc, fcLen) = finalConv.apply(chMajor, inLength: hLen)

        // sigmoid → mean over time = single speech probability.
        var sum: Float = 0
        for t in 0..<fcLen { sum += VADMath.sigmoid(fc[t]) }
        let prob = fcLen > 0 ? sum / Float(fcLen) : 0
        return (prob, finalH, finalC)
    }
}

// ─── Right-side reflect padding ──────────────────────────────────────

/// Reflect-pad a 1-D signal on the right by `pad` samples, mirroring
/// `numpy`/`torch` reflect mode (`x[n-2], x[n-3], …`).
func reflectPadRight(_ x: [Float], pad: Int) -> [Float] {
    guard pad > 0 else { return x }
    let n = x.count
    precondition(n > pad, "reflect pad of \(pad) needs more than \(pad) samples (got \(n))")
    var out = x
    out.reserveCapacity(n + pad)
    for i in 0..<pad {
        out.append(x[n - 2 - i])
    }
    return out
}

// ─── SileroVAD model ─────────────────────────────────────────────────

/// Loaded SileroVAD model. Audio-in / speech-probability-out — this is
/// a VAD family, so it does not conform to `LanguageModel` and is
/// reached via `VADModelRegistry`, not `ModelRegistry`.
public final class SileroVADModel: @unchecked Sendable {
    public let config: SileroVADConfig
    let branch16k: SileroVADBranch
    let branch8k: SileroVADBranch

    init(config: SileroVADConfig, branch16k: SileroVADBranch, branch8k: SileroVADBranch) {
        self.config = config
        self.branch16k = branch16k
        self.branch8k = branch8k
    }

    private func branch(forSampleRate sr: Int) throws -> SileroVADBranch {
        switch sr {
        case 16000: return branch16k
        case 8000: return branch8k
        default: throw SileroVADError.unsupportedSampleRate(sr)
        }
    }

    // ─── Forward over a full clip ────────────────────────────────────

    /// Run VAD over a mono `audio` clip and return the per-chunk speech
    /// probability stream plus post-processed speech segments.
    ///
    /// - Parameters:
    ///   - audio: mono PCM samples in `[-1, 1]`.
    ///   - sampleRate: 8000 or 16000 Hz.
    public func detect(audio: [Float], sampleRate: Int = 16000) throws -> VADOutput {
        let b = try branch(forSampleRate: sampleRate)
        let cs = b.config.chunkSize
        let ctx = b.config.contextSize

        if audio.isEmpty {
            return VADOutput(probabilities: [], frameStrideSamples: cs,
                             sampleRate: sampleRate, segments: [])
        }

        // Right-pad audio to a whole number of chunks, then prepend a
        // zero context window.
        var a = audio
        let rem = a.count % cs
        if rem != 0 { a.append(contentsOf: [Float](repeating: 0, count: cs - rem)) }
        let preCtx = [Float](repeating: 0, count: ctx)
        a = preCtx + a

        var probs: [Float] = []
        var lstmHidden: [Float]? = nil
        var lstmCell: [Float]? = nil
        var pos = ctx
        while pos < a.count {
            let window = Array(a[(pos - ctx)..<(pos + cs)])
            let (p, h, c) = b.forward(window: window, lstmHidden: lstmHidden, lstmCell: lstmCell)
            probs.append(p)
            lstmHidden = h
            lstmCell = c
            pos += cs
        }

        let segments = Self.probsToSegments(
            probs, audioLen: audio.count, sampleRate: sampleRate,
            chunkSize: cs, threshold: config.threshold,
            minSpeechDurationMs: config.minSpeechDurationMs,
            minSilenceDurationMs: config.minSilenceDurationMs,
            speechPadMs: config.speechPadMs)

        return VADOutput(probabilities: probs, frameStrideSamples: cs,
                         sampleRate: sampleRate, segments: segments)
    }

    // ─── Probability stream → segments ───────────────────────────────

    /// Convert a per-chunk probability stream into speech segments using
    /// threshold + hysteresis + min-duration smoothing. Mirrors the
    /// reference `probsToTimestamps` in mlx-audio-swift.
    static func probsToSegments(
        _ probs: [Float], audioLen: Int, sampleRate: Int,
        chunkSize: Int, threshold: Float,
        minSpeechDurationMs: Int, minSilenceDurationMs: Int, speechPadMs: Int
    ) -> [VADSpeechSegment] {
        let minSpeechSamples = Float(sampleRate) * Float(minSpeechDurationMs) / 1000
        let minSilenceSamples = Float(sampleRate) * Float(minSilenceDurationMs) / 1000
        let speechPadSamples = Int(Float(sampleRate) * Float(speechPadMs) / 1000)
        // Hysteresis: a lower threshold ends a speech run than starts it.
        let negThreshold = max(threshold - 0.15, 0.01)

        struct Run { var start: Int; var end: Int }
        var speeches: [Run] = []
        var triggered = false
        var currentStart = 0
        var tempEnd = 0

        for (idx, p) in probs.enumerated() {
            let chunkStart = idx * chunkSize
            if p >= threshold && !triggered {
                triggered = true
                currentStart = chunkStart
                tempEnd = 0
                continue
            }
            if triggered && p >= threshold {
                tempEnd = 0
                continue
            }
            if triggered && p < negThreshold {
                if tempEnd == 0 { tempEnd = chunkStart }
                if Float(chunkStart - tempEnd) >= minSilenceSamples {
                    if Float(tempEnd - currentStart) >= minSpeechSamples {
                        speeches.append(Run(start: currentStart, end: tempEnd))
                    }
                    triggered = false
                    tempEnd = 0
                }
            }
        }
        if triggered {
            let end = min(audioLen, probs.count * chunkSize)
            if Float(end - currentStart) >= minSpeechSamples {
                speeches.append(Run(start: currentStart, end: end))
            }
        }

        // Pad each run by `speechPadSamples`, merging overlaps.
        var padded: [Run] = []
        for s in speeches {
            let start = max(0, s.start - speechPadSamples)
            let end = min(audioLen, s.end + speechPadSamples)
            if !padded.isEmpty, start <= padded[padded.count - 1].end {
                padded[padded.count - 1].end = max(padded[padded.count - 1].end, end)
            } else {
                padded.append(Run(start: start, end: end))
            }
        }
        return padded.map {
            VADSpeechSegment(startSample: $0.start, endSample: $0.end, sampleRate: sampleRate)
        }
    }

    // ─── Loading ─────────────────────────────────────────────────────

    /// Remap a raw checkpoint key to the branch-prefixed name this
    /// loader expects. Drops `val_*` validation tensors.
    static func remap(_ key: String) -> String? {
        if key.hasPrefix("val_") { return nil }
        if key.hasPrefix("vad_16k.") {
            return "branch16k." + String(key.dropFirst("vad_16k.".count))
        }
        if key.hasPrefix("vad_8k.") {
            return "branch8k." + String(key.dropFirst("vad_8k.".count))
        }
        return key
    }

    /// Load a SileroVAD checkpoint from a local snapshot directory.
    public static func loadFromDirectory(_ directory: URL,
                                         device: Device = .shared) throws -> SileroVADModel {
        // config.json is optional — every field has a published default.
        var config = SileroVADConfig()
        let configURL = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
           let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            config = SileroVADConfig.decode(from: raw)
        }

        // The bundle exposes raw checkpoint keys (`vad_16k.*` etc.);
        // SileroVADBranch expects branch-prefixed names. Build a
        // remapped name → tensor table once, then hand the branches a
        // lookup closure over it.
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        var table: [String: Tensor] = [:]
        for key in bundle.allKeys {
            guard let mapped = remap(key) else { continue }
            table[mapped] = try bundle.tensor(named: key)
        }
        let lookup: (String) -> Tensor? = { table[$0] }

        let branch16k = try SileroVADBranch(
            config: config.branch16k, prefix: "branch16k", lookup: lookup)
        let branch8k = try SileroVADBranch(
            config: config.branch8k, prefix: "branch8k", lookup: lookup)
        return SileroVADModel(config: config, branch16k: branch16k, branch8k: branch8k)
    }

    /// Download (or hit cache) a SileroVAD checkpoint from HuggingFace
    /// and load it.
    public static func fromPretrained(_ idOrPath: String,
                                      device: Device = .shared) async throws -> SileroVADModel {
        let locator = ModelLocator()
        let dir = try await locator.resolve(idOrPath: idOrPath)
        return try loadFromDirectory(dir, device: device)
    }
}
