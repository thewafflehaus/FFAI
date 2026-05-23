// TenVAD — TEN-framework lightweight voice-activity-detection family.
//
// TEN-VAD is a small neural VAD from the TEN Framework (Agora).
// Unlike SileroVAD (which ships as PyTorch weights that FFAI re-runs as
// a CPU forward), TEN-VAD distributes as a pre-compiled C library plus
// an embedded ONNX model. The macOS build ships as a universal
// `ten_vad.framework` inside the `TEN-framework/ten-vad` HuggingFace
// repo (lib/macOS/ten_vad.framework/).
//
// Architecture (from aed_st.h in the upstream source):
//   40-channel log-mel filterbank (LFBE) features @ 16 kHz,
//   hop_size = 256 samples (16 ms) by default.
//   Input features: 41 dims per frame (40 mel + 1 energy).
//   Neural backbone: SeparableConv2d → SeparableConv1d × 2 → 2× LSTM
//     (hidden=64) → Dense → Sigmoid.
//   The full forward is stateful — LSTM states carry between hops.
//
// C API (ten_vad.h):
//   ten_vad_create(handle, hop_size, threshold) → 0 / -1
//   ten_vad_process(handle, int16 audio_data, length,
//                   out_probability, out_flag) → 0 / -1
//   ten_vad_destroy(handle)
//
// Weight storage: embedded inside the compiled framework binary; no
// safetensors are involved. Loading happens via `dlopen` against the
// framework binary checked out from the HuggingFace snapshot.
//
// HuggingFace repo: `TEN-framework/ten-vad`
// No mlx-community conversion exists as of 2026-05-22 — a conversion
// would require exporting the ONNX weights to safetensors (the ONNX
// file is 315 kB and sits at src/onnx_model/ten-vad.onnx in the repo).
// If a safetensors-based conversion appears at mlx-community/TEN-VAD,
// replace the dlopen approach with a pure-Swift LSTM forward matching
// the SileroVAD pattern.

import Foundation

// ─── Errors ──────────────────────────────────────────────────────────

public enum TenVADError: Error, CustomStringConvertible {
    case unsupportedSampleRate(Int)
    case frameworkNotFound(URL)
    case frameworkSymbolMissing(String)
    case createFailed
    case processFailed

    public var description: String {
        switch self {
        case .unsupportedSampleRate(let s):
            return "TenVAD: supports 16000 Hz audio (got \(s))"
        case .frameworkNotFound(let url):
            return "TenVAD: ten_vad framework binary not found at \(url.path)"
        case .frameworkSymbolMissing(let sym):
            return "TenVAD: required symbol '\(sym)' missing from framework"
        case .createFailed:
            return "TenVAD: ten_vad_create() returned -1"
        case .processFailed:
            return "TenVAD: ten_vad_process() returned -1"
        }
    }
}

// ─── Config ──────────────────────────────────────────────────────────

/// TEN-VAD configuration. All fields have published defaults matching
/// the upstream TEN-VAD reference implementation.
public struct TenVADConfig: Sendable {
    /// Samples per analysis hop at 16 kHz (16 ms = 256 samples).
    public let hopSize: Int
    /// Speech-detection threshold in `[0, 1]`.
    public let threshold: Float
    /// Minimum speech segment duration (ms) for post-processing.
    public let minSpeechDurationMs: Int
    /// Minimum silence gap (ms) for post-processing.
    public let minSilenceDurationMs: Int
    /// Padding added to each detected segment (ms).
    public let speechPadMs: Int

    public init(hopSize: Int = 256,
                threshold: Float = 0.5,
                minSpeechDurationMs: Int = 250,
                minSilenceDurationMs: Int = 100,
                speechPadMs: Int = 30) {
        self.hopSize = hopSize
        self.threshold = threshold
        self.minSpeechDurationMs = minSpeechDurationMs
        self.minSilenceDurationMs = minSilenceDurationMs
        self.speechPadMs = speechPadMs
    }

    /// Decode from a HuggingFace `config.json` dictionary. Every field
    /// has a published default so a missing / sparse config is fine.
    public static func decode(from raw: [String: Any]) -> TenVADConfig {
        func i(_ k: String, _ fb: Int) -> Int { (raw[k] as? Int) ?? fb }
        func f(_ k: String, _ fb: Float) -> Float {
            (raw[k] as? NSNumber)?.floatValue ?? fb
        }
        return TenVADConfig(
            hopSize: i("hop_size", 256),
            threshold: f("threshold", 0.5),
            minSpeechDurationMs: i("min_speech_duration_ms", 250),
            minSilenceDurationMs: i("min_silence_duration_ms", 100),
            speechPadMs: i("speech_pad_ms", 30))
    }
}

// ─── C function pointer types ─────────────────────────────────────────

// The TEN-VAD C API exposed by ten_vad.framework. All symbols are
// resolved at runtime via dlopen / dlsym so the package has no link-time
// dependency on the framework binary.

private typealias TenVadCreateFn = @convention(c) (
    UnsafeMutablePointer<UnsafeMutableRawPointer?>,
    Int,
    Float
) -> Int32

private typealias TenVadProcessFn = @convention(c) (
    UnsafeMutableRawPointer?,
    UnsafePointer<Int16>,
    Int,
    UnsafeMutablePointer<Float>,
    UnsafeMutablePointer<Int32>
) -> Int32

private typealias TenVadDestroyFn = @convention(c) (
    UnsafeMutablePointer<UnsafeMutableRawPointer?>
) -> Int32

// ─── Native wrapper ───────────────────────────────────────────────────

/// Manages one TEN-VAD C instance (create / process / destroy lifecycle).
/// All ops are single-threaded; callers must serialise access.
private final class TenVADNative {
    private var handle: UnsafeMutableRawPointer?
    private let createFn: TenVadCreateFn
    private let processFn: TenVadProcessFn
    private let destroyFn: TenVadDestroyFn

    init(libraryURL: URL, hopSize: Int, threshold: Float) throws {
        guard let lib = dlopen(libraryURL.path, RTLD_LAZY) else {
            throw TenVADError.frameworkNotFound(libraryURL)
        }

        func sym<T>(_ name: String) throws -> T {
            guard let ptr = dlsym(lib, name) else {
                throw TenVADError.frameworkSymbolMissing(name)
            }
            return unsafeBitCast(ptr, to: T.self)
        }
        self.createFn  = try sym("ten_vad_create")
        self.processFn = try sym("ten_vad_process")
        self.destroyFn = try sym("ten_vad_destroy")

        var h: UnsafeMutableRawPointer? = nil
        let ret = createFn(&h, hopSize, threshold)
        guard ret == 0, let created = h else {
            throw TenVADError.createFailed
        }
        self.handle = created
    }

    deinit {
        if handle != nil {
            _ = destroyFn(&handle)
        }
    }

    /// Process one hop of `hopSize` int16 samples. Returns (probability, flag).
    func process(_ pcm: [Int16]) throws -> (prob: Float, flag: Bool) {
        precondition(!pcm.isEmpty, "TenVADNative: empty PCM buffer")
        var prob: Float = 0
        var flag: Int32 = 0
        let ret = pcm.withUnsafeBufferPointer { buf in
            processFn(handle, buf.baseAddress!, pcm.count, &prob, &flag)
        }
        guard ret == 0 else { throw TenVADError.processFailed }
        return (prob, flag != 0)
    }
}

// ─── TenVADModel ─────────────────────────────────────────────────────

/// Loaded TEN-VAD model. Audio-in / speech-probability-out — reached via
/// `VADModelRegistry`, not `ModelRegistry`. `fromPretrained` downloads
/// the `TEN-framework/ten-vad` HuggingFace snapshot and links the
/// macOS framework binary that ships inside it.
public final class TenVADModel: @unchecked Sendable {
    public let config: TenVADConfig
    /// URL of the loaded ten_vad framework binary.
    public let frameworkBinaryURL: URL
    /// Snapshot directory from which this model was loaded.
    let directory: URL

    init(config: TenVADConfig, frameworkBinaryURL: URL, directory: URL) {
        self.config = config
        self.frameworkBinaryURL = frameworkBinaryURL
        self.directory = directory
    }

    // ─── Forward over a full clip ────────────────────────────────────

    /// Run VAD over a mono `audio` clip and return the per-frame speech
    /// probability stream plus post-processed speech segments.
    ///
    /// TEN-VAD processes audio as 16-bit PCM (int16 in `[-32768, 32767]`).
    /// Float samples in `[-1, 1]` are scaled accordingly before each hop.
    ///
    /// - Parameters:
    ///   - audio: Mono PCM samples in `[-1, 1]`.
    ///   - sampleRate: Must be 16000 Hz.
    public func detect(audio: [Float], sampleRate: Int = 16000) throws -> VADOutput {
        guard sampleRate == 16000 else {
            throw TenVADError.unsupportedSampleRate(sampleRate)
        }
        if audio.isEmpty {
            return VADOutput(probabilities: [], frameStrideSamples: config.hopSize,
                             sampleRate: sampleRate, segments: [])
        }

        // Create a fresh native instance for this detection pass. Each
        // call gets its own stateful LSTM context so concurrent calls on
        // the same model are safe.
        let native = try TenVADNative(
            libraryURL: frameworkBinaryURL,
            hopSize: config.hopSize,
            threshold: config.threshold)

        let hs = config.hopSize
        // Right-pad to a whole number of hops.
        var padded = audio
        let rem = audio.count % hs
        if rem != 0 { padded.append(contentsOf: [Float](repeating: 0, count: hs - rem)) }

        var probs: [Float] = []
        probs.reserveCapacity(padded.count / hs)

        // Process hop by hop, converting float → int16 each time.
        var pcm16 = [Int16](repeating: 0, count: hs)
        for hopIdx in 0..<(padded.count / hs) {
            let base = hopIdx * hs
            // Scale [-1, 1] → int16. Values outside [-1, 1] are clamped.
            for i in 0..<hs {
                let s = padded[base + i]
                let clamped = max(-1, min(1, s))
                pcm16[i] = Int16(clamped * 32767)
            }
            let (prob, _) = try native.process(pcm16)
            probs.append(prob)
        }

        let segments = Self.probsToSegments(
            probs, audioLen: audio.count, sampleRate: sampleRate,
            hopSize: hs, threshold: config.threshold,
            minSpeechDurationMs: config.minSpeechDurationMs,
            minSilenceDurationMs: config.minSilenceDurationMs,
            speechPadMs: config.speechPadMs)

        return VADOutput(probabilities: probs, frameStrideSamples: hs,
                         sampleRate: sampleRate, segments: segments)
    }

    // ─── Probability stream → segments ───────────────────────────────

    /// Convert the per-hop probability stream into speech segments using
    /// threshold + hysteresis + min-duration smoothing. Mirrors
    /// `SileroVADModel.probsToSegments`.
    static func probsToSegments(
        _ probs: [Float], audioLen: Int, sampleRate: Int,
        hopSize: Int, threshold: Float,
        minSpeechDurationMs: Int, minSilenceDurationMs: Int, speechPadMs: Int
    ) -> [VADSpeechSegment] {
        let minSpeechSamples = Float(sampleRate) * Float(minSpeechDurationMs) / 1000
        let minSilenceSamples = Float(sampleRate) * Float(minSilenceDurationMs) / 1000
        let speechPadSamples = Int(Float(sampleRate) * Float(speechPadMs) / 1000)
        // Hysteresis: a lower threshold ends speech than starts it.
        let negThreshold = max(threshold - 0.15, 0.01)

        struct Run { var start: Int; var end: Int }
        var speeches: [Run] = []
        var triggered = false
        var currentStart = 0
        var tempEnd = 0

        for (idx, p) in probs.enumerated() {
            let hopStart = idx * hopSize
            if p >= threshold && !triggered {
                triggered = true
                currentStart = hopStart
                tempEnd = 0
                continue
            }
            if triggered && p >= threshold {
                tempEnd = 0
                continue
            }
            if triggered && p < negThreshold {
                if tempEnd == 0 { tempEnd = hopStart }
                if Float(hopStart - tempEnd) >= minSilenceSamples {
                    if Float(tempEnd - currentStart) >= minSpeechSamples {
                        speeches.append(Run(start: currentStart, end: tempEnd))
                    }
                    triggered = false
                    tempEnd = 0
                }
            }
        }
        if triggered {
            let end = min(audioLen, probs.count * hopSize)
            if Float(end - currentStart) >= minSpeechSamples {
                speeches.append(Run(start: currentStart, end: end))
            }
        }

        // Pad each segment by speechPadSamples, merging overlaps.
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
            VADSpeechSegment(startSample: $0.start, endSample: $0.end,
                             sampleRate: sampleRate)
        }
    }

    // ─── Loading ─────────────────────────────────────────────────────

    /// Resolve the ten_vad macOS framework binary inside a snapshot
    /// directory. The HF repo ships it at:
    /// `lib/macOS/ten_vad.framework/Versions/A/ten_vad`
    ///
    /// Returns nil if the binary is absent (integration test disables
    /// itself in that case rather than throwing a misleading error).
    public static func frameworkBinary(in directory: URL) -> URL? {
        // Canonical path inside the TEN-framework/ten-vad snapshot.
        let canonical = directory
            .appendingPathComponent("lib/macOS/ten_vad.framework")
            .appendingPathComponent("Versions/A/ten_vad")
        if FileManager.default.fileExists(atPath: canonical.path) {
            return canonical
        }
        // Flat layout fallback (in case the framework is unpacked).
        let flat = directory.appendingPathComponent("ten_vad")
        if FileManager.default.fileExists(atPath: flat.path) {
            return flat
        }
        return nil
    }

    /// Load a TEN-VAD model from a local snapshot directory.
    ///
    /// Throws `TenVADError.frameworkNotFound` if the ten_vad framework
    /// binary is not present under the snapshot's `lib/macOS/` path.
    public static func loadFromDirectory(_ directory: URL,
                                          device _: Device = .shared) throws -> TenVADModel {
        // config.json is optional; published defaults are sufficient.
        var config = TenVADConfig()
        let configURL = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
           let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            config = TenVADConfig.decode(from: raw)
        }

        guard let binaryURL = frameworkBinary(in: directory) else {
            throw TenVADError.frameworkNotFound(
                directory.appendingPathComponent("lib/macOS/ten_vad.framework/Versions/A/ten_vad"))
        }
        return TenVADModel(config: config, frameworkBinaryURL: binaryURL,
                           directory: directory)
    }

    /// Download (or hit cache) the `TEN-framework/ten-vad` HuggingFace
    /// snapshot and load the model from the local directory.
    public static func fromPretrained(_ idOrPath: String,
                                       device: Device = .shared) async throws -> TenVADModel {
        let dir = try await ModelLocator().resolve(idOrPath: idOrPath)
        return try loadFromDirectory(dir, device: device)
    }
}
