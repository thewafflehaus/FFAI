// AudioModel — protocol + shared types for TTS / audio-generation
// families. The runtime registry is `AudioModelRegistry` in
// `AudioModelRegistry.swift`; this file declares the model-facing protocol
// + generation-parameter struct + error enum that audio families implement.
//
// Audio models differ from `LanguageModel` in that their primary output is
// waveform samples (or quantized audio codes), not text tokens. They may
// internally use an LLM backbone for semantic token generation but expose
// a `synthesize(...)` entry point instead of `generate(...)`.

import Foundation
import Metal

// ─── Errors ──────────────────────────────────────────────────────────────

/// Errors raised by audio-generation (TTS / dual-AR) families.
/// Distinct from `AudioModelError` in `VADModelRegistry.swift` (which
/// covers VAD-registry routing failures).
public enum AudioGenerationError: Error, CustomStringConvertible {
    case missingConfig(String)
    case codecNotAvailable(String)
    case generationFailed(String)
    case invalidInput(String)
    /// Requested voice isn't one of the model's `availableVoices`.
    case voiceNotAvailable(requested: String, available: [String])

    public var description: String {
        switch self {
        case .missingConfig(let m): return "AudioGeneration: missing config field: \(m)"
        case .codecNotAvailable(let m): return "AudioGeneration: codec not available — \(m)"
        case .generationFailed(let m): return "AudioGeneration: generation failed: \(m)"
        case .invalidInput(let m): return "AudioGeneration: invalid input: \(m)"
        case .voiceNotAvailable(let requested, let available):
            return "AudioGeneration: voice '\(requested)' not available — "
                + "model offers \(available.joined(separator: ", "))"
        }
    }
}

// ─── Protocol ────────────────────────────────────────────────────────────

/// Common surface for TTS / audio-generation families. Callers interact
/// with this protocol so they don't need to know the concrete family type.
public protocol AudioModel: Module {
    /// Audio sample rate in Hz (e.g. 44100 for FishSpeech).
    var sampleRate: Int { get }

    /// Synthesise speech for `text`. Returns a flat `Float32` waveform in
    /// CPU-accessible memory.
    func synthesize(
        text: String,
        parameters: AudioGenerationParameters,
        device: Device
    ) throws -> [Float]
}

// ─── Voice selection ─────────────────────────────────────────────────────
//
// Every AudioModel exposes a voice surface: a list of available voices, a
// model-specific default, and a `setVoice(_:)` hot-swap. Models that don't
// have a meaningful voice concept (most STS denoisers / source-separators,
// some single-speaker TTS) inherit the protocol-extension defaults below —
// `availableVoices == ["default"]` and `setVoice("default")` no-ops.
// Multi-voice families (Kokoro, FishSpeech, …) override `availableVoices`
// to expose their voice catalogue and override `setVoice` to swap the
// active style vector.

public extension AudioModel {
    /// Names of voices this model can use. Default is a single
    /// `"default"` entry — multi-voice families (Kokoro, FishSpeech, …)
    /// override this to expose their voice catalogue by name.
    var availableVoices: [String] { ["default"] }

    /// The voice activated when the caller passes
    /// `AudioGenerationParameters.voice == "default"`. Override per
    /// family — e.g. Kokoro defaults to `"af_heart"`.
    var defaultVoice: String { "default" }

    /// Activate a named voice. The default implementation accepts only
    /// `"default"` (matching the single-voice protocol-extension surface);
    /// multi-voice families override it to hot-load and cache the
    /// requested style vector.
    func setVoice(_ name: String) throws {
        let valid = availableVoices
        guard valid.contains(name) || name == "default" else {
            throw AudioGenerationError.voiceNotAvailable(
                requested: name, available: valid)
        }
    }
}

// ─── Generation parameters ───────────────────────────────────────────────

/// Hyper-parameters for audio generation. All fields have defaults so a
/// caller can override only what they care about per family.
public struct AudioGenerationParameters: Sendable {
    /// Voice name. `"default"` (the default value) resolves to the
    /// model's `defaultVoice`. Pass any other string from
    /// `model.availableVoices` to override per-call. Models with only
    /// the `["default"]` voice catalogue ignore non-`"default"` values
    /// or throw `AudioGenerationError.voiceNotAvailable`.
    public var voice: String
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    /// Speed multiplier (1.0 = natural rate). Applied post-decode.
    public var speed: Float

    public init(
        voice: String = "default",
        maxTokens: Int = 1024,
        temperature: Float = 0.7,
        topP: Float = 0.7,
        topK: Int = 30,
        speed: Float = 1.0
    ) {
        self.voice = voice
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.speed = speed
    }
}
