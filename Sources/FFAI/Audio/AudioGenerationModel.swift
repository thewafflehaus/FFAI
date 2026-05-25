// AudioGenerationModel — protocol + shared types for TTS / audio-generation
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

    public var description: String {
        switch self {
        case .missingConfig(let m): return "AudioGeneration: missing config field: \(m)"
        case .codecNotAvailable(let m): return "AudioGeneration: codec not available — \(m)"
        case .generationFailed(let m): return "AudioGeneration: generation failed: \(m)"
        case .invalidInput(let m): return "AudioGeneration: invalid input: \(m)"
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

// ─── Generation parameters ───────────────────────────────────────────────

/// Hyper-parameters for audio generation. All fields have defaults so a
/// caller can override only what they care about per family.
public struct AudioGenerationParameters: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    /// Speed multiplier (1.0 = natural rate). Applied post-decode.
    public var speed: Float

    public init(
        maxTokens: Int = 1024,
        temperature: Float = 0.7,
        topP: Float = 0.7,
        topK: Int = 30,
        speed: Float = 1.0
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.speed = speed
    }
}
