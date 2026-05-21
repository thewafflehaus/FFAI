// VADOutput — shared result types for voice-activity-detection (VAD)
// model families.
//
// VAD models consume an audio waveform and emit a per-frame
// speech-probability stream plus optional post-processed speech
// segments. Unlike the causal-LM families (`LanguageModel`), VAD models
// are audio-in / probability-out, so they don't flow through
// `ModelRegistry` / `Model`. They live under `VADModelRegistry`
// instead and expose their own `loadFromDirectory` / `fromPretrained`
// entry points.

import Foundation

// ─── Speech segment ──────────────────────────────────────────────────

/// A contiguous run of detected speech, expressed in audio sample
/// indices relative to the start of the input clip.
public struct VADSpeechSegment: Sendable, Equatable {
    /// Inclusive start sample index.
    public let startSample: Int
    /// Exclusive end sample index.
    public let endSample: Int
    /// Sample rate the indices are expressed in (Hz).
    public let sampleRate: Int

    public init(startSample: Int, endSample: Int, sampleRate: Int) {
        self.startSample = startSample
        self.endSample = endSample
        self.sampleRate = sampleRate
    }

    /// Segment start in seconds.
    public var startSeconds: Double { Double(startSample) / Double(sampleRate) }
    /// Segment end in seconds.
    public var endSeconds: Double { Double(endSample) / Double(sampleRate) }
    /// Segment duration in seconds.
    public var durationSeconds: Double { endSeconds - startSeconds }
}

// ─── VAD output ──────────────────────────────────────────────────────

/// Result of running a VAD model over a clip.
///
/// `probabilities` is the raw per-frame speech-probability stream; each
/// value is the model's estimate that the corresponding analysis frame
/// contains speech. `segments` is the post-processed list of speech
/// runs derived from that stream (threshold + hysteresis + min-duration
/// smoothing — exact post-processing is family-specific).
public struct VADOutput: Sendable {
    /// Per-frame speech probabilities, in `[0, 1]`.
    public let probabilities: [Float]
    /// Number of audio samples each probability frame advances by. For
    /// SileroVAD this is the chunk size (512 @ 16 kHz). Useful for
    /// mapping a frame index back to an audio offset.
    public let frameStrideSamples: Int
    /// Sample rate the analysis was run at (Hz).
    public let sampleRate: Int
    /// Post-processed speech segments. Empty if no speech crossed the
    /// detection threshold.
    public let segments: [VADSpeechSegment]

    public init(
        probabilities: [Float],
        frameStrideSamples: Int,
        sampleRate: Int,
        segments: [VADSpeechSegment]
    ) {
        self.probabilities = probabilities
        self.frameStrideSamples = frameStrideSamples
        self.sampleRate = sampleRate
        self.segments = segments
    }

    /// True if the probability stream is finite and every value lies in
    /// `[0, 1]`. A failed forward pass (NaN / Inf / out-of-range)
    /// trips this — used by integration tests as a sanity gate.
    public var isWellFormed: Bool {
        probabilities.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 }
    }

    /// Total speech time across all detected segments, in seconds.
    public var totalSpeechSeconds: Double {
        segments.reduce(0) { $0 + $1.durationSeconds }
    }
}

// ─── Endpoint output (turn-detection style) ──────────────────────────

/// Result of a binary turn-/endpoint-detection model (e.g. SmartTurn).
/// These models emit a single utterance-level probability rather than a
/// per-frame stream.
public struct VADEndpointOutput: Sendable {
    /// Probability in `[0, 1]` that the utterance has ended (a complete
    /// turn / endpoint was reached).
    public let probability: Float
    /// `1` if `probability` crossed the model's threshold, else `0`.
    public let prediction: Int

    public init(probability: Float, prediction: Int) {
        self.probability = probability
        self.prediction = prediction
    }
}

// ─── Diarization output (multi-speaker) ──────────────────────────────

/// A speaker-attributed segment from a diarization model (e.g.
/// Sortformer).
public struct DiarizationSegment: Sendable, Equatable {
    /// Segment start in seconds.
    public let startSeconds: Double
    /// Segment end in seconds.
    public let endSeconds: Double
    /// Zero-based speaker index.
    public let speaker: Int

    public init(startSeconds: Double, endSeconds: Double, speaker: Int) {
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.speaker = speaker
    }

    /// Segment duration in seconds.
    public var durationSeconds: Double { endSeconds - startSeconds }
}

/// Result of running a diarization model over a clip.
public struct DiarizationOutput: Sendable {
    /// Per-frame, per-speaker activity probabilities. Outer index is the
    /// analysis frame, inner index is the speaker.
    public let speakerProbabilities: [[Float]]
    /// Number of audio samples each probability frame advances by.
    public let frameStrideSamples: Int
    /// Sample rate the analysis was run at (Hz).
    public let sampleRate: Int
    /// Post-processed speaker segments.
    public let segments: [DiarizationSegment]
    /// Number of distinct speakers the model is configured for.
    public let numSpeakers: Int

    public init(
        speakerProbabilities: [[Float]],
        frameStrideSamples: Int,
        sampleRate: Int,
        segments: [DiarizationSegment],
        numSpeakers: Int
    ) {
        self.speakerProbabilities = speakerProbabilities
        self.frameStrideSamples = frameStrideSamples
        self.sampleRate = sampleRate
        self.segments = segments
        self.numSpeakers = numSpeakers
    }

    /// True if every probability is finite and in `[0, 1]`.
    public var isWellFormed: Bool {
        speakerProbabilities.allSatisfy { row in
            row.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 }
        }
    }
}
