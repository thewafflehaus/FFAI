// Capability — what a model can do. Multi-modal models declare which
// capabilities they support via their family file; users pick which to
// enable at load time. Disabled modalities skip weight allocation.
//
// textIn / textOut are universal for LLMs and always implicitly enabled.

import Foundation

public enum Capability: String, Sendable, Hashable, CaseIterable, Codable {
    case textIn
    case textOut
    case visionIn
    /// A model that can consume video — i.e. a temporally-ordered
    /// sequence of frames. Distinct from `visionIn` because not every
    /// vision-language model wires the multi-frame temporal-patch
    /// path: a `videoIn` model accepts `[Tensor]` (one frame each) via
    /// `VisionEncoder.encode(frames:device:)` and folds them into the
    /// vision-token stream in temporal-patch chunks, while a
    /// `visionIn`-only model takes a single image and treats the
    /// temporal axis as a degenerate repeat.
    case videoIn
    case audioIn
    case audioOut
    case toolCalling
    /// Model supports chain-of-thought / "thinking" generation — emits
    /// a private reasoning trace before the final answer (Qwen 3 thinking,
    /// DeepSeek-R1, Claude extended thinking). The trace is typically
    /// fenced in `<think>…</think>` (or family-specific tokens) and
    /// stripped by `ThinkingSplit` before being shown to the user.
    case thinking
    /// Model supports a user-tunable reasoning-effort dial (minimal /
    /// low / medium / high), distinct from just having `.thinking`.
    /// The selected level is set on `GenerationParameters`, not here —
    /// this capability just advertises that the model honours it.
    case reasoningLevel
}

/// User-tunable reasoning effort for models that advertise the
/// `Capability.reasoningLevel` capability. Models without it ignore
/// the setting.
public enum ReasoningLevel: String, Sendable, Hashable, CaseIterable, Codable {
    case minimal
    case low
    case medium
    case high
}

extension Capability {
    public static let textOnly: Set<Capability> = [.textIn, .textOut]
    public static let textWithTools: Set<Capability> = [.textIn, .textOut, .toolCalling]

    /// Speech-to-text models (Whisper): audio in, text out.
    public static let speechToText: Set<Capability> = [.audioIn, .textOut]
    /// Text-to-speech models (Kokoro): text in, audio out.
    public static let textToSpeech: Set<Capability> = [.textIn, .audioOut]
    /// Omni-modal models (Qwen-Omni): text + audio in, text out.
    public static let omniAudio: Set<Capability> = [.textIn, .audioIn, .textOut]
    /// Speech enhancement / source separation / audio segmentation
    /// (DeepFilterNet, MossFormer2-SE, SAMAudio): audio in, audio out.
    public static let speechToSpeech: Set<Capability> = [.audioIn, .audioOut]
}
