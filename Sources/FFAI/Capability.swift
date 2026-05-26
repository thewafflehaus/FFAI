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
// Capability — what a model can do. Multi-modal models declare which
// capabilities they support via their family file; users pick which to
// enable at load time. Disabled modalities skip weight allocation.
//
// textIn / textOut are universal for LLMs and always implicitly enabled.

import Foundation

public enum Capability: String, Sendable, Hashable, CaseIterable, Codable {
    case textIn
    case textOut
    /// Image input — image-only VL models (Gemma 3/4-VL, Nemotron-VL,
    /// Idefics3, GlmOcr, FastVLM, Mistral3, Paligemma, …). Video-capable
    /// families declare `.videoIn` in addition to `.imageIn`.
    case imageIn
    /// A model that can consume video — i.e. a temporally-ordered
    /// sequence of frames. Distinct from `imageIn` because not every
    /// vision-language model wires the multi-frame temporal-patch
    /// path: a `videoIn` model accepts `[Tensor]` (one frame each) via
    /// `VisionEncoder.encode(frames:device:)` and folds them into the
    /// vision-token stream in temporal-patch chunks, while an
    /// `imageIn`-only model takes a single image and treats the
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
    /// Model supports a user-tunable reasoning-effort dial (none / low /
    /// medium / high / extra-high / max), distinct from just having
    /// `.thinking`. The selected level is set on `GenerationParameters`,
    /// not here — this capability just advertises that the model
    /// honours it.
    case reasoningLevel
}

/// User-tunable reasoning effort for models that advertise the
/// `Capability.reasoningLevel` capability. Models without it ignore
/// the setting.
///
/// Values follow the Claude Opus convention (`none` → `max`) that
/// other reasoning-tuned model families are adopting. `.none` disables
/// the reasoning trace entirely; `.max` lets the model spend as long
/// as it needs to.
///
/// FFAI exposes the **full** enum as the common user-facing dial.
/// Models that natively only understand a subset (e.g. GPT-OSS-20B
/// supports `{.low, .medium, .high}`) declare their native set via
/// `ReasoningCapable.supportedReasoningLevels` and the resolver
/// `clamped(to:)` below maps the user's request to the nearest native
/// value. `.none` always maps to `.none` — explicit disable wins.
public enum ReasoningLevel: String, Sendable, Hashable, CaseIterable, Codable {
    case none
    case low
    case medium
    case high
    /// Raw value `"extra-high"` to match the hyphenated convention
    /// other model families use on the wire.
    case extraHigh = "extra-high"
    case max
}

public extension ReasoningLevel {
    /// Canonical "more reasoning" ordering. Used by `clamped(to:)` to
    /// pick the nearest native level by index distance.
    static let canonicalOrder: [ReasoningLevel] = [
        .none, .low, .medium, .high, .extraHigh, .max
    ]

    /// Map this user-requested level to the nearest value the model
    /// actually understands. `.none` is always honoured (returns
    /// `.none` regardless of `supported`) — it's an explicit
    /// "disable" signal. For any other value:
    ///
    ///   • If `supported` already contains it, return it unchanged.
    ///   • Otherwise pick the nearest member of `supported` by
    ///     canonical-order distance. Ties break toward the **lower**
    ///     (cheaper) level — better to under-reason than to silently
    ///     burn extra tokens.
    ///
    /// Example — GPT-OSS-20B has `supported = {.low, .medium, .high}`:
    ///   - `.none` → `.none`              (explicit disable)
    ///   - `.low / .medium / .high`        unchanged
    ///   - `.extraHigh / .max`            → `.high` (clamped)
    func clamped(to supported: Set<ReasoningLevel>) -> ReasoningLevel {
        if self == .none { return .none }
        if supported.contains(self) { return self }
        let order = ReasoningLevel.canonicalOrder
        guard let myIdx = order.firstIndex(of: self) else { return self }
        let candidates = order.enumerated().filter { supported.contains($0.element) }
        guard !candidates.isEmpty else { return self }
        let best = candidates.min { lhs, rhs in
            let dl = abs(lhs.offset - myIdx)
            let dr = abs(rhs.offset - myIdx)
            if dl != dr { return dl < dr }
            // Tie-break toward the lower (cheaper) level.
            return lhs.offset < rhs.offset
        }
        return best!.element
    }
}

/// Conformance marker for models that advertise
/// `Capability.reasoningLevel`. Declares which native levels the
/// model recognises so the runtime can clamp user requests.
///
/// Conformance is by family variant struct (e.g.
/// `GPTOSSMoEVariant`) — the same shape as
/// `defaultGenerationParameters`. Non-reasoning families simply
/// don't conform.
public protocol ReasoningCapable {
    /// Levels the model natively recognises (excluding `.none`,
    /// which every reasoning-capable model honours implicitly as
    /// "disable reasoning"). For GPT-OSS-20B this is
    /// `[.low, .medium, .high]`; new models add whatever they
    /// support.
    static var supportedReasoningLevels: Set<ReasoningLevel> { get }
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
