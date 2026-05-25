// GenerationParameters — knobs that control a single `generate(...)` call.
//
// Composition with per-family defaults: each model family declares its own
// default GenerationParameters via the family Variant protocol's `defaults`
// property. `Model.defaultGenerationParameters` returns that default; the user
// either uses it as-is, mutates fields on it, or constructs their own.
//
// The shipped greedy decode path consumes `maxTokens`, `stopOnEOS`, and
// (in planned) `prefillStepSize`. Sampling fields (`temperature`, `topP`,
// `topK`, `minP`, `repetitionPenalty`, `presencePenalty`) are wired through
// the API surface today but only take effect once GPU sampling kernels land
// — see planning/roadmap.md. Declaring them now keeps `GenerationParameters`
// a stable surface so per-family defaults don't churn when sampling lands.

import Foundation

public struct GenerationParameters: Sendable, Equatable {
    // ─── Length / stopping ───────────────────────────────────────────

    /// Hard cap on generated tokens. The decode loop also stops at EOS
    /// when `stopOnEOS` is true.
    public var maxTokens: Int

    /// Stop at the model's `eosTokenId` when set.
    public var stopOnEOS: Bool

    /// Override / extend the model's EOS set. Empty = use the model's
    /// declared EOS only.
    public var extraStopTokens: Set<Int>

    // ─── Prefill ─────────────────────────────────────────────────────

    /// Prefill chunk size (tokens) — how many prompt tokens flow
    /// through `engine.forwardMulti(...)` per dispatch. `nil` (the
    /// default) defers to the engine's `defaultPrefillStepSize` (1024
    /// generic, 2048 GPT-OSS, 4096 Gemma 4 / Qwen 3.5 MoE — the values
    /// `mlx-swift-lm` benched). Explicit non-nil value overrides the
    /// engine default. Used as the chunk size in
    /// `Generate.driveGeneration` (the chunked-prefill path).
    public var prefillStepSize: Int?

    // ─── Sampling (planned; no-op today on the greedy path) ──────────

    /// Softmax temperature. `0` = greedy argmax. Wired through but only
    /// honored once GPU sampling kernels ship; the current decode path
    /// is greedy regardless.
    public var temperature: Float

    /// Nucleus sampling cutoff. `1.0` = disabled.
    public var topP: Float

    /// Top-K sampling cutoff. `0` = disabled.
    public var topK: Int

    /// Min-P sampling cutoff (Qwen-style). `0` = disabled.
    public var minP: Float

    /// Repetition penalty (`<1` encourages, `>1` discourages). `1.0` = disabled.
    public var repetitionPenalty: Float

    /// Presence penalty (additive). `0` = disabled.
    public var presencePenalty: Float

    /// Optional sampling seed for reproducibility. `nil` = system-random.
    public var seed: UInt64?

    // ─── Reasoning ───────────────────────────────────────────────────

    /// User-requested reasoning effort. `nil` (the default) means
    /// "don't override the model" — the family's own
    /// `defaultGenerationParameters.reasoningLevel` then decides, and
    /// non-reasoning families ignore the field entirely.
    ///
    /// Reasoning-capable families (those declaring
    /// `Capability.reasoningLevel`) conform to `ReasoningCapable` and
    /// publish a `supportedReasoningLevels` set. The user-facing dial
    /// here always accepts the full `ReasoningLevel` enum
    /// (`none / low / medium / high / extraHigh / max`); each model
    /// clamps to what it natively understands via
    /// `ReasoningLevel.clamped(to:)`. `.none` always disables
    /// reasoning regardless of the model's catalogue.
    public var reasoningLevel: ReasoningLevel?

    public init(
        maxTokens: Int = 256,
        stopOnEOS: Bool = true,
        extraStopTokens: Set<Int> = [],
        prefillStepSize: Int? = nil,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        repetitionPenalty: Float = 1.0,
        presencePenalty: Float = 0.0,
        seed: UInt64? = nil,
        reasoningLevel: ReasoningLevel? = nil
    ) {
        self.maxTokens = maxTokens
        self.stopOnEOS = stopOnEOS
        self.extraStopTokens = extraStopTokens
        self.prefillStepSize = prefillStepSize
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.reasoningLevel = reasoningLevel
    }

    /// Returns a copy with `body` applied — convenient for "family default
    /// + tweak one field" call sites.
    public func with(_ body: (inout GenerationParameters) -> Void) -> GenerationParameters {
        var copy = self
        body(&copy)
        return copy
    }
}
