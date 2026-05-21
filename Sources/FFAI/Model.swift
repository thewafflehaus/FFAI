// Model — public entry point users interact with. Resolves a model
// id-or-path, downloads via HF if needed, decodes config, dispatches to
// the right family file, loads weights, and exposes a forward()/generate()
// surface.

import Foundation
import Tokenizers

public enum ModelError: Error, CustomStringConvertible {
    case unsupportedArchitecture(String)
    case unsupportedModelType(String)
    case capabilityNotAvailable(Capability)
    case visionModelNotIntegrated(String)

    public var description: String {
        switch self {
        case .unsupportedArchitecture(let a): return "Unsupported architecture: \(a)"
        case .unsupportedModelType(let m): return "Unsupported model_type: \(m)"
        case .capabilityNotAvailable(let c): return "Capability not available: \(c)"
        case .visionModelNotIntegrated(let a):
            return "Vision-language checkpoint '\(a)' detected. The FFAI "
                + "vision foundation (VisionEncoder, ImagePreprocessing, "
                + "VLModel cross-modal splice, conv2d/patch_embed/rope_2d "
                + "Ops) is in tree, but this VL family is not yet wired to "
                + "a checkpoint loader. Load the text-only checkpoint, or "
                + "compose a VLModel directly from VisionEncoder + the text "
                + "engine."
        }
    }
}

/// Architecture strings that identify a vision-language checkpoint. A
/// VL checkpoint carries a `vision_config` block and prefixes its text
/// weights under `language_model.*`; the registry recognizes these so a
/// VL load fails with an actionable `visionModelNotIntegrated` error
/// rather than a generic "unsupported architecture".
public enum VisionLanguageArchitectures {
    public static let architectures: Set<String> = [
        "Gemma3ForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        // Note: `Gemma4ForConditionalGeneration` is intentionally NOT
        // listed — it is shared by text-only Gemma 4 checkpoints. The
        // `vision_config`-presence check below distinguishes the VL
        // conversion, which the dispatch routes to `Gemma4VL.load`.
    ]

    /// True if `config` describes a VL checkpoint — by architecture
    /// string or by the presence of a `vision_config` block.
    public static func isVisionLanguage(_ config: ModelConfig) -> Bool {
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        return config.nested("vision_config") != nil
    }
}

/// Routes a config to the right family file. Family files declare which
/// architecture / model_type strings they handle. Add a new family by
/// extending `dispatchAndLoad` here.
public enum ModelRegistry {
    /// Engine + the variant-declared generation defaults. The defaults
    /// flow into the `Model` so callers can read them off without
    /// knowing the concrete family.
    public struct Loaded {
        public let engine: any LanguageModel
        public let defaultGenerationParameters: GenerationParameters
        /// Capabilities the loaded variant supports. Text-only families
        /// report `Capability.textOnly`; VL variants add `.visionIn`.
        public let availableCapabilities: Set<Capability>
        /// The composed vision-language model, when the checkpoint is a
        /// VLM. `nil` for text-only families. The `engine` is the VL
        /// model's text backbone, so text-only generation works
        /// regardless; `vlModel` adds the cross-modal image path.
        public let vlModel: VLModel?

        public init(engine: any LanguageModel,
                    defaultGenerationParameters: GenerationParameters,
                    availableCapabilities: Set<Capability> = Capability.textOnly,
                    vlModel: VLModel? = nil) {
            self.engine = engine
            self.defaultGenerationParameters = defaultGenerationParameters
            self.availableCapabilities = availableCapabilities
            self.vlModel = vlModel
        }
    }

    public static func dispatchAndLoad(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Loaded {
        // Vision-language checkpoints carry a nested `vision_config` and
        // prefix their text weights under `language_model.*`.
        if VisionLanguageArchitectures.isVisionLanguage(config) {
            // Gemma 3 VL — SigLIP tower + Gemma 3 text backbone. Fully
            // wired: the SigLIP architecture is exactly `VisionEncoder`.
            if config.architecture == "Gemma3ForConditionalGeneration" {
                let vlm = try Gemma3VL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Gemma3Dense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.visionIn]),
                    vlModel: vlm)
            }
            // Qwen 2.5-VL — dynamic-resolution windowed-attention ViT
            // tower + the Qwen 2.x text backbone (routed through the
            // Llama dense engine, which now supports embedding-input
            // forward for the VLM splice).
            if config.architecture == "Qwen2_5_VLForConditionalGeneration" {
                let vlm = try Qwen25VL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: LlamaDense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.visionIn]),
                    vlModel: vlm)
            }
            // Qwen 3-VL — dynamic-resolution full-attention ViT tower
            // (LayerNorm pre-norms, GELU MLP, learned position table) +
            // the Qwen 3 dense text backbone, joined by the splice.
            if config.architecture == "Qwen3VLForConditionalGeneration" {
                let vlm = try Qwen3VL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Qwen3Dense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.visionIn]),
                    vlModel: vlm)
            }
            // Qwen 3-VL-MoE — the Qwen3-VL vision tower + the Qwen 3.5
            // mixture-of-experts hybrid text backbone (Gated Delta Net ↔
            // attention, block-sparse MoE FFN), joined by the splice.
            if config.architecture == "Qwen3VLMoeForConditionalGeneration" {
                let vlm = try Qwen3VLMoe.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Qwen35Hybrid.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.visionIn]),
                    vlModel: vlm)
            }
            // Gemma 4 VL — the bespoke Gemma 4 ViT tower (RoPE attention,
            // q/k/v norms, attention-pooling head) + multi-modal embedder
            // + the Gemma 4 text backbone, joined by the splice.
            if config.architecture == "Gemma4ForConditionalGeneration" {
                let vlm = try Gemma4VL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Gemma4Dense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.visionIn]),
                    vlModel: vlm)
            }
            // Other VL families (Nemotron-VLM, …) — the FFAI vision
            // foundation (VisionEncoder, ImagePreprocessing, VLModel
            // splice, conv2d/patch_embed/rope_2d Ops) is in tree, but
            // these towers are not yet wired to a checkpoint loader.
            // Fail with an actionable error rather than a generic
            // "unsupported".
            throw ModelError.visionModelNotIntegrated(
                config.architecture ?? config.modelType ?? "<unknown>")
        }
        if let arch = config.architecture, Llama.architectures.contains(arch) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let mt = config.modelType, Llama.modelTypes.contains(mt) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        // Mistral and Llama share the same weight layout + forward shape.
        // The Mistral family enum routes through the Llama loader so
        // every Mistral 7B / Nemo / Small checkpoint Just Works without
        // a separate dense engine.
        if let arch = config.architecture, Mistral.architectures.contains(arch) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let mt = config.modelType, Mistral.modelTypes.contains(mt) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        // Llama-compatible families: SmolLM 1/2/3, OLMo 1/2, Granite,
        // Yi, InternLM 2, Starcoder 2. Same weight layout + forward
        // shape as Llama 3; optional QKV biases auto-detected by
        // loadLinear. Each gets a six-line registry entry in
        // Models/LlamaCompatibles.swift instead of its own family file
        // (until / unless it diverges from the Llama-3 shape).
        if let arch = config.architecture, LlamaCompatibles.architectures.contains(arch) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let mt = config.modelType, LlamaCompatibles.modelTypes.contains(mt) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let arch = config.architecture, Phi.architectures.contains(arch) {
            return try loadPhi(config: config, weights: weights,
                               options: options, device: device)
        }
        if let mt = config.modelType, Phi.modelTypes.contains(mt) {
            return try loadPhi(config: config, weights: weights,
                               options: options, device: device)
        }
        // Qwen 2 / 2.5 — Llama-shaped arch with QKV biases. The
        // bias-aware Linear in Layers.swift handles the layout
        // transparently; just route the dispatch.
        if let arch = config.architecture, Qwen2.architectures.contains(arch) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let mt = config.modelType, Qwen2.modelTypes.contains(mt) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let arch = config.architecture, Gemma3.architectures.contains(arch) {
            return try loadGemma3(config: config, weights: weights,
                                  options: options, device: device)
        }
        if let mt = config.modelType, Gemma3.modelTypes.contains(mt) {
            return try loadGemma3(config: config, weights: weights,
                                  options: options, device: device)
        }
        // Gemma 4 — dense / PLE (E2B, E4B) / MoE (26B-A4B). All three
        // ship under the `gemma4` model_type; the family file picks the
        // variant from config. Checked before Qwen3 so the `gemma4`
        // model_type isn't shadowed.
        if let arch = config.architecture, Gemma4.architectures.contains(arch) {
            return try loadGemma4(config: config, weights: weights,
                                  options: options, device: device)
        }
        if let mt = config.modelType, Gemma4.modelTypes.contains(mt) {
            return try loadGemma4(config: config, weights: weights,
                                  options: options, device: device)
        }
        if let arch = config.architecture, Qwen3.architectures.contains(arch) {
            return try loadQwen3(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let mt = config.modelType, Qwen3.modelTypes.contains(mt) {
            return try loadQwen3(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let arch = config.architecture, Mamba2.architectures.contains(arch) {
            return try loadMamba2(config: config, weights: weights,
                                  options: options, device: device)
        }
        if let mt = config.modelType, Mamba2.modelTypes.contains(mt) {
            return try loadMamba2(config: config, weights: weights,
                                  options: options, device: device)
        }
        // FalconH1 — the first Phase 5e hybrid (Mamba 2 + attention in
        // every layer). Routes through its own family file + engine.
        if let arch = config.architecture, FalconH1.architectures.contains(arch) {
            return try loadFalconH1(config: config, weights: weights,
                                    options: options, device: device)
        }
        if let mt = config.modelType, FalconH1.modelTypes.contains(mt) {
            return try loadFalconH1(config: config, weights: weights,
                                    options: options, device: device)
        }
        // NemotronH — a Phase 5e stack-interleaved hybrid (Mamba 2 /
        // attention / dense-MLP layers selected per-layer by a
        // hybrid_override_pattern). Routes through its own family file.
        if let arch = config.architecture, NemotronH.architectures.contains(arch) {
            return try loadNemotronH(config: config, weights: weights,
                                     options: options, device: device)
        }
        if let mt = config.modelType, NemotronH.modelTypes.contains(mt) {
            return try loadNemotronH(config: config, weights: weights,
                                     options: options, device: device)
        }
        // GraniteMoeHybrid — a Phase 5e stack-interleaved hybrid (Mamba 2
        // / attention layers selected by `layer_types`) with an MoE +
        // shared-expert feed-forward. Routes through its own family file.
        if let arch = config.architecture, GraniteMoeHybrid.architectures.contains(arch) {
            return try loadGraniteMoeHybrid(config: config, weights: weights,
                                            options: options, device: device)
        }
        if let mt = config.modelType, GraniteMoeHybrid.modelTypes.contains(mt) {
            return try loadGraniteMoeHybrid(config: config, weights: weights,
                                            options: options, device: device)
        }
        // Jamba — a Phase 5e stack-interleaved hybrid (Mamba 1 / attention
        // layers selected by `layers_block_type`) with a dense SwiGLU or
        // MoE feed-forward. Routes through its own family file.
        if let arch = config.architecture, Jamba.architectures.contains(arch) {
            return try loadJamba(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let mt = config.modelType, Jamba.modelTypes.contains(mt) {
            return try loadJamba(config: config, weights: weights,
                                 options: options, device: device)
        }

        // Qwen3.5 — a Phase 5e stack-interleaved hybrid (Gated Delta Net /
        // full-attention layers alternating every `full_attention_interval`)
        // with a dense SwiGLU or MoE feed-forward. Routes through its own
        // family file.
        if let arch = config.architecture, Qwen35.architectures.contains(arch) {
            return try loadQwen35(config: config, weights: weights,
                                  options: options, device: device)
        }
        if let mt = config.modelType, Qwen35.modelTypes.contains(mt) {
            return try loadQwen35(config: config, weights: weights,
                                  options: options, device: device)
        }

        // GPT-OSS — a mixture-of-experts transformer with an alternating
        // sliding/full attention schedule, learned per-head attention
        // sinks, and bias-corrected projections. Routes through its own
        // family file.
        if let arch = config.architecture, GPTOSS.architectures.contains(arch) {
            return try loadGPTOSS(config: config, weights: weights,
                                  options: options, device: device)
        }
        if let mt = config.modelType, GPTOSS.modelTypes.contains(mt) {
            return try loadGPTOSS(config: config, weights: weights,
                                  options: options, device: device)
        }

        // Nemotron-Labs-Diffusion — tri-mode (AR / diffusion /
        // self-speculation) dense transformer. Distinct from the
        // NemotronH stack-interleaved hybrid family above.
        if let arch = config.architecture, NemotronLabsDiffusion.architectures.contains(arch) {
            return try loadNemotronLabsDiffusion(config: config, weights: weights,
                                                 options: options, device: device)
        }
        if let mt = config.modelType, NemotronLabsDiffusion.modelTypes.contains(mt) {
            return try loadNemotronLabsDiffusion(config: config, weights: weights,
                                                 options: options, device: device)
        }
        throw ModelError.unsupportedArchitecture(
            config.architecture ?? config.modelType ?? "<unknown>"
        )
    }

    public static func loadLlama(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Llama.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadQwen3(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Qwen3.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadPhi(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Phi.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGemma3(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Gemma3.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGemma4(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Gemma4.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadMamba2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Mamba2.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadFalconH1(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try FalconH1.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadNemotronH(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try NemotronH.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGraniteMoeHybrid(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try GraniteMoeHybrid.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadJamba(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Jamba.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadQwen35(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Qwen35.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGPTOSS(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try GPTOSS.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadNemotronLabsDiffusion(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try NemotronLabsDiffusion.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(engine: engine,
                      defaultGenerationParameters: variant.defaultGenerationParameters)
    }
}

/// High-level loaded model with tokenizer attached. The public API users
/// touch.
public final class Model: @unchecked Sendable {
    /// The concrete model engine (LlamaModel, Qwen3Model, …). For a VLM
    /// this is the text backbone — text-only generation works
    /// regardless of whether `.visionIn` is enabled.
    public let engine: any LanguageModel
    public let tokenizer: any Tokenizer
    public let config: ModelConfig
    public let modelDirectory: URL
    public let availableCapabilities: Set<Capability>

    /// The composed vision-language model — `nil` unless the checkpoint
    /// is a VLM. Use `vlModel.generate(...)` for an image+text prompt;
    /// available only when `availableCapabilities` contains `.visionIn`.
    public let vlModel: VLModel?

    /// Currently-enabled capabilities. Mutated via `enable(_:)` /
    /// `disable(_:)`; guarded by `capabilityLock` for thread safety.
    private var _enabledCapabilities: Set<Capability>
    private let capabilityLock = NSLock()

    /// Snapshot of the enabled-capability set.
    public var enabledCapabilities: Set<Capability> {
        capabilityLock.lock(); defer { capabilityLock.unlock() }
        return _enabledCapabilities
    }
    /// Default generation parameters declared by the model's family
    /// variant. Use as-is, or call `.with { $0.maxTokens = ... }` to
    /// tweak a field without losing the family-tuned baseline.
    public let defaultGenerationParameters: GenerationParameters

    /// Convenience accessor for tests + tools that want the Llama-typed
    /// model. Returns nil if the loaded engine isn't Llama.
    public var llama: LlamaModel? { engine as? LlamaModel }

    /// Convenience accessor for the Qwen3 engine.
    public var qwen3: Qwen3Model? { engine as? Qwen3Model }

    /// Convenience accessor for the Mamba 2 engine.
    public var mamba2: Mamba2Model? { engine as? Mamba2Model }

    /// Convenience accessor for the FalconH1 hybrid engine.
    public var falconH1: FalconH1Model? { engine as? FalconH1Model }

    /// Convenience accessor for the NemotronH hybrid engine.
    public var nemotronH: NemotronHModel? { engine as? NemotronHModel }

    /// Convenience accessor for the GraniteMoeHybrid hybrid engine.
    public var graniteMoeHybrid: GraniteMoeHybridModel? {
        engine as? GraniteMoeHybridModel
    }

    /// Convenience accessor for the Jamba hybrid engine.
    public var jamba: JambaModel? { engine as? JambaModel }

    /// Convenience accessor for the Qwen3.5 hybrid engine.
    public var qwen35: Qwen35Model? { engine as? Qwen35Model }

    /// Convenience accessor for the GPT-OSS MoE engine.
    public var gptOSS: GPTOSSModel? { engine as? GPTOSSModel }

    /// Convenience accessor for the Nemotron-Labs-Diffusion engine.
    public var nemotronLabsDiffusion: NemotronLabsDiffusionModel? {
        engine as? NemotronLabsDiffusionModel
    }

    private let stateLock = NSLock()
    private var _currentState: ModelLifecycleState = .ready

    public var currentState: ModelLifecycleState {
        stateLock.lock(); defer { stateLock.unlock() }
        return _currentState
    }

    /// Maximum number of lifecycle events buffered when no consumer is
    /// reading from `events`. The default `AsyncStream` policy is
    /// `.unbounded`, which leaks events forever if nobody subscribes
    /// (the common case — most callers don't attach an `events` task).
    /// 64 is well above the typical event count per generation
    /// (~6: idle → loading → ready → generating → idle, plus a few
    /// capability flips) but small enough that the unconsumed-events
    /// retention is bounded.
    public static let eventsBufferCapacity = 64

    public let events: AsyncStream<ModelLifecycleEvent>
    private let eventsContinuation: AsyncStream<ModelLifecycleEvent>.Continuation

    init(engine: any LanguageModel, tokenizer: any Tokenizer, config: ModelConfig,
         modelDirectory: URL,
         availableCapabilities: Set<Capability>,
         enabledCapabilities: Set<Capability>,
         defaultGenerationParameters: GenerationParameters,
         vlModel: VLModel? = nil) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.config = config
        self.modelDirectory = modelDirectory
        self.availableCapabilities = availableCapabilities
        self.vlModel = vlModel
        // textIn / textOut are universal — always enabled. Other
        // requested capabilities are honored only if the model declares
        // them available.
        self._enabledCapabilities = enabledCapabilities
            .union(Capability.textOnly)
            .intersection(availableCapabilities.union(Capability.textOnly))
        self.defaultGenerationParameters = defaultGenerationParameters
        // Bounded buffer — when no consumer is reading, keep the most
        // recent `eventsBufferCapacity` events and drop older ones.
        // Avoids the unbounded-growth leak from the default policy.
        let (stream, cont) = AsyncStream<ModelLifecycleEvent>.makeStream(
            bufferingPolicy: .bufferingNewest(Self.eventsBufferCapacity)
        )
        self.events = stream
        self.eventsContinuation = cont
    }

    deinit {
        eventsContinuation.finish()
    }

    fileprivate func emit(_ event: ModelLifecycleEvent) {
        stateLock.lock()
        _currentState = event.state
        stateLock.unlock()
        eventsContinuation.yield(event)
    }

    // ─── Capability enable / disable ─────────────────────────────────

    /// Whether a capability is currently enabled.
    public func isEnabled(_ capability: Capability) -> Bool {
        enabledCapabilities.contains(capability)
    }

    /// Enable a capability at runtime — e.g. `enable(.visionIn)` lights
    /// up the vision path on a model loaded text-only. No-op if the
    /// capability isn't in `availableCapabilities` (a text-only model
    /// can't gain vision) or is already enabled. Emits a lifecycle
    /// event tagged with the capability so consumers can react.
    ///
    /// `textIn` / `textOut` are universal and always enabled — calling
    /// `enable` / `disable` on them is a harmless no-op.
    @discardableResult
    public func enable(_ capability: Capability) -> Bool {
        guard availableCapabilities.contains(capability)
            || Capability.textOnly.contains(capability) else { return false }
        capabilityLock.lock()
        let changed = !_enabledCapabilities.contains(capability)
        _enabledCapabilities.insert(capability)
        capabilityLock.unlock()
        if changed {
            eventsContinuation.yield(
                ModelLifecycleEvent(capability: capability, state: currentState))
        }
        return changed
    }

    /// Disable a capability at runtime. `textIn` / `textOut` are
    /// universal and cannot be disabled — those calls are a no-op.
    /// Emits a capability-tagged lifecycle event when the set changes.
    @discardableResult
    public func disable(_ capability: Capability) -> Bool {
        guard !Capability.textOnly.contains(capability) else { return false }
        capabilityLock.lock()
        let changed = _enabledCapabilities.contains(capability)
        _enabledCapabilities.remove(capability)
        capabilityLock.unlock()
        if changed {
            eventsContinuation.yield(
                ModelLifecycleEvent(capability: capability, state: currentState))
        }
        return changed
    }

    // ─── Top-level loader ────────────────────────────────────────────

    /// Resolve an id-or-path, download if needed, decode config, load
    /// weights, build the family-specific model, attach tokenizer.
    public static func load(
        _ idOrPath: String,
        options: LoadOptions = LoadOptions(),
        device: Device = .shared
    ) async throws -> Model {
        Debug.log(.load, "Model.load id-or-path=\(idOrPath)")
        let model = try await Profile.timeAsync("model_load") {
            try await Profile.signpostAsync("model_load") {
                let locator = ModelLocator(downloader: ModelDownloader(cacheDirectory: options.cacheDirectory))
                let dir = try await locator.resolve(idOrPath: idOrPath, revision: options.revision)
                Debug.log(.loader, "resolved snapshot dir: \(dir.path)")
                let config = try ModelConfig.load(from: dir)
                Debug.log(.load, "config: arch=\(config.architecture ?? "?") model_type=\(config.modelType ?? "?") hidden=\(config.hiddenSize ?? 0) layers=\(config.numLayers ?? 0)")
                let bundle = try SafeTensorsBundle(directory: dir, device: device)
                let loaded = try ModelRegistry.dispatchAndLoad(
                    config: config, weights: bundle, options: options, device: device
                )
                // Nemotron-Labs-Diffusion ships an optional
                // `linear_spec_lora` adapter that sharpens the
                // self-speculation diffusion drafter — attach it if the
                // checkpoint included the subfolder.
                if let nd = loaded.engine as? NemotronLabsDiffusionModel {
                    nd.attachLoRA(from: dir, device: device)
                }
                let tokenizer = try await TokenizerLoader().load(from: dir)
                return Model(
                    engine: loaded.engine, tokenizer: tokenizer, config: config,
                    modelDirectory: dir,
                    availableCapabilities: loaded.availableCapabilities,
                    enabledCapabilities: options.capabilities,
                    defaultGenerationParameters: loaded.defaultGenerationParameters,
                    vlModel: loaded.vlModel
                )
            }
        }

        // Prewarm just touches the embedding lookup once so the PSO is
        // compiled before the first user-visible decode. Captured as a
        // separate phase so `--profiling 1` shows it broken out from
        // model_load.
        if options.prewarm {
            await Profile.timeAsync("prewarm") {
                await Profile.signpostAsync("prewarm") {
                    await model.prewarm()
                }
            }
        }

        model.emit(ModelLifecycleEvent(state: .ready))
        return model
    }

    /// Compile PSOs for the kernels we'll need during decode by running
    /// one no-op forward step. Costs ~100ms-1s on first load.
    public func prewarm() async {
        let cache = engine.makeLayerCaches()
        _ = engine.forward(tokenId: 0, position: 0, caches: cache)
    }
}

// ─── Hot LoRA adapter management ─────────────────────────────────────

public extension Model {
    /// Whether a LoRA adapter is currently attached. Always `false` for
    /// families that don't support adapters (only Nemotron-Labs-
    /// Diffusion does today).
    var hasLoRA: Bool { nemotronLabsDiffusion?.hasLoRA ?? false }

    /// Hot-load a LoRA adapter at runtime. `directory` may be the model
    /// directory (the adapter is resolved under `linear_spec_lora/`) or
    /// a directory holding `adapter_model.safetensors` directly — so the
    /// same call swaps in the bundled adapter or an external one. Any
    /// currently-attached adapter is replaced. No-op on families that
    /// don't support adapters. Do not call during an active generate.
    func loadLoRA(from directory: URL, device: Device = .shared) {
        guard let nd = nemotronLabsDiffusion else { return }
        nd.detachLoRA()
        nd.attachLoRA(from: directory, device: device)
    }

    /// Hot-unload the current LoRA adapter. No-op when none is attached.
    func unloadLoRA() { nemotronLabsDiffusion?.detachLoRA() }
}
