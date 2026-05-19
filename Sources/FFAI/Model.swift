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

    public var description: String {
        switch self {
        case .unsupportedArchitecture(let a): return "Unsupported architecture: \(a)"
        case .unsupportedModelType(let m): return "Unsupported model_type: \(m)"
        case .capabilityNotAvailable(let c): return "Capability not available: \(c)"
        }
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
    }

    public static func dispatchAndLoad(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Loaded {
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
}

/// High-level loaded model with tokenizer attached. The public API users
/// touch.
public final class Model: @unchecked Sendable {
    /// The concrete model engine (LlamaModel, Qwen3Model, …).
    public let engine: any LanguageModel
    public let tokenizer: any Tokenizer
    public let config: ModelConfig
    public let modelDirectory: URL
    public let availableCapabilities: Set<Capability>
    public let enabledCapabilities: Set<Capability>
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
         defaultGenerationParameters: GenerationParameters) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.config = config
        self.modelDirectory = modelDirectory
        self.availableCapabilities = availableCapabilities
        self.enabledCapabilities = enabledCapabilities
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
                let tokenizer = try await TokenizerLoader().load(from: dir)
                return Model(
                    engine: loaded.engine, tokenizer: tokenizer, config: config,
                    modelDirectory: dir,
                    availableCapabilities: Capability.textOnly,
                    enabledCapabilities: options.capabilities,
                    defaultGenerationParameters: loaded.defaultGenerationParameters
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
