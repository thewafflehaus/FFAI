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

/// Phase 2 ModelRegistry — only Llama for now. Family files register
/// their architectures here at compile time.
public enum ModelRegistry {
    public static func loadLlama(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LlamaModel {
        let variant = try Llama.variant(for: config)
        return try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
    }

    public static func dispatchAndLoad(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LlamaModel {
        if let arch = config.architecture, Llama.architectures.contains(arch) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        if let mt = config.modelType, Llama.modelTypes.contains(mt) {
            return try loadLlama(config: config, weights: weights,
                                 options: options, device: device)
        }
        throw ModelError.unsupportedArchitecture(
            config.architecture ?? config.modelType ?? "<unknown>"
        )
    }
}

/// High-level loaded model with tokenizer attached. The public API users
/// touch.
public final class Model: @unchecked Sendable {
    public let llama: LlamaModel
    public let tokenizer: any Tokenizer
    public let config: ModelConfig
    public let modelDirectory: URL
    public let availableCapabilities: Set<Capability>
    public let enabledCapabilities: Set<Capability>

    private let stateLock = NSLock()
    private var _currentState: ModelLifecycleState = .ready

    public var currentState: ModelLifecycleState {
        stateLock.lock(); defer { stateLock.unlock() }
        return _currentState
    }

    public let events: AsyncStream<ModelLifecycleEvent>
    private let eventsContinuation: AsyncStream<ModelLifecycleEvent>.Continuation

    init(llama: LlamaModel, tokenizer: any Tokenizer, config: ModelConfig,
         modelDirectory: URL,
         availableCapabilities: Set<Capability>,
         enabledCapabilities: Set<Capability>) {
        self.llama = llama
        self.tokenizer = tokenizer
        self.config = config
        self.modelDirectory = modelDirectory
        self.availableCapabilities = availableCapabilities
        self.enabledCapabilities = enabledCapabilities
        var cont: AsyncStream<ModelLifecycleEvent>.Continuation!
        self.events = AsyncStream { c in cont = c }
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
        let locator = ModelLocator()
        let dir = try await locator.resolve(idOrPath: idOrPath, revision: options.revision)
        let config = try ModelConfig.load(from: dir)
        let bundle = try SafeTensorsBundle(directory: dir, device: device)
        let llama = try ModelRegistry.dispatchAndLoad(
            config: config, weights: bundle, options: options, device: device
        )
        let tokenizer = try await TokenizerLoader().load(from: dir)

        let model = Model(
            llama: llama, tokenizer: tokenizer, config: config,
            modelDirectory: dir,
            availableCapabilities: LlamaDense.availableCapabilities,
            enabledCapabilities: options.capabilities
        )

        // Phase 2: prewarm just touches the embedding lookup once so the
        // PSO is compiled before the first user-visible decode.
        if options.prewarm {
            await model.prewarm()
        }

        model.emit(ModelLifecycleEvent(state: .ready))
        return model
    }

    /// Compile PSOs for the kernels we'll need during decode by running
    /// one no-op forward step. Costs ~100ms-1s on first load.
    public func prewarm() async {
        let cache = llama.makeKVCache()
        // Run one decode step on token 0 to warm every PSO.
        _ = llama.forward(tokenId: 0, position: 0, caches: cache)
    }
}
