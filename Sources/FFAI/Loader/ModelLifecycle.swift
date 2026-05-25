// Model lifecycle states + per-event stream value. Observable via
// Model.events: AsyncStream<ModelLifecycleEvent>.

import Foundation

public struct LoadProgress: Sendable {
    public let stage: String           // "config", "weights", "modules", "prewarm", ...
    public let completed: Int64        // bytes or tensors loaded
    public let total: Int64            // total bytes or tensors

    public var fraction: Double {
        total > 0 ? Double(completed) / Double(total) : 0
    }

    public init(stage: String, completed: Int64, total: Int64) {
        self.stage = stage
        self.completed = completed
        self.total = total
    }
}

/// Wrapped error type so ModelLifecycleState can stay Sendable.
public struct ModelLifecycleError: Error, Sendable, CustomStringConvertible {
    public let message: String
    public init(_ underlying: Error) {
        self.message = String(describing: underlying)
    }
    public init(message: String) {
        self.message = message
    }
    public var description: String { message }
}

public enum ModelLifecycleState: Sendable {
    case idle
    case downloading(Progress)
    case loading(LoadProgress)
    case loaded
    case ready
    case failed(ModelLifecycleError)
}

public struct ModelLifecycleEvent: Sendable {
    /// Capability whose loading state changed; nil = whole-model event.
    public let capability: Capability?
    public let state: ModelLifecycleState

    public init(capability: Capability? = nil, state: ModelLifecycleState) {
        self.capability = capability
        self.state = state
    }
}
