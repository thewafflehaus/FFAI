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
