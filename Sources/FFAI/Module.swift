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
// Module protocol — minimal interface every model layer implements.
//
// Initial cut: just parameter discovery + weight loading. No autograd, no
// state-dict serialization. Adding more later as needs arise.

import Foundation

public protocol Module: AnyObject {
    /// Flat list of (name, tensor) pairs for everything this module owns.
    /// Names use HF SafeTensors convention (e.g. "self_attn.q_proj.weight").
    /// Container modules prefix their child names with their own.
    func parameters() -> [(String, Tensor)]
}

public extension Module {
    /// Pretty-print parameter names (debugging aid).
    func parameterSummary() -> String {
        parameters()
            .map { "\($0.0): \($0.1.shape) \($0.1.dtype)" }
            .joined(separator: "\n")
    }
}
