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
