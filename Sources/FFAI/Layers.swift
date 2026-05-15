// Standard transformer layer building blocks: Linear, Embedding,
// RMSNorm. Each holds its weight tensors as fields and exposes
// `parameters()` for SafeTensors weight binding.

import Foundation
import Metal

// ─── Linear (no bias for now — Llama doesn't use biases) ─────────────

public final class Linear: Module {
    /// weight shape [out_features, in_features], row-major.
    public let weight: Tensor

    public init(weight: Tensor) {
        precondition(weight.shape.count == 2, "Linear: weight must be 2D")
        self.weight = weight
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.gemv(weight: weight, input: x, on: cmd)
    }
}

// ─── Embedding ───────────────────────────────────────────────────────

public final class Embedding: Module {
    /// weight shape [vocab_size, hidden_size]
    public let weight: Tensor

    public init(weight: Tensor) {
        precondition(weight.shape.count == 2, "Embedding: weight must be 2D")
        self.weight = weight
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight)]
    }

    /// Look up `tokenIds` (one-element u32 tensor for decode) and return
    /// [n_tokens, hidden] in the table's dtype.
    public func callAsFunction(_ tokenIds: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.gather(table: weight, tokenIds: tokenIds, on: cmd)
    }
}

// ─── RMSNorm ─────────────────────────────────────────────────────────

public final class RMSNorm: Module {
    /// weight shape [n] — per-channel scale.
    public let weight: Tensor
    public let eps: Float

    public init(weight: Tensor, eps: Float) {
        self.weight = weight
        self.eps = eps
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.rmsNorm(x, weight: weight, eps: eps, on: cmd)
    }
}
