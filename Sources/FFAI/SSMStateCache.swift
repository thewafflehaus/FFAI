// SSMStateCache — per-layer recurrent state for Mamba 2 (and other
// state-space-model layers). Unlike `KVCache`, the state has a fixed
// shape (`[nHeads, stateDim, headDim]`) and doesn't grow with the
// number of generated tokens — selective scan compresses an
// arbitrary-length history into that fixed state. Decode is therefore
// O(1) memory and O(stateDim * headDim) compute per token.
//
// Storage dtype is **fp32**: the state accumulates `exp(A*dt) * h +
// dt*B*x` over many decode steps; bf16's 7-bit mantissa drifts in a
// few dozen steps. mlx-swift-lm makes the same call.
//
// This class ships the storage + reset/step plumbing. The Mamba 2
// family file (Phase 5e+) calls `Ops.ssmStep(...)` between
// `appendOnGPU(...)`-equivalent invocations; the conv-state buffer
// that hybrid models need (Mamba 2 has a 1D depthwise conv on
// the input projection) is a follow-up.

import Foundation
import Metal

public final class SSMStateCache: @unchecked Sendable {
    public let nHeads: Int
    public let stateDim: Int
    public let headDim: Int

    /// Recurrent state h, shape `[nHeads, stateDim, headDim]`. Always fp32.
    public let h: Tensor

    public init(nHeads: Int, stateDim: Int, headDim: Int,
                device: Device = .shared) {
        self.nHeads = nHeads
        self.stateDim = stateDim
        self.headDim = headDim
        self.h = Tensor.empty(shape: [nHeads, stateDim, headDim],
                              dtype: .f32, device: device)
        self.h.zero()
    }

    /// Reset the recurrent state to zero. Cheap (zero-fill of a small
    /// fp32 buffer); cache size doesn't depend on sequence length.
    public func reset() { h.zero() }

    /// Bytes occupied by the recurrent state. Independent of how many
    /// tokens have been processed — that's the SSM compression win.
    public var bytesAllocated: Int {
        nHeads * stateDim * headDim * DType.f32.byteSize
    }
}

public extension Array where Element == SSMStateCache {
    /// Sum of `bytesAllocated` across all per-layer SSM caches.
    var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
}
