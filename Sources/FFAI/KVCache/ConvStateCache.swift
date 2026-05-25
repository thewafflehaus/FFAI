// ConvStateCache — per-layer rolling window for Mamba 2's 1D
// depthwise causal conv on the input projection (`d_conv` taps,
// typically 4). Holds the last `kernelSize - 1` inputs per channel
// so the conv can be evaluated as a single-token streaming step
// instead of recomputing over the full prefix every time.
//
// Storage shape: `[kernelSize - 1, nChannels]` in the layer's dtype.
// The buffer is shifted in-place by `Ops.conv1dCausalStep(...)`:
// after each call, state[0] is dropped and the current input x[d]
// becomes the new state[K-2][d].

import Foundation
import Metal

public final class ConvStateCache: @unchecked Sendable {
    public let nChannels: Int
    public let kernelSize: Int
    public let dtype: DType

    /// Rolling window, shape `[kernelSize - 1, nChannels]`.
    public let state: Tensor

    public init(nChannels: Int, kernelSize: Int, dtype: DType,
                device: Device = .shared) {
        precondition(kernelSize >= 2,
                     "ConvStateCache: kernelSize must be >= 2 (1-tap conv has no state)")
        self.nChannels = nChannels
        self.kernelSize = kernelSize
        self.dtype = dtype
        self.state = Tensor.empty(shape: [kernelSize - 1, nChannels],
                                  dtype: dtype, device: device)
        self.state.zero()
    }

    /// Reset to zero. Used between sessions; cheap (small fp16/bf16 buffer).
    public func reset() { state.zero() }

    /// Bytes occupied by the rolling window. Constant w.r.t. sequence
    /// length — that's the streaming-decode design.
    public var bytesAllocated: Int {
        (kernelSize - 1) * nChannels * dtype.byteSize
    }
}

public extension Array where Element == ConvStateCache {
    /// Sum of `bytesAllocated` across all per-layer conv caches.
    var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
}
