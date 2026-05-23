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
        // Conv state lives for the lifetime of a generation. Pin it in
        // the device residency set — conv1dCausalStep touches it every
        // token, every layer.
        device.markWeightsResident([self.state.buffer])
    }

    /// Reset to zero. Used between sessions; cheap (small fp16/bf16 buffer).
    public func reset() { state.zero() }

    /// Snapshot the current rolling window to a fresh `Tensor`. Used by
    /// speculative decode — the conv1d_causal_step kernel mutates this
    /// window in-place each token (drops state[0], appends current
    /// input as state[K-2]), and the mutation isn't trivially
    /// reversible without remembering the exact dropped slot and old
    /// state contents. Snapshot/restore is the cleanest rollback.
    ///
    /// Cost: `(kernelSize - 1) * nChannels` × dtype-size bytes per
    /// layer. At Qwen3.6-A3B convDim=5120, kernelSize=4, bf16:
    /// 3 * 5120 * 2 = 30 720 bytes per layer × 30 GDN layers ≈ 900 KB
    /// — small enough to snapshot every spec step without measurable
    /// host overhead.
    public func snapshot(device: Device = .shared) -> Tensor {
        let snap = Tensor.empty(shape: [kernelSize - 1, nChannels],
                                 dtype: dtype, device: device)
        let cmd = device.makeCommandBuffer()
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure("ConvStateCache.snapshot: makeBlitCommandEncoder failed")
        }
        let bytes = (kernelSize - 1) * nChannels * dtype.byteSize
        blit.copy(from: state.buffer, sourceOffset: state.offset,
                  to: snap.buffer, destinationOffset: snap.offset,
                  size: bytes)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return snap
    }

    /// Restore the rolling window from a snapshot. Overwrites the
    /// state buffer with the snapshot contents.
    public func restore(from snapshot: Tensor, device: Device = .shared) {
        let expected = (kernelSize - 1) * nChannels
        precondition(snapshot.elementCount == expected,
                     "ConvStateCache.restore: snapshot has \(snapshot.elementCount) elements, expected \(expected)")
        precondition(snapshot.dtype == dtype,
                     "ConvStateCache.restore: snapshot dtype \(snapshot.dtype) ≠ state dtype \(dtype)")
        let cmd = device.makeCommandBuffer()
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure("ConvStateCache.restore: makeBlitCommandEncoder failed")
        }
        let bytes = expected * dtype.byteSize
        blit.copy(from: snapshot.buffer, sourceOffset: snapshot.offset,
                  to: state.buffer, destinationOffset: state.offset,
                  size: bytes)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

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
