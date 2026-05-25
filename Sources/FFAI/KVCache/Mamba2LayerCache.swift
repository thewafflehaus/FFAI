// Mamba2LayerCache — per-layer state bundle for a Mamba 2 mixer block.
// Wraps the two caches a Mamba 2 layer needs:
//   - `ssm`  : fp32 recurrent state h[nHeads, stateDim, headDim]
//              (selective-scan compresses arbitrary history into this)
//   - `conv` : rolling window of the last `(kernelSize - 1)` post-projection
//              inputs, shape [kernelSize - 1, nChannels], in layer dtype
//
// Conforms to `LayerCacheProtocol` (not `KVCacheProtocol`) — Mamba 2 has
// no K / V tensors and the engine never calls attention methods on these
// caches. The Mamba 2 model.forward casts each per-layer cache back to
// `Mamba2LayerCache` to reach the typed members.
//
// Length semantics: `length` increments by 1 per decode step purely for
// accounting; the underlying storage is constant-size. `maxSeq` is
// reported as `.max` so external code that gates on capacity treats SSM
// layers as unlimited.

import Foundation
import Metal

public final class Mamba2LayerCache: LayerCacheProtocol, @unchecked Sendable {
    public let ssm: SSMStateCache
    public let conv: ConvStateCache

    public private(set) var length: Int = 0
    public let maxSeq: Int = .max

    public init(nHeads: Int, stateDim: Int, headDim: Int,
                convChannels: Int, convKernelSize: Int,
                dtype: DType, device: Device = .shared) {
        self.ssm = SSMStateCache(nHeads: nHeads, stateDim: stateDim,
                                 headDim: headDim, device: device)
        self.conv = ConvStateCache(nChannels: convChannels,
                                   kernelSize: convKernelSize,
                                   dtype: dtype, device: device)
    }

    public func reset() {
        ssm.reset()
        conv.reset()
        length = 0
    }

    /// Called by the Mamba 2 layer after each decode step to advance the
    /// step counter. The actual state mutation happens inside the SSM /
    /// conv kernels via the wrapped caches.
    public func advance() { length += 1 }

    public var bytesAllocated: Int {
        ssm.bytesAllocated + conv.bytesAllocated
    }

    public var bytesInUse: Int {
        // SSM + conv storage is constant-size; "in use" equals allocated
        // once any step has executed, zero before then.
        length == 0 ? 0 : bytesAllocated
    }
}
