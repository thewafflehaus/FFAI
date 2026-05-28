// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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

    /// The device whose residency set holds `state` — kept so `deinit`
    /// can release it.
    private let device: Device

    public init(
        nChannels: Int, kernelSize: Int, dtype: DType,
        device: Device = .shared
    ) {
        precondition(
            kernelSize >= 2,
            "ConvStateCache: kernelSize must be >= 2 (1-tap conv has no state)")
        self.device = device
        self.nChannels = nChannels
        self.kernelSize = kernelSize
        self.dtype = dtype
        self.state = Tensor.empty(
            shape: [kernelSize - 1, nChannels],
            dtype: dtype, device: device)
        self.state.zero()
        // Conv rolling window persists across every decode step; pin it
        // in the residency set so the driver skips per-dispatch
        // residency validation on the read+shift that fires each token.
        device.markWeightsResident([self.state.buffer])
    }

    deinit {
        // Release the conv state buffer from the residency set so the
        // wired memory is freed when the cache is dropped.
        device.unmarkWeightsResident([state.buffer])
    }

    /// Reset to zero. Used between sessions; cheap (small fp16/bf16 buffer).
    public func reset() { state.zero() }

    /// Bytes occupied by the rolling window. Constant w.r.t. sequence
    /// length — that's the streaming-decode design.
    public var bytesAllocated: Int {
        (kernelSize - 1) * nChannels * dtype.byteSize
    }

    /// Cached snapshot tensor. Reused across snapshot calls so
    /// spec-decode doesn't churn ~30 KB per layer per verify step.
    private var cachedSnapshot: Tensor?

    /// Copy `state` into a fresh (or cached) snapshot tensor. Used by
    /// spec-decode to roll back the conv rolling window on draft reject.
    ///
    /// Reuse contract: this method returns a single per-instance
    /// scratch tensor. Calling `snapshot()` a second time before
    /// `restore(from:)` will overwrite the prior snapshot. Nested or
    /// concurrent snapshot usage on the same cache is not supported.
    public func snapshot(device: Device = .shared) -> Tensor {
        let shape = [kernelSize - 1, nChannels]
        if cachedSnapshot == nil
            || cachedSnapshot!.shape != shape
            || cachedSnapshot!.dtype != dtype
        {
            cachedSnapshot = Tensor.empty(
                shape: shape, dtype: dtype, device: device)
        }
        let snap = cachedSnapshot!
        let cmd = device.makeCommandBuffer()
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure(
                "ConvStateCache.snapshot: makeBlitCommandEncoder failed")
        }
        let bytes = bytesAllocated
        blit.copy(
            from: state.buffer, sourceOffset: state.offset,
            to: snap.buffer, destinationOffset: snap.offset,
            size: bytes)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return snap
    }

    /// Restore `state` from a snapshot taken via `snapshot()`.
    public func restore(from snapshot: Tensor, device: Device = .shared) {
        precondition(
            snapshot.elementCount == state.elementCount,
            "ConvStateCache.restore: snapshot element count mismatch")
        precondition(
            snapshot.dtype == dtype,
            "ConvStateCache.restore: snapshot dtype mismatch")
        let cmd = device.makeCommandBuffer()
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure(
                "ConvStateCache.restore: makeBlitCommandEncoder failed")
        }
        let bytes = bytesAllocated
        blit.copy(
            from: snapshot.buffer, sourceOffset: snapshot.offset,
            to: state.buffer, destinationOffset: state.offset,
            size: bytes)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
}

extension Array where Element == ConvStateCache {
    /// Sum of `bytesAllocated` across all per-layer conv caches.
    public var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
}
