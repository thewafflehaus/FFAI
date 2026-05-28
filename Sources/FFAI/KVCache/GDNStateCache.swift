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
// GDNStateCache — per-layer recurrent state for a Gated Delta Net
// (GDN) mixer block. Like `SSMStateCache`, the state has a fixed shape
// and does NOT grow with the number of generated tokens: the GDN
// recurrence compresses arbitrary history into a per-head matrix
// `S[Hv, Dv, Dk]`. Decode is O(1) memory and O(Dv * Dk) compute per
// token.
//
// Storage dtype is **fp32**: the recurrence
//   S_t = g_t · S_{t-1} + β_t · k_t · (v_t − k_tᵀ · S_{t-1})ᵀ
// accumulates a multiplicative gate + a rank-1 update over many decode
// steps; bf16's 7-bit mantissa drifts fast. The metaltile kernel runs
// `S` in fp32 regardless of the activation dtype, so the cache matches.
//
// Double-buffered. The `mt_gated_delta_step` kernel reads `state_in`
// and writes a distinct `state_out` (it is NOT an in-place update like
// `ssm_step`). The cache holds both buffers and `swap()` ping-pongs
// them after each decode step so the freshly-written state becomes the
// input for the next step.
//
// State layout is `[Hv, Dv, Dk]` fp32 — matches the kernel's
// `state_base = n·Dv·Dk + dv_idx·Dk + s_idx` indexing (n = batch·Hv +
// hv; decode is single-batch so n = hv).
//
// Forward-only. This is scope: there are deliberately NO
// `record()` / `rollback()` hooks. Partial-accept replay (rewinding
// GDN state when speculative tokens are rejected) is deferred to
// — adding replay support now would mean snapshotting the
// per-head matrix on every step with no consumer for it.
//
// Conforms to `LayerCacheProtocol` (not `KVCacheProtocol`) — GDN has
// no K / V tensors and the engine never calls attention methods on
// these caches. A hybrid GDN model's `forward` casts each per-layer
// cache back to `GDNStateCache` to reach the typed members.

import Foundation
import Metal

public final class GDNStateCache: LayerCacheProtocol, @unchecked Sendable {
    /// Number of value heads (`Hv`). The per-head state slab index is
    /// `n = batch·Hv + hv`; decode is single-batch so `n = hv`.
    public let numValueHeads: Int
    /// Value-head dimension (`Dv`) — the row count of the state matrix.
    public let valueHeadDim: Int
    /// Key-head dimension (`Dk`) — the column count of the state matrix.
    public let keyHeadDim: Int

    /// The two recurrent-state buffers, each shape `[Hv, Dv, Dk]`, fp32.
    /// `current` is the live state (kernel input); `next` receives the
    /// kernel's output. `swap()` exchanges them after each step.
    ///
    /// Both the legacy `Ops.gatedDeltaStep` path and the fused
    /// `Ops.gatedDeltaPrepStep` path share these slots: the fused
    /// kernel runs in fp32 (against bf16 model activations cast to
    /// fp32 on the GPU first) so the state precision matches the
    /// canonical legacy path exactly.
    public private(set) var current: Tensor
    public private(set) var next: Tensor

    /// Step counter — incremented once per decode step for accounting.
    /// The underlying storage is constant-size regardless of length.
    public private(set) var length: Int = 0

    /// GDN state is not length-bound; report `.max` so capacity gates
    /// treat GDN layers as unlimited (mirrors `SSMStateCache`).
    public let capacity: Int = .max

    /// The device whose residency set holds the state buffers — kept so
    /// `deinit` can release them.
    private let device: Device

    public init(
        numValueHeads: Int, valueHeadDim: Int, keyHeadDim: Int,
        device: Device = .shared
    ) {
        self.device = device
        precondition(
            numValueHeads > 0,
            "GDNStateCache: numValueHeads must be positive")
        precondition(
            valueHeadDim > 0,
            "GDNStateCache: valueHeadDim must be positive")
        precondition(
            keyHeadDim > 0,
            "GDNStateCache: keyHeadDim must be positive")
        self.numValueHeads = numValueHeads
        self.valueHeadDim = valueHeadDim
        self.keyHeadDim = keyHeadDim
        let shape = [numValueHeads, valueHeadDim, keyHeadDim]
        self.current = Tensor.empty(shape: shape, dtype: .f32, device: device)
        self.next = Tensor.empty(shape: shape, dtype: .f32, device: device)
        self.current.zero()
        self.next.zero()
        // Recurrent state buffers persist across every decode step; pin
        // them in the residency set so the driver skips per-dispatch
        // residency validation on the read/write/swap that fires each
        // token through the GDN layer.
        device.markWeightsResident([self.current.buffer, self.next.buffer])
    }

    deinit {
        // Release the recurrent state buffers from the residency set so
        // the wired memory is freed when the cache is dropped.
        device.unmarkWeightsResident([current.buffer, next.buffer])
    }

    /// Exchange `current` and `next`. Call this after `Ops.gatedDeltaStep`
    /// or `Ops.gatedDeltaPrepStep` has written the updated state into
    /// `next`, so the next decode step reads the fresh state via
    /// `current`. Increments `length`.
    public func swap() {
        Swift.swap(&current, &next)
        length += 1
    }

    /// Reset the recurrent state to zero. Cheap (zero-fill of two small
    /// fp32 buffers); cache size doesn't depend on sequence length.
    public func reset() {
        current.zero()
        next.zero()
        length = 0
    }

    /// Bytes physically allocated — both state buffers. Independent of
    /// how many tokens have been processed (the GDN compression win).
    public var bytesAllocated: Int {
        2 * numValueHeads * valueHeadDim * keyHeadDim * DType.f32.byteSize
    }

    /// GDN state storage is constant-size; "in use" equals allocated
    /// once any step has executed, zero before then.
    public var bytesInUse: Int {
        length == 0 ? 0 : bytesAllocated
    }

    /// Cached snapshot tensor. Reused across snapshot calls so
    /// spec-decode doesn't churn ~2 MB per layer per verify step.
    private var cachedSnapshot: Tensor?

    /// Copy `current` into a fresh (or cached) snapshot tensor. Used
    /// by spec-decode to roll back the recurrent state on draft reject.
    /// Returns the snapshot tensor; caller stores it until needed.
    ///
    /// Reuse contract: this method returns a single per-instance
    /// scratch tensor. Calling `snapshot()` a second time before
    /// `restore(from:)` will overwrite the prior snapshot. Nested or
    /// concurrent snapshot usage on the same cache is not supported.
    public func snapshot(device: Device = .shared) -> Tensor {
        let shape = [numValueHeads, valueHeadDim, keyHeadDim]
        if cachedSnapshot == nil {
            cachedSnapshot = Tensor.empty(
                shape: shape, dtype: .f32, device: device)
        }
        let snap = cachedSnapshot!
        let cmd = device.makeCommandBuffer()
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure(
                "GDNStateCache.snapshot: makeBlitCommandEncoder failed")
        }
        let bytes =
            numValueHeads * valueHeadDim * keyHeadDim * DType.f32.byteSize
        blit.copy(
            from: current.buffer, sourceOffset: current.offset,
            to: snap.buffer, destinationOffset: snap.offset,
            size: bytes)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        return snap
    }

    /// Restore from a snapshot taken via `snapshot()`. Overwrites
    /// `current` with the snapshot contents; leaves `next` alone (it'll
    /// be overwritten on the next kernel dispatch). Does NOT decrement
    /// `length` — caller must `setLength(...)` to match.
    public func restore(from snapshot: Tensor, device: Device = .shared) {
        let expected = numValueHeads * valueHeadDim * keyHeadDim
        precondition(
            snapshot.elementCount == expected,
            "GDNStateCache.restore: snapshot has \(snapshot.elementCount) elements, expected \(expected)"
        )
        precondition(
            snapshot.dtype == .f32,
            "GDNStateCache.restore: snapshot must be f32")
        let cmd = device.makeCommandBuffer()
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure(
                "GDNStateCache.restore: makeBlitCommandEncoder failed")
        }
        let bytes = expected * DType.f32.byteSize
        blit.copy(
            from: snapshot.buffer, sourceOffset: snapshot.offset,
            to: current.buffer, destinationOffset: current.offset,
            size: bytes)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    /// Set the position counter directly without zeroing buffers.
    /// Spec-decode restore path uses this after writing the snapshot
    /// tensor into `current`. `reset()` + `swap()` would wipe the
    /// just-restored state.
    public func setLength(_ length: Int) {
        precondition(length >= 0, "GDNStateCache.setLength: must be ≥ 0")
        self.length = length
    }
}

extension Array where Element == GDNStateCache {
    /// Sum of `bytesAllocated` across all per-layer GDN caches.
    public var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
}
