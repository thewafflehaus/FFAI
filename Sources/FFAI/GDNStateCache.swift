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
// Forward-only. This is Phase 5e.C scope: there are deliberately NO
// `record()` / `rollback()` hooks. Partial-accept replay (rewinding
// GDN state when speculative tokens are rejected) is deferred to
// Phase 8 — adding replay support now would mean snapshotting the
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
    public let maxSeq: Int = .max

    public init(numValueHeads: Int, valueHeadDim: Int, keyHeadDim: Int,
                device: Device = .shared) {
        precondition(numValueHeads > 0,
                     "GDNStateCache: numValueHeads must be positive")
        precondition(valueHeadDim > 0,
                     "GDNStateCache: valueHeadDim must be positive")
        precondition(keyHeadDim > 0,
                     "GDNStateCache: keyHeadDim must be positive")
        self.numValueHeads = numValueHeads
        self.valueHeadDim = valueHeadDim
        self.keyHeadDim = keyHeadDim
        let shape = [numValueHeads, valueHeadDim, keyHeadDim]
        self.current = Tensor.empty(shape: shape, dtype: .f32, device: device)
        self.next = Tensor.empty(shape: shape, dtype: .f32, device: device)
        self.current.zero()
        self.next.zero()
        // GDN state buffers live for the lifetime of a generation. Pin
        // them in the device residency set — every fused-GDN dispatch
        // reads + writes them, so skipping per-encode residency
        // tracking matters at 30 GDN layers × per-token cadence.
        device.markWeightsResident([self.current.buffer, self.next.buffer])
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

    /// Snapshot the current state to a fresh `Tensor`. Used by
    /// speculative-decode to remember the pre-speculative state so it
    /// can be restored if drafted tokens are rejected.
    ///
    /// GDN is recurrent — once the kernel has folded a wrong token
    /// into `current`, you cannot subtract it back. Snapshot before
    /// each speculative forward, restore on full rejection.
    ///
    /// Snapshot cost: one `memcpy` of `[Hv, Dv, Dk]` fp32 = ~2 MiB on
    /// Qwen3.6-A3B per layer × 30 layers = ~60 MiB. On Apple silicon
    /// unified memory at ~400 GB/s this is ~150 µs per layer in
    /// aggregate — negligible against a ~60 ms decode step.
    ///
    /// Returns the snapshot tensor; caller stores it until needed.
    public func snapshot(device: Device = .shared) -> Tensor {
        let shape = [numValueHeads, valueHeadDim, keyHeadDim]
        let snap = Tensor.empty(shape: shape, dtype: .f32, device: device)
        // Use a blit encoder to copy on the GPU side — keeps the host
        // out of the loop. Both buffers are shared-storage, so the
        // copy is just a unified-memory blit.
        let cmd = device.makeCommandBuffer()
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure("GDNStateCache.snapshot: makeBlitCommandEncoder failed")
        }
        let bytes = numValueHeads * valueHeadDim * keyHeadDim * DType.f32.byteSize
        blit.copy(from: current.buffer, sourceOffset: current.offset,
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
    /// `length` — caller is responsible for matching the position
    /// counter to the restored state.
    public func restore(from snapshot: Tensor, device: Device = .shared) {
        let expected = numValueHeads * valueHeadDim * keyHeadDim
        precondition(snapshot.elementCount == expected,
                     "GDNStateCache.restore: snapshot has \(snapshot.elementCount) elements, expected \(expected)")
        precondition(snapshot.dtype == .f32,
                     "GDNStateCache.restore: snapshot must be f32 (matches state buffer dtype)")
        let cmd = device.makeCommandBuffer()
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure("GDNStateCache.restore: makeBlitCommandEncoder failed")
        }
        let bytes = expected * DType.f32.byteSize
        blit.copy(from: snapshot.buffer, sourceOffset: snapshot.offset,
                  to: current.buffer, destinationOffset: current.offset,
                  size: bytes)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
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
}

public extension Array where Element == GDNStateCache {
    /// Sum of `bytesAllocated` across all per-layer GDN caches.
    var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
}
