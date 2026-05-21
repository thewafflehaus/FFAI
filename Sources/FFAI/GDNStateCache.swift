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
}

public extension Array where Element == GDNStateCache {
    /// Sum of `bytesAllocated` across all per-layer GDN caches.
    var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
}
