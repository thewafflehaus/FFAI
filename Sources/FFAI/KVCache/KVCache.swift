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
// KVCache — one cache per attention layer. Pre-allocated to maxSeq
// capacity; `length` tracks how many positions are currently filled.
//
// Two implementations conform to `KVCacheProtocol`:
//   - `KVCache`                  : raw fp16/bf16 K/V (default)
//   - `AffineQuantizedKVCache`   : int8 group-quantized K/V.
//     int4 + int6 land as later commits — kernels exist for those
//     bit widths but the cache is int8-only today.
//
// The forward path calls `cache.prepareForAttention(on: cmd)` before
// SDPA, which is a no-op for raw caches and runs the bulk-dequant
// kernel for quantized caches. The returned (k, v) tensors have
// shape [nKVHeads, maxSeq, headDim] — the kvStride passed to SDPA
// is `maxSeq`, n_kv is the live `length`.

import Foundation
import Metal

// MARK: - LayerCacheProtocol (parent, used by Mamba 2 too)

/// Universal per-layer state cache contract. All recurrent / attention
/// caches conform to this so the engine + Generate can drive memory
/// accounting and reset without caring about the layer's flavor.
/// Attention layers add the K/V-specific surface in `KVCacheProtocol`
/// below; SSM layers conform to this protocol directly (see
/// `Mamba2LayerCache`).
public protocol LayerCacheProtocol: AnyObject, Sendable {
    /// Number of timesteps this cache has consumed. Grows monotonically
    /// for attention caches, fixed at 0 for stateless variants. SSM
    /// caches count steps for accounting purposes but the state size
    /// itself is constant.
    var length: Int { get }
    /// Maximum number of timesteps the cache was sized for. SSM caches
    /// report `.max` since their state is not length-bound.
    var maxSeq: Int { get }
    /// Bytes physically allocated for this cache's persistent storage.
    var bytesAllocated: Int { get }
    /// Bytes used by the live slice (`length` rows).
    var bytesInUse: Int { get }

    /// Reset internal state to zero. Doesn't reclaim memory.
    func reset()
}

// MARK: - KVCacheProtocol (attention-specific)

public protocol KVCacheProtocol: LayerCacheProtocol {
    var nKVHeads: Int { get }
    var headDim: Int { get }
    var dtype: DType { get }

    /// Eviction policy. `.unbounded` means classic monotonic growth
    /// up to `maxSeq` and a panic on overflow. `.window(maxSize:keep:)`
    /// enables FIFO ring-buffer rotation past `maxSize` positions,
    /// optionally pinning `keep` attention-sink slots at the front.
    var eviction: KVEviction { get }

    /// Maximum positions the cache retains — `maxSize` from
    /// `.window(...)` or `maxSeq` for `.unbounded`. Returned to the
    /// engine for cache-size reporting + sliding-window attention
    /// mask construction.
    var effectiveMaxSize: Int { get }

    /// Monotonic count of tokens ever appended (does not reset on
    /// eviction). Used by the engine to derive the next RoPE position.
    var absolutePosition: Int { get }

    /// Append one timestep's K and V (each [nKVHeads, headDim]) on the
    /// GPU. Queued on `cmd`; no commit/wait. Bumps `length`.
    func appendOnGPU(kFlat: Tensor, vFlat: Tensor, on cmd: MTLCommandBuffer)

    /// Produce K and V tensors usable by `Ops.sdpaDecode(...)`. For
    /// raw caches this returns the storage directly with no work
    /// queued. For quantized caches this queues a bulk-dequant kernel
    /// onto `cmd` writing into the shared working buffer pair, and
    /// returns that pair. Both returned tensors have shape
    /// `[nKVHeads, maxSeq, headDim]` — SDPA's `kvStride = maxSeq`,
    /// `nKV = length`.
    func prepareForAttention(on cmd: MTLCommandBuffer) -> (k: Tensor, v: Tensor)

    /// Roll the cache back to `length` filled positions, discarding the
    /// tail. Physical K/V storage is left intact — the next append
    /// overwrites the discarded slots. Used by speculative decoding to
    /// drop rejected draft tokens after an AR verify pass (e.g.
    /// Nemotron-Labs-Diffusion self-speculation). `.unbounded` caches
    /// only; `.window` rejects once the ring buffer has rotated — see
    /// `KVEvictionState.truncate(toLength:)`.
    func truncate(toLength length: Int)
}

extension KVCacheProtocol {
    /// Default for callers that pre-date the sliding-window addition —
    /// behaves like a non-rotating cache. Concrete classes override
    /// when they wire `KVEvictionState` in.
    public var eviction: KVEviction { .unbounded }
    public var effectiveMaxSize: Int { maxSeq }
    public var absolutePosition: Int { length }

    /// Sink + window bounds for `Ops.sdpaDecode`'s sliding-window fast
    /// path, derived from the eviction policy. Returns
    /// `(sinkEnd, windowStart)` for a cache whose live span is `nKV`
    /// physical slots.
    ///
    /// FFAI's KV caches (raw / affine / AURA) all back a `.window`
    /// policy with a ring buffer that keeps live data **contiguous** in
    /// physical slots `[0, length)` — `length` saturates at `maxSize`,
    /// and `KVEvictionState.reserveNextSlot()` writes sinks into
    /// `[0, keep)` then rings within `[keep, maxSize)`. So the kernel
    /// already sees a flat, gap-free `[0, nKV)` range and the dense
    /// path `(0, 0)` is numerically exact. The sparse fast path only
    /// pays off for a non-ring "grow-then-mask" layout (e.g. a future
    /// paged / batched cache) where the live window is a sub-range of a
    /// larger contiguous buffer — such a cache overrides this method.
    ///
    /// Returning `(0, 0)` here means windowed callers can unconditionally
    /// thread `cache.sdpaSinkWindow(nKV:)` into `Ops.sdpaDecode` without
    /// branching on the cache kind; the API is wired end-to-end and a
    /// new cache layout only has to change this one method.
    public func sdpaSinkWindow(nKV: Int) -> (sinkEnd: Int, windowStart: Int) {
        (sinkEnd: 0, windowStart: 0)
    }
}

// MARK: - KVCache (raw fp16 / bf16)

public final class KVCache: KVCacheProtocol, @unchecked Sendable {
    /// Default starting capacity for an incrementally-grown unbounded
    /// cache. The buffer is allocated to this depth (or the context
    /// ceiling, whichever is smaller) at init, and `ensureCapacity`
    /// doubles it on demand as `length` approaches the current
    /// capacity.
    ///
    /// 2048 is chosen so the entire common operating range incurs ZERO
    /// reallocations: decode throughput peaks between ~2K–4K context on
    /// Apple Silicon before the quadratic attention term and KV-bandwidth
    /// pressure start to dominate (past that, sparse-decode + KV
    /// retention/eviction strategies are the lever — a later phase). A
    /// session that never exceeds 2048 tokens therefore allocates once
    /// and never copies; longer sessions double from here. The footprint
    /// is modest even on big models (e.g. Qwen3.6-27B at 16 attention
    /// layers × 4 KV heads × 2048 × 256 × 2(K+V) × 2 bytes ≈ 128 MB vs
    /// the ~16 GB a full 256K-context pre-allocation would cost).
    ///
    /// `public static var` so it's a global tuning knob —
    /// `KVCache.defaultInitialCapacity = 1024` lowers the baseline
    /// allocation; a per-cache override is also available via the
    /// `initialCapacity:` init parameter.
    ///
    /// `nonisolated(unsafe)`: this is a process-global startup tuning
    /// knob, expected to be set once before any model load (model loads
    /// serialise behind `ModelLoadLock`, and cache construction is
    /// single-threaded per decode). It is not mutated during concurrent
    /// inference, so no synchronisation is warranted.
    nonisolated(unsafe) public static var defaultInitialCapacity = 2048

    public let nKVHeads: Int
    public let headDim: Int
    public let dtype: DType

    /// The maximum depth this cache may grow to — the chosen context
    /// window (`maxContextLength`, or the model's
    /// `max_position_embeddings`). NOT the physical allocation; see
    /// `capacity` for that. Exposed as the growth ceiling so callers
    /// can reason about the worst-case footprint.
    public let contextCeiling: Int

    /// Physical depth currently allocated for `kBuffer` / `vBuffer`
    /// (the middle dim of `[nKVHeads, capacity, headDim]`). This is the
    /// `kvStride` every SDPA + append dispatch must use. Grows (via
    /// `ensureCapacity`) up to `contextCeiling`; never shrinks. For a
    /// pre-allocated cache it equals `contextCeiling` from the start.
    public private(set) var capacity: Int

    /// `maxSeq` is the SDPA / append buffer stride — i.e. the CURRENT
    /// physical `capacity`, not the growth ceiling. Returning capacity
    /// here keeps every existing `kvStride: cache.maxSeq` call site
    /// correct as the buffer grows (the buffer's middle-dim stride IS
    /// the current capacity). Use `contextCeiling` for the max-growth
    /// bound and `effectiveMaxSize` for the sliding-window / reporting
    /// size.
    public var maxSeq: Int { capacity }

    public private(set) var kBuffer: Tensor  // [nKVHeads, capacity, headDim]
    public private(set) var vBuffer: Tensor  // [nKVHeads, capacity, headDim]

    /// Retained device handle for reallocating the backing buffers on
    /// growth (and re-pinning them into the residency set).
    private let device: Device
    /// Old backing buffers kept alive until at least one growth-copy
    /// command buffer has completed. Cleared lazily on the next grow.
    private var retiredBuffers: [Tensor] = []

    /// Lock-protected fill state. Safe today even without the lock —
    /// single-threaded decode — but planned's batched / speculative
    /// decode coordinates multiple Tasks against one cache. The lock
    /// makes the `(reserve slot, queue dispatch, increment)` sequence
    /// in `appendOnGPU` atomic so concurrent appenders don't write to
    /// the same physical slot.
    ///
    /// `NSLock` rather than `OSAllocatedUnfairLock<KVEvictionState>`
    /// because the critical section captures `MTLCommandBuffer` (not
    /// Sendable), which `OSAllocatedUnfairLock.withLock`'s `@Sendable`
    /// closure signature rejects. NSLock has no such restriction and
    /// the performance difference at decode-step granularity is noise.
    private let lengthLock = NSLock()
    private var _evictionState: KVEvictionState
    public var length: Int { lengthLock.withLock { _evictionState.length } }
    public var absolutePosition: Int { lengthLock.withLock { _evictionState.absolutePosition } }
    public var eviction: KVEviction { _evictionState.policy }
    public var effectiveMaxSize: Int {
        switch _evictionState.policy {
        // Unbounded reports the growth ceiling (the chosen context),
        // NOT the current physical capacity — `effectiveMaxSize` is the
        // "how long can this conversation get" number used for
        // reporting + mask construction, independent of how much has
        // been allocated so far.
        case .unbounded: return contextCeiling
        case .window(let m, _): return m
        }
    }

    public convenience init(
        nKVHeads: Int, headDim: Int, maxSeq: Int, dtype: DType,
        device: Device = .shared
    ) {
        self.init(
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
            dtype: dtype, eviction: .unbounded, device: device)
    }

    /// - Parameters:
    ///   - maxSeq: the context ceiling — the maximum depth the cache may
    ///     grow to. With `preallocate == false` (the default for
    ///     `.unbounded`) the physical buffer starts at
    ///     `min(maxSeq, defaultInitialCapacity)` and doubles on demand;
    ///     with `preallocate == true` it is allocated to `maxSeq` up
    ///     front (the legacy behaviour, and the path callers that stage
    ///     into the free tail — e.g. diffusion-block forwards — must
    ///     use).
    ///   - preallocate: force full-`maxSeq` allocation at init. Defaults
    ///     to `false` for `.unbounded` (incremental growth) and is
    ///     forced `true` for `.window` (the ring buffer is sized to its
    ///     `maxSize` window and never grows).
    ///   - initialCapacity: starting physical depth for an unbounded,
    ///     non-preallocated cache. `nil` (default) uses the global
    ///     `KVCache.defaultInitialCapacity`. Clamped to `maxSeq` (never
    ///     start larger than the ceiling). Ignored when `preallocate`
    ///     is true or the policy is `.window`.
    public init(
        nKVHeads: Int, headDim: Int, maxSeq: Int, dtype: DType,
        eviction: KVEviction,
        preallocate: Bool = false,
        initialCapacity: Int? = nil,
        device: Device = .shared
    ) {
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.contextCeiling = maxSeq
        self.dtype = dtype
        self.device = device

        // Initial physical capacity. Both unbounded AND window caches
        // start small and grow on demand — a window grows linearly up to
        // its `maxSize` ring and only then begins rotating, so a 16K
        // window that only sees 3K tokens never allocates the full 16K.
        // `preallocate` forces the full target up front (the ceiling for
        // unbounded, the ring size for a window).
        let startCapacity: Int
        switch eviction {
        case .window(let maxSize, _):
            if preallocate {
                startCapacity = maxSize
            } else {
                let requested = initialCapacity ?? KVCache.defaultInitialCapacity
                startCapacity = Swift.min(maxSize, Swift.max(1, requested))
            }
        case .unbounded:
            if preallocate {
                startCapacity = maxSeq
            } else {
                let requested = initialCapacity ?? KVCache.defaultInitialCapacity
                startCapacity = Swift.min(maxSeq, Swift.max(1, requested))
            }
        }
        self.capacity = startCapacity

        self.kBuffer = Tensor.empty(
            shape: [nKVHeads, startCapacity, headDim], dtype: dtype, device: device)
        self.vBuffer = Tensor.empty(
            shape: [nKVHeads, startCapacity, headDim], dtype: dtype, device: device)
        self.kBuffer.zero()
        self.vBuffer.zero()
        self._evictionState = KVEvictionState(policy: eviction, bufferCapacity: startCapacity)
        // KV buffers live for the entire generation — pin them in the
        // device's residency set so per-dispatch residency tracking
        // doesn't fire on the thousands of decode-step appends + reads.
        device.markWeightsResident([self.kBuffer.buffer, self.vBuffer.buffer])
    }

    /// Ensure the backing buffers can hold `needed` total live rows,
    /// growing (realloc + GPU blit-copy of the live region) if the
    /// current `capacity` is short and the ceiling allows. Called at the
    /// top of every append path, BEFORE any work for the current token
    /// is queued on the caller's command buffer.
    ///
    /// GPU-timeline safety: the cache's live K/V (rows `[0, length)`) is
    /// always the product of PRIOR, already-committed forwards — the
    /// token being appended now still lives in the caller's scratch
    /// tensor, not the cache. So the growth copy (old buffer → new
    /// buffer) reads only quiesced data and runs on its own command
    /// buffer that is committed + waited here, with no dependency on the
    /// caller's in-flight `cmd`. The new (larger) buffer is then the
    /// target of the caller's append + the source of its SDPA read,
    /// both correctly ordered on the caller's `cmd`.
    ///
    /// Grows for both `.unbounded` (toward `contextCeiling`) and
    /// `.window` (toward `maxSize`, after which rotation takes over —
    /// the ring's linear pre-fill region grows on demand). No-op once
    /// `capacity` has reached the relevant ceiling.
    private func ensureCapacityLocked(_ needed: Int) {
        // The depth this cache may grow to: the rotation ring size for a
        // window, the context ceiling otherwise.
        let growthCeiling: Int
        switch _evictionState.policy {
        case .unbounded: growthCeiling = contextCeiling
        case .window(let maxSize, _): growthCeiling = maxSize
        }
        guard needed > capacity, capacity < growthCeiling else { return }

        // Geometric growth (double) gives O(log N) reallocs + O(N) total
        // copy while keeping reads contiguous; clamp to the ceiling and
        // never below `needed`.
        var newCapacity = capacity
        while newCapacity < needed { newCapacity *= 2 }
        newCapacity = Swift.min(newCapacity, growthCeiling)
        if newCapacity < needed {
            // `needed` exceeds the ceiling — let reserveNextSlot's own
            // precondition fire with its actionable message rather than
            // silently truncating here.
            newCapacity = growthCeiling
        }
        guard newCapacity > capacity else { return }

        let liveRows = _evictionState.length
        let newK = Tensor.empty(
            shape: [nKVHeads, newCapacity, headDim], dtype: dtype, device: device)
        let newV = Tensor.empty(
            shape: [nKVHeads, newCapacity, headDim], dtype: dtype, device: device)
        newK.zero()
        newV.zero()

        // Copy the live region per head: head h's rows [0, liveRows) move
        // from old stride `capacity` to new stride `newCapacity`. Both
        // buffers are contiguous per head, so this is `nKVHeads` linear
        // blits. Runs on a dedicated command buffer committed + waited
        // here (old data is quiesced — see the doc comment).
        if liveRows > 0 {
            let cmd = device.makeCommandBuffer()
            guard let blit = cmd.makeBlitCommandEncoder() else {
                fatalError("KVCache.ensureCapacity: makeBlitCommandEncoder returned nil")
            }
            let rowBytes = headDim * dtype.byteSize
            let copyBytes = liveRows * rowBytes
            for h in 0 ..< nKVHeads {
                let oldHeadOffset = h * capacity * rowBytes
                let newHeadOffset = h * newCapacity * rowBytes
                blit.copy(
                    from: kBuffer.buffer, sourceOffset: kBuffer.offset + oldHeadOffset,
                    to: newK.buffer, destinationOffset: newK.offset + newHeadOffset,
                    size: copyBytes)
                blit.copy(
                    from: vBuffer.buffer, sourceOffset: vBuffer.offset + oldHeadOffset,
                    to: newV.buffer, destinationOffset: newV.offset + newHeadOffset,
                    size: copyBytes)
            }
            blit.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // Retire old buffers (kept until the next grow so nothing
        // dangles), swap in the new ones, bump capacity + eviction
        // bookkeeping, and pin the new buffers resident.
        retiredBuffers.append(contentsOf: [kBuffer, vBuffer])
        if retiredBuffers.count > 4 { retiredBuffers.removeFirst(retiredBuffers.count - 4) }
        kBuffer = newK
        vBuffer = newV
        capacity = newCapacity
        _evictionState.grow(to: newCapacity)
        device.markWeightsResident([newK.buffer, newV.buffer])
    }

    /// CPU-side legacy append. Caller must have already sync'd the
    /// command buffer that produced kFlat / vFlat. Kept for tests +
    /// callers that don't have a live MTLCommandBuffer; the inference
    /// path uses `appendOnGPU` instead, which is sync-free.
    public func append(kFlat: Tensor, vFlat: Tensor) {
        precondition(kFlat.dtype == dtype && vFlat.dtype == dtype, "KVCache: dtype mismatch")
        lengthLock.withLock {
            ensureCapacityLocked(_evictionState.length + 1)
            let pos = _evictionState.reserveNextSlot()
            let bytesPerHead = headDim * dtype.byteSize
            let kSrc = kFlat.buffer.contents().advanced(by: kFlat.offset)
            let vSrc = vFlat.buffer.contents().advanced(by: vFlat.offset)
            for h in 0 ..< nKVHeads {
                let dstHeadOffset = (h * maxSeq + pos) * headDim * dtype.byteSize
                let kDst = kBuffer.buffer.contents().advanced(by: kBuffer.offset + dstHeadOffset)
                let vDst = vBuffer.buffer.contents().advanced(by: vBuffer.offset + dstHeadOffset)
                let srcOffset = h * bytesPerHead
                kDst.copyMemory(from: kSrc.advanced(by: srcOffset), byteCount: bytesPerHead)
                vDst.copyMemory(from: vSrc.advanced(by: srcOffset), byteCount: bytesPerHead)
            }
        }
    }

    /// Append one timestep on the GPU via Ops.kvCacheUpdate. The dispatch
    /// is queued on `cmd`; no commit/wait happens here. Caller is
    /// responsible for ensuring the command buffer that produced kFlat /
    /// vFlat is the same one (or a strictly-prior one) so dependencies
    /// are honored. Bumps `length` immediately so subsequent SDPA
    /// dispatches see the updated count.
    public func appendOnGPU(
        kFlat: Tensor, vFlat: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(kFlat.dtype == dtype && vFlat.dtype == dtype, "KVCache: dtype mismatch")
        lengthLock.withLock {
            ensureCapacityLocked(_evictionState.length + 1)
            let pos = _evictionState.reserveNextSlot()
            Ops.kvCacheUpdate(
                src: kFlat, into: kBuffer,
                nKVHeads: nKVHeads, headDim: headDim,
                maxSeq: maxSeq, position: pos, on: cmd)
            Ops.kvCacheUpdate(
                src: vFlat, into: vBuffer,
                nKVHeads: nKVHeads, headDim: headDim,
                maxSeq: maxSeq, position: pos, on: cmd)
        }
    }

    public func reset() { lengthLock.withLock { _evictionState.reset() } }

    public func truncate(toLength length: Int) {
        lengthLock.withLock { _evictionState.truncate(toLength: length) }
    }

    /// Append `kRows`/`vRows` (each `[nKVHeads, headDim]`) as consecutive
    /// timesteps in one call, bumping `length` by `kRows.count`. Used by
    /// multi-token forwards (diffusion-block commit, multi-token
    /// prefill) — equivalent to N back-to-back `appendOnGPU` calls but
    /// takes the length lock once.
    public func appendRangeOnGPU(
        kRows: [Tensor], vRows: [Tensor],
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            kRows.count == vRows.count,
            "KVCache.appendRangeOnGPU: kRows (\(kRows.count)) / vRows "
                + "(\(vRows.count)) count mismatch")
        lengthLock.withLock {
            // Grow once for the whole range before queuing any writes —
            // mid-range growth would orphan the writes already queued on
            // `cmd` against the old buffer.
            ensureCapacityLocked(_evictionState.length + kRows.count)
            for (kFlat, vFlat) in zip(kRows, vRows) {
                precondition(
                    kFlat.dtype == dtype && vFlat.dtype == dtype,
                    "KVCache.appendRangeOnGPU: dtype mismatch")
                let pos = _evictionState.reserveNextSlot()
                Ops.kvCacheUpdate(
                    src: kFlat, into: kBuffer,
                    nKVHeads: nKVHeads, headDim: headDim,
                    maxSeq: maxSeq, position: pos, on: cmd)
                Ops.kvCacheUpdate(
                    src: vFlat, into: vBuffer,
                    nKVHeads: nKVHeads, headDim: headDim,
                    maxSeq: maxSeq, position: pos, on: cmd)
            }
        }
    }

    /// Batched range append: takes contiguous `[T, nKVHeads, headDim]`
    /// flat K + V tensors and writes them all in ONE shared encoder
    /// (2 dispatches: K then V) using a `[T]` u32 positions buffer.
    /// Caller provides the positions buffer (allocated + filled with
    /// the freshly-reserved slot indices). Replaces the T-loop of
    /// `Ops.kvCacheUpdateKV` calls.
    public func appendRangeOnGPUMany(
        kFlat: Tensor, vFlat: Tensor,
        t: Int, positions: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            kFlat.dtype == dtype && vFlat.dtype == dtype,
            "KVCache.appendRangeOnGPUMany: dtype mismatch")
        precondition(
            positions.dtype == .u32 && positions.elementCount == t,
            "KVCache.appendRangeOnGPUMany: positions must be .u32[T]")
        Ops.kvCacheUpdateKVMany(
            kSrc: kFlat, kCache: kBuffer,
            vSrc: vFlat, vCache: vBuffer,
            positions: positions, t: t,
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
            on: cmd)
    }

    /// Reserve T sequential physical slots in the cache, writing the
    /// chosen indices into `positionsOut` (a u32 buffer length ≥ T).
    /// Atomic under `lengthLock`. Returns nothing — caller passes the
    /// positions tensor straight to `appendRangeOnGPUMany`.
    public func reserveSlotsManyOnHost(t: Int, into positionsOut: Tensor) {
        precondition(
            positionsOut.dtype == .u32,
            "KVCache.reserveSlotsManyOnHost: positionsOut must be .u32")
        precondition(
            positionsOut.elementCount >= t,
            "KVCache.reserveSlotsManyOnHost: positionsOut shorter than T")
        let ptr = positionsOut.buffer.contents().advanced(by: positionsOut.offset)
            .bindMemory(to: UInt32.self, capacity: t)
        lengthLock.withLock {
            // Grow before reserving so the subsequent
            // `appendRangeOnGPUMany` (which reads `kBuffer`/`maxSeq` at
            // call time) targets the grown buffer at the correct stride.
            ensureCapacityLocked(_evictionState.length + t)
            for r in 0 ..< t { ptr[r] = UInt32(_evictionState.reserveNextSlot()) }
        }
    }

    /// Write one timestep's K/V at an explicit physical slot **without**
    /// touching `length`. Diffusion-block forwards stage their scratch
    /// K/V in the buffer's free region `[length, maxSeq)` across denoise
    /// iterations before a final commit. Caller guarantees the slot is
    /// free. No lock — `length` is unchanged, so no shared state moves.
    public func writeTimestepOnGPU(
        kFlat: Tensor, vFlat: Tensor,
        atSlot slot: Int, on cmd: MTLCommandBuffer
    ) {
        precondition(
            kFlat.dtype == dtype && vFlat.dtype == dtype,
            "KVCache.writeTimestepOnGPU: dtype mismatch")
        precondition(
            slot >= 0 && slot < maxSeq,
            "KVCache.writeTimestepOnGPU: slot \(slot) out of range 0..<\(maxSeq)")
        Ops.kvCacheUpdate(
            src: kFlat, into: kBuffer,
            nKVHeads: nKVHeads, headDim: headDim,
            maxSeq: maxSeq, position: slot, on: cmd)
        Ops.kvCacheUpdate(
            src: vFlat, into: vBuffer,
            nKVHeads: nKVHeads, headDim: headDim,
            maxSeq: maxSeq, position: slot, on: cmd)
    }

    /// Raw cache exposes its storage buffers directly; no per-step
    /// work is needed before SDPA.
    public func prepareForAttention(on cmd: MTLCommandBuffer) -> (k: Tensor, v: Tensor) {
        return (kBuffer, vBuffer)
    }

    // ─── Memory accounting (used by --stats / bench harness) ─────────

    /// Bytes the K + V buffers physically occupy in device memory. Set
    /// at construction time to `2 * nKVHeads * maxSeq * headDim *
    /// dtype.byteSize` and immutable thereafter — the buffer is
    /// preallocated.
    public var bytesAllocated: Int {
        2 * nKVHeads * maxSeq * headDim * dtype.byteSize
    }

    /// Bytes occupied by the in-use K + V slice (`length` rows out of
    /// `maxSeq`). What you'd report as the "live KV" delta in stats.
    public var bytesInUse: Int {
        2 * nKVHeads * length * headDim * dtype.byteSize
    }
}

// MARK: - AffineQuantizedKVCache

/// Affine group-quantized KV cache. Stores K and V in packed int8
/// form (one byte per element) plus per-group fp16/bf16 scales +
/// biases. Memory vs the raw cache: ~40% less at int8 for typical
/// shapes (Qwen3 1.7B 28×head_dim=128 fits in roughly 268MB vs 448MB
/// raw at maxSeq=4096).
///
/// All caches built in a single `makeKVCache(...)` call share **one**
/// working buffer pair (`sharedWorkingK` + `sharedWorkingV`) — sized
/// at `[nKVHeads, maxSeq, headDim]` in the model dtype. Per attention
/// step the cache writes its dequantized rows into the shared buffer
/// pair and returns those for SDPA. Metal's default hazard tracking
/// serializes the buffer reuse across layers within a cmdbuf.
///
/// Bit-width support today: **8**. int4 + int6 land as follow-up
/// commits — the metaltile-side kernels are bit-specific
/// (`quantize_kv_int8` / `bulk_dequant_kv_int8`); int4 + int6 will
/// add their own kernel pairs alongside.
public final class AffineQuantizedKVCache: KVCacheProtocol, @unchecked Sendable {
    public let nKVHeads: Int
    public let headDim: Int
    public let maxSeq: Int
    public let dtype: DType  // dtype of scales/biases + dequant output
    public let bits: Int  // 8 for now
    public let groupSize: Int

    // Compressed storage (int8 packed 4-per-uint32)
    public let kWeights: Tensor  // [nKVHeads, maxSeq, headDim / 4] u32
    public let vWeights: Tensor
    public let kScales: Tensor  // [nKVHeads, maxSeq, headDim / groupSize] T
    public let vScales: Tensor
    public let kBiases: Tensor
    public let vBiases: Tensor

    // Shared working buffers (passed in at construction by the
    // family's makeKVCache; reused across every layer's cache).
    public let sharedWorkingK: Tensor  // [nKVHeads, maxSeq, headDim] T
    public let sharedWorkingV: Tensor

    /// Lock-protected fill state. See `KVCache.lengthLock` for the
    /// rationale — same Phase-8 concurrent-decode prep.
    private let lengthLock = NSLock()
    private var _evictionState: KVEvictionState
    public var length: Int { lengthLock.withLock { _evictionState.length } }
    public var absolutePosition: Int { lengthLock.withLock { _evictionState.absolutePosition } }
    public var eviction: KVEviction { _evictionState.policy }
    public var effectiveMaxSize: Int {
        switch _evictionState.policy {
        case .unbounded: return maxSeq
        case .window(let m, _): return m
        }
    }

    public convenience init(
        nKVHeads: Int, headDim: Int, maxSeq: Int, dtype: DType,
        bits: Int, groupSize: Int,
        sharedWorkingK: Tensor, sharedWorkingV: Tensor,
        device: Device = .shared
    ) {
        self.init(
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
            dtype: dtype, bits: bits, groupSize: groupSize,
            sharedWorkingK: sharedWorkingK, sharedWorkingV: sharedWorkingV,
            eviction: .unbounded, device: device)
    }

    public init(
        nKVHeads: Int, headDim: Int, maxSeq: Int, dtype: DType,
        bits: Int, groupSize: Int,
        sharedWorkingK: Tensor, sharedWorkingV: Tensor,
        eviction: KVEviction,
        device: Device = .shared
    ) {
        precondition(
            bits == 4 || bits == 8,
            "AffineQuantizedKVCache: bits must be 4 or 8 today (int6 is a follow-up)")
        let valuesPerWord = 32 / bits  // int8 → 4, int4 → 8
        precondition(
            headDim % valuesPerWord == 0,
            "headDim must be divisible by \(valuesPerWord) for int\(bits) packing")
        precondition(headDim % groupSize == 0, "headDim must be divisible by groupSize")
        precondition(
            sharedWorkingK.shape == [nKVHeads, maxSeq, headDim],
            "sharedWorkingK shape mismatch")
        precondition(
            sharedWorkingV.shape == [nKVHeads, maxSeq, headDim],
            "sharedWorkingV shape mismatch")
        precondition(
            sharedWorkingK.dtype == dtype && sharedWorkingV.dtype == dtype,
            "sharedWorking dtype mismatch")

        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.bits = bits
        self.groupSize = groupSize

        let packsPerRow = headDim / valuesPerWord
        let groupsPerRow = headDim / groupSize
        self.kWeights = Tensor.empty(
            shape: [nKVHeads, maxSeq, packsPerRow], dtype: .u32, device: device)
        self.vWeights = Tensor.empty(
            shape: [nKVHeads, maxSeq, packsPerRow], dtype: .u32, device: device)
        self.kScales = Tensor.empty(
            shape: [nKVHeads, maxSeq, groupsPerRow], dtype: dtype, device: device)
        self.vScales = Tensor.empty(
            shape: [nKVHeads, maxSeq, groupsPerRow], dtype: dtype, device: device)
        self.kBiases = Tensor.empty(
            shape: [nKVHeads, maxSeq, groupsPerRow], dtype: dtype, device: device)
        self.vBiases = Tensor.empty(
            shape: [nKVHeads, maxSeq, groupsPerRow], dtype: dtype, device: device)

        self.sharedWorkingK = sharedWorkingK
        self.sharedWorkingV = sharedWorkingV

        kWeights.zero()
        vWeights.zero()
        kScales.zero()
        vScales.zero()
        kBiases.zero()
        vBiases.zero()

        self._evictionState = KVEvictionState(policy: eviction, bufferCapacity: maxSeq)
    }

    public func reset() { lengthLock.withLock { _evictionState.reset() } }

    public func truncate(toLength length: Int) {
        lengthLock.withLock { _evictionState.truncate(toLength: length) }
    }

    public func appendOnGPU(
        kFlat: Tensor, vFlat: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            kFlat.dtype == dtype && vFlat.dtype == dtype,
            "AffineQuantizedKVCache: dtype mismatch")
        lengthLock.withLock {
            let pos = _evictionState.reserveNextSlot()
            Ops.quantizeKVAffine(
                src: kFlat,
                weights: kWeights, scales: kScales, biases: kBiases,
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                groupSize: groupSize, position: pos, bits: bits, on: cmd)
            Ops.quantizeKVAffine(
                src: vFlat,
                weights: vWeights, scales: vScales, biases: vBiases,
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                groupSize: groupSize, position: pos, bits: bits, on: cmd)
        }
    }

    public func prepareForAttention(on cmd: MTLCommandBuffer) -> (k: Tensor, v: Tensor) {
        // Bulk-dequant into the shared working buffers. SDPA reads
        // from the working buffers; the cache's compressed storage
        // is the persistent state.
        guard length > 0 else { return (sharedWorkingK, sharedWorkingV) }
        Ops.bulkDequantKVAffine(
            weights: kWeights, scales: kScales, biases: kBiases,
            into: sharedWorkingK,
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
            groupSize: groupSize, nPositions: length, bits: bits, on: cmd)
        Ops.bulkDequantKVAffine(
            weights: vWeights, scales: vScales, biases: vBiases,
            into: sharedWorkingV,
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
            groupSize: groupSize, nPositions: length, bits: bits, on: cmd)
        return (sharedWorkingK, sharedWorkingV)
    }

    /// Storage cost for this cache's compressed buffers only.
    /// The shared working buffer is accounted once at the caller
    /// (model engine), not multiplied across layers.
    public var bytesAllocated: Int {
        let valuesPerWord = 32 / bits
        let packs = headDim / valuesPerWord
        let groups = headDim / groupSize
        // K + V × (weights + scales + biases)
        return 2
            * (nKVHeads * maxSeq * packs * 4
                + 2 * nKVHeads * maxSeq * groups * dtype.byteSize)
    }

    public var bytesInUse: Int {
        let valuesPerWord = 32 / bits
        let packs = headDim / valuesPerWord
        let groups = headDim / groupSize
        return 2
            * (nKVHeads * length * packs * 4
                + 2 * nKVHeads * length * groups * dtype.byteSize)
    }
}

// MARK: - Array helpers

extension Array where Element == any LayerCacheProtocol {
    /// Sum of `bytesAllocated` across all per-layer caches.
    public var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
    /// Sum of `bytesInUse` across all per-layer caches.
    public var totalBytesInUse: Int { reduce(0) { $0 + $1.bytesInUse } }
}

extension Array where Element == any KVCacheProtocol {
    /// Sum of `bytesAllocated` across all per-layer caches.
    public var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
    /// Sum of `bytesInUse` across all per-layer caches.
    public var totalBytesInUse: Int { reduce(0) { $0 + $1.bytesInUse } }
}

extension Array where Element == KVCache {
    /// Sum of `bytesAllocated` across all per-layer caches.
    public var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
    /// Sum of `bytesInUse` across all per-layer caches.
    public var totalBytesInUse: Int { reduce(0) { $0 + $1.bytesInUse } }
}
