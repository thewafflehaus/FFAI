// AURAQuantizedKVCache — KVCacheProtocol conformance that stores K and V
// as AURA-codec packed indices + per-position norm corrections.
//
// Storage layout (per layer, both K and V):
//   packed  [nKVHeads, maxSeq, packedWidth] u32   — bit-packed codebook indices
//   norms   [nKVHeads, maxSeq]              f32   — per-position norm correction
//   working [nKVHeads, maxSeq, headDim]     dtype — shared across layers; bulk-dequant target
//
// `packedWidth = ceil(headDim * bits / 32)`. K and V use independent bit
// widths (the production aura4v2 recipe is K=4-bit, V=2-bit per
// `papers/aura-compression-algorithm.md` §2.5).
//
// ─── First-light rotation choice ────────────────────────────────────
//
// This implementation uses `AURARotation.identityMatrix(dim:)` as Π.
// Identity rotation is the "first-light" path the AURARotation doc
// explicitly calls out: it exercises the encode/decode kernels end-to-
// end without requiring the SRHT-based rotation + W_o offline fold
// described in §2.2 of the AURA paper. Compression quality is worse
// than SRHT (the codebook assumes Beta-distributed rotated coords);
// the smoke-test bar is "produces coherent output," not optimal PPL.
//
// SRHT integration + W_o pre-multiply lands as a Phase 5d.E follow-up.
//
// ─── Dispatch granularity ───────────────────────────────────────────
//
// `appendOnGPU` runs `nKVHeads` encode dispatches per K (and the same
// per V): the encode kernel writes `[rows, packed_width]` contiguous
// output, but our storage layout interleaves heads as the outer
// dimension so per-position writes for different heads are not
// contiguous. We work around this by issuing one encode per head with
// a per-head buffer view. For Qwen3 1.7B (nKVHeads=8, 28 layers) that's
// 16 encode dispatches per token per layer = 448 per token total — a
// real perf cost vs raw bf16 KV. Production speedup paths:
//
//   1. Strided-output encode kernel (one dispatch writes all heads).
//   2. Layout change to `[maxSeq, nKVHeads, packedWidth]` + matching
//      dequant kernel rewrite.
//
// Both queued for the Phase 5d.E perf pass.

import Foundation
import Metal

public final class AURAQuantizedKVCache: KVCacheProtocol, @unchecked Sendable {
    public let nKVHeads: Int
    public let headDim: Int
    public let maxSeq: Int
    public let dtype: DType
    public let scheme: AURAScheme

    public let kPackedWidth: Int
    public let vPackedWidth: Int

    // Codec data — shared across all layers' caches in a model.
    // Constructed once by makeLayerCaches and passed in here.
    public let rotation: Tensor       // [headDim, headDim] f32
    public let kCodebook: Tensor      // [2^keyBits]   f32
    public let kBoundaries: Tensor    // [2^keyBits-1] f32
    public let vCodebook: Tensor      // [2^valueBits] f32
    public let vBoundaries: Tensor    // [2^valueBits-1] f32

    // Per-cache compressed storage.
    public let kPacked: Tensor        // [nKVHeads, maxSeq, kPackedWidth] u32
    public let vPacked: Tensor        // [nKVHeads, maxSeq, vPackedWidth] u32
    public let kNorms: Tensor         // [nKVHeads, maxSeq] f32
    public let vNorms: Tensor         // [nKVHeads, maxSeq] f32

    // Shared working buffers — bulk-dequant target; reused across layers.
    public let sharedWorkingK: Tensor // [nKVHeads, maxSeq, headDim] dtype
    public let sharedWorkingV: Tensor

    /// Lock-protected fill state. Same Phase-8 concurrent-decode prep
    /// reasoning as `KVCache.lengthLock` — see `Sources/FFAI/KVCache.swift`.
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
        scheme: AURAScheme,
        rotation: Tensor,
        kCodebook: Tensor, kBoundaries: Tensor,
        vCodebook: Tensor, vBoundaries: Tensor,
        sharedWorkingK: Tensor, sharedWorkingV: Tensor,
        device: Device = .shared
    ) {
        self.init(nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                  dtype: dtype, scheme: scheme, rotation: rotation,
                  kCodebook: kCodebook, kBoundaries: kBoundaries,
                  vCodebook: vCodebook, vBoundaries: vBoundaries,
                  sharedWorkingK: sharedWorkingK, sharedWorkingV: sharedWorkingV,
                  eviction: .unbounded, device: device)
    }

    public init(
        nKVHeads: Int, headDim: Int, maxSeq: Int, dtype: DType,
        scheme: AURAScheme,
        rotation: Tensor,
        kCodebook: Tensor, kBoundaries: Tensor,
        vCodebook: Tensor, vBoundaries: Tensor,
        sharedWorkingK: Tensor, sharedWorkingV: Tensor,
        eviction: KVEviction,
        device: Device = .shared
    ) {
        precondition(rotation.shape == [headDim, headDim],
                     "AURAQuantizedKVCache: rotation must be [headDim, headDim]")
        precondition(rotation.dtype == .f32,
                     "AURAQuantizedKVCache: rotation must be f32")
        precondition(kCodebook.dtype == .f32 && kBoundaries.dtype == .f32,
                     "AURAQuantizedKVCache: K codebook/boundaries must be f32")
        precondition(vCodebook.dtype == .f32 && vBoundaries.dtype == .f32,
                     "AURAQuantizedKVCache: V codebook/boundaries must be f32")
        precondition(sharedWorkingK.shape == [nKVHeads, maxSeq, headDim],
                     "AURAQuantizedKVCache: sharedWorkingK shape mismatch")
        precondition(sharedWorkingV.shape == [nKVHeads, maxSeq, headDim],
                     "AURAQuantizedKVCache: sharedWorkingV shape mismatch")
        precondition(sharedWorkingK.dtype == dtype && sharedWorkingV.dtype == dtype,
                     "AURAQuantizedKVCache: sharedWorking dtype mismatch")

        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.scheme = scheme
        self.rotation = rotation
        self.kCodebook = kCodebook
        self.kBoundaries = kBoundaries
        self.vCodebook = vCodebook
        self.vBoundaries = vBoundaries
        self.sharedWorkingK = sharedWorkingK
        self.sharedWorkingV = sharedWorkingV

        self.kPackedWidth = AURACodebook.packedWidth(dim: headDim, bits: scheme.keyBits)
        self.vPackedWidth = AURACodebook.packedWidth(dim: headDim, bits: scheme.valueBits)

        self.kPacked = Tensor.empty(
            shape: [nKVHeads, maxSeq, kPackedWidth], dtype: .u32, device: device)
        self.vPacked = Tensor.empty(
            shape: [nKVHeads, maxSeq, vPackedWidth], dtype: .u32, device: device)
        self.kNorms = Tensor.empty(
            shape: [nKVHeads, maxSeq], dtype: .f32, device: device)
        self.vNorms = Tensor.empty(
            shape: [nKVHeads, maxSeq], dtype: .f32, device: device)

        // Codec is purely additive in atomic_or terms, so packed slots
        // MUST start zeroed. Norms slots get overwritten per encode but
        // zero is a safe default.
        kPacked.zero(); vPacked.zero()
        kNorms.zero();  vNorms.zero()

        self._evictionState = KVEvictionState(policy: eviction, bufferCapacity: maxSeq)
    }

    public func reset() { lengthLock.withLock { _evictionState.reset() } }

    /// Encode the current step's K + V into the compressed storage.
    /// `kFlat` and `vFlat` come in as [nKVHeads, headDim] in the model's
    /// dtype (bf16/f16/f32) — the encode kernel is dtype-generic, so no
    /// upcast is needed.
    public func appendOnGPU(kFlat: Tensor, vFlat: Tensor,
                            on cmd: MTLCommandBuffer) {
        precondition(kFlat.dtype == dtype && vFlat.dtype == dtype,
                     "AURAQuantizedKVCache: kFlat/vFlat dtype must match cache dtype")
        precondition(kFlat.elementCount == nKVHeads * headDim,
                     "AURAQuantizedKVCache: kFlat shape mismatch")
        precondition(vFlat.elementCount == nKVHeads * headDim,
                     "AURAQuantizedKVCache: vFlat shape mismatch")

        lengthLock.withLock {
            let pos = _evictionState.reserveNextSlot()
            // AURA encode atomic_or-accumulates into `packed[pos]`,
            // so on a rotated slot we MUST zero the prior contents
            // before the encode runs, or stale bits will OR through.
            // Cheap (one packed_width × u32 row per head per cache).
            if case .window = _evictionState.policy,
               _evictionState.absolutePosition > _evictionState.length {
                zeroPackedSlot(packed: kPacked, packedWidth: kPackedWidth, pos: pos, on: cmd)
                zeroPackedSlot(packed: vPacked, packedWidth: vPackedWidth, pos: pos, on: cmd)
            }
            encodePerHead(
                inputFlat: kFlat, packed: kPacked, norms: kNorms,
                codebook: kCodebook, boundaries: kBoundaries,
                packedWidth: kPackedWidth, bits: scheme.keyBits,
                pos: pos, on: cmd)
            encodePerHead(
                inputFlat: vFlat, packed: vPacked, norms: vNorms,
                codebook: vCodebook, boundaries: vBoundaries,
                packedWidth: vPackedWidth, bits: scheme.valueBits,
                pos: pos, on: cmd)
        }
    }

    /// Clear the packed-u32 row at `pos` across all heads, in
    /// preparation for the atomic_or-accumulating encode kernel.
    /// Called only when rotating into a previously-occupied slot.
    private func zeroPackedSlot(
        packed: Tensor, packedWidth: Int, pos: Int,
        on cmd: MTLCommandBuffer
    ) {
        let packedBytesPerSlot = packedWidth * 4
        let packedBytesPerHead = maxSeq * packedBytesPerSlot
        guard let blit = cmd.makeBlitCommandEncoder() else { return }
        for h in 0..<nKVHeads {
            let off = packed.offset + h * packedBytesPerHead + pos * packedBytesPerSlot
            blit.fill(buffer: packed.buffer,
                      range: off..<(off + packedBytesPerSlot),
                      value: 0)
        }
        blit.endEncoding()
    }

    /// Bulk-dequant the entire filled prefix of K and V into the shared
    /// working buffers. The output lives in **rotated** codec space; with
    /// identity rotation that's the same as the original K/V space, so
    /// downstream SDPA on the working buffers is correct. The SRHT path
    /// would inject an inverse rotation here (or fold it into W_o
    /// offline).
    public func prepareForAttention(on cmd: MTLCommandBuffer) -> (k: Tensor, v: Tensor) {
        guard length > 0 else { return (sharedWorkingK, sharedWorkingV) }
        Ops.auraDequantRotated(
            packed: kPacked, norms: kNorms, codebook: kCodebook,
            into: sharedWorkingK,
            nKVHeads: nKVHeads, dim: headDim, packedWidth: kPackedWidth,
            tokens: length, bits: scheme.keyBits, on: cmd)
        Ops.auraDequantRotated(
            packed: vPacked, norms: vNorms, codebook: vCodebook,
            into: sharedWorkingV,
            nKVHeads: nKVHeads, dim: headDim, packedWidth: vPackedWidth,
            tokens: length, bits: scheme.valueBits, on: cmd)
        return (sharedWorkingK, sharedWorkingV)
    }

    // MARK: - bytesAllocated / bytesInUse

    public var bytesAllocated: Int {
        // packed buffers + norms buffers (per cache, not counting the
        // shared working buffer or the shared codec tensors — those are
        // accounted once at the caller).
        let packedBytes = nKVHeads * maxSeq * (kPackedWidth + vPackedWidth) * 4
        let normBytes = nKVHeads * maxSeq * 2 * 4  // f32
        return packedBytes + normBytes
    }

    public var bytesInUse: Int {
        let filled = length
        let packedBytes = nKVHeads * filled * (kPackedWidth + vPackedWidth) * 4
        let normBytes = nKVHeads * filled * 2 * 4
        return packedBytes + normBytes
    }

    // MARK: - Internals

    /// Issue one `Ops.auraEncode` dispatch per head, with per-head
    /// buffer views, so each head's encoded output lands at its
    /// non-contiguous slot in the [nKVHeads, maxSeq, packedWidth]
    /// storage layout. See class header for the perf-followup note.
    private func encodePerHead(
        inputFlat: Tensor, packed: Tensor, norms: Tensor,
        codebook: Tensor, boundaries: Tensor,
        packedWidth: Int, bits: Int,
        pos: Int,
        on cmd: MTLCommandBuffer
    ) {
        let inputBytesPerHead = headDim * dtype.byteSize
        let packedBytesPerSlot = packedWidth * 4         // u32
        let packedBytesPerHead = maxSeq * packedBytesPerSlot
        let normBytesPerHead = maxSeq * 4                // f32

        for h in 0..<nKVHeads {
            let inputView = Tensor(
                buffer: inputFlat.buffer,
                offset: inputFlat.offset + h * inputBytesPerHead,
                shape: [1, headDim], dtype: dtype)
            let packedView = Tensor(
                buffer: packed.buffer,
                offset: packed.offset + h * packedBytesPerHead + pos * packedBytesPerSlot,
                shape: [1, packedWidth], dtype: .u32)
            let normsView = Tensor(
                buffer: norms.buffer,
                offset: norms.offset + h * normBytesPerHead + pos * 4,
                shape: [1], dtype: .f32)
            Ops.auraEncode(
                input: inputView, rotation: rotation,
                boundaries: boundaries, codebook: codebook,
                packedOut: packedView, normsOut: normsView,
                rows: 1, dim: headDim, packedWidth: packedWidth, bits: bits,
                on: cmd)
        }
    }
}
