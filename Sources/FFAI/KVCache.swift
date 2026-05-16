// KVCache — one cache per attention layer. Pre-allocated to maxSeq
// capacity; `length` tracks how many positions are currently filled.
//
// Two implementations conform to `KVCacheProtocol`:
//   - `KVCache`                  : raw fp16/bf16 K/V (default, Phase 2)
//   - `AffineQuantizedKVCache`   : int8 group-quantized K/V (Phase 5c).
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

// MARK: - KVCacheProtocol

public protocol KVCacheProtocol: AnyObject, Sendable {
    var nKVHeads: Int { get }
    var headDim: Int { get }
    var maxSeq: Int { get }
    var dtype: DType { get }
    var length: Int { get }
    var bytesAllocated: Int { get }
    var bytesInUse: Int { get }

    /// Reset to length 0. Doesn't reclaim memory.
    func reset()

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
}

// MARK: - KVCache (raw fp16 / bf16)

public final class KVCache: KVCacheProtocol, @unchecked Sendable {
    public let nKVHeads: Int
    public let headDim: Int
    public let maxSeq: Int
    public let dtype: DType

    public let kBuffer: Tensor   // [nKVHeads, maxSeq, headDim]
    public let vBuffer: Tensor   // [nKVHeads, maxSeq, headDim]

    public private(set) var length: Int = 0

    public init(nKVHeads: Int, headDim: Int, maxSeq: Int, dtype: DType,
                device: Device = .shared) {
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.kBuffer = Tensor.empty(shape: [nKVHeads, maxSeq, headDim], dtype: dtype, device: device)
        self.vBuffer = Tensor.empty(shape: [nKVHeads, maxSeq, headDim], dtype: dtype, device: device)
        self.kBuffer.zero()
        self.vBuffer.zero()
    }

    /// CPU-side legacy append. Caller must have already sync'd the
    /// command buffer that produced kFlat / vFlat. Kept for tests +
    /// callers that don't have a live MTLCommandBuffer; the inference
    /// path uses `appendOnGPU` instead, which is sync-free.
    public func append(kFlat: Tensor, vFlat: Tensor) {
        precondition(length < maxSeq, "KVCache: capacity exhausted")
        precondition(kFlat.dtype == dtype && vFlat.dtype == dtype, "KVCache: dtype mismatch")
        let bytesPerHead = headDim * dtype.byteSize
        let kSrc = kFlat.buffer.contents().advanced(by: kFlat.offset)
        let vSrc = vFlat.buffer.contents().advanced(by: vFlat.offset)
        for h in 0..<nKVHeads {
            let dstHeadOffset = (h * maxSeq + length) * headDim * dtype.byteSize
            let kDst = kBuffer.buffer.contents().advanced(by: kBuffer.offset + dstHeadOffset)
            let vDst = vBuffer.buffer.contents().advanced(by: vBuffer.offset + dstHeadOffset)
            let srcOffset = h * bytesPerHead
            kDst.copyMemory(from: kSrc.advanced(by: srcOffset), byteCount: bytesPerHead)
            vDst.copyMemory(from: vSrc.advanced(by: srcOffset), byteCount: bytesPerHead)
        }
        length += 1
    }

    /// Append one timestep on the GPU via Ops.kvCacheUpdate. The dispatch
    /// is queued on `cmd`; no commit/wait happens here. Caller is
    /// responsible for ensuring the command buffer that produced kFlat /
    /// vFlat is the same one (or a strictly-prior one) so dependencies
    /// are honored. Bumps `length` immediately so subsequent SDPA
    /// dispatches see the updated count.
    public func appendOnGPU(kFlat: Tensor, vFlat: Tensor,
                            on cmd: MTLCommandBuffer) {
        precondition(length < maxSeq, "KVCache: capacity exhausted")
        precondition(kFlat.dtype == dtype && vFlat.dtype == dtype, "KVCache: dtype mismatch")
        Ops.kvCacheUpdate(src: kFlat, into: kBuffer,
                          nKVHeads: nKVHeads, headDim: headDim,
                          maxSeq: maxSeq, position: length, on: cmd)
        Ops.kvCacheUpdate(src: vFlat, into: vBuffer,
                          nKVHeads: nKVHeads, headDim: headDim,
                          maxSeq: maxSeq, position: length, on: cmd)
        length += 1
    }

    public func reset() { length = 0 }

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

// MARK: - AffineQuantizedKVCache (Phase 5c)

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
    public let dtype: DType        // dtype of scales/biases + dequant output
    public let bits: Int           // 8 for now
    public let groupSize: Int

    // Compressed storage (int8 packed 4-per-uint32)
    public let kWeights: Tensor    // [nKVHeads, maxSeq, headDim / 4] u32
    public let vWeights: Tensor
    public let kScales: Tensor     // [nKVHeads, maxSeq, headDim / groupSize] T
    public let vScales: Tensor
    public let kBiases: Tensor
    public let vBiases: Tensor

    // Shared working buffers (passed in at construction by the
    // family's makeKVCache; reused across every layer's cache).
    public let sharedWorkingK: Tensor   // [nKVHeads, maxSeq, headDim] T
    public let sharedWorkingV: Tensor

    public private(set) var length: Int = 0

    public init(nKVHeads: Int, headDim: Int, maxSeq: Int, dtype: DType,
                bits: Int, groupSize: Int,
                sharedWorkingK: Tensor, sharedWorkingV: Tensor,
                device: Device = .shared) {
        precondition(bits == 4 || bits == 8,
                     "AffineQuantizedKVCache: bits must be 4 or 8 today (int6 is a Phase 5c follow-up)")
        let valuesPerWord = 32 / bits   // int8 → 4, int4 → 8
        precondition(headDim % valuesPerWord == 0,
                     "headDim must be divisible by \(valuesPerWord) for int\(bits) packing")
        precondition(headDim % groupSize == 0, "headDim must be divisible by groupSize")
        precondition(sharedWorkingK.shape == [nKVHeads, maxSeq, headDim],
                     "sharedWorkingK shape mismatch")
        precondition(sharedWorkingV.shape == [nKVHeads, maxSeq, headDim],
                     "sharedWorkingV shape mismatch")
        precondition(sharedWorkingK.dtype == dtype && sharedWorkingV.dtype == dtype,
                     "sharedWorking dtype mismatch")

        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.bits = bits
        self.groupSize = groupSize

        let packsPerRow = headDim / valuesPerWord
        let groupsPerRow = headDim / groupSize
        self.kWeights = Tensor.empty(shape: [nKVHeads, maxSeq, packsPerRow], dtype: .u32, device: device)
        self.vWeights = Tensor.empty(shape: [nKVHeads, maxSeq, packsPerRow], dtype: .u32, device: device)
        self.kScales = Tensor.empty(shape: [nKVHeads, maxSeq, groupsPerRow], dtype: dtype, device: device)
        self.vScales = Tensor.empty(shape: [nKVHeads, maxSeq, groupsPerRow], dtype: dtype, device: device)
        self.kBiases = Tensor.empty(shape: [nKVHeads, maxSeq, groupsPerRow], dtype: dtype, device: device)
        self.vBiases = Tensor.empty(shape: [nKVHeads, maxSeq, groupsPerRow], dtype: dtype, device: device)

        self.sharedWorkingK = sharedWorkingK
        self.sharedWorkingV = sharedWorkingV

        kWeights.zero(); vWeights.zero()
        kScales.zero(); vScales.zero()
        kBiases.zero(); vBiases.zero()
    }

    public func reset() { length = 0 }

    public func appendOnGPU(kFlat: Tensor, vFlat: Tensor,
                            on cmd: MTLCommandBuffer) {
        precondition(length < maxSeq, "AffineQuantizedKVCache: capacity exhausted")
        precondition(kFlat.dtype == dtype && vFlat.dtype == dtype,
                     "AffineQuantizedKVCache: dtype mismatch")
        Ops.quantizeKVAffine(src: kFlat,
                             weights: kWeights, scales: kScales, biases: kBiases,
                             nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                             groupSize: groupSize, position: length, bits: bits, on: cmd)
        Ops.quantizeKVAffine(src: vFlat,
                             weights: vWeights, scales: vScales, biases: vBiases,
                             nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                             groupSize: groupSize, position: length, bits: bits, on: cmd)
        length += 1
    }

    public func prepareForAttention(on cmd: MTLCommandBuffer) -> (k: Tensor, v: Tensor) {
        // Bulk-dequant into the shared working buffers. SDPA reads
        // from the working buffers; the cache's compressed storage
        // is the persistent state.
        guard length > 0 else { return (sharedWorkingK, sharedWorkingV) }
        Ops.bulkDequantKVAffine(weights: kWeights, scales: kScales, biases: kBiases,
                                into: sharedWorkingK,
                                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                                groupSize: groupSize, nPositions: length, bits: bits, on: cmd)
        Ops.bulkDequantKVAffine(weights: vWeights, scales: vScales, biases: vBiases,
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
        return 2 * (
            nKVHeads * maxSeq * packs * 4
            + 2 * nKVHeads * maxSeq * groups * dtype.byteSize
        )
    }

    public var bytesInUse: Int {
        let valuesPerWord = 32 / bits
        let packs = headDim / valuesPerWord
        let groups = headDim / groupSize
        return 2 * (
            nKVHeads * length * packs * 4
            + 2 * nKVHeads * length * groups * dtype.byteSize
        )
    }
}

// MARK: - Array helpers

public extension Array where Element == any KVCacheProtocol {
    /// Sum of `bytesAllocated` across all per-layer caches.
    var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
    /// Sum of `bytesInUse` across all per-layer caches.
    var totalBytesInUse: Int { reduce(0) { $0 + $1.bytesInUse } }
}

public extension Array where Element == KVCache {
    /// Sum of `bytesAllocated` across all per-layer caches.
    var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
    /// Sum of `bytesInUse` across all per-layer caches.
    var totalBytesInUse: Int { reduce(0) { $0 + $1.bytesInUse } }
}
