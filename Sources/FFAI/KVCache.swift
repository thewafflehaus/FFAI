// KVCache — one cache per attention layer. Pre-allocated to maxSeq
// capacity; `length` tracks how many positions are currently filled.
//
// Layout: K and V each as [maxSeq, nKVHeads * headDim] flat. Writing
// position p is a contiguous block at byte offset `p * rowBytes`.
//
// Reading: SDPA decode kernel takes K/V as [nKVHeads, nKV, headDim].
// We expose K/V tensors with shape [nKVHeads, length, headDim] but the
// underlying memory layout is [maxSeq, nKVHeads * headDim] — same data,
// different logical shape. The current SDPA kernel treats K/V as
// [nKVHeads, nKV, headDim] which means we need to transpose.
//
// SIMPLIFICATION: lay out the cache as [nKVHeads, maxSeq, headDim] so
// indexing matches what sdpa_decode_naive expects. Per-position write
// then becomes scattered (one slice per kv head). For Phase 2 we do the
// scatter on CPU after the K/V projection — slow but correct.
//
// Better: lay out K/V as [maxSeq, nKVHeads, headDim]. Per-position write
// is a contiguous block of size nKVHeads * headDim. SDPA kernel reads
// scattered. We'll go with this layout and update SDPA later if needed.
//
// FOR NOW: I'll use [nKVHeads, maxSeq, headDim] layout. This requires
// the K/V projection output to be re-strided into the cache. The simplest
// approach is to project into a [nKVHeads, headDim] temporary, then run
// a small kernel to write each head's row into the cache at the right
// position. That's a separate kernel.
//
// SIMPLEST SIMPLIFICATION: lay out as [maxSeq, nKVHeads, headDim] AND
// change the SDPA kernel to read this layout. Done in next iteration.

import Foundation
import Metal

public final class KVCache: @unchecked Sendable {
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

    /// Append one timestep of K and V. `kFlat` and `vFlat` are
    /// [nKVHeads, headDim] (flat row-major). For each head h, copy
    /// kFlat[h, :] into kBuffer[h, length, :].
    public func append(kFlat: Tensor, vFlat: Tensor) {
        precondition(length < maxSeq, "KVCache: capacity exhausted")
        precondition(kFlat.dtype == dtype && vFlat.dtype == dtype, "KVCache: dtype mismatch")
        let bytesPerHead = headDim * dtype.byteSize
        let kSrc = kFlat.buffer.contents().advanced(by: kFlat.offset)
        let vSrc = vFlat.buffer.contents().advanced(by: vFlat.offset)
        for h in 0..<nKVHeads {
            // dst offset within kBuffer: h * maxSeq * headDim + length * headDim
            let dstHeadOffset = (h * maxSeq + length) * headDim * dtype.byteSize
            let kDst = kBuffer.buffer.contents().advanced(by: kBuffer.offset + dstHeadOffset)
            let vDst = vBuffer.buffer.contents().advanced(by: vBuffer.offset + dstHeadOffset)
            let srcOffset = h * bytesPerHead
            kDst.copyMemory(from: kSrc.advanced(by: srcOffset), byteCount: bytesPerHead)
            vDst.copyMemory(from: vSrc.advanced(by: srcOffset), byteCount: bytesPerHead)
        }
        length += 1
    }

    public func reset() { length = 0 }
}
