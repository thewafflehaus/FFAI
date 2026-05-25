// `Ops.kvCacheUpdateKVMany` — batched K+V append in one shared encoder.
// Verifies the batched dispatch matches the per-row `kvCacheUpdateKV`
// loop at Qwen3.5/3.6 attention shapes.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.kvCacheUpdateKVMany — batched K+V append")
struct KvCacheUpdateManyTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("bf16 Qwen3.6 K+V shape (T=8, nKV=8, headDim=256) matches per-row")
    func bf16Qwen36() {
        runCase(t: 8, nKVHeads: 8, headDim: 256, maxSeq: 1024,
                dtype: .bf16)
    }

    @Test("f16 mid (T=16, nKV=4, headDim=128) matches per-row")
    func f16Mid() {
        runCase(t: 16, nKVHeads: 4, headDim: 128, maxSeq: 512,
                dtype: .f16)
    }

    @Test("f32 small (T=4, nKV=2, headDim=64) matches per-row")
    func f32Small() {
        runCase(t: 4, nKVHeads: 2, headDim: 64, maxSeq: 256,
                dtype: .f32)
    }

    private func runCase(t: Int, nKVHeads: Int, headDim: Int, maxSeq: Int,
                          dtype: DType) {
        let rowSize = nKVHeads * headDim
        var seed: UInt64 = 0xA1B2_C3D4
        @inline(__always) func xs() -> UInt32 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return UInt32(truncatingIfNeeded: seed)
        }
        @inline(__always) func rsmall() -> Float {
            Float(Int32(truncatingIfNeeded: xs())) / Float(Int32.max) * 0.5
        }
        let kSrcF = (0..<(t * rowSize)).map { _ in rsmall() }
        let vSrcF = (0..<(t * rowSize)).map { _ in rsmall() }

        let basePos = 5
        var positions = [UInt32](); positions.reserveCapacity(t)
        for r in 0..<t { positions.append(UInt32(basePos + r)) }

        let cacheKRef = Tensor.empty(shape: [nKVHeads, maxSeq, headDim], dtype: dtype)
        let cacheVRef = Tensor.empty(shape: [nKVHeads, maxSeq, headDim], dtype: dtype)
        Self.zero(cacheKRef); Self.zero(cacheVRef)
        let cmdRef = Device.shared.makeCommandBuffer()
        for r in 0..<t {
            let kSrc = Tensor.empty(shape: [rowSize], dtype: dtype)
            let vSrc = Tensor.empty(shape: [rowSize], dtype: dtype)
            Self.writeF32(kSrc, Array(kSrcF[(r * rowSize)..<((r + 1) * rowSize)]), dtype: dtype)
            Self.writeF32(vSrc, Array(vSrcF[(r * rowSize)..<((r + 1) * rowSize)]), dtype: dtype)
            Ops.kvCacheUpdateKV(
                kSrc: kSrc, kCache: cacheKRef,
                vSrc: vSrc, vCache: cacheVRef,
                nKVHeads: nKVHeads, headDim: headDim,
                maxSeq: maxSeq, position: basePos + r, on: cmdRef)
        }
        cmdRef.commit(); cmdRef.waitUntilCompleted()

        let kSrcAll = Tensor.empty(shape: [t * rowSize], dtype: dtype)
        let vSrcAll = Tensor.empty(shape: [t * rowSize], dtype: dtype)
        Self.writeF32(kSrcAll, kSrcF, dtype: dtype)
        Self.writeF32(vSrcAll, vSrcF, dtype: dtype)
        let cacheKBatch = Tensor.empty(shape: [nKVHeads, maxSeq, headDim], dtype: dtype)
        let cacheVBatch = Tensor.empty(shape: [nKVHeads, maxSeq, headDim], dtype: dtype)
        Self.zero(cacheKBatch); Self.zero(cacheVBatch)
        let positionsT = Tensor.empty(shape: [t], dtype: .u32)
        positionsT.copyIn(from: positions)
        let cmdBatch = Device.shared.makeCommandBuffer()
        Ops.kvCacheUpdateKVMany(
            kSrc: kSrcAll, kCache: cacheKBatch,
            vSrc: vSrcAll, vCache: cacheVBatch,
            positions: positionsT, t: t,
            nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
            on: cmdBatch)
        cmdBatch.commit(); cmdBatch.waitUntilCompleted()
        Self.flushQueue()

        let kRefArr = cacheKRef.toFloatArray()
        let kBatchArr = cacheKBatch.toFloatArray()
        let vRefArr = cacheVRef.toFloatArray()
        let vBatchArr = cacheVBatch.toFloatArray()
        #expect(kRefArr == kBatchArr,
                "K cache mismatch (dtype \(dtype) t=\(t))")
        #expect(vRefArr == vBatchArr,
                "V cache mismatch (dtype \(dtype) t=\(t))")
    }

    private static func writeF32(_ t: Tensor, _ src: [Float], dtype: DType) {
        switch dtype {
        case .f32: t.copyIn(from: src)
        case .f16: t.copyIn(from: src.map { Float16($0) })
        case .bf16: t.copyIn(from: src.map { UInt16($0.bitPattern >> 16) })
        default: preconditionFailure("unsupported dtype")
        }
    }

    private static func zero(_ t: Tensor) {
        let bytes = t.elementCount * t.dtype.byteSize
        memset(t.buffer.contents().advanced(by: t.offset), 0, bytes)
    }
}
