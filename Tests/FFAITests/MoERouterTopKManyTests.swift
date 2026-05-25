// `Ops.moeRouterTopKMany` — T-batched MoE router. Verifies that
// dispatching the kernel with T rows produces per-row results
// identical to running `Ops.moeRouterTopK` once per row.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.moeRouterTopKMany — T-batched MoE router")
struct MoERouterTopKManyTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("f32 T=4 nExperts=128 k=8: many matches per-row reference")
    func f32Production() {
        runCase(t: 4, nExperts: 128, k: 8, dtype: .f32, tolerance: 1e-5)
    }

    @Test("bf16 T=8 nExperts=128 k=8: many matches per-row reference")
    func bf16Production() {
        runCase(t: 8, nExperts: 128, k: 8, dtype: .bf16, tolerance: 1e-2)
    }

    @Test("f16 T=2 nExperts=64 k=4: many matches per-row reference")
    func f16Small() {
        runCase(t: 2, nExperts: 64, k: 4, dtype: .f16, tolerance: 1e-3)
    }

    private func runCase(t: Int, nExperts: Int, k: Int, dtype: DType,
                          tolerance: Float) {
        var seed: UInt64 = 0xCAFE_BABE
        func rand() -> Float {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return Float(Int32(truncatingIfNeeded: seed)) / Float(Int32.max) * 4.0
        }
        let logits = Tensor.empty(shape: [t, nExperts], dtype: dtype)
        let logitsFlat: [Float] = (0..<(t * nExperts)).map { _ in rand() }
        Self.writeF32(logits, logitsFlat, dtype: dtype)

        // Reference: T separate `moeRouterTopK` calls (each on its own
        // [nExperts] / [k] row slice).
        let refIndices = Tensor.empty(shape: [t, k], dtype: .u32)
        let refWeights = Tensor.empty(shape: [t, k], dtype: dtype)
        let cmdRef = Device.shared.makeCommandBuffer()
        let dtBytes = dtype.byteSize
        for r in 0..<t {
            let lRow = Tensor(buffer: logits.buffer,
                              offset: logits.offset + r * nExperts * dtBytes,
                              shape: [nExperts], dtype: dtype)
            let iRow = Tensor(buffer: refIndices.buffer,
                              offset: refIndices.offset + r * k * 4,
                              shape: [k], dtype: .u32)
            let wRow = Tensor(buffer: refWeights.buffer,
                              offset: refWeights.offset + r * k * dtBytes,
                              shape: [k], dtype: dtype)
            Ops.moeRouterTopK(
                logits: lRow, indicesOut: iRow, weightsOut: wRow,
                nExperts: nExperts, k: k, normTopkProb: true, on: cmdRef)
        }
        cmdRef.commit(); cmdRef.waitUntilCompleted()

        // T-batched: ONE dispatch.
        let manyIndices = Tensor.empty(shape: [t, k], dtype: .u32)
        let manyWeights = Tensor.empty(shape: [t, k], dtype: dtype)
        let cmdMany = Device.shared.makeCommandBuffer()
        Ops.moeRouterTopKMany(
            logits: logits, indicesOut: manyIndices, weightsOut: manyWeights,
            t: t, nExperts: nExperts, k: k, normTopkProb: true, on: cmdMany)
        cmdMany.commit(); cmdMany.waitUntilCompleted()
        Self.flushQueue()

        // Index match must be exact.
        let refIdxArr: [UInt32] = refIndices.toArray(as: UInt32.self)
        let manyIdxArr: [UInt32] = manyIndices.toArray(as: UInt32.self)
        #expect(refIdxArr == manyIdxArr,
                 "T=\(t) k=\(k): topK indices differ between many and reference")

        // Weights match within dtype tolerance.
        let refWArr = refWeights.toFloatArray()
        let manyWArr = manyWeights.toFloatArray()
        var maxDiff: Float = 0
        for i in 0..<refWArr.count {
            let d = abs(refWArr[i] - manyWArr[i]); if d > maxDiff { maxDiff = d }
        }
        if maxDiff >= tolerance {
            print("[\(dtype) T=\(t)] weights maxDiff=\(maxDiff)")
        }
        #expect(maxDiff < tolerance)
    }

    private static func writeF32(_ t: Tensor, _ src: [Float], dtype: DType) {
        switch dtype {
        case .f32: t.copyIn(from: src)
        case .f16: t.copyIn(from: src.map { Float16($0) })
        case .bf16:
            t.copyIn(from: src.map { UInt16($0.bitPattern >> 16) })
        default: preconditionFailure("unsupported dtype")
        }
    }
}
