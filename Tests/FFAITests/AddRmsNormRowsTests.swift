// `Ops.addRmsNormRows` — multi-row variant of the fused residual+norm
// kernel. Verifies the fused 2-output dispatch matches the separate
// `Ops.add + Ops.rmsNormRows` reference chain it replaces in the
// `decodeMany` prefill paths (ITER 73).

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.addRmsNormRows — multi-row fused residual+RMSNorm")
struct AddRmsNormRowsTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("f32 T=4 hidden=128: fused matches `Ops.add + Ops.rmsNormRows`")
    func f32MultiRow() {
        runCase(t: 4, hidden: 128, dtype: .f32, tolerance: 1e-4)
    }

    @Test("bf16 T=8 hidden=256: fused matches reference within bf16 tolerance")
    func bf16MultiRow() {
        runCase(t: 8, hidden: 256, dtype: .bf16, tolerance: 5e-2)
    }

    @Test("f16 T=2 hidden=512: fused matches reference within f16 tolerance")
    func f16MultiRow() {
        runCase(t: 2, hidden: 512, dtype: .f16, tolerance: 5e-3)
    }

    private func runCase(t: Int, hidden: Int, dtype: DType, tolerance: Float) {
        let total = t * hidden
        let a = Tensor.empty(shape: [total], dtype: dtype)
        let b = Tensor.empty(shape: [total], dtype: dtype)
        let weight = Tensor.empty(shape: [hidden], dtype: dtype)
        var seed: UInt64 = 0xFEED_BEEF
        func rand() -> Float {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return Float(Int32(truncatingIfNeeded: seed)) / Float(Int32.max) * 0.3
        }
        Self.writeF32(a, (0..<total).map { _ in rand() }, dtype: dtype)
        Self.writeF32(b, (0..<total).map { _ in rand() }, dtype: dtype)
        // Norm weight near 1.0 so the output magnitude stays sensible.
        Self.writeF32(weight, (0..<hidden).map { _ in 1.0 + rand() * 0.1 },
                      dtype: dtype)

        // Reference: separate Ops.add + Ops.rmsNormRows.
        let cmdRef = Device.shared.makeCommandBuffer()
        let refSum = Ops.add(a, b, on: cmdRef)
        let refNorm = Ops.rmsNormRows(
            refSum, weight: weight, eps: 1e-5,
            nRows: t, rowSize: hidden, on: cmdRef)
        cmdRef.commit()
        cmdRef.waitUntilCompleted()

        // Fused: one dispatch produces both outputs.
        let fusedSum = Tensor.empty(shape: [total], dtype: dtype)
        let fusedNorm = Tensor.empty(shape: [total], dtype: dtype)
        let cmdFused = Device.shared.makeCommandBuffer()
        Ops.addRmsNormRows(
            a: a, b: b, weight: weight, eps: 1e-5,
            nRows: t, rowSize: hidden,
            residualOut: fusedSum, normedOut: fusedNorm, on: cmdFused)
        cmdFused.commit()
        cmdFused.waitUntilCompleted()
        Self.flushQueue()

        let refSumArr = refSum.toFloatArray()
        let fusedSumArr = fusedSum.toFloatArray()
        let refNormArr = refNorm.toFloatArray()
        let fusedNormArr = fusedNorm.toFloatArray()
        var maxSumDiff: Float = 0
        var maxNormDiff: Float = 0
        for i in 0..<total {
            let ds = abs(refSumArr[i] - fusedSumArr[i])
            if ds > maxSumDiff { maxSumDiff = ds }
            let dn = abs(refNormArr[i] - fusedNormArr[i])
            if dn > maxNormDiff { maxNormDiff = dn }
        }
        #expect(maxSumDiff < tolerance)
        #expect(maxNormDiff < tolerance)
        if maxSumDiff >= tolerance || maxNormDiff >= tolerance {
            print("[\(dtype) T=\(t) hidden=\(hidden)] sumDiff=\(maxSumDiff) normDiff=\(maxNormDiff)")
        }
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
