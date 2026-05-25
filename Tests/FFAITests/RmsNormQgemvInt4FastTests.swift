// `Ops.rmsNormQgemvInt4Fast` — fused RMSNorm + int4 dequant-GEMV.
// Verifies the fused kernel matches the separate `Ops.rmsNorm +
// Ops.dequantGemvInt4` reference chain it will replace at finalNorm
// + lmHead and similar pre-norm + single-projection sites.
//
// Kernel constraints (per `ffai_rms_norm_qgemv_fast`):
//   in_dim multiple of 512, out_dim multiple of 8, group_size = 64.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.rmsNormQgemvInt4Fast — fused RMSNorm + int4 GEMV")
struct RmsNormQgemvInt4FastTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    /// Tolerance is RELATIVE to the reference's max abs value, since the
    /// fused kernel skips the intermediate bf16/f16 round-trip on the
    /// normed activation that the reference chain does. Absolute diff
    /// scales with output magnitude; relative captures the math-
    /// equivalence claim independently of accumulator range.
    ///
    /// Smallest constraint-satisfying shape: in_dim=512, out_dim=8.
    @Test("f32 in_dim=512 out_dim=8: fused matches separate chain (rel<2%)")
    func f32Smallest() {
        runCase(inDim: 512, outDim: 8, dtype: .f32, relativeTolerance: 2e-2)
    }

    /// Qwen3.6-A3B finalNorm+lmHead-like shape: in_dim=2048, out_dim=128.
    @Test("bf16 in_dim=2048 out_dim=128: fused matches separate chain (rel<5%)")
    func bf16Production() {
        runCase(inDim: 2048, outDim: 128, dtype: .bf16, relativeTolerance: 5e-2)
    }

    @Test("f16 in_dim=1024 out_dim=64: fused matches separate chain (rel<3%)")
    func f16Mid() {
        runCase(inDim: 1024, outDim: 64, dtype: .f16, relativeTolerance: 3e-2)
    }

    private func runCase(inDim: Int, outDim: Int, dtype: DType,
                          relativeTolerance: Float) {
        let groupSize = 64
        let nGroups = inDim / groupSize
        let packedPerRow = inDim / 8
        // Random data — deterministic xorshift.
        var seed: UInt64 = 0xFEED_C0FFEE
        @inline(__always)
        func xs() -> UInt32 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return UInt32(truncatingIfNeeded: seed)
        }
        @inline(__always)
        func rsmall() -> Float {
            Float(Int32(truncatingIfNeeded: xs())) / Float(Int32.max) * 0.2
        }

        // Pack random u32 weights (any bit pattern is a valid int4-packed weight).
        let qWeight = Tensor.empty(shape: [outDim, packedPerRow], dtype: .u32)
        var wBytes = [UInt32](); wBytes.reserveCapacity(outDim * packedPerRow)
        for _ in 0..<(outDim * packedPerRow) { wBytes.append(xs()) }
        qWeight.copyIn(from: wBytes)

        // Scales / biases — small magnitude to keep activations bounded.
        let qScales = Tensor.empty(shape: [outDim, nGroups], dtype: dtype)
        let qBiases = Tensor.empty(shape: [outDim, nGroups], dtype: dtype)
        Self.writeF32(qScales, (0..<(outDim * nGroups)).map { _ in rsmall() + 0.05 }, dtype: dtype)
        Self.writeF32(qBiases, (0..<(outDim * nGroups)).map { _ in rsmall() }, dtype: dtype)

        // x and norm_weight.
        let x = Tensor.empty(shape: [inDim], dtype: dtype)
        let normWeight = Tensor.empty(shape: [inDim], dtype: dtype)
        Self.writeF32(x, (0..<inDim).map { _ in rsmall() }, dtype: dtype)
        Self.writeF32(normWeight, (0..<inDim).map { _ in 1.0 + rsmall() * 0.05 },
                      dtype: dtype)

        let eps: Float = 1e-5

        // Reference: separate Ops.rmsNorm + Ops.dequantGemvInt4.
        let cmdRef = Device.shared.makeCommandBuffer()
        let normed = Ops.rmsNorm(x, weight: normWeight, eps: eps, on: cmdRef)
        let refOut = Ops.dequantGemvInt4(
            weight: qWeight, scales: qScales, biases: qBiases,
            input: normed, groupSize: groupSize, on: cmdRef)
        cmdRef.commit()
        cmdRef.waitUntilCompleted()

        // Fused: one dispatch.
        let fusedOut = Tensor.empty(shape: [outDim], dtype: dtype)
        let cmdFused = Device.shared.makeCommandBuffer()
        Ops.rmsNormQgemvInt4Fast(
            x: x, normWeight: normWeight, eps: eps,
            qWeight: qWeight, qScales: qScales, qBiases: qBiases,
            on: cmdFused, into: fusedOut)
        cmdFused.commit()
        cmdFused.waitUntilCompleted()
        Self.flushQueue()

        let refArr = refOut.toFloatArray()
        let fusedArr = fusedOut.toFloatArray()
        var maxDiff: Float = 0
        var maxAbsRef: Float = 0
        for i in 0..<outDim {
            let d = abs(refArr[i] - fusedArr[i])
            if d > maxDiff { maxDiff = d }
            let a = abs(refArr[i])
            if a > maxAbsRef { maxAbsRef = a }
        }
        // Guard against tiny outputs causing meaningless ratios.
        let denom = max(maxAbsRef, 1.0)
        let relativeDiff = maxDiff / denom
        #expect(relativeDiff < relativeTolerance)
        if relativeDiff >= relativeTolerance {
            print("[\(dtype) in=\(inDim) out=\(outDim)] maxDiff=\(maxDiff) " +
                  "maxAbsRef=\(maxAbsRef) rel=\(relativeDiff)")
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
