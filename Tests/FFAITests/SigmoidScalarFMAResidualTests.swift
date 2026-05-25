// `Ops.sigmoidScalarFMAResidual` correctness — verifies the fused
// 4-input form matches the separate `sigmoidScalarFMA + Ops.add`
// chain it replaces. Wired into `qwen35ApplyFFN .moe` by ITER 66
// to collapse the post-MoE-FFN residual add into one dispatch.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.sigmoidScalarFMAResidual — fused vs reference")
struct SigmoidScalarFMAResidualTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    /// Compare fused (one dispatch) vs reference (sigmoidScalarFMA +
    /// Ops.add) over a Qwen3.6-A3B-like shape (hidden=2048, bf16).
    @Test("bf16 hidden=2048: fused matches reference within bf16 tolerance")
    func bf16Production() {
        runCase(hidden: 2048, dtype: .bf16, tolerance: 5e-2)
    }

    @Test("f16 hidden=2048: fused matches reference within f16 tolerance")
    func f16Production() {
        runCase(hidden: 2048, dtype: .f16, tolerance: 5e-3)
    }

    @Test("f32 hidden=128: fused matches reference within f32 tolerance")
    func f32Small() {
        runCase(hidden: 128, dtype: .f32, tolerance: 1e-5)
    }

    private func runCase(hidden: Int, dtype: DType, tolerance: Float) {
        // Synth inputs.
        let gate = Tensor.empty(shape: [1], dtype: dtype)
        let value = Tensor.empty(shape: [hidden], dtype: dtype)
        let base = Tensor.empty(shape: [hidden], dtype: dtype)
        let residual = Tensor.empty(shape: [hidden], dtype: dtype)
        Self.writeF32(gate, [0.3], dtype: dtype)
        let valuesF32 = (0..<hidden).map { Float(($0 % 11) - 5) * 0.03 }
        let basesF32 = (0..<hidden).map { Float(($0 % 7) - 3) * 0.05 + 0.1 }
        let residualsF32 = (0..<hidden).map { Float(($0 % 13) - 6) * 0.04 - 0.2 }
        Self.writeF32(value, valuesF32, dtype: dtype)
        Self.writeF32(base, basesF32, dtype: dtype)
        Self.writeF32(residual, residualsF32, dtype: dtype)

        // Reference: sigmoidScalarFMA then Ops.add.
        let refIntermediate = Tensor.empty(shape: [hidden], dtype: dtype)
        let refOut = Tensor.empty(shape: [hidden], dtype: dtype)
        let cmdRef = Device.shared.makeCommandBuffer()
        Ops.sigmoidScalarFMA(gate: gate, value: value, base: base,
                              into: refIntermediate, on: cmdRef)
        _ = Ops.add(residual, refIntermediate, on: cmdRef, into: refOut)
        cmdRef.commit()
        cmdRef.waitUntilCompleted()

        // Fused: one dispatch.
        let fusedOut = Tensor.empty(shape: [hidden], dtype: dtype)
        let cmdFused = Device.shared.makeCommandBuffer()
        Ops.sigmoidScalarFMAResidual(gate: gate, value: value, base: base,
                                       residual: residual,
                                       into: fusedOut, on: cmdFused)
        cmdFused.commit()
        cmdFused.waitUntilCompleted()
        Self.flushQueue()

        let refArr = refOut.toFloatArray()
        let fusedArr = fusedOut.toFloatArray()
        var maxDiff: Float = 0
        for i in 0..<hidden {
            let d = abs(refArr[i] - fusedArr[i])
            if d > maxDiff { maxDiff = d }
        }
        #expect(maxDiff < tolerance)
        if maxDiff >= tolerance {
            print("[\(dtype) hidden=\(hidden)] maxDiff=\(maxDiff)")
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
