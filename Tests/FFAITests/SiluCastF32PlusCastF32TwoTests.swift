// `Ops.siluCastF32PlusCastF32Two` — shared-encoder fused silu+cast +
// two plain casts. Verifies the 3-dispatch shared encoder matches the
// separate `siluCastToF32 + castToF32Two` reference chain across
// bf16/f16.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.siluCastF32PlusCastF32Two — shared-encoder triple cast")
struct SiluCastF32PlusCastF32TwoTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("bf16 — matches separate siluCast + castTwo")
    func bf16Match() { runCase(dtype: .bf16, n: 256, tolerance: 1e-3) }

    @Test("f16 — matches separate siluCast + castTwo")
    func f16Match() { runCase(dtype: .f16, n: 128, tolerance: 1e-3) }

    private func runCase(dtype: DType, n: Int, tolerance: Float) {
        var seed: UInt64 = 0xC1AD_E571
        func rand() -> Float {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return Float(Int32(truncatingIfNeeded: seed)) / Float(Int32.max) * 0.5
        }
        let siluIn = Tensor.empty(shape: [n], dtype: dtype)
        let a = Tensor.empty(shape: [n], dtype: dtype)
        let b = Tensor.empty(shape: [n], dtype: dtype)
        Self.writeF32(siluIn, (0..<n).map { _ in rand() }, dtype: dtype)
        Self.writeF32(a, (0..<n).map { _ in rand() }, dtype: dtype)
        Self.writeF32(b, (0..<n).map { _ in rand() }, dtype: dtype)

        // Reference.
        let refSilu = Tensor.empty(shape: [n], dtype: .f32)
        let refA = Tensor.empty(shape: [n], dtype: .f32)
        let refB = Tensor.empty(shape: [n], dtype: .f32)
        let cmdRef = Device.shared.makeCommandBuffer()
        Ops.siluCastToF32(siluIn, into: refSilu, on: cmdRef)
        Ops.castToF32Two(a, into: refA, b, into: refB, on: cmdRef)
        cmdRef.commit(); cmdRef.waitUntilCompleted()

        // Fused.
        let fusedSilu = Tensor.empty(shape: [n], dtype: .f32)
        let fusedA = Tensor.empty(shape: [n], dtype: .f32)
        let fusedB = Tensor.empty(shape: [n], dtype: .f32)
        let cmdFused = Device.shared.makeCommandBuffer()
        Ops.siluCastF32PlusCastF32Two(
            siluIn: siluIn, into: fusedSilu,
            a, into: fusedA,
            b, into: fusedB, on: cmdFused)
        cmdFused.commit(); cmdFused.waitUntilCompleted()
        Self.flushQueue()

        func cmp(_ ref: Tensor, _ got: Tensor, _ label: String) -> Bool {
            let r = ref.toFloatArray()
            let g = got.toFloatArray()
            var maxD: Float = 0
            for i in 0..<r.count { let d = abs(r[i] - g[i]); if d > maxD { maxD = d } }
            if maxD >= tolerance {
                print("[\(dtype) \(label)] maxDiff=\(maxD)")
            }
            return maxD < tolerance
        }
        #expect(cmp(refSilu, fusedSilu, "silu"))
        #expect(cmp(refA, fusedA, "a"))
        #expect(cmp(refB, fusedB, "b"))
    }

    private static func writeF32(_ t: Tensor, _ src: [Float], dtype: DType) {
        switch dtype {
        case .f16: t.copyIn(from: src.map { Float16($0) })
        case .bf16:
            t.copyIn(from: src.map { UInt16($0.bitPattern >> 16) })
        default: preconditionFailure("unsupported dtype")
        }
    }
}
