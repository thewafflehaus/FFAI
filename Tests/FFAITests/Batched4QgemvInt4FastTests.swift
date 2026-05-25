// `Ops.batched4QgemvInt4Fast` — fused 4-output int4 GEMV in one
// dispatch. Verifies the fused single-kernel result matches the
// `dequantGemvInt4Four` 4-dispatch shared-encoder reference at
// production-like Qwen3.5 GDN shapes.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.batched4QgemvInt4Fast — fused 4-output int4 GEMV")
struct Batched4QgemvInt4FastTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("bf16 GDN-shape hidden=2048 outs=(512,1024,64,64): fused matches dequantGemvInt4Four")
    func bf16Gdn() {
        runCase(inDim: 2048, outA: 512, outB: 1024, outC: 64, outD: 64,
                 dtype: .bf16, relativeTolerance: 5e-2)
    }

    @Test("f16 mid hidden=1024 outs=(256,512,32,32): fused matches reference")
    func f16Mid() {
        runCase(inDim: 1024, outA: 256, outB: 512, outC: 32, outD: 32,
                 dtype: .f16, relativeTolerance: 3e-2)
    }

    @Test("f32 small hidden=512 outs=(8,8,8,8): fused matches reference")
    func f32Smallest() {
        runCase(inDim: 512, outA: 8, outB: 8, outC: 8, outD: 8,
                 dtype: .f32, relativeTolerance: 2e-2)
    }

    private func runCase(inDim: Int,
                          outA: Int, outB: Int, outC: Int, outD: Int,
                          dtype: DType, relativeTolerance: Float) {
        let groupSize = 64
        let packedPerRow = inDim / 8
        let nGroups = inDim / groupSize
        var seed: UInt64 = 0xFACE_4ABC
        @inline(__always) func xs() -> UInt32 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return UInt32(truncatingIfNeeded: seed)
        }
        @inline(__always) func rsmall() -> Float {
            Float(Int32(truncatingIfNeeded: xs())) / Float(Int32.max) * 0.2
        }
        func makeWeights(_ outDim: Int) -> (Tensor, Tensor, Tensor) {
            let w = Tensor.empty(shape: [outDim, packedPerRow], dtype: .u32)
            var wBytes = [UInt32](); wBytes.reserveCapacity(outDim * packedPerRow)
            for _ in 0..<(outDim * packedPerRow) { wBytes.append(xs()) }
            w.copyIn(from: wBytes)
            let s = Tensor.empty(shape: [outDim, nGroups], dtype: dtype)
            Self.writeF32(s, (0..<(outDim * nGroups)).map { _ in rsmall() + 0.05 },
                          dtype: dtype)
            let b = Tensor.empty(shape: [outDim, nGroups], dtype: dtype)
            Self.writeF32(b, (0..<(outDim * nGroups)).map { _ in rsmall() },
                          dtype: dtype)
            return (w, s, b)
        }
        let (wA, sA, bA) = makeWeights(outA)
        let (wB, sB, bB) = makeWeights(outB)
        let (wC, sC, bC) = makeWeights(outC)
        let (wD, sD, bD) = makeWeights(outD)
        let x = Tensor.empty(shape: [inDim], dtype: dtype)
        Self.writeF32(x, (0..<inDim).map { _ in rsmall() }, dtype: dtype)

        // Reference: 4-dispatch shared-encoder path.
        let refA = Tensor.empty(shape: [outA], dtype: dtype)
        let refB = Tensor.empty(shape: [outB], dtype: dtype)
        let refC = Tensor.empty(shape: [outC], dtype: dtype)
        let refD = Tensor.empty(shape: [outD], dtype: dtype)
        let cmdRef = Device.shared.makeCommandBuffer()
        Ops.dequantGemvInt4Four(
            input: x,
            w0: wA, s0: sA, b0: bA, out0: refA,
            w1: wB, s1: sB, b1: bB, out1: refB,
            w2: wC, s2: sC, b2: bC, out2: refC,
            w3: wD, s3: sD, b3: bD, out3: refD,
            groupSize: groupSize, on: cmdRef)
        cmdRef.commit(); cmdRef.waitUntilCompleted()

        // Fused single dispatch into 4 separate output buffers.
        let fusedA = Tensor.empty(shape: [outA], dtype: dtype)
        let fusedB = Tensor.empty(shape: [outB], dtype: dtype)
        let fusedC = Tensor.empty(shape: [outC], dtype: dtype)
        let fusedD = Tensor.empty(shape: [outD], dtype: dtype)
        let cmdFused = Device.shared.makeCommandBuffer()
        Ops.batched4QgemvInt4Fast(
            input: x,
            wA: wA, scalesA: sA, biasesA: bA, outA: fusedA,
            wB: wB, scalesB: sB, biasesB: bB, outB: fusedB,
            wC: wC, scalesC: sC, biasesC: bC, outC: fusedC,
            wD: wD, scalesD: sD, biasesD: bD, outD: fusedD,
            groupSize: groupSize, on: cmdFused)
        cmdFused.commit(); cmdFused.waitUntilCompleted()
        Self.flushQueue()

        func relCheck(_ ref: Tensor, _ got: Tensor, _ label: String) -> Bool {
            let r = ref.toFloatArray()
            let g = got.toFloatArray()
            var maxDiff: Float = 0, maxAbs: Float = 0
            for i in 0..<r.count {
                let d = abs(r[i] - g[i]); if d > maxDiff { maxDiff = d }
                let a = abs(r[i]); if a > maxAbs { maxAbs = a }
            }
            let rel = maxDiff / max(maxAbs, 1.0)
            if rel >= relativeTolerance {
                print("[\(dtype) in=\(inDim) \(label)] maxDiff=\(maxDiff) " +
                      "maxAbs=\(maxAbs) rel=\(rel)")
            }
            return rel < relativeTolerance
        }
        #expect(relCheck(refA, fusedA, "A"))
        #expect(relCheck(refB, fusedB, "B"))
        #expect(relCheck(refC, fusedC, "C"))
        #expect(relCheck(refD, fusedD, "D"))
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
