// `Ops.batched4QmmFast` — M>1 (batched) 4-output int4 GEMM in one
// dispatch. Verifies against the M=1 oracle (`Ops.batched4QgemvInt4Fast`)
// looped over M.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.batched4QmmFast — M>1 4-output int4 GEMM")
struct Batched4QmmFastTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("bf16 GDN shape (M=2 in=2048 outs=(2048,2048,16,16))")
    func bf16Gdn() {
        runCase(m: 2, inDim: 2048, outA: 2048, outB: 2048, outC: 16, outD: 16,
                dtype: .bf16, tolerance: 1e-2)
    }

    @Test("f16 mid (M=8 in=1024 outs=(512,256,8,8))")
    func f16Mid() {
        runCase(m: 8, inDim: 1024, outA: 512, outB: 256, outC: 8, outD: 8,
                dtype: .f16, tolerance: 3e-3)
    }

    @Test("f32 small (M=2 in=512 outs=(8,8,8,8))")
    func f32Small() {
        runCase(m: 2, inDim: 512, outA: 8, outB: 8, outC: 8, outD: 8,
                dtype: .f32, tolerance: 1e-3)
    }

    private func runCase(m: Int, inDim: Int,
                          outA: Int, outB: Int, outC: Int, outD: Int,
                          dtype: DType, tolerance: Float) {
        let groupSize = 64
        let packedPerRow = inDim / 8
        let nGroups = inDim / groupSize
        var seed: UInt64 = 0xC0FFEE_BEEF
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
        let xAll = Tensor.empty(shape: [m * inDim], dtype: dtype)
        Self.writeF32(xAll, (0..<(m * inDim)).map { _ in rsmall() }, dtype: dtype)

        // ── Reference: M-loop of M=1 batched4QgemvInt4Fast ──
        let refA = Tensor.empty(shape: [m * outA], dtype: dtype)
        let refB = Tensor.empty(shape: [m * outB], dtype: dtype)
        let refC = Tensor.empty(shape: [m * outC], dtype: dtype)
        let refD = Tensor.empty(shape: [m * outD], dtype: dtype)
        let cmdRef = Device.shared.makeCommandBuffer()
        let dtBytes = dtype.byteSize
        for r in 0..<m {
            let xRow = Tensor(buffer: xAll.buffer,
                              offset: xAll.offset + r * inDim * dtBytes,
                              shape: [inDim], dtype: dtype)
            let aRow = Tensor(buffer: refA.buffer,
                              offset: refA.offset + r * outA * dtBytes,
                              shape: [outA], dtype: dtype)
            let bRow = Tensor(buffer: refB.buffer,
                              offset: refB.offset + r * outB * dtBytes,
                              shape: [outB], dtype: dtype)
            let cRow = Tensor(buffer: refC.buffer,
                              offset: refC.offset + r * outC * dtBytes,
                              shape: [outC], dtype: dtype)
            let dRow = Tensor(buffer: refD.buffer,
                              offset: refD.offset + r * outD * dtBytes,
                              shape: [outD], dtype: dtype)
            Ops.batched4QgemvInt4Fast(
                input: xRow,
                wA: wA, scalesA: sA, biasesA: bA, outA: aRow,
                wB: wB, scalesB: sB, biasesB: bB, outB: bRow,
                wC: wC, scalesC: sC, biasesC: bC, outC: cRow,
                wD: wD, scalesD: sD, biasesD: bD, outD: dRow,
                groupSize: groupSize, on: cmdRef)
        }
        cmdRef.commit(); cmdRef.waitUntilCompleted()

        // ── Fused: ONE dispatch ──
        let fusedA = Tensor.empty(shape: [m * outA], dtype: dtype)
        let fusedB = Tensor.empty(shape: [m * outB], dtype: dtype)
        let fusedC = Tensor.empty(shape: [m * outC], dtype: dtype)
        let fusedD = Tensor.empty(shape: [m * outD], dtype: dtype)
        let cmdFused = Device.shared.makeCommandBuffer()
        Ops.batched4QmmFast(
            input: xAll, m: m,
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
            if rel >= tolerance {
                print("[\(dtype) M=\(m) in=\(inDim) \(label)] maxDiff=\(maxDiff) " +
                      "maxAbs=\(maxAbs) rel=\(rel)")
            }
            return rel < tolerance
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
        case .bf16: t.copyIn(from: src.map { UInt16($0.bitPattern >> 16) })
        default: preconditionFailure("unsupported dtype")
        }
    }
}
