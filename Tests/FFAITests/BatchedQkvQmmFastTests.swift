// `Ops.batchedQkvQmmFast` — M>1 fused Q/K/V int4 qmm. Verifies that
// dispatching the new kernel against M rows of input matches running
// `Ops.batchedQkvQgemvInt4Fast` (the M=1 sibling) M times against the
// same per-row inputs/outputs.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.batchedQkvQmmFast — fused QKV int4 QMM (M>1)")
struct BatchedQkvQmmFastTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("bf16 M=4 in=512 q/k/v=256/64/64: M>1 matches the GEMV-M=1 oracle")
    func bf16M4() {
        runCase(m: 4, inDim: 512, outQ: 256, outK: 64, outV: 64,
                dtype: .bf16, relativeTolerance: 5e-2)
    }

    @Test("f16 M=8 in=1024 q/k/v=512/128/128: M>1 matches the GEMV-M=1 oracle")
    func f16M8() {
        runCase(m: 8, inDim: 1024, outQ: 512, outK: 128, outV: 128,
                dtype: .f16, relativeTolerance: 3e-2)
    }

    @Test("f32 M=2 in=512 q/k/v=8/8/8: M>1 matches the GEMV-M=1 oracle")
    func f32M2Tiny() {
        runCase(m: 2, inDim: 512, outQ: 8, outK: 8, outV: 8,
                dtype: .f32, relativeTolerance: 2e-2)
    }

    private func runCase(m: Int, inDim: Int, outQ: Int, outK: Int, outV: Int,
                          dtype: DType, relativeTolerance: Float) {
        let groupSize = 64
        let packedPerRow = inDim / 8
        let nGroups = inDim / groupSize
        var seed: UInt64 = 0xBEEF_FACE
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
        let (wQ, sQ, bQ) = makeWeights(outQ)
        let (wK, sK, bK) = makeWeights(outK)
        let (wV, sV, bV) = makeWeights(outV)
        let x = Tensor.empty(shape: [m, inDim], dtype: dtype)
        Self.writeF32(x, (0..<(m * inDim)).map { _ in rsmall() }, dtype: dtype)

        // Oracle: dispatch the M=1 fused kernel M times against per-row x slices.
        let dtBytes = dtype.byteSize
        let oracleOut = Tensor.empty(shape: [m, outQ + outK + outV], dtype: dtype)
        let cmdRef = Device.shared.makeCommandBuffer()
        for r in 0..<m {
            let xRow = Tensor(buffer: x.buffer,
                              offset: x.offset + r * inDim * dtBytes,
                              shape: [inDim], dtype: dtype)
            let outRow = Tensor(buffer: oracleOut.buffer,
                                offset: oracleOut.offset + r * (outQ + outK + outV) * dtBytes,
                                shape: [outQ + outK + outV], dtype: dtype)
            Ops.batchedQkvQgemvInt4Fast(
                x: xRow,
                wQ: wQ, scalesQ: sQ, biasesQ: bQ,
                wK: wK, scalesK: sK, biasesK: bK,
                wV: wV, scalesV: sV, biasesV: bV,
                outQ: outQ, outK: outK, outV: outV,
                on: cmdRef, into: outRow)
        }
        cmdRef.commit(); cmdRef.waitUntilCompleted()

        // Fused M>1 single dispatch into 3 separate output buffers.
        let fusedQ = Tensor.empty(shape: [m, outQ], dtype: dtype)
        let fusedK = Tensor.empty(shape: [m, outK], dtype: dtype)
        let fusedV = Tensor.empty(shape: [m, outV], dtype: dtype)
        let cmdFused = Device.shared.makeCommandBuffer()
        Ops.batchedQkvQmmFast(
            x: x,
            wQ: wQ, scalesQ: sQ, biasesQ: bQ,
            wK: wK, scalesK: sK, biasesK: bK,
            wV: wV, scalesV: sV, biasesV: bV,
            m: m, outQ: outQ, outK: outK, outV: outV,
            on: cmdFused,
            qBuf: fusedQ, kBuf: fusedK, vBuf: fusedV)
        cmdFused.commit(); cmdFused.waitUntilCompleted()
        Self.flushQueue()

        // Reassemble fused into row-major `[M, q|k|v]` to match the
        // oracle's layout.
        let fQ = fusedQ.toFloatArray()
        let fK = fusedK.toFloatArray()
        let fV = fusedV.toFloatArray()
        var fusedArr: [Float] = []
        fusedArr.reserveCapacity(m * (outQ + outK + outV))
        for row in 0..<m {
            fusedArr.append(contentsOf: fQ[row * outQ..<(row + 1) * outQ])
            fusedArr.append(contentsOf: fK[row * outK..<(row + 1) * outK])
            fusedArr.append(contentsOf: fV[row * outV..<(row + 1) * outV])
        }
        let oracleArr = oracleOut.toFloatArray()
        var maxDiff: Float = 0
        var maxAbs: Float = 0
        for i in 0..<oracleArr.count {
            let d = abs(oracleArr[i] - fusedArr[i]); if d > maxDiff { maxDiff = d }
            let a = abs(oracleArr[i]); if a > maxAbs { maxAbs = a }
        }
        let denom = max(maxAbs, 1.0)
        let rel = maxDiff / denom
        if rel >= relativeTolerance {
            print("[\(dtype) M=\(m)] maxDiff=\(maxDiff) maxAbs=\(maxAbs) rel=\(rel)")
        }
        #expect(rel < relativeTolerance)
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
