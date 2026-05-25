// `Ops.batchedQkvQgemvInt4Fast` — fused Q/K/V int4 GEMV in one
// dispatch. Verifies the single-kernel-call result matches the
// `dequantGemvInt4Three` shared-encoder reference at production-like
// Qwen3.6-A3B shapes.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.batchedQkvQgemvInt4Fast — fused QKV int4 GEMV")
struct BatchedQkvQgemvInt4FastTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("bf16 hidden=512, q/k/v=256/64/64: fused matches dequantGemvInt4Three")
    func bf16Production() {
        runCase(inDim: 512, outQ: 256, outK: 64, outV: 64,
                 dtype: .bf16, relativeTolerance: 5e-2)
    }

    @Test("f16 hidden=1024, q/k/v=512/128/128: fused matches reference")
    func f16Production() {
        runCase(inDim: 1024, outQ: 512, outK: 128, outV: 128,
                 dtype: .f16, relativeTolerance: 3e-2)
    }

    @Test("f32 hidden=512, q/k/v=8/8/8 (smallest): fused matches reference")
    func f32Smallest() {
        runCase(inDim: 512, outQ: 8, outK: 8, outV: 8,
                 dtype: .f32, relativeTolerance: 2e-2)
    }

    private func runCase(inDim: Int, outQ: Int, outK: Int, outV: Int,
                          dtype: DType, relativeTolerance: Float) {
        let groupSize = 64
        let packedPerRow = inDim / 8
        let nGroupsQ = inDim / groupSize
        var seed: UInt64 = 0xC0FFEE_BABE
        @inline(__always) func xs() -> UInt32 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return UInt32(truncatingIfNeeded: seed)
        }
        @inline(__always) func rsmall() -> Float {
            Float(Int32(truncatingIfNeeded: xs())) / Float(Int32.max) * 0.2
        }
        // Build weights for Q, K, V independently.
        func makeWeights(_ outDim: Int) -> (Tensor, Tensor, Tensor) {
            let w = Tensor.empty(shape: [outDim, packedPerRow], dtype: .u32)
            var wBytes = [UInt32](); wBytes.reserveCapacity(outDim * packedPerRow)
            for _ in 0..<(outDim * packedPerRow) { wBytes.append(xs()) }
            w.copyIn(from: wBytes)
            let s = Tensor.empty(shape: [outDim, nGroupsQ], dtype: dtype)
            Self.writeF32(s, (0..<(outDim * nGroupsQ)).map { _ in rsmall() + 0.05 },
                          dtype: dtype)
            let b = Tensor.empty(shape: [outDim, nGroupsQ], dtype: dtype)
            Self.writeF32(b, (0..<(outDim * nGroupsQ)).map { _ in rsmall() },
                          dtype: dtype)
            return (w, s, b)
        }
        let (wQ, sQ, bQ) = makeWeights(outQ)
        let (wK, sK, bK) = makeWeights(outK)
        let (wV, sV, bV) = makeWeights(outV)
        let x = Tensor.empty(shape: [inDim], dtype: dtype)
        Self.writeF32(x, (0..<inDim).map { _ in rsmall() }, dtype: dtype)

        // Reference: dequantGemvInt4Three on a shared encoder.
        let refOutQ = Tensor.empty(shape: [outQ], dtype: dtype)
        let refOutK = Tensor.empty(shape: [outK], dtype: dtype)
        let refOutV = Tensor.empty(shape: [outV], dtype: dtype)
        let cmdRef = Device.shared.makeCommandBuffer()
        Ops.dequantGemvInt4Three(
            input: x,
            w0: wQ, s0: sQ, b0: bQ, out0: refOutQ,
            w1: wK, s1: sK, b1: bK, out1: refOutK,
            w2: wV, s2: sV, b2: bV, out2: refOutV,
            groupSize: groupSize, on: cmdRef)
        cmdRef.commit()
        cmdRef.waitUntilCompleted()

        // Fused: single dispatch into one concatenated [q+k+v] output.
        let fusedOut = Tensor.empty(shape: [outQ + outK + outV], dtype: dtype)
        let cmdFused = Device.shared.makeCommandBuffer()
        Ops.batchedQkvQgemvInt4Fast(
            x: x,
            wQ: wQ, scalesQ: sQ, biasesQ: bQ,
            wK: wK, scalesK: sK, biasesK: bK,
            wV: wV, scalesV: sV, biasesV: bV,
            outQ: outQ, outK: outK, outV: outV,
            on: cmdFused, into: fusedOut)
        cmdFused.commit()
        cmdFused.waitUntilCompleted()
        Self.flushQueue()

        // Compare each slice independently.
        let refQ = refOutQ.toFloatArray()
        let refK = refOutK.toFloatArray()
        let refV = refOutV.toFloatArray()
        let fused = fusedOut.toFloatArray()
        let fusedQ = Array(fused[0..<outQ])
        let fusedK = Array(fused[outQ..<(outQ + outK)])
        let fusedV = Array(fused[(outQ + outK)..<(outQ + outK + outV)])

        func relCheck(_ ref: [Float], _ got: [Float], _ label: String) -> Bool {
            var maxDiff: Float = 0
            var maxAbs: Float = 0
            for i in 0..<ref.count {
                let d = abs(ref[i] - got[i]); if d > maxDiff { maxDiff = d }
                let a = abs(ref[i]); if a > maxAbs { maxAbs = a }
            }
            let denom = max(maxAbs, 1.0)
            let rel = maxDiff / denom
            if rel >= relativeTolerance {
                print("[\(dtype) in=\(inDim) \(label)] maxDiff=\(maxDiff) " +
                      "maxAbs=\(maxAbs) rel=\(rel)")
            }
            return rel < relativeTolerance
        }
        #expect(relCheck(refQ, fusedQ, "Q"))
        #expect(relCheck(refK, fusedK, "K"))
        #expect(relCheck(refV, fusedV, "V"))
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
