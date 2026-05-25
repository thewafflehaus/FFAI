// `Ops.gatedRmsNormQgemvInt4Fast` — fused gated-RMSNorm + int4 GEMV.
// Verifies the single-dispatch fused result matches the unfused
// `Ops.gatedMixerNorm + Ops.dequantGemvInt4` chain at Qwen3.5/3.6 GDN
// production shapes (hv=16, dv=128, out_dim=2048).

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.gatedRmsNormQgemvInt4Fast — fused GDN gatedMixerNorm + outProj")
struct GatedRmsNormQgemvInt4FastTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("bf16 Qwen3.6-A3B shape (hv=16 dv=128 out=2048): fused matches reference")
    func bf16Qwen36() {
        runCase(hv: 16, dv: 128, outDim: 2048, dtype: .bf16, tolerance: 5e-2)
    }

    @Test("f16 mid (hv=8 dv=128 out=1024): fused matches reference")
    func f16Mid() {
        runCase(hv: 8, dv: 128, outDim: 1024, dtype: .f16, tolerance: 3e-2)
    }

    @Test("f32 small (hv=4 dv=128 out=512): fused matches reference")
    func f32Small() {
        runCase(hv: 4, dv: 128, outDim: 512, dtype: .f32, tolerance: 1e-2)
    }

    private func runCase(hv: Int, dv: Int, outDim: Int,
                          dtype: DType, tolerance: Float) {
        let inDim = hv * dv
        let groupSize = 64
        let packedPerRow = inDim / 8
        let nGroups = inDim / groupSize
        var seed: UInt64 = 0xDEAD_F00D
        @inline(__always) func xs() -> UInt32 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return UInt32(truncatingIfNeeded: seed)
        }
        @inline(__always) func rsmall() -> Float {
            Float(Int32(truncatingIfNeeded: xs())) / Float(Int32.max) * 0.2
        }

        // y: [Hv, Dv] fp32 from gatedDeltaPrepStep.
        let y = Tensor.empty(shape: [hv, dv], dtype: .f32)
        y.copyIn(from: (0..<inDim).map { _ in rsmall() })

        // z: [Hv*Dv] gate, model dtype.
        let z = Tensor.empty(shape: [inDim], dtype: dtype)
        Self.writeF32(z, (0..<inDim).map { _ in rsmall() }, dtype: dtype)

        // norm_weight: [Dv] T.
        let normWeight = Tensor.empty(shape: [dv], dtype: dtype)
        Self.writeF32(normWeight, (0..<dv).map { _ in 1.0 + rsmall() * 0.05 },
                      dtype: dtype)

        // Quantized weight: [out_dim, in_dim/8] u32.
        let qWeight = Tensor.empty(shape: [outDim, packedPerRow], dtype: .u32)
        var wBytes = [UInt32](); wBytes.reserveCapacity(outDim * packedPerRow)
        for _ in 0..<(outDim * packedPerRow) { wBytes.append(xs()) }
        qWeight.copyIn(from: wBytes)
        let qScales = Tensor.empty(shape: [outDim, nGroups], dtype: dtype)
        Self.writeF32(qScales,
                      (0..<(outDim * nGroups)).map { _ in rsmall() + 0.05 },
                      dtype: dtype)
        let qBiases = Tensor.empty(shape: [outDim, nGroups], dtype: dtype)
        Self.writeF32(qBiases,
                      (0..<(outDim * nGroups)).map { _ in rsmall() },
                      dtype: dtype)

        let eps: Float = 1e-6

        // ── Reference: gatedMixerNorm + dequantGemvInt4 ───────────────
        let yGated = Tensor.empty(shape: [inDim], dtype: dtype)
        let epsBuf = Device.shared.makeBuffer(length: 4)
        var epsCopy = eps
        memcpy(epsBuf.contents(), &epsCopy, 4)
        let epsBufTensor = Tensor(buffer: epsBuf, offset: 0,
                                   shape: [1], dtype: .f32)
        let cmdRef = Device.shared.makeCommandBuffer()
        Ops.gatedMixerNorm(
            y: y, z: z, weight: normWeight, epsBuf: epsBufTensor,
            into: yGated,
            numValueHeads: hv, valueHeadDim: dv,
            on: cmdRef)
        let refOut = Ops.dequantGemvInt4(
            weight: qWeight, scales: qScales, biases: qBiases,
            input: yGated, groupSize: groupSize, on: cmdRef)
        cmdRef.commit(); cmdRef.waitUntilCompleted()

        // ── Fused: one dispatch ───────────────────────────────────────
        let fusedOut = Tensor.empty(shape: [outDim], dtype: dtype)
        let cmdFused = Device.shared.makeCommandBuffer()
        Ops.gatedRmsNormQgemvInt4Fast(
            y: y, z: z, normWeight: normWeight, eps: eps,
            qWeight: qWeight, qScales: qScales, qBiases: qBiases,
            hv: hv, dv: dv, outDim: outDim, groupSize: groupSize,
            on: cmdFused, into: fusedOut)
        cmdFused.commit(); cmdFused.waitUntilCompleted()
        Self.flushQueue()

        let refArr = refOut.toFloatArray()
        let fusedArr = fusedOut.toFloatArray()
        var maxDiff: Float = 0, maxAbs: Float = 0
        for i in 0..<outDim {
            let d = abs(refArr[i] - fusedArr[i]); if d > maxDiff { maxDiff = d }
            let a = abs(refArr[i]); if a > maxAbs { maxAbs = a }
        }
        let denom = max(maxAbs, 1.0)
        let rel = maxDiff / denom
        if rel >= tolerance {
            print("[\(dtype) hv=\(hv) dv=\(dv) out=\(outDim)] " +
                  "maxDiff=\(maxDiff) maxAbs=\(maxAbs) rel=\(rel)")
        }
        #expect(rel < tolerance)
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
