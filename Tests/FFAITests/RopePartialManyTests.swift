// `Ops.ropePartialMany` / `Ops.ropePartialManyTwo` — batched per-row RoPE.
// Verifies the single-dispatch batched result matches the per-row
// `Ops.ropePartial` loop oracle at production-like Qwen3.5/3.6 shapes.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.ropePartialMany — batched per-row RoPE")
struct RopePartialManyTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("bf16 Qwen3.6 attn shape (T=8 Q + K) matches per-row ropePartial")
    func bf16Qwen36() {
        runCase(t: 8, nHeadsQ: 32, nKVHeads: 8, headDim: 256, rotaryDim: 256,
                dtype: .bf16, tolerance: 5e-2)
    }

    @Test("f16 mid (T=16, nHeads=16, headDim=128) matches per-row")
    func f16Mid() {
        runCase(t: 16, nHeadsQ: 16, nKVHeads: 4, headDim: 128, rotaryDim: 128,
                dtype: .f16, tolerance: 3e-2)
    }

    @Test("f32 small (T=4, nHeads=4, headDim=64) matches per-row")
    func f32Small() {
        runCase(t: 4, nHeadsQ: 4, nKVHeads: 2, headDim: 64, rotaryDim: 64,
                dtype: .f32, tolerance: 1e-4)
    }

    private func runCase(t: Int, nHeadsQ: Int, nKVHeads: Int,
                         headDim: Int, rotaryDim: Int,
                         dtype: DType, tolerance: Float) {
        let qRowStride = nHeadsQ * headDim
        let kRowStride = nKVHeads * headDim
        let thetaBase: Float = 10_000.0
        let startPosition = 5
        var seed: UInt64 = 0xCAFE_FACE
        @inline(__always) func xs() -> UInt32 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return UInt32(truncatingIfNeeded: seed)
        }
        @inline(__always) func rsmall() -> Float {
            Float(Int32(truncatingIfNeeded: xs())) / Float(Int32.max) * 0.5
        }
        let qSrc = (0..<(t * qRowStride)).map { _ in rsmall() }
        let kSrc = (0..<(t * kRowStride)).map { _ in rsmall() }

        // ── Reference: per-row ropePartial T-loop (the "before" state) ──
        let qRef = Tensor.empty(shape: [t * qRowStride], dtype: dtype)
        let kRef = Tensor.empty(shape: [t * kRowStride], dtype: dtype)
        Self.writeF32(qRef, qSrc, dtype: dtype)
        Self.writeF32(kRef, kSrc, dtype: dtype)
        let cmdRef = Device.shared.makeCommandBuffer()
        let dtBytes = dtype.byteSize
        for r in 0..<t {
            let qRow = Tensor(buffer: qRef.buffer,
                              offset: qRef.offset + r * qRowStride * dtBytes,
                              shape: [qRowStride], dtype: dtype)
            let kRow = Tensor(buffer: kRef.buffer,
                              offset: kRef.offset + r * kRowStride * dtBytes,
                              shape: [kRowStride], dtype: dtype)
            Ops.ropePartialTwo(qRow, kRow, position: startPosition + r,
                                headDim: headDim, rotaryDim: rotaryDim,
                                thetaBase: thetaBase, on: cmdRef)
        }
        cmdRef.commit(); cmdRef.waitUntilCompleted()

        // ── Batched: ONE dispatch per buffer (Q + K on shared encoder) ──
        let qBatch = Tensor.empty(shape: [t * qRowStride], dtype: dtype)
        let kBatch = Tensor.empty(shape: [t * kRowStride], dtype: dtype)
        Self.writeF32(qBatch, qSrc, dtype: dtype)
        Self.writeF32(kBatch, kSrc, dtype: dtype)
        let positions = Tensor.empty(shape: [t], dtype: .u32)
        positions.copyIn(from: (0..<t).map { UInt32(startPosition + $0) })

        let cmdBatch = Device.shared.makeCommandBuffer()
        Ops.ropePartialManyTwo(
            q: qBatch, qNHeads: nHeadsQ, qRowStride: qRowStride,
            k: kBatch, kNHeads: nKVHeads, kRowStride: kRowStride,
            positions: positions, t: t,
            headDim: headDim, rotaryDim: rotaryDim,
            thetaBase: thetaBase, on: cmdBatch)
        cmdBatch.commit(); cmdBatch.waitUntilCompleted()
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
                print("[\(dtype) T=\(t) \(label)] maxDiff=\(maxDiff) " +
                      "maxAbs=\(maxAbs) rel=\(rel)")
            }
            return rel < tolerance
        }
        #expect(relCheck(qRef, qBatch, "Q"))
        #expect(relCheck(kRef, kBatch, "K"))
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
