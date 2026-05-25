// `Ops.moeDownSwigluAccumInt4Chain8` — fused MoE phase 1b + 2 + 3 in
// one kernel launch. Verifies the fused single-dispatch output matches
// the unfused 3-stage reference (`swigluMany` + per-expert
// `dequantGemvInt4ExpertIndexed` × 8 + `scalarFMAChain8`).

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.moeDownSwigluAccumInt4Chain8 — fused MoE down+swiglu+chain8")
struct MoEDownSwigluAccumTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("bf16 Qwen3.6-A3B shape (hidden=2048 inter=768 k=8): fused matches reference")
    func bf16Production() {
        runCase(nExperts: 128, inDim: 768, outDim: 2048,
                 dtype: .bf16, relativeTolerance: 5e-2)
    }

    @Test("f16 mid shape (hidden=1024 inter=512 k=8): fused matches reference")
    func f16Mid() {
        runCase(nExperts: 32, inDim: 512, outDim: 1024,
                 dtype: .f16, relativeTolerance: 3e-2)
    }

    @Test("f32 small shape (hidden=128 inter=64 k=8 nExperts=8): fused matches reference")
    func f32Smallest() {
        runCase(nExperts: 8, inDim: 64, outDim: 128,
                 dtype: .f32, relativeTolerance: 1e-3)
    }

    private func runCase(nExperts: Int, inDim: Int, outDim: Int,
                          dtype: DType, relativeTolerance: Float) {
        let groupSize = 64
        let k = 8
        precondition(inDim % groupSize == 0,
                     "test: inDim must be multiple of groupSize")
        precondition(outDim % 8 == 0,
                     "test: outDim must be multiple of 8 (kernel constraint)")
        var seed: UInt64 = 0xBEEF_CAFE
        @inline(__always) func xs() -> UInt32 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return UInt32(truncatingIfNeeded: seed)
        }
        @inline(__always) func rsmall() -> Float {
            Float(Int32(truncatingIfNeeded: xs())) / Float(Int32.max) * 0.2
        }

        // Gates / ups — 8 slots, each `[inDim]` in model dtype.
        var gates: [Tensor] = []
        var ups: [Tensor] = []
        for _ in 0..<k {
            let g = Tensor.empty(shape: [inDim], dtype: dtype)
            let u = Tensor.empty(shape: [inDim], dtype: dtype)
            Self.writeF32(g, (0..<inDim).map { _ in rsmall() }, dtype: dtype)
            Self.writeF32(u, (0..<inDim).map { _ in rsmall() }, dtype: dtype)
            gates.append(g)
            ups.append(u)
        }

        // Expert indices and slot weights (un-normalised).
        let exIdxBytes: [UInt32] = (0..<k).map { i in
            UInt32((i * 13 + 3) % nExperts)  // deterministic, distinct
        }
        let expertIndices = Tensor.empty(shape: [k], dtype: .u32)
        expertIndices.copyIn(from: exIdxBytes)

        let slotWeights = Tensor.empty(shape: [k], dtype: dtype)
        let slotWeightsF32: [Float] = [0.31, 0.19, 0.12, 0.08,
                                        0.06, 0.05, 0.04, 0.03]
        Self.writeF32(slotWeights, slotWeightsF32, dtype: dtype)

        // Stacked weights — `[nExperts, outDim, inDim/8]` u32.
        let packedPerRow = inDim / 8
        let nGroups = inDim / groupSize
        let wStacked = Tensor.empty(shape: [nExperts, outDim, packedPerRow],
                                     dtype: .u32)
        var wBytes = [UInt32](); wBytes.reserveCapacity(nExperts * outDim * packedPerRow)
        for _ in 0..<(nExperts * outDim * packedPerRow) { wBytes.append(xs()) }
        wStacked.copyIn(from: wBytes)
        let sStacked = Tensor.empty(shape: [nExperts, outDim, nGroups], dtype: dtype)
        Self.writeF32(sStacked,
                      (0..<(nExperts * outDim * nGroups)).map { _ in rsmall() + 0.05 },
                      dtype: dtype)
        let bStacked = Tensor.empty(shape: [nExperts, outDim, nGroups], dtype: dtype)
        Self.writeF32(bStacked,
                      (0..<(nExperts * outDim * nGroups)).map { _ in rsmall() },
                      dtype: dtype)

        // ── Reference (3 separate dispatches) ───────────────────────────
        let cmdRef = Device.shared.makeCommandBuffer()
        // (1) swigluMany — element-wise silu(gate) * up over the 8 slots.
        var inners: [Tensor] = []
        for _ in 0..<k {
            inners.append(Tensor.empty(shape: [inDim], dtype: dtype))
        }
        Ops.swigluMany(gates: gates, ups: ups, outs: inners, on: cmdRef)

        // (2) per-slot dequant-gemv-expert-indexed: out[k] = W_down[expert] · inner[k].
        var perSlotOuts: [Tensor] = []
        let dtBytes = dtype.byteSize
        var expertIdxScratches: [Tensor] = []
        for slot in 0..<k {
            perSlotOuts.append(Tensor.empty(shape: [outDim], dtype: dtype))
            expertIdxScratches.append(Tensor(
                buffer: expertIndices.buffer,
                offset: expertIndices.offset + slot * 4,
                shape: [1], dtype: .u32))
        }
        Ops.dequantGemvInt4ExpertIndexedMany(
            weightsStacked: Array(repeating: wStacked, count: k),
            scalesStacked: Array(repeating: sStacked, count: k),
            biasesStacked: Array(repeating: bStacked, count: k),
            inputs: inners,
            expertIndices: expertIdxScratches,
            outputs: perSlotOuts,
            groupSize: groupSize, on: cmdRef)

        // (3) scalarFMAChain8: acc[i] = Σ slot_weight[k] * out[k][i].
        var scalarSlots: [Tensor] = []
        for slot in 0..<k {
            scalarSlots.append(Tensor(
                buffer: slotWeights.buffer,
                offset: slotWeights.offset + slot * dtBytes,
                shape: [1], dtype: dtype))
        }
        let refOut = Tensor.empty(shape: [outDim], dtype: dtype)
        Ops.scalarFMAChain8(scalars: scalarSlots, values: perSlotOuts,
                             out: refOut, on: cmdRef)
        cmdRef.commit(); cmdRef.waitUntilCompleted()

        // ── Fused single dispatch ───────────────────────────────────────
        let fusedOut = Tensor.empty(shape: [outDim], dtype: dtype)
        let cmdFused = Device.shared.makeCommandBuffer()
        Ops.moeDownSwigluAccumInt4Chain8(
            gates: gates, ups: ups,
            expertIndices: expertIndices,
            slotWeights: slotWeights,
            weightsStacked: wStacked,
            scalesStacked: sStacked,
            biasesStacked: bStacked,
            output: fusedOut,
            inDim: inDim, outDim: outDim, groupSize: groupSize,
            on: cmdFused)
        cmdFused.commit(); cmdFused.waitUntilCompleted()
        Self.flushQueue()

        let refArr = refOut.toFloatArray()
        let fusedArr = fusedOut.toFloatArray()
        var maxDiff: Float = 0
        var maxAbs: Float = 0
        for i in 0..<outDim {
            let d = abs(refArr[i] - fusedArr[i]); if d > maxDiff { maxDiff = d }
            let a = abs(refArr[i]); if a > maxAbs { maxAbs = a }
        }
        let denom = max(maxAbs, 1.0)
        let rel = maxDiff / denom
        if rel >= relativeTolerance {
            print("[\(dtype) hidden=\(outDim) inter=\(inDim)] " +
                  "maxDiff=\(maxDiff) maxAbs=\(maxAbs) rel=\(rel)")
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
