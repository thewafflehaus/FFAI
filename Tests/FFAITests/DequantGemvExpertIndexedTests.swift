// `dequant_gemv_int4_expert_indexed` GPU kernel correctness tests.
//
// The indexed kernel (ITER 55 wrapper) operates on stacked-expert
// weights `[n_experts, out_dim, in_dim/8]` u32 and reads the expert
// slot from a GPU-resident `expert_index: Tensor<u32>`. Compared to
// the original `dequant_gemv_int4` which expects single-expert
// weights, the math is bit-identical — same threadgroup geometry,
// same reduce_sum tree, same per-row inner loop. The ONLY difference
// is two row-base offsets computed from the loaded expert id.
//
// Strategy: synthesize a stacked weight slab, dispatch the indexed
// kernel with `expert_index[0] = E` into `out_indexed`. Independently
// dispatch the original `dequantGemvInt4` against a Tensor view that
// points to the same `expert E`'s slice of the stacked buffer into
// `out_ref`. Outputs MUST match within f32 precision (same math).

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.dequantGemvInt4ExpertIndexed — GPU correctness vs dequantGemvInt4")
struct DequantGemvExpertIndexedTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    /// One expert in a stacked weight equals the standalone
    /// `dequant_gemv_int4` against that expert's row band.
    /// Shape: nExperts=4, outDim=32, inDim=64, groupSize=64, dtype=f32.
    /// Pick expertIndex=2 — the middle of the slab — to catch any
    /// offset arithmetic bug that would misfire at the ends.
    @Test("f32: indexed[expert=2] == standalone against the same expert slice")
    func f32IndexedMatchesStandalone() {
        runCase(nExperts: 4, outDim: 32, inDim: 64, groupSize: 64,
                expert: 2, dtype: .f32, tolerance: 1e-5)
    }

    @Test("f16: indexed[expert=0] matches standalone (first-row edge)")
    func f16IndexedFirstExpert() {
        runCase(nExperts: 4, outDim: 32, inDim: 64, groupSize: 64,
                expert: 0, dtype: .f16, tolerance: 5e-3)
    }

    @Test("f16: indexed[expert=N-1] matches standalone (last-row edge)")
    func f16IndexedLastExpert() {
        runCase(nExperts: 4, outDim: 32, inDim: 64, groupSize: 64,
                expert: 3, dtype: .f16, tolerance: 5e-3)
    }

    /// Bigger production-like shape: Qwen3.6-A3B has hidden=2048,
    /// moeIntermediate=768. Reduce to fit in a unit test:
    /// outDim=64, inDim=128, groupSize=64, nExperts=8.
    @Test("f32: production-like shape (outDim=64, inDim=128)")
    func f32ProductionLikeShape() {
        runCase(nExperts: 8, outDim: 64, inDim: 128, groupSize: 64,
                expert: 5, dtype: .f32, tolerance: 1e-5)
    }

    /// ITER 58 `Many` variant: dispatching 8 indexed qmms on ONE shared
    /// encoder must produce byte-identical output to 8 single-call
    /// dispatches. Tests the shared-encoder path used by the GPU MoE
    /// router branch (gate+up fused encoder, down encoder).
    @Test("f16: Many variant matches per-call dispatches")
    func f16ManyMatchesPerCall() {
        runManyCase(nExperts: 8, outDim: 32, inDim: 64, groupSize: 64,
                    slotExperts: [0, 1, 2, 3, 4, 5, 6, 7],
                    dtype: .f16, tolerance: 5e-3)
    }

    /// Many variant with REPEATED experts (production case: topK=8 of
    /// nExperts=128 may pick the same expert multiple times under
    /// degenerate logits — kernel must handle it without crashing).
    @Test("f16: Many variant handles repeated expert indices")
    func f16ManyRepeatedExperts() {
        runManyCase(nExperts: 8, outDim: 32, inDim: 64, groupSize: 64,
                    slotExperts: [3, 3, 3, 3, 5, 5, 5, 5],
                    dtype: .f16, tolerance: 5e-3)
    }

    private func runManyCase(
        nExperts: Int, outDim: Int, inDim: Int, groupSize: Int,
        slotExperts: [Int], dtype: DType, tolerance: Float
    ) {
        precondition(inDim % 8 == 0)
        precondition(inDim % groupSize == 0)
        let n = slotExperts.count
        let packedPerRow = inDim / 8
        let nGroups = inDim / groupSize

        // Same synthesis as runCase but factored out as a closure for
        // reuse; we need ONE stacked slab + ONE input shared between
        // ref and Many.
        let weights = Tensor.empty(shape: [nExperts, outDim, packedPerRow], dtype: .u32)
        var wBytes = [UInt32]()
        var seed: UInt64 = 0xDEAD_BEEF_CA11
        for _ in 0..<(nExperts * outDim * packedPerRow) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            wBytes.append(UInt32(truncatingIfNeeded: seed))
        }
        weights.copyIn(from: wBytes)
        let scales = Tensor.empty(shape: [nExperts, outDim, nGroups], dtype: dtype)
        let biases = Tensor.empty(shape: [nExperts, outDim, nGroups], dtype: dtype)
        let sCount = nExperts * outDim * nGroups
        Self.writeF32IntoTypedTensor(scales,
            (0..<sCount).map { Float(($0 % 11) - 5) * 0.02 + 0.07 }, dtype: dtype)
        Self.writeF32IntoTypedTensor(biases,
            (0..<sCount).map { Float(($0 % 7) - 3) * 0.015 }, dtype: dtype)
        let input = Tensor.empty(shape: [inDim], dtype: dtype)
        Self.writeF32IntoTypedTensor(input,
            (0..<inDim).map { Float(($0 % 13) - 6) * 0.04 }, dtype: dtype)

        // n expert-index tensors (each [1] u32 holding the slot's expert).
        let idxBuf = Tensor.empty(shape: [n], dtype: .u32)
        idxBuf.copyIn(from: slotExperts.map { UInt32($0) })
        var idxTensors: [Tensor] = []
        for slot in 0..<n {
            idxTensors.append(Tensor(buffer: idxBuf.buffer,
                                      offset: idxBuf.offset + slot * 4,
                                      shape: [1], dtype: .u32))
        }

        // Reference: n single-call dispatches.
        var refOuts: [Tensor] = []
        for _ in 0..<n {
            refOuts.append(Tensor.empty(shape: [outDim], dtype: dtype))
        }
        let cmdRef = Device.shared.makeCommandBuffer()
        for slot in 0..<n {
            Ops.dequantGemvInt4ExpertIndexed(
                weightsStacked: weights, scalesStacked: scales, biasesStacked: biases,
                input: input, expertIndex: idxTensors[slot],
                groupSize: groupSize, on: cmdRef, into: refOuts[slot])
        }
        cmdRef.commit()
        cmdRef.waitUntilCompleted()

        // Many: 1 shared encoder, n dispatches.
        var manyOuts: [Tensor] = []
        for _ in 0..<n {
            manyOuts.append(Tensor.empty(shape: [outDim], dtype: dtype))
        }
        let ws = Array(repeating: weights, count: n)
        let ss = Array(repeating: scales, count: n)
        let bs = Array(repeating: biases, count: n)
        let ins = Array(repeating: input, count: n)
        let cmdMany = Device.shared.makeCommandBuffer()
        Ops.dequantGemvInt4ExpertIndexedMany(
            weightsStacked: ws, scalesStacked: ss, biasesStacked: bs,
            inputs: ins, expertIndices: idxTensors, outputs: manyOuts,
            groupSize: groupSize, on: cmdMany)
        cmdMany.commit()
        cmdMany.waitUntilCompleted()
        Self.flushQueue()

        // Compare per-slot.
        var globalMaxDiff: Float = 0
        for slot in 0..<n {
            let refArr = refOuts[slot].toFloatArray()
            let manyArr = manyOuts[slot].toFloatArray()
            for i in 0..<outDim {
                let diff = abs(refArr[i] - manyArr[i])
                if diff > globalMaxDiff { globalMaxDiff = diff }
            }
        }
        #expect(globalMaxDiff < tolerance)
        if globalMaxDiff >= tolerance {
            print("[Many \(dtype)] globalMaxDiff=\(globalMaxDiff) slots=\(slotExperts)")
        }
    }

    // ─── Helper: shared test body ───────────────────────────────────

    /// Synthesize a stacked weight slab, dispatch both kernels,
    /// compare output element-by-element.
    private func runCase(
        nExperts: Int, outDim: Int, inDim: Int, groupSize: Int,
        expert: Int, dtype: DType, tolerance: Float
    ) {
        precondition(inDim % 8 == 0, "inDim must pack into u32")
        precondition(inDim % groupSize == 0, "inDim must divide groupSize")
        let packedPerRow = inDim / 8 // int4 → 8 per u32
        let nGroups = inDim / groupSize

        // Stacked weight buffer — random u32 (any 32-bit pattern is a
        // valid int4-packed weight; kernel masks to 0xF per nibble).
        let weights = Tensor.empty(shape: [nExperts, outDim, packedPerRow], dtype: .u32)
        var wBytes = [UInt32]()
        wBytes.reserveCapacity(nExperts * outDim * packedPerRow)
        var seed: UInt64 = 0xC0FFEE_FA11
        for _ in 0..<(nExperts * outDim * packedPerRow) {
            // xorshift64 — deterministic pseudo-random.
            seed ^= seed << 13
            seed ^= seed >> 7
            seed ^= seed << 17
            wBytes.append(UInt32(truncatingIfNeeded: seed))
        }
        weights.copyIn(from: wBytes)

        // Stacked scales / biases.
        let scales = Tensor.empty(shape: [nExperts, outDim, nGroups], dtype: dtype)
        let biases = Tensor.empty(shape: [nExperts, outDim, nGroups], dtype: dtype)
        let sCount = nExperts * outDim * nGroups
        let scalesF32 = (0..<sCount).map { Float(($0 % 7) - 3) * 0.01 + 0.05 }
        let biasesF32 = (0..<sCount).map { Float(($0 % 5) - 2) * 0.02 }
        Self.writeF32IntoTypedTensor(scales, scalesF32, dtype: dtype)
        Self.writeF32IntoTypedTensor(biases, biasesF32, dtype: dtype)

        // Input.
        let input = Tensor.empty(shape: [inDim], dtype: dtype)
        let inputF32 = (0..<inDim).map { Float(($0 % 11) - 5) * 0.03 }
        Self.writeF32IntoTypedTensor(input, inputF32, dtype: dtype)

        // Indexed dispatch: expertIndex tensor with value `expert`.
        let expertIdx = Tensor.empty(shape: [1], dtype: .u32)
        expertIdx.copyIn(from: [UInt32(expert)])
        let outIndexed = Tensor.empty(shape: [outDim], dtype: dtype)
        let cmd1 = Device.shared.makeCommandBuffer()
        Ops.dequantGemvInt4ExpertIndexed(
            weightsStacked: weights, scalesStacked: scales, biasesStacked: biases,
            input: input, expertIndex: expertIdx,
            groupSize: groupSize, on: cmd1, into: outIndexed)
        cmd1.commit()
        cmd1.waitUntilCompleted()

        // Standalone dispatch: Tensor views into the same stacked buffer
        // at the expert's row band offset. The standalone kernel sees a
        // [outDim, packedPerRow] u32 / [outDim, nGroups] T weight slab.
        let weightView = Tensor(
            buffer: weights.buffer,
            offset: weights.offset + expert * outDim * packedPerRow * 4,
            shape: [outDim, packedPerRow], dtype: .u32)
        let scalesView = Tensor(
            buffer: scales.buffer,
            offset: scales.offset + expert * outDim * nGroups * dtype.byteSize,
            shape: [outDim, nGroups], dtype: dtype)
        let biasesView = Tensor(
            buffer: biases.buffer,
            offset: biases.offset + expert * outDim * nGroups * dtype.byteSize,
            shape: [outDim, nGroups], dtype: dtype)
        let outRef = Tensor.empty(shape: [outDim], dtype: dtype)
        let cmd2 = Device.shared.makeCommandBuffer()
        _ = Ops.dequantGemvInt4(
            weight: weightView, scales: scalesView, biases: biasesView,
            input: input, groupSize: groupSize, on: cmd2, into: outRef)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        Self.flushQueue()

        // Compare. Drop failure message — Swift Testing's `#expect`
        // shows the failing expression by default; explicit messages
        // need `Comment(...)` ceremony that's not worth the noise.
        let indexedArr = outIndexed.toFloatArray()
        let refArr = outRef.toFloatArray()
        var maxDiff: Float = 0
        var firstFailRow = -1
        for i in 0..<outDim {
            let diff = abs(indexedArr[i] - refArr[i])
            if diff > maxDiff { maxDiff = diff }
            if diff >= tolerance && firstFailRow < 0 { firstFailRow = i }
        }
        #expect(maxDiff < tolerance)
        if maxDiff >= tolerance {
            // Print on failure for diagnosis.
            print("[\(dtype) expert=\(expert)] maxDiff=\(maxDiff) "
                  + "first-fail row=\(firstFailRow) "
                  + "indexed=\(firstFailRow >= 0 ? indexedArr[firstFailRow] : 0) "
                  + "ref=\(firstFailRow >= 0 ? refArr[firstFailRow] : 0)")
        }
    }

    /// Write a `[Float]` into a typed Tensor (`f32` / `f16` / `bf16`).
    private static func writeF32IntoTypedTensor(_ t: Tensor, _ src: [Float],
                                                  dtype: DType) {
        switch dtype {
        case .f32: t.copyIn(from: src)
        case .f16: t.copyIn(from: src.map { Float16($0) })
        case .bf16:
            t.copyIn(from: src.map { UInt16($0.bitPattern >> 16) })
        default: preconditionFailure("unsupported dtype \(dtype)")
        }
    }
}
