// `Ops.conv1dCausalStepSiluCastMany` — batched conv1d + silu + cast
// across T tokens in one dispatch. Verifies output + state_out match
// the per-token `Ops.conv1dCausalStep` + `Ops.siluCastToF32` reference
// at Qwen3.6/Mamba2 shapes (kernel=4).

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.conv1dCausalStepSiluCastMany — batched conv+silu+cast")
struct Conv1dCausalStepSiluCastManyTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("bf16 Qwen3.6 conv shape (T=8, convDim=2048, K=4) matches per-token")
    func bf16Qwen36() {
        runCase(t: 8, convDim: 2048, dtype: .bf16, tolerance: 5e-2)
    }

    @Test("f16 mid (T=16, convDim=256, K=4) matches per-token")
    func f16Mid() {
        runCase(t: 16, convDim: 256, dtype: .f16, tolerance: 5e-3)
    }

    @Test("f32 small (T=4, convDim=64, K=4) matches per-token")
    func f32Small() {
        runCase(t: 4, convDim: 64, dtype: .f32, tolerance: 1e-4)
    }

    private func runCase(t: Int, convDim: Int, dtype: DType, tolerance: Float) {
        let convKernel = 4
        var seed: UInt64 = 0xBEEF_CAFE
        @inline(__always) func xs() -> UInt32 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return UInt32(truncatingIfNeeded: seed)
        }
        @inline(__always) func rsmall() -> Float {
            Float(Int32(truncatingIfNeeded: xs())) / Float(Int32.max) * 0.3
        }

        let srcAll = (0..<(t * convDim)).map { _ in rsmall() }
        let wAll = (0..<(convKernel * convDim)).map { _ in rsmall() }
        let bAll = (0..<convDim).map { _ in rsmall() }
        let stateAll = (0..<((convKernel - 1) * convDim)).map { _ in rsmall() }

        let src = Tensor.empty(shape: [t, convDim], dtype: dtype)
        let w = Tensor.empty(shape: [convKernel, convDim], dtype: dtype)
        let bT = Tensor.empty(shape: [convDim], dtype: dtype)
        Self.writeF32(src, srcAll, dtype: dtype)
        Self.writeF32(w, wAll, dtype: dtype)
        Self.writeF32(bT, bAll, dtype: dtype)

        // ── Reference: per-token conv1d + silu_cast T-loop ──
        let stateRef = Tensor.empty(shape: [convKernel - 1, convDim], dtype: dtype)
        Self.writeF32(stateRef, stateAll, dtype: dtype)
        let convScratchRef = Tensor.empty(shape: [convDim], dtype: dtype)
        let outRefAll = Tensor.empty(shape: [t * convDim], dtype: .f32)
        let cmdRef = Device.shared.makeCommandBuffer()
        let dtBytes = dtype.byteSize
        for r in 0..<t {
            let srcRow = Tensor(buffer: src.buffer,
                                offset: src.offset + r * convDim * dtBytes,
                                shape: [convDim], dtype: dtype)
            Ops.conv1dCausalStep(
                x: srcRow, w: w, b: bT,
                state: stateRef, into: convScratchRef,
                nChannels: convDim, kernelSize: convKernel, on: cmdRef)
            let outRowF32 = Tensor(buffer: outRefAll.buffer,
                                    offset: outRefAll.offset + r * convDim * 4,
                                    shape: [convDim], dtype: .f32)
            if dtype == .f32 {
                _ = Ops.silu(convScratchRef, on: cmdRef, into: convScratchRef)
                Ops.castToF32(convScratchRef, into: outRowF32, on: cmdRef)
            } else {
                Ops.siluCastToF32(convScratchRef, into: outRowF32, on: cmdRef)
            }
        }
        cmdRef.commit(); cmdRef.waitUntilCompleted()

        // ── Batched: ONE dispatch, in-place state update ──
        let stateBatch = Tensor.empty(shape: [convKernel - 1, convDim], dtype: dtype)
        Self.writeF32(stateBatch, stateAll, dtype: dtype)
        let outBatchAll = Tensor.empty(shape: [t * convDim], dtype: .f32)
        let cmdBatch = Device.shared.makeCommandBuffer()
        Ops.conv1dCausalStepSiluCastMany(
            src: src, w: w, b: bT,
            stateIn: stateBatch, outF32: outBatchAll, stateOut: stateBatch,
            t: t, convDim: convDim, convKernel: convKernel,
            on: cmdBatch)
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
                print("[\(dtype) T=\(t) convDim=\(convDim) \(label)] " +
                      "maxDiff=\(maxDiff) maxAbs=\(maxAbs) rel=\(rel)")
            }
            return rel < tolerance
        }
        #expect(relCheck(outRefAll, outBatchAll, "out_f32"))
        #expect(relCheck(stateRef, stateBatch, "state_out"))
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
