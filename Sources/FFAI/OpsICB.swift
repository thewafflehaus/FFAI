// OpsICB — ICB-recording variants of the simplest hot Ops.
//
// The direct Ops API (`Ops.silu`, `Ops.sigmoid`, `Ops.rmsNorm`, ...)
// takes `on cmd: MTLCommandBuffer` and dispatches via a fresh
// MTLComputeCommandEncoder per call. Each encoder costs ~4-17 µs of
// host dispatch overhead on Apple silicon.
//
// `OpsICB` mirrors the API but writes into an `ICBRecorder`. Per-token
// decode replay can then call `recorder.execute(on: cmd)` once instead
// of ~600 encoder dispatches.
//
// Status (Day 1):
//   * Implements silu / sigmoid / rmsNorm.
//   * Test: a 600-call elementwise chain records once + replays 50
//     times; compares total wall to direct-Ops 50× and verifies
//     bit-identical output.
//
// Multi-day rework left:
//   * Quant-aware ops: dequantGemv / qmm (heavy on params, weight
//     buffers must be use(:_)-registered).
//   * GDN, conv1d, sdpa: kernels with multiple buffers + per-token
//     varying scalars (position, length).
//   * Full Qwen35Model.forwardICB: walks every layer's decode dispatch
//     into an ICBRecorder; per-token mutates the scalar slots
//     (position, token id) in the recorder's paramsBuffer.

import Foundation
import Metal
import MetalTileSwift

public enum OpsICB {

    /// SiLU into `out` (must be pre-allocated). Records into `recorder`.
    public static func silu(_ x: Tensor, into out: Tensor,
                            recorder: ICBRecorder) {
        precondition(x.shape == out.shape, "OpsICB.silu: shape mismatch")
        precondition(x.dtype == out.dtype, "OpsICB.silu: dtype mismatch")
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        let psize: Int
        switch x.dtype {
        case .f32: psize = MetalTileKernels.mt_silu_f32_params_size
        case .f16: psize = MetalTileKernels.mt_silu_f16_params_size
        case .bf16: psize = MetalTileKernels.mt_silu_bf16_params_size
        default: fatalError("OpsICB.silu: unsupported dtype \(x.dtype)")
        }
        let slot = recorder.next(paramsSize: psize)
        switch x.dtype {
        case .f32:
            MetalTileKernels.mt_silu_f32_record(
                a: x.buffer, aOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        case .f16:
            MetalTileKernels.mt_silu_f16_record(
                a: x.buffer, aOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        case .bf16:
            MetalTileKernels.mt_silu_bf16_record(
                a: x.buffer, aOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        default: fatalError("OpsICB.silu: unsupported dtype \(x.dtype)")
        }
        recorder.use(x.buffer, usage: .read)
        recorder.use(out.buffer, usage: .write)
    }

    /// Sigmoid into `out`. Records into `recorder`.
    public static func sigmoid(_ x: Tensor, into out: Tensor,
                               recorder: ICBRecorder) {
        precondition(x.shape == out.shape, "OpsICB.sigmoid: shape mismatch")
        precondition(x.dtype == out.dtype, "OpsICB.sigmoid: dtype mismatch")
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        let psize: Int
        switch x.dtype {
        case .f32: psize = MetalTileKernels.mt_sigmoid_f32_params_size
        case .f16: psize = MetalTileKernels.mt_sigmoid_f16_params_size
        case .bf16: psize = MetalTileKernels.mt_sigmoid_bf16_params_size
        default: fatalError("OpsICB.sigmoid: unsupported dtype \(x.dtype)")
        }
        let slot = recorder.next(paramsSize: psize)
        switch x.dtype {
        case .f32:
            MetalTileKernels.mt_sigmoid_f32_record(
                a: x.buffer, aOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        case .f16:
            MetalTileKernels.mt_sigmoid_f16_record(
                a: x.buffer, aOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        case .bf16:
            MetalTileKernels.mt_sigmoid_bf16_record(
                a: x.buffer, aOffset: x.offset,
                out: out.buffer, outOffset: out.offset,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        default: fatalError("OpsICB.sigmoid: unsupported dtype \(x.dtype)")
        }
        recorder.use(x.buffer, usage: .read)
        recorder.use(out.buffer, usage: .write)
    }

    /// Elementwise mul: out = a * b. Both inputs must share shape + dtype.
    public static func mul(_ a: Tensor, _ b: Tensor, into out: Tensor,
                           recorder: ICBRecorder) {
        precondition(a.shape == b.shape && a.shape == out.shape,
                     "OpsICB.mul: shape mismatch")
        precondition(a.dtype == b.dtype && a.dtype == out.dtype,
                     "OpsICB.mul: dtype mismatch")
        precondition(a.dtype == .f32,
                     "OpsICB.mul: only f32 kernel generated today (extend to f16/bf16 when needed)")
        let n = a.elementCount
        let (grid, tg) = elementwiseGrid(n)
        let psize = MetalTileKernels.mt_mul_f32_params_size
        let slot = recorder.next(paramsSize: psize)
        MetalTileKernels.mt_mul_f32_record(
            a: a.buffer, aOffset: a.offset,
            b: b.buffer, bOffset: b.offset,
            out: out.buffer, outOffset: out.offset,
            paramsBuffer: recorder.paramsBuffer,
            paramsBufferOffset: slot.paramsOffset,
            gridSize: grid, threadgroupSize: tg,
            into: slot.command)
        recorder.use(a.buffer, usage: .read)
        recorder.use(b.buffer, usage: .read)
        recorder.use(out.buffer, usage: .write)
    }

    /// Compute the same (grid, threadgroup) tuple `Ops.elementwiseGrid`
    /// produces. Local helper — we don't want OpsICB to import the
    /// private Ops machinery, and these kernels are simple enough that
    /// reproducing the math is cheaper than refactoring.
    @inline(__always)
    private static func elementwiseGrid(_ n: Int) -> (MTLSize, MTLSize) {
        // Match Ops.swift: 1024-wide threadgroups, grid = n.
        let tgWidth = Swift.min(1024, n)
        return (MTLSize(width: n, height: 1, depth: 1),
                MTLSize(width: tgWidth, height: 1, depth: 1))
    }
}
