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

    /// RMSNorm: `out = rms_norm(x, w, eps)`. `epsBuf` is a 1-element
    /// fp32 buffer holding epsilon (so it can be mutated per-call
    /// without re-recording). Single-row variant (n = `x.elementCount`).
    public static func rmsNorm(_ x: Tensor, weight w: Tensor, epsBuf: Tensor,
                               into out: Tensor, recorder: ICBRecorder) {
        precondition(x.shape == out.shape, "OpsICB.rmsNorm: shape mismatch")
        precondition(x.dtype == out.dtype && x.dtype == w.dtype,
                     "OpsICB.rmsNorm: dtype mismatch")
        precondition(epsBuf.dtype == .f32 && epsBuf.elementCount == 1,
                     "OpsICB.rmsNorm: epsBuf must be .f32[1]")
        let n = UInt32(x.elementCount)
        let tgWidth = Swift.min(1024, Int(n))
        let grid = MTLSize(width: tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let psize: Int
        switch x.dtype {
        case .f32: psize = MetalTileKernels.mt_rms_norm_f32_params_size
        case .f16: psize = MetalTileKernels.mt_rms_norm_f16_params_size
        case .bf16: psize = MetalTileKernels.mt_rms_norm_bf16_params_size
        default: fatalError("OpsICB.rmsNorm: unsupported dtype \(x.dtype)")
        }
        let slot = recorder.next(paramsSize: psize)
        switch x.dtype {
        case .f32:
            MetalTileKernels.mt_rms_norm_f32_record(
                x: x.buffer, xOffset: x.offset,
                w: w.buffer, wOffset: w.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: n,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        case .f16:
            MetalTileKernels.mt_rms_norm_f16_record(
                x: x.buffer, xOffset: x.offset,
                w: w.buffer, wOffset: w.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: n,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        case .bf16:
            MetalTileKernels.mt_rms_norm_bf16_record(
                x: x.buffer, xOffset: x.offset,
                w: w.buffer, wOffset: w.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: n,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        default: fatalError("OpsICB.rmsNorm: unsupported dtype \(x.dtype)")
        }
        recorder.use(x.buffer, usage: .read)
        recorder.use(w.buffer, usage: .read)
        recorder.use(out.buffer, usage: .write)
        recorder.use(epsBuf.buffer, usage: .read)
    }

    /// Fused residual + RMSNorm: `residualOut = a + b`,
    /// `normedOut = rms_norm(residualOut, w, eps)`. Two outputs.
    public static func addRmsNorm(a: Tensor, b: Tensor,
                                   weight w: Tensor, epsBuf: Tensor,
                                   residualOut: Tensor, normedOut: Tensor,
                                   recorder: ICBRecorder) {
        precondition(a.shape == b.shape && a.shape == residualOut.shape && a.shape == normedOut.shape,
                     "OpsICB.addRmsNorm: shape mismatch")
        precondition(a.dtype == b.dtype && a.dtype == residualOut.dtype && a.dtype == normedOut.dtype && a.dtype == w.dtype,
                     "OpsICB.addRmsNorm: dtype mismatch")
        precondition(epsBuf.dtype == .f32 && epsBuf.elementCount == 1,
                     "OpsICB.addRmsNorm: epsBuf must be .f32[1]")
        let n = UInt32(a.elementCount)
        let tgWidth = Swift.min(1024, Int(n))
        let grid = MTLSize(width: tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let psize: Int
        switch a.dtype {
        case .f32: psize = MetalTileKernels.mt_add_rms_norm_f32_params_size
        case .f16: psize = MetalTileKernels.mt_add_rms_norm_f16_params_size
        case .bf16: psize = MetalTileKernels.mt_add_rms_norm_bf16_params_size
        default: fatalError("OpsICB.addRmsNorm: unsupported dtype \(a.dtype)")
        }
        let slot = recorder.next(paramsSize: psize)
        switch a.dtype {
        case .f32:
            MetalTileKernels.mt_add_rms_norm_f32_record(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                w: w.buffer, wOffset: w.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                residual_out: residualOut.buffer, residual_outOffset: residualOut.offset,
                normed_out: normedOut.buffer, normed_outOffset: normedOut.offset,
                n: n,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        case .f16:
            MetalTileKernels.mt_add_rms_norm_f16_record(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                w: w.buffer, wOffset: w.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                residual_out: residualOut.buffer, residual_outOffset: residualOut.offset,
                normed_out: normedOut.buffer, normed_outOffset: normedOut.offset,
                n: n,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        case .bf16:
            MetalTileKernels.mt_add_rms_norm_bf16_record(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                w: w.buffer, wOffset: w.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                residual_out: residualOut.buffer, residual_outOffset: residualOut.offset,
                normed_out: normedOut.buffer, normed_outOffset: normedOut.offset,
                n: n,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        default: fatalError("OpsICB.addRmsNorm: unsupported dtype \(a.dtype)")
        }
        recorder.use(a.buffer, usage: .read)
        recorder.use(b.buffer, usage: .read)
        recorder.use(w.buffer, usage: .read)
        recorder.use(epsBuf.buffer, usage: .read)
        recorder.use(residualOut.buffer, usage: .write)
        recorder.use(normedOut.buffer, usage: .write)
    }

    // NOTE: `dequantGemvInt4` ICB wrapper exists below but produces
    // INCORRECT results when chained with dependent reads, because
    // `MTLIndirectCommandBuffer` with `commandTypes = .concurrentDispatch`
    // + `inheritBuffers = false` runs commands in parallel and Metal's
    // hazard tracking can't see ICB bindings. The dependent qgemv
    // reads-before-write races its producer. Day 1's smoke test
    // didn't catch this because all 600 sigmoid dispatches wrote the
    // same buffer (no real dependency).
    //
    // Day 2 path forward: split ICB execution into dependency groups
    // separated by `MTLComputeCommandEncoder.memoryBarrier(scope:
    // .buffers)` calls, OR use multiple ICBs with barriers in the
    // enclosing encoder between executeCommandsInBuffer calls. This
    // wrapper is kept below for the moment a dep-aware execution
    // helper lands; do NOT use it in dependent chains today.

    /// int4-quantized GEMV: `out = dequant(weight) @ input`. `weight` is
    /// packed int4 (u32 storage). `input` is `[in_dim]`. `output` is
    /// `[out_dim]`. Constexprs: `in_dim`, `group_size`.
    public static func dequantGemvInt4(weight: Tensor, scales: Tensor,
                                        biases: Tensor, input: Tensor,
                                        into output: Tensor,
                                        groupSize: Int = 64,
                                        recorder: ICBRecorder) {
        precondition(weight.dtype == .u32, "OpsICB.dequantGemvInt4: weight must be u32")
        precondition(scales.dtype == biases.dtype && scales.dtype == input.dtype && scales.dtype == output.dtype,
                     "OpsICB.dequantGemvInt4: dtype mismatch")
        let outDim = weight.shape[0]
        let inDim = input.elementCount
        precondition(output.elementCount == outDim,
                     "OpsICB.dequantGemvInt4: output size mismatch")
        // Reduction kernel — one 256-thread TG per output row. Matches
        // `Ops.dequantGemv`'s `grid = outDim * 256, tg = 256` geometry.
        let tgWidth = 256
        let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let psize: Int
        switch scales.dtype {
        case .f32: psize = MetalTileKernels.dequant_gemv_int4_f32_params_size
        case .f16: psize = MetalTileKernels.dequant_gemv_int4_f16_params_size
        case .bf16: psize = MetalTileKernels.dequant_gemv_int4_bf16_params_size
        default: fatalError("OpsICB.dequantGemvInt4: unsupported dtype \(scales.dtype)")
        }
        let slot = recorder.next(paramsSize: psize)
        let inD = UInt32(inDim), gs = UInt32(groupSize)
        switch scales.dtype {
        case .f32:
            MetalTileKernels.dequant_gemv_int4_f32_record(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: output.buffer, outputOffset: output.offset,
                in_dim: inD, group_size: gs,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        case .f16:
            MetalTileKernels.dequant_gemv_int4_f16_record(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: output.buffer, outputOffset: output.offset,
                in_dim: inD, group_size: gs,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        case .bf16:
            MetalTileKernels.dequant_gemv_int4_bf16_record(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: output.buffer, outputOffset: output.offset,
                in_dim: inD, group_size: gs,
                paramsBuffer: recorder.paramsBuffer,
                paramsBufferOffset: slot.paramsOffset,
                gridSize: grid, threadgroupSize: tg,
                into: slot.command)
        default: fatalError("OpsICB.dequantGemvInt4: unsupported dtype \(scales.dtype)")
        }
        recorder.use(weight.buffer, usage: .read)
        recorder.use(scales.buffer, usage: .read)
        recorder.use(biases.buffer, usage: .read)
        recorder.use(input.buffer, usage: .read)
        recorder.use(output.buffer, usage: .write)
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
