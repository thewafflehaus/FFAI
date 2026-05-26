// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// OpsFused — fused activation + norm + projection kernels.
//
// These ops compose multiple primitive operations into one kernel
// launch, eliminating intermediate writes to global memory. They are
// the highest-perf path for the model layers that use them:
//   * `fusedGateGelu` / `fusedGateClippedSwiglu` / `sigmoidMul` — fused
//     gate-and-value activations for MLP / MoE blocks
//   * `rmsNormResidual` — fused `add(x, residual)` + `rmsNorm` (one of
//     the two transformer-block fused norms — `addAndRmsNorm` in
//     `Ops.swift` is the sibling for the post-MLP path)
//   * `rmsNormRope` — fused per-head RMSNorm + RoPE; used by Qwen3 /
//     Llama4 / Gemma3 etc. where Q/K go through `q_norm` / `k_norm`
//     before RoPE
//   * `gatedRmsNorm` — Mamba2-style fused `silu(z) * w * rmsnorm(y)`
//     used in GDN mixer norms
//   * `batchedQKVQGemv` — fused triple-projection int4 dequant + GEMV
//     for the attention QKV step at decode (T=1)
//   * `rmsNormSmall` — RMSNorm specialized for "small" row widths
//     (n < 128) where the wide variant would be wasteful.
//   * `scalarFMA` / `scalarFMAMany` / `scalarFMAChain8` —
//     scalar-broadcast FMA for the MoE top-K expert accumulator path
//     (avoids materializing a `Tensor.filled([hidden], weight)`
//     broadcast buffer per expert).
//   * `sigmoidScalarFMAResidual` — `sigmoidScalarFMA` with the
//     residual add folded in for the post-MoE-FFN path (the bare
//     `sigmoidScalarFMA` sibling lives in `Ops.swift`).

import Foundation
import Metal
import MetalTileSwift

extension Ops {

    // ─── fused gate-Gelu / clipped-SwiGLU / sigmoid-mul ──────────────
    //
    // These are pure elementwise fusions of `f(gate) ⊗ up` in one
    // kernel — saves the intermediate `f(gate)` write that `silu(gate)`
    // + `mul(result, up)` would otherwise spill. Used by MoE families
    // whose activation graph emits the pattern.

    /// Element-wise `out[i] = gelu(gate[i]) * up[i]`.
    public static func fusedGateGelu(
        gate: Tensor, up: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(gate.shape == up.shape, "Ops.fusedGateGelu: shape mismatch")
        precondition(gate.dtype == up.dtype, "Ops.fusedGateGelu: dtype mismatch")
        let result = out ?? Tensor.empty(shape: gate.shape, dtype: gate.dtype)
        let (grid, tg) = elementwiseGrid(gate.elementCount)
        switch gate.dtype {
        case .f32:
            MetalTileKernels.mt_fused_gate_gelu_f32(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_fused_gate_gelu_f16(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_fused_gate_gelu_bf16(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.fusedGateGelu: unsupported dtype \(gate.dtype)")
        }
        return result
    }

    /// Element-wise `out[i] = silu(min(gate[i], clip)) * (up[i] + 1)`
    /// — the GPT-OSS / SmolLM3 "clipped SwiGLU" variant.
    public static func fusedGateClippedSwiglu(
        gate: Tensor, up: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(gate.shape == up.shape, "Ops.fusedGateClippedSwiglu: shape mismatch")
        precondition(gate.dtype == up.dtype, "Ops.fusedGateClippedSwiglu: dtype mismatch")
        let result = out ?? Tensor.empty(shape: gate.shape, dtype: gate.dtype)
        let (grid, tg) = elementwiseGrid(gate.elementCount)
        switch gate.dtype {
        case .f32:
            MetalTileKernels.mt_fused_gate_clipped_swiglu_f32(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_fused_gate_clipped_swiglu_f16(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_fused_gate_clipped_swiglu_bf16(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.fusedGateClippedSwiglu: unsupported dtype \(gate.dtype)")
        }
        return result
    }

    /// Element-wise `out[i] = a[i] * sigmoid(b[i])`. Used by Qwen3.5's
    /// gated-attention output (`o = sdpa_out * sigmoid(gate_proj(x))`).
    /// Note the argument order: `a` is the value, `b` is the gate.
    public static func sigmoidMul(
        _ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(a.shape == b.shape, "Ops.sigmoidMul: shape mismatch")
        precondition(a.dtype == b.dtype, "Ops.sigmoidMul: dtype mismatch")
        let result = out ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let (grid, tg) = elementwiseGrid(a.elementCount)
        switch a.dtype {
        case .f32:
            MetalTileKernels.mt_sigmoid_mul_f32(
                a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_sigmoid_mul_f16(
                a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_sigmoid_mul_bf16(
                a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sigmoidMul: unsupported dtype \(a.dtype)")
        }
        return result
    }

    // ─── ffai_rms_norm_residual ──────────────────────────────────────
    //
    // `out[r, i] = residual[r, i] + w[i] * x[r, i] * rsqrt(mean(x[r]²) + eps)`.
    // Reduction-mode (TPG = n/4); n must be multiple of 128 and ≤ 4096.

    public static func rmsNormResidual(
        x: Tensor, residual: Tensor, weight: Tensor, epsBuf: Tensor,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            x.shape == residual.shape,
            "Ops.rmsNormResidual: x/residual shape mismatch")
        precondition(
            x.dtype == residual.dtype && residual.dtype == weight.dtype,
            "Ops.rmsNormResidual: x/residual/weight dtype mismatch")
        precondition(
            epsBuf.dtype == .f32 && epsBuf.elementCount == 1,
            "Ops.rmsNormResidual: epsBuf must be a single f32 element")
        precondition(!x.shape.isEmpty, "Ops.rmsNormResidual: x must be non-empty")
        let n = x.shape.last!
        let rows = x.elementCount / n
        precondition(
            weight.elementCount == n,
            "Ops.rmsNormResidual: weight (\(weight.elementCount)) must have n=\(n) elements")
        if let reason = OpsValidation.validateRmsNormResidual(n: n) {
            preconditionFailure("Ops.rmsNormResidual: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        // Kernel invariants: TPG = n/4 (multiple of 32, ≤ 1024); 1 TG/row.
        let tgSize = n / 4
        let grid = MTLSize(width: rows * tgSize, height: 1, depth: 1)
        let tg = MTLSize(width: tgSize, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.ffai_rms_norm_residual_f32(
                x: x.buffer, xOffset: x.offset,
                residual: residual.buffer, residualOffset: residual.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_rms_norm_residual_f16(
                x: x.buffer, xOffset: x.offset,
                residual: residual.buffer, residualOffset: residual.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_rms_norm_residual_bf16(
                x: x.buffer, xOffset: x.offset,
                residual: residual.buffer, residualOffset: residual.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rmsNormResidual: unsupported dtype \(x.dtype)")
        }
        return result
    }

    // ─── ffai_gated_rmsnorm ──────────────────────────────────────────
    //
    // `out[r, i] = w[i] · y[r, i] · rsqrt(mean(y[r]²) + eps) · silu(z[r, i])`.
    // y is fp32 (the GDN recurrence output); z/w/out are T. TPG = n/4
    // (multiple of 32, ≤ 1024).

    public static func gatedRmsNorm(
        y: Tensor, z: Tensor, weight: Tensor, epsBuf: Tensor,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(y.shape == z.shape, "Ops.gatedRmsNorm: y/z shape mismatch")
        precondition(y.dtype == .f32, "Ops.gatedRmsNorm: y must be f32 (GDN recurrence output)")
        precondition(
            z.dtype == weight.dtype,
            "Ops.gatedRmsNorm: z/weight dtype must match")
        precondition(
            epsBuf.dtype == .f32 && epsBuf.elementCount == 1,
            "Ops.gatedRmsNorm: epsBuf must be a single f32 element")
        precondition(!y.shape.isEmpty, "Ops.gatedRmsNorm: y must be non-empty")
        let n = y.shape.last!
        let rows = y.elementCount / n
        precondition(
            weight.elementCount == n,
            "Ops.gatedRmsNorm: weight (\(weight.elementCount)) must have n=\(n) elements")
        if let reason = OpsValidation.validateGatedRmsNorm(n: n) {
            preconditionFailure("Ops.gatedRmsNorm: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: y.shape, dtype: z.dtype)
        let tgSize = n / 4
        let grid = MTLSize(width: rows * tgSize, height: 1, depth: 1)
        let tg = MTLSize(width: tgSize, height: 1, depth: 1)
        switch z.dtype {
        case .f32:
            MetalTileKernels.ffai_gated_rmsnorm_f32(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_gated_rmsnorm_f16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gated_rmsnorm_bf16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedRmsNorm: unsupported dtype \(z.dtype)")
        }
        return result
    }

    // ─── mt_rms_norm_small ───────────────────────────────────────────
    //
    // RMSNorm specialised for "small" row widths (n ∈ {64, 96, 128, …}
    // for per-head qnorm / knorm). The default `mt_rms_norm` uses 4
    // elements/thread so `TPG = n/4`; that hits the `TPG ≥ 32`
    // simdgroup floor only at `n ≥ 128`. The small variant uses 2
    // elements/thread (`TPG = n/2`), so `n ≥ 64` is supported, and
    // it caps at `n ≤ 2048` (TPG ≤ 1024).

    public static func rmsNormSmall(
        _ x: Tensor, weight: Tensor, epsBuf: Tensor,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            x.dtype == weight.dtype,
            "Ops.rmsNormSmall: x/weight dtype mismatch")
        precondition(
            epsBuf.dtype == .f32 && epsBuf.elementCount == 1,
            "Ops.rmsNormSmall: epsBuf must be a single f32 element")
        precondition(!x.shape.isEmpty, "Ops.rmsNormSmall: x must be non-empty")
        let n = x.shape.last!
        let rows = x.elementCount / n
        precondition(
            weight.elementCount == n,
            "Ops.rmsNormSmall: weight (\(weight.elementCount)) must have n=\(n) elements")
        if let reason = OpsValidation.validateRmsNormSmall(n: n) {
            preconditionFailure("Ops.rmsNormSmall: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        // TPG = n/2 (2 elements/thread). Must be ≥ 32 (simdgroup) and
        // ≤ 1024 (Apple cap) — see validateRmsNormSmall.
        let tgSize = n / 2
        let grid = MTLSize(width: rows * tgSize, height: 1, depth: 1)
        let tg = MTLSize(width: tgSize, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.mt_rms_norm_small_f32(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_rms_norm_small_f16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_rms_norm_small_bf16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rmsNormSmall: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// `out[i] = base[i] + scalar[0] * value[i]` with `scalar` a
    /// 1-element buffer. Replaces the MoE top-K weighted-add chain at
    /// decode T=1, which would otherwise be `Tensor.filled([hidden],
    /// weight)` + `Ops.mul(expertOut, broadcast)` + `Ops.add(acc,
    /// scaled)` — collapsing 1 host alloc + 2 dispatches into 1
    /// dispatch + a 4-byte scalar buffer. Aliasing `out == base` is
    /// safe: the kernel reads `value[idx]` and `base[idx]` then writes
    /// `out[idx]`.
    public static func scalarFMA(
        scalar: Tensor, value: Tensor, base: Tensor,
        into out: Tensor, on cmd: MTLCommandBuffer
    ) {
        precondition(
            scalar.dtype == value.dtype && value.dtype == base.dtype
                && base.dtype == out.dtype,
            "Ops.scalarFMA: all tensors must share dtype")
        precondition(
            scalar.elementCount == 1,
            "Ops.scalarFMA: scalar must be [1] (got \(scalar.elementCount))")
        precondition(
            value.elementCount == base.elementCount
                && base.elementCount == out.elementCount,
            "Ops.scalarFMA: value / base / out must have matching elementCount")
        let n = value.elementCount
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch out.dtype {
        case .f32:
            MetalTileKernels.mt_scalar_fma_f32(
                scalar: scalar.buffer, scalarOffset: scalar.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_scalar_fma_f16(
                scalar: scalar.buffer, scalarOffset: scalar.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_scalar_fma_bf16(
                scalar: scalar.buffer, scalarOffset: scalar.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.scalarFMA: unsupported dtype \(out.dtype)")
        }
    }

    /// N back-to-back `scalarFMA` dispatches accumulating into the same
    /// `acc` tensor on ONE compute encoder. Saves N-1 encoder
    /// begin/end pairs versus N independent `scalarFMA` calls. Used by
    /// the MoE top-K accumulator path (N=topK, ×nMoELayers per decode
    /// token). Metal's in-order execution within a single encoder makes
    /// the serial accumulation safe.
    public static func scalarFMAMany(
        scalars: [Tensor], values: [Tensor], acc: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            scalars.count == values.count, "Ops.scalarFMAMany: count mismatch")
        precondition(!scalars.isEmpty, "Ops.scalarFMAMany: empty")
        let dt = acc.dtype
        for (i, s) in scalars.enumerated() {
            precondition(
                s.elementCount == 1,
                "Ops.scalarFMAMany: scalar[\(i)] must be [1]")
            precondition(
                s.dtype == dt && values[i].dtype == dt,
                "Ops.scalarFMAMany: dtype mismatch at \(i)")
            precondition(
                values[i].elementCount == acc.elementCount,
                "Ops.scalarFMAMany: value[\(i)] / acc size mismatch")
        }
        let n = acc.elementCount
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        let psoName: String
        switch dt {
        case .f32: psoName = "mt_scalar_fma_f32"
        case .f16: psoName = "mt_scalar_fma_f16"
        case .bf16: psoName = "mt_scalar_fma_bf16"
        default: fatalError("Ops.scalarFMAMany: unsupported dtype \(dt)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        enc.setComputePipelineState(pso)
        // Kernel buffer bindings: 0 scalar, 1 value, 2 base, 3 out.
        // Base and out are always `acc`; only scalar/value rotate per
        // dispatch within this encoder.
        enc.setBuffer(acc.buffer, offset: acc.offset, index: 2)
        enc.setBuffer(acc.buffer, offset: acc.offset, index: 3)
        for i in 0 ..< scalars.count {
            enc.setBuffer(scalars[i].buffer, offset: scalars[i].offset, index: 0)
            enc.setBuffer(values[i].buffer, offset: values[i].offset, index: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        enc.endEncoding()
    }

    /// 8-way fused scalarFMA chain. Computes
    /// `out[i] = sum_{k=0..8} scalars[k][0] * values[k][i]` in ONE
    /// kernel dispatch. Collapses the topK=8 expert accumulator chain
    /// (8 sequential `mt_scalar_fma` dispatches + 1 zero-fill of `acc`)
    /// into a single dispatch with 16 input buffers, saving 7 read +
    /// 7 write roundtrips of `acc` per MoE layer.
    public static func scalarFMAChain8(
        scalars: [Tensor], values: [Tensor], out: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            scalars.count == 8 && values.count == 8,
            "Ops.scalarFMAChain8: requires exactly 8 of each")
        let dt = out.dtype
        for i in 0 ..< 8 {
            precondition(
                scalars[i].elementCount == 1,
                "Ops.scalarFMAChain8: scalar[\(i)] must be [1]")
            precondition(
                scalars[i].dtype == dt && values[i].dtype == dt,
                "Ops.scalarFMAChain8: dtype mismatch at \(i)")
            precondition(
                values[i].elementCount == out.elementCount,
                "Ops.scalarFMAChain8: value[\(i)] / out size mismatch")
        }
        let n = out.elementCount
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch dt {
        case .f32:
            MetalTileKernels.mt_scalar_fma_chain8_f32(
                scalar0: scalars[0].buffer, scalar0Offset: scalars[0].offset,
                value0: values[0].buffer, value0Offset: values[0].offset,
                scalar1: scalars[1].buffer, scalar1Offset: scalars[1].offset,
                value1: values[1].buffer, value1Offset: values[1].offset,
                scalar2: scalars[2].buffer, scalar2Offset: scalars[2].offset,
                value2: values[2].buffer, value2Offset: values[2].offset,
                scalar3: scalars[3].buffer, scalar3Offset: scalars[3].offset,
                value3: values[3].buffer, value3Offset: values[3].offset,
                scalar4: scalars[4].buffer, scalar4Offset: scalars[4].offset,
                value4: values[4].buffer, value4Offset: values[4].offset,
                scalar5: scalars[5].buffer, scalar5Offset: scalars[5].offset,
                value5: values[5].buffer, value5Offset: values[5].offset,
                scalar6: scalars[6].buffer, scalar6Offset: scalars[6].offset,
                value6: values[6].buffer, value6Offset: values[6].offset,
                scalar7: scalars[7].buffer, scalar7Offset: scalars[7].offset,
                value7: values[7].buffer, value7Offset: values[7].offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_scalar_fma_chain8_f16(
                scalar0: scalars[0].buffer, scalar0Offset: scalars[0].offset,
                value0: values[0].buffer, value0Offset: values[0].offset,
                scalar1: scalars[1].buffer, scalar1Offset: scalars[1].offset,
                value1: values[1].buffer, value1Offset: values[1].offset,
                scalar2: scalars[2].buffer, scalar2Offset: scalars[2].offset,
                value2: values[2].buffer, value2Offset: values[2].offset,
                scalar3: scalars[3].buffer, scalar3Offset: scalars[3].offset,
                value3: values[3].buffer, value3Offset: values[3].offset,
                scalar4: scalars[4].buffer, scalar4Offset: scalars[4].offset,
                value4: values[4].buffer, value4Offset: values[4].offset,
                scalar5: scalars[5].buffer, scalar5Offset: scalars[5].offset,
                value5: values[5].buffer, value5Offset: values[5].offset,
                scalar6: scalars[6].buffer, scalar6Offset: scalars[6].offset,
                value6: values[6].buffer, value6Offset: values[6].offset,
                scalar7: scalars[7].buffer, scalar7Offset: scalars[7].offset,
                value7: values[7].buffer, value7Offset: values[7].offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_scalar_fma_chain8_bf16(
                scalar0: scalars[0].buffer, scalar0Offset: scalars[0].offset,
                value0: values[0].buffer, value0Offset: values[0].offset,
                scalar1: scalars[1].buffer, scalar1Offset: scalars[1].offset,
                value1: values[1].buffer, value1Offset: values[1].offset,
                scalar2: scalars[2].buffer, scalar2Offset: scalars[2].offset,
                value2: values[2].buffer, value2Offset: values[2].offset,
                scalar3: scalars[3].buffer, scalar3Offset: scalars[3].offset,
                value3: values[3].buffer, value3Offset: values[3].offset,
                scalar4: scalars[4].buffer, scalar4Offset: scalars[4].offset,
                value4: values[4].buffer, value4Offset: values[4].offset,
                scalar5: scalars[5].buffer, scalar5Offset: scalars[5].offset,
                value5: values[5].buffer, value5Offset: values[5].offset,
                scalar6: scalars[6].buffer, scalar6Offset: scalars[6].offset,
                value6: values[6].buffer, value6Offset: values[6].offset,
                scalar7: scalars[7].buffer, scalar7Offset: scalars[7].offset,
                value7: values[7].buffer, value7Offset: values[7].offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.scalarFMAChain8: unsupported dtype \(dt)")
        }
    }

    /// Fused MoE phase 1b + phase 2 + phase 3 in ONE kernel launch via
    /// `ffai_moe_down_swiglu_accum_int4_chain8`.
    ///
    /// The GPU-router MoE path runs three back-to-back dispatches per
    /// layer:
    ///   1. `swigluMany`           — `inner[k][i] = silu(gate[k][i]) * up[k][i]`
    ///   2. `dequantGemvInt4ExpertIndexedMany` (down)
    ///                             — `out[k]     = W_down[expert_idx[k]] · inner[k]`
    ///   3. `scalarFMAChain8`      — `acc[i]     = Σ_k slot_weight[k] · out[k][i]`
    ///
    /// This wrapper collapses all three into one dispatch. Each
    /// threadgroup owns one output row of `[out_dim]`, iterates the 8
    /// slots sequentially: stages `inner` in 3 KiB threadgroup memory,
    /// runs the dequant-gemv inner-product against
    /// `W_down[expert_idx[k]]` for that slot, accumulates into a
    /// per-thread `acc` with the slot scalar baked in, then reduce-
    /// sums across threads at the end. Eliminates the `inner[k]` and
    /// `out[k]` DRAM roundtrips between phases.
    ///
    /// Kernel constraints (preconditioned):
    /// - `in_dim` (moeIntermediate) MUST be ≤ 768. The kernel's
    ///   TG-memory alloc is hardcoded at 768 floats (3 KiB).
    /// - `group_size` MUST be 64.
    /// - TPG = 128. Grid = `[out_dim · TPG, 1, 1]` (one TG per output
    ///   row).
    /// - `gates` / `ups` each contain 8 tensors of shape `[in_dim]` in
    ///   the model dtype.
    /// - `expertIndices` is `[8] u32`. `slotWeights` is `[8] T`.
    /// - `weightsStacked` is `[nExperts, out_dim, in_dim/8]` u32-packed.
    /// - `scalesStacked` / `biasesStacked` are
    ///   `[nExperts, out_dim, in_dim/group_size]` T.
    /// - `output` is `[out_dim]` T.
    public static func moeDownSwigluAccumInt4Chain8(
        gates: [Tensor],  // 8 tensors, each [moeIntermediate]
        ups: [Tensor],  // 8 tensors, each [moeIntermediate]
        expertIndices: Tensor,  // [8] u32
        slotWeights: Tensor,  // [8] T
        weightsStacked: Tensor,  // [nExperts, hidden, moeIntermediate/8] u32
        scalesStacked: Tensor,  // [nExperts, hidden, moeIntermediate/groupSize] T
        biasesStacked: Tensor,  // [nExperts, hidden, moeIntermediate/groupSize] T
        output: Tensor,  // [hidden] T
        inDim: Int,  // moeIntermediate
        outDim: Int,  // hidden
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            gates.count == 8 && ups.count == 8,
            "Ops.moeDownSwigluAccumInt4Chain8: k must be 8 (got \(gates.count) gates, \(ups.count) ups)")
        precondition(
            inDim <= 768,
            "Ops.moeDownSwigluAccumInt4Chain8: in_dim must be ≤ 768 (kernel TG-mem alloc), got \(inDim)")
        precondition(
            groupSize == 64,
            "Ops.moeDownSwigluAccumInt4Chain8: group_size must be 64, got \(groupSize)")
        precondition(
            expertIndices.dtype == .u32 && expertIndices.elementCount == 8,
            "Ops.moeDownSwigluAccumInt4Chain8: expert_indices must be [8] u32")
        precondition(
            slotWeights.dtype == output.dtype && slotWeights.elementCount == 8,
            "Ops.moeDownSwigluAccumInt4Chain8: slot_weights must be [8] matching output dtype")
        precondition(
            output.elementCount == outDim,
            "Ops.moeDownSwigluAccumInt4Chain8: output must be [out_dim]")
        let tpg = 128
        let grid = MTLSize(width: outDim * tpg, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch output.dtype {
        case .f32:
            MetalTileKernels.ffai_moe_down_swiglu_accum_int4_chain8_f32(
                gate_0: gates[0].buffer, gate_0Offset: gates[0].offset,
                up_0: ups[0].buffer, up_0Offset: ups[0].offset,
                gate_1: gates[1].buffer, gate_1Offset: gates[1].offset,
                up_1: ups[1].buffer, up_1Offset: ups[1].offset,
                gate_2: gates[2].buffer, gate_2Offset: gates[2].offset,
                up_2: ups[2].buffer, up_2Offset: ups[2].offset,
                gate_3: gates[3].buffer, gate_3Offset: gates[3].offset,
                up_3: ups[3].buffer, up_3Offset: ups[3].offset,
                gate_4: gates[4].buffer, gate_4Offset: gates[4].offset,
                up_4: ups[4].buffer, up_4Offset: ups[4].offset,
                gate_5: gates[5].buffer, gate_5Offset: gates[5].offset,
                up_5: ups[5].buffer, up_5Offset: ups[5].offset,
                gate_6: gates[6].buffer, gate_6Offset: gates[6].offset,
                up_6: ups[6].buffer, up_6Offset: ups[6].offset,
                gate_7: gates[7].buffer, gate_7Offset: gates[7].offset,
                up_7: ups[7].buffer, up_7Offset: ups[7].offset,
                expert_indices: expertIndices.buffer, expert_indicesOffset: expertIndices.offset,
                slot_weights: slotWeights.buffer, slot_weightsOffset: slotWeights.offset,
                weights_stacked: weightsStacked.buffer, weights_stackedOffset: weightsStacked.offset,
                scales_stacked: scalesStacked.buffer, scales_stackedOffset: scalesStacked.offset,
                biases_stacked: biasesStacked.buffer, biases_stackedOffset: biasesStacked.offset,
                output: output.buffer, outputOffset: output.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim),
                group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_moe_down_swiglu_accum_int4_chain8_f16(
                gate_0: gates[0].buffer, gate_0Offset: gates[0].offset,
                up_0: ups[0].buffer, up_0Offset: ups[0].offset,
                gate_1: gates[1].buffer, gate_1Offset: gates[1].offset,
                up_1: ups[1].buffer, up_1Offset: ups[1].offset,
                gate_2: gates[2].buffer, gate_2Offset: gates[2].offset,
                up_2: ups[2].buffer, up_2Offset: ups[2].offset,
                gate_3: gates[3].buffer, gate_3Offset: gates[3].offset,
                up_3: ups[3].buffer, up_3Offset: ups[3].offset,
                gate_4: gates[4].buffer, gate_4Offset: gates[4].offset,
                up_4: ups[4].buffer, up_4Offset: ups[4].offset,
                gate_5: gates[5].buffer, gate_5Offset: gates[5].offset,
                up_5: ups[5].buffer, up_5Offset: ups[5].offset,
                gate_6: gates[6].buffer, gate_6Offset: gates[6].offset,
                up_6: ups[6].buffer, up_6Offset: ups[6].offset,
                gate_7: gates[7].buffer, gate_7Offset: gates[7].offset,
                up_7: ups[7].buffer, up_7Offset: ups[7].offset,
                expert_indices: expertIndices.buffer, expert_indicesOffset: expertIndices.offset,
                slot_weights: slotWeights.buffer, slot_weightsOffset: slotWeights.offset,
                weights_stacked: weightsStacked.buffer, weights_stackedOffset: weightsStacked.offset,
                scales_stacked: scalesStacked.buffer, scales_stackedOffset: scalesStacked.offset,
                biases_stacked: biasesStacked.buffer, biases_stackedOffset: biasesStacked.offset,
                output: output.buffer, outputOffset: output.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim),
                group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_moe_down_swiglu_accum_int4_chain8_bf16(
                gate_0: gates[0].buffer, gate_0Offset: gates[0].offset,
                up_0: ups[0].buffer, up_0Offset: ups[0].offset,
                gate_1: gates[1].buffer, gate_1Offset: gates[1].offset,
                up_1: ups[1].buffer, up_1Offset: ups[1].offset,
                gate_2: gates[2].buffer, gate_2Offset: gates[2].offset,
                up_2: ups[2].buffer, up_2Offset: ups[2].offset,
                gate_3: gates[3].buffer, gate_3Offset: gates[3].offset,
                up_3: ups[3].buffer, up_3Offset: ups[3].offset,
                gate_4: gates[4].buffer, gate_4Offset: gates[4].offset,
                up_4: ups[4].buffer, up_4Offset: ups[4].offset,
                gate_5: gates[5].buffer, gate_5Offset: gates[5].offset,
                up_5: ups[5].buffer, up_5Offset: ups[5].offset,
                gate_6: gates[6].buffer, gate_6Offset: gates[6].offset,
                up_6: ups[6].buffer, up_6Offset: ups[6].offset,
                gate_7: gates[7].buffer, gate_7Offset: gates[7].offset,
                up_7: ups[7].buffer, up_7Offset: ups[7].offset,
                expert_indices: expertIndices.buffer, expert_indicesOffset: expertIndices.offset,
                slot_weights: slotWeights.buffer, slot_weightsOffset: slotWeights.offset,
                weights_stacked: weightsStacked.buffer, weights_stackedOffset: weightsStacked.offset,
                scales_stacked: scalesStacked.buffer, scales_stackedOffset: scalesStacked.offset,
                biases_stacked: biasesStacked.buffer, biases_stackedOffset: biasesStacked.offset,
                output: output.buffer, outputOffset: output.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim),
                group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeDownSwigluAccumInt4Chain8: unsupported dtype \(output.dtype)")
        }
    }

    /// `sigmoidScalarFMA` with an extra residual term folded in:
    /// `out[i] = residual[i] + base[i] + sigmoid(gate[0]) * value[i]`.
    /// Collapses the MoE post-FFN two-step chain
    /// `sigmoidScalarFMA(gate, sharedOut, routed) -> ffnOut` followed
    /// by `Ops.add(postMix, ffnOut)` into one kernel, saving a
    /// `[hidden]` DRAM roundtrip per MoE layer per decode token.
    public static func sigmoidScalarFMAResidual(
        gate: Tensor, value: Tensor, base: Tensor, residual: Tensor,
        into out: Tensor, on cmd: MTLCommandBuffer
    ) {
        precondition(
            gate.dtype == value.dtype && value.dtype == base.dtype
                && base.dtype == residual.dtype && residual.dtype == out.dtype,
            "Ops.sigmoidScalarFMAResidual: all tensors must share dtype")
        precondition(
            gate.elementCount == 1,
            "Ops.sigmoidScalarFMAResidual: gate must be [1] (got \(gate.elementCount))")
        precondition(
            value.elementCount == base.elementCount
                && base.elementCount == residual.elementCount
                && residual.elementCount == out.elementCount,
            "Ops.sigmoidScalarFMAResidual: value/base/residual/out must match elementCount")
        let n = value.elementCount
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch out.dtype {
        case .f32:
            MetalTileKernels.mt_sigmoid_scalar_fma_residual_f32(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                residual: residual.buffer, residualOffset: residual.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_sigmoid_scalar_fma_residual_f16(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                residual: residual.buffer, residualOffset: residual.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_sigmoid_scalar_fma_residual_bf16(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                residual: residual.buffer, residualOffset: residual.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sigmoidScalarFMAResidual: unsupported dtype \(out.dtype)")
        }
    }
}
