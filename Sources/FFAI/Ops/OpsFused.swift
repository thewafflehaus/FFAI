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
        precondition(x.shape == residual.shape,
                     "Ops.rmsNormResidual: x/residual shape mismatch")
        precondition(x.dtype == residual.dtype && residual.dtype == weight.dtype,
                     "Ops.rmsNormResidual: x/residual/weight dtype mismatch")
        precondition(epsBuf.dtype == .f32 && epsBuf.elementCount == 1,
                     "Ops.rmsNormResidual: epsBuf must be a single f32 element")
        precondition(!x.shape.isEmpty, "Ops.rmsNormResidual: x must be non-empty")
        let n = x.shape.last!
        let rows = x.elementCount / n
        precondition(weight.elementCount == n,
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
        precondition(z.dtype == weight.dtype,
                     "Ops.gatedRmsNorm: z/weight dtype must match")
        precondition(epsBuf.dtype == .f32 && epsBuf.elementCount == 1,
                     "Ops.gatedRmsNorm: epsBuf must be a single f32 element")
        precondition(!y.shape.isEmpty, "Ops.gatedRmsNorm: y must be non-empty")
        let n = y.shape.last!
        let rows = y.elementCount / n
        precondition(weight.elementCount == n,
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
        precondition(x.dtype == weight.dtype,
                     "Ops.rmsNormSmall: x/weight dtype mismatch")
        precondition(epsBuf.dtype == .f32 && epsBuf.elementCount == 1,
                     "Ops.rmsNormSmall: epsBuf must be a single f32 element")
        precondition(!x.shape.isEmpty, "Ops.rmsNormSmall: x must be non-empty")
        let n = x.shape.last!
        let rows = x.elementCount / n
        precondition(weight.elementCount == n,
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
}
