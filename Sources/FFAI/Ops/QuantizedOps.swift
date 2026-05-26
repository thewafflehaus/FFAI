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
// QuantizedOps — general affine quantize / dequantize for arbitrary
// buffers.
//
// `Ops.quantizeKV*` / `Ops.bulkDequantKV*` (in `Ops.swift`) are the
// per-row KV-cache specialisations. The kernels here are the
// "weights / activations" siblings: they quantize an arbitrary `[N]`
// (or `[…, N]`) tensor in `group_size`-element groups and store the
// affine scale + bias per group. Round-trips through `quantizeAffine`
// + `dequantizeAffine` recover the input to within the quantization
// step.
//
// Wraps `mt_affine_quantize_int{2,4,8}_*` + `mt_affine_dequantize_int{2,4,8}_*`
// from `metaltile-std/src/mlx/quantized.rs`. Bits ∈ {3, 5, 6} use a
// bit-stream packing path and are documented in
// `OpsCoverageNotes.swift` — not yet wrapped (no production caller
// needs them at this layer; embedding / weight quant already routes
// through `Ops.dequantGemv` / `Ops.dequantGather`).
//
// ## Layout
//
// For `numel` input elements grouped by `group_size`:
//   * `n_groups = numel / group_size`
//   * `pack_factor = 32 / bits` (8 for int4, 4 for int8, 16 for int2)
//   * Packed weight shape: `[numel / pack_factor]` as `u32`
//   * `scales` / `biases`: `[n_groups]` in the input dtype.
//
// ## Dispatch (kernel invariants — see `mlx/quantized.rs`)
//
//   * Dequantize: Grid3D mode, one thread per packed `u32`. No reduction.
//   * Quantize: Reduction mode (simd_min / simd_max), TPG = 32 (one
//     simdgroup), grid = `[n_groups, 1, 1]`.

import Foundation
import Metal
import MetalTileSwift

public enum QuantizedOps {

    // ─── dequantizeAffine: packed u32 → fp32/16/bf16 ───────────────

    /// Affine-dequantize `w`'s packed `u32` entries into `out` for the
    /// given `bits` ∈ {2, 4, 8} and `group_size`. `out.elementCount`
    /// must equal `n_groups * group_size`; `w.elementCount` must equal
    /// `n_groups * (group_size / pack_factor)`.
    public static func dequantizeAffine(
        weight: Tensor, scales: Tensor, biases: Tensor,
        into out: Tensor,
        bits: Int, groupSize: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            weight.dtype == .u32,
            "QuantizedOps.dequantizeAffine: weight must be u32 (packed)")
        precondition(
            scales.dtype == out.dtype && biases.dtype == out.dtype,
            "QuantizedOps.dequantizeAffine: scales/biases dtype must match out")
        if let reason = QuantizedOpsValidation.validateAffineDequantize(
            numel: out.elementCount,
            packedCount: weight.elementCount,
            scalesCount: scales.elementCount,
            biasesCount: biases.elementCount,
            bits: bits, groupSize: groupSize
        ) {
            preconditionFailure("QuantizedOps.dequantizeAffine: \(reason)")
        }
        // Grid3D: one thread per packed word.
        let nPacks = weight.elementCount
        let tgSize = min(256, nPacks)
        let grid = MTLSize(width: nPacks, height: 1, depth: 1)
        let tg = MTLSize(width: tgSize, height: 1, depth: 1)
        let gs = UInt32(groupSize)
        switch (bits, out.dtype) {
        case (4, .f32):
            MetalTileKernels.mt_affine_dequantize_int4_f32(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .f16):
            MetalTileKernels.mt_affine_dequantize_int4_f16(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .bf16):
            MetalTileKernels.mt_affine_dequantize_int4_bf16(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f32):
            MetalTileKernels.mt_affine_dequantize_int8_f32(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f16):
            MetalTileKernels.mt_affine_dequantize_int8_f16(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .bf16):
            MetalTileKernels.mt_affine_dequantize_int8_bf16(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .f32):
            MetalTileKernels.mt_affine_dequantize_int2_f32(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .f16):
            MetalTileKernels.mt_affine_dequantize_int2_f16(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .bf16):
            MetalTileKernels.mt_affine_dequantize_int2_bf16(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "QuantizedOps.dequantizeAffine: unsupported (bits=\(bits), dtype=\(out.dtype)); int3/5/6 not wrapped at this layer (see OpsCoverageNotes.swift)"
            )
        }
    }

    // ─── quantizeAffine: fp32/16/bf16 → packed u32 ─────────────────

    /// Affine-quantize `weight` into packed `u32` `out` (+ per-group
    /// scale + bias outputs). bits ∈ {2, 4, 8}.
    public static func quantizeAffine(
        weight: Tensor,
        packed: Tensor, scales: Tensor, biases: Tensor,
        bits: Int, groupSize: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            packed.dtype == .u32,
            "QuantizedOps.quantizeAffine: packed must be u32")
        precondition(
            scales.dtype == weight.dtype && biases.dtype == weight.dtype,
            "QuantizedOps.quantizeAffine: scales/biases dtype must match input")
        if let reason = QuantizedOpsValidation.validateAffineQuantize(
            numel: weight.elementCount,
            packedCount: packed.elementCount,
            scalesCount: scales.elementCount,
            biasesCount: biases.elementCount,
            bits: bits, groupSize: groupSize
        ) {
            preconditionFailure("QuantizedOps.quantizeAffine: \(reason)")
        }
        // Reduction mode: TPG = 32, one threadgroup per group.
        let nGroups = weight.elementCount / groupSize
        let tgSize = 32
        let grid = MTLSize(width: nGroups * tgSize, height: 1, depth: 1)
        let tg = MTLSize(width: tgSize, height: 1, depth: 1)
        let gs = UInt32(groupSize)
        switch (bits, weight.dtype) {
        case (4, .f32):
            MetalTileKernels.mt_affine_quantize_int4_f32(
                w: weight.buffer, wOffset: weight.offset,
                out: packed.buffer, outOffset: packed.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .f16):
            MetalTileKernels.mt_affine_quantize_int4_f16(
                w: weight.buffer, wOffset: weight.offset,
                out: packed.buffer, outOffset: packed.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .bf16):
            MetalTileKernels.mt_affine_quantize_int4_bf16(
                w: weight.buffer, wOffset: weight.offset,
                out: packed.buffer, outOffset: packed.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f32):
            MetalTileKernels.mt_affine_quantize_int8_f32(
                w: weight.buffer, wOffset: weight.offset,
                out: packed.buffer, outOffset: packed.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f16):
            MetalTileKernels.mt_affine_quantize_int8_f16(
                w: weight.buffer, wOffset: weight.offset,
                out: packed.buffer, outOffset: packed.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .bf16):
            MetalTileKernels.mt_affine_quantize_int8_bf16(
                w: weight.buffer, wOffset: weight.offset,
                out: packed.buffer, outOffset: packed.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .f32):
            MetalTileKernels.mt_affine_quantize_int2_f32(
                w: weight.buffer, wOffset: weight.offset,
                out: packed.buffer, outOffset: packed.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .f16):
            MetalTileKernels.mt_affine_quantize_int2_f16(
                w: weight.buffer, wOffset: weight.offset,
                out: packed.buffer, outOffset: packed.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .bf16):
            MetalTileKernels.mt_affine_quantize_int2_bf16(
                w: weight.buffer, wOffset: weight.offset,
                out: packed.buffer, outOffset: packed.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                group_size: gs, gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "QuantizedOps.quantizeAffine: unsupported (bits=\(bits), dtype=\(weight.dtype)); int3/5/6 not wrapped at this layer (see OpsCoverageNotes.swift)"
            )
        }
    }
}
