// Copyright 2026 Tom Turney (@TheTom)
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
// GGUF block-dequant Ops — Swift wrappers over the metaltile
// `ffai_gguf_dequant_*` kernel family.
//
// Each Op takes the GPU-resident split the loader produces (packed
// quants + per-block scales + LUT tables) and dispatches the matching
// per-dtype kernel. The input/output dtypes are fixed by the kernel
// family:
//
//   - `Tensor<u8>` / `Tensor<u32>` — packed quant bytes
//   - `Tensor<f32>`                — per-block fp16-converted scales
//   - `Tensor<u8>`                 — iq2xxs grid + ksigns tables
//   - `Tensor<T>`                  — output (T = f32 / f16 / bf16)

import Foundation
import Metal
import MetalTileSwift

extension Ops {
    /// Q8_0 — `out[i] = qs_signed[i] * scales[i/32]`. Block size 32.
    ///
    /// - Parameters:
    ///   - qsSigned: `[n_blocks * 32]` `u8` — int8 quants, sign-reconstructed
    ///     inside the kernel via `select(q >= 128, q-256, q)`.
    ///   - scales: `[n_blocks]` `f32` — host-extracted block super-scales
    ///     (fp16 → f32 at load time).
    ///   - outDtype: target output dtype. Allocates the result tensor.
    public static func ggufDequantQ8_0(
        qsSigned: Tensor, scales: Tensor, nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(qsSigned.dtype == .u8, "ggufDequantQ8_0: qsSigned must be u8")
        precondition(scales.dtype == .f32, "ggufDequantQ8_0: scales must be f32")
        precondition(nValues % 32 == 0, "ggufDequantQ8_0: nValues must be multiple of 32")
        let result = out ?? Tensor.empty(shape: [nValues], dtype: outDtype)
        let (grid, tg) = elementwiseGrid(nValues)
        let n = UInt32(nValues)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_gguf_dequant_q8_0_f32(
                qs_signed: qsSigned.buffer, qs_signedOffset: qsSigned.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_gguf_dequant_q8_0_f16(
                qs_signed: qsSigned.buffer, qs_signedOffset: qsSigned.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gguf_dequant_q8_0_bf16(
                qs_signed: qsSigned.buffer, qs_signedOffset: qsSigned.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("ggufDequantQ8_0: unsupported output dtype \(outDtype)")
        }
        return result
    }

    /// Q2_K — `out[i] = d * scale_4bit * q_2bit - dmin * min_4bit`.
    /// Block size 256, two-level scales.
    public static func ggufDequantQ2_K(
        qsPacked: Tensor, scales: Tensor, dF32: Tensor, dminF32: Tensor,
        nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(qsPacked.dtype == .u32, "ggufDequantQ2_K: qsPacked must be u32")
        precondition(scales.dtype == .u8, "ggufDequantQ2_K: scales must be u8")
        precondition(dF32.dtype == .f32, "ggufDequantQ2_K: d_f32 must be f32")
        precondition(dminF32.dtype == .f32, "ggufDequantQ2_K: dmin_f32 must be f32")
        precondition(nValues % 256 == 0, "ggufDequantQ2_K: nValues must be multiple of 256")
        let result = out ?? Tensor.empty(shape: [nValues], dtype: outDtype)
        let (grid, tg) = elementwiseGrid(nValues)
        let n = UInt32(nValues)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_gguf_dequant_q2_k_f32(
                qs_packed: qsPacked.buffer, qs_packedOffset: qsPacked.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                dmin_f32: dminF32.buffer, dmin_f32Offset: dminF32.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_gguf_dequant_q2_k_f16(
                qs_packed: qsPacked.buffer, qs_packedOffset: qsPacked.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                dmin_f32: dminF32.buffer, dmin_f32Offset: dminF32.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gguf_dequant_q2_k_bf16(
                qs_packed: qsPacked.buffer, qs_packedOffset: qsPacked.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                dmin_f32: dminF32.buffer, dmin_f32Offset: dminF32.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("ggufDequantQ2_K: unsupported output dtype \(outDtype)")
        }
        return result
    }

    /// IQ2_XXS — codebook lookup against `iq2xxs_grid[256][8]` modulated
    /// by the `ksigns_iq2xs[128]` sign-mask table. Block size 256.
    public static func ggufDequantIQ2_XXS(
        qsU32: Tensor, dF32: Tensor, grid: Tensor, signs: Tensor,
        nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(qsU32.dtype == .u32, "ggufDequantIQ2_XXS: qsU32 must be u32")
        precondition(dF32.dtype == .f32, "ggufDequantIQ2_XXS: d_f32 must be f32")
        precondition(grid.dtype == .u8, "ggufDequantIQ2_XXS: grid must be u8")
        precondition(signs.dtype == .u8, "ggufDequantIQ2_XXS: signs must be u8")
        precondition(
            grid.elementCount == 2048, "ggufDequantIQ2_XXS: grid must be 2048 bytes (256×8)")
        precondition(signs.elementCount == 128, "ggufDequantIQ2_XXS: signs must be 128 bytes")
        precondition(nValues % 256 == 0, "ggufDequantIQ2_XXS: nValues must be multiple of 256")
        let result = out ?? Tensor.empty(shape: [nValues], dtype: outDtype)
        let (gridDim, tg) = elementwiseGrid(nValues)
        let n = UInt32(nValues)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_gguf_dequant_iq2_xxs_f32(
                qs_u32: qsU32.buffer, qs_u32Offset: qsU32.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: gridDim, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_gguf_dequant_iq2_xxs_f16(
                qs_u32: qsU32.buffer, qs_u32Offset: qsU32.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: gridDim, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gguf_dequant_iq2_xxs_bf16(
                qs_u32: qsU32.buffer, qs_u32Offset: qsU32.offset,
                d_f32: dF32.buffer, d_f32Offset: dF32.offset,
                grid: grid.buffer, gridOffset: grid.offset,
                signs: signs.buffer, signsOffset: signs.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: gridDim, threadgroupSize: tg, on: cmd)
        default:
            fatalError("ggufDequantIQ2_XXS: unsupported output dtype \(outDtype)")
        }
        return result
    }
}
