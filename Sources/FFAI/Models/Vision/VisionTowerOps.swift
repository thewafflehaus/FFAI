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
// VisionTowerOps — module-internal helpers shared by every dynamic-
// resolution vision tower in the FFAI VLM family.
//
// Pre-extract, each Qwen 2-VL / Qwen 2.5-VL / Qwen 3-VL family file
// either redefined the same static helpers or called into
// `Qwen25VLVisionModel.<helper>` across the family boundary. Gemma 4-VL
// and Nemotron-VL did the same. Pixtral shipped its own private
// `addRowBias`. The vision-tower helpers are pure functions that
// implement weight-layout repacking, GEMM tile alignment padding, and
// bias broadcast — none of which depends on per-family architecture.
//
// This file centralizes them as module-internal top-level functions:
//   • `addRowBias(_:bias:nRows:rowSize:on:)` — broadcast a `[rowSize]`
//     bias into a `[nRows, rowSize]` tensor and add.
//   • `padLinearRows(_:toRows:device:)` — zero-extend a `Linear`'s
//     output rows up to a GEMM-tile-aligned row count (and its bias).
//   • `padLinearCols(_:toCols:device:)` (Linear overload) — zero-
//     extend a `Linear`'s input columns up to a tile-aligned column
//     count. Bias is unchanged.
//   • `padLinearColsTo(_:toCols:device:)` (Tensor overload) — same
//     operation on a raw weight tensor (used when the caller doesn't
//     own a `Linear` wrapper, e.g. Gemma 4-VL's patch-embed Conv2d
//     reshape path).
//   • `flattenPatchEmbed(_:hidden:patchDim:patchDimPadded:device:)` —
//     repack a 5-D Conv3d patch-embed weight into a 2-D GEMM weight
//     `[hidden, patchDimPadded]`. Detects MLX channel-last vs
//     PyTorch channel-first layout from the trailing-dim size. Used by
//     all three Qwen-VL families and any future dynamic-resolution
//     tower with a Conv3d patch-embed.
//   • `gemmKTileWidth` — the K-tile width every dynamic-resolution
//     tower aligns to so the patch-embed projection dispatches as a
//     single `Ops.gemm`. Shared constant so all callers stay aligned.
//
// All helpers are CPU-side weight-prep utilities — they touch tensors
// at load time, not in the hot forward path.

import Foundation
import Metal

// MARK: - Constants

/// The `Ops.gemm` K-tile width. Dynamic-resolution vision towers must
/// align their patch-embed input dim + SwiGLU intermediate dim to a
/// multiple of this so the projection dispatches as a single
/// `Ops.gemm`. 16 is the metaltile `mt_gemm` minimum K-tile.
let gemmKTileWidth: Int = 16

// MARK: - Bias broadcast

/// Broadcast-add a `[rowSize]` bias to each of `nRows` rows of a flat
/// `[nRows, rowSize]` tensor. CPU-side tile + one `Ops.add` dispatch —
/// not in the hot path, runs at most a few hundred ops per encode.
func addRowBias(
    _ x: Tensor, bias: Tensor, nRows: Int,
    rowSize: Int, on cmd: MTLCommandBuffer
) -> Tensor {
    let biasVals = bias.toFloatArray()
    var flat = [Float](repeating: 0, count: nRows * rowSize)
    for r in 0 ..< nRows {
        for c in 0 ..< rowSize { flat[r * rowSize + c] = biasVals[c] }
    }
    let tiled = Tensor.empty(shape: [nRows, rowSize], dtype: x.dtype)
    ImagePreprocessing.copyFloats(flat, into: tiled)
    return Ops.add(x, tiled, on: cmd)
}

// MARK: - Linear padding

/// Zero-extend a `Linear`'s output rows from `[outOld, inDim]` to
/// `[toRows, inDim]` (and its bias to `[toRows]`). The extra rows are
/// zero, so the extra outputs are zero — used to pad e.g. the SwiGLU
/// `gate`/`up` outputs up to the K-tile-aligned intermediate dim.
func padLinearRows(_ linear: Linear, toRows: Int, device: Device) -> Linear {
    let outOld = linear.weight.shape[0]
    let inDim = linear.weight.shape[1]
    if outOld == toRows { return linear }
    precondition(
        toRows >= outOld,
        "padLinearRows: target \(toRows) < \(outOld)")
    let src = linear.weight.toFloatArray()
    var dst = [Float](repeating: 0, count: toRows * inDim)
    for r in 0 ..< outOld {
        for c in 0 ..< inDim { dst[r * inDim + c] = src[r * inDim + c] }
    }
    let w = ImagePreprocessing.makeTensor(
        from: dst, shape: [toRows, inDim], dtype: linear.weight.dtype,
        device: device)
    var b: Tensor?
    if let bias = linear.bias {
        let bs = bias.toFloatArray()
        var bd = [Float](repeating: 0, count: toRows)
        for i in 0 ..< outOld { bd[i] = bs[i] }
        b = ImagePreprocessing.makeTensor(
            from: bd, shape: [toRows], dtype: bias.dtype, device: device)
    }
    return Linear(weight: w, bias: b)
}

/// Zero-extend a `Linear`'s input columns from `[outDim, inOld]` to
/// `[outDim, toCols]`. Bias is unchanged. The extra columns are zero,
/// so they contribute nothing to the dot product — used to pad e.g.
/// the SwiGLU `down` projection's `inDim` up to the K-tile-aligned
/// intermediate.
func padLinearCols(_ linear: Linear, toCols: Int, device: Device) -> Linear {
    let outDim = linear.weight.shape[0]
    let inOld = linear.weight.shape[1]
    if inOld == toCols { return linear }
    precondition(
        toCols >= inOld,
        "padLinearCols: target \(toCols) < \(inOld)")
    let src = linear.weight.toFloatArray()
    var dst = [Float](repeating: 0, count: outDim * toCols)
    for r in 0 ..< outDim {
        for c in 0 ..< inOld { dst[r * toCols + c] = src[r * inOld + c] }
    }
    let w = ImagePreprocessing.makeTensor(
        from: dst, shape: [outDim, toCols], dtype: linear.weight.dtype,
        device: device)
    return Linear(weight: w, bias: linear.bias)
}

/// Tensor-overload of `padLinearCols`. Same op (zero-extend input
/// columns) but operates on a raw weight tensor instead of a `Linear`
/// wrapper — used by Gemma 4-VL's patch-embed Conv2d reshape path that
/// doesn't carry a `Linear`. Returns a fresh tensor with the padded
/// columns.
func padLinearColsTo(_ w: Tensor, toCols: Int, device: Device) -> Tensor {
    let outDim = w.shape[0]
    let inOld = w.shape[1]
    if inOld == toCols { return w }
    precondition(
        toCols >= inOld,
        "padLinearColsTo: target \(toCols) < \(inOld)")
    let src = w.toFloatArray()
    var dst = [Float](repeating: 0, count: outDim * toCols)
    for r in 0 ..< outDim {
        for c in 0 ..< inOld { dst[r * toCols + c] = src[r * inOld + c] }
    }
    return ImagePreprocessing.makeTensor(
        from: dst, shape: [outDim, toCols], dtype: w.dtype, device: device)
}

// MARK: - Conv3d patch-embed reshape

/// Repack a 5-D Conv3d patch-embed weight into a 2-D GEMM weight
/// `[hidden, patchDimPadded]`. Each row of the output holds the
/// `tP × py × px × inCh` values for one output channel, in
/// `((t·inCh + ch)·p + py)·p + px` column order; the trailing
/// `patchDimPadded - patchDim` columns are zero-pad.
///
/// Detects layout from the trailing dim: small last dim (in-channels
/// ≤ 4) → MLX channel-last `[hidden, tP, py, px, inCh]`. Larger
/// trailing dim → PyTorch channel-first `[hidden, inCh, tP, py, px]`.
/// Used by every dynamic-resolution Qwen-VL vision tower (and any
/// future tower with a Conv3d patch-embed of the same shape).
func flattenPatchEmbed(
    _ w: Tensor, hidden: Int, patchDim: Int,
    patchDimPadded: Int, device: Device
) -> Tensor {
    precondition(
        w.shape.count == 5,
        "flattenPatchEmbed: patch-embed weight must be 5D Conv3d, "
            + "got \(w.shape)")
    let src = w.toFloatArray()
    // Zero-initialized — the pad columns stay zero.
    var dst = [Float](repeating: 0, count: hidden * patchDimPadded)
    // dst column order: (((t·inCh + ch)·p + py)·p + px).
    let mlxLayout = w.shape[4] <= 4  // trailing dim is in_channels
    if mlxLayout {
        // src `[hidden, tP, py, px, inCh]` — channel last.
        let tP = w.shape[1]
        let p = w.shape[2]
        let inCh = w.shape[4]
        for o in 0 ..< hidden {
            for t in 0 ..< tP {
                for py in 0 ..< p {
                    for px in 0 ..< p {
                        for ch in 0 ..< inCh {
                            let s = ((((o * tP + t) * p + py) * p + px) * inCh + ch)
                            let col = (((t * inCh + ch) * p + py) * p + px)
                            dst[o * patchDimPadded + col] = src[s]
                        }
                    }
                }
            }
        }
    } else {
        // src `[hidden, inCh, tP, py, px]` — PyTorch channel-first.
        let inCh = w.shape[1]
        let tP = w.shape[2]
        let p = w.shape[3]
        for o in 0 ..< hidden {
            for ch in 0 ..< inCh {
                for t in 0 ..< tP {
                    for py in 0 ..< p {
                        for px in 0 ..< p {
                            let s = ((((o * inCh + ch) * tP + t) * p + py) * p + px)
                            let col = (((t * inCh + ch) * p + py) * p + px)
                            dst[o * patchDimPadded + col] = src[s]
                        }
                    }
                }
            }
        }
    }
    return ImagePreprocessing.makeTensor(
        from: dst, shape: [hidden, patchDimPadded], dtype: w.dtype,
        device: device)
}
