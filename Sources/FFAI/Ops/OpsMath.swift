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
// OpsMath — element-wise math primitives + per-row reductions.
//
// Wraps the `mt_*` math kernels from `metaltile-std/src/mlx/unary.rs`,
// `binary.rs`, `reduce.rs`, `softmax.rs`, `logsumexp.rs`, `arg_reduce.rs`,
// `copy.rs`, `arange.rs`, and `strided.rs`.
//
// These are pure elementwise / per-row kernels. Most are Grid3D mode
// (`elementwiseGrid` does the right thing); a few are reductions with a
// row-per-threadgroup layout (`softmax`, `logsumexp`, `argmin`, the
// `row_reduce_*` family). Reductions hard-fix TPG = 256 so the contract
// becomes `n` must be a positive integer — same as `argmax` already
// shipped in `Ops.swift`.

import Foundation
import Metal
import MetalTileSwift

extension Ops {

    // ─── Binary elementwise: sub, div, pow, maxElem, minElem ─────────

    /// Element-wise `out[i] = a[i] - b[i]`. Wraps `mt_sub_*`.
    public static func sub(_ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
                           into out: Tensor? = nil) -> Tensor {
        precondition(a.shape == b.shape, "sub: shape mismatch \(a.shape) vs \(b.shape)")
        precondition(a.dtype == b.dtype, "sub: dtype mismatch")
        let result = out ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let (grid, tg) = elementwiseGrid(a.elementCount)
        switch a.dtype {
        case .f32:  MetalTileKernels.mt_sub_f32(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:  MetalTileKernels.mt_sub_f16(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16: MetalTileKernels.mt_sub_bf16(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        default: fatalError("Ops.sub: unsupported dtype \(a.dtype)")
        }
        return result
    }

    /// Element-wise `out[i] = a[i] / b[i]`. Wraps `mt_div_*`.
    public static func div(_ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
                           into out: Tensor? = nil) -> Tensor {
        precondition(a.shape == b.shape, "div: shape mismatch \(a.shape) vs \(b.shape)")
        precondition(a.dtype == b.dtype, "div: dtype mismatch")
        let result = out ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let (grid, tg) = elementwiseGrid(a.elementCount)
        switch a.dtype {
        case .f32:  MetalTileKernels.mt_div_f32(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:  MetalTileKernels.mt_div_f16(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16: MetalTileKernels.mt_div_bf16(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        default: fatalError("Ops.div: unsupported dtype \(a.dtype)")
        }
        return result
    }

    /// Element-wise `out[i] = a[i] ^ b[i]`. Wraps `mt_pow_*`.
    public static func pow(_ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
                           into out: Tensor? = nil) -> Tensor {
        precondition(a.shape == b.shape, "pow: shape mismatch")
        precondition(a.dtype == b.dtype, "pow: dtype mismatch")
        let result = out ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let (grid, tg) = elementwiseGrid(a.elementCount)
        switch a.dtype {
        case .f32:  MetalTileKernels.mt_pow_f32(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:  MetalTileKernels.mt_pow_f16(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16: MetalTileKernels.mt_pow_bf16(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        default: fatalError("Ops.pow: unsupported dtype \(a.dtype)")
        }
        return result
    }

    /// Element-wise `out[i] = max(a[i], b[i])`. Wraps `mt_max_elem_*`.
    /// Distinct from `Ops.argmax` / `mt_max_elem` per-tensor; this is
    /// per-element maximum across two tensors.
    public static func maxElem(_ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
                               into out: Tensor? = nil) -> Tensor {
        precondition(a.shape == b.shape, "maxElem: shape mismatch")
        precondition(a.dtype == b.dtype, "maxElem: dtype mismatch")
        let result = out ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let (grid, tg) = elementwiseGrid(a.elementCount)
        switch a.dtype {
        case .f32:  MetalTileKernels.mt_max_elem_f32(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:  MetalTileKernels.mt_max_elem_f16(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16: MetalTileKernels.mt_max_elem_bf16(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        default: fatalError("Ops.maxElem: unsupported dtype \(a.dtype)")
        }
        return result
    }

    /// Element-wise `out[i] = min(a[i], b[i])`. Wraps `mt_min_elem_*`.
    public static func minElem(_ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
                               into out: Tensor? = nil) -> Tensor {
        precondition(a.shape == b.shape, "minElem: shape mismatch")
        precondition(a.dtype == b.dtype, "minElem: dtype mismatch")
        let result = out ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let (grid, tg) = elementwiseGrid(a.elementCount)
        switch a.dtype {
        case .f32:  MetalTileKernels.mt_min_elem_f32(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:  MetalTileKernels.mt_min_elem_f16(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16: MetalTileKernels.mt_min_elem_bf16(a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset, out: result.buffer, outOffset: result.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        default: fatalError("Ops.minElem: unsupported dtype \(a.dtype)")
        }
        return result
    }

    // ─── Unary elementwise: neg, abs, exp, log, sqrt, square, recip ──

    /// Helper that runs a unary `(in, out)` elementwise kernel for the
    /// three float dtypes. Builds the standard `elementwiseGrid`.
    @inline(__always)
    private static func runUnary(
        _ x: Tensor, _ out: Tensor?,
        _ name: String,
        _ f32: (MTLBuffer, Int, MTLBuffer, Int, MTLSize, MTLSize, MTLCommandBuffer) -> Void,
        _ f16: (MTLBuffer, Int, MTLBuffer, Int, MTLSize, MTLSize, MTLCommandBuffer) -> Void,
        _ bf16: (MTLBuffer, Int, MTLBuffer, Int, MTLSize, MTLSize, MTLCommandBuffer) -> Void,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let (grid, tg) = elementwiseGrid(x.elementCount)
        switch x.dtype {
        case .f32:  f32(x.buffer, x.offset, result.buffer, result.offset, grid, tg, cmd)
        case .f16:  f16(x.buffer, x.offset, result.buffer, result.offset, grid, tg, cmd)
        case .bf16: bf16(x.buffer, x.offset, result.buffer, result.offset, grid, tg, cmd)
        default: fatalError("Ops.\(name): unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// Element-wise `out[i] = -x[i]`.
    public static func neg(_ x: Tensor, on cmd: MTLCommandBuffer, into out: Tensor? = nil) -> Tensor {
        runUnary(x, out, "neg",
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_neg_f32(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_neg_f16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_neg_bf16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            on: cmd)
    }

    /// Element-wise `out[i] = |x[i]|`.
    public static func abs(_ x: Tensor, on cmd: MTLCommandBuffer, into out: Tensor? = nil) -> Tensor {
        runUnary(x, out, "abs",
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_abs_f32(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_abs_f16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_abs_bf16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            on: cmd)
    }

    /// Element-wise `out[i] = exp(x[i])`.
    public static func exp(_ x: Tensor, on cmd: MTLCommandBuffer, into out: Tensor? = nil) -> Tensor {
        runUnary(x, out, "exp",
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_exp_f32(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_exp_f16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_exp_bf16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            on: cmd)
    }

    /// Element-wise `out[i] = log(x[i])`.
    public static func log(_ x: Tensor, on cmd: MTLCommandBuffer, into out: Tensor? = nil) -> Tensor {
        runUnary(x, out, "log",
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_log_f32(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_log_f16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_log_bf16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            on: cmd)
    }

    /// Element-wise `out[i] = sqrt(x[i])`.
    public static func sqrt(_ x: Tensor, on cmd: MTLCommandBuffer, into out: Tensor? = nil) -> Tensor {
        runUnary(x, out, "sqrt",
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_sqrt_f32(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_sqrt_f16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_sqrt_bf16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            on: cmd)
    }

    /// Element-wise `out[i] = x[i] * x[i]`.
    public static func square(_ x: Tensor, on cmd: MTLCommandBuffer, into out: Tensor? = nil) -> Tensor {
        runUnary(x, out, "square",
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_square_f32(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_square_f16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_square_bf16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            on: cmd)
    }

    /// Element-wise `out[i] = 1 / x[i]`.
    public static func recip(_ x: Tensor, on cmd: MTLCommandBuffer, into out: Tensor? = nil) -> Tensor {
        runUnary(x, out, "recip",
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_recip_f32(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_recip_f16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_recip_bf16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            on: cmd)
    }

    /// Element-wise `out[i] = floor(x[i])`.
    public static func floor(_ x: Tensor, on cmd: MTLCommandBuffer, into out: Tensor? = nil) -> Tensor {
        runUnary(x, out, "floor",
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_floor_f32(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_floor_f16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_floor_bf16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            on: cmd)
    }

    /// Element-wise `out[i] = ceil(x[i])`.
    public static func ceil(_ x: Tensor, on cmd: MTLCommandBuffer, into out: Tensor? = nil) -> Tensor {
        runUnary(x, out, "ceil",
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_ceil_f32(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_ceil_f16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_ceil_bf16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            on: cmd)
    }

    /// Element-wise `out[i] = round(x[i])` (banker's rounding via the
    /// Metal IEEE-754 `rint` semantic).
    public static func round(_ x: Tensor, on cmd: MTLCommandBuffer, into out: Tensor? = nil) -> Tensor {
        runUnary(x, out, "round",
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_round_f32(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_round_f16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            { a, ao, o, oo, g, t, c in MetalTileKernels.mt_round_bf16(a: a, aOffset: ao, out: o, outOffset: oo, gridSize: g, threadgroupSize: t, on: c) },
            on: cmd)
    }

    // ─── GPU copy (kernel) ───────────────────────────────────────────
    //
    // `Ops.copy` (in Ops.swift) is a blit copy. `copyKernel` exposes the
    // metaltile `mt_copy_*` kernel — same semantics, but it runs as a
    // compute kernel rather than a blit, which is occasionally useful
    // for kernel-fusion / command-encoder ordering reasons (a blit forces
    // an encoder boundary; a compute dispatch chains with neighbouring
    // compute work).

    public static func copyKernel(_ src: Tensor, into dst: Tensor,
                                  on cmd: MTLCommandBuffer) {
        precondition(src.elementCount == dst.elementCount,
                     "Ops.copyKernel: element-count mismatch")
        precondition(src.dtype == dst.dtype, "Ops.copyKernel: dtype mismatch")
        let (grid, tg) = elementwiseGrid(src.elementCount)
        switch src.dtype {
        case .f32:  MetalTileKernels.mt_copy_f32(a: src.buffer, aOffset: src.offset, out: dst.buffer, outOffset: dst.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:  MetalTileKernels.mt_copy_f16(a: src.buffer, aOffset: src.offset, out: dst.buffer, outOffset: dst.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16: MetalTileKernels.mt_copy_bf16(a: src.buffer, aOffset: src.offset, out: dst.buffer, outOffset: dst.offset, gridSize: grid, threadgroupSize: tg, on: cmd)
        default: fatalError("Ops.copyKernel: unsupported dtype \(src.dtype)")
        }
    }

    // ─── arange ──────────────────────────────────────────────────────
    //
    // `mt_arange_*` fills `out[i] = start + i * step` for `i ∈ [0, n)`.
    // The kernel takes `start` and `step` as 1-element scalar buffers
    // (not constexpr scalars) so the same PSO works across calls — see
    // `metaltile-std/src/mlx/arange.rs`.

    /// Fill `out` with the linspace `[start, start + step, …,
    /// start + (n-1)*step]`. `out.elementCount` is treated as `n`.
    public static func arange(start: Float, step: Float,
                              into out: Tensor, on cmd: MTLCommandBuffer) {
        let n = out.elementCount
        precondition(n > 0, "Ops.arange: out must be non-empty")
        // Scalar buffers are allocated as small 4-byte slabs. For f16 /
        // bf16 the kernel expects a half-width value; we cap to f32 and
        // route through the f32 variant for f32 outputs only — that's
        // the only variant currently used by production callers. f16 /
        // bf16 arange is rare; route to the matching kernel if we need
        // it later.
        let dev = Device.shared
        switch out.dtype {
        case .f32:
            let startBuf = dev.makeBuffer(length: 4)
            let stepBuf = dev.makeBuffer(length: 4)
            startBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee = start
            stepBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee = step
            let (grid, tg) = elementwiseGrid(n)
            MetalTileKernels.mt_arange_f32(
                out: out.buffer, outOffset: out.offset,
                start: startBuf, startOffset: 0,
                step: stepBuf, stepOffset: 0,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.arange: unsupported dtype \(out.dtype) (f32 only for now)")
        }
    }

    // ─── Per-row reductions: softmax, logsumexp, argmin ──────────────

    /// Numerically-stable softmax over the last dim of a row-major
    /// 2D `[rows, n]` tensor (or a 1D `[n]` tensor treated as `rows=1`).
    /// Wraps `mt_softmax_*`: one threadgroup per row, online max + sum
    /// reduction, then a second pass writes the normalised output. The
    /// vocab length `n` is a runtime constexpr — no per-vocab PSO churn.
    public static func softmax(_ x: Tensor, on cmd: MTLCommandBuffer,
                               into out: Tensor? = nil) -> Tensor {
        precondition(!x.shape.isEmpty, "Ops.softmax: x must be non-empty")
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.shape.last!
        let rows = x.elementCount / n
        precondition(n > 0 && rows > 0, "Ops.softmax: n>0 and rows>0")
        // Reduction-mode: one threadgroup per row, TPG = 256 (matches
        // the kernel's tested geometry — pure `lsize`-looped, no fixed
        // multiplicity required).
        let tgSize = 256
        let grid = MTLSize(width: rows * tgSize, height: 1, depth: 1)
        let tg = MTLSize(width: tgSize, height: 1, depth: 1)
        switch x.dtype {
        case .f32:  MetalTileKernels.mt_softmax_f32(inp: x.buffer, inpOffset: x.offset, out: result.buffer, outOffset: result.offset, n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:  MetalTileKernels.mt_softmax_f16(inp: x.buffer, inpOffset: x.offset, out: result.buffer, outOffset: result.offset, n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16: MetalTileKernels.mt_softmax_bf16(inp: x.buffer, inpOffset: x.offset, out: result.buffer, outOffset: result.offset, n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default: fatalError("Ops.softmax: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// `out[r] = log(sum_i exp(x[r, i]))`. One threadgroup per row,
    /// fp32 accumulation. Output element-count equals row count.
    public static func logsumexp(_ x: Tensor, on cmd: MTLCommandBuffer,
                                 into out: Tensor? = nil) -> Tensor {
        precondition(!x.shape.isEmpty, "Ops.logsumexp: x must be non-empty")
        let n = x.shape.last!
        let rows = x.elementCount / n
        let outShape = Array(x.shape.dropLast())
        let resultShape = outShape.isEmpty ? [1] : outShape
        let result = out ?? Tensor.empty(shape: resultShape, dtype: x.dtype)
        precondition(result.elementCount == rows,
                     "Ops.logsumexp: out element-count (\(result.elementCount)) must match row count \(rows)")
        let tgSize = 256
        let grid = MTLSize(width: rows * tgSize, height: 1, depth: 1)
        let tg = MTLSize(width: tgSize, height: 1, depth: 1)
        switch x.dtype {
        case .f32:  MetalTileKernels.mt_logsumexp_f32(inp: x.buffer, inpOffset: x.offset, out: result.buffer, outOffset: result.offset, n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:  MetalTileKernels.mt_logsumexp_f16(inp: x.buffer, inpOffset: x.offset, out: result.buffer, outOffset: result.offset, n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16: MetalTileKernels.mt_logsumexp_bf16(inp: x.buffer, inpOffset: x.offset, out: result.buffer, outOffset: result.offset, n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default: fatalError("Ops.logsumexp: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// `out[r] = argmin_i x[r, i]` as a u32 buffer.
    /// One threadgroup per row, TPG = 256.
    public static func argmin(_ x: Tensor, into out: Tensor,
                              on cmd: MTLCommandBuffer) {
        precondition(out.dtype == .u32, "Ops.argmin: out must be u32")
        precondition(!x.shape.isEmpty, "Ops.argmin: x must be non-empty")
        let n = x.shape.last!
        let rows = x.elementCount / n
        precondition(out.elementCount == rows,
                     "Ops.argmin: out element-count (\(out.elementCount)) must match row count \(rows)")
        let tgSize = 256
        let grid = MTLSize(width: rows * tgSize, height: 1, depth: 1)
        let tg = MTLSize(width: tgSize, height: 1, depth: 1)
        switch x.dtype {
        case .f32:  MetalTileKernels.mt_argmin_f32(inp: x.buffer, inpOffset: x.offset, out: out.buffer, outOffset: out.offset, n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:  MetalTileKernels.mt_argmin_f16(inp: x.buffer, inpOffset: x.offset, out: out.buffer, outOffset: out.offset, n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16: MetalTileKernels.mt_argmin_bf16(inp: x.buffer, inpOffset: x.offset, out: out.buffer, outOffset: out.offset, n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default: fatalError("Ops.argmin: unsupported dtype \(x.dtype)")
        }
    }
}
