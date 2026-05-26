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
// Ops — ergonomic Tensor-based dispatch over MetalTileKernels.
//
// Each op picks the right kernel for the input dtype, fills in default
// grid/threadgroup sizing, encodes on the supplied command buffer, and
// returns a fresh output Tensor (or writes into a caller-supplied one).
//
// Initial cut: only the kernels Llama needs. Adding more in follow-ups.

import Foundation
import Metal
import MetalTileSwift

public enum Ops {
    public static let device: Device = .shared

    // ─── Sizing helpers ──────────────────────────────────────────────

    /// Threadgroup width for elementwise kernels. Matches what we know
    /// PSO maxTotalThreadsPerThreadgroup will accept on M-series.
    public static let elementwiseTgSize = 256

    /// Internal because the per-Ops extension files (`OpsMath.swift`,
    /// `OpsLogits.swift`, `OpsFused.swift`) call it.
    static func elementwiseGrid(_ n: Int) -> (MTLSize, MTLSize) {
        let tg = MTLSize(width: min(elementwiseTgSize, n), height: 1, depth: 1)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        return (grid, tg)
    }

    // ─── GPU copy (blit, no kernel dispatch) ─────────────────────────

    /// GPU-side copy of `src` into `dst`. Encoded as an `MTLBlit`
    /// command on `cmd` — no compute kernel, no PSO dispatch overhead.
    /// Same dtype, same elementCount; layouts must match.
    ///
    /// Use when the caller already has `dst` allocated at a specific
    /// row inside a larger contiguous buffer and needs to deposit the
    /// op-produced `src` there (e.g. per-row GDN-layer write-back in
    /// batched-prefill, where the per-token decode returns a fresh
    /// tensor but the model's `[T, hidden]` running buffer needs that
    /// row updated). Prefer `Ops.add(into:)` when an arithmetic
    /// combine is what's wanted; reach for `copy` for the pure-move
    /// case.
    public static func copy(
        _ src: Tensor, into dst: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            src.elementCount == dst.elementCount,
            "Ops.copy: src/dst element-count mismatch (\(src.elementCount) vs \(dst.elementCount))")
        precondition(
            src.dtype == dst.dtype,
            "Ops.copy: src/dst dtype mismatch (\(src.dtype) vs \(dst.dtype))")
        let bytes = src.elementCount * src.dtype.byteSize
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(
            from: src.buffer, sourceOffset: src.offset,
            to: dst.buffer, destinationOffset: dst.offset,
            size: bytes)
        blit.endEncoding()
    }

    // ─── Element-wise binary: add ────────────────────────────────────

    public static func add(
        _ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(a.shape == b.shape, "add: shape mismatch \(a.shape) vs \(b.shape)")
        precondition(a.dtype == b.dtype, "add: dtype mismatch")
        let result = out ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let n = a.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch a.dtype {
        case .f32:
            MetalTileKernels.vector_add_f32(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: result.buffer, cOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.vector_add_f16(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: result.buffer, cOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.vector_add_bf16(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: result.buffer, cOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.add: unsupported dtype \(a.dtype)")
        }
        return result
    }

    public static func mul(
        _ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(a.shape == b.shape, "mul: shape mismatch")
        precondition(a.dtype == b.dtype, "mul: dtype mismatch")
        let result = out ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let n = a.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch a.dtype {
        case .f32:
            MetalTileKernels.mt_mul_f32(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_mul_f16(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_mul_bf16(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.mul: unsupported dtype \(a.dtype)")
        }
        return result
    }

    public static func silu(
        _ x: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch x.dtype {
        case .f32:
            MetalTileKernels.mt_silu_f32(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_silu_f16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_silu_bf16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.silu: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// Sigmoid: out[i] = 1 / (1 + exp(-x[i])). Wraps metaltile's
    /// `mt_sigmoid_*` element-wise kernel. Qwen3.5's gated attention
    /// output (`attn_output_gate`) multiplies the SDPA result by
    /// `sigmoid(gate)` before `o_proj`.
    public static func sigmoid(
        _ x: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch x.dtype {
        case .f32:
            MetalTileKernels.mt_sigmoid_f32(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_sigmoid_f16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_sigmoid_bf16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sigmoid: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// ReLU: out[i] = max(x[i], 0). Wraps metaltile's `mt_relu_*`
    /// element-wise kernel. NemotronH's MLP / MoE feed-forward blocks
    /// use squared-ReLU (`relu(x)^2`) as their activation; the squaring
    /// is a follow-up `Ops.mul(relu, relu)` on the caller side.
    public static func relu(
        _ x: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch x.dtype {
        case .f32:
            MetalTileKernels.mt_relu_f32(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_relu_f16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_relu_bf16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.relu: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// GELU (tanh-approximate, matching the math used by Gemma 3 / GPT-2
    /// / etc.). out[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/π) * (x[i] + 0.044715 * x[i]³))).
    /// Wraps metaltile's `mt_gelu_*` element-wise kernel.
    public static func gelu(
        _ x: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch x.dtype {
        case .f32:
            MetalTileKernels.mt_gelu_f32(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_gelu_f16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_gelu_bf16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gelu: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// Softplus: out[i] = log(1 + exp(x[i])). Numerically stable across
    /// the full input range. Used by Mamba 2's `dt = softplus(dt_raw +
    /// dt_bias)` per-head time-step computation.
    public static func softplus(
        _ x: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount
        let (grid, tg) = elementwiseGrid(n)
        switch x.dtype {
        case .f32:
            MetalTileKernels.mt_softplus_f32(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_softplus_f16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_softplus_bf16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.softplus: unsupported dtype \(x.dtype)")
        }
        return result
    }

    /// Embedding lookup. `table` is [vocab, dim], `tokenIds` is [n_tokens]
    /// (u32), output is [n_tokens, dim].
    public static func gather(
        table: Tensor, tokenIds: Tensor,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(table.shape.count == 2, "gather: table must be 2D")
        precondition(tokenIds.dtype == .u32, "gather: tokenIds must be u32")
        let dim = table.shape[1]
        let n = tokenIds.elementCount
        let result = out ?? Tensor.empty(shape: [n, dim], dtype: table.dtype)
        let totalThreads = n * dim
        let (grid, tg) = elementwiseGrid(totalThreads)
        switch table.dtype {
        case .f32:
            MetalTileKernels.ffai_gather_f32(
                table: table.buffer, tableOffset: table.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                dim: UInt32(dim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_gather_f16(
                table: table.buffer, tableOffset: table.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                dim: UInt32(dim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gather_bf16(
                table: table.buffer, tableOffset: table.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                dim: UInt32(dim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gather: unsupported dtype \(table.dtype)")
        }
        return result
    }

    /// Two embedding lookups against the SAME `table` on one compute
    /// encoder. Used when two parallel id streams need to read the
    /// same embedding (or, in Qwen3 attention, when a single linear
    /// projection result is split into two head-half slices via
    /// `ids = [0..nHeads]` + `ids = [nHeads..2·nHeads]`). The table
    /// binding + `dim` constant are set once and the per-call
    /// `(ids, out)` pair rotates per dispatch.
    public static func gatherTwo(
        table: Tensor,
        ids1: Tensor, into out1: Tensor,
        ids2: Tensor, into out2: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(table.shape.count == 2, "Ops.gatherTwo: table must be 2D")
        precondition(
            ids1.dtype == .u32 && ids2.dtype == .u32,
            "Ops.gatherTwo: ids must be u32")
        precondition(
            out1.dtype == table.dtype && out2.dtype == table.dtype,
            "Ops.gatherTwo: out dtype must match table")
        let dim = table.shape[1]
        let psoName: String
        switch table.dtype {
        case .f32: psoName = "ffai_gather_f32"
        case .f16: psoName = "ffai_gather_f16"
        case .bf16: psoName = "ffai_gather_bf16"
        default: fatalError("Ops.gatherTwo: unsupported dtype \(table.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        var dimV = UInt32(dim)
        enc.setBytes(&dimV, length: 4, index: 3)
        enc.setBuffer(table.buffer, offset: table.offset, index: 0)
        @inline(__always)
        func dispatch(_ ids: Tensor, _ out: Tensor) {
            let n = ids.elementCount
            let totalThreads = n * dim
            let (grid, tg) = elementwiseGrid(totalThreads)
            enc.setBuffer(ids.buffer, offset: ids.offset, index: 1)
            enc.setBuffer(out.buffer, offset: out.offset, index: 2)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(ids1, out1)
        dispatch(ids2, out2)
        enc.endEncoding()
    }

    /// Cooperative-thread matrix-vector multiply. weight: [out_dim, in_dim],
    /// input: [in_dim], output: [out_dim]. One threadgroup per output row;
    /// threads cooperate on the dot-product reduction.
    public static func gemv(
        weight: Tensor, input: Tensor,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.shape.count == 2, "gemv: weight must be 2D")
        precondition(input.shape.count == 1, "gemv: input must be 1D")
        precondition(
            weight.shape[1] == input.shape[0],
            "gemv: in_dim mismatch \(weight.shape[1]) vs \(input.shape[0])")
        precondition(weight.dtype == input.dtype, "gemv: dtype mismatch")
        let outDim = weight.shape[0]
        let inDim = weight.shape[1]
        if let reason = OpsValidation.validateGemv(outDim: outDim, inDim: inDim) {
            preconditionFailure("Ops.gemv: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: [outDim], dtype: weight.dtype)
        // Reduction kernel: one threadgroup per output row.
        // dispatchThreads dispatches outDim*tgWidth threads in groups
        // of tgWidth — yielding outDim threadgroups, each cooperating
        // over the in_dim axis.
        let tgWidth = 256
        let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch weight.dtype {
        case .f32:
            MetalTileKernels.mt_gemv_f32(
                mat: weight.buffer, matOffset: weight.offset,
                vec: input.buffer, vecOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                k: UInt32(inDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_gemv_f16(
                mat: weight.buffer, matOffset: weight.offset,
                vec: input.buffer, vecOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                k: UInt32(inDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_gemv_bf16(
                mat: weight.buffer, matOffset: weight.offset,
                vec: input.buffer, vecOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                k: UInt32(inDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gemv: unsupported dtype \(weight.dtype)")
        }
        return result
    }

    /// Multi-row GEMM — `out[r, :] = weight · input[r, :]` for a block
    /// of `nRows` rows in one dispatch. `weight` is `[outDim, inDim]`,
    /// `input` is `[nRows, inDim]`, output is `[nRows, outDim]`.
    ///
    /// `ffai_gemm` tiles the output 32×32 and stages weight + input
    /// tiles in threadgroup memory, so the weight is read once and
    /// reused across the block's rows — the projection-bandwidth win
    /// the diffusion / self-speculation block forward depends on.
    /// Reduction-mode kernel; the 1024-thread dispatch is hard.
    public static func gemm(
        weight: Tensor, input: Tensor, nRows: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.shape.count == 2, "Ops.gemm: weight must be 2D")
        let outDim = weight.shape[0]
        let inDim = weight.shape[1]
        if let reason = OpsValidation.validateGemm(inDim: inDim, outDim: outDim, nRows: nRows) {
            preconditionFailure("Ops.gemm: \(reason)")
        }
        precondition(
            input.elementCount == nRows * inDim,
            "Ops.gemm: input has \(input.elementCount) elements, expected "
                + "nRows*inDim = \(nRows * inDim)")
        precondition(weight.dtype == input.dtype, "Ops.gemm: weight/input dtype mismatch")
        let result = out ?? Tensor.empty(shape: [nRows, outDim], dtype: weight.dtype)
        // 32×32 output tiles, TPG = 1024. 1-D thread grid that Metal
        // slices into (outDim/32 ceil) × (nRows/32 ceil) threadgroups.
        let threadsPerGroup = 1024
        let nTiles = (outDim + 31) / 32
        let mTiles = (nRows + 31) / 32
        let grid = MTLSize(width: nTiles * threadsPerGroup, height: mTiles, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        switch weight.dtype {
        case .f32:
            MetalTileKernels.ffai_gemm_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                input: input.buffer, inputOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim), n_rows: UInt32(nRows),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_gemm_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                input: input.buffer, inputOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim), n_rows: UInt32(nRows),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gemm_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                input: input.buffer, inputOffset: input.offset,
                out: result.buffer, outOffset: result.offset,
                in_dim: UInt32(inDim), out_dim: UInt32(outDim), n_rows: UInt32(nRows),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gemm: unsupported dtype \(weight.dtype)")
        }
        return result
    }

    /// RMSNorm. x: [n], weight: [n], eps: scalar.
    /// Internally bound as a 1-element f32 buffer.
    /// Reduction kernel — one threadgroup per row.
    public static func rmsNorm(
        _ x: Tensor, weight: Tensor, eps: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(x.shape == weight.shape, "rmsNorm: weight/x shape mismatch")
        precondition(x.dtype == weight.dtype, "rmsNorm: dtype mismatch")
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount

        // Kernel-invariant validation. See OpsValidation.swift for the
        // full reasoning + a CI-runnable test of each precondition.
        if let reason = OpsValidation.validateRmsNorm(n: n) {
            preconditionFailure("Ops.rmsNorm: \(reason)")
        }
        dispatchRmsNorm(
            x: x, weight: weight, result: result,
            eps: eps, n: n, nRows: 1, on: cmd)
        return result
    }

    /// Multi-row RMSNorm. Input is [nRows, n]; weight is [n] (shared
    /// across all rows). Each row gets its own threadgroup. Used by
    /// Qwen3 to dispatch all per-head q_norm / k_norm in one call
    /// instead of one per head.
    public static func rmsNormRows(
        _ x: Tensor, weight: Tensor, eps: Float,
        nRows: Int, rowSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            x.elementCount == nRows * rowSize,
            "rmsNormRows: x size \(x.elementCount) ≠ nRows*rowSize")
        precondition(weight.elementCount == rowSize, "rmsNormRows: weight must be [rowSize]")
        precondition(x.dtype == weight.dtype, "rmsNormRows: dtype mismatch")
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)

        // Reduction kernel: one threadgroup per row. Per-row invariant
        // is the same as the single-row dispatch — see OpsValidation
        // for the full reasoning and CI-runnable tests.
        if let reason = OpsValidation.validateRmsNorm(n: rowSize) {
            preconditionFailure("Ops.rmsNormRows: \(reason)")
        }
        dispatchRmsNorm(
            x: x, weight: weight, result: result,
            eps: eps, n: rowSize, nRows: nRows, on: cmd)
        return result
    }

    /// Two `rmsNormRows` dispatches in ONE compute encoder. Used by
    /// the Qwen3 attention mixer's pre-RoPE Q-norm + K-norm pair: both
    /// run the fast `mt_rms_norm_*` kernel (TPG = rowSize / 4), share
    /// the same eps when set from the same model config, and there's
    /// no data dependency between them.
    ///
    /// `rowSize1` and `rowSize2` must both clear the fast-path bounds
    /// (`≤ 4096` and a multiple of 128 — checked via
    /// `OpsValidation.validateRmsNorm`). For wider rows the caller
    /// must fall back to two `rmsNormRows` calls so the wide kernel is
    /// picked individually.
    ///
    /// `eps` is forwarded via 1-element f32 buffers. If both eps
    /// values are equal we share a single alloc, otherwise two are
    /// allocated.
    public static func rmsNormRowsTwo(
        _ x1: Tensor, weight w1: Tensor, eps1: Float,
        nRows1: Int, rowSize1: Int, into out1: Tensor,
        _ x2: Tensor, weight w2: Tensor, eps2: Float,
        nRows2: Int, rowSize2: Int, into out2: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            x1.dtype == w1.dtype && x2.dtype == w2.dtype && x1.dtype == x2.dtype,
            "Ops.rmsNormRowsTwo: dtype mismatch")
        precondition(
            x1.elementCount == nRows1 * rowSize1,
            "Ops.rmsNormRowsTwo: x1 size \(x1.elementCount) ≠ nRows1·rowSize1")
        precondition(
            x2.elementCount == nRows2 * rowSize2,
            "Ops.rmsNormRowsTwo: x2 size \(x2.elementCount) ≠ nRows2·rowSize2")
        precondition(
            w1.elementCount == rowSize1 && w2.elementCount == rowSize2,
            "Ops.rmsNormRowsTwo: weight size must equal rowSize")
        precondition(
            out1.elementCount == x1.elementCount && out2.elementCount == x2.elementCount,
            "Ops.rmsNormRowsTwo: output element-count must match input")
        precondition(
            out1.dtype == x1.dtype && out2.dtype == x2.dtype,
            "Ops.rmsNormRowsTwo: output dtype must match input")
        if let reason = OpsValidation.validateRmsNorm(n: rowSize1) {
            preconditionFailure("Ops.rmsNormRowsTwo (#1): \(reason)")
        }
        if let reason = OpsValidation.validateRmsNorm(n: rowSize2) {
            preconditionFailure("Ops.rmsNormRowsTwo (#2): \(reason)")
        }
        let psoName: String
        switch x1.dtype {
        case .f32: psoName = "mt_rms_norm_f32"
        case .f16: psoName = "mt_rms_norm_f16"
        case .bf16: psoName = "mt_rms_norm_bf16"
        default: fatalError("Ops.rmsNormRowsTwo: unsupported dtype \(x1.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        // Allocate one or two eps buffers depending on whether the two
        // eps values match — RMSNorm pairs typically come from the
        // same checkpoint config and so share a single eps in practice.
        let epsBuf1: MTLBuffer = {
            let b = device.makeBuffer(length: 4)
            var v = eps1
            memcpy(b.contents(), &v, 4)
            return b
        }()
        let epsBuf2: MTLBuffer = {
            if eps1 == eps2 { return epsBuf1 }
            let b = device.makeBuffer(length: 4)
            var v = eps2
            memcpy(b.contents(), &v, 4)
            return b
        }()
        @inline(__always)
        func dispatch(
            _ x: Tensor, _ w: Tensor, _ out: Tensor,
            _ epsBuf: MTLBuffer, _ n: Int, _ nRows: Int
        ) {
            let tgWidth = n / 4
            let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            enc.setBuffer(x.buffer, offset: x.offset, index: 0)
            enc.setBuffer(w.buffer, offset: w.offset, index: 1)
            enc.setBuffer(out.buffer, offset: out.offset, index: 2)
            enc.setBuffer(epsBuf, offset: 0, index: 3)
            var nU = UInt32(n)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(x1, w1, out1, epsBuf1, rowSize1, nRows1)
        dispatch(x2, w2, out2, epsBuf2, rowSize2, nRows2)
        enc.endEncoding()
    }

    /// Fused (residual `a + b`) + RMSNorm. Returns **both** outputs:
    /// `residual` = `a + b` (the next layer's residual stream) and
    /// `normed` = `RMSNorm(a + b, weight)` (the input to the next
    /// sublayer). One dispatch instead of two — eliminates one round
    /// trip to GPU for the post-attention / post-FFN residual+norm
    /// pattern that every transformer block runs twice per token.
    ///
    /// Shape: `a`, `b` are `[rows, n]` (or any contiguous layout that
    /// flattens to `rows * n`); `weight` is `[n]`. Both outputs match
    /// `a`'s shape + dtype. `eps` is passed via a 1-element f32 buffer
    /// the wrapper allocates.
    ///
    /// **Row-width limit.** Kernel uses TPG = n / 4, so n must be ≤
    /// `OpsValidation.addRmsNormMaxRowSize` (4096) and a multiple of 4.
    /// Larger hidden sizes (Gemma 4 27B+ at 5376) MUST use separate
    /// `Ops.add` + `Ops.rmsNorm` calls. Pre-check with
    /// `OpsValidation.validateAddRmsNorm(n:)` if you don't know the
    /// model's hidden size statically.
    public static func addAndRmsNorm(
        _ a: Tensor, _ b: Tensor, weight: Tensor, eps: Float,
        nRows: Int, rowSize: Int, on cmd: MTLCommandBuffer,
        residualOut: Tensor? = nil, normedOut: Tensor? = nil
    ) -> (residual: Tensor, normed: Tensor) {
        precondition(
            a.elementCount == nRows * rowSize,
            "Ops.addAndRmsNorm: a size \(a.elementCount) ≠ nRows*rowSize")
        precondition(
            b.elementCount == nRows * rowSize,
            "Ops.addAndRmsNorm: b size \(b.elementCount) ≠ nRows*rowSize")
        precondition(
            weight.elementCount == rowSize,
            "Ops.addAndRmsNorm: weight must be [rowSize]")
        precondition(
            a.dtype == b.dtype && a.dtype == weight.dtype,
            "Ops.addAndRmsNorm: dtype mismatch")

        // Kernel-invariant validation. See OpsValidation.swift.
        if let reason = OpsValidation.validateAddRmsNorm(n: rowSize) {
            preconditionFailure("Ops.addAndRmsNorm: \(reason)")
        }

        let resid = residualOut ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let normed = normedOut ?? Tensor.empty(shape: a.shape, dtype: a.dtype)

        // eps as a 1-element f32 buffer.
        var epsValue = eps
        let epsBuf = device.makeBuffer(length: 4)
        memcpy(epsBuf.contents(), &epsValue, 4)

        // TPG = n / 4 (kernel vectorises in 4-elem chunks). One
        // threadgroup per row → grid = nRows * (n/4) threads × 1 × 1.
        // Validator above ensures n/4 ∈ [32, 1024].
        let tgWidth = rowSize / 4
        let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let nU = UInt32(rowSize)
        switch a.dtype {
        case .f32:
            MetalTileKernels.mt_add_rms_norm_f32(
                a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset,
                w: weight.buffer, wOffset: weight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                residual_out: resid.buffer, residual_outOffset: resid.offset,
                normed_out: normed.buffer, normed_outOffset: normed.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_add_rms_norm_f16(
                a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset,
                w: weight.buffer, wOffset: weight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                residual_out: resid.buffer, residual_outOffset: resid.offset,
                normed_out: normed.buffer, normed_outOffset: normed.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_add_rms_norm_bf16(
                a: a.buffer, aOffset: a.offset, b: b.buffer, bOffset: b.offset,
                w: weight.buffer, wOffset: weight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                residual_out: resid.buffer, residual_outOffset: resid.offset,
                normed_out: normed.buffer, normed_outOffset: normed.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.addAndRmsNorm: unsupported dtype \(a.dtype)")
        }
        return (resid, normed)
    }

    /// Shared RMSNorm dispatch for `rmsNorm` (nRows = 1) and
    /// `rmsNormRows`. Routes by row width:
    ///   • `n ≤ 4096` AND `n` is a multiple of 128 → `mt_rms_norm`,
    ///     4 elements per thread, TPG = n / 4 (the fast straight-line
    ///     kernel — every standard transformer hidden size).
    ///   • Anything else → `mt_rms_norm_wide`, whose strided loop covers
    ///     any width at a fixed TPG of 1024. Used by large-hidden text
    ///     models (Gemma 4 27B+ hidden 5376) AND by vision/audio towers
    ///     whose hidden dim isn't a 128-multiple (SmolVLM2 d=960 etc).
    /// One threadgroup per row in both cases.
    private static func dispatchRmsNorm(
        x: Tensor, weight: Tensor, result: Tensor,
        eps: Float, n: Int, nRows: Int, on cmd: MTLCommandBuffer
    ) {
        // eps as a 1-element f32 buffer.
        var epsValue = eps
        let epsBuf = device.makeBuffer(length: 4)
        memcpy(epsBuf.contents(), &epsValue, 4)

        // Fast kernel needs TPG = n/4 with TPG a multiple of 32 and
        // ≤ 1024. Anything outside that — too wide, too narrow, or not
        // a 128-multiple — goes through the always-correct wide kernel.
        let useWide = n > 4096 || !n.isMultiple(of: 128)
        let tgWidth = useWide ? 1024 : n / 4
        let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch (useWide, x.dtype) {
        case (false, .f32):
            MetalTileKernels.mt_rms_norm_f32(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (false, .f16):
            MetalTileKernels.mt_rms_norm_f16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (false, .bf16):
            MetalTileKernels.mt_rms_norm_bf16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (true, .f32):
            MetalTileKernels.mt_rms_norm_wide_f32(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (true, .f16):
            MetalTileKernels.mt_rms_norm_wide_f16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (true, .bf16):
            MetalTileKernels.mt_rms_norm_wide_bf16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rmsNorm: unsupported dtype \(x.dtype)")
        }
    }

    /// Llama-3-style RoPE with frequency-band scaling. Pass `scaleFactor=1`
    /// + `originalMaxPosition` very large to disable scaling.
    public struct RoPEScaling: Sendable {
        public var scaleFactor: Float
        public var lowFreqFactor: Float
        public var highFreqFactor: Float
        public var originalMaxPosition: Float

        public init(
            scaleFactor: Float = 1, lowFreqFactor: Float = 1,
            highFreqFactor: Float = 4,
            originalMaxPosition: Float = 1e9
        ) {
            self.scaleFactor = scaleFactor
            self.lowFreqFactor = lowFreqFactor
            self.highFreqFactor = highFreqFactor
            self.originalMaxPosition = originalMaxPosition
        }

        public static let none = RoPEScaling()
    }

    public static func rope(
        _ qk: Tensor, position: Int, headDim: Int,
        thetaBase: Float,
        scaling: RoPEScaling = .none,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(qk.elementCount % headDim == 0, "rope: qk size must be multiple of headDim")
        let nHeads = qk.elementCount / headDim
        let halfDim = headDim / 2
        let result = out ?? Tensor.empty(shape: qk.shape, dtype: qk.dtype)
        let grid = MTLSize(width: nHeads, height: halfDim, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch qk.dtype {
        case .f32:
            MetalTileKernels.ffai_rope_llama_f32(
                qk: qk.buffer, qkOffset: qk.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim),
                position: UInt32(position),
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_rope_llama_f16(
                qk: qk.buffer, qkOffset: qk.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim),
                position: UInt32(position),
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_rope_llama_bf16(
                qk: qk.buffer, qkOffset: qk.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim),
                position: UInt32(position),
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rope: unsupported dtype \(qk.dtype)")
        }
        return result
    }

    /// Partial-rotary RoPE. Rotates only the first `rotaryDim` elements
    /// of each `headDim`-strided head, leaving the remaining
    /// `headDim - rotaryDim` elements untouched. Qwen3.5 sets
    /// `partial_rotary_factor = 0.25` so a 256-dim head rotates only its
    /// first 64 dims.
    ///
    /// The `ffai_rope_llama_*` kernel takes the per-head stride
    /// (`head_dim`) and the rotate-half pairing offset / grid height
    /// (`half_dim`) as independent constants. Driving it with
    /// `head_dim = headDim` (true stride) but `half_dim = rotaryDim / 2`
    /// rotates the pairs `(i, i + rotaryDim/2)` for `i ∈ [0, rotaryDim/2)`
    /// inside each head — exactly the partial-rotary subset. Dims
    /// `[rotaryDim, headDim)` are never written, so this MUST run
    /// in-place (`out` aliasing `qk`) — the caller's buffer already holds
    /// the correct pass-through values for the unrotated tail.
    ///
    /// `rotaryDim` must be even and ≤ `headDim`; `qk` must be a flat
    /// `[nHeads * headDim]` tensor.
    public static func ropePartial(
        _ qk: Tensor, position: Int,
        headDim: Int, rotaryDim: Int,
        thetaBase: Float,
        scaling: RoPEScaling = .none,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            qk.elementCount % headDim == 0,
            "ropePartial: qk size must be a multiple of headDim")
        precondition(
            rotaryDim > 0 && rotaryDim <= headDim,
            "ropePartial: rotaryDim (\(rotaryDim)) must be in 1...headDim (\(headDim))")
        precondition(
            rotaryDim % 2 == 0,
            "ropePartial: rotaryDim (\(rotaryDim)) must be even (rotate-half pairs)")
        let nHeads = qk.elementCount / headDim
        let halfRotary = rotaryDim / 2
        // Grid: one thread per (head, rotary-pair). Writes in-place.
        let grid = MTLSize(width: nHeads, height: halfRotary, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch qk.dtype {
        case .f32:
            MetalTileKernels.ffai_rope_llama_f32(
                qk: qk.buffer, qkOffset: qk.offset,
                out: qk.buffer, outOffset: qk.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfRotary),
                position: UInt32(position),
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_rope_llama_f16(
                qk: qk.buffer, qkOffset: qk.offset,
                out: qk.buffer, outOffset: qk.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfRotary),
                position: UInt32(position),
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_rope_llama_bf16(
                qk: qk.buffer, qkOffset: qk.offset,
                out: qk.buffer, outOffset: qk.offset,
                head_dim: UInt32(headDim),
                half_dim: UInt32(halfRotary),
                position: UInt32(position),
                theta_base: thetaBase,
                scale_factor: scaling.scaleFactor,
                low_freq_factor: scaling.lowFreqFactor,
                high_freq_factor: scaling.highFreqFactor,
                original_max_position: scaling.originalMaxPosition,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.ropePartial: unsupported dtype \(qk.dtype)")
        }
    }

    /// Partial RoPE rotation on TWO tensors (typically Q + K) in ONE
    /// compute encoder. Both tensors share the same `(headDim,
    /// rotaryDim, position, thetaBase, scaling)` and dtype. The
    /// kernel writes in-place exactly like `ropePartial`, so passing
    /// `out == qk` is required.
    public static func ropePartialTwo(
        _ q: Tensor, _ k: Tensor, position: Int,
        headDim: Int, rotaryDim: Int,
        thetaBase: Float,
        scaling: RoPEScaling = .none,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            q.dtype == k.dtype,
            "Ops.ropePartialTwo: dtype mismatch")
        precondition(
            q.elementCount % headDim == 0 && k.elementCount % headDim == 0,
            "Ops.ropePartialTwo: sizes must be multiples of headDim")
        precondition(
            rotaryDim > 0 && rotaryDim <= headDim && rotaryDim % 2 == 0,
            "Ops.ropePartialTwo: rotaryDim (\(rotaryDim)) must be even and in 1...headDim")
        let psoName: String
        switch q.dtype {
        case .f32: psoName = "ffai_rope_llama_f32"
        case .f16: psoName = "ffai_rope_llama_f16"
        case .bf16: psoName = "ffai_rope_llama_bf16"
        default: fatalError("Ops.ropePartialTwo: unsupported dtype \(q.dtype)")
        }
        let halfRotary = rotaryDim / 2
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        // RoPE constants are shared across q and k — set ONCE.
        var hd = UInt32(headDim)
        var half = UInt32(halfRotary)
        var pos = UInt32(position)
        var theta = thetaBase
        var scaleFactor = scaling.scaleFactor
        var lowFreq = scaling.lowFreqFactor
        var highFreq = scaling.highFreqFactor
        var origMax = scaling.originalMaxPosition
        enc.setBytes(&hd, length: 4, index: 2)
        enc.setBytes(&half, length: 4, index: 3)
        enc.setBytes(&pos, length: 4, index: 4)
        enc.setBytes(&theta, length: 4, index: 5)
        enc.setBytes(&scaleFactor, length: 4, index: 6)
        enc.setBytes(&lowFreq, length: 4, index: 7)
        enc.setBytes(&highFreq, length: 4, index: 8)
        enc.setBytes(&origMax, length: 4, index: 9)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        @inline(__always)
        func dispatch(_ t: Tensor) {
            let nHeads = t.elementCount / headDim
            let grid = MTLSize(width: nHeads, height: halfRotary, depth: 1)
            enc.setBuffer(t.buffer, offset: t.offset, index: 0)
            enc.setBuffer(t.buffer, offset: t.offset, index: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(q)
        dispatch(k)
        enc.endEncoding()
    }

    /// Batched per-row RoPE: applies position-dependent rotation to T
    /// rows of `qk` in ONE dispatch. `positions` is a `[T]` u32 buffer
    /// (per-row physical slot index). In-place rotation. Saves T-1
    /// encoder begin/end pairs versus a per-row `ropePartial` T-loop;
    /// at Qwen3.6-A3B prefill T=512 × 10 attn layers that's ≈ 5100
    /// fewer dispatches per prefill call.
    public static func ropePartialMany(
        _ qk: Tensor, positions: Tensor,
        t: Int, nHeads: Int,
        headDim: Int, rotaryDim: Int,
        rowStride: Int,
        thetaBase: Float,
        scaling: RoPEScaling = .none,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            positions.dtype == .u32,
            "Ops.ropePartialMany: positions must be .u32")
        precondition(
            positions.elementCount == t,
            "Ops.ropePartialMany: positions count must equal T")
        precondition(
            qk.elementCount == t * rowStride,
            "Ops.ropePartialMany: qk size \(qk.elementCount) ≠ T·rowStride = \(t * rowStride)")
        precondition(
            rotaryDim > 0 && rotaryDim <= headDim && rotaryDim % 2 == 0,
            "Ops.ropePartialMany: rotaryDim must be even and in 1...headDim")
        let halfRotary = rotaryDim / 2
        let psoName: String
        switch qk.dtype {
        case .f32: psoName = "ffai_rope_llama_many_f32"
        case .f16: psoName = "ffai_rope_llama_many_f16"
        case .bf16: psoName = "ffai_rope_llama_many_bf16"
        default: fatalError("Ops.ropePartialMany: unsupported dtype \(qk.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(qk.buffer, offset: qk.offset, index: 0)
        enc.setBuffer(positions.buffer, offset: positions.offset, index: 1)
        enc.setBuffer(qk.buffer, offset: qk.offset, index: 2)
        var hd = UInt32(headDim)
        var half = UInt32(halfRotary)
        var stride = UInt32(rowStride)
        var theta = thetaBase
        var sf = scaling.scaleFactor
        var lf = scaling.lowFreqFactor
        var hf = scaling.highFreqFactor
        var om = scaling.originalMaxPosition
        enc.setBytes(&hd, length: 4, index: 3)
        enc.setBytes(&half, length: 4, index: 4)
        enc.setBytes(&stride, length: 4, index: 5)
        enc.setBytes(&theta, length: 4, index: 6)
        enc.setBytes(&sf, length: 4, index: 7)
        enc.setBytes(&lf, length: 4, index: 8)
        enc.setBytes(&hf, length: 4, index: 9)
        enc.setBytes(&om, length: 4, index: 10)
        let grid = MTLSize(width: t, height: nHeads, depth: halfRotary)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    /// Pair of `ropePartialMany` calls (Q + K) sharing ONE compute
    /// encoder. Both buffers consume the same `positions`, `headDim`,
    /// `rotaryDim`, `thetaBase`, and `scaling`; only the per-buffer
    /// `nHeads` × `rowStride` differs.
    public static func ropePartialManyTwo(
        q: Tensor, qNHeads: Int, qRowStride: Int,
        k: Tensor, kNHeads: Int, kRowStride: Int,
        positions: Tensor, t: Int,
        headDim: Int, rotaryDim: Int,
        thetaBase: Float,
        scaling: RoPEScaling = .none,
        on cmd: MTLCommandBuffer
    ) {
        precondition(q.dtype == k.dtype, "Ops.ropePartialManyTwo: dtype mismatch")
        precondition(
            positions.dtype == .u32,
            "Ops.ropePartialManyTwo: positions must be .u32")
        precondition(
            positions.elementCount == t,
            "Ops.ropePartialManyTwo: positions count must equal T")
        precondition(
            rotaryDim > 0 && rotaryDim <= headDim && rotaryDim % 2 == 0,
            "Ops.ropePartialManyTwo: rotaryDim must be even and in 1...headDim")
        let halfRotary = rotaryDim / 2
        let psoName: String
        switch q.dtype {
        case .f32: psoName = "ffai_rope_llama_many_f32"
        case .f16: psoName = "ffai_rope_llama_many_f16"
        case .bf16: psoName = "ffai_rope_llama_many_bf16"
        default: fatalError("Ops.ropePartialManyTwo: unsupported dtype \(q.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        // Shared bindings: positions + constants set ONCE.
        enc.setBuffer(positions.buffer, offset: positions.offset, index: 1)
        var hd = UInt32(headDim)
        var half = UInt32(halfRotary)
        var theta = thetaBase
        var sf = scaling.scaleFactor
        var lf = scaling.lowFreqFactor
        var hf = scaling.highFreqFactor
        var om = scaling.originalMaxPosition
        enc.setBytes(&hd, length: 4, index: 3)
        enc.setBytes(&half, length: 4, index: 4)
        enc.setBytes(&theta, length: 4, index: 6)
        enc.setBytes(&sf, length: 4, index: 7)
        enc.setBytes(&lf, length: 4, index: 8)
        enc.setBytes(&hf, length: 4, index: 9)
        enc.setBytes(&om, length: 4, index: 10)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        @inline(__always)
        func dispatch(_ buf: Tensor, _ nHeads: Int, _ rowStride: Int) {
            var stride = UInt32(rowStride)
            enc.setBuffer(buf.buffer, offset: buf.offset, index: 0)
            enc.setBuffer(buf.buffer, offset: buf.offset, index: 2)
            enc.setBytes(&stride, length: 4, index: 5)
            let grid = MTLSize(width: t, height: nHeads, depth: halfRotary)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(q, qNHeads, qRowStride)
        dispatch(k, kNHeads, kRowStride)
        enc.endEncoding()
    }

    /// YaRN RoPE parameters. `low` / `high` are the correction-range
    /// bounds — precomputed via `RoPEYaRN.from(...)` since they need a
    /// `floor`/`ceil`/`ln` computation that is constant across the
    /// dispatch. `factor == 1` collapses YaRN to plain RoPE.
    public struct RoPEYaRN: Sendable {
        public var factor: Float
        public var low: Float
        public var high: Float
        public var attnFactor: Float

        public init(factor: Float, low: Float, high: Float, attnFactor: Float = 1) {
            self.factor = factor
            self.low = low
            self.high = high
            self.attnFactor = attnFactor
        }

        /// Plain RoPE — `factor == 1` makes interpolation == extrapolation,
        /// so the ramp blend is a no-op.
        public static let plain = RoPEYaRN(factor: 1, low: 0, high: 1, attnFactor: 1)

        /// Build YaRN parameters from a checkpoint's `rope_parameters`
        /// block. Computes the correction-range bounds (`low` / `high`)
        /// from `beta_fast` / `beta_slow` and the YaRN mscale attention
        /// factor from `mscale` / `mscale_all_dim`.
        public static func from(
            headDim: Int, thetaBase: Float, factor: Float,
            betaFast: Float, betaSlow: Float,
            originalMaxPosition: Float,
            mscale: Float = 1, mscaleAllDim: Float = 1
        ) -> RoPEYaRN {
            // find_correction_dim — the dimension index at which a given
            // number of rotations occurs over the original context.
            func correctionDim(_ numRotations: Float) -> Float {
                (Float(headDim) * Foundation.log(originalMaxPosition / (numRotations * 2 * .pi)))
                    / (2 * Foundation.log(thetaBase))
            }
            var low = correctionDim(betaFast).rounded(.down)
            var high = correctionDim(betaSlow).rounded(.up)
            low = max(low, 0)
            high = min(high, Float(headDim - 1))
            if high <= low { high = low + 0.001 }  // avoid a zero-width ramp

            // YaRN mscale attention factor. When mscale == mscale_all_dim
            // (the common case) the ratio is exactly 1.
            func yarnMscale(_ scale: Float, _ m: Float) -> Float {
                scale <= 1 ? 1 : 0.1 * m * Foundation.log(scale) + 1
            }
            let attnFactor = yarnMscale(factor, mscale) / yarnMscale(factor, mscaleAllDim)
            return RoPEYaRN(factor: factor, low: low, high: high, attnFactor: attnFactor)
        }
    }

    /// YaRN RoPE — context-extended rotary embedding. Same Grid3D
    /// dispatch shape as `rope`: one threadgroup per (head, half-dim
    /// index). `factor == 1` reproduces plain RoPE bit-for-bit.
    public static func ropeYaRN(
        _ qk: Tensor, position: Int, headDim: Int,
        thetaBase: Float, yarn: RoPEYaRN,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            qk.elementCount % headDim == 0,
            "ropeYaRN: qk size must be a multiple of headDim")
        let nHeads = qk.elementCount / headDim
        let halfDim = headDim / 2
        let result = out ?? Tensor.empty(shape: qk.shape, dtype: qk.dtype)
        let grid = MTLSize(width: nHeads, height: halfDim, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch qk.dtype {
        case .f32:
            MetalTileKernels.ffai_rope_yarn_f32(
                qk: qk.buffer, qkOffset: qk.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), half_dim: UInt32(halfDim),
                position: UInt32(position), theta_base: thetaBase,
                factor: yarn.factor, low: yarn.low, high: yarn.high,
                attn_factor: yarn.attnFactor,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_rope_yarn_f16(
                qk: qk.buffer, qkOffset: qk.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), half_dim: UInt32(halfDim),
                position: UInt32(position), theta_base: thetaBase,
                factor: yarn.factor, low: yarn.low, high: yarn.high,
                attn_factor: yarn.attnFactor,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_rope_yarn_bf16(
                qk: qk.buffer, qkOffset: qk.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), half_dim: UInt32(halfDim),
                position: UInt32(position), theta_base: thetaBase,
                factor: yarn.factor, low: yarn.low, high: yarn.high,
                attn_factor: yarn.attnFactor,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.ropeYaRN: unsupported dtype \(qk.dtype)")
        }
        return result
    }

    /// MLX-format dequantizing gather (embedding lookup). bits ∈ {4, 8}.
    public static func dequantGather(
        weight: Tensor, scales: Tensor, biases: Tensor,
        tokenIds: Tensor, hidden: Int, bits: Int, groupSize: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.dtype == .u32, "dequantGather: weight must be u32 packed")
        precondition(tokenIds.dtype == .u32, "dequantGather: tokenIds must be u32")
        precondition(scales.dtype == biases.dtype, "dequantGather: scales/biases dtype mismatch")
        // Kernel-invariant validation (bit-width + silent-miscompute
        // footgun: partial trailing group). See
        // OpsValidation.validateDequantGather.
        if let reason = OpsValidation.validateDequantGather(
            hidden: hidden, bits: bits, groupSize: groupSize
        ) {
            preconditionFailure("Ops.dequantGather: \(reason)")
        }
        let n = tokenIds.elementCount
        let result = out ?? Tensor.empty(shape: [n, hidden], dtype: scales.dtype)
        let totalThreads = n * hidden
        let (grid, tg) = elementwiseGrid(totalThreads)
        let hiddenU = UInt32(hidden)
        let groupSizeU = UInt32(groupSize)
        switch (bits, scales.dtype) {
        case (4, .f32):
            MetalTileKernels.dequant_gather_int4_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .f16):
            MetalTileKernels.dequant_gather_int4_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .bf16):
            MetalTileKernels.dequant_gather_int4_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f32):
            MetalTileKernels.dequant_gather_int8_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f16):
            MetalTileKernels.dequant_gather_int8_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .bf16):
            MetalTileKernels.dequant_gather_int8_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .f32):
            MetalTileKernels.dequant_gather_int6_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .f16):
            MetalTileKernels.dequant_gather_int6_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .bf16):
            MetalTileKernels.dequant_gather_int6_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f32):
            MetalTileKernels.dequant_gather_int3_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f16):
            MetalTileKernels.dequant_gather_int3_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .bf16):
            MetalTileKernels.dequant_gather_int3_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .f32):
            MetalTileKernels.dequant_gather_int5_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .f16):
            MetalTileKernels.dequant_gather_int5_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .bf16):
            MetalTileKernels.dequant_gather_int5_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                hidden: hiddenU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.dequantGather: unsupported (bits=\(bits), dtype=\(scales.dtype))")
        }
        return result
    }

    /// Backwards-compatible 4-bit alias.
    public static func dequantGatherInt4(
        weight: Tensor, scales: Tensor, biases: Tensor,
        tokenIds: Tensor, hidden: Int, groupSize: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        dequantGather(
            weight: weight, scales: scales, biases: biases,
            tokenIds: tokenIds, hidden: hidden, bits: 4,
            groupSize: groupSize, on: cmd, into: out)
    }

    /// MLX-format dequantizing GEMV. Weight is packed uint32; either 8
    /// 4-bit values per word (`bits == 4`) or 4 8-bit values per word
    /// (`bits == 8`). Per `groupSize` stripe of the in_dim axis, a
    /// per-row (scale, bias) pair dequantizes via `w_real = q*scale + bias`.
    /// Fused with the gemv accumulator — full weight matrix never
    /// materialized.
    public static func dequantGemv(
        weight: Tensor, scales: Tensor, biases: Tensor,
        input: Tensor, bits: Int, groupSize: Int = 64,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.shape.count == 2, "dequantGemv: weight must be 2D")
        precondition(weight.dtype == .u32, "dequantGemv: weight must be u32 (packed)")
        precondition(
            scales.dtype == input.dtype && biases.dtype == input.dtype,
            "dequantGemv: scales/biases dtype must match input")
        let outDim = weight.shape[0]
        let packedPerRow = weight.shape[1]
        // Storage layout: bytes per row = in_dim * bits / 8.
        // packedPerRow uint32 = (in_dim * bits / 8) / 4 bytes, so:
        //   in_dim = packedPerRow * 32 / bits
        let inDim = packedPerRow * 32 / bits
        precondition(
            input.elementCount == inDim,
            "dequantGemv: input \(input.elementCount) ≠ in_dim \(inDim)")
        // Kernel-invariant validation (silent-miscompute footguns:
        // partial trailing group, unaligned pack tail, undersized
        // scales/biases). See OpsValidation.validateDequantGemv.
        if let reason = OpsValidation.validateDequantGemv(
            outDim: outDim, inDim: inDim, bits: bits, groupSize: groupSize,
            scalesCount: scales.elementCount, biasesCount: biases.elementCount
        ) {
            preconditionFailure("Ops.dequantGemv: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: [outDim], dtype: input.dtype)
        // Reduction kernel: one threadgroup per output row.
        let tgWidth = 256
        let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let inDimU = UInt32(inDim)
        let groupSizeU = UInt32(groupSize)

        switch (bits, input.dtype) {
        case (4, .f32):
            MetalTileKernels.dequant_gemv_int4_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .f16):
            MetalTileKernels.dequant_gemv_int4_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .bf16):
            MetalTileKernels.dequant_gemv_int4_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f32):
            MetalTileKernels.dequant_gemv_int8_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f16):
            MetalTileKernels.dequant_gemv_int8_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .bf16):
            MetalTileKernels.dequant_gemv_int8_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .f32):
            MetalTileKernels.dequant_gemv_int6_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .f16):
            MetalTileKernels.dequant_gemv_int6_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (6, .bf16):
            MetalTileKernels.dequant_gemv_int6_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f32):
            MetalTileKernels.dequant_gemv_int3_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f16):
            MetalTileKernels.dequant_gemv_int3_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .bf16):
            MetalTileKernels.dequant_gemv_int3_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .f32):
            MetalTileKernels.dequant_gemv_int5_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .f16):
            MetalTileKernels.dequant_gemv_int5_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (5, .bf16):
            MetalTileKernels.dequant_gemv_int5_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: inDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.dequantGemv: unsupported (bits=\(bits), dtype=\(input.dtype))")
        }
        return result
    }

    /// Backwards-compatible 4-bit alias.
    public static func dequantGemvInt4(
        weight: Tensor, scales: Tensor, biases: Tensor,
        input: Tensor, groupSize: Int = 64,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        dequantGemv(
            weight: weight, scales: scales, biases: biases,
            input: input, bits: 4, groupSize: groupSize,
            on: cmd, into: out)
    }

    /// Batched int4 dequantGemv on TWO projections sharing one input
    /// in ONE compute encoder. Used by `Qwen35MoEFFN.forward` for the
    /// per-expert gate + up pair: both projections read the same
    /// `[hidden]` post-norm activation, so we set the `input` binding
    /// once and rotate `(weight, scales, biases, output)` per dispatch.
    /// Saves one encoder begin/end pair per call.
    ///
    /// All outputs must share dtype with `input`; `groupSize` and the
    /// 4-bit packing are identical across the two projections.
    public static func dequantGemvInt4Two(
        input: Tensor,
        w0: Tensor, s0: Tensor, b0: Tensor, out0: Tensor,
        w1: Tensor, s1: Tensor, b1: Tensor, out1: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            input.dtype == out0.dtype && input.dtype == out1.dtype,
            "Ops.dequantGemvInt4Two: dtype mismatch")
        let psoName: String
        switch input.dtype {
        case .f32: psoName = "dequant_gemv_int4_f32"
        case .f16: psoName = "dequant_gemv_int4_f16"
        case .bf16: psoName = "dequant_gemv_int4_bf16"
        default: fatalError("Ops.dequantGemvInt4Two: unsupported dtype \(input.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 3)
        // `inDim` derives from the packed-row width: 4-bit packs 8
        // weights per u32 word and `weight.shape[1]` counts words.
        let packedPerRow = w0.shape[1]
        let inDim = packedPerRow * 32 / 4
        var inDimV = UInt32(inDim)
        var groupSizeV = UInt32(groupSize)
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&groupSizeV, length: 4, index: 6)
        let tgWidth = 256
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        @inline(__always)
        func dispatch(_ w: Tensor, _ s: Tensor, _ b: Tensor, _ out: Tensor) {
            enc.setBuffer(w.buffer, offset: w.offset, index: 0)
            enc.setBuffer(s.buffer, offset: s.offset, index: 1)
            enc.setBuffer(b.buffer, offset: b.offset, index: 2)
            enc.setBuffer(out.buffer, offset: out.offset, index: 4)
            let outDim = w.shape[0]
            let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(w0, s0, b0, out0)
        dispatch(w1, s1, b1, out1)
        enc.endEncoding()
    }

    /// Batched int4 dequantGemv on THREE projections sharing one input
    /// in ONE compute encoder. Used by the Qwen3.5/3.6 attention mixer
    /// for the q/k/v projection triplet.
    public static func dequantGemvInt4Three(
        input: Tensor,
        w0: Tensor, s0: Tensor, b0: Tensor, out0: Tensor,
        w1: Tensor, s1: Tensor, b1: Tensor, out1: Tensor,
        w2: Tensor, s2: Tensor, b2: Tensor, out2: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            input.dtype == out0.dtype,
            "Ops.dequantGemvInt4Three: dtype mismatch")
        let psoName: String
        switch input.dtype {
        case .f32: psoName = "dequant_gemv_int4_f32"
        case .f16: psoName = "dequant_gemv_int4_f16"
        case .bf16: psoName = "dequant_gemv_int4_bf16"
        default: fatalError("Ops.dequantGemvInt4Three: unsupported dtype \(input.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 3)
        let packedPerRow = w0.shape[1]
        let inDim = packedPerRow * 32 / 4
        var inDimV = UInt32(inDim)
        var groupSizeV = UInt32(groupSize)
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&groupSizeV, length: 4, index: 6)
        let tgWidth = 256
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        @inline(__always)
        func dispatch(_ w: Tensor, _ s: Tensor, _ b: Tensor, _ out: Tensor) {
            enc.setBuffer(w.buffer, offset: w.offset, index: 0)
            enc.setBuffer(s.buffer, offset: s.offset, index: 1)
            enc.setBuffer(b.buffer, offset: b.offset, index: 2)
            enc.setBuffer(out.buffer, offset: out.offset, index: 4)
            let outDim = w.shape[0]
            let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(w0, s0, b0, out0)
        dispatch(w1, s1, b1, out1)
        dispatch(w2, s2, b2, out2)
        enc.endEncoding()
    }

    /// Batched int4 dequantGemv on FOUR projections sharing one input
    /// in ONE compute encoder. Used by the Qwen3.5/3.6 GDN mixer where
    /// the four input projections (qkv, z, b, a) all read the same
    /// xNorm output.
    public static func dequantGemvInt4Four(
        input: Tensor,
        w0: Tensor, s0: Tensor, b0: Tensor, out0: Tensor,
        w1: Tensor, s1: Tensor, b1: Tensor, out1: Tensor,
        w2: Tensor, s2: Tensor, b2: Tensor, out2: Tensor,
        w3: Tensor, s3: Tensor, b3: Tensor, out3: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            input.dtype == out0.dtype,
            "Ops.dequantGemvInt4Four: dtype mismatch")
        let psoName: String
        switch input.dtype {
        case .f32: psoName = "dequant_gemv_int4_f32"
        case .f16: psoName = "dequant_gemv_int4_f16"
        case .bf16: psoName = "dequant_gemv_int4_bf16"
        default: fatalError("Ops.dequantGemvInt4Four: unsupported dtype \(input.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(input.buffer, offset: input.offset, index: 3)
        let packedPerRow = w0.shape[1]
        let inDim = packedPerRow * 32 / 4
        var inDimV = UInt32(inDim)
        var groupSizeV = UInt32(groupSize)
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&groupSizeV, length: 4, index: 6)
        let tgWidth = 256
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        @inline(__always)
        func dispatch(_ w: Tensor, _ s: Tensor, _ b: Tensor, _ out: Tensor) {
            enc.setBuffer(w.buffer, offset: w.offset, index: 0)
            enc.setBuffer(s.buffer, offset: s.offset, index: 1)
            enc.setBuffer(b.buffer, offset: b.offset, index: 2)
            enc.setBuffer(out.buffer, offset: out.offset, index: 4)
            let outDim = w.shape[0]
            let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(w0, s0, b0, out0)
        dispatch(w1, s1, b1, out1)
        dispatch(w2, s2, b2, out2)
        dispatch(w3, s3, b3, out3)
        enc.endEncoding()
    }

    /// Batched int4 dequant-GEMV on N projections with DIFFERENT
    /// inputs sharing ONE compute encoder. Unlike `dequantGemvInt4Two`
    /// / `Three` / `Four` (which all share a single input), the per-
    /// expert MoE down phase has N independent activations — one per
    /// chosen expert — but all projections share PSO, `inDim`, and
    /// `groupSize`. The wrapper sets the constexprs once and rotates
    /// `(weight, scales, biases, input, output)` per dispatch. Saves
    /// N-1 encoder begin/end pairs versus N independent
    /// `dequantGemvInt4` calls.
    public static func dequantGemvInt4Many(
        weights: [Tensor], scales: [Tensor], biases: [Tensor],
        inputs: [Tensor], outputs: [Tensor],
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        let n = weights.count
        precondition(
            scales.count == n && biases.count == n && inputs.count == n && outputs.count == n,
            "Ops.dequantGemvInt4Many: count mismatch")
        guard n > 0 else { return }
        let dtype = inputs[0].dtype
        let psoName: String
        switch dtype {
        case .f32: psoName = "dequant_gemv_int4_f32"
        case .f16: psoName = "dequant_gemv_int4_f16"
        case .bf16: psoName = "dequant_gemv_int4_bf16"
        default: fatalError("Ops.dequantGemvInt4Many: unsupported dtype \(dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        let packedPerRow = weights[0].shape[1]
        let inDim = packedPerRow * 32 / 4  // bits = 4 → 8 weights / u32
        var inDimV = UInt32(inDim)
        var groupSizeV = UInt32(groupSize)
        enc.setBytes(&inDimV, length: 4, index: 5)
        enc.setBytes(&groupSizeV, length: 4, index: 6)
        let tgWidth = 256
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        for i in 0 ..< n {
            precondition(
                weights[i].shape[1] == packedPerRow,
                "Ops.dequantGemvInt4Many: inDim varies at index \(i)")
            precondition(
                inputs[i].dtype == dtype && outputs[i].dtype == dtype,
                "Ops.dequantGemvInt4Many: dtype varies at index \(i)")
            enc.setBuffer(weights[i].buffer, offset: weights[i].offset, index: 0)
            enc.setBuffer(scales[i].buffer, offset: scales[i].offset, index: 1)
            enc.setBuffer(biases[i].buffer, offset: biases[i].offset, index: 2)
            enc.setBuffer(inputs[i].buffer, offset: inputs[i].offset, index: 3)
            enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 4)
            let outDim = weights[i].shape[0]
            let grid = MTLSize(width: outDim * tgWidth, height: 1, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        enc.endEncoding()
    }

    /// Fused Q/K/V int4 dequant-GEMV in ONE dispatch via
    /// `ffai_batched_qkv_qgemv_fast`. Replaces the 3-dispatch
    /// `dequantGemvInt4Three` shared-encoder form: the matrix is
    /// selected by `program_id<2>()` across a `[ceil(maxOut/8), 1, 3]`
    /// threadgroup grid, so all 3 projections share a single kernel
    /// launch and read `x` once into TG memory instead of three times
    /// from DRAM. Saves 2 encoder begin/end pairs per attention layer
    /// at decode T=1.
    ///
    /// Output layout: q, k, v concatenated in that order in `out`;
    /// caller slices into the three regions.
    ///
    /// Kernel constraints (mirror `ffai_rms_norm_qgemv_fast`):
    /// - `in_dim` MUST be a multiple of 512.
    /// - Each of `out_q`, `out_k`, `out_v` MUST be a multiple of 8.
    /// - `group_size` MUST be 64.
    /// - TPG = 64.
    public static func batchedQkvQgemvInt4Fast(
        x: Tensor,
        wQ: Tensor, scalesQ: Tensor, biasesQ: Tensor,
        wK: Tensor, scalesK: Tensor, biasesK: Tensor,
        wV: Tensor, scalesV: Tensor, biasesV: Tensor,
        outQ: Int, outK: Int, outV: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            wQ.dtype == .u32 && wK.dtype == .u32 && wV.dtype == .u32,
            "Ops.batchedQkvQgemvInt4Fast: w_* must be u32-packed")
        let packedPerRow = wQ.shape[1]
        let inDim = packedPerRow * 8
        precondition(
            x.elementCount == inDim,
            "Ops.batchedQkvQgemvInt4Fast: x.elementCount \(x.elementCount) ≠ inDim \(inDim)")
        precondition(
            out.elementCount == outQ + outK + outV,
            "Ops.batchedQkvQgemvInt4Fast: out.elementCount must be q+k+v")
        precondition(
            inDim % 512 == 0,
            "Ops.batchedQkvQgemvInt4Fast: in_dim must be a multiple of 512")
        precondition(
            outQ % 8 == 0 && outK % 8 == 0 && outV % 8 == 0,
            "Ops.batchedQkvQgemvInt4Fast: out_q/k/v must each be a multiple of 8")
        let groupSize = 64
        let maxOut = max(outQ, max(outK, outV))
        // Grid: [ceil(maxOut/8) * TPG, 1, 3] — TG count per matrix axis
        // times TPG=64 lanes; 3 matrices.
        let tpg = 64
        let nTiles = maxOut / 8
        let grid = MTLSize(width: nTiles * tpg, height: 1, depth: 3)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.ffai_batched_qkv_qgemv_fast_f32(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                out: out.buffer, outOffset: out.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_batched_qkv_qgemv_fast_f16(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                out: out.buffer, outOffset: out.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_batched_qkv_qgemv_fast_bf16(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                out: out.buffer, outOffset: out.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.batchedQkvQgemvInt4Fast: unsupported dtype \(x.dtype)")
        }
    }

    /// M>1 (batched-prefill) sibling of `batchedQkvQgemvInt4Fast`. Reads
    /// `x = [M, in_dim]` once into TG memory and produces all 3
    /// projections into THREE separate contiguous output tensors
    /// (`qBuf [M, out_q]`, `kBuf [M, out_k]`, `vBuf [M, out_v]`). The
    /// split-output layout lets downstream Q/K/V code read each
    /// projection as `[M, dim]` without strided views (vs. a single
    /// concat buffer). One dispatch replaces three independent qmm
    /// dispatches; the `xNorm` DRAM roundtrip is paid once instead of
    /// three times.
    ///
    /// Constraints (mirror `ffai_batched_qkv_qmm_fast`):
    /// - `in_dim` MUST be a multiple of 512.
    /// - Each of `out_q`, `out_k`, `out_v` MUST be a multiple of 8.
    /// - `group_size` MUST be 64. TPG = 64.
    /// - All three outputs MUST be pre-zeroed (kernel only writes
    ///   valid `row0 < out_*` tiles per matrix branch). The wrapper
    ///   handles this — callers do not need to zero ahead of time.
    public static func batchedQkvQmmFast(
        x: Tensor,  // [M, in_dim]
        wQ: Tensor, scalesQ: Tensor, biasesQ: Tensor,
        wK: Tensor, scalesK: Tensor, biasesK: Tensor,
        wV: Tensor, scalesV: Tensor, biasesV: Tensor,
        m: Int, outQ: Int, outK: Int, outV: Int,
        on cmd: MTLCommandBuffer,
        qBuf: Tensor,  // [M, out_q]
        kBuf: Tensor,  // [M, out_k]
        vBuf: Tensor  // [M, out_v]
    ) {
        precondition(
            wQ.dtype == .u32 && wK.dtype == .u32 && wV.dtype == .u32,
            "Ops.batchedQkvQmmFast: w_* must be u32-packed")
        let packedPerRow = wQ.shape[1]
        let inDim = packedPerRow * 8
        precondition(
            x.elementCount == m * inDim,
            "Ops.batchedQkvQmmFast: x.elementCount \(x.elementCount) ≠ M·inDim \(m * inDim)")
        precondition(
            qBuf.elementCount == m * outQ,
            "Ops.batchedQkvQmmFast: qBuf.elementCount must be M·out_q")
        precondition(
            kBuf.elementCount == m * outK,
            "Ops.batchedQkvQmmFast: kBuf.elementCount must be M·out_k")
        precondition(
            vBuf.elementCount == m * outV,
            "Ops.batchedQkvQmmFast: vBuf.elementCount must be M·out_v")
        precondition(
            inDim % 512 == 0,
            "Ops.batchedQkvQmmFast: in_dim must be a multiple of 512")
        precondition(
            outQ % 8 == 0 && outK % 8 == 0 && outV % 8 == 0,
            "Ops.batchedQkvQmmFast: out_q/k/v must each be a multiple of 8")
        // Pre-zero outputs: the kernel tile mechanic depends on the
        // buffer starting at zero where the smaller two matrices' tiles
        // past their `out_*` count no-op the store.
        qBuf.zero()
        kBuf.zero()
        vBuf.zero()
        let groupSize = 64
        let tpg = 64
        let maxOut = max(outQ, max(outK, outV))
        let nTiles = maxOut / 8
        let grid = MTLSize(width: nTiles * tpg, height: m, depth: 3)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.ffai_batched_qkv_qmm_fast_f32(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                q_buf: qBuf.buffer, q_bufOffset: qBuf.offset,
                k_buf: kBuf.buffer, k_bufOffset: kBuf.offset,
                v_buf: vBuf.buffer, v_bufOffset: vBuf.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_batched_qkv_qmm_fast_f16(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                q_buf: qBuf.buffer, q_bufOffset: qBuf.offset,
                k_buf: kBuf.buffer, k_bufOffset: kBuf.offset,
                v_buf: vBuf.buffer, v_bufOffset: vBuf.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_batched_qkv_qmm_fast_bf16(
                x: x.buffer, xOffset: x.offset,
                w_q: wQ.buffer, w_qOffset: wQ.offset,
                scales_q: scalesQ.buffer, scales_qOffset: scalesQ.offset,
                biases_q: biasesQ.buffer, biases_qOffset: biasesQ.offset,
                w_k: wK.buffer, w_kOffset: wK.offset,
                scales_k: scalesK.buffer, scales_kOffset: scalesK.offset,
                biases_k: biasesK.buffer, biases_kOffset: biasesK.offset,
                w_v: wV.buffer, w_vOffset: wV.offset,
                scales_v: scalesV.buffer, scales_vOffset: scalesV.offset,
                biases_v: biasesV.buffer, biases_vOffset: biasesV.offset,
                q_buf: qBuf.buffer, q_bufOffset: qBuf.offset,
                k_buf: kBuf.buffer, k_bufOffset: kBuf.offset,
                v_buf: vBuf.buffer, v_bufOffset: vBuf.offset,
                out_q: UInt32(outQ), out_k: UInt32(outK), out_v: UInt32(outV),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.batchedQkvQmmFast: unsupported dtype \(x.dtype)")
        }
    }

    /// Fused 4-output int4 dequant-GEMV in ONE dispatch via
    /// `ffai_batched_4_qgemv_fast`. Extends the QKV (3-output) pattern
    /// to 4 projections sharing the same input. Used by the Qwen3.5
    /// GDN mixer where the four input projections (qkv, z, b, a) all
    /// read the same `xNorm` and can be fused into a single kernel
    /// launch. Replaces the 4-dispatch shared-encoder form
    /// (`Ops.dequantGemvInt4Four`) with one launch, paying the input
    /// DRAM roundtrip once instead of four times.
    ///
    /// Constraints (mirror `ffai_batched_qkv_qgemv_fast`):
    /// - `in_dim` MUST be a multiple of 512.
    /// - Each of `out_a / out_b / out_c / out_d` MUST be a multiple
    ///   of 8.
    /// - `group_size` MUST be 64.
    /// - TPG = 64. Grid: `[ceil(max(out_*) / 8) * TPG, 1, 4]`.
    public static func batched4QgemvInt4Fast(
        input x: Tensor,
        wA: Tensor, scalesA: Tensor, biasesA: Tensor, outA: Tensor,
        wB: Tensor, scalesB: Tensor, biasesB: Tensor, outB: Tensor,
        wC: Tensor, scalesC: Tensor, biasesC: Tensor, outC: Tensor,
        wD: Tensor, scalesD: Tensor, biasesD: Tensor, outD: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            wA.dtype == .u32 && wB.dtype == .u32 && wC.dtype == .u32 && wD.dtype == .u32,
            "Ops.batched4QgemvInt4Fast: w_* must be u32-packed")
        let packedPerRow = wA.shape[1]
        let inDim = packedPerRow * 8
        precondition(
            x.elementCount == inDim,
            "Ops.batched4QgemvInt4Fast: x.elementCount \(x.elementCount) ≠ inDim \(inDim)")
        let outA_ = outA.elementCount
        let outB_ = outB.elementCount
        let outC_ = outC.elementCount
        let outD_ = outD.elementCount
        precondition(
            inDim % 512 == 0,
            "Ops.batched4QgemvInt4Fast: in_dim must be a multiple of 512")
        precondition(
            outA_ % 8 == 0 && outB_ % 8 == 0 && outC_ % 8 == 0 && outD_ % 8 == 0,
            "Ops.batched4QgemvInt4Fast: each out_* must be a multiple of 8")
        let tpg = 64
        let maxOut = max(max(outA_, outB_), max(outC_, outD_))
        let nTiles = maxOut / 8
        let grid = MTLSize(width: nTiles * tpg, height: 1, depth: 4)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.ffai_batched_4_qgemv_fast_f32(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_out: outA.buffer, a_outOffset: outA.offset,
                b_out: outB.buffer, b_outOffset: outB.offset,
                c_out: outC.buffer, c_outOffset: outC.offset,
                d_out: outD.buffer, d_outOffset: outD.offset,
                out_a: UInt32(outA_), out_b: UInt32(outB_),
                out_c: UInt32(outC_), out_d: UInt32(outD_),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_batched_4_qgemv_fast_f16(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_out: outA.buffer, a_outOffset: outA.offset,
                b_out: outB.buffer, b_outOffset: outB.offset,
                c_out: outC.buffer, c_outOffset: outC.offset,
                d_out: outD.buffer, d_outOffset: outD.offset,
                out_a: UInt32(outA_), out_b: UInt32(outB_),
                out_c: UInt32(outC_), out_d: UInt32(outD_),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_batched_4_qgemv_fast_bf16(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_out: outA.buffer, a_outOffset: outA.offset,
                b_out: outB.buffer, b_outOffset: outB.offset,
                c_out: outC.buffer, c_outOffset: outC.offset,
                d_out: outD.buffer, d_outOffset: outD.offset,
                out_a: UInt32(outA_), out_b: UInt32(outB_),
                out_c: UInt32(outC_), out_d: UInt32(outD_),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.batched4QgemvInt4Fast: unsupported dtype \(x.dtype)")
        }
    }

    /// M>1 (batched-prefill) sibling of `batched4QgemvInt4Fast`. Reads
    /// `x = [M, in_dim]` once into TG memory and produces all 4
    /// outputs (each `[M, out_*]`) in a single dispatch. Eliminates
    /// the 3 redundant input DRAM reads of the unfused `callMany`
    /// chain used by the GDN mixer's batched forward (qkv, z, b, a all
    /// project from the same `xNorm`).
    ///
    /// Output tensors are caller-allocated and indexed independently
    /// (`outA / outB / outC / outD`, each `[M, out_*]`).
    ///
    /// Constraints (mirror `ffai_batched_4_qgemv_fast`):
    /// - `in_dim` MUST be a multiple of 512.
    /// - Each of `out_a / out_b / out_c / out_d` MUST be a multiple
    ///   of 8.
    /// - `group_size` MUST be 64. TPG = 64.
    public static func batched4QmmFast(
        input x: Tensor, m: Int,
        wA: Tensor, scalesA: Tensor, biasesA: Tensor, outA: Tensor,
        wB: Tensor, scalesB: Tensor, biasesB: Tensor, outB: Tensor,
        wC: Tensor, scalesC: Tensor, biasesC: Tensor, outC: Tensor,
        wD: Tensor, scalesD: Tensor, biasesD: Tensor, outD: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            wA.dtype == .u32 && wB.dtype == .u32 && wC.dtype == .u32 && wD.dtype == .u32,
            "Ops.batched4QmmFast: w_* must be u32-packed")
        let packedPerRow = wA.shape[1]
        let inDim = packedPerRow * 8
        precondition(
            x.elementCount == m * inDim,
            "Ops.batched4QmmFast: x size \(x.elementCount) ≠ M·inDim \(m * inDim)")
        let outADim = wA.shape[0]
        let outBDim = wB.shape[0]
        let outCDim = wC.shape[0]
        let outDDim = wD.shape[0]
        precondition(
            outA.elementCount == m * outADim,
            "Ops.batched4QmmFast: outA size \(outA.elementCount) ≠ M·outADim \(m * outADim)")
        precondition(
            outB.elementCount == m * outBDim,
            "Ops.batched4QmmFast: outB size \(outB.elementCount) ≠ M·outBDim \(m * outBDim)")
        precondition(
            outC.elementCount == m * outCDim,
            "Ops.batched4QmmFast: outC size \(outC.elementCount) ≠ M·outCDim \(m * outCDim)")
        precondition(
            outD.elementCount == m * outDDim,
            "Ops.batched4QmmFast: outD size \(outD.elementCount) ≠ M·outDDim \(m * outDDim)")
        precondition(
            inDim % 512 == 0,
            "Ops.batched4QmmFast: in_dim must be a multiple of 512")
        precondition(
            outADim % 8 == 0 && outBDim % 8 == 0 && outCDim % 8 == 0 && outDDim % 8 == 0,
            "Ops.batched4QmmFast: each out_* must be a multiple of 8")
        let tpg = 64
        let maxOut = max(max(outADim, outBDim), max(outCDim, outDDim))
        let nTiles = maxOut / 8
        let grid = MTLSize(width: nTiles * tpg, height: m, depth: 4)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.ffai_batched_4_qmm_fast_f32(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_buf: outA.buffer, a_bufOffset: outA.offset,
                b_buf: outB.buffer, b_bufOffset: outB.offset,
                c_buf: outC.buffer, c_bufOffset: outC.offset,
                d_buf: outD.buffer, d_bufOffset: outD.offset,
                out_a: UInt32(outADim), out_b: UInt32(outBDim),
                out_c: UInt32(outCDim), out_d: UInt32(outDDim),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_batched_4_qmm_fast_f16(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_buf: outA.buffer, a_bufOffset: outA.offset,
                b_buf: outB.buffer, b_bufOffset: outB.offset,
                c_buf: outC.buffer, c_bufOffset: outC.offset,
                d_buf: outD.buffer, d_bufOffset: outD.offset,
                out_a: UInt32(outADim), out_b: UInt32(outBDim),
                out_c: UInt32(outCDim), out_d: UInt32(outDDim),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_batched_4_qmm_fast_bf16(
                x: x.buffer, xOffset: x.offset,
                w_a: wA.buffer, w_aOffset: wA.offset,
                scales_a: scalesA.buffer, scales_aOffset: scalesA.offset,
                biases_a: biasesA.buffer, biases_aOffset: biasesA.offset,
                w_b: wB.buffer, w_bOffset: wB.offset,
                scales_b: scalesB.buffer, scales_bOffset: scalesB.offset,
                biases_b: biasesB.buffer, biases_bOffset: biasesB.offset,
                w_c: wC.buffer, w_cOffset: wC.offset,
                scales_c: scalesC.buffer, scales_cOffset: scalesC.offset,
                biases_c: biasesC.buffer, biases_cOffset: biasesC.offset,
                w_d: wD.buffer, w_dOffset: wD.offset,
                scales_d: scalesD.buffer, scales_dOffset: scalesD.offset,
                biases_d: biasesD.buffer, biases_dOffset: biasesD.offset,
                a_buf: outA.buffer, a_bufOffset: outA.offset,
                b_buf: outB.buffer, b_bufOffset: outB.offset,
                c_buf: outC.buffer, c_bufOffset: outC.offset,
                d_buf: outD.buffer, d_bufOffset: outD.offset,
                out_a: UInt32(outADim), out_b: UInt32(outBDim),
                out_c: UInt32(outCDim), out_d: UInt32(outDDim),
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.batched4QmmFast: unsupported dtype \(x.dtype)")
        }
    }

    /// Per-expert indexed int4 dequant-GEMV. The caller stacks every
    /// expert's weight slab into one `[nExperts, outDim, inDim/8]`
    /// u32-packed tensor (and matching scales / biases stacks); the
    /// kernel reads `expertIndex[0]` to pick the slab to dequantize on
    /// this call. Paired with `Ops.moeRouterTopK`, the top-K indices
    /// stay GPU-resident and the host-side `route → dispatch expert
    /// gemv` sync at every MoE layer collapses to one
    /// `expertIndex.buffer + offset = slot · 4` view per slot.
    public static func dequantGemvInt4ExpertIndexed(
        weightsStacked: Tensor, scalesStacked: Tensor, biasesStacked: Tensor,
        input: Tensor, expertIndex: Tensor,
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            weightsStacked.shape.count == 3,
            "Ops.dequantGemvInt4ExpertIndexed: weightsStacked must be [nExperts, outDim, inDim/8]")
        precondition(
            weightsStacked.dtype == .u32,
            "Ops.dequantGemvInt4ExpertIndexed: weightsStacked must be u32 (packed)")
        precondition(
            expertIndex.dtype == .u32 && expertIndex.elementCount == 1,
            "Ops.dequantGemvInt4ExpertIndexed: expertIndex must be [1] u32")
        precondition(
            scalesStacked.dtype == input.dtype && biasesStacked.dtype == input.dtype,
            "Ops.dequantGemvInt4ExpertIndexed: scales / biases dtype must match input")
        precondition(
            out.dtype == input.dtype,
            "Ops.dequantGemvInt4ExpertIndexed: out dtype must match input")
        let outDim = weightsStacked.shape[1]
        let packedPerRow = weightsStacked.shape[2]
        let inDim = packedPerRow * 8  // int4 packs 8 weights per u32 word
        precondition(
            input.elementCount == inDim,
            "Ops.dequantGemvInt4ExpertIndexed: input \(input.elementCount) ≠ inDim \(inDim)")
        precondition(
            out.elementCount == outDim,
            "Ops.dequantGemvInt4ExpertIndexed: out \(out.elementCount) ≠ outDim \(outDim)")
        // Kernel runs in Reduction mode with tpg=32 (one simdgroup per
        // output row). `dispatchThreads` counts THREADS not threadgroups
        // — total threads = outDim · 32 gives outDim threadgroups, one
        // per output row.
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let grid = MTLSize(width: outDim * 32, height: 1, depth: 1)
        let inDimU = UInt32(inDim)
        let outDimU = UInt32(outDim)
        let groupSizeU = UInt32(groupSize)
        switch input.dtype {
        case .f32:
            MetalTileKernels.dequant_gemv_int4_expert_indexed_f32(
                weights_stacked: weightsStacked.buffer, weights_stackedOffset: weightsStacked.offset,
                scales_stacked: scalesStacked.buffer, scales_stackedOffset: scalesStacked.offset,
                biases_stacked: biasesStacked.buffer, biases_stackedOffset: biasesStacked.offset,
                input: input.buffer, inputOffset: input.offset,
                expert_index: expertIndex.buffer, expert_indexOffset: expertIndex.offset,
                output: out.buffer, outputOffset: out.offset,
                in_dim: inDimU, out_dim: outDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.dequant_gemv_int4_expert_indexed_f16(
                weights_stacked: weightsStacked.buffer, weights_stackedOffset: weightsStacked.offset,
                scales_stacked: scalesStacked.buffer, scales_stackedOffset: scalesStacked.offset,
                biases_stacked: biasesStacked.buffer, biases_stackedOffset: biasesStacked.offset,
                input: input.buffer, inputOffset: input.offset,
                expert_index: expertIndex.buffer, expert_indexOffset: expertIndex.offset,
                output: out.buffer, outputOffset: out.offset,
                in_dim: inDimU, out_dim: outDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.dequant_gemv_int4_expert_indexed_bf16(
                weights_stacked: weightsStacked.buffer, weights_stackedOffset: weightsStacked.offset,
                scales_stacked: scalesStacked.buffer, scales_stackedOffset: scalesStacked.offset,
                biases_stacked: biasesStacked.buffer, biases_stackedOffset: biasesStacked.offset,
                input: input.buffer, inputOffset: input.offset,
                expert_index: expertIndex.buffer, expert_indexOffset: expertIndex.offset,
                output: out.buffer, outputOffset: out.offset,
                in_dim: inDimU, out_dim: outDimU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.dequantGemvInt4ExpertIndexed: unsupported input dtype \(input.dtype)")
        }
    }

    /// Encoder-batched form of `dequantGemvInt4ExpertIndexed`. All N
    /// calls must share `(inDim, outDim, groupSize, dtype)` so they
    /// reuse the same PSO + constexpr binding; only the per-call
    /// `(weights, scales, biases, input, expertIndex, output)` rotate
    /// per dispatch. Saves N-1 encoder begin/end pairs versus N
    /// independent calls. At Qwen3.6-A3B's topK=8 the gate+up phase
    /// becomes 16 calls on one encoder and the down phase 8 calls on
    /// a second encoder per MoE layer.
    public static func dequantGemvInt4ExpertIndexedMany(
        weightsStacked: [Tensor], scalesStacked: [Tensor], biasesStacked: [Tensor],
        inputs: [Tensor], expertIndices: [Tensor], outputs: [Tensor],
        groupSize: Int = 64,
        on cmd: MTLCommandBuffer
    ) {
        let n = weightsStacked.count
        precondition(
            scalesStacked.count == n && biasesStacked.count == n
                && inputs.count == n && expertIndices.count == n
                && outputs.count == n,
            "Ops.dequantGemvInt4ExpertIndexedMany: count mismatch")
        guard n > 0 else { return }
        let dtype = inputs[0].dtype
        let psoName: String
        switch dtype {
        case .f32: psoName = "dequant_gemv_int4_expert_indexed_f32"
        case .f16: psoName = "dequant_gemv_int4_expert_indexed_f16"
        case .bf16: psoName = "dequant_gemv_int4_expert_indexed_bf16"
        default:
            fatalError(
                "Ops.dequantGemvInt4ExpertIndexedMany: unsupported dtype \(dtype)")
        }
        let outDim = weightsStacked[0].shape[1]
        let packedPerRow = weightsStacked[0].shape[2]
        let inDim = packedPerRow * 8
        var inDimV = UInt32(inDim)
        var outDimV = UInt32(outDim)
        var groupSizeV = UInt32(groupSize)
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        // Shared constexprs (kernel buffer indices 6, 7, 8) set once.
        enc.setBytes(&inDimV, length: 4, index: 6)
        enc.setBytes(&outDimV, length: 4, index: 7)
        enc.setBytes(&groupSizeV, length: 4, index: 8)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let grid = MTLSize(width: outDim * 32, height: 1, depth: 1)
        for i in 0 ..< n {
            precondition(
                weightsStacked[i].shape[1] == outDim
                    && weightsStacked[i].shape[2] == packedPerRow,
                "Ops.dequantGemvInt4ExpertIndexedMany: weight shape varies at \(i)")
            precondition(
                inputs[i].dtype == dtype && outputs[i].dtype == dtype,
                "Ops.dequantGemvInt4ExpertIndexedMany: dtype varies at \(i)")
            precondition(
                expertIndices[i].dtype == .u32 && expertIndices[i].elementCount == 1,
                "Ops.dequantGemvInt4ExpertIndexedMany: expertIndices[\(i)] must be [1] u32")
            enc.setBuffer(weightsStacked[i].buffer, offset: weightsStacked[i].offset, index: 0)
            enc.setBuffer(scalesStacked[i].buffer, offset: scalesStacked[i].offset, index: 1)
            enc.setBuffer(biasesStacked[i].buffer, offset: biasesStacked[i].offset, index: 2)
            enc.setBuffer(inputs[i].buffer, offset: inputs[i].offset, index: 3)
            enc.setBuffer(expertIndices[i].buffer, offset: expertIndices[i].offset, index: 4)
            enc.setBuffer(outputs[i].buffer, offset: outputs[i].offset, index: 5)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        enc.endEncoding()
    }

    /// MoE top-K router. Reads `[nExperts]` raw logits and writes
    /// `[k]` u32 expert indices + `[k]` weights of the same dtype.
    /// `normTopkProb` matches Qwen3-MoE convention when `true`
    /// (softmax restricted to the chosen k, weights renormalise to
    /// sum-to-1); `false` matches the Qwen3-Next style (softmax over
    /// all experts, then pick top-k without renormalisation).
    ///
    /// Single-row form — `logits: [nExperts]`. Internally calls the
    /// T-batched variant with `t = 1`.
    public static func moeRouterTopK(
        logits: Tensor, indicesOut: Tensor, weightsOut: Tensor,
        nExperts: Int, k: Int, normTopkProb: Bool,
        on cmd: MTLCommandBuffer
    ) {
        moeRouterTopKMany(
            logits: logits, indicesOut: indicesOut, weightsOut: weightsOut,
            t: 1, nExperts: nExperts, k: k, normTopkProb: normTopkProb,
            on: cmd)
    }

    /// T-batched MoE top-K router. `logits: [T, nExperts]`,
    /// `indicesOut: [T, k]` u32, `weightsOut: [T, k]` (matching logits
    /// dtype). The kernel iterates `T` rows along `program_id<0>()`
    /// (one threadgroup per row), so a single dispatch covers all
    /// prefill rows without the host commit + wait + CPU
    /// `MoERouter.route` round-trip that the per-row form forced.
    public static func moeRouterTopKMany(
        logits: Tensor, indicesOut: Tensor, weightsOut: Tensor,
        t: Int, nExperts: Int, k: Int, normTopkProb: Bool,
        on cmd: MTLCommandBuffer
    ) {
        precondition(t > 0, "Ops.moeRouterTopKMany: t must be positive")
        precondition(
            logits.elementCount == t * nExperts,
            "Ops.moeRouterTopKMany: logits must be [T·nExperts]")
        precondition(
            indicesOut.elementCount == t * k && indicesOut.dtype == .u32,
            "Ops.moeRouterTopKMany: indicesOut must be [T·k] u32")
        precondition(
            weightsOut.elementCount == t * k && weightsOut.dtype == logits.dtype,
            "Ops.moeRouterTopKMany: weightsOut must be [T·k] matching logits dtype")
        // Kernel pins tpg = 32 (one simdgroup per token row, Reduction
        // mode). Grid total threads = T · 32 → T threadgroups.
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let grid = MTLSize(width: t * 32, height: 1, depth: 1)
        let normFlag: UInt32 = normTopkProb ? 1 : 0
        switch logits.dtype {
        case .f32:
            MetalTileKernels.mt_moe_router_topk_f32(
                router_logits: logits.buffer, router_logitsOffset: logits.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k), norm_topk_prob: normFlag,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_moe_router_topk_f16(
                router_logits: logits.buffer, router_logitsOffset: logits.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k), norm_topk_prob: normFlag,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_moe_router_topk_bf16(
                router_logits: logits.buffer, router_logitsOffset: logits.offset,
                indices_out: indicesOut.buffer, indices_outOffset: indicesOut.offset,
                weights_out: weightsOut.buffer, weights_outOffset: weightsOut.offset,
                n_experts: UInt32(nExperts), k: UInt32(k), norm_topk_prob: normFlag,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.moeRouterTopKMany: unsupported logits dtype \(logits.dtype)")
        }
    }

    /// GPU argmax over a 1D logits tensor. Caller supplies a 1-element
    /// u32 output buffer. Uses the cooperative 256-thread Reduction
    /// kernel — one threadgroup, ~80-300 KB / vocab logits in registers.
    /// Mamba 2 / Mamba 1D depthwise causal-conv step — streaming
    /// decode form. One thread per channel. `state` is the rolling
    /// window of the last `kernelSize - 1` inputs (shape
    /// `[kernelSize - 1, nChannels]`); shifted in-place after compute.
    /// Activation (Mamba 2 follows the conv with SiLU) is the caller's
    /// concern — kept separate for composability.
    public static func conv1dCausalStep(
        x: Tensor, w: Tensor, b: Tensor,
        state: Tensor, into y: Tensor,
        nChannels: Int, kernelSize: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            w.dtype == x.dtype && b.dtype == x.dtype
                && state.dtype == x.dtype && y.dtype == x.dtype,
            "Ops.conv1dCausalStep: every tensor must share dtype")
        let grid = MTLSize(width: nChannels, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.conv1d_causal_step_f32(
                x: x.buffer, xOffset: x.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state: state.buffer, stateOffset: state.offset,
                y: y.buffer, yOffset: y.offset,
                n_channels: UInt32(nChannels), kernel_size: UInt32(kernelSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.conv1d_causal_step_f16(
                x: x.buffer, xOffset: x.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state: state.buffer, stateOffset: state.offset,
                y: y.buffer, yOffset: y.offset,
                n_channels: UInt32(nChannels), kernel_size: UInt32(kernelSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.conv1d_causal_step_bf16(
                x: x.buffer, xOffset: x.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state: state.buffer, stateOffset: state.offset,
                y: y.buffer, yOffset: y.offset,
                n_channels: UInt32(nChannels), kernel_size: UInt32(kernelSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.conv1dCausalStep: unsupported dtype \(x.dtype)")
        }
    }

    /// Batched depthwise causal conv1d + SiLU + cast-to-f32. Sweeps
    /// `t` tokens in ONE dispatch with the conv state held in
    /// per-channel registers across the sweep — `stateIn` is read
    /// once at the top, `stateOut` is written once at the bottom.
    /// Replaces the per-token `conv1dCausalStep` + `siluCastToF32`
    /// T-loop in `Qwen35GDNMixer.forwardManyChunked`.
    ///
    /// `convKernel` must be 4 (Qwen3.5 / 3.6, Mamba 2, NemotronH all
    /// use kernel=4; the kernel signature carries it as a constexpr
    /// for shape uniformity but the body is K=4 hardcoded).
    ///
    /// State in-place safety: each grid thread owns one channel;
    /// reads `stateIn[*, c]` once at the start of the sweep and
    /// writes `stateOut[*, c]` once at the end. Passing the same
    /// buffer for both is safe.
    public static func conv1dCausalStepSiluCastMany(
        src: Tensor, w: Tensor, b: Tensor,
        stateIn: Tensor, outF32: Tensor, stateOut: Tensor,
        t: Int, convDim: Int, convKernel: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            convKernel == 4,
            "Ops.conv1dCausalStepSiluCastMany: requires conv_kernel = 4")
        precondition(
            src.dtype == w.dtype && src.dtype == b.dtype
                && src.dtype == stateIn.dtype && src.dtype == stateOut.dtype,
            "Ops.conv1dCausalStepSiluCastMany: dtype mismatch")
        precondition(
            outF32.dtype == .f32,
            "Ops.conv1dCausalStepSiluCastMany: outF32 must be .f32")
        let grid = MTLSize(width: convDim, height: 1, depth: 1)
        let tg = MTLSize(width: min(convDim, 256), height: 1, depth: 1)
        let tLenU = UInt32(t)
        let cdU = UInt32(convDim)
        let ckU = UInt32(convKernel)
        switch src.dtype {
        case .f32:
            MetalTileKernels.ffai_conv1d_causal_step_silu_cast_many_f32(
                src: src.buffer, srcOffset: src.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                out_f32: outF32.buffer, out_f32Offset: outF32.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                t_len: tLenU, conv_dim: cdU, conv_kernel: ckU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_conv1d_causal_step_silu_cast_many_f16(
                src: src.buffer, srcOffset: src.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                out_f32: outF32.buffer, out_f32Offset: outF32.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                t_len: tLenU, conv_dim: cdU, conv_kernel: ckU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_conv1d_causal_step_silu_cast_many_bf16(
                src: src.buffer, srcOffset: src.offset,
                w: w.buffer, wOffset: w.offset,
                b: b.buffer, bOffset: b.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                out_f32: outF32.buffer, out_f32Offset: outF32.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                t_len: tLenU, conv_dim: cdU, conv_kernel: ckU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.conv1dCausalStepSiluCastMany: unsupported dtype \(src.dtype)")
        }
    }

    /// Mamba 2 selective-scan single-token decode step. Updates the
    /// per-layer recurrent state `h` in place and writes the output
    /// channel vector `y`. `h` lives in fp32 (state accumulates over
    /// many decode steps; bf16's 7-bit mantissa drifts fast). One
    /// thread per `(head, channel)` — total `nHeads * headDim` threads.
    ///
    /// See `SSMStateCache` for the storage class that wraps the per-layer
    /// `h` buffer; Mamba 2 family files call this through
    /// that cache.
    public static func ssmStep(
        x: Tensor, a: Tensor, b: Tensor, c: Tensor, dt: Tensor,
        state h: Tensor, into y: Tensor,
        nHeads: Int, headDim: Int, stateDim: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(h.dtype == .f32, "Ops.ssmStep: state h must be f32")
        precondition(x.dtype == y.dtype, "Ops.ssmStep: x and y dtype must match")
        precondition(
            a.dtype == x.dtype && b.dtype == x.dtype
                && c.dtype == x.dtype && dt.dtype == x.dtype,
            "Ops.ssmStep: a/b/c/dt dtype must match x")
        let grid = MTLSize(width: nHeads * headDim, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.ssm_step_f32(
                x: x.buffer, xOffset: x.offset,
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: c.buffer, cOffset: c.offset,
                dt: dt.buffer, dtOffset: dt.offset,
                h: h.buffer, hOffset: h.offset,
                y: y.buffer, yOffset: y.offset,
                head_dim: UInt32(headDim), state_dim: UInt32(stateDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ssm_step_f16(
                x: x.buffer, xOffset: x.offset,
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: c.buffer, cOffset: c.offset,
                dt: dt.buffer, dtOffset: dt.offset,
                h: h.buffer, hOffset: h.offset,
                y: y.buffer, yOffset: y.offset,
                head_dim: UInt32(headDim), state_dim: UInt32(stateDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ssm_step_bf16(
                x: x.buffer, xOffset: x.offset,
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: c.buffer, cOffset: c.offset,
                dt: dt.buffer, dtOffset: dt.offset,
                h: h.buffer, hOffset: h.offset,
                y: y.buffer, yOffset: y.offset,
                head_dim: UInt32(headDim), state_dim: UInt32(stateDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.ssmStep: unsupported dtype \(x.dtype)")
        }
    }

    /// Gated Delta Net (GDN) recurrent single-token decode step.
    ///
    /// Computes, per decode step and per value head, the recurrence
    ///   S_t = g_t · S_{t-1} + β_t · k_t · (v_t − k_tᵀ · S_{t-1})ᵀ
    /// then the output `o_t = S_t · q_t`. The per-head state matrix
    /// `S [Hv, Dv, Dk]` stays fp32 throughout — the gate + rank-1
    /// update accumulate across many steps and bf16 drifts fast.
    ///
    /// The underlying kernel is **reduction-mode** with strict
    /// dispatch-shape invariants (see
    /// `crates/metaltile-std/src/ffai/gated_delta.rs`
    /// §"DISPATCH INVARIANTS" and `OpsValidation.validateGatedDeltaStep`):
    /// TPG = 32 (one simdgroup), one threadgroup per `(dv, n)` pair,
    /// `Dk % 32 == 0`, `Hv % Hk == 0`. The `(Dk, Dv, Hk, Hv)` dimensions
    /// are runtime `#[constexpr]` scalars — a single PSO serves every
    /// model config, no per-tuple specialization.
    ///
    /// `q` / `k` are expected pre-normalised (the GDN block applies the
    /// rmsNorm + scale before calling this — the standard, non-fused
    /// kernel variant). The kernel reads `stateIn` and writes a distinct
    /// `stateOut`; callers double-buffer via `GDNStateCache.swap()`.
    ///
    /// All tensors are f32 — the kernel runs the recurrence state in f32
    /// regardless of activation dtype. This wraps `mt_gated_delta_step`,
    /// which performs exactly one recurrence step; multi-step prefill
    /// loops this call per token at the caller.
    public static func gatedDeltaStep(
        q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
        stateIn: Tensor, into y: Tensor, stateOut: Tensor,
        numKeyHeads: Int, numValueHeads: Int,
        keyHeadDim: Int, valueHeadDim: Int,
        on cmd: MTLCommandBuffer
    ) {
        // Kernel-invariant validation — see OpsValidation.swift. A bad
        // dispatch shape on a reduction kernel ranges from silent
        // miscompute to a non-preemptive GPU pin.
        if let reason = OpsValidation.validateGatedDeltaStep(
            keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim,
            numKeyHeads: numKeyHeads, numValueHeads: numValueHeads
        ) {
            preconditionFailure("Ops.gatedDeltaStep: \(reason)")
        }
        precondition(
            q.dtype == .f32 && k.dtype == .f32 && v.dtype == .f32
                && g.dtype == .f32 && beta.dtype == .f32,
            "Ops.gatedDeltaStep: q/k/v/g/beta must be f32")
        precondition(
            stateIn.dtype == .f32 && stateOut.dtype == .f32
                && y.dtype == .f32,
            "Ops.gatedDeltaStep: state + output tensors must be f32")

        // Dispatch derived from the invariants: 32 threads per group
        // (one simdgroup), one group per (dv_idx, n) pair. The kernel
        // reads tid as the lane (dk_idx), tgid_x as dv_idx, tgid_y as n
        // = batch·Hv + hv. Decode is single-batch so n ranges [0, Hv).
        // Generated bindings use `dispatchThreads`, so the grid is
        // counted in THREADS: width = Dv · 32, height = Hv.
        let lanesPerGroup = 32
        let grid = MTLSize(
            width: valueHeadDim * lanesPerGroup,
            height: numValueHeads, depth: 1)
        let tg = MTLSize(width: lanesPerGroup, height: 1, depth: 1)

        // `mt_gated_delta_step` is a single PSO: the (Dk, Dv, Hv, Hk)
        // dimensions are runtime `#[constexpr]` scalars rather than
        // baked compile-time constants, so one dispatch serves every
        // model config.
        MetalTileKernels.mt_gated_delta_step_f32(
            q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
            v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
            beta: beta.buffer, betaOffset: beta.offset,
            state_in: stateIn.buffer, state_inOffset: stateIn.offset,
            state_out: stateOut.buffer, state_outOffset: stateOut.offset,
            y: y.buffer, yOffset: y.offset,
            dk: UInt32(keyHeadDim), dv: UInt32(valueHeadDim),
            hv: UInt32(numValueHeads), hk: UInt32(numKeyHeads),
            gridSize: grid, threadgroupSize: tg, on: cmd)
    }

    public static func argmax(
        _ logits: Tensor, into out: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(out.dtype == .u32, "Ops.argmax: output must be u32")
        precondition(out.elementCount == 1, "Ops.argmax: output must be a single element")
        let n = logits.elementCount
        // Reduction: dispatch 256 threads in 1 group (full simdgroups).
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: 256, height: 1, depth: 1)
        switch logits.dtype {
        case .f32:
            MetalTileKernels.ffai_argmax_f32(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_argmax_f16(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_argmax_bf16(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.argmax: unsupported dtype \(logits.dtype)")
        }
    }

    /// GPU softmax + categorical sample over a 1D logits tensor.
    /// Caller supplies:
    ///   - 1-element u32 output buffer for the sampled token id
    ///   - 1-element f32 temperature buffer (must be > 0; T=0 should
    ///     route to `argmax` instead)
    ///   - 1-element f32 uniform draw in [0, 1) — CPU-generated each
    ///     decode step from `GenerationParameters.makeRNG()`
    ///
    /// Used by the Generate decode loop for the pure-temperature
    /// sampling path (T > 0, no top-K / top-P / min-P / rep-penalty)
    /// to keep logits on the GPU. Top-K / top-P / min-P /
    /// rep-penalty still flow through the CPU `Sampling.sample(...)`
    /// path until separate filter kernels land.
    public static func softmaxCategoricalSample(
        _ logits: Tensor,
        into out: Tensor,
        temperature: Tensor,
        uniform: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(out.dtype == .u32, "Ops.softmaxCategoricalSample: output must be u32")
        precondition(
            out.elementCount == 1, "Ops.softmaxCategoricalSample: output must be a single element")
        precondition(
            temperature.dtype == .f32 && temperature.elementCount == 1,
            "Ops.softmaxCategoricalSample: temperature must be a 1-element f32 tensor")
        precondition(
            uniform.dtype == .f32 && uniform.elementCount == 1,
            "Ops.softmaxCategoricalSample: uniform must be a 1-element f32 tensor")
        let n = logits.elementCount
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let grid = MTLSize(width: 256, height: 1, depth: 1)
        switch logits.dtype {
        case .f32:
            MetalTileKernels.softmax_categorical_sample_f32(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                temperature_in: temperature.buffer, temperature_inOffset: temperature.offset,
                uniform_in: uniform.buffer, uniform_inOffset: uniform.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.softmax_categorical_sample_f16(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                temperature_in: temperature.buffer, temperature_inOffset: temperature.offset,
                uniform_in: uniform.buffer, uniform_inOffset: uniform.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.softmax_categorical_sample_bf16(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                temperature_in: temperature.buffer, temperature_inOffset: temperature.offset,
                uniform_in: uniform.buffer, uniform_inOffset: uniform.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.softmaxCategoricalSample: unsupported dtype \(logits.dtype)")
        }
    }

    /// Affine-quantize one K (or V) row into an int4 or int8 KV cache
    /// slot. Dispatches to the right `quantize_kv_int{4,8}` kernel
    /// based on `bits`.
    public static func quantizeKVAffine(
        src: Tensor,
        weights: Tensor, scales: Tensor, biases: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, position: Int, bits: Int,
        on cmd: MTLCommandBuffer
    ) {
        switch bits {
        case 4:
            quantizeKVInt4(
                src: src, weights: weights, scales: scales, biases: biases,
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                groupSize: groupSize, position: position, on: cmd)
        case 8:
            quantizeKVInt8(
                src: src, weights: weights, scales: scales, biases: biases,
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                groupSize: groupSize, position: position, on: cmd)
        default:
            fatalError("Ops.quantizeKVAffine: unsupported bits=\(bits) (use 4 or 8)")
        }
    }

    /// Bulk-dequant int4/int8 KV cache → working buffer. Dispatches to
    /// the right `bulk_dequant_kv_int{4,8}` kernel based on `bits`.
    public static func bulkDequantKVAffine(
        weights: Tensor, scales: Tensor, biases: Tensor,
        into out: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, nPositions: Int, bits: Int,
        on cmd: MTLCommandBuffer
    ) {
        switch bits {
        case 4:
            bulkDequantKVInt4(
                weights: weights, scales: scales, biases: biases,
                into: out, nKVHeads: nKVHeads, headDim: headDim,
                maxSeq: maxSeq, groupSize: groupSize,
                nPositions: nPositions, on: cmd)
        case 8:
            bulkDequantKVInt8(
                weights: weights, scales: scales, biases: biases,
                into: out, nKVHeads: nKVHeads, headDim: headDim,
                maxSeq: maxSeq, groupSize: groupSize,
                nPositions: nPositions, on: cmd)
        default:
            fatalError("Ops.bulkDequantKVAffine: unsupported bits=\(bits) (use 4 or 8)")
        }
    }

    /// Affine-quantize one K (or V) row into an int4 KV cache slot.
    /// Packs 8 nibbles per uint32. One thread per group.
    public static func quantizeKVInt4(
        src: Tensor,
        weights: Tensor, scales: Tensor, biases: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, position: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(weights.dtype == .u32, "quantizeKVInt4: weights must be u32")
        precondition(
            scales.dtype == src.dtype && biases.dtype == src.dtype,
            "quantizeKVInt4: scales/biases dtype must match src")
        // Kernel-invariant validation (silent-miscompute footguns:
        // partial trailing group, unaligned pack stride). See
        // OpsValidation.validateQuantizeKV.
        if let reason = OpsValidation.validateQuantizeKV(
            nKVHeads: nKVHeads, headDim: headDim, groupSize: groupSize, bits: 4
        ) {
            preconditionFailure("Ops.quantizeKVInt4: \(reason)")
        }
        let groupsPerHead = headDim / groupSize
        let grid = MTLSize(width: nKVHeads * groupsPerHead, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch src.dtype {
        case .f32:
            MetalTileKernels.quantize_kv_int4_f32(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.quantize_kv_int4_f16(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.quantize_kv_int4_bf16(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.quantizeKVInt4: unsupported dtype \(src.dtype)")
        }
    }

    /// Bulk-dequant the live slice of an int4 KV cache into a
    /// working buffer for SDPA. Unpacks 8 nibbles per uint32.
    public static func bulkDequantKVInt4(
        weights: Tensor, scales: Tensor, biases: Tensor,
        into out: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, nPositions: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(weights.dtype == .u32, "bulkDequantKVInt4: weights must be u32")
        precondition(
            scales.dtype == out.dtype && biases.dtype == out.dtype,
            "bulkDequantKVInt4: scales/biases dtype must match output")
        // Kernel-invariant validation — shares the int4 group/pack
        // alignment contract with quantizeKVInt4. See
        // OpsValidation.validateQuantizeKV.
        if let reason = OpsValidation.validateQuantizeKV(
            nKVHeads: nKVHeads, headDim: headDim, groupSize: groupSize, bits: 4
        ) {
            preconditionFailure("Ops.bulkDequantKVInt4: \(reason)")
        }
        let total = nKVHeads * nPositions * headDim
        let grid = MTLSize(width: total, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch out.dtype {
        case .f32:
            MetalTileKernels.bulk_dequant_kv_int4_f32(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.bulk_dequant_kv_int4_f16(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.bulk_dequant_kv_int4_bf16(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.bulkDequantKVInt4: unsupported dtype \(out.dtype)")
        }
    }

    /// Affine-quantize one K (or V) row into an int8 KV cache slot.
    /// `src` is `[nKVHeads, headDim]` in fp16/bf16; outputs are the
    /// cache's packed weights + per-group scales + biases (see
    /// `AffineQuantizedKVCache`). One thread per group.
    public static func quantizeKVInt8(
        src: Tensor,
        weights: Tensor, scales: Tensor, biases: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, position: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(weights.dtype == .u32, "quantizeKVInt8: weights must be u32")
        precondition(
            scales.dtype == src.dtype && biases.dtype == src.dtype,
            "quantizeKVInt8: scales/biases dtype must match src")
        // Kernel-invariant validation (silent-miscompute footguns:
        // partial trailing group, unaligned pack stride). See
        // OpsValidation.validateQuantizeKV.
        if let reason = OpsValidation.validateQuantizeKV(
            nKVHeads: nKVHeads, headDim: headDim, groupSize: groupSize, bits: 8
        ) {
            preconditionFailure("Ops.quantizeKVInt8: \(reason)")
        }
        let groupsPerHead = headDim / groupSize
        let grid = MTLSize(width: nKVHeads * groupsPerHead, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch src.dtype {
        case .f32:
            MetalTileKernels.quantize_kv_int8_f32(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.quantize_kv_int8_f16(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.quantize_kv_int8_bf16(
                src: src.buffer, srcOffset: src.offset,
                out_w: weights.buffer, out_wOffset: weights.offset,
                out_s: scales.buffer, out_sOffset: scales.offset,
                out_b: biases.buffer, out_bOffset: biases.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), position: UInt32(position),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.quantizeKVInt8: unsupported dtype \(src.dtype)")
        }
    }

    /// Bulk-dequantize the live slice of an int8 KV cache into a
    /// working buffer that SDPA can read directly. Output buffer
    /// shape `[nKVHeads, maxSeq, headDim]`; only positions `[0,
    /// nPositions)` are written. One thread per output element.
    public static func bulkDequantKVInt8(
        weights: Tensor, scales: Tensor, biases: Tensor,
        into out: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        groupSize: Int, nPositions: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(weights.dtype == .u32, "bulkDequantKVInt8: weights must be u32")
        precondition(
            scales.dtype == out.dtype && biases.dtype == out.dtype,
            "bulkDequantKVInt8: scales/biases dtype must match output")
        // Kernel-invariant validation — shares the int8 group/pack
        // alignment contract with quantizeKVInt8. See
        // OpsValidation.validateQuantizeKV.
        if let reason = OpsValidation.validateQuantizeKV(
            nKVHeads: nKVHeads, headDim: headDim, groupSize: groupSize, bits: 8
        ) {
            preconditionFailure("Ops.bulkDequantKVInt8: \(reason)")
        }
        let total = nKVHeads * nPositions * headDim
        let grid = MTLSize(width: total, height: 1, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch out.dtype {
        case .f32:
            MetalTileKernels.bulk_dequant_kv_int8_f32(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.bulk_dequant_kv_int8_f16(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.bulk_dequant_kv_int8_bf16(
                in_w: weights.buffer, in_wOffset: weights.offset,
                in_s: scales.buffer, in_sOffset: scales.offset,
                in_b: biases.buffer, in_bOffset: biases.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), max_seq: UInt32(maxSeq),
                group_size: UInt32(groupSize), n_positions: UInt32(nPositions),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.bulkDequantKVInt8: unsupported dtype \(out.dtype)")
        }
    }

    /// Append one timestep to a KV cache on the GPU. `src` is
    /// [nKVHeads, headDim] (rotated K or V for the current token);
    /// `cache` is the full [nKVHeads, maxSeq, headDim] buffer.
    /// `position` is the slot to write into. Replaces the CPU-side
    /// memcpy + mid-layer commit/wait pattern from the initial cut.
    public static func kvCacheUpdate(
        src: Tensor, into cache: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int, position: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(src.dtype == cache.dtype, "kvCacheUpdate: dtype mismatch")
        let total = nKVHeads * headDim
        let (grid, tg) = elementwiseGrid(total)
        let hd = UInt32(headDim)
        let ms = UInt32(maxSeq)
        let pos = UInt32(position)
        switch src.dtype {
        case .f32:
            MetalTileKernels.kv_cache_update_f32(
                src: src.buffer, srcOffset: src.offset,
                out: cache.buffer, outOffset: cache.offset,
                head_dim: hd, max_seq: ms, position: pos,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.kv_cache_update_f16(
                src: src.buffer, srcOffset: src.offset,
                out: cache.buffer, outOffset: cache.offset,
                head_dim: hd, max_seq: ms, position: pos,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.kv_cache_update_bf16(
                src: src.buffer, srcOffset: src.offset,
                out: cache.buffer, outOffset: cache.offset,
                head_dim: hd, max_seq: ms, position: pos,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.kvCacheUpdate: unsupported dtype \(src.dtype)")
        }
    }

    /// Append the current token's K AND V rows to their caches in ONE
    /// compute encoder. The cache buffers, layout, and writing thread
    /// are identical to `kvCacheUpdate` — this just amortises the
    /// encoder begin/end across both projections (saving one pair per
    /// attention layer per decode token).
    public static func kvCacheUpdateKV(
        kSrc: Tensor, kCache: Tensor,
        vSrc: Tensor, vCache: Tensor,
        nKVHeads: Int, headDim: Int, maxSeq: Int, position: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            kSrc.dtype == kCache.dtype && vSrc.dtype == vCache.dtype
                && kSrc.dtype == vSrc.dtype,
            "Ops.kvCacheUpdateKV: dtype mismatch")
        let total = nKVHeads * headDim
        let (grid, tg) = elementwiseGrid(total)
        let hd = UInt32(headDim)
        let ms = UInt32(maxSeq)
        let pos = UInt32(position)
        let psoName: String
        switch kSrc.dtype {
        case .f32: psoName = "kv_cache_update_f32"
        case .f16: psoName = "kv_cache_update_f16"
        case .bf16: psoName = "kv_cache_update_bf16"
        default: fatalError("Ops.kvCacheUpdateKV: unsupported dtype \(kSrc.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        @inline(__always)
        func dispatch(_ src: Tensor, _ cache: Tensor) {
            enc.setBuffer(src.buffer, offset: src.offset, index: 0)
            enc.setBuffer(cache.buffer, offset: cache.offset, index: 1)
            var headDimV = hd
            var maxSeqV = ms
            var positionV = pos
            enc.setBytes(&headDimV, length: 4, index: 2)
            enc.setBytes(&maxSeqV, length: 4, index: 3)
            enc.setBytes(&positionV, length: 4, index: 4)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(kSrc, kCache)
        dispatch(vSrc, vCache)
        enc.endEncoding()
    }

    /// Batched K + V cache append: writes T tokens' K rows AND V rows
    /// into their respective caches in ONE shared encoder (2 batched
    /// dispatches). Each per-token write goes to a slot indexed by
    /// `positions[t]`. Replaces the per-token `kvCacheUpdateKV` loop
    /// at `decodeMany` / `forwardMany` prefill — at T=512 × 10 attn
    /// layers that's ≈ 5100 fewer dispatches per prefill call.
    ///
    /// Shapes:
    ///   - `kSrc` / `vSrc`: `[T, nKVHeads, headDim]` (flat).
    ///   - `kCache` / `vCache`: `[nKVHeads, maxSeq, headDim]`.
    ///   - `positions`: `[T]` u32 slot indices into the caches.
    public static func kvCacheUpdateKVMany(
        kSrc: Tensor, kCache: Tensor,
        vSrc: Tensor, vCache: Tensor,
        positions: Tensor, t: Int,
        nKVHeads: Int, headDim: Int, maxSeq: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            kSrc.dtype == kCache.dtype && vSrc.dtype == vCache.dtype
                && kSrc.dtype == vSrc.dtype,
            "Ops.kvCacheUpdateKVMany: dtype mismatch")
        precondition(
            positions.dtype == .u32,
            "Ops.kvCacheUpdateKVMany: positions must be .u32")
        precondition(
            positions.elementCount == t,
            "Ops.kvCacheUpdateKVMany: positions count must equal T")
        let elementsPerRow = nKVHeads * headDim
        let total = t * elementsPerRow
        let (grid, tg) = elementwiseGrid(total)
        let psoName: String
        switch kSrc.dtype {
        case .f32: psoName = "kv_cache_update_many_f32"
        case .f16: psoName = "kv_cache_update_many_f16"
        case .bf16: psoName = "kv_cache_update_many_bf16"
        default: fatalError("Ops.kvCacheUpdateKVMany: unsupported dtype \(kSrc.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        // Shared bindings: positions + scalars set ONCE.
        enc.setBuffer(positions.buffer, offset: positions.offset, index: 1)
        var hd = UInt32(headDim)
        var ms = UInt32(maxSeq)
        var nhd = UInt32(elementsPerRow)
        enc.setBytes(&hd, length: 4, index: 3)
        enc.setBytes(&ms, length: 4, index: 4)
        enc.setBytes(&nhd, length: 4, index: 5)
        @inline(__always)
        func dispatch(_ src: Tensor, _ cache: Tensor) {
            enc.setBuffer(src.buffer, offset: src.offset, index: 0)
            enc.setBuffer(cache.buffer, offset: cache.offset, index: 2)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        dispatch(kSrc, kCache)
        dispatch(vSrc, vCache)
        enc.endEncoding()
    }

    /// SDPA decode. q: [n_q_heads, head_dim]. k/v cache layout:
    /// [n_kv_heads, kv_stride, head_dim] where kv_stride is the physical
    /// capacity (maxSeq) and nKV is how many positions to attend to.
    /// Output: [n_q_heads, head_dim].
    ///
    /// Dispatch invariants (see `crates/metaltile-std/src/ffai/sdpa_decode.rs`
    /// + `sdpa_decode_d64.rs`):
    ///   * `head_dim ∈ {64, 128}` — one threadgroup is 32 simdgroups ×
    ///     32 lanes; each lane owns `head_dim / 32` consecutive Q/K/V
    ///     elements. 128 covers full attention heads on Llama 3.2 3B+ /
    ///     Qwen3 / GPT-OSS full layers; 64 covers Llama 3.2 1B and
    ///     GPT-OSS sliding-window layers. head_dim=256 (some Gemma
    ///     configs) is queued but not yet emitted.
    ///   * 1 threadgroup per Q head, 1024 threads per threadgroup
    ///     (32 simdgroups × 32 lanes). `tgid_x = q_head`.
    ///
    /// Sliding-window / attention-sink fast path (`head_dim = 128`
    /// variants only — metaltile PR #50): when `sinkEnd` and/or
    /// `windowStart` are non-zero the kernel attends `[0, sinkEnd)`
    /// (the pinned attention sinks) plus `[windowStart, nKV)` (the
    /// sliding window) and skips the masked range
    /// `[sinkEnd, windowStart)` at the loop-bound level — no
    /// per-position branching. Both default to 0, which is exactly
    /// dense full attention over `[0, nKV)`. Callers with a windowed
    /// KV cache derive these from the eviction policy: for a window of
    /// `W` retained positions with `S` pinned sink slots,
    /// `windowStart = max(0, nKV - W)` and `sinkEnd = S`. The d64 /
    /// d256 kernel variants are dense-only — passing non-zero sink /
    /// window with those head dims is a precondition failure.
    public static func sdpaDecode(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        nKV: Int, kvStride: Int,
        scale: Float, on cmd: MTLCommandBuffer,
        sinkEnd: Int = 0, windowStart: Int = 0,
        into out: Tensor? = nil
    ) -> Tensor {
        // Kernel-invariant validation — see OpsValidation.swift for the
        // full reasoning + CI-runnable tests. The 2026-05-19 GPU freeze
        // came from this wrapper accepting head_dim=4 with no check.
        if let reason = OpsValidation.validateSdpaDecode(
            headDim: headDim, nQHeads: nQHeads, nKVHeads: nKVHeads,
            nKV: nKV, kvStride: kvStride,
            sinkEnd: sinkEnd, windowStart: windowStart
        ) {
            preconditionFailure("Ops.sdpaDecode: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: [nQHeads, headDim], dtype: q.dtype)
        // One threadgroup per q-head. Reduction-mode kernel reads `tgid_x`
        // as the q-head; we lay out a 1D thread grid of
        // `nQHeads * threadsPerGroup` threads so Metal slices it into
        // `nQHeads` threadgroups.
        //
        // The d64/d128/d256 variants run 1024 threads (32 simdgroups).
        // The d512 variant runs 512 (16 simdgroups): its 16-wide per-lane
        // register footprint pushes the pipeline's
        // maxTotalThreadsPerThreadgroup below 1024, and a 1024-thread
        // dispatch silently no-ops (command buffer errors, output stays
        // zero). The kernel body is parametric in `n_simd`, so 512 is
        // correct — see sdpa_decode_d512.rs §"DISPATCH INVARIANTS".
        let threadsPerGroup = headDim == 512 ? 512 : 1024
        let grid = MTLSize(width: nQHeads * threadsPerGroup, height: 1, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let headsPerGroup = nQHeads / nKVHeads

        // Route to the head_dim-specialized kernel. Each variant has its
        // own per-lane width baked into the kernel (head_dim=128 → 4
        // elt/lane, head_dim=64 → 2 elt/lane). A generic kernel would
        // either lose register efficiency or trip up the codegen's
        // vectorize pass; per-head-dim variants keep both kernels fast.
        switch (headDim, q.dtype) {
        case (128, .f32):
            MetalTileKernels.ffai_sdpa_decode_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                // Sliding-window / sink fast path. Both 0 → dense full
                // attention over [0, n_kv); non-zero values let the
                // kernel skip the masked range [sink_end, window_start)
                // at the loop-bound level. See the doc comment above.
                sink_end: UInt32(sinkEnd), window_start: UInt32(windowStart),
                // has_sink: 0 → no learned attention-sink logit folded
                // into the softmax denominator (GPT-OSS-style sinks).
                // Ops.sdpaDecode does not yet expose a sink-logit param;
                // 0 reproduces the pre-#145 dense-attention behavior.
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (128, .f16):
            MetalTileKernels.ffai_sdpa_decode_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                sink_end: UInt32(sinkEnd), window_start: UInt32(windowStart),
                // has_sink: 0 → dense softmax denominator (see d128 f32
                // case above for the GPT-OSS attention-sink rationale).
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (128, .bf16):
            MetalTileKernels.ffai_sdpa_decode_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                sink_end: UInt32(sinkEnd), window_start: UInt32(windowStart),
                // has_sink: 0 → dense softmax denominator (see d128 f32
                // case above for the GPT-OSS attention-sink rationale).
                has_sink: 0, sink_logit: 0.0,
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        // NOTE: the d64/d256 specialized kernels in metaltile-std
        // don't accept `sink_end` / `window_start` parameters — they're
        // perf-tuned for dense full attention. Callers needing sliding-
        // window / attention-sink at these head_dims must precondition
        // `sinkEnd == 0 && windowStart == 0` (validated upstream by
        // OpsValidation.validateSdpaDecode). The d128 / d512 cases use
        // the generic `ffai_sdpa_decode_*` kernel with constexpr
        // head_dim which DOES carry both params.
        case (64, .f32):
            MetalTileKernels.ffai_sdpa_decode_d64_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (64, .f16):
            MetalTileKernels.ffai_sdpa_decode_d64_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (64, .bf16):
            MetalTileKernels.ffai_sdpa_decode_d64_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (256, .f32):
            MetalTileKernels.ffai_sdpa_decode_d256_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (256, .f16):
            MetalTileKernels.ffai_sdpa_decode_d256_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (256, .bf16):
            MetalTileKernels.ffai_sdpa_decode_d256_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        // d512 routes to the dedicated `ffai_sdpa_decode_d512_*` kernel.
        // metaltile #145 auto-discovers `sdpa_decode_d512.rs`, so the
        // per-head-dim kernel is emitted again — the generic-kernel
        // reroute used while it was missing produced all-zero output
        // past offset 128 (the generic body's per-lane width assumes
        // head_dim ≤ 128). Like d64/d256, the d512 kernel is perf-tuned
        // for dense full attention and does not accept sink params;
        // OpsValidation preconditions `sinkEnd == 0 && windowStart == 0`.
        case (512, .f32):
            MetalTileKernels.ffai_sdpa_decode_d512_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (512, .f16):
            MetalTileKernels.ffai_sdpa_decode_d512_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (512, .bf16):
            MetalTileKernels.ffai_sdpa_decode_d512_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            // OpsValidation rejected anything we don't have a kernel for;
            // an unsupported dtype lands here.
            fatalError("Ops.sdpaDecode: unsupported (head_dim=\(headDim), dtype=\(q.dtype))")
        }
        return result
    }

    /// Two-pass Flash-Decoding SDPA. Pass 1 partitions the KV cache
    /// into `blocks` even tiles and writes a per-block partial
    /// `(O, max, lse)` triple; pass 2 merges them with the log-sum-exp
    /// trick. Wins over single-pass `sdpaDecode` at long KV (≥ 1K
    /// typical), where the single-pass kernel starves the GPU with
    /// one threadgroup per Q-head.
    ///
    /// **Block-count contract.** The pass2 kernel hardcodes a 32-wide
    /// reduction (`bn = 32` per lane) over `blocks / 32` chunks, so
    /// `blocks` MUST be a multiple of 32 and at least 32. Passing
    /// `blocks < 32` causes pass2 to iterate zero chunks and emit an
    /// all-zero output silently — the precondition below catches that.
    /// 32 is the standard short-context tile; 64 / 96 / 128 are
    /// reasonable for very long contexts.
    ///
    /// **Scratch buffer sizing.** Caller owns the partials; reuse
    /// across decode steps is safe because the shape depends only on
    /// `(nQHeads, headDim, blocks)`:
    ///   - `partialO` : `[nQHeads, blocks, headDim]` matching `q.dtype`
    ///   - `partialM` : `[nQHeads, blocks]` fp32 (running max)
    ///   - `partialL` : `[nQHeads, blocks]` fp32 (running lse)
    public static func sdpaDecode2Pass(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        nKV: Int, kvStride: Int, blocks: Int,
        scale: Float,
        partialO: Tensor, partialM: Tensor, partialL: Tensor,
        into out: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            nQHeads % nKVHeads == 0,
            "Ops.sdpaDecode2Pass: nQHeads must be a multiple of nKVHeads")
        let gqaFactor = nQHeads / nKVHeads
        precondition(
            blocks >= 32 && blocks % 32 == 0,
            "Ops.sdpaDecode2Pass: blocks (\(blocks)) must be a multiple of 32 "
                + "and ≥ 32 — pass2 hardcodes a 32-lane block-merge reduction")
        precondition(
            partialO.elementCount == nQHeads * blocks * headDim,
            "Ops.sdpaDecode2Pass: partialO must be [nQHeads, blocks, headDim]")
        precondition(
            partialM.elementCount == nQHeads * blocks && partialM.dtype == .f32,
            "Ops.sdpaDecode2Pass: partialM must be [nQHeads, blocks] f32")
        precondition(
            partialL.elementCount == nQHeads * blocks && partialL.dtype == .f32,
            "Ops.sdpaDecode2Pass: partialL must be [nQHeads, blocks] f32")
        precondition(
            partialO.dtype == q.dtype && out.dtype == q.dtype,
            "Ops.sdpaDecode2Pass: partialO / out dtype must match q")

        // Pass 1: per-block partial O / m / l. One TG per
        // (kv_head, block) with `gqa_factor` simdgroups × 32 lanes,
        // each lane owning `headDim / 32` consecutive elements.
        let pass1TgWidth = gqaFactor * 32
        let pass1Grid = MTLSize(
            width: nKVHeads * pass1TgWidth, height: blocks, depth: 1)
        let pass1Tg = MTLSize(width: pass1TgWidth, height: 1, depth: 1)

        // Pass 2: merge partials across blocks. One TG per Q-head with
        // 32 simdgroups × 32 lanes = 1024 threads.
        let pass2TgWidth = 1024
        let pass2Grid = MTLSize(
            width: nQHeads * pass2TgWidth, height: 1, depth: 1)
        let pass2Tg = MTLSize(width: pass2TgWidth, height: 1, depth: 1)

        switch q.dtype {
        case .f32:
            MetalTileKernels.sdpa_decode_2pass_pass1_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                gqa_factor: UInt32(gqaFactor), blocks: UInt32(blocks),
                scale: scale,
                gridSize: pass1Grid, threadgroupSize: pass1Tg, on: cmd)
            MetalTileKernels.sdpa_decode_2pass_pass2_f32(
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), blocks: UInt32(blocks),
                gridSize: pass2Grid, threadgroupSize: pass2Tg, on: cmd)
        case .f16:
            MetalTileKernels.sdpa_decode_2pass_pass1_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                gqa_factor: UInt32(gqaFactor), blocks: UInt32(blocks),
                scale: scale,
                gridSize: pass1Grid, threadgroupSize: pass1Tg, on: cmd)
            MetalTileKernels.sdpa_decode_2pass_pass2_f16(
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), blocks: UInt32(blocks),
                gridSize: pass2Grid, threadgroupSize: pass2Tg, on: cmd)
        case .bf16:
            MetalTileKernels.sdpa_decode_2pass_pass1_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                head_dim: UInt32(headDim), n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                gqa_factor: UInt32(gqaFactor), blocks: UInt32(blocks),
                scale: scale,
                gridSize: pass1Grid, threadgroupSize: pass1Tg, on: cmd)
            MetalTileKernels.sdpa_decode_2pass_pass2_bf16(
                partial_o: partialO.buffer, partial_oOffset: partialO.offset,
                partial_m: partialM.buffer, partial_mOffset: partialM.offset,
                partial_l: partialL.buffer, partial_lOffset: partialL.offset,
                out: out.buffer, outOffset: out.offset,
                head_dim: UInt32(headDim), blocks: UInt32(blocks),
                gridSize: pass2Grid, threadgroupSize: pass2Tg, on: cmd)
        default:
            fatalError("Ops.sdpaDecode2Pass: unsupported dtype \(q.dtype)")
        }
    }

    /// Multi-query SDPA — attends `nQuery` query rows against a shared
    /// K/V cache in one dispatch. `q` / output are `[nQuery, nQHeads,
    /// headDim]`; `k` / `v` are the cache buffers `[nKVHeads, kvStride,
    /// headDim]`. `causal == false` → every query attends
    /// `[0, baseKV + nQuery)` (bidirectional); `causal == true` → query
    /// `r` attends `[0, baseKV + r + 1)`.
    ///
    /// `ffai_sdpa_multi` is a reduction kernel — its threadgroup
    /// geometry is part of the contract. Each invariant below is
    /// `precondition`-checked, citing `crates/metaltile-std/src/ffai/
    /// sdpa_multi.rs`. The 1024-thread dispatch is hard: a smaller TPG
    /// makes the kernel's `n_simd` zero and the K walk an infinite GPU
    /// loop (the documented machine-freeze hazard).
    public static func sdpaMulti(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        baseKV: Int, nQuery: Int, kvStride: Int,
        causal: Bool, scale: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        // ## DISPATCH INVARIANTS — ffai/sdpa_multi.rs. See
        // OpsValidation.validateSdpaMulti for the full reasoning +
        // CI-runnable tests.
        if let reason = OpsValidation.validateSdpaMulti(
            headDim: headDim, nQHeads: nQHeads, nKVHeads: nKVHeads,
            baseKV: baseKV, nQuery: nQuery, kvStride: kvStride
        ) {
            preconditionFailure("Ops.sdpaMulti: \(reason)")
        }
        let headsPerGroup = nQHeads / nKVHeads
        let result = out ?? Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: q.dtype)
        // TPG = 1024 (32 simdgroups × 32 lanes), one threadgroup per
        // (query, q_head). Lay out a 1D thread grid so Metal slices it
        // into nQHeads*nQuery threadgroups of 1024 — NEVER fewer than
        // 32 threads per group (the freeze condition).
        let threadsPerGroup = 1024
        let grid = MTLSize(width: nQHeads * nQuery * threadsPerGroup, height: 1, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let causalFlag = UInt32(causal ? 1 : 0)
        // Route by (dtype, headDim). d=128 → `ffai_sdpa_multi_*`; d=256
        // → `ffai_sdpa_multi_d256_*`, the variant added for Qwen3.6-A3B
        // full-attention layers (head_dim=256). Both kernels share the
        // same Swift wrapper shape; the dispatch grid / TPG / online
        // softmax / causal mask / GQA fan-out semantics are identical.
        // The d=256 kernel uses a 2-phase output reduction internally
        // to stay under Apple's 32 KB threadgroup-memory cap.
        switch (q.dtype, headDim) {
        case (.f32, 128):
            MetalTileKernels.ffai_sdpa_multi_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.f16, 128):
            MetalTileKernels.ffai_sdpa_multi_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.bf16, 128):
            MetalTileKernels.ffai_sdpa_multi_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.f32, 256):
            MetalTileKernels.ffai_sdpa_multi_d256_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.f16, 256):
            MetalTileKernels.ffai_sdpa_multi_d256_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (.bf16, 256):
            MetalTileKernels.ffai_sdpa_multi_d256_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                causal: causalFlag, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.sdpaMulti: unsupported dtype \(q.dtype) at headDim \(headDim) "
                    + "— only (f32/f16/bf16) × (128, 256) are emitted")
        }
        return result
    }

    /// Tree-causal `sdpaMulti` — same dispatch contract but the
    /// scalar `causal` boolean is replaced by an additive
    /// `[nQuery, nQuery]` mask tensor consulted only for in-block KV
    /// positions (positions `< baseKV` — the cached prefix — are
    /// always fully attended). The mask is `0.0` for "allow" and
    /// `-inf` for "block"; the kernel adds it directly to the
    /// pre-softmax scores.
    ///
    /// Used by speculative-decode tree-verify: one verifier call
    /// attends every leaf-to-root path of the draft tree at once,
    /// gated by a tree-causal mask that lets each leaf only see its
    /// own ancestors. Companion to `DraftTreeNode.treeCausalMask()`.
    ///
    /// Mask dtype must match `q`. `headDim` shares `sdpaMulti`'s
    /// constraint: only the 128-variant kernel is emitted on dev
    /// (tree-verify has no d256 use case today).
    public static func sdpaMultiTreeMask(
        q: Tensor, k: Tensor, v: Tensor, mask: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        baseKV: Int, nQuery: Int, kvStride: Int,
        scale: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        if let reason = OpsValidation.validateSdpaMulti(
            headDim: headDim, nQHeads: nQHeads, nKVHeads: nKVHeads,
            baseKV: baseKV, nQuery: nQuery, kvStride: kvStride
        ) {
            preconditionFailure("Ops.sdpaMultiTreeMask: \(reason)")
        }
        precondition(
            headDim == 128,
            "Ops.sdpaMultiTreeMask: only headDim=128 has a tree-mask kernel variant")
        precondition(
            mask.dtype == q.dtype,
            "Ops.sdpaMultiTreeMask: mask dtype (\(mask.dtype)) must match q dtype (\(q.dtype))")
        precondition(
            mask.elementCount == nQuery * nQuery,
            "Ops.sdpaMultiTreeMask: mask elementCount \(mask.elementCount) ≠ nQuery·nQuery \(nQuery * nQuery)")
        let headsPerGroup = nQHeads / nKVHeads
        let result = out ?? Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: q.dtype)
        let threadsPerGroup = 1024
        let grid = MTLSize(
            width: nQHeads * nQuery * threadsPerGroup,
            height: 1, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        switch q.dtype {
        case .f32:
            MetalTileKernels.ffai_sdpa_multi_tree_mask_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                mask: mask.buffer, maskOffset: mask.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_sdpa_multi_tree_mask_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                mask: mask.buffer, maskOffset: mask.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_sdpa_multi_tree_mask_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                mask: mask.buffer, maskOffset: mask.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim), n_q_heads: UInt32(nQHeads),
                base_kv: UInt32(baseKV), n_query: UInt32(nQuery),
                kv_stride: UInt32(kvStride), heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sdpaMultiTreeMask: unsupported dtype \(q.dtype)")
        }
        return result
    }

    /// Multi-query **bidirectional** SDPA — every query attends the
    /// full `[0, baseKV + nQuery)` range. Specialized variants for
    /// head_dim ∈ {32, 64, 72} cover the VLM vision-tower spectrum
    /// (FastViT-HD = 32; SigLIP / CLIP-L = 64; PaliGemma SigLIP-So400m
    /// = 72). For head_dim = 128 use `sdpaMulti(causal: false)`.
    ///
    /// Same shape contract as `sdpaMulti`: Q / `out` layout
    /// `[nQuery, nQHeads, headDim]`, K / V layout `[nKVHeads, kvStride,
    /// headDim]`. The d=72 kernel uses a ragged 3-elements-per-lane
    /// layout with bounds masking — lanes 24..31 are idle (25% lane
    /// occupancy loss, kernel-internal, transparent to the wrapper).
    ///
    /// `ffai_sdpa_bidirectional_dN` is a reduction kernel — TPG=1024
    /// is hard. See `OpsValidation.validateSdpaBidirectional` for the
    /// invariant list and `metaltile/crates/metaltile-std/src/ffai/
    /// sdpa_bidirectional.rs` for the dispatch contract.
    public static func sdpaBidirectional(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        baseKV: Int, nQuery: Int, kvStride: Int,
        scale: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        if let reason = OpsValidation.validateSdpaBidirectional(
            headDim: headDim, nQHeads: nQHeads, nKVHeads: nKVHeads,
            baseKV: baseKV, nQuery: nQuery, kvStride: kvStride
        ) {
            preconditionFailure("Ops.sdpaBidirectional: \(reason)")
        }
        let headsPerGroup = nQHeads / nKVHeads
        let result = out ?? Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: q.dtype)
        // TPG = 1024 (32 simdgroups × 32 lanes), one threadgroup per
        // (query, q_head). Same machine-freeze hazard as sdpa_multi —
        // never use elementwiseGrid (would make n_simd=0).
        let threadsPerGroup = 1024
        let grid = MTLSize(width: nQHeads * nQuery * threadsPerGroup, height: 1, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let hd = UInt32(headDim)
        let nqh = UInt32(nQHeads)
        let bkv = UInt32(baseKV)
        let nq = UInt32(nQuery)
        let kvs = UInt32(kvStride)
        let hpg = UInt32(headsPerGroup)
        switch (headDim, q.dtype) {
        case (32, .f32):
            MetalTileKernels.ffai_sdpa_bidirectional_d32_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (32, .f16):
            MetalTileKernels.ffai_sdpa_bidirectional_d32_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (32, .bf16):
            MetalTileKernels.ffai_sdpa_bidirectional_d32_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (64, .f32):
            MetalTileKernels.ffai_sdpa_bidirectional_d64_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (64, .f16):
            MetalTileKernels.ffai_sdpa_bidirectional_d64_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (64, .bf16):
            MetalTileKernels.ffai_sdpa_bidirectional_d64_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (72, .f32):
            MetalTileKernels.ffai_sdpa_bidirectional_d72_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (72, .f16):
            MetalTileKernels.ffai_sdpa_bidirectional_d72_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (72, .bf16):
            MetalTileKernels.ffai_sdpa_bidirectional_d72_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (80, .f32):
            MetalTileKernels.ffai_sdpa_bidirectional_d80_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (80, .f16):
            MetalTileKernels.ffai_sdpa_bidirectional_d80_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (80, .bf16):
            MetalTileKernels.ffai_sdpa_bidirectional_d80_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (96, .f32):
            MetalTileKernels.ffai_sdpa_bidirectional_d96_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (96, .f16):
            MetalTileKernels.ffai_sdpa_bidirectional_d96_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (96, .bf16):
            MetalTileKernels.ffai_sdpa_bidirectional_d96_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_q_heads: nqh, base_kv: bkv, n_query: nq,
                kv_stride: kvs, heads_per_group: hpg, scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sdpaBidirectional: unsupported (headDim=\(headDim), dtype=\(q.dtype))")
        }
        return result
    }

    // MARK: - AURA

    /// AURA fused encode for `rows` flat vectors of length `dim`.
    /// Computes per-row L2 norm, rotates by `rotation` (`[dim×dim]`
    /// f32), quantises against `boundaries`, packs codebook indices
    /// into `packed_out`, and writes the norm-correction factor to
    /// `norms_out`. One threadgroup per row, `dim` threads per group.
    ///
    /// `input` accepts f32 / f16 / bf16 — the kernel casts to f32 at
    /// the load, so production code can feed the K/V projection
    /// directly without an explicit upcast pass. `rotation`,
    /// `boundaries`, and `codebook` are always f32 (precision matters
    /// for the rotation matmul + Lloyd-Max boundary comparisons).
    ///
    /// Per-(bits, dtype) dispatch. `bits ∈ {2, 3, 4, 8}` × dtype ∈
    /// {f32, f16, bf16} → 12 generated kernel variants.
    public static func auraEncode(
        input: Tensor, rotation: Tensor, boundaries: Tensor, codebook: Tensor,
        packedOut: Tensor, normsOut: Tensor,
        rows: Int, dim: Int, packedWidth: Int, bits: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            rotation.dtype == .f32 && boundaries.dtype == .f32 && codebook.dtype == .f32,
            "Ops.auraEncode: rotation/boundaries/codebook must be f32")
        precondition(packedOut.dtype == .u32, "Ops.auraEncode: packed_out must be u32")
        precondition(normsOut.dtype == .f32, "Ops.auraEncode: norms_out must be f32")
        // Kernel-invariant validation — see OpsValidation.swift.
        if let reason = OpsValidation.validateAuraEncode(rows: rows, dim: dim, bits: bits) {
            preconditionFailure("Ops.auraEncode: \(reason)")
        }
        // One threadgroup per row; `dim` threads per group (each thread
        // owns one rotated coordinate). The kernel reads
        // `row = program_id::<0>() = tgid_x`. We dispatch `rows * dim`
        // total threads in a 1D grid so Metal slices into `rows`
        // threadgroups of `dim` threads — that gives `tgid_x` the row
        // index we want.
        //
        // The earlier shape `(dim, rows, 1)` with `dispatchThreads`
        // computed threadgroups as `(1, rows, 1)`, making `tgid_x` always
        // 0 and processing only row 0. Latent because no production
        // caller existed; AURAQuantizedKVCache is the first.
        let grid = MTLSize(width: dim * rows, height: 1, depth: 1)
        let tg = MTLSize(width: dim, height: 1, depth: 1)
        let dimU = UInt32(dim)
        let pwU = UInt32(packedWidth)

        // Per-(bits, dtype) routing. The kernel layout, argument order,
        // and grid shape are identical across all 12 variants — only the
        // function name changes — so a small local helper keeps the
        // switch readable.
        @inline(__always)
        func dispatchEncode(
            _ kernel: (
                MTLBuffer, Int, MTLBuffer, Int, MTLBuffer, Int,
                MTLBuffer, Int, MTLBuffer, Int, MTLBuffer, Int,
                UInt32, UInt32, MTLSize, MTLSize, MTLCommandBuffer
            ) -> Void
        ) {
            kernel(
                input.buffer, input.offset,
                rotation.buffer, rotation.offset,
                boundaries.buffer, boundaries.offset,
                codebook.buffer, codebook.offset,
                packedOut.buffer, packedOut.offset,
                normsOut.buffer, normsOut.offset,
                dimU, pwU, grid, tg, cmd
            )
        }

        switch (bits, input.dtype) {
        case (2, .f32):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int2_f32(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (2, .f16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int2_f16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (2, .bf16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int2_bf16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (3, .f32):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int3_f32(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (3, .f16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int3_f16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (3, .bf16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int3_bf16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (4, .f32):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int4_f32(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (4, .f16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int4_f16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (4, .bf16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int4_bf16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (8, .f32):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int8_f32(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (8, .f16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int8_f16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        case (8, .bf16):
            dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
                MetalTileKernels.aura_encode_int8_bf16(
                    input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b,
                    boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p,
                    packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d,
                    packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd)
            }
        default:
            fatalError("Ops.auraEncode: unsupported (bits=\(bits), input.dtype=\(input.dtype))")
        }
    }

    /// AURA bulk dequant — codebook lookup + norm rescale for every
    /// stored token, into a fp32 / fp16 / bf16 working buffer in
    /// rotated codec space. The caller applies the inverse rotation
    /// (e.g. by folding Π into W_o offline, or via a follow-up
    /// matmul / fused pass2_with_rot kernel).
    ///
    /// Grid: `(packed_width, tokens, B*H)` — `dispatchThreads` does
    /// not pad, so no per-thread bounds guards are needed.
    /// Bulk-dequant AURA-packed K/V into `out`.
    ///
    /// `tokens` is the number of valid token rows to process (the
    /// dispatch grid height). `cacheStride` is the per-head row stride
    /// of the `packed` / `norms` / `out` buffers — i.e. the *allocated*
    /// sequence capacity. The dequant kernel keys all per-head offset
    /// arithmetic off its `tokens` constexpr, so for a buffer laid out
    /// `[nKVHeads, maxSeq, …]` the kernel must be fed `cacheStride =
    /// maxSeq`, NOT the fill count — otherwise heads 1…n read/write at
    /// the wrong offset and the error grows with fill length (the AURA
    /// "coherent then collapse" bug). `cacheStride` defaults to `tokens`
    /// for callers whose buffers are exactly `[nKVHeads, tokens, …]`.
    public static func auraDequantRotated(
        packed: Tensor, norms: Tensor, codebook: Tensor,
        into out: Tensor,
        nKVHeads: Int, dim: Int, packedWidth: Int, tokens: Int, bits: Int,
        cacheStride: Int? = nil,
        on cmd: MTLCommandBuffer
    ) {
        precondition(packed.dtype == .u32, "Ops.auraDequantRotated: packed must be u32")
        precondition(
            norms.dtype == .f32 && codebook.dtype == .f32,
            "Ops.auraDequantRotated: norms/codebook must be f32")
        let stride = cacheStride ?? tokens
        // Kernel-invariant validation (cacheStride row-stride contract,
        // packedWidth dim coverage, bit-width). See
        // OpsValidation.validateAuraDequantRotated.
        if let reason = OpsValidation.validateAuraDequantRotated(
            dim: dim, packedWidth: packedWidth, tokens: tokens, bits: bits,
            cacheStride: stride
        ) {
            preconditionFailure("Ops.auraDequantRotated: \(reason)")
        }
        // Grid height = rows to process (`tokens`); the kernel's `tokens`
        // constexpr carries the per-head buffer stride (`stride`).
        let grid = MTLSize(width: packedWidth, height: tokens, depth: nKVHeads)
        let tg = MTLSize(width: packedWidth, height: 1, depth: 1)
        switch (bits, out.dtype) {
        case (2, .f32):
            MetalTileKernels.aura_dequant_rotated_int2_f32(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .f16):
            MetalTileKernels.aura_dequant_rotated_int2_f16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (2, .bf16):
            MetalTileKernels.aura_dequant_rotated_int2_bf16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f32):
            MetalTileKernels.aura_dequant_rotated_int3_f32(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .f16):
            MetalTileKernels.aura_dequant_rotated_int3_f16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (3, .bf16):
            MetalTileKernels.aura_dequant_rotated_int3_bf16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .f32):
            MetalTileKernels.aura_dequant_rotated_int4_f32(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .f16):
            MetalTileKernels.aura_dequant_rotated_int4_f16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (4, .bf16):
            MetalTileKernels.aura_dequant_rotated_int4_bf16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f32):
            MetalTileKernels.aura_dequant_rotated_int8_f32(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .f16):
            MetalTileKernels.aura_dequant_rotated_int8_f16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (8, .bf16):
            MetalTileKernels.aura_dequant_rotated_int8_bf16(
                packed: packed.buffer, packedOffset: packed.offset,
                norms: norms.buffer, normsOffset: norms.offset,
                codebook: codebook.buffer, codebookOffset: codebook.offset,
                out: out.buffer, outOffset: out.offset,
                dim: UInt32(dim), packed_width: UInt32(packedWidth), tokens: UInt32(stride),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.auraDequantRotated: unsupported (bits=\(bits), dtype=\(out.dtype))")
        }
    }

    /// Apply a shared `[headDim, headDim]` rotation matrix to each
    /// head's `[headDim]` slice of a flat `[nHeads * headDim]` tensor.
    /// Used for AURA's Q post-RoPE rotation and the attention-output
    /// un-rotation (see `AURAQuantizedKVCache` header).
    ///
    /// The rotation matrix is shared across heads within a layer; we
    /// fan out one `Ops.gemv` dispatch per head with per-head buffer
    /// views, so the cost is `nHeads` gemv launches per call. For Qwen3
    /// 1.7B that's 16 dispatches per Q rotation × 2 calls per layer × 28
    /// layers = 896 dispatches per token. Fine for correctness-first
    /// Stage 1a; Stage 1b folds this work into the compressed-domain
    /// `aura_flash_p1` / pass2 kernels which expect pre-rotated Q and
    /// produce un-rotated output directly.
    ///
    /// Preconditions:
    ///   * `x` is `[nHeads * headDim]`
    ///   * `rotation` is `[headDim, headDim]`
    ///   * `x.dtype == rotation.dtype` (gemv requires matched dtypes —
    ///     the caller is expected to keep an activation-dtype copy of
    ///     the rotation alongside the f32 copy required by `auraEncode`)
    public static func auraRotatePerHead(
        _ x: Tensor, rotation: Tensor,
        nHeads: Int, headDim: Int,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        // Pure shape / dtype contract validation — see
        // OpsValidation.validateAuraRotatePerHead.
        if let reason = OpsValidation.validateAuraRotatePerHead(
            xElementCount: x.elementCount, rotationShape: rotation.shape,
            rotationDtypeMatchesX: rotation.dtype == x.dtype,
            nHeads: nHeads, headDim: headDim
        ) {
            preconditionFailure("Ops.auraRotatePerHead: \(reason)")
        }

        let result = Tensor.empty(shape: [nHeads * headDim], dtype: x.dtype)
        let bytesPerHead = headDim * x.dtype.byteSize

        for h in 0 ..< nHeads {
            let xView = Tensor(
                buffer: x.buffer,
                offset: x.offset + h * bytesPerHead,
                shape: [headDim], dtype: x.dtype)
            let outView = Tensor(
                buffer: result.buffer,
                offset: result.offset + h * bytesPerHead,
                shape: [headDim], dtype: x.dtype)
            _ = Ops.gemv(weight: rotation, input: xView, on: cmd, into: outView)
        }
        return result
    }

    // ─── MoE batched gather GEMM (int4) ──────────────────────────────
    //
    // Wraps `mt_moe_gather_qmm_mma_int4_bm16_*` — one kernel launch that
    // processes `mTotal` rows of activations, each row tagged with the
    // expert id it routes to via `indices[mTotal]`. Replaces a Python-
    // style serial-expert loop (`topK` dispatches × 3 projections = 3K
    // launches) with one batched dispatch per projection.
    //
    // Weight layout (mlx affine int4 stacked):
    //   `weight   [E, N, K/8]` u32 packed, row-major within each expert.
    //   `scales`, `biases` `[E, N, K/groupSize]` in the activation dtype.
    // Activation layout:
    //   `input    [mTotal, K]` contiguous in `input.dtype`.
    //   `indices  [mTotal]` u32 — expert id per row. ROWS MUST BE SORTED
    //                            BY EXPERT (the kernel scans forward
    //                            looking for an expert-id change to find
    //                            the per-tile sub-range).
    // Output:
    //   `out      [mTotal, N]` in `input.dtype` (caller pre-allocates).
    //
    // Tile contract: N % 32 == 0, K % 32 == 0 (the kernel pads m_total
    // internally — BM=16 with boundary masking). Grid =
    //   [N / 32, ceil(mTotal / 16), 1]
    // Threadgroup = [64, 1, 1] (2 simdgroups).
    public static func moeGatherDequantGemmInt4(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        // Shape / dtype invariants. The kernel itself does no validation
        // so any mismatch silently produces garbage.
        precondition(weight.dtype == .u32, "moeGatherDequantGemmInt4: weight must be u32 packed")
        precondition(
            indices.dtype == .u32,
            "moeGatherDequantGemmInt4: indices must be u32 (\(indices.dtype))")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "moeGatherDequantGemmInt4: input/scales/biases dtype must match")
        precondition(
            out.dtype == input.dtype,
            "moeGatherDequantGemmInt4: out dtype must match input")
        precondition(
            input.elementCount == mTotal * kIn,
            "moeGatherDequantGemmInt4: input has \(input.elementCount) elements, expected \(mTotal * kIn) (mTotal=\(mTotal) * kIn=\(kIn))"
        )
        precondition(
            out.elementCount == mTotal * nOut,
            "moeGatherDequantGemmInt4: out has \(out.elementCount) elements, expected \(mTotal * nOut)"
        )
        precondition(
            indices.elementCount == mTotal,
            "moeGatherDequantGemmInt4: indices has \(indices.elementCount) elements, expected mTotal=\(mTotal)"
        )
        precondition(
            nOut % 32 == 0,
            "moeGatherDequantGemmInt4: nOut (\(nOut)) must be multiple of 32 for bm16 kernel")
        precondition(
            kIn % 32 == 0,
            "moeGatherDequantGemmInt4: kIn (\(kIn)) must be multiple of 32 for bm16 kernel")

        // BM=16, BN=32. The simdgroup-matrix variant uses 64 threads
        // (2 SGs); the MPP / NAX cooperative-tensor variant uses 32
        // threads (1 SG). Canonical metaltile dispatch
        //   dispatch_with_grid(grid=[N/32, ceil(T/16), 1], tg=[TG_W, 1, 1])
        // calls `dispatchThreadgroups` (grid counted in TGs); the
        // generated Swift wrapper calls `dispatchThreads` (grid counted
        // in TOTAL threads) so total threads = grid × tg per axis.
        //
        // FFAI_MOE_BGEMM_MPP=1 opts into the NAX cooperative-tensor
        // path — macOS 26.2+ on gen-≥17 GPU silicon only (M5 Max yes).
        let useMpp = ProcessInfo.processInfo.environment["FFAI_MOE_BGEMM_MPP"] != nil
        let tgWidth = useMpp ? 32 : 64
        let grid = MTLSize(
            width: (nOut / 32) * tgWidth,
            height: (mTotal + 15) / 16,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let mTotalU = UInt32(mTotal)
        let nOutU = UInt32(nOut)
        let kInU = UInt32(kIn)
        let groupSizeU = UInt32(groupSize)

        if useMpp {
            switch input.dtype {
            case .f32:
                MetalTileKernels.mt_moe_gather_qmm_mma_int4_bm16_mpp_f32(
                    x: input.buffer, xOffset: input.offset,
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    indices: indices.buffer, indicesOffset: indices.offset,
                    out: out.buffer, outOffset: out.offset,
                    m_total: mTotalU, n_out: nOutU, k_in: kInU,
                    group_size: groupSizeU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            case .f16:
                MetalTileKernels.mt_moe_gather_qmm_mma_int4_bm16_mpp_f16(
                    x: input.buffer, xOffset: input.offset,
                    w: weight.buffer, wOffset: weight.offset,
                    scales: scales.buffer, scalesOffset: scales.offset,
                    biases: biases.buffer, biasesOffset: biases.offset,
                    indices: indices.buffer, indicesOffset: indices.offset,
                    out: out.buffer, outOffset: out.offset,
                    m_total: mTotalU, n_out: nOutU, k_in: kInU,
                    group_size: groupSizeU,
                    gridSize: grid, threadgroupSize: tg, on: cmd)
            default:
                fatalError(
                    "Ops.moeGatherDequantGemmInt4: MPP variant only emits f32/f16 (got \(input.dtype))"
                )
            }
            return
        }

        switch input.dtype {
        case .f32:
            MetalTileKernels.mt_moe_gather_qmm_mma_int4_bm16_f32(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_moe_gather_qmm_mma_int4_bm16_f16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_moe_gather_qmm_mma_int4_bm16_bf16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeGatherDequantGemmInt4: unsupported dtype \(input.dtype)")
        }
    }

    // MARK: - Batched-T prefill kernels (Qwen3.5 / Qwen3.6 hybrid prefill)

    /// Multi-token Gated Delta Net recurrence over a chunk of `T` tokens.
    ///
    /// Wraps `mt_gated_delta_chunk` — same recurrence math as
    /// `gatedDeltaStep` but runs the per-token loop *inside* the kernel
    /// with the recurrent state kept in per-lane registers across the
    /// entire `T` sweep. A single dispatch replaces `T` independent
    /// `gatedDeltaStep` calls; the state buffer is read once at entry
    /// and written once at exit.
    ///
    /// Layout (matches the MLX-LM `_make_gated_delta_kernel` convention
    /// the metaltile-ffai kernel implements):
    ///   * `q, k`  : `[T, Hk, Dk]`   row-major  (B=1)
    ///   * `v, y`  : `[T, Hv, Dv]`   row-major
    ///   * `g, beta` : `[T, Hv]`     row-major
    ///   * `stateIn / stateOut` : `[Hv, Dv, Dk]` (one state per `hv`)
    ///   * `tLen`  : `[1]` u32 — number of tokens in this chunk (runtime
    ///                scalar, NOT a constexpr; same PSO works for every
    ///                chunk length).
    ///
    /// All input tensors must share the activation dtype `T` — the
    /// kernel is emitted in f32 / f16 / bf16 variants. For Qwen3.5 the
    /// state is f32 (see `GDNStateCache.dtype`); pass q/k/v/g/beta as f32
    /// tensors as well.
    ///
    /// Dispatch: grid `(Dv, Hv)` threadgroups, 32 threads (one simdgroup)
    /// per group — identical to `mt_gated_delta_step` apart from the
    /// runtime `tLen`. `Dk % 32 == 0` invariant applies (max Dk = 256, so
    /// `n_per_t = Dk/32 ≤ 8` register entries per lane).
    public static func gatedDeltaChunk(
        q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
        stateIn: Tensor, into y: Tensor, stateOut: Tensor,
        tLen: Tensor,
        numKeyHeads: Int, numValueHeads: Int,
        keyHeadDim: Int, valueHeadDim: Int,
        on cmd: MTLCommandBuffer
    ) {
        if let reason = OpsValidation.validateGatedDeltaStep(
            keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim,
            numKeyHeads: numKeyHeads, numValueHeads: numValueHeads
        ) {
            preconditionFailure("Ops.gatedDeltaChunk: \(reason)")
        }
        precondition(
            q.dtype == k.dtype && k.dtype == v.dtype
                && v.dtype == g.dtype && g.dtype == beta.dtype
                && beta.dtype == stateIn.dtype && stateIn.dtype == stateOut.dtype
                && stateOut.dtype == y.dtype,
            "Ops.gatedDeltaChunk: every tensor must share dtype")
        precondition(
            tLen.dtype == .u32 && tLen.elementCount == 1,
            "Ops.gatedDeltaChunk: tLen must be a [1] u32 scalar buffer")

        let lanesPerGroup = 32
        let grid = MTLSize(
            width: valueHeadDim * lanesPerGroup,
            height: numValueHeads, depth: 1)
        let tg = MTLSize(width: lanesPerGroup, height: 1, depth: 1)

        switch q.dtype {
        case .f32:
            MetalTileKernels.mt_gated_delta_chunk_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: UInt32(keyHeadDim), dv: UInt32(valueHeadDim),
                hv: UInt32(numValueHeads), hk: UInt32(numKeyHeads),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_gated_delta_chunk_f16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: UInt32(keyHeadDim), dv: UInt32(valueHeadDim),
                hv: UInt32(numValueHeads), hk: UInt32(numKeyHeads),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_gated_delta_chunk_bf16(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: UInt32(keyHeadDim), dv: UInt32(valueHeadDim),
                hv: UInt32(numValueHeads), hk: UInt32(numKeyHeads),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedDeltaChunk: unsupported dtype \(q.dtype)")
        }
    }

    /// Batched-Q causal SDPA prefill via the simdgroup-matrix MMA kernel.
    /// Replaces `T` independent `sdpaDecode` calls (each O(K) per
    /// q-head, T separate dispatches) with one dispatch that tiles the
    /// `T × K` attention matrix 32×16 with simdgroup_matrix MMAs.
    ///
    /// Layout (per `mt_sdpa_prefill_mma` in metaltile-ffai):
    ///   * `q`   : `[n_q_heads, q_len, head_dim]`   row-major (B=1)
    ///   * `k,v` : `[n_kv_heads, k_len, head_dim]`  row-major
    ///   * `out` : `[n_q_heads, q_len, head_dim]`
    ///
    /// `qLen` must be a multiple of 32 (BQ tile). Caller is responsible
    /// for padding (or for splitting the prefill into a 32-aligned chunk
    /// plus a per-token tail that falls back to `sdpaDecode`).
    /// `kLen ≥ qLen` covers the causal mask; the kernel internally bounds
    /// each row's K walk to its absolute position (assumes the last
    /// `qLen` queries correspond to the last `qLen` K rows — i.e. the
    /// queries are appended at the end of the K cache).
    ///
    /// Grid: `(qLen / 32, n_q_heads, 1)` threadgroups, 128 threads per
    /// group (4 simdgroups). Reduction-mode kernel.
    public static func sdpaPrefillMma(
        q: Tensor, k: Tensor, v: Tensor,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        qLen: Int, kLen: Int, scale: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            qLen > 0 && qLen % 32 == 0,
            "Ops.sdpaPrefillMma: qLen \(qLen) must be a positive multiple of 32")
        precondition(
            kLen >= qLen,
            "Ops.sdpaPrefillMma: kLen \(kLen) must be >= qLen \(qLen)")
        precondition(
            nQHeads % nKVHeads == 0,
            "Ops.sdpaPrefillMma: nQHeads \(nQHeads) must be a multiple of nKVHeads \(nKVHeads)")
        precondition(
            q.dtype == k.dtype && k.dtype == v.dtype,
            "Ops.sdpaPrefillMma: q/k/v must share dtype")
        let result = out ?? Tensor.empty(shape: [nQHeads, qLen, headDim], dtype: q.dtype)
        // Grid: (qLen/BQ, n_q_heads, 1) threadgroups, 128 threads per
        // group (4 simdgroups). dispatchThreads counts threads, so width
        // = (qLen/32) * 128, height = n_q_heads.
        let threadsPerGroup = 128
        let bq = 32
        let grid = MTLSize(
            width: (qLen / bq) * threadsPerGroup,
            height: nQHeads, depth: 1)
        let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let gqaFactor = nQHeads / nKVHeads

        switch q.dtype {
        case .f32:
            MetalTileKernels.mt_sdpa_prefill_mma_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                q_len: UInt32(qLen), k_len: UInt32(kLen),
                gqa_factor: UInt32(gqaFactor),
                n_q_heads: UInt32(nQHeads), n_kv_heads: UInt32(nKVHeads),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_sdpa_prefill_mma_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                q_len: UInt32(qLen), k_len: UInt32(kLen),
                gqa_factor: UInt32(gqaFactor),
                n_q_heads: UInt32(nQHeads), n_kv_heads: UInt32(nKVHeads),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_sdpa_prefill_mma_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                q_len: UInt32(qLen), k_len: UInt32(kLen),
                gqa_factor: UInt32(gqaFactor),
                n_q_heads: UInt32(nQHeads), n_kv_heads: UInt32(nKVHeads),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sdpaPrefillMma: unsupported dtype \(q.dtype)")
        }
        return result
    }

    // ─── MoE gather BGEMM int4 — BM=8 MPP variant ────────────────────
    //
    // Half-height counterpart of `moeGatherDequantGemmInt4` tuned for
    // topK=8 decode where `mTotal = 8`. BM=16 wastes 50% of the tile
    // rows on the trailing boundary; BM=8 fills the tile exactly.
    // Uses the MPP destination-only-cooperative path, so the kernel
    // descriptor (M=8, N=32, K=16) clears the simdgroup-scope
    // constraint that the all-cooperative `matmul2d` would reject.
    // Requires macOS 26.2+ / Apple10 GPU (M5 Max).
    /// BM=64 MPP / NAX cooperative-tensor MoE gather BGEMM. Half-height
    /// counterpart of the Bm16 default — fills `ceil(m/64)` tiles
    /// instead of `ceil(m/16)`. Sweet spot for the batched-prefill
    /// regime where `mTotal = T·topK ≥ 64`: at Qwen3.6-A3B T=32 topK=8
    /// mTotal=256 = exactly 4 BM=64 tiles, no boundary waste.
    ///
    /// Requires macOS 26.2+ on Apple10+ GPU (M5 Max) — the MPP path
    /// uses the cooperative-tensor `mpp::tensor_ops::matmul2d` per the
    /// metaltile MPP NAX primitive landing.
    ///
    /// WARNING — KNOWN CORRECTNESS ISSUE (2026-05-21). Dispatch shape
    /// matches the metaltile-std `moe_gather_qmm_mpp_bm64_correctness`
    /// test
    /// (grid = `[ceil(N/64), ceil(M/64), 1]` threadgroups,
    /// tg = `[128, 1, 1]` for 4 SGs WM=WN=2), but `forwardManyEquivalence`
    /// at T=8 mTotal=64 produces top-1 argmax 279 with logit 12.25 vs
    /// reference argmax 52290 logit 17.125 — the tokens are present in
    /// both top-5 lists but with a ~5-logit scale drift. Suspected
    /// bf16-output-scale or buffer-binding mismatch between the
    /// Swift-side wrapper and the kernel's expected layout. Cosine
    /// ≥ 0.999 in the upstream test (f32 output) — bf16 output path
    /// may need its own verification. Use **only** behind
    /// `FFAI_MOE_BGEMM_BM64=1` after re-verifying against the kernel
    /// source. Bm16 default ships the 2.69× T=32 win.
    public static func moeGatherDequantGemmInt4Bm64Mpp(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            weight.dtype == .u32,
            "moeGatherDequantGemmInt4Bm64Mpp: weight must be u32 packed")
        precondition(
            indices.dtype == .u32,
            "moeGatherDequantGemmInt4Bm64Mpp: indices must be u32")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "moeGatherDequantGemmInt4Bm64Mpp: input/scales/biases dtype must match")
        precondition(
            out.dtype == input.dtype,
            "moeGatherDequantGemmInt4Bm64Mpp: out dtype must match input")
        precondition(
            input.elementCount == mTotal * kIn,
            "moeGatherDequantGemmInt4Bm64Mpp: input elements \(input.elementCount) ≠ mTotal·kIn \(mTotal * kIn)"
        )
        precondition(
            out.elementCount == mTotal * nOut,
            "moeGatherDequantGemmInt4Bm64Mpp: out elements \(out.elementCount) ≠ mTotal·nOut \(mTotal * nOut)"
        )
        precondition(
            indices.elementCount == mTotal,
            "moeGatherDequantGemmInt4Bm64Mpp: indices elements \(indices.elementCount) ≠ mTotal \(mTotal)"
        )
        precondition(
            nOut % 64 == 0,
            "moeGatherDequantGemmInt4Bm64Mpp: nOut (\(nOut)) must be multiple of 64 (BN tile)")
        precondition(
            kIn % 32 == 0,
            "moeGatherDequantGemmInt4Bm64Mpp: kIn (\(kIn)) must be multiple of 32 (BK tile)")

        // Per `metaltile-std/src/ffai/moe_mpp_bm64.rs`:
        //   BM = BN = 64, BK = 32. 4 SGs per TG, WM = WN = 2.
        //   Threadgroup size [128, 1, 1]. Grid [N/64, ceil(M/64), 1].
        // dispatchThreads (total threads): width = (N/64)·128.
        let tgWidth = 128
        let grid = MTLSize(
            width: (nOut / 64) * tgWidth,
            height: (mTotal + 63) / 64,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let mTotalU = UInt32(mTotal)
        let nOutU = UInt32(nOut)
        let kInU = UInt32(kIn)
        let groupSizeU = UInt32(groupSize)

        switch input.dtype {
        case .f32:
            MetalTileKernels.mt_moe_gather_qmm_mma_int4_bm64_mpp_f32(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_moe_gather_qmm_mma_int4_bm64_mpp_f16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_moe_gather_qmm_mma_int4_bm64_mpp_bf16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeGatherDequantGemmInt4Bm64Mpp: unsupported dtype \(input.dtype)")
        }
    }

    /// Fused MoE unpermute + weighted scatter-sum back to per-token
    /// outputs. Replaces the per-row `mTotal·2` scalar mul+add loop
    /// (`Tensor.filled([hidden])` × mTotal) on the FFAI batched-prefill
    /// MoE path — one dispatch over T·hidden elements vs the
    /// `mTotal × 2` separate dispatches.
    ///
    /// Layout (per `metaltile-std/src/ffai/moe.rs`):
    ///   - `expertOutputs`: `[k·B·T, hidden]` per-expert dense outputs at
    ///     expert-sorted positions
    ///   - `invPerm`: `[B·T, k]` u32 — where (token, slot) was placed in
    ///     `expertOutputs` (caller's sort step)
    ///   - `topKWeights`: `[B·T, k]` routing weights in `out.dtype`
    ///   - `out`: `[B·T, hidden]` weighted sum across `k` experts
    ///
    /// Geometry: TG=`[128, 1, 1]`, grid=`[B·T, 1, 1]` TGs. Reduction
    /// mode — one TG per token. Bandwidth-bound.
    public static func moeUnpermute(
        expertOutputs: Tensor,
        invPerm: Tensor,
        topKWeights: Tensor,
        into out: Tensor,
        nRows: Int, hidden: Int, k: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            invPerm.dtype == .u32,
            "Ops.moeUnpermute: invPerm must be u32 (got \(invPerm.dtype))")
        precondition(
            expertOutputs.dtype == topKWeights.dtype && topKWeights.dtype == out.dtype,
            "Ops.moeUnpermute: expertOutputs/topKWeights/out must share dtype")
        precondition(
            expertOutputs.elementCount == nRows * k * hidden,
            "Ops.moeUnpermute: expertOutputs has \(expertOutputs.elementCount) elements, expected nRows·k·hidden = \(nRows * k * hidden)"
        )
        precondition(
            invPerm.elementCount == nRows * k,
            "Ops.moeUnpermute: invPerm has \(invPerm.elementCount) elements, expected nRows·k = \(nRows * k)"
        )
        precondition(
            topKWeights.elementCount == nRows * k,
            "Ops.moeUnpermute: topKWeights has \(topKWeights.elementCount) elements, expected nRows·k = \(nRows * k)"
        )
        precondition(
            out.elementCount == nRows * hidden,
            "Ops.moeUnpermute: out has \(out.elementCount) elements, expected nRows·hidden = \(nRows * hidden)"
        )

        let tgWidth = 128
        let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let hiddenU = UInt32(hidden)
        let kU = UInt32(k)
        switch out.dtype {
        case .f32:
            MetalTileKernels.mt_moe_unpermute_f32(
                expert_outputs: expertOutputs.buffer, expert_outputsOffset: expertOutputs.offset,
                inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
                top_k_weights: topKWeights.buffer, top_k_weightsOffset: topKWeights.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU, k: kU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_moe_unpermute_f16(
                expert_outputs: expertOutputs.buffer, expert_outputsOffset: expertOutputs.offset,
                inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
                top_k_weights: topKWeights.buffer, top_k_weightsOffset: topKWeights.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU, k: kU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_moe_unpermute_bf16(
                expert_outputs: expertOutputs.buffer, expert_outputsOffset: expertOutputs.offset,
                inv_perm: invPerm.buffer, inv_permOffset: invPerm.offset,
                top_k_weights: topKWeights.buffer, top_k_weightsOffset: topKWeights.offset,
                out: out.buffer, outOffset: out.offset,
                hidden: hiddenU, k: kU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeUnpermute: unsupported dtype \(out.dtype)")
        }
    }

    public static func moeGatherDequantGemmInt4Bm8(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        indices: Tensor,
        mTotal: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            weight.dtype == .u32,
            "moeGatherDequantGemmInt4Bm8: weight must be u32 packed")
        precondition(
            indices.dtype == .u32,
            "moeGatherDequantGemmInt4Bm8: indices must be u32")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "moeGatherDequantGemmInt4Bm8: input/scales/biases dtype must match")
        precondition(
            out.dtype == input.dtype,
            "moeGatherDequantGemmInt4Bm8: out dtype must match input")
        precondition(
            input.elementCount == mTotal * kIn,
            "moeGatherDequantGemmInt4Bm8: input elements \(input.elementCount) != mTotal*kIn \(mTotal*kIn)"
        )
        precondition(
            out.elementCount == mTotal * nOut,
            "moeGatherDequantGemmInt4Bm8: out elements \(out.elementCount) != mTotal*nOut \(mTotal*nOut)"
        )
        precondition(
            indices.elementCount == mTotal,
            "moeGatherDequantGemmInt4Bm8: indices elements \(indices.elementCount) != mTotal \(mTotal)"
        )
        precondition(
            nOut % 32 == 0,
            "moeGatherDequantGemmInt4Bm8: nOut \(nOut) must be multiple of 32")
        precondition(
            kIn % 32 == 0,
            "moeGatherDequantGemmInt4Bm8: kIn \(kIn) must be multiple of 32")

        // 1 simdgroup per TG, BM=8 → grid Y = ceil(m/8).
        let tgWidth = 32
        let grid = MTLSize(
            width: (nOut / 32) * tgWidth,
            height: (mTotal + 7) / 8,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let mTotalU = UInt32(mTotal)
        let nOutU = UInt32(nOut)
        let kInU = UInt32(kIn)
        let groupSizeU = UInt32(groupSize)

        switch input.dtype {
        case .f32:
            MetalTileKernels.mt_moe_gather_qmm_mma_int4_bm8_mpp_f32(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_moe_gather_qmm_mma_int4_bm8_mpp_f16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_moe_gather_qmm_mma_int4_bm8_mpp_bf16(
                x: input.buffer, xOffset: input.offset,
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                indices: indices.buffer, indicesOffset: indices.offset,
                out: out.buffer, outOffset: out.offset,
                m_total: mTotalU, n_out: nOutU, k_in: kInU,
                group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeGatherDequantGemmInt4Bm8: unsupported dtype \(input.dtype)")
        }
    }

    // ─── Fused GDN prep + recurrence step ─────────────────────────────
    //
    // One dispatch absorbs the per-head q/k RMSNorm + g/beta math +
    // the existing GDN recurrence step, collapsing the 3 host
    // commit+wait pairs in Qwen35GDNMixer.forward down to 1. Same
    // dispatch geometry as `mt_gated_delta_step`:
    //   grid = [dv, B·hv, 1]
    //   tg   = [32, 1, 1]
    // `convOut` layout: `[B, 2·Hk·Dk + Hv·Dv]` — q | k | v slabs.
    // `qNormWeight` / `kNormWeight`: `[Hk·Dk]` (pass 1.0 × invKeyScale
    // for the unweighted path).
    public static func gatedDeltaPrepStep(
        convOut: Tensor,
        aLog: Tensor, dtBias: Tensor,
        aRaw: Tensor, bRaw: Tensor,
        qNormWeight: Tensor, kNormWeight: Tensor,
        stateIn: Tensor, stateOut: Tensor, y: Tensor,
        batchSize: Int, dk: Int, dv: Int, hv: Int, hk: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            convOut.dtype == aLog.dtype && aLog.dtype == dtBias.dtype
                && dtBias.dtype == aRaw.dtype && aRaw.dtype == bRaw.dtype
                && bRaw.dtype == qNormWeight.dtype && qNormWeight.dtype == kNormWeight.dtype
                && kNormWeight.dtype == stateIn.dtype && stateIn.dtype == stateOut.dtype
                && stateOut.dtype == y.dtype,
            "Ops.gatedDeltaPrepStep: every tensor must share dtype")
        precondition(
            dk % 32 == 0,
            "Ops.gatedDeltaPrepStep: dk \(dk) must be multiple of 32")
        precondition(
            dv % 32 == 0,
            "Ops.gatedDeltaPrepStep: dv \(dv) must be multiple of 32")
        precondition(
            hv % hk == 0,
            "Ops.gatedDeltaPrepStep: hv \(hv) must be a multiple of hk \(hk) (GQA)")

        // `dispatchThreads` counts TOTAL threads per axis, so the
        // X axis is `dv` (TGs along X) × `tgWidth` (threads per TG).
        // `tgid_x` inside the kernel therefore ranges 0..dv-1, matching
        // the kernel's `dv_idx = tgid_x` contract. Y axis is one TG
        // per (batch · Hv) slab.
        let tgWidth = 32
        let grid = MTLSize(
            width: dv * tgWidth,
            height: batchSize * hv,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let dkU = UInt32(dk)
        let dvU = UInt32(dv)
        let hvU = UInt32(hv)
        let hkU = UInt32(hk)

        switch convOut.dtype {
        case .f32:
            MetalTileKernels.mt_gated_delta_prep_step_f32(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_gated_delta_prep_step_f16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_gated_delta_prep_step_bf16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedDeltaPrepStep: unsupported dtype \(convOut.dtype)")
        }
    }

    /// Chunked-prefill counterpart to `gatedDeltaPrepStep`. Runs the
    /// fused GDN prep + recurrence over a contiguous range of tokens
    /// in one dispatch instead of looping the single-token kernel,
    /// keeping the per-head recurrent state in registers across the
    /// chunk. `tLen` is a GPU-resident `[1]` u32 scalar so the host
    /// can chain chunked dispatches without a CPU readback for the
    /// chunk-size constant.
    ///
    /// Shape contract (all share dtype except `tLen`):
    ///   - `convOut`   : `[batchSize, tLen, hv·dv]`
    ///   - `aLog`      : `[hv]` per-head log-decay
    ///   - `dtBias`    : `[hv]` per-head delta-t bias
    ///   - `aRaw`/`bRaw` : `[batchSize, tLen, hv]` chunk gates
    ///   - `qNormWeight` / `kNormWeight` : `[dk]` Q-norm / K-norm
    ///     applied before the recurrence
    ///   - `stateIn` / `stateOut` : `[batchSize, hv, dv, dk]` fp32
    ///     recurrent state (read-once / write-once across the chunk)
    ///   - `y`         : `[batchSize, tLen, hv·dv]` output activations
    public static func gatedDeltaPrepChunk(
        convOut: Tensor,
        aLog: Tensor, dtBias: Tensor,
        aRaw: Tensor, bRaw: Tensor,
        qNormWeight: Tensor, kNormWeight: Tensor,
        stateIn: Tensor, stateOut: Tensor, y: Tensor,
        tLen: Tensor,
        batchSize: Int, dk: Int, dv: Int, hv: Int, hk: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            convOut.dtype == aLog.dtype && aLog.dtype == dtBias.dtype
                && dtBias.dtype == aRaw.dtype && aRaw.dtype == bRaw.dtype
                && bRaw.dtype == qNormWeight.dtype && qNormWeight.dtype == kNormWeight.dtype
                && kNormWeight.dtype == stateIn.dtype && stateIn.dtype == stateOut.dtype
                && stateOut.dtype == y.dtype,
            "Ops.gatedDeltaPrepChunk: every tensor must share dtype")
        precondition(
            dk % 32 == 0,
            "Ops.gatedDeltaPrepChunk: dk (\(dk)) must be a multiple of 32")
        precondition(
            dv % 32 == 0,
            "Ops.gatedDeltaPrepChunk: dv (\(dv)) must be a multiple of 32")
        precondition(
            hv % hk == 0,
            "Ops.gatedDeltaPrepChunk: hv (\(hv)) must be a multiple of hk (\(hk)) (GQA)")
        precondition(
            tLen.dtype == .u32 && tLen.elementCount == 1,
            "Ops.gatedDeltaPrepChunk: tLen must be a [1] u32 scalar buffer")
        // One simdgroup per (head, dv index) ; grid sweeps batch · hv
        // along Y. Kernel walks tLen tokens internally.
        let tgWidth = 32
        let grid = MTLSize(
            width: dv * tgWidth,
            height: batchSize * hv,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let dkU = UInt32(dk)
        let dvU = UInt32(dv)
        let hvU = UInt32(hv)
        let hkU = UInt32(hk)
        switch convOut.dtype {
        case .f32:
            MetalTileKernels.mt_gated_delta_prep_chunk_f32(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_gated_delta_prep_chunk_f16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_gated_delta_prep_chunk_bf16(
                conv_out: convOut.buffer, conv_outOffset: convOut.offset,
                a_log: aLog.buffer, a_logOffset: aLog.offset,
                dt_bias: dtBias.buffer, dt_biasOffset: dtBias.offset,
                a_raw: aRaw.buffer, a_rawOffset: aRaw.offset,
                b_raw: bRaw.buffer, b_rawOffset: bRaw.offset,
                q_norm_weight: qNormWeight.buffer, q_norm_weightOffset: qNormWeight.offset,
                k_norm_weight: kNormWeight.buffer, k_norm_weightOffset: kNormWeight.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                y: y.buffer, yOffset: y.offset,
                t_len: tLen.buffer, t_lenOffset: tLen.offset,
                dk: dkU, dv: dvU, hv: hvU, hk: hkU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedDeltaPrepChunk: unsupported dtype \(convOut.dtype)")
        }
    }

    // ─── Dynamic-M batched-prefill int4 qmm ───────────────────────────
    //
    // Host-side driver around `mt_qmm_mma`. The kernel's dispatch grid is
    // `[N/32, ceil(M/32), 1]` × tg `[128, 1, 1]` — M is purely grid-Y
    // sized at runtime, so any `M % 32 == 0` is a clean dispatch. The
    // driver pads ragged `T → mPadded = ceil(T/32) * 32` with zero rows
    // and slices the first `T` output rows on return; zero-row contributions
    // collapse to `0` at every valid output column (`s · Σ q·x + b · Σ x`
    // with `x ≡ 0`).
    //
    // The Rust-side metaltile-ffai counterpart lives in
    // `crates/metaltile-std/src/mlx/quantized_mma_dynamic_m.rs` and has
    // 7/7 GPU correctness cells including the bf16 T=4096 N=K=2048 Qwen3.6
    // production shape (cos 0.999999). This Swift wrapper is the host-side
    // equivalent — same padding contract, same dispatch geometry, dispatched
    // through the regenerated `MetalTileKernels.mt_qmm_mma_*` bindings.
    //
    // Use this when `T > 1` (the batched-prefill case). For T=1 the regular
    // per-token dequant path is faster (the mma kernel pays full weight-tile
    // load cost regardless of how many valid rows are in the M tile).
    public static func dequantGemmDynamicM(
        input: Tensor,
        weight: Tensor, scales: Tensor, biases: Tensor,
        t: Int, nOut: Int, kIn: Int,
        groupSize: Int,
        on cmd: MTLCommandBuffer,
        device: Device,
        into out: Tensor
    ) {
        precondition(
            weight.dtype == .u32,
            "Ops.dequantGemmDynamicM: weight must be u32 packed (got \(weight.dtype))")
        precondition(
            input.dtype == scales.dtype && scales.dtype == biases.dtype,
            "Ops.dequantGemmDynamicM: input/scales/biases must share dtype")
        precondition(
            out.dtype == input.dtype,
            "Ops.dequantGemmDynamicM: out dtype must match input (got \(out.dtype) vs \(input.dtype))"
        )
        precondition(
            input.elementCount == t * kIn,
            "Ops.dequantGemmDynamicM: input has \(input.elementCount) elements, expected t*kIn = \(t * kIn)"
        )
        precondition(
            out.elementCount == t * nOut,
            "Ops.dequantGemmDynamicM: out has \(out.elementCount) elements, expected t*nOut = \(t * nOut)"
        )
        precondition(
            nOut % 32 == 0,
            "Ops.dequantGemmDynamicM: nOut (\(nOut)) must be multiple of 32 (BN tile)")
        precondition(
            kIn % 32 == 0,
            "Ops.dequantGemmDynamicM: kIn (\(kIn)) must be multiple of 32 (BK tile)")

        let mPadded = ((t + 31) / 32) * 32
        let gsPerRow = kIn / groupSize

        // Fast path: T already multiple of 32 — dispatch directly without
        // padding. The hot-loop case for Qwen3.6 prefill (T = 32, 256, 1024,
        // 4096, 32768 are all naturally aligned).
        if t == mPadded {
            dispatchQmmMma(
                weight: weight, scales: scales, biases: biases,
                input: input, output: out,
                m: mPadded, n: nOut, k: kIn, gsPerRow: gsPerRow,
                on: cmd)
            return
        }

        // Slow path: pad X to mPadded with zero rows, dispatch into a
        // padded output, copy the first T rows back to `out`. The trailing
        // (mPadded - T) rows are dispatched but their outputs are discarded.
        //
        // Both the input → xPadded copy and the outPadded → out slice run
        // as MTLBlit on `cmd`. Previously they were host `memcpy`s through
        // `buffer.contents()`, which assumes the input tensor is RESIDENT
        // — fine for chunked-prefill callers that hand it a pre-committed
        // buffer, but UNSAFE for the batched-prefill caller chain where
        // `input` is the in-flight output of an upstream kernel on the
        // same `cmd`. Host-side memcpy of in-flight data reads stale
        // memory and silently produces garbage projections. Blits on
        // `cmd` get Metal hazard-tracking between the prior write and
        // the copy.
        let xPadded = Tensor.empty(
            shape: [mPadded, kIn], dtype: input.dtype,
            device: device)
        let validInBytes = t * kIn * input.dtype.byteSize
        let tailZeroBytes = (mPadded - t) * kIn * input.dtype.byteSize
        let blit = cmd.makeBlitCommandEncoder()!
        blit.copy(
            from: input.buffer, sourceOffset: input.offset,
            to: xPadded.buffer, destinationOffset: xPadded.offset,
            size: validInBytes)
        if tailZeroBytes > 0 {
            blit.fill(
                buffer: xPadded.buffer,
                range: (xPadded.offset + validInBytes)
                    ..< (xPadded.offset + validInBytes + tailZeroBytes),
                value: 0)
        }
        blit.endEncoding()

        let outPadded = Tensor.empty(
            shape: [mPadded, nOut], dtype: input.dtype,
            device: device)
        dispatchQmmMma(
            weight: weight, scales: scales, biases: biases,
            input: xPadded, output: outPadded,
            m: mPadded, n: nOut, k: kIn, gsPerRow: gsPerRow,
            on: cmd)

        // Slice first T rows of outPadded → out (MTLBlit on `cmd`).
        let validOutBytes = t * nOut * input.dtype.byteSize
        let outBlit = cmd.makeBlitCommandEncoder()!
        outBlit.copy(
            from: outPadded.buffer, sourceOffset: outPadded.offset,
            to: out.buffer, destinationOffset: out.offset,
            size: validOutBytes)
        outBlit.endEncoding()
    }

    /// Inner dispatcher for `dequantGemmDynamicM`. Grid is `[N/32, M/32, 1]`
    /// × tg `[128, 1, 1]` (4 SGs WM=WN=2 per the canonical `mt_qmm_mma`).
    /// `dispatchThreads` counts total threads per axis, so grid.x = N/32·128.
    private static func dispatchQmmMma(
        weight: Tensor, scales: Tensor, biases: Tensor,
        input: Tensor, output: Tensor,
        m: Int, n: Int, k: Int, gsPerRow: Int,
        on cmd: MTLCommandBuffer
    ) {
        let tgWidth = 128
        let grid = MTLSize(
            width: (n / 32) * tgWidth,
            height: m / 32,
            depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let kU = UInt32(k)
        let nU = UInt32(n)
        let gsU = UInt32(gsPerRow)
        switch input.dtype {
        case .f32:
            MetalTileKernels.mt_qmm_mma_f32(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_qmm_mma_f16(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_qmm_mma_bf16(
                w: weight.buffer, wOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                x: input.buffer, xOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                k: kU, n: nU, gs_per_row: gsU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.dequantGemmDynamicM: unsupported dtype \(input.dtype)")
        }
    }

    // ─── GPU-side cast to fp32 ────────────────────────────────────────
    //
    // Wraps `mt_cast_to_f32_{bf16,f16,f32}`. Per-element copy with dtype
    // promotion. Used to bridge bf16 / f16 model activations into kernels
    // that require fp32 inputs (the fused GDN prep step is the immediate
    // consumer — it runs against the fp32 recurrence state to avoid the
    // 7-bit-mantissa drift that bf16 state would accumulate). f32→f32 is
    // a memory copy on the GPU, retained for dispatch-table uniformity.
    /// Fused silu + bf16/f16 → fp32 cast in one dispatch. Collapses the
    /// `Ops.silu(...) → Ops.castToF32(...)` two-dispatch chain used in
    /// FFAI's batched-prefill GDN inner loop into one. At Qwen3.6-A3B
    /// T=512 × 30 GDN layers = 15360 dispatches removed per prefill.
    /// Input is bf16 or f16; output must be f32.
    public static func siluCastToF32(
        _ input: Tensor, into output: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            output.dtype == .f32,
            "Ops.siluCastToF32: output dtype must be f32, got \(output.dtype)")
        precondition(
            input.elementCount == output.elementCount,
            "Ops.siluCastToF32: element count mismatch (\(input.elementCount) vs \(output.elementCount))"
        )
        let n = input.elementCount
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch input.dtype {
        case .f16:
            MetalTileKernels.mt_silu_cast_to_f32_f16(
                input: input.buffer, inputOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_silu_cast_to_f32_bf16(
                input: input.buffer, inputOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError(
                "Ops.siluCastToF32: unsupported input dtype \(input.dtype) — bf16 / f16 only")
        }
    }

    /// Cast TWO same-dtype tensors to f32 on the same compute encoder.
    /// Saves one encoder begin/end pair versus two `castToF32` calls.
    /// Used inside the Qwen3 GDN T-loop where the two raw gate inputs
    /// (`aRaw`, `bRaw`) need to be promoted to fp32 every token.
    public static func castToF32Two(
        _ a: Tensor, into outA: Tensor,
        _ b: Tensor, into outB: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            a.dtype == b.dtype,
            "Ops.castToF32Two: inputs must share dtype")
        precondition(
            outA.dtype == .f32 && outB.dtype == .f32,
            "Ops.castToF32Two: outputs must be f32")
        precondition(
            a.elementCount == outA.elementCount,
            "Ops.castToF32Two: a / outA element-count mismatch")
        precondition(
            b.elementCount == outB.elementCount,
            "Ops.castToF32Two: b / outB element-count mismatch")
        let psoName: String
        switch a.dtype {
        case .bf16: psoName = "mt_cast_to_f32_bf16"
        case .f16: psoName = "mt_cast_to_f32_f16"
        case .f32: psoName = "mt_cast_to_f32_f32"
        default: fatalError("Ops.castToF32Two: unsupported dtype \(a.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        @inline(__always)
        func dispatch(_ input: Tensor, _ out: Tensor) {
            enc.setBuffer(input.buffer, offset: input.offset, index: 0)
            enc.setBuffer(out.buffer, offset: out.offset, index: 1)
            let n = input.elementCount
            let tgWidth = min(n, 256)
            enc.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        dispatch(a, outA)
        dispatch(b, outB)
        enc.endEncoding()
    }

    /// Cast THREE same-dtype tensors to f32 on the same compute
    /// encoder. Used in the Qwen3 GDN fused-prep path where
    /// `convAct`, `aRaw`, and `bRaw` all need f32 promotion before
    /// the recurrence kernel.
    public static func castToF32Three(
        _ a: Tensor, into outA: Tensor,
        _ b: Tensor, into outB: Tensor,
        _ c: Tensor, into outC: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            a.dtype == b.dtype && b.dtype == c.dtype,
            "Ops.castToF32Three: all inputs must share dtype")
        precondition(
            outA.dtype == .f32 && outB.dtype == .f32 && outC.dtype == .f32,
            "Ops.castToF32Three: outputs must all be f32")
        precondition(
            a.elementCount == outA.elementCount,
            "Ops.castToF32Three: a / outA element-count mismatch")
        precondition(
            b.elementCount == outB.elementCount,
            "Ops.castToF32Three: b / outB element-count mismatch")
        precondition(
            c.elementCount == outC.elementCount,
            "Ops.castToF32Three: c / outC element-count mismatch")
        let psoName: String
        switch a.dtype {
        case .bf16: psoName = "mt_cast_to_f32_bf16"
        case .f16: psoName = "mt_cast_to_f32_f16"
        case .f32: psoName = "mt_cast_to_f32_f32"
        default: fatalError("Ops.castToF32Three: unsupported dtype \(a.dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        @inline(__always)
        func dispatch(_ input: Tensor, _ out: Tensor) {
            enc.setBuffer(input.buffer, offset: input.offset, index: 0)
            enc.setBuffer(out.buffer, offset: out.offset, index: 1)
            let n = input.elementCount
            let tgWidth = min(n, 256)
            enc.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        dispatch(a, outA)
        dispatch(b, outB)
        dispatch(c, outC)
        enc.endEncoding()
    }

    public static func castToF32(
        _ input: Tensor, into output: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            output.dtype == .f32,
            "Ops.castToF32: output dtype must be f32, got \(output.dtype)")
        precondition(
            input.elementCount == output.elementCount,
            "Ops.castToF32: element count mismatch (\(input.elementCount) vs \(output.elementCount))"
        )
        let n = input.elementCount
        // Pick a TG width that divides n evenly when possible; cap at
        // 256 (well under the Apple TG limit of 1024).
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch input.dtype {
        case .f32:
            MetalTileKernels.mt_cast_to_f32_f32(
                input: input.buffer, inputOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_cast_to_f32_f16(
                input: input.buffer, inputOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_cast_to_f32_bf16(
                input: input.buffer, inputOffset: input.offset,
                out: output.buffer, outOffset: output.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.castToF32: unsupported input dtype \(input.dtype)")
        }
    }

    /// One silu-cast plus two plain casts to f32 on the SAME compute
    /// encoder. Used inside the Qwen3.5 GDN T-loop where, every token,
    /// the input goes through `siluCastToF32` while `aRaw` / `bRaw`
    /// (the gate scalars) go through plain `castToF32`. The wrapper
    /// sets the silu PSO once, dispatches the silu, then switches to
    /// the plain-cast PSO and dispatches the two raw casts. Saves the
    /// encoder begin/end pair between dispatches that would otherwise
    /// fire on every GDN token.
    public static func siluCastF32PlusCastF32Two(
        siluIn: Tensor, into siluOut: Tensor,
        _ a: Tensor, into outA: Tensor,
        _ b: Tensor, into outB: Tensor,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            siluIn.dtype == a.dtype && a.dtype == b.dtype,
            "Ops.siluCastF32PlusCastF32Two: all inputs must share dtype")
        precondition(
            siluOut.dtype == .f32 && outA.dtype == .f32 && outB.dtype == .f32,
            "Ops.siluCastF32PlusCastF32Two: outputs must be f32")
        precondition(
            siluIn.elementCount == siluOut.elementCount,
            "Ops.siluCastF32PlusCastF32Two: silu in/out count mismatch")
        precondition(
            a.elementCount == outA.elementCount,
            "Ops.siluCastF32PlusCastF32Two: a/outA count mismatch")
        precondition(
            b.elementCount == outB.elementCount,
            "Ops.siluCastF32PlusCastF32Two: b/outB count mismatch")
        let siluPso: String
        let castPso: String
        switch siluIn.dtype {
        case .f16:
            siluPso = "mt_silu_cast_to_f32_f16"
            castPso = "mt_cast_to_f32_f16"
        case .bf16:
            siluPso = "mt_silu_cast_to_f32_bf16"
            castPso = "mt_cast_to_f32_bf16"
        default:
            fatalError(
                "Ops.siluCastF32PlusCastF32Two: unsupported dtype \(siluIn.dtype)")
        }
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        @inline(__always)
        func dispatch(_ input: Tensor, _ out: Tensor) {
            enc.setBuffer(input.buffer, offset: input.offset, index: 0)
            enc.setBuffer(out.buffer, offset: out.offset, index: 1)
            let n = input.elementCount
            let tgWidth = min(n, 256)
            enc.dispatchThreads(
                MTLSize(width: n, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        enc.setComputePipelineState(PSOCache.shared.pipelineState(for: siluPso))
        dispatch(siluIn, siluOut)
        enc.setComputePipelineState(PSOCache.shared.pipelineState(for: castPso))
        dispatch(a, outA)
        dispatch(b, outB)
        enc.endEncoding()
    }

    // ─── Fused gated mixer norm ───────────────────────────────────────
    //
    // Wraps `mt_gated_mixer_norm_{f32,f16,bf16}`. Computes
    // `out = rms_norm(y, w) · silu(z)` per row across `[Hv, Dv]` in a
    // single dispatch. Used by FFAI's Qwen3.5 / Qwen3.6 GDN mixer to
    // eliminate the phase-2 host round-trip the legacy path needed
    // for the gated norm + silu math — 30 host commit+waits per
    // Qwen3.6-A3B decode token recovered (one per GDN layer).
    //
    // Tensor contracts (matching the kernel sig):
    //   y       [Hv, Dv]  fp32     — recurrence output (fp32 state)
    //   z       [Hv, Dv]  T        — gate, in model dtype
    //   w       [Dv]      T        — `mixer.norm.weight`, in model dtype
    //   out     [Hv, Dv]  T        — gated output, in model dtype
    //   eps_buf [1]       fp32     — epsilon as a 1-element buffer
    // Constexpr:
    //   n = Dv (must be a multiple of 4 — kernel reads 4 elts/thread)
    // Dispatch:
    //   grid = [Hv * (Dv/4), 1, 1] threads (Hv TGs × Dv/4 threads per TG)
    //   tg   = [Dv/4, 1, 1] threads
    public static func gatedMixerNorm(
        y: Tensor, z: Tensor, weight: Tensor,
        epsBuf: Tensor,
        into out: Tensor,
        numValueHeads: Int, valueHeadDim: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            y.dtype == .f32,
            "Ops.gatedMixerNorm: y must be f32 (got \(y.dtype))")
        precondition(
            z.dtype == weight.dtype && weight.dtype == out.dtype,
            "Ops.gatedMixerNorm: z / weight / out must share dtype")
        precondition(
            epsBuf.dtype == .f32,
            "Ops.gatedMixerNorm: epsBuf must be f32")
        precondition(
            valueHeadDim.isMultiple(of: 4),
            "Ops.gatedMixerNorm: valueHeadDim (\(valueHeadDim)) must be multiple of 4")
        precondition(
            y.elementCount == numValueHeads * valueHeadDim,
            "Ops.gatedMixerNorm: y has \(y.elementCount) elements, expected \(numValueHeads * valueHeadDim)"
        )
        precondition(
            z.elementCount == numValueHeads * valueHeadDim,
            "Ops.gatedMixerNorm: z has \(z.elementCount) elements, expected \(numValueHeads * valueHeadDim)"
        )
        precondition(
            weight.elementCount == valueHeadDim,
            "Ops.gatedMixerNorm: weight has \(weight.elementCount) elements, expected \(valueHeadDim)"
        )
        precondition(
            out.elementCount == numValueHeads * valueHeadDim,
            "Ops.gatedMixerNorm: out has \(out.elementCount) elements, expected \(numValueHeads * valueHeadDim)"
        )

        // One thread per 4 consecutive Dv elements → tpg = Dv / 4.
        // `dispatchThreads` counts total threads per axis, so grid.x =
        // numValueHeads × tpg (Hv TGs × tpg threads per TG).
        let tpg = valueHeadDim / 4
        let grid = MTLSize(width: numValueHeads * tpg, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        let nU = UInt32(valueHeadDim)
        switch out.dtype {
        case .f32:
            MetalTileKernels.mt_gated_mixer_norm_f32(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_gated_mixer_norm_f16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_gated_mixer_norm_bf16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedMixerNorm: unsupported out dtype \(out.dtype)")
        }
    }

    /// T-batched `gatedMixerNorm`. Same kernel, same per-row geometry
    /// (one threadgroup per row, `Dv / 4` threads per TG); the grid X
    /// axis widens from `Hv` to `T · Hv` so a single dispatch covers
    /// every prefill row. Used in the Qwen3.5 GDN mixer's prefill
    /// path where the T-loop over per-token norms was an explicit
    /// performance hotspot.
    ///
    /// Tensor shapes:
    ///   - `y` `[T, Hv, Dv]` fp32 (recurrence output stays fp32)
    ///   - `z` / `out` `[T, Hv, Dv]` in model dtype
    ///   - `weight` `[Dv]` in model dtype
    ///   - `epsBuf` `[1]` fp32
    public static func gatedMixerNormMany(
        y: Tensor, z: Tensor, weight: Tensor,
        epsBuf: Tensor,
        into out: Tensor,
        t: Int, numValueHeads: Int, valueHeadDim: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(t > 0, "Ops.gatedMixerNormMany: t must be positive")
        precondition(
            y.dtype == .f32,
            "Ops.gatedMixerNormMany: y must be f32 (got \(y.dtype))")
        precondition(
            z.dtype == weight.dtype && weight.dtype == out.dtype,
            "Ops.gatedMixerNormMany: z / weight / out must share dtype")
        precondition(
            epsBuf.dtype == .f32,
            "Ops.gatedMixerNormMany: epsBuf must be f32")
        precondition(
            valueHeadDim.isMultiple(of: 4),
            "Ops.gatedMixerNormMany: valueHeadDim (\(valueHeadDim)) must be a multiple of 4")
        let expected = t * numValueHeads * valueHeadDim
        precondition(
            y.elementCount == expected,
            "Ops.gatedMixerNormMany: y has \(y.elementCount), expected T·Hv·Dv = \(expected)")
        precondition(
            z.elementCount == expected,
            "Ops.gatedMixerNormMany: z has \(z.elementCount), expected \(expected)")
        precondition(
            weight.elementCount == valueHeadDim,
            "Ops.gatedMixerNormMany: weight has \(weight.elementCount), expected Dv = \(valueHeadDim)")
        precondition(
            out.elementCount == expected,
            "Ops.gatedMixerNormMany: out has \(out.elementCount), expected \(expected)")
        let tpg = valueHeadDim / 4
        let nRows = t * numValueHeads
        let grid = MTLSize(width: nRows * tpg, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        let nU = UInt32(valueHeadDim)
        switch out.dtype {
        case .f32:
            MetalTileKernels.mt_gated_mixer_norm_f32(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_gated_mixer_norm_f16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_gated_mixer_norm_bf16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: out.buffer, outOffset: out.offset,
                eps_buf: epsBuf.buffer, eps_bufOffset: epsBuf.offset,
                n: nU, gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedMixerNormMany: unsupported out dtype \(out.dtype)")
        }
    }

    // ─── Fused RMSNorm + int4 dequant-GEMV ────────────────────────────
    //
    // Wraps `ffai_rms_norm_qgemv_fast` (RMSNorm-fused) and
    // `ffai_gated_rms_norm_qgemv_int4_fast` (Mamba2-gated-RMSNorm-
    // fused). Both keep the normalised activation in registers and
    // feed it straight into the int4 dequant-GEMV, replacing a
    // 2-dispatch chain (`Ops.rmsNorm` / `Ops.gatedRmsNorm` + `Ops.
    // dequantGemvInt4`) with a single kernel. Saves a full `[in_dim]`
    // DRAM roundtrip on the intermediate `normed` per call.
    //
    // Use at every site where a pre-norm immediately precedes a
    // single int4 qmm projection — most notably the finalNorm + lmHead
    // boundary (one dispatch per token) and the GDN mixer's
    // post-recurrence gatedRmsNorm + outProj pair.

    /// Fused RMSNorm + int4 dequant-GEMV in ONE dispatch via
    /// `ffai_rms_norm_qgemv_fast`. Computes
    ///   `y[row] = Σ_i (q[row,i]·scale + bias) ·
    ///            (x[i] · norm_weight[i] · inv_rms)`
    /// with `inv_rms = rsqrt(mean(x²) + eps)` and the normalised
    /// activation never leaving registers.
    ///
    /// Kernel constraints (`ffai_rms_norm_qgemv_fast`):
    /// - `in_dim` MUST be a multiple of 512 (kernel block size = 512
    ///   K-elements per outer iter).
    /// - `out_dim` MUST be a multiple of 8 (kernel processes 8 output
    ///   rows per TG).
    /// - `group_size` MUST equal 64.
    /// - TPG = 64 (2 simdgroups × 32 lanes).
    public static func rmsNormQgemvInt4Fast(
        x: Tensor, normWeight: Tensor, eps: Float,
        qWeight: Tensor, qScales: Tensor, qBiases: Tensor,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            qWeight.dtype == .u32,
            "Ops.rmsNormQgemvInt4Fast: qWeight must be u32-packed")
        precondition(
            qWeight.shape.count == 2,
            "Ops.rmsNormQgemvInt4Fast: qWeight must be [outDim, inDim/8]")
        let outDim = qWeight.shape[0]
        let packedPerRow = qWeight.shape[1]
        let inDim = packedPerRow * 8
        precondition(
            x.elementCount == inDim,
            "Ops.rmsNormQgemvInt4Fast: x.elementCount \(x.elementCount) ≠ inDim \(inDim)")
        precondition(
            normWeight.elementCount == inDim,
            "Ops.rmsNormQgemvInt4Fast: normWeight.elementCount \(normWeight.elementCount) ≠ inDim \(inDim)")
        precondition(
            out.elementCount == outDim,
            "Ops.rmsNormQgemvInt4Fast: out.elementCount \(out.elementCount) ≠ outDim \(outDim)")
        precondition(
            x.dtype == normWeight.dtype && normWeight.dtype == qScales.dtype
                && qScales.dtype == qBiases.dtype && qBiases.dtype == out.dtype,
            "Ops.rmsNormQgemvInt4Fast: all non-weight tensors must share dtype")
        precondition(
            inDim % 512 == 0,
            "Ops.rmsNormQgemvInt4Fast: in_dim \(inDim) must be a multiple of 512 (fast variant)")
        precondition(
            outDim % 8 == 0,
            "Ops.rmsNormQgemvInt4Fast: out_dim \(outDim) must be a multiple of 8")
        // INVARIANT: kernel pins group_size=64 + TPG=64 for the fast
        // 8-rows-per-TG variant. group_size is checked as a constexpr
        // by the kernel; we re-assert here for early failure.
        let groupSize = 64
        precondition(
            inDim % groupSize == 0,
            "Ops.rmsNormQgemvInt4Fast: in_dim must divide group_size=64")
        // eps as a 1-element f32 buffer.
        var epsValue = eps
        let epsBuf = device.makeBuffer(length: 4)
        memcpy(epsBuf.contents(), &epsValue, 4)
        let tg = MTLSize(width: 64, height: 1, depth: 1)
        let nTiles = outDim / 8
        let grid = MTLSize(width: nTiles * 64, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.ffai_rms_norm_qgemv_fast_f32(
                x: x.buffer, xOffset: x.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                weight: qWeight.buffer, weightOffset: qWeight.offset,
                scales: qScales.buffer, scalesOffset: qScales.offset,
                biases: qBiases.buffer, biasesOffset: qBiases.offset,
                output: out.buffer, outputOffset: out.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_rms_norm_qgemv_fast_f16(
                x: x.buffer, xOffset: x.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                weight: qWeight.buffer, weightOffset: qWeight.offset,
                scales: qScales.buffer, scalesOffset: qScales.offset,
                biases: qBiases.buffer, biasesOffset: qBiases.offset,
                output: out.buffer, outputOffset: out.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_rms_norm_qgemv_fast_bf16(
                x: x.buffer, xOffset: x.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                weight: qWeight.buffer, weightOffset: qWeight.offset,
                scales: qScales.buffer, scalesOffset: qScales.offset,
                biases: qBiases.buffer, biasesOffset: qBiases.offset,
                output: out.buffer, outputOffset: out.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                in_dim: UInt32(inDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rmsNormQgemvInt4Fast: unsupported dtype \(x.dtype)")
        }
    }

    /// Fused Mamba2-style gated RMSNorm + int4 dequant-GEMV in ONE
    /// dispatch via `ffai_gated_rms_norm_qgemv_int4_fast`. Computes
    /// the GDN mixer's `silu(z) · w · rmsnorm(y)` and feeds the result
    /// straight into the int4 outProj — replaces the 2-dispatch
    /// chain (`Ops.gatedRmsNorm` + `Ops.dequantGemvInt4`) with a
    /// single kernel. Saves a full `[Hv·Dv]` DRAM roundtrip on the
    /// intermediate gated-norm output per call.
    ///
    /// Use at the GDN mixer post-recurrence boundary, where the
    /// gated-norm output is the direct input to the outProj projection.
    ///
    /// Kernel constraints:
    /// - `Hv·Dv` (in_dim) MUST be a multiple of 512.
    /// - `Hv·Dv` MUST be ≤ 8192 (kernel TG-memory cap; the gated-norm
    ///   pass stages the full input in TG memory).
    /// - `out_dim` MUST be a multiple of 8.
    /// - `group_size` MUST be 64. TPG = 64.
    public static func gatedRmsNormQgemvInt4Fast(
        y: Tensor,  // [Hv, Dv] f32 (GDN recurrence output)
        z: Tensor,  // [Hv*Dv] T
        normWeight: Tensor,  // [Dv] T
        eps: Float,
        qWeight: Tensor,  // [out_dim, in_dim/8] u32
        qScales: Tensor,  // [out_dim, in_dim/group_size] T
        qBiases: Tensor,
        hv: Int, dv: Int, outDim: Int, groupSize: Int = 64,
        on cmd: MTLCommandBuffer,
        into out: Tensor  // [out_dim] T
    ) {
        precondition(
            y.dtype == .f32,
            "Ops.gatedRmsNormQgemvInt4Fast: y must be f32")
        precondition(
            z.dtype == normWeight.dtype && normWeight.dtype == out.dtype,
            "Ops.gatedRmsNormQgemvInt4Fast: z/weight/out dtype mismatch")
        precondition(
            qWeight.dtype == .u32,
            "Ops.gatedRmsNormQgemvInt4Fast: q_weight must be u32-packed")
        let inDim = hv * dv
        precondition(
            inDim % 512 == 0,
            "Ops.gatedRmsNormQgemvInt4Fast: Hv·Dv (\(inDim)) must be multiple of 512")
        precondition(
            inDim <= 8192,
            "Ops.gatedRmsNormQgemvInt4Fast: Hv·Dv (\(inDim)) must be ≤ 8192 (kernel TG-mem cap)")
        precondition(
            outDim % 8 == 0,
            "Ops.gatedRmsNormQgemvInt4Fast: out_dim must be multiple of 8")
        precondition(
            groupSize == 64,
            "Ops.gatedRmsNormQgemvInt4Fast: group_size must be 64")
        precondition(
            out.elementCount == outDim,
            "Ops.gatedRmsNormQgemvInt4Fast: out element count mismatch")
        let tpg = 64
        let nTiles = outDim / 8
        let grid = MTLSize(width: nTiles * tpg, height: 1, depth: 1)
        let tg = MTLSize(width: tpg, height: 1, depth: 1)
        // eps as a 1-element f32 buffer.
        var epsValue = eps
        let epsBuf = device.makeBuffer(length: 4)
        memcpy(epsBuf.contents(), &epsValue, 4)
        switch out.dtype {
        case .f32:
            MetalTileKernels.ffai_gated_rms_norm_qgemv_int4_fast_f32(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                q_weight: qWeight.buffer, q_weightOffset: qWeight.offset,
                q_scales: qScales.buffer, q_scalesOffset: qScales.offset,
                q_biases: qBiases.buffer, q_biasesOffset: qBiases.offset,
                out: out.buffer, outOffset: out.offset,
                hv: UInt32(hv), dv: UInt32(dv),
                out_dim: UInt32(outDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_gated_rms_norm_qgemv_int4_fast_f16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                q_weight: qWeight.buffer, q_weightOffset: qWeight.offset,
                q_scales: qScales.buffer, q_scalesOffset: qScales.offset,
                q_biases: qBiases.buffer, q_biasesOffset: qBiases.offset,
                out: out.buffer, outOffset: out.offset,
                hv: UInt32(hv), dv: UInt32(dv),
                out_dim: UInt32(outDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_gated_rms_norm_qgemv_int4_fast_bf16(
                y: y.buffer, yOffset: y.offset,
                z: z.buffer, zOffset: z.offset,
                norm_weight: normWeight.buffer, norm_weightOffset: normWeight.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                q_weight: qWeight.buffer, q_weightOffset: qWeight.offset,
                q_scales: qScales.buffer, q_scalesOffset: qScales.offset,
                q_biases: qBiases.buffer, q_biasesOffset: qBiases.offset,
                out: out.buffer, outOffset: out.offset,
                hv: UInt32(hv), dv: UInt32(dv),
                out_dim: UInt32(outDim), group_size: UInt32(groupSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.gatedRmsNormQgemvInt4Fast: unsupported dtype \(out.dtype)")
        }
    }

    // ─── Fused scalar-sigmoid fan-out + FMA ───────────────────────────
    //
    // Wraps `mt_sigmoid_scalar_fma_{f32,f16,bf16}`. Computes
    //   `out[i] = base[i] + sigmoid(gate[0]) * value[i]`
    // for `i in 0..hidden`, broadcasting the scalar `gate` across the
    // `[hidden]` vectors. Used by `Qwen35MoEFFN.forward` to fuse the
    // shared-expert gate's sigmoid + broadcast-mul + residual add into
    // one GPU dispatch, eliminating the `gateLogit.toFloatArray()` host
    // detour and the `commit + wait` that comes with it.
    //
    // Tensor contracts (matching the kernel sig):
    //   gate    [1]       T — scalar logit (raw, not yet sigmoided)
    //   value   [hidden]  T — shared-expert output
    //   base    [hidden]  T — running combine (routed-expert sum)
    //   out     [hidden]  T — gated + accumulated result
    // Dispatch:
    //   grid = [hidden, 1, 1] threads (one per element)
    //   tg   = [tgWidth, 1, 1] threads
    public static func sigmoidScalarFMA(
        gate: Tensor, value: Tensor, base: Tensor,
        into out: Tensor, on cmd: MTLCommandBuffer
    ) {
        precondition(
            gate.dtype == value.dtype && value.dtype == base.dtype && base.dtype == out.dtype,
            "Ops.sigmoidScalarFMA: all tensors must share dtype")
        precondition(
            gate.elementCount == 1,
            "Ops.sigmoidScalarFMA: gate must be [1] (got \(gate.elementCount))")
        precondition(
            value.elementCount == base.elementCount && base.elementCount == out.elementCount,
            "Ops.sigmoidScalarFMA: value / base / out must have matching elementCount")

        let n = value.elementCount
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch out.dtype {
        case .f32:
            MetalTileKernels.mt_sigmoid_scalar_fma_f32(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_sigmoid_scalar_fma_f16(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_sigmoid_scalar_fma_bf16(
                gate: gate.buffer, gateOffset: gate.offset,
                value: value.buffer, valueOffset: value.offset,
                base: base.buffer, baseOffset: base.offset,
                out: out.buffer, outOffset: out.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sigmoidScalarFMA: unsupported dtype \(out.dtype)")
        }
    }

    // ─── Indirect dequant GEMV ────────────────────────────────────────
    //
    // Variant of `dequantGemv` that takes its dispatch shape from a GPU
    // buffer instead of a host-computed `MTLSize`. The buffer holds
    // `MTLDispatchThreadgroupsIndirectArguments` — 3 × u32 = threadgroup
    // counts for x/y/z (NOT thread counts). For the reduction kernel
    // shape used here, write `[outDim, 1, 1]` at the right offset (one
    // threadgroup per output row, `threadgroupSize = 256`).
    //
    // Plumbing for FFAI's GPU-router Day 1 work. The win lands once
    // multiple chained MoE-layer expert dispatches share one indirect
    // buffer + one command buffer (Day 1.5 cross-layer chain). Single-
    // layer use is correctness validation for the indirect path.
    //
    // Only the f16 and bf16 4-bit paths get an indirect emit (Qwen3.6-A3B
    // is bf16 + int4). f32 / 8-bit fall through to a fatalError — no
    // production MoE checkpoint uses those at decode.
    public static func dequantGemvIndirect(
        weight: Tensor, scales: Tensor, biases: Tensor,
        input: Tensor, bits: Int, groupSize: Int = 64,
        indirectBuffer: MTLBuffer, indirectBufferOffset: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor
    ) {
        precondition(
            bits == 4,
            "Ops.dequantGemvIndirect: only 4-bit weights have indirect variants (got bits=\(bits))")
        precondition(weight.shape.count == 2, "dequantGemvIndirect: weight must be 2D")
        precondition(weight.dtype == .u32, "dequantGemvIndirect: weight must be u32 (packed)")
        precondition(
            scales.dtype == input.dtype && biases.dtype == input.dtype,
            "dequantGemvIndirect: scales/biases dtype must match input")
        precondition(
            out.dtype == input.dtype,
            "dequantGemvIndirect: out dtype must match input")
        let outDim = weight.shape[0]
        let packedPerRow = weight.shape[1]
        let inDim = packedPerRow * 32 / bits
        precondition(
            input.elementCount == inDim,
            "dequantGemvIndirect: input \(input.elementCount) ≠ in_dim \(inDim)")
        precondition(
            out.elementCount == outDim,
            "dequantGemvIndirect: out \(out.elementCount) ≠ outDim \(outDim)")
        if let reason = OpsValidation.validateDequantGemv(
            outDim: outDim, inDim: inDim, bits: bits, groupSize: groupSize,
            scalesCount: scales.elementCount, biasesCount: biases.elementCount
        ) {
            preconditionFailure("Ops.dequantGemvIndirect: \(reason)")
        }
        let tgWidth = 256
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        let inDimU = UInt32(inDim)
        let groupSizeU = UInt32(groupSize)

        switch input.dtype {
        case .f16:
            MetalTileKernels.dequant_gemv_int4_f16_indirect(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: out.buffer, outputOffset: out.offset,
                in_dim: inDimU, group_size: groupSizeU,
                indirectBuffer: indirectBuffer, indirectBufferOffset: indirectBufferOffset,
                threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.dequant_gemv_int4_bf16_indirect(
                weight: weight.buffer, weightOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                input: input.buffer, inputOffset: input.offset,
                output: out.buffer, outputOffset: out.offset,
                in_dim: inDimU, group_size: groupSizeU,
                indirectBuffer: indirectBuffer, indirectBufferOffset: indirectBufferOffset,
                threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.dequantGemvIndirect: unsupported input dtype \(input.dtype)")
        }
    }

    // ─── Fused SwiGLU ─────────────────────────────────────────────────
    //
    // Wraps `mt_swiglu_{f32,f16,bf16}`. Computes `out[i] = silu(gate[i]) * up[i]`
    // in one element-wise dispatch — replaces the two-launch `silu` +
    // `mul` chain (`Ops.silu(gate) → Ops.mul(_, up)`). Half the bandwidth
    // on the activation tensor (the intermediate `silu(gate)` stays in
    // registers) plus one fewer commit/encode round-trip per call.
    //
    // Used by FFAI's per-expert MoE SwiGLU + Qwen3 dense MLPs. fp32 path
    // is the canonical reference; f16 / bf16 narrow on store, accumulate
    // in fp32.
    public static func swiglu(
        gate: Tensor, up: Tensor, on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            gate.dtype == up.dtype,
            "Ops.swiglu: gate / up dtype mismatch")
        precondition(
            gate.elementCount == up.elementCount,
            "Ops.swiglu: gate / up size mismatch")
        let result = out ?? Tensor.empty(shape: gate.shape, dtype: gate.dtype)
        precondition(
            result.dtype == gate.dtype && result.elementCount == gate.elementCount,
            "Ops.swiglu: out dtype / size mismatch")
        let n = gate.elementCount
        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch gate.dtype {
        case .f32:
            MetalTileKernels.mt_swiglu_f32(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_swiglu_f16(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_swiglu_bf16(
                gate: gate.buffer, gateOffset: gate.offset,
                up: up.buffer, upOffset: up.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.swiglu: unsupported dtype \(gate.dtype)")
        }
        return result
    }

    /// N independent SwiGLU dispatches sharing ONE compute encoder.
    /// Each `(gate, up, out)` triple is dispatched in sequence after
    /// setting the same `mt_swiglu_*` PSO once. Used by the MoE
    /// per-expert SwiGLU phase at decode (one dispatch per chosen
    /// expert × topK experts × MoE layers) where the encoder
    /// begin/end pairs dominated CPU side.
    public static func swigluMany(
        gates: [Tensor], ups: [Tensor], outs: [Tensor],
        on cmd: MTLCommandBuffer
    ) {
        let n = gates.count
        precondition(
            ups.count == n && outs.count == n,
            "Ops.swigluMany: count mismatch")
        guard n > 0 else { return }
        let dtype = gates[0].dtype
        let psoName: String
        switch dtype {
        case .f32: psoName = "mt_swiglu_f32"
        case .f16: psoName = "mt_swiglu_f16"
        case .bf16: psoName = "mt_swiglu_bf16"
        default: fatalError("Ops.swigluMany: unsupported dtype \(dtype)")
        }
        let pso = PSOCache.shared.pipelineState(for: psoName)
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        for i in 0 ..< n {
            let count = gates[i].elementCount
            precondition(
                ups[i].elementCount == count && outs[i].elementCount == count,
                "Ops.swigluMany: shape mismatch at index \(i)")
            precondition(
                ups[i].dtype == dtype && outs[i].dtype == dtype,
                "Ops.swigluMany: dtype mismatch at index \(i)")
            let tgWidth = min(count, 256)
            enc.setBuffer(gates[i].buffer, offset: gates[i].offset, index: 0)
            enc.setBuffer(ups[i].buffer, offset: ups[i].offset, index: 1)
            enc.setBuffer(outs[i].buffer, offset: outs[i].offset, index: 2)
            enc.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        enc.endEncoding()
    }

    // ─── MoE gather quantised matmul, scalar m1 ────────────────────────
    //
    // Wraps `mt_moe_gather_qmm_int4_{f32,f16,bf16}`. One TG per
    // (output column m, input row t). Each TG resolves the row's
    // expert via the CSR `expert_offsets` and computes one scalar
    // dot-product with `simd_sum` reduction. No cooperative-tensor
    // (MPP) overhead — at decode T=1 the cooperative variants
    // (bm8 / bm16 / bm64) regress because the descriptor + tensor
    // setup dominates the tiny compute. This kernel skips that
    // overhead and walks K with the same per-lane stride pattern as
    // `mt_dequant_gemv_int4`.
    //
    // Inputs:
    //   x              [T_rows, k_in]                 T  — sorted by expert (caller responsibility)
    //   weight         [n_experts, m_out, k_in/8]     u32 (int4 packed)
    //   scales         [n_experts, m_out, k_in/group] T
    //   biases         [n_experts, m_out, k_in/group] T
    //   expertOffsets  [n_experts + 1]                u32 — CSR row offsets
    //   out            [T_rows, m_out]                T
    //
    // Constexpr: k_in, m_out, n_experts, group_size.
    public static func moeGatherDequantGemmInt4M1(
        _ x: Tensor, _ weight: Tensor, _ scales: Tensor, _ biases: Tensor,
        _ expertOffsets: Tensor,
        _ tRows: Int, _ mOut: Int, _ kIn: Int,
        _ nExperts: Int, _ groupSize: Int,
        _ cmd: MTLCommandBuffer, _ out: Tensor
    ) {
        precondition(weight.dtype == .u32, "moeGatherDequantGemmInt4M1: weight must be u32 packed")
        precondition(
            expertOffsets.dtype == .u32, "moeGatherDequantGemmInt4M1: expertOffsets must be u32")
        precondition(
            scales.dtype == x.dtype && biases.dtype == x.dtype && out.dtype == x.dtype,
            "moeGatherDequantGemmInt4M1: dtype mismatch")
        precondition(
            kIn.isMultiple(of: 32), "moeGatherDequantGemmInt4M1: k_in must be multiple of 32")
        precondition(
            kIn.isMultiple(of: groupSize), "moeGatherDequantGemmInt4M1: group_size must divide k_in"
        )
        precondition(
            expertOffsets.elementCount == nExperts + 1,
            "moeGatherDequantGemmInt4M1: expertOffsets must have n_experts+1 entries")
        // Swift binding uses dispatchThreads — total-thread semantics.
        // Kernel wants ONE TG per (output col m, input row t). With TG
        // width 32, grid.x = mOut * 32 gives mOut threadgroups in x; the
        // y axis maps one TG per row.
        let grid = MTLSize(width: mOut * 32, height: tRows, depth: 1)
        let tg = MTLSize(width: 32, height: 1, depth: 1)
        let kInU = UInt32(kIn)
        let mOutU = UInt32(mOut)
        let nExpertsU = UInt32(nExperts)
        let groupSizeU = UInt32(groupSize)
        switch x.dtype {
        case .f32:
            MetalTileKernels.mt_moe_gather_qmm_int4_f32(
                x: x.buffer, xOffset: x.offset,
                weight_packed: weight.buffer, weight_packedOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                expert_offsets: expertOffsets.buffer, expert_offsetsOffset: expertOffsets.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, n_experts: nExpertsU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_moe_gather_qmm_int4_f16(
                x: x.buffer, xOffset: x.offset,
                weight_packed: weight.buffer, weight_packedOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                expert_offsets: expertOffsets.buffer, expert_offsetsOffset: expertOffsets.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, n_experts: nExpertsU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_moe_gather_qmm_int4_bf16(
                x: x.buffer, xOffset: x.offset,
                weight_packed: weight.buffer, weight_packedOffset: weight.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                biases: biases.buffer, biasesOffset: biases.offset,
                expert_offsets: expertOffsets.buffer, expert_offsetsOffset: expertOffsets.offset,
                out: out.buffer, outOffset: out.offset,
                k_in: kInU, m_out: mOutU, n_experts: nExpertsU, group_size: groupSizeU,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.moeGatherDequantGemmInt4M1: unsupported dtype \(x.dtype)")
        }
    }

    /// LayerNorm: `out = (x − mean) / sqrt(var + eps) · weight + bias`.
    /// Backed by `mt_layer_norm_*` (one TG per row, TPG=1024). Used by
    /// vision-transformer encoders (SigLIP / CLIP) and audio frontends —
    /// not on the Qwen3.5/3.6 hot path, but kept here so VLM/audio
    /// models in this package compile against `Ops`.
    public static func layerNorm(
        _ x: Tensor, weight: Tensor, bias: Tensor,
        eps: Float, nRows: Int, rowSize: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            x.elementCount == nRows * rowSize,
            "Ops.layerNorm: x size \(x.elementCount) ≠ nRows*rowSize \(nRows * rowSize)")
        precondition(
            weight.elementCount == rowSize,
            "Ops.layerNorm: weight must be [rowSize]")
        precondition(
            bias.elementCount == rowSize,
            "Ops.layerNorm: bias must be [rowSize]")
        precondition(
            x.dtype == weight.dtype && weight.dtype == bias.dtype,
            "Ops.layerNorm: x/weight/bias dtype mismatch")
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        var epsValue = eps
        let epsBuf = device.makeBuffer(length: 4)
        memcpy(epsBuf.contents(), &epsValue, 4)
        // TPG=1024 per the kernel's reduce-tree contract. One TG per row.
        let tgWidth = 1024
        let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.mt_layer_norm_f32(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                b: bias.buffer, bOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(rowSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mt_layer_norm_f16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                b: bias.buffer, bOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(rowSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mt_layer_norm_bf16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                b: bias.buffer, bOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0, n: UInt32(rowSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.layerNorm: unsupported dtype \(x.dtype)")
        }
        return result
    }

    // ─── Vision: conv2d ──────────────────────────────────────────────

    /// 2D convolution for vision-transformer patch embedding.
    ///
    /// Backed by `metaltile-std/src/ffai/conv2d.rs` (`conv2d_generic`).
    /// One thread per output element `(n, oc, oh, ow)` — a genuine
    /// Grid3D / one-thread-per-output kernel, dispatched flat over the
    /// `batch * out_ch * out_h * out_w` output count, exactly like
    /// `gather`.
    ///
    /// Layouts (NCHW input, OIHW weight — the PyTorch / safetensors
    /// default every VLM checkpoint ships):
    ///   input  `[batch, in_ch,  in_h,  in_w]`
    ///   weight `[out_ch, in_ch, kh,    kw]`
    ///   bias   `[out_ch]`
    ///   out    `[batch, out_ch, out_h, out_w]`
    ///
    /// ## DISPATCH INVARIANTS (conv2d.rs)
    ///   * `out_h = (in_h + 2*pad_h - kh) / stride_h + 1`
    ///   * `out_w = (in_w + 2*pad_w - kw) / stride_w + 1`
    ///   * `input`, `weight`, `bias`, `out` share one floating dtype.
    public static func conv2d(
        input: Tensor, weight: Tensor, bias: Tensor,
        strideH: Int, strideW: Int,
        padH: Int = 0, padW: Int = 0,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            input.shape.count == 4,
            "Ops.conv2d: input must be 4D [batch,in_ch,in_h,in_w] (conv2d.rs)")
        precondition(
            weight.shape.count == 4,
            "Ops.conv2d: weight must be 4D [out_ch,in_ch,kh,kw] (conv2d.rs)")
        precondition(
            bias.shape.count == 1,
            "Ops.conv2d: bias must be 1D [out_ch] (conv2d.rs)")
        precondition(
            input.dtype == weight.dtype && weight.dtype == bias.dtype,
            "Ops.conv2d: input/weight/bias dtype mismatch (conv2d.rs)")
        precondition(
            strideH > 0 && strideW > 0,
            "Ops.conv2d: stride must be positive (conv2d.rs)")

        let batch = input.shape[0]
        let inCh = input.shape[1]
        let inH = input.shape[2]
        let inW = input.shape[3]
        let outCh = weight.shape[0]
        let kh = weight.shape[2]
        let kw = weight.shape[3]
        precondition(
            weight.shape[1] == inCh,
            "Ops.conv2d: weight in_ch \(weight.shape[1]) != input in_ch \(inCh) (conv2d.rs)")
        precondition(
            bias.shape[0] == outCh,
            "Ops.conv2d: bias \(bias.shape[0]) != out_ch \(outCh) (conv2d.rs)")

        // Output spatial dims — the conv2d.rs DISPATCH INVARIANT.
        let outH = (inH + 2 * padH - kh) / strideH + 1
        let outW = (inW + 2 * padW - kw) / strideW + 1
        precondition(
            outH > 0 && outW > 0,
            "Ops.conv2d: degenerate output \(outH)x\(outW) — kernel "
                + "larger than padded input (conv2d.rs)")

        let result =
            out
            ?? Tensor.empty(
                shape: [batch, outCh, outH, outW],
                dtype: input.dtype)
        precondition(
            result.shape == [batch, outCh, outH, outW],
            "Ops.conv2d: out shape \(result.shape) != expected "
                + "\([batch, outCh, outH, outW]) (conv2d.rs)")

        // Grid3D — one thread per output element, dispatched flat.
        let totalThreads = batch * outCh * outH * outW
        let (grid, tg) = elementwiseGrid(totalThreads)

        func dispatch(
            _ fn: (
                MTLBuffer, Int, MTLBuffer, Int, MTLBuffer, Int, MTLBuffer, Int,
                UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                MTLSize, MTLSize, MTLCommandBuffer
            ) -> Void
        ) {
            fn(
                input.buffer, input.offset, weight.buffer, weight.offset,
                bias.buffer, bias.offset, result.buffer, result.offset,
                UInt32(batch), UInt32(inCh), UInt32(inH), UInt32(inW),
                UInt32(outCh), UInt32(outH), UInt32(outW),
                UInt32(kh), UInt32(kw), UInt32(strideH), UInt32(strideW),
                UInt32(padH), UInt32(padW), grid, tg, cmd)
        }
        switch input.dtype {
        case .f32:
            dispatch {
                MetalTileKernels.conv2d_generic_f32(
                    input: $0, inputOffset: $1, weight: $2, weightOffset: $3,
                    bias: $4, biasOffset: $5, out: $6, outOffset: $7,
                    batch: $8, in_ch: $9, in_h: $10, in_w: $11,
                    out_ch: $12, out_h: $13, out_w: $14,
                    kh: $15, kw: $16, stride_h: $17, stride_w: $18,
                    pad_h: $19, pad_w: $20, gridSize: $21, threadgroupSize: $22,
                    on: $23)
            }
        case .f16:
            dispatch {
                MetalTileKernels.conv2d_generic_f16(
                    input: $0, inputOffset: $1, weight: $2, weightOffset: $3,
                    bias: $4, biasOffset: $5, out: $6, outOffset: $7,
                    batch: $8, in_ch: $9, in_h: $10, in_w: $11,
                    out_ch: $12, out_h: $13, out_w: $14,
                    kh: $15, kw: $16, stride_h: $17, stride_w: $18,
                    pad_h: $19, pad_w: $20, gridSize: $21, threadgroupSize: $22,
                    on: $23)
            }
        case .bf16:
            dispatch {
                MetalTileKernels.conv2d_generic_bf16(
                    input: $0, inputOffset: $1, weight: $2, weightOffset: $3,
                    bias: $4, biasOffset: $5, out: $6, outOffset: $7,
                    batch: $8, in_ch: $9, in_h: $10, in_w: $11,
                    out_ch: $12, out_h: $13, out_w: $14,
                    kh: $15, kw: $16, stride_h: $17, stride_w: $18,
                    pad_h: $19, pad_w: $20, gridSize: $21, threadgroupSize: $22,
                    on: $23)
            }
        default:
            fatalError("Ops.conv2d: unsupported dtype \(input.dtype)")
        }
        return result
    }

    // ─── Vision: patch_embed ─────────────────────────────────────────

    /// Fused image-unfold + linear-projection patch embedding for vision
    /// transformers — the ViT stem in one dispatch.
    ///
    /// Backed by `metaltile-std/src/ffai/patch_embed.rs`. One thread per
    /// output element `(patch, h)` — Grid3D / one-thread-per-output,
    /// dispatched flat over `num_patches * hidden`.
    ///
    /// Layouts (NCHW image, flat linear weight):
    ///   image  `[in_ch, in_h, in_w]`  (single image)
    ///   weight `[hidden, in_ch * patch_h * patch_w]`
    ///   bias   `[hidden]`
    ///   out    `[num_patches, hidden]`
    ///
    /// ## DISPATCH INVARIANTS (patch_embed.rs)
    ///   * `in_h` divisible by `patch_h`, `in_w` by `patch_w` — the
    ///     patch grid tiles the image exactly (no padding / clamp).
    ///   * `weight` second dim == `in_ch * patch_h * patch_w`.
    ///   * `image`, `weight`, `bias`, `out` share one floating dtype.
    public static func patchEmbed(
        image: Tensor, weight: Tensor, bias: Tensor,
        patchH: Int, patchW: Int,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            image.shape.count == 3,
            "Ops.patchEmbed: image must be 3D [in_ch,in_h,in_w] (patch_embed.rs)")
        precondition(
            weight.shape.count == 2,
            "Ops.patchEmbed: weight must be 2D [hidden,patch_dim] (patch_embed.rs)")
        precondition(
            bias.shape.count == 1,
            "Ops.patchEmbed: bias must be 1D [hidden] (patch_embed.rs)")
        precondition(
            image.dtype == weight.dtype && weight.dtype == bias.dtype,
            "Ops.patchEmbed: image/weight/bias dtype mismatch (patch_embed.rs)")

        let inCh = image.shape[0]
        let inH = image.shape[1]
        let inW = image.shape[2]
        let hidden = weight.shape[0]
        precondition(
            inH % patchH == 0 && inW % patchW == 0,
            "Ops.patchEmbed: image \(inH)x\(inW) not divisible by patch "
                + "\(patchH)x\(patchW) (patch_embed.rs)")
        let patchDim = inCh * patchH * patchW
        precondition(
            weight.shape[1] == patchDim,
            "Ops.patchEmbed: weight patch_dim \(weight.shape[1]) != "
                + "in_ch*patch_h*patch_w \(patchDim) (patch_embed.rs)")
        precondition(
            bias.shape[0] == hidden,
            "Ops.patchEmbed: bias \(bias.shape[0]) != hidden \(hidden) (patch_embed.rs)")

        let numPatches = (inH / patchH) * (inW / patchW)
        let result =
            out
            ?? Tensor.empty(
                shape: [numPatches, hidden],
                dtype: image.dtype)
        precondition(
            result.shape == [numPatches, hidden],
            "Ops.patchEmbed: out shape \(result.shape) != expected "
                + "\([numPatches, hidden]) (patch_embed.rs)")

        let totalThreads = numPatches * hidden
        let (grid, tg) = elementwiseGrid(totalThreads)
        switch image.dtype {
        case .f32:
            MetalTileKernels.patch_embed_f32(
                image: image.buffer, imageOffset: image.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                in_ch: UInt32(inCh), in_h: UInt32(inH), in_w: UInt32(inW),
                patch_h: UInt32(patchH), patch_w: UInt32(patchW),
                hidden: UInt32(hidden),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.patch_embed_f16(
                image: image.buffer, imageOffset: image.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                in_ch: UInt32(inCh), in_h: UInt32(inH), in_w: UInt32(inW),
                patch_h: UInt32(patchH), patch_w: UInt32(patchW),
                hidden: UInt32(hidden),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.patch_embed_bf16(
                image: image.buffer, imageOffset: image.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                in_ch: UInt32(inCh), in_h: UInt32(inH), in_w: UInt32(inW),
                patch_h: UInt32(patchH), patch_w: UInt32(patchW),
                hidden: UInt32(hidden),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.patchEmbed: unsupported dtype \(image.dtype)")
        }
        return result
    }

    // ─── Vision: rope_2d ─────────────────────────────────────────────

    /// 2D positional RoPE for vision transformers — the "M-RoPE" spatial
    /// component. Splits each head's `head_dim` into two halves: the
    /// first rotated by the token's row index, the second by its column.
    ///
    /// Backed by `metaltile-std/src/ffai/rope_2d.rs`. Grid3D — one thread
    /// per `(token, head, j)` with `j ∈ [0, quarter_dim)`; each thread
    /// emits four output values.
    ///
    /// Layout:
    ///   qk        `[n_tokens, n_heads, head_dim]`
    ///   positions `[n_tokens, 2]`  u32 — `(row, col)` per token
    ///   out       `[n_tokens, n_heads, head_dim]`
    ///
    /// ## DISPATCH INVARIANTS (rope_2d.rs)
    ///   * `head_dim` divisible by 4 (`quarter_dim = head_dim / 4`).
    ///   * `positions` is u32, `[n_tokens, 2]`.
    ///   * `qk` element count == `n_tokens * n_heads * head_dim`.
    public static func rope2D(
        _ qk: Tensor, positions: Tensor,
        nTokens: Int, nHeads: Int, headDim: Int,
        thetaBase: Float,
        on cmd: MTLCommandBuffer,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            headDim % 4 == 0,
            "Ops.rope2D: head_dim \(headDim) must be a multiple of 4 (rope_2d.rs)")
        precondition(
            positions.dtype == .u32,
            "Ops.rope2D: positions must be u32 (rope_2d.rs)")
        precondition(
            positions.elementCount == nTokens * 2,
            "Ops.rope2D: positions count \(positions.elementCount) != "
                + "n_tokens*2 \(nTokens * 2) (rope_2d.rs)")
        precondition(
            qk.elementCount == nTokens * nHeads * headDim,
            "Ops.rope2D: qk count \(qk.elementCount) != "
                + "n_tokens*n_heads*head_dim \(nTokens * nHeads * headDim) (rope_2d.rs)")

        let halfDim = headDim / 2
        let quarterDim = headDim / 4
        let result = out ?? Tensor.empty(shape: qk.shape, dtype: qk.dtype)

        // Grid3D: one thread per (token, head, j).
        let grid = MTLSize(width: nTokens, height: nHeads, depth: quarterDim)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch qk.dtype {
        case .f32:
            MetalTileKernels.ffai_rope_2d_f32(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: positions.buffer, positionsOffset: positions.offset,
                out: result.buffer, outOffset: result.offset,
                n_heads: UInt32(nHeads), head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim), quarter_dim: UInt32(quarterDim),
                theta_base: thetaBase,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_rope_2d_f16(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: positions.buffer, positionsOffset: positions.offset,
                out: result.buffer, outOffset: result.offset,
                n_heads: UInt32(nHeads), head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim), quarter_dim: UInt32(quarterDim),
                theta_base: thetaBase,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_rope_2d_bf16(
                qk: qk.buffer, qkOffset: qk.offset,
                positions: positions.buffer, positionsOffset: positions.offset,
                out: result.buffer, outOffset: result.offset,
                n_heads: UInt32(nHeads), head_dim: UInt32(headDim),
                half_dim: UInt32(halfDim), quarter_dim: UInt32(quarterDim),
                theta_base: thetaBase,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rope2D: unsupported dtype \(qk.dtype)")
        }
        return result
    }

    // ─── Audio front-end / vocoder ───────────────────────────────────

    /// Log-Mel spectrogram — the STT / audio-in front-end. Fuses the
    /// short-time Fourier transform, the Mel filterbank projection and
    /// the log into one dispatch. Wraps `mel_spectrogram_{f32,f16}`.
    ///
    /// Inputs (all share `dtype`, f32 or f16):
    ///   * `audio`      — `[nSamples]` mono waveform, pre-padded so every
    ///                    frame is in-bounds (Whisper reflect-pads by
    ///                    `nFFT/2` on each side before calling this).
    ///   * `window`     — `[nFFT]` analysis window (periodic Hann).
    ///   * `melWeight`  — `[nMels, nFreq]` Mel filterbank, row-major.
    ///
    /// Output: `[nFrames, nMels]` log-Mel.
    ///
    /// The kernel is a plain Grid3D one-thread-per-output kernel
    /// (`KernelMode::Grid3D` in `mel_spectrogram.rs`), so `elementwiseGrid`
    /// is the correct dispatch — NOT a reduction kernel.
    ///
    /// ## DISPATCH INVARIANTS (from `ffai/mel_spectrogram.rs`)
    ///   * `nFreq == nFFT / 2 + 1` — the non-redundant real-FFT bins.
    ///   * `nSamples >= (nFrames - 1) * hopLength + nFFT` — the kernel
    ///     does no bounds check on the frame walk; the caller pre-pads.
    public static func melSpectrogram(
        audio: Tensor, window: Tensor, melWeight: Tensor,
        nFFT: Int, nMels: Int, hopLength: Int,
        nFrames: Int, logEps: Float = 1e-10,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            audio.dtype == window.dtype && audio.dtype == melWeight.dtype,
            "Ops.melSpectrogram: audio/window/melWeight must share dtype")
        precondition(
            audio.dtype == .f32 || audio.dtype == .f16,
            "Ops.melSpectrogram: dtype must be f32 or f16")
        let nFreq = nFFT / 2 + 1
        // Invariants cited from ffai/mel_spectrogram.rs §"Layouts".
        precondition(
            window.elementCount == nFFT,
            "Ops.melSpectrogram: window must be [nFFT=\(nFFT)] "
                + "(ffai/mel_spectrogram.rs)")
        precondition(
            melWeight.elementCount == nMels * nFreq,
            "Ops.melSpectrogram: melWeight must be [nMels, nFreq] "
                + "= [\(nMels), \(nFreq)] (ffai/mel_spectrogram.rs)")
        precondition(
            audio.elementCount >= (nFrames - 1) * hopLength + nFFT,
            "Ops.melSpectrogram: audio too short — kernel does no "
                + "bounds check on the frame walk; pre-pad so "
                + "nSamples >= (nFrames-1)*hop + nFFT "
                + "(ffai/mel_spectrogram.rs)")
        let result = out ?? Tensor.empty(shape: [nFrames, nMels], dtype: audio.dtype)
        // One thread per output element (frame, mel_bin).
        let (grid, tg) = elementwiseGrid(nFrames * nMels)
        switch audio.dtype {
        case .f32:
            MetalTileKernels.mel_spectrogram_f32(
                audio: audio.buffer, audioOffset: audio.offset,
                window: window.buffer, windowOffset: window.offset,
                mel_weight: melWeight.buffer, mel_weightOffset: melWeight.offset,
                out: result.buffer, outOffset: result.offset,
                n_fft: UInt32(nFFT), n_freq: UInt32(nFreq),
                n_mels: UInt32(nMels), hop_length: UInt32(hopLength),
                log_eps: logEps,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mel_spectrogram_f16(
                audio: audio.buffer, audioOffset: audio.offset,
                window: window.buffer, windowOffset: window.offset,
                mel_weight: melWeight.buffer, mel_weightOffset: melWeight.offset,
                out: result.buffer, outOffset: result.offset,
                n_fft: UInt32(nFFT), n_freq: UInt32(nFreq),
                n_mels: UInt32(nMels), hop_length: UInt32(hopLength),
                log_eps: logEps,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.melSpectrogram: unsupported dtype \(audio.dtype)")
        }
        return result
    }

    /// Wide-stride multi-channel 1D convolution — the STT audio patch
    /// embedding. Dense, strided, NCL layout (PyTorch `nn.Conv1d`).
    /// Wraps `audio_conv1d_{f32,f16,bf16}`.
    ///
    /// Inputs (all share `dtype`):
    ///   * `input`  — `[batch, inCh, inLen]`.
    ///   * `weight` — `[outCh, inCh, k]`.
    ///   * `bias`   — `[outCh]`.
    ///
    /// Output: `[batch, outCh, outLen]` with
    /// `outLen = (inLen + 2*pad - k) / stride + 1`.
    ///
    /// Grid3D one-thread-per-output kernel (`KernelMode::Grid3D` in
    /// `audio_conv1d.rs`); `elementwiseGrid` is the correct dispatch.
    ///
    /// ## DISPATCH INVARIANTS (from `ffai/audio_conv1d.rs`)
    ///   * `outLen == (inLen + 2*pad - k) / stride + 1`.
    ///   * `stride >= 1`, `k >= 1`.
    public static func audioConv1d(
        input: Tensor, weight: Tensor, bias: Tensor,
        batch: Int, inCh: Int, inLen: Int, outCh: Int,
        k: Int, stride: Int, pad: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            input.dtype == weight.dtype && input.dtype == bias.dtype,
            "Ops.audioConv1d: input/weight/bias must share dtype")
        precondition(
            stride >= 1 && k >= 1,
            "Ops.audioConv1d: stride and k must be >= 1 "
                + "(ffai/audio_conv1d.rs)")
        precondition(
            input.elementCount == batch * inCh * inLen,
            "Ops.audioConv1d: input must be [batch, inCh, inLen]")
        precondition(
            weight.elementCount == outCh * inCh * k,
            "Ops.audioConv1d: weight must be [outCh, inCh, k]")
        precondition(
            bias.elementCount == outCh,
            "Ops.audioConv1d: bias must be [outCh]")
        let outLen = (inLen + 2 * pad - k) / stride + 1
        precondition(
            outLen >= 1,
            "Ops.audioConv1d: degenerate outLen=\(outLen) "
                + "(ffai/audio_conv1d.rs)")
        let result =
            out
            ?? Tensor.empty(
                shape: [batch, outCh, outLen],
                dtype: input.dtype)
        // One thread per output element (n, oc, op).
        let (grid, tg) = elementwiseGrid(batch * outCh * outLen)
        switch input.dtype {
        case .f32:
            MetalTileKernels.audio_conv1d_f32(
                input: input.buffer, inputOffset: input.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                batch: UInt32(batch), in_ch: UInt32(inCh), in_len: UInt32(inLen),
                out_ch: UInt32(outCh), out_len: UInt32(outLen),
                k: UInt32(k), stride: UInt32(stride), pad: UInt32(pad),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.audio_conv1d_f16(
                input: input.buffer, inputOffset: input.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                batch: UInt32(batch), in_ch: UInt32(inCh), in_len: UInt32(inLen),
                out_ch: UInt32(outCh), out_len: UInt32(outLen),
                k: UInt32(k), stride: UInt32(stride), pad: UInt32(pad),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.audio_conv1d_bf16(
                input: input.buffer, inputOffset: input.offset,
                weight: weight.buffer, weightOffset: weight.offset,
                bias: bias.buffer, biasOffset: bias.offset,
                out: result.buffer, outOffset: result.offset,
                batch: UInt32(batch), in_ch: UInt32(inCh), in_len: UInt32(inLen),
                out_ch: UInt32(outCh), out_len: UInt32(outLen),
                k: UInt32(k), stride: UInt32(stride), pad: UInt32(pad),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.audioConv1d: unsupported dtype \(input.dtype)")
        }
        return result
    }

    /// Inverse-STFT overlap-add — the TTS vocoder waveform-synthesis
    /// tail. Inverse-DFTs each frame, applies the synthesis window,
    /// overlap-adds with COLA normalisation. Wraps
    /// `vocoder_istft_{f32,f16,bf16}`.
    ///
    /// Inputs (all share `dtype`):
    ///   * `specRe` / `specIm` — `[nFrames, nFreq]` real / imaginary
    ///     planes of the predicted STFT.
    ///   * `window` — `[nFFT]` synthesis window.
    ///
    /// Output: `[outLen]` reconstructed waveform with
    /// `outLen = (nFrames - 1) * hopLength + nFFT`.
    ///
    /// Grid3D one-thread-per-output-sample kernel (`KernelMode::Grid3D`
    /// in `vocoder.rs`); `elementwiseGrid` is the correct dispatch.
    ///
    /// ## DISPATCH INVARIANTS (from `ffai/vocoder.rs`)
    ///   * `nFreq == nFFT / 2 + 1`.
    ///   * `outLen == (nFrames - 1) * hopLength + nFFT`.
    public static func vocoderISTFT(
        specRe: Tensor, specIm: Tensor, window: Tensor,
        nFrames: Int, nFFT: Int, hopLength: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            specRe.dtype == specIm.dtype && specRe.dtype == window.dtype,
            "Ops.vocoderISTFT: specRe/specIm/window must share dtype")
        let nFreq = nFFT / 2 + 1
        precondition(
            window.elementCount == nFFT,
            "Ops.vocoderISTFT: window must be [nFFT=\(nFFT)] "
                + "(ffai/vocoder.rs)")
        precondition(
            specRe.elementCount == nFrames * nFreq,
            "Ops.vocoderISTFT: specRe must be [nFrames, nFreq] "
                + "= [\(nFrames), \(nFreq)] (ffai/vocoder.rs)")
        precondition(
            specIm.elementCount == nFrames * nFreq,
            "Ops.vocoderISTFT: specIm must be [nFrames, nFreq] "
                + "= [\(nFrames), \(nFreq)] (ffai/vocoder.rs)")
        let outLen = (nFrames - 1) * hopLength + nFFT
        let result = out ?? Tensor.empty(shape: [outLen], dtype: specRe.dtype)
        // One thread per output sample.
        let (grid, tg) = elementwiseGrid(outLen)
        switch specRe.dtype {
        case .f32:
            MetalTileKernels.vocoder_istft_f32(
                spec_re: specRe.buffer, spec_reOffset: specRe.offset,
                spec_im: specIm.buffer, spec_imOffset: specIm.offset,
                window: window.buffer, windowOffset: window.offset,
                out: result.buffer, outOffset: result.offset,
                n_frames: UInt32(nFrames), n_fft: UInt32(nFFT),
                n_freq: UInt32(nFreq), hop_length: UInt32(hopLength),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.vocoder_istft_f16(
                spec_re: specRe.buffer, spec_reOffset: specRe.offset,
                spec_im: specIm.buffer, spec_imOffset: specIm.offset,
                window: window.buffer, windowOffset: window.offset,
                out: result.buffer, outOffset: result.offset,
                n_frames: UInt32(nFrames), n_fft: UInt32(nFFT),
                n_freq: UInt32(nFreq), hop_length: UInt32(hopLength),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.vocoder_istft_bf16(
                spec_re: specRe.buffer, spec_reOffset: specRe.offset,
                spec_im: specIm.buffer, spec_imOffset: specIm.offset,
                window: window.buffer, windowOffset: window.offset,
                out: result.buffer, outOffset: result.offset,
                n_frames: UInt32(nFrames), n_fft: UInt32(nFFT),
                n_freq: UInt32(nFreq), hop_length: UInt32(hopLength),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.vocoderISTFT: unsupported dtype \(specRe.dtype)")
        }
        return result
    }
}
