// Ops — ergonomic Tensor-based dispatch over MetalTileKernels.
//
// Each op picks the right kernel for the input dtype, fills in default
// grid/threadgroup sizing, encodes on the supplied command buffer, and
// returns a fresh output Tensor (or writes into a caller-supplied one).
//
// Phase 2: only the kernels Llama needs. Adding more in later phases.

import Foundation
import Metal
import MetalTileSwift

public enum Ops {
    public static let device: Device = .shared

    // ─── Sizing helpers ──────────────────────────────────────────────

    /// Threadgroup width for elementwise kernels. Matches what we know
    /// PSO maxTotalThreadsPerThreadgroup will accept on M-series.
    public static let elementwiseTgSize = 256

    private static func elementwiseGrid(_ n: Int) -> (MTLSize, MTLSize) {
        let tg = MTLSize(width: min(elementwiseTgSize, n), height: 1, depth: 1)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        return (grid, tg)
    }

    // ─── Element-wise binary: add ────────────────────────────────────

    public static func add(_ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
                           into out: Tensor? = nil) -> Tensor {
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

    public static func mul(_ a: Tensor, _ b: Tensor, on cmd: MTLCommandBuffer,
                           into out: Tensor? = nil) -> Tensor {
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

    public static func silu(_ x: Tensor, on cmd: MTLCommandBuffer,
                            into out: Tensor? = nil) -> Tensor {
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
    public static func sigmoid(_ x: Tensor, on cmd: MTLCommandBuffer,
                               into out: Tensor? = nil) -> Tensor {
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
    public static func relu(_ x: Tensor, on cmd: MTLCommandBuffer,
                            into out: Tensor? = nil) -> Tensor {
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
    public static func gelu(_ x: Tensor, on cmd: MTLCommandBuffer,
                            into out: Tensor? = nil) -> Tensor {
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
    public static func softplus(_ x: Tensor, on cmd: MTLCommandBuffer,
                                into out: Tensor? = nil) -> Tensor {
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
    public static func gather(table: Tensor, tokenIds: Tensor,
                              on cmd: MTLCommandBuffer,
                              into out: Tensor? = nil) -> Tensor {
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

    /// Cooperative-thread matrix-vector multiply. weight: [out_dim, in_dim],
    /// input: [in_dim], output: [out_dim]. One threadgroup per output row;
    /// threads cooperate on the dot-product reduction.
    public static func gemv(weight: Tensor, input: Tensor,
                            on cmd: MTLCommandBuffer,
                            into out: Tensor? = nil) -> Tensor {
        precondition(weight.shape.count == 2, "gemv: weight must be 2D")
        precondition(input.shape.count == 1, "gemv: input must be 1D")
        precondition(weight.shape[1] == input.shape[0],
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

    /// RMSNorm. x: [n], weight: [n], eps: scalar.
    /// Internally bound as a 1-element f32 buffer.
    /// Reduction kernel — one threadgroup per row.
    public static func rmsNorm(_ x: Tensor, weight: Tensor, eps: Float,
                               on cmd: MTLCommandBuffer,
                               into out: Tensor? = nil) -> Tensor {
        precondition(x.shape == weight.shape, "rmsNorm: weight/x shape mismatch")
        precondition(x.dtype == weight.dtype, "rmsNorm: dtype mismatch")
        let result = out ?? Tensor.empty(shape: x.shape, dtype: x.dtype)
        let n = x.elementCount

        // Kernel-invariant validation. See OpsValidation.swift for the
        // full reasoning + a CI-runnable test of each precondition.
        if let reason = OpsValidation.validateRmsNorm(n: n) {
            preconditionFailure("Ops.rmsNorm: \(reason)")
        }
        dispatchRmsNorm(x: x, weight: weight, result: result,
                        eps: eps, n: n, nRows: 1, on: cmd)
        return result
    }

    /// Multi-row RMSNorm. Input is [nRows, n]; weight is [n] (shared
    /// across all rows). Each row gets its own threadgroup. Used by
    /// Qwen3 to dispatch all per-head q_norm / k_norm in one call
    /// instead of one per head.
    public static func rmsNormRows(_ x: Tensor, weight: Tensor, eps: Float,
                                   nRows: Int, rowSize: Int,
                                   on cmd: MTLCommandBuffer,
                                   into out: Tensor? = nil) -> Tensor {
        precondition(x.elementCount == nRows * rowSize,
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
        dispatchRmsNorm(x: x, weight: weight, result: result,
                        eps: eps, n: rowSize, nRows: nRows, on: cmd)
        return result
    }

    /// Shared RMSNorm dispatch for `rmsNorm` (nRows = 1) and
    /// `rmsNormRows`. Routes by row width:
    ///   • `n ≤ 4096` → `mt_rms_norm`, 4 elements per thread,
    ///     TPG = n / 4 (the fast straight-line kernel).
    ///   • `n > 4096` → `mt_rms_norm_wide`, whose strided loop covers
    ///     any width at a fixed TPG of 1024 (large-hidden models such
    ///     as Gemma 4 31B, hidden 5376).
    /// One threadgroup per row in both cases.
    private static func dispatchRmsNorm(
        x: Tensor, weight: Tensor, result: Tensor,
        eps: Float, n: Int, nRows: Int, on cmd: MTLCommandBuffer
    ) {
        // eps as a 1-element f32 buffer.
        var epsValue = eps
        let epsBuf = device.makeBuffer(length: 4)
        memcpy(epsBuf.contents(), &epsValue, 4)

        let useWide = n > 4096
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

        public init(scaleFactor: Float = 1, lowFreqFactor: Float = 1,
                    highFreqFactor: Float = 4,
                    originalMaxPosition: Float = 1e9) {
            self.scaleFactor = scaleFactor
            self.lowFreqFactor = lowFreqFactor
            self.highFreqFactor = highFreqFactor
            self.originalMaxPosition = originalMaxPosition
        }

        public static let none = RoPEScaling()
    }

    public static func rope(_ qk: Tensor, position: Int, headDim: Int,
                            thetaBase: Float,
                            scaling: RoPEScaling = .none,
                            on cmd: MTLCommandBuffer,
                            into out: Tensor? = nil) -> Tensor {
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
    public static func ropePartial(_ qk: Tensor, position: Int,
                                   headDim: Int, rotaryDim: Int,
                                   thetaBase: Float,
                                   scaling: RoPEScaling = .none,
                                   on cmd: MTLCommandBuffer) {
        precondition(qk.elementCount % headDim == 0,
                     "ropePartial: qk size must be a multiple of headDim")
        precondition(rotaryDim > 0 && rotaryDim <= headDim,
                     "ropePartial: rotaryDim (\(rotaryDim)) must be in 1...headDim (\(headDim))")
        precondition(rotaryDim % 2 == 0,
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
        dequantGather(weight: weight, scales: scales, biases: biases,
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
        precondition(scales.dtype == input.dtype && biases.dtype == input.dtype,
                     "dequantGemv: scales/biases dtype must match input")
        let outDim = weight.shape[0]
        let packedPerRow = weight.shape[1]
        // Storage layout: bytes per row = in_dim * bits / 8.
        // packedPerRow uint32 = (in_dim * bits / 8) / 4 bytes, so:
        //   in_dim = packedPerRow * 32 / bits
        let inDim = packedPerRow * 32 / bits
        precondition(input.elementCount == inDim,
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
        dequantGemv(weight: weight, scales: scales, biases: biases,
                    input: input, bits: 4, groupSize: groupSize,
                    on: cmd, into: out)
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
        precondition(w.dtype == x.dtype && b.dtype == x.dtype
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

    /// Mamba 2 selective-scan single-token decode step. Updates the
    /// per-layer recurrent state `h` in place and writes the output
    /// channel vector `y`. `h` lives in fp32 (state accumulates over
    /// many decode steps; bf16's 7-bit mantissa drifts fast). One
    /// thread per `(head, channel)` — total `nHeads * headDim` threads.
    ///
    /// See `SSMStateCache` for the storage class that wraps the per-layer
    /// `h` buffer; Mamba 2 family files (Phase 5e+) call this through
    /// that cache.
    public static func ssmStep(
        x: Tensor, a: Tensor, b: Tensor, c: Tensor, dt: Tensor,
        state h: Tensor, into y: Tensor,
        nHeads: Int, headDim: Int, stateDim: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(h.dtype == .f32, "Ops.ssmStep: state h must be f32")
        precondition(x.dtype == y.dtype, "Ops.ssmStep: x and y dtype must match")
        precondition(a.dtype == x.dtype && b.dtype == x.dtype
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
    /// `crates/metaltile-std/src/ffai/gated_delta_step.rs`
    /// §"DISPATCH INVARIANTS" and `OpsValidation.validateGatedDeltaStep`):
    /// TPG = 32 (one simdgroup), one threadgroup per `(dv, n)` pair,
    /// `Dk % 32 == 0`, `Hv % Hk == 0`, and `(Dk, Dv, Hk, Hv)` baked in
    /// as compile-time constants.
    ///
    /// `q` / `k` are expected pre-normalised (the GDN block applies the
    /// rmsNorm + scale before calling this — the standard, non-fused
    /// kernel variant). The kernel reads `stateIn` and writes a distinct
    /// `stateOut`; callers double-buffer via `GDNStateCache.swap()`.
    ///
    /// All tensors are f32 — the only emitted kernel dtype. `tSteps` is
    /// the number of sequential recurrence steps packed into this call
    /// (1 for pure decode; > 1 when replaying a short prompt chunk).
    public static func gatedDeltaStep(
        q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
        stateIn: Tensor, into y: Tensor, stateOut: Tensor,
        numKeyHeads: Int, numValueHeads: Int,
        keyHeadDim: Int, valueHeadDim: Int,
        tSteps: Int = 1,
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
        precondition(q.dtype == .f32 && k.dtype == .f32 && v.dtype == .f32
                     && g.dtype == .f32 && beta.dtype == .f32,
                     "Ops.gatedDeltaStep: q/k/v/g/beta must be f32")
        precondition(stateIn.dtype == .f32 && stateOut.dtype == .f32
                     && y.dtype == .f32,
                     "Ops.gatedDeltaStep: state + output tensors must be f32")
        precondition(tSteps >= 1,
                     "Ops.gatedDeltaStep: tSteps must be >= 1 (got \(tSteps))")

        // Dispatch derived from the invariants: 32 threads per group
        // (one simdgroup), one group per (dv_idx, n) pair. The kernel
        // reads tid as the lane (dk_idx), tgid_x as dv_idx, tgid_y as n
        // = batch·Hv + hv. Decode is single-batch so n ranges [0, Hv).
        // Generated bindings use `dispatchThreads`, so the grid is
        // counted in THREADS: width = Dv · 32, height = Hv.
        let lanesPerGroup = 32
        let grid = MTLSize(width: valueHeadDim * lanesPerGroup,
                           height: numValueHeads, depth: 1)
        let tg = MTLSize(width: lanesPerGroup, height: 1, depth: 1)

        // Route to the (Dk, Dv, Hk, Hv)-specialized kernel — each tuple
        // bakes its pointer strides in as compile-time constants.
        switch (keyHeadDim, valueHeadDim, numKeyHeads, numValueHeads) {
        case (192, 128, 4, 4):
            MetalTileKernels.gated_delta_step_192_128_4_4_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                y: y.buffer, yOffset: y.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                t_steps: UInt32(tSteps),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (128, 128, 8, 8):
            MetalTileKernels.gated_delta_step_128_128_8_8_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                y: y.buffer, yOffset: y.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                t_steps: UInt32(tSteps),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (128, 128, 16, 16):
            MetalTileKernels.gated_delta_step_128_128_16_16_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                y: y.buffer, yOffset: y.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                t_steps: UInt32(tSteps),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (128, 128, 16, 32):
            MetalTileKernels.gated_delta_step_128_128_16_32_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                y: y.buffer, yOffset: y.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                t_steps: UInt32(tSteps),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (128, 128, 16, 48):
            MetalTileKernels.gated_delta_step_128_128_16_48_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                y: y.buffer, yOffset: y.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                t_steps: UInt32(tSteps),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case (64, 64, 8, 8):
            MetalTileKernels.gated_delta_step_64_64_8_8_f32(
                q: q.buffer, qOffset: q.offset, k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset, g: g.buffer, gOffset: g.offset,
                beta: beta.buffer, betaOffset: beta.offset,
                state_in: stateIn.buffer, state_inOffset: stateIn.offset,
                y: y.buffer, yOffset: y.offset,
                state_out: stateOut.buffer, state_outOffset: stateOut.offset,
                t_steps: UInt32(tSteps),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            // Unreachable — validateGatedDeltaStep already rejected any
            // tuple without an emitted kernel.
            fatalError("Ops.gatedDeltaStep: unsupported config "
                       + "(\(keyHeadDim),\(valueHeadDim),\(numKeyHeads),\(numValueHeads))")
        }
    }

    public static func argmax(_ logits: Tensor, into out: Tensor,
                              on cmd: MTLCommandBuffer) {
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
        precondition(out.elementCount == 1, "Ops.softmaxCategoricalSample: output must be a single element")
        precondition(temperature.dtype == .f32 && temperature.elementCount == 1,
                     "Ops.softmaxCategoricalSample: temperature must be a 1-element f32 tensor")
        precondition(uniform.dtype == .f32 && uniform.elementCount == 1,
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
            quantizeKVInt4(src: src, weights: weights, scales: scales, biases: biases,
                           nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                           groupSize: groupSize, position: position, on: cmd)
        case 8:
            quantizeKVInt8(src: src, weights: weights, scales: scales, biases: biases,
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
            bulkDequantKVInt4(weights: weights, scales: scales, biases: biases,
                              into: out, nKVHeads: nKVHeads, headDim: headDim,
                              maxSeq: maxSeq, groupSize: groupSize,
                              nPositions: nPositions, on: cmd)
        case 8:
            bulkDequantKVInt8(weights: weights, scales: scales, biases: biases,
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
        precondition(scales.dtype == src.dtype && biases.dtype == src.dtype,
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
        precondition(scales.dtype == out.dtype && biases.dtype == out.dtype,
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
        precondition(scales.dtype == src.dtype && biases.dtype == src.dtype,
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
        precondition(scales.dtype == out.dtype && biases.dtype == out.dtype,
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
    /// memcpy + mid-layer commit/wait pattern from Phase 2.
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
    public static func sdpaDecode(q: Tensor, k: Tensor, v: Tensor,
                                  nQHeads: Int, nKVHeads: Int, headDim: Int,
                                  nKV: Int, kvStride: Int,
                                  scale: Float, on cmd: MTLCommandBuffer,
                                  sinkEnd: Int = 0, windowStart: Int = 0,
                                  into out: Tensor? = nil) -> Tensor {
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
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
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

    // MARK: - AURA (Phase 5d)

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
        precondition(rotation.dtype == .f32 && boundaries.dtype == .f32 && codebook.dtype == .f32,
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
            _ kernel: (MTLBuffer, Int, MTLBuffer, Int, MTLBuffer, Int,
                       MTLBuffer, Int, MTLBuffer, Int, MTLBuffer, Int,
                       UInt32, UInt32, MTLSize, MTLSize, MTLCommandBuffer) -> Void
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
        case (2, .f32):  dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int2_f32(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
        case (2, .f16):  dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int2_f16(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
        case (2, .bf16): dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int2_bf16(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
        case (3, .f32):  dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int3_f32(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
        case (3, .f16):  dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int3_f16(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
        case (3, .bf16): dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int3_bf16(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
        case (4, .f32):  dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int4_f32(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
        case (4, .f16):  dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int4_f16(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
        case (4, .bf16): dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int4_bf16(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
        case (8, .f32):  dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int8_f32(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
        case (8, .f16):  dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int8_f16(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
        case (8, .bf16): dispatchEncode { i, io, r, ro, b, bo, c, co, p, po, n, no, d, pw, g, t, cmd in
            MetalTileKernels.aura_encode_int8_bf16(input: i, inputOffset: io, rotation: r, rotationOffset: ro, boundaries: b, boundariesOffset: bo, codebook: c, codebookOffset: co, packed_out: p, packed_outOffset: po, norms_out: n, norms_outOffset: no, dim: d, packed_width: pw, gridSize: g, threadgroupSize: t, on: cmd) }
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
        precondition(norms.dtype == .f32 && codebook.dtype == .f32,
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

        for h in 0..<nHeads {
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
}
