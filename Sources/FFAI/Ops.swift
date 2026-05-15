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
            MetalTileKernels.add_f32(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: result.buffer, cOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.add_f16(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: result.buffer, cOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.add_bf16(
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
            MetalTileKernels.mul_f32(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: result.buffer, cOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.mul_f16(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: result.buffer, cOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.mul_bf16(
                a: a.buffer, aOffset: a.offset,
                b: b.buffer, bOffset: b.offset,
                c: result.buffer, cOffset: result.offset,
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
            MetalTileKernels.silu_f32(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.silu_f16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.silu_bf16(
                a: x.buffer, aOffset: x.offset,
                out: result.buffer, outOffset: result.offset,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.silu: unsupported dtype \(x.dtype)")
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
            MetalTileKernels.gather_f32(
                table: table.buffer, tableOffset: table.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                dim: UInt32(dim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.gather_f16(
                table: table.buffer, tableOffset: table.offset,
                indices: tokenIds.buffer, indicesOffset: tokenIds.offset,
                out: result.buffer, outOffset: result.offset,
                dim: UInt32(dim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.gather_bf16(
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
            MetalTileKernels.gemv_f32(
                weight: weight.buffer, weightOffset: weight.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.gemv_f16(
                weight: weight.buffer, weightOffset: weight.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.gemv_bf16(
                weight: weight.buffer, weightOffset: weight.offset,
                input: input.buffer, inputOffset: input.offset,
                output: result.buffer, outputOffset: result.offset,
                in_dim: UInt32(inDim),
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

        // eps as 1-element f32 buffer
        var epsValue = eps
        let epsBuf = device.makeBuffer(length: 4)
        memcpy(epsBuf.contents(), &epsValue, 4)

        // Reduction kernel: dispatchThreads with grid in THREADS, not
        // threadgroups. For 1 row × 256 cooperating threads, we need
        // grid=(256,1,1) so simd_sum sees a full active simdgroup.
        // _tgid3.x = 0 (1 threadgroup), _lsize3.x = 256 (active threads).
        let tgWidth = 256
        let grid = MTLSize(width: tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.rms_norm_f32(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.rms_norm_f16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.rms_norm_bf16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                n: UInt32(n),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rmsNorm: unsupported dtype \(x.dtype)")
        }
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

        var epsValue = eps
        let epsBuf = device.makeBuffer(length: 4)
        memcpy(epsBuf.contents(), &epsValue, 4)

        // Reduction kernel: one threadgroup per row.
        let tgWidth = 256
        let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        switch x.dtype {
        case .f32:
            MetalTileKernels.rms_norm_f32(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                n: UInt32(rowSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.rms_norm_f16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                n: UInt32(rowSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.rms_norm_bf16(
                x: x.buffer, xOffset: x.offset,
                w: weight.buffer, wOffset: weight.offset,
                out: result.buffer, outOffset: result.offset,
                eps_buf: epsBuf, eps_bufOffset: 0,
                n: UInt32(rowSize),
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.rmsNormRows: unsupported dtype \(x.dtype)")
        }
        return result
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
            MetalTileKernels.rope_f32(
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
            MetalTileKernels.rope_f16(
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
            MetalTileKernels.rope_bf16(
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

    /// MLX-format dequantizing gather (embedding lookup). bits ∈ {4, 8}.
    public static func dequantGather(
        weight: Tensor, scales: Tensor, biases: Tensor,
        tokenIds: Tensor, hidden: Int, bits: Int, groupSize: Int,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(weight.dtype == .u32, "dequantGather: weight must be u32 packed")
        precondition(tokenIds.dtype == .u32, "dequantGather: tokenIds must be u32")
        precondition(scales.dtype == biases.dtype, "dequantGather: scales/biases dtype mismatch")
        precondition(bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8,
                     "dequantGather: bits must be one of 3, 4, 5, 6, or 8")
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
        precondition(bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8,
                     "dequantGemv: bits must be one of 3, 4, 5, 6, or 8")
        let outDim = weight.shape[0]
        let packedPerRow = weight.shape[1]
        // Storage layout: bytes per row = in_dim * bits / 8.
        // packedPerRow uint32 = (in_dim * bits / 8) / 4 bytes, so:
        //   in_dim = packedPerRow * 32 / bits
        let inDim = packedPerRow * 32 / bits
        precondition(input.elementCount == inDim,
                     "dequantGemv: input \(input.elementCount) ≠ in_dim \(inDim)")
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
            MetalTileKernels.argmax_f32(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.argmax_f16(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.argmax_bf16(
                inp: logits.buffer, inpOffset: logits.offset,
                out: out.buffer, outOffset: out.offset,
                n: UInt32(n), gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.argmax: unsupported dtype \(logits.dtype)")
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
    public static func sdpaDecode(q: Tensor, k: Tensor, v: Tensor,
                                  nQHeads: Int, nKVHeads: Int, headDim: Int,
                                  nKV: Int, kvStride: Int,
                                  scale: Float, on cmd: MTLCommandBuffer,
                                  into out: Tensor? = nil) -> Tensor {
        let result = out ?? Tensor.empty(shape: [nQHeads, headDim], dtype: q.dtype)
        let totalThreads = nQHeads * headDim
        let (grid, tg) = elementwiseGrid(totalThreads)
        let headsPerGroup = nQHeads / nKVHeads
        switch q.dtype {
        case .f32:
            MetalTileKernels.sdpa_decode_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.sdpa_decode_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.sdpa_decode_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: UInt32(headDim),
                n_kv: UInt32(nKV),
                kv_stride: UInt32(kvStride),
                heads_per_group: UInt32(headsPerGroup),
                scale: scale,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.sdpaDecode: unsupported dtype \(q.dtype)")
        }
        return result
    }
}
