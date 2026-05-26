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
// OpsLogits — sampling-pipeline logits processors.
//
// Wraps the `logits_*` kernels from `metaltile-std/src/ffai/`:
//   * `logits_temperature_*` — `out[i] = inp[i] / temperature` (Grid3D)
//   * `logits_repetition_penalty_*` — in-place division/multiplication
//     of the logits at the supplied token IDs (Grid3D, one thread per
//     token ID; caller must dedupe — see kernel header).
//   * `logits_topk_mask_*` — masks every logit `< threshold` to
//     `-INFINITY`. Caller computes the K-th-largest threshold on the
//     host (e.g. via `Ops.sort` head + a CPU read), then the kernel
//     does the mask in one Grid3D pass.
//   * `logits_min_p_mask_*` — reduction-mode: one threadgroup per row;
//     computes row max in pass 1, masks `< max * min_p` in pass 2.
//   * `logits_top_p_mask_*` — reduction-mode: bisection over the
//     softmax CDF to find the cumulative-probability cutoff, then mask.
//
// All five are pure elementwise / per-row reductions with no machine-
// freeze hazard: TPG choices are tested-stable (256), the row-mode
// kernels handle any `n` (looped over `lsize`).

import Foundation
import Metal
import MetalTileSwift

extension Ops {

    // ─── temperature ────────────────────────────────────────────────

    /// `out[i] = inp[i] / temperature`. In-place if `out == inp`.
    /// Grid3D mode, one thread per logit. No reduction; caller picks
    /// any `temperature > 0` — `1.0` disables (a no-op divide). If
    /// `temperature == 0` (greedy sampling) callers should branch to
    /// `Ops.argmax` instead — the kernel would divide by zero.
    public static func logitsTemperature(
        _ inp: Tensor, temperature: Float,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        if let reason = OpsValidation.validateLogitsTemperature(
            n: inp.elementCount, temperature: temperature
        ) {
            preconditionFailure("Ops.logitsTemperature: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: inp.shape, dtype: inp.dtype)
        precondition(
            result.shape == inp.shape && result.dtype == inp.dtype,
            "Ops.logitsTemperature: out shape/dtype must match inp")
        let (grid, tg) = elementwiseGrid(inp.elementCount)
        switch inp.dtype {
        case .f32:
            MetalTileKernels.logits_temperature_f32(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                temperature: temperature,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.logits_temperature_f16(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                temperature: temperature,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.logits_temperature_bf16(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                temperature: temperature,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.logitsTemperature: unsupported dtype \(inp.dtype)")
        }
        return result
    }

    // ─── repetition penalty ─────────────────────────────────────────

    /// In-place repetition penalty: for every `t = tokenIds[i]`, multiplies
    /// `logits[t]` by `1/penalty` when `logits[t] > 0` and by `penalty`
    /// when `logits[t] < 0` (matching the HuggingFace formulation). The
    /// kernel is Grid3D over `tokenIds`. **Caller must pre-dedupe**
    /// `tokenIds` — duplicate IDs would each apply the penalty, causing
    /// quadratic blow-up. The wrapper does not enforce dedupe.
    public static func logitsRepetitionPenalty(
        logits: Tensor, tokenIds: Tensor, penalty: Float,
        on cmd: MTLCommandBuffer
    ) {
        precondition(
            tokenIds.dtype == .u32 || tokenIds.dtype == .i32,
            "Ops.logitsRepetitionPenalty: tokenIds must be u32 or i32")
        if let reason = OpsValidation.validateLogitsRepetitionPenalty(
            vocab: logits.elementCount, nTokenIds: tokenIds.elementCount,
            penalty: penalty
        ) {
            preconditionFailure("Ops.logitsRepetitionPenalty: \(reason)")
        }
        // One thread per token-id (TPG=256 is the tested geometry — any
        // size works since the kernel is pure-elementwise).
        let (grid, tg) = elementwiseGrid(tokenIds.elementCount)
        switch logits.dtype {
        case .f32:
            MetalTileKernels.logits_repetition_penalty_f32(
                logits: logits.buffer, logitsOffset: logits.offset,
                token_ids: tokenIds.buffer, token_idsOffset: tokenIds.offset,
                penalty: penalty,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.logits_repetition_penalty_f16(
                logits: logits.buffer, logitsOffset: logits.offset,
                token_ids: tokenIds.buffer, token_idsOffset: tokenIds.offset,
                penalty: penalty,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.logits_repetition_penalty_bf16(
                logits: logits.buffer, logitsOffset: logits.offset,
                token_ids: tokenIds.buffer, token_idsOffset: tokenIds.offset,
                penalty: penalty,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.logitsRepetitionPenalty: unsupported dtype \(logits.dtype)")
        }
    }

    // ─── topK mask ──────────────────────────────────────────────────

    /// Mask every logit `< threshold` to `-INFINITY`. Grid3D mode, one
    /// thread per vocab position. Caller is responsible for computing
    /// `threshold` as the K-th-largest logit value (host-side after a
    /// sort, or via a future on-GPU partial-sort wrapper).
    public static func logitsTopKMask(
        _ inp: Tensor, threshold: Float,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        if let reason = OpsValidation.validateLogitsTopKMask(
            n: inp.elementCount
        ) {
            preconditionFailure("Ops.logitsTopKMask: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: inp.shape, dtype: inp.dtype)
        precondition(
            result.shape == inp.shape && result.dtype == inp.dtype,
            "Ops.logitsTopKMask: out shape/dtype must match inp")
        let (grid, tg) = elementwiseGrid(inp.elementCount)
        switch inp.dtype {
        case .f32:
            MetalTileKernels.logits_topk_mask_f32(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                threshold: threshold,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.logits_topk_mask_f16(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                threshold: threshold,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.logits_topk_mask_bf16(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                threshold: threshold,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.logitsTopKMask: unsupported dtype \(inp.dtype)")
        }
        return result
    }

    // ─── min-p mask ─────────────────────────────────────────────────

    /// Mask every logit below `max * min_p` to `-INFINITY`. One
    /// threadgroup per row (treats the input as `[rows, n]` with `n =
    /// shape.last`). Caller contract: `0 < min_p < 1`. TPG = 256 — the
    /// row loop is `lsize`-based, no fixed multiplicity required.
    public static func logitsMinPMask(
        _ inp: Tensor, minP: Float,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(!inp.shape.isEmpty, "Ops.logitsMinPMask: inp must be non-empty")
        let n = inp.shape.last!
        let rows = inp.elementCount / n
        if let reason = OpsValidation.validateLogitsMinPMask(
            n: n, rows: rows, minP: minP
        ) {
            preconditionFailure("Ops.logitsMinPMask: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: inp.shape, dtype: inp.dtype)
        let tgSize = 256
        let grid = MTLSize(width: rows * tgSize, height: 1, depth: 1)
        let tg = MTLSize(width: tgSize, height: 1, depth: 1)
        switch inp.dtype {
        case .f32:
            MetalTileKernels.logits_min_p_mask_f32(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                n: UInt32(n), min_p: minP,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.logits_min_p_mask_f16(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                n: UInt32(n), min_p: minP,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.logits_min_p_mask_bf16(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                n: UInt32(n), min_p: minP,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.logitsMinPMask: unsupported dtype \(inp.dtype)")
        }
        return result
    }

    // ─── top-p mask ─────────────────────────────────────────────────

    /// Mask the tail of the softmax CDF so the surviving probability
    /// mass is `≥ top_p`. One threadgroup per row, bisection search.
    /// Caller contract: `0 < top_p < 1`.
    public static func logitsTopPMask(
        _ inp: Tensor, topP: Float,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(!inp.shape.isEmpty, "Ops.logitsTopPMask: inp must be non-empty")
        let n = inp.shape.last!
        let rows = inp.elementCount / n
        if let reason = OpsValidation.validateLogitsTopPMask(
            n: n, rows: rows, topP: topP
        ) {
            preconditionFailure("Ops.logitsTopPMask: \(reason)")
        }
        let result = out ?? Tensor.empty(shape: inp.shape, dtype: inp.dtype)
        let tgSize = 256
        let grid = MTLSize(width: rows * tgSize, height: 1, depth: 1)
        let tg = MTLSize(width: tgSize, height: 1, depth: 1)
        switch inp.dtype {
        case .f32:
            MetalTileKernels.logits_top_p_mask_f32(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                n: UInt32(n), top_p: topP,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.logits_top_p_mask_f16(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                n: UInt32(n), top_p: topP,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.logits_top_p_mask_bf16(
                inp: inp.buffer, inpOffset: inp.offset,
                out: result.buffer, outOffset: result.offset,
                n: UInt32(n), top_p: topP,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Ops.logitsTopPMask: unsupported dtype \(inp.dtype)")
        }
        return result
    }
}
