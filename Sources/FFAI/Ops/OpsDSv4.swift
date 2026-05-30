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
// DeepSeek V4 architecture-specific kernel wrappers. Each Op below
// dispatches one of the metaltile `ffai_dsv4_*` (plus `ffai_moe_router_*`
// and `ffai_sdpa_decode_d512_sink`) kernels with the right
// dtype-suffixed entry point, grid shape, and constexpr bindings.

import Foundation
import Metal
import MetalTileSwift

extension Ops {

    // ─── MoE router ──────────────────────────────────────────────────

    /// DSv4 MoE router scoring: `sqrt(softplus(logits))` with a
    /// `noaux_tc` bias-correction. Returns `(score_unbiased, score_biased)`
    /// side-by-side — downstream top-k uses `score_biased` for
    /// selection and `score_unbiased * routed_scaling_factor` for the
    /// gather weight.
    public static func dsv4MoeRouterSqrtsoftplus(
        logits: Tensor, bias: Tensor, on cmd: MTLCommandBuffer,
        scoreUnbiased: Tensor? = nil, scoreBiased: Tensor? = nil
    ) -> (scoreUnbiased: Tensor, scoreBiased: Tensor) {
        precondition(logits.dtype == .f32, "dsv4MoeRouterSqrtsoftplus: logits must be f32")
        precondition(bias.dtype == .f32, "dsv4MoeRouterSqrtsoftplus: bias must be f32")
        precondition(
            logits.elementCount == bias.elementCount,
            "dsv4MoeRouterSqrtsoftplus: logits / bias element-count mismatch")
        let n = logits.elementCount
        let unbiased = scoreUnbiased ?? Tensor.empty(shape: [n], dtype: .f32)
        let biased = scoreBiased ?? Tensor.empty(shape: [n], dtype: .f32)
        let (grid, tg) = elementwiseGrid(n)
        MetalTileKernels.ffai_moe_router_sqrtsoftplus_f32(
            logits: logits.buffer, logitsOffset: logits.offset,
            bias: bias.buffer, biasOffset: bias.offset,
            score_unbiased: unbiased.buffer, score_unbiasedOffset: unbiased.offset,
            score_biased: biased.buffer, score_biasedOffset: biased.offset,
            gridSize: grid, threadgroupSize: tg, on: cmd)
        return (unbiased, biased)
    }

    // ─── MXFP4 dequant (OCP FP4 e2m1, block size 32) ────────────────

    /// Dequantize a buffer of OCP-spec MXFP4 blocks. Each block
    /// carries 16 packed bytes (32 × 4-bit codes) + one host-extracted
    /// fp32 scale (from the E8M0 raw scale byte).
    public static func dsv4Mxfp4Dequant(
        qsPacked: Tensor, scales: Tensor, lut: Tensor,
        nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(qsPacked.dtype == .u32, "dsv4Mxfp4Dequant: qsPacked must be u32")
        precondition(scales.dtype == .f32, "dsv4Mxfp4Dequant: scales must be f32")
        precondition(lut.dtype == .f32, "dsv4Mxfp4Dequant: lut must be f32")
        precondition(lut.elementCount == 16, "dsv4Mxfp4Dequant: lut must be 16 entries")
        precondition(nValues % 32 == 0, "dsv4Mxfp4Dequant: nValues must be multiple of 32")
        let result = out ?? Tensor.empty(shape: [nValues], dtype: outDtype)
        let (grid, tg) = elementwiseGrid(nValues)
        let n = UInt32(nValues)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_dsv4_mxfp4_dequant_f32(
                qs_packed: qsPacked.buffer, qs_packedOffset: qsPacked.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                lut: lut.buffer, lutOffset: lut.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_dsv4_mxfp4_dequant_f16(
                qs_packed: qsPacked.buffer, qs_packedOffset: qsPacked.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                lut: lut.buffer, lutOffset: lut.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_dsv4_mxfp4_dequant_bf16(
                qs_packed: qsPacked.buffer, qs_packedOffset: qsPacked.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                lut: lut.buffer, lutOffset: lut.offset,
                out: result.buffer, outOffset: result.offset,
                n_values: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("dsv4Mxfp4Dequant: unsupported output dtype \(outDtype)")
        }
        return result
    }

    // ─── FP8 block dequant (e4m3, 128×128 block scales) ─────────────

    /// Dequantize FP8 e4m3 weights using a 256-entry LUT (byte → fp32)
    /// and per-(128×128)-block fp32 scales. Apple has no native FP8
    /// type — the LUT path is the proven fast path.
    public static func dsv4Fp8BlockDequant(
        weightBytes: Tensor, scales: Tensor, lut: Tensor,
        mDim: Int, nDim: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(weightBytes.dtype == .u8, "dsv4Fp8BlockDequant: weightBytes must be u8")
        precondition(scales.dtype == .f32, "dsv4Fp8BlockDequant: scales must be f32")
        precondition(lut.dtype == .f32, "dsv4Fp8BlockDequant: lut must be f32")
        precondition(lut.elementCount == 256, "dsv4Fp8BlockDequant: lut must be 256 entries")
        precondition(mDim % 128 == 0 && nDim % 128 == 0, "dsv4Fp8BlockDequant: dims must be multiples of 128")
        let total = mDim * nDim
        let result = out ?? Tensor.empty(shape: [mDim, nDim], dtype: outDtype)
        let (grid, tg) = elementwiseGrid(total)
        let m = UInt32(mDim)
        let n = UInt32(nDim)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_dsv4_fp8_block_dequant_f32(
                weight_bytes: weightBytes.buffer, weight_bytesOffset: weightBytes.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                fp8_lut: lut.buffer, fp8_lutOffset: lut.offset,
                out: result.buffer, outOffset: result.offset,
                m_dim: m, n_dim: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_dsv4_fp8_block_dequant_f16(
                weight_bytes: weightBytes.buffer, weight_bytesOffset: weightBytes.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                fp8_lut: lut.buffer, fp8_lutOffset: lut.offset,
                out: result.buffer, outOffset: result.offset,
                m_dim: m, n_dim: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_dsv4_fp8_block_dequant_bf16(
                weight_bytes: weightBytes.buffer, weight_bytesOffset: weightBytes.offset,
                scales: scales.buffer, scalesOffset: scales.offset,
                fp8_lut: lut.buffer, fp8_lutOffset: lut.offset,
                out: result.buffer, outOffset: result.offset,
                m_dim: m, n_dim: n,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("dsv4Fp8BlockDequant: unsupported output dtype \(outDtype)")
        }
        return result
    }

    // ─── CSA / HCA compressor pool ───────────────────────────────────

    /// Softmax-gated weighted pool used by CSA (`pool_len=8`) and HCA
    /// (`pool_len=128`) compressors:
    ///   `out[d] = sum_w softmax(gate)[w] * (raw_kv[w, d] + ape[w, d])`
    public static func dsv4CompressorPool(
        rawKv: Tensor, gate: Tensor, ape: Tensor,
        headDim: Int, poolLen: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(gate.dtype == .f32, "dsv4CompressorPool: gate must be f32")
        precondition(rawKv.dtype == outDtype, "dsv4CompressorPool: rawKv dtype must match outDtype")
        precondition(ape.dtype == outDtype, "dsv4CompressorPool: ape dtype must match outDtype")
        let result = out ?? Tensor.empty(shape: [headDim], dtype: outDtype)
        let (grid, tg) = elementwiseGrid(headDim)
        let hd = UInt32(headDim)
        let pl = UInt32(poolLen)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_dsv4_compressor_pool_f32(
                raw_kv: rawKv.buffer, raw_kvOffset: rawKv.offset,
                gate: gate.buffer, gateOffset: gate.offset,
                ape: ape.buffer, apeOffset: ape.offset,
                compressed: result.buffer, compressedOffset: result.offset,
                head_dim: hd, pool_len: pl,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_dsv4_compressor_pool_f16(
                raw_kv: rawKv.buffer, raw_kvOffset: rawKv.offset,
                gate: gate.buffer, gateOffset: gate.offset,
                ape: ape.buffer, apeOffset: ape.offset,
                compressed: result.buffer, compressedOffset: result.offset,
                head_dim: hd, pool_len: pl,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_dsv4_compressor_pool_bf16(
                raw_kv: rawKv.buffer, raw_kvOffset: rawKv.offset,
                gate: gate.buffer, gateOffset: gate.offset,
                ape: ape.buffer, apeOffset: ape.offset,
                compressed: result.buffer, compressedOffset: result.offset,
                head_dim: hd, pool_len: pl,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("dsv4CompressorPool: unsupported output dtype \(outDtype)")
        }
        return result
    }

    // ─── mHC residual mix ────────────────────────────────────────────

    /// mHC pre-mix — `out[d] = sum_c pre[c] * state[c, d]`.
    public static func dsv4MhcPre(
        state: Tensor, pre: Tensor,
        hiddenDim: Int, nCh: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(pre.dtype == .f32, "dsv4MhcPre: pre must be f32")
        precondition(state.dtype == outDtype, "dsv4MhcPre: state dtype must match outDtype")
        let result = out ?? Tensor.empty(shape: [hiddenDim], dtype: outDtype)
        let (grid, tg) = elementwiseGrid(hiddenDim)
        let hd = UInt32(hiddenDim)
        let nc = UInt32(nCh)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_dsv4_mhc_pre_f32(
                state: state.buffer, stateOffset: state.offset,
                pre: pre.buffer, preOffset: pre.offset,
                out: result.buffer, outOffset: result.offset,
                hidden_dim: hd, n_ch: nc,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_dsv4_mhc_pre_f16(
                state: state.buffer, stateOffset: state.offset,
                pre: pre.buffer, preOffset: pre.offset,
                out: result.buffer, outOffset: result.offset,
                hidden_dim: hd, n_ch: nc,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_dsv4_mhc_pre_bf16(
                state: state.buffer, stateOffset: state.offset,
                pre: pre.buffer, preOffset: pre.offset,
                out: result.buffer, outOffset: result.offset,
                hidden_dim: hd, n_ch: nc,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("dsv4MhcPre: unsupported output dtype \(outDtype)")
        }
        return result
    }

    /// mHC post-mix — in-place outer-product accumulate:
    ///   `state[c, d] += post[c] * sublayer_out[d]` for c ∈ [0, n_ch).
    public static func dsv4MhcPost(
        sublayerOut: Tensor, post: Tensor, state: Tensor,
        hiddenDim: Int, nCh: Int,
        on cmd: MTLCommandBuffer
    ) {
        precondition(post.dtype == .f32, "dsv4MhcPost: post must be f32")
        precondition(state.dtype == sublayerOut.dtype, "dsv4MhcPost: state/sublayerOut dtype mismatch")
        let (grid, tg) = elementwiseGrid(hiddenDim)
        let hd = UInt32(hiddenDim)
        let nc = UInt32(nCh)
        switch state.dtype {
        case .f32:
            MetalTileKernels.ffai_dsv4_mhc_post_f32(
                sublayer_out: sublayerOut.buffer, sublayer_outOffset: sublayerOut.offset,
                post: post.buffer, postOffset: post.offset,
                state: state.buffer, stateOffset: state.offset,
                hidden_dim: hd, n_ch: nc,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_dsv4_mhc_post_f16(
                sublayer_out: sublayerOut.buffer, sublayer_outOffset: sublayerOut.offset,
                post: post.buffer, postOffset: post.offset,
                state: state.buffer, stateOffset: state.offset,
                hidden_dim: hd, n_ch: nc,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_dsv4_mhc_post_bf16(
                sublayer_out: sublayerOut.buffer, sublayer_outOffset: sublayerOut.offset,
                post: post.buffer, postOffset: post.offset,
                state: state.buffer, stateOffset: state.offset,
                hidden_dim: hd, n_ch: nc,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("dsv4MhcPost: unsupported state dtype \(state.dtype)")
        }
    }

    // ─── Lightning Indexer ───────────────────────────────────────────

    /// Per-position aggregate score:
    ///   `score[t] = sum_h w[h] * ReLU(q_idx[h] · k_idx[t, h])`.
    public static func dsv4IndexerScore(
        qIdx: Tensor, kIdx: Tensor, w: Tensor,
        nHeads: Int, dIdx: Int, nKv: Int,
        on cmd: MTLCommandBuffer, into score: Tensor? = nil
    ) -> Tensor {
        precondition(w.dtype == .f32, "dsv4IndexerScore: w must be f32")
        precondition(qIdx.dtype == kIdx.dtype, "dsv4IndexerScore: qIdx / kIdx dtype mismatch")
        let result = score ?? Tensor.empty(shape: [nKv], dtype: .f32)
        let (grid, tg) = elementwiseGrid(nKv)
        let nh = UInt32(nHeads)
        let di = UInt32(dIdx)
        let nk = UInt32(nKv)
        switch qIdx.dtype {
        case .f32:
            MetalTileKernels.ffai_dsv4_indexer_score_f32(
                q_idx: qIdx.buffer, q_idxOffset: qIdx.offset,
                k_idx: kIdx.buffer, k_idxOffset: kIdx.offset,
                w: w.buffer, wOffset: w.offset,
                score: result.buffer, scoreOffset: result.offset,
                n_heads: nh, d_idx: di, n_kv: nk,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_dsv4_indexer_score_f16(
                q_idx: qIdx.buffer, q_idxOffset: qIdx.offset,
                k_idx: kIdx.buffer, k_idxOffset: kIdx.offset,
                w: w.buffer, wOffset: w.offset,
                score: result.buffer, scoreOffset: result.offset,
                n_heads: nh, d_idx: di, n_kv: nk,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_dsv4_indexer_score_bf16(
                q_idx: qIdx.buffer, q_idxOffset: qIdx.offset,
                k_idx: kIdx.buffer, k_idxOffset: kIdx.offset,
                w: w.buffer, wOffset: w.offset,
                score: result.buffer, scoreOffset: result.offset,
                n_heads: nh, d_idx: di, n_kv: nk,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("dsv4IndexerScore: unsupported qIdx dtype \(qIdx.dtype)")
        }
        return result
    }

    /// Single-block bitonic top-K over `score[n_kv]`. Returns the
    /// `k` largest entries' original cache-position indices as `u32`.
    /// Caller must satisfy `n_kv <= 1024`.
    public static func dsv4IndexerTopkBlock(
        score: Tensor, nKv: Int, k: Int,
        on cmd: MTLCommandBuffer, into outIndices: Tensor? = nil
    ) -> Tensor {
        precondition(score.dtype == .f32, "dsv4IndexerTopkBlock: score must be f32")
        precondition(nKv <= 1024, "dsv4IndexerTopkBlock: nKv > 1024 needs the multi-block variant")
        precondition(k <= nKv, "dsv4IndexerTopkBlock: k must be <= nKv")
        let result = outIndices ?? Tensor.empty(shape: [k], dtype: .u32)
        MetalTileKernels.ffai_dsv4_indexer_topk_block_f32(
            score: score.buffer, scoreOffset: score.offset,
            out_indices: result.buffer, out_indicesOffset: result.offset,
            n_kv: UInt32(nKv), k: UInt32(k),
            gridSize: MTLSize(width: 1, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: 256, height: 1, depth: 1),
            on: cmd)
        return result
    }

    // ─── SDPA: HCA dense + attn_sink ────────────────────────────────

    /// Single-token SDPA decode for `head_dim == 512` with a per-head
    /// learnable softmax sink term. Used by DSv4 full-attention layers
    /// (0, 1, 42) and HCA dense layers.
    public static func dsv4SdpaDecodeD512Sink(
        q: Tensor, k: Tensor, v: Tensor, sinkLogit: Tensor,
        nQHeads: Int, nKvHeads: Int, headDim: Int, nKv: Int, kvStride: Int,
        scale: Float, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(headDim == 512, "dsv4SdpaDecodeD512Sink: head_dim must be 512")
        precondition(nQHeads % nKvHeads == 0, "dsv4SdpaDecodeD512Sink: GQA must be integer")
        precondition(sinkLogit.dtype == .f32, "dsv4SdpaDecodeD512Sink: sinkLogit must be f32")
        let result = out ?? Tensor.empty(shape: [nQHeads, headDim], dtype: outDtype)
        let gridSize = MTLSize(width: nQHeads, height: 1, depth: 1)
        let tg = MTLSize(width: 512, height: 1, depth: 1)
        let hd = UInt32(headDim)
        let nk = UInt32(nKv)
        let ks = UInt32(kvStride)
        let hpg = UInt32(nQHeads / nKvHeads)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_sdpa_decode_d512_sink_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                sink_logit: sinkLogit.buffer, sink_logitOffset: sinkLogit.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_kv: nk, kv_stride: ks, heads_per_group: hpg, scale: scale,
                gridSize: gridSize, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_sdpa_decode_d512_sink_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                sink_logit: sinkLogit.buffer, sink_logitOffset: sinkLogit.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_kv: nk, kv_stride: ks, heads_per_group: hpg, scale: scale,
                gridSize: gridSize, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_sdpa_decode_d512_sink_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                sink_logit: sinkLogit.buffer, sink_logitOffset: sinkLogit.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_kv: nk, kv_stride: ks, heads_per_group: hpg, scale: scale,
                gridSize: gridSize, threadgroupSize: tg, on: cmd)
        default:
            fatalError("dsv4SdpaDecodeD512Sink: unsupported output dtype \(outDtype)")
        }
        return result
    }

    // ─── SDPA: CSA sparse-gather ────────────────────────────────────

    /// Sparse-gather SDPA decode for `head_dim == 512`. Attention is
    /// taken over the cache positions listed in `selectedIndices`
    /// (typically Lightning Indexer top-K unioned with the trailing
    /// sliding window).
    public static func dsv4CsaSdpaDecode(
        q: Tensor, k: Tensor, v: Tensor, selectedIndices: Tensor,
        nQHeads: Int, nKvHeads: Int, headDim: Int,
        nSelected: Int, kvStride: Int, scale: Float, outDtype: DType,
        on cmd: MTLCommandBuffer, into out: Tensor? = nil
    ) -> Tensor {
        precondition(headDim == 512, "dsv4CsaSdpaDecode: head_dim must be 512")
        precondition(selectedIndices.dtype == .u32, "dsv4CsaSdpaDecode: selectedIndices must be u32")
        precondition(nQHeads % nKvHeads == 0, "dsv4CsaSdpaDecode: GQA must be integer")
        let result = out ?? Tensor.empty(shape: [nQHeads, headDim], dtype: outDtype)
        let gridSize = MTLSize(width: nQHeads, height: 1, depth: 1)
        let tg = MTLSize(width: 512, height: 1, depth: 1)
        let hd = UInt32(headDim)
        let ns = UInt32(nSelected)
        let ks = UInt32(kvStride)
        let hpg = UInt32(nQHeads / nKvHeads)
        switch outDtype {
        case .f32:
            MetalTileKernels.ffai_dsv4_csa_sdpa_decode_f32(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                selected_indices: selectedIndices.buffer, selected_indicesOffset: selectedIndices.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_selected: ns, kv_stride: ks, heads_per_group: hpg, scale: scale,
                gridSize: gridSize, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_dsv4_csa_sdpa_decode_f16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                selected_indices: selectedIndices.buffer, selected_indicesOffset: selectedIndices.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_selected: ns, kv_stride: ks, heads_per_group: hpg, scale: scale,
                gridSize: gridSize, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_dsv4_csa_sdpa_decode_bf16(
                q: q.buffer, qOffset: q.offset,
                k: k.buffer, kOffset: k.offset,
                v: v.buffer, vOffset: v.offset,
                selected_indices: selectedIndices.buffer, selected_indicesOffset: selectedIndices.offset,
                out: result.buffer, outOffset: result.offset,
                head_dim: hd, n_selected: ns, kv_stride: ks, heads_per_group: hpg, scale: scale,
                gridSize: gridSize, threadgroupSize: tg, on: cmd)
        default:
            fatalError("dsv4CsaSdpaDecode: unsupported output dtype \(outDtype)")
        }
        return result
    }
}
