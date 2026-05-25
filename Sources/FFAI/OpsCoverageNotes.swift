// OpsCoverageNotes — documentation of metaltile kernels intentionally
// **not** exposed as standalone `Ops.*` wrappers.
//
// FFAI's `Ops` surface is the production-callable API. Not every
// `#[kernel]` in `metaltile-std` belongs there:
//
//   1. Some kernels are internal building blocks composed by a
//      higher-level wrapper (e.g. `aura_flash_p1_*` is used inside
//      `Ops.auraFlashAttention`-style composites, not directly).
//   2. Some are bit-width / tile-shape variants that the existing
//      wrapper switches over by parameter (e.g. `mt_qmv_b3` … `mt_qmv_b8`
//      are auto-routed inside `Ops.dequantGemv(bits:)`).
//   3. Some are `_record` / `state_replay_*` infrastructure for the
//      `dispatch_chain` indirect-command-buffer mode — those aren't
//      standalone-callable kernels, they're recorded into an ICB.
//   4. Some are bench-only / probe-only kernels with no production
//      caller (`mt_sgload_smoke`, `mma_layout_probe`, `mpp_matmul_smoke`).
//   5. Some require dtypes (fp4 / fp8 / mxfp) that FFAI's `DType` enum
//      doesn't yet expose; wrapping them would require a DType
//      expansion first (tracked separately).
//
// This file is a manifest of those choices so the next sweep over
// `Sources/MetalTileSwift/Generated/MetalTileKernels.swift` can quickly
// distinguish "intentionally skipped" from "missing wrapper, please add".
//
// Format: each `KernelKind` case captures the *family* (not every
// per-dtype / per-bit-width variant). The `static var notWrapped`
// dictionary records the rationale.

import Foundation

public enum OpsCoverageNotes {

    /// One entry per intentionally-unwrapped metaltile kernel family.
    /// Pure data — no Metal symbols referenced, so this builds even if
    /// a kernel is later removed from metaltile.
    public struct UncoveredKernel: Sendable, Equatable {
        public let familyName: String
        public let rationale: String
    }

    public static let intentionallyUnwrapped: [UncoveredKernel] = [
        // ── Internal building blocks for AURA flash attention ───────
        UncoveredKernel(familyName: "aura_flash_p1_*",
                        rationale: "Internal pass-1 partial of the AURA flash-attention composite. Exposed through Ops.auraFlash* family, not standalone."),
        UncoveredKernel(familyName: "aura_flash_pass2_*",
                        rationale: "Internal pass-2 partial of the AURA flash-attention composite. Exposed through Ops.auraFlash* family."),
        UncoveredKernel(familyName: "aura_flash_sdpa_*",
                        rationale: "Single-pass AURA flash SDPA variant used internally by the AURA fast path. Composed inside AURAQuantizedKVCache decode."),
        UncoveredKernel(familyName: "aura_score_int{2,3,4,6,8}",
                        rationale: "AURA per-token score kernel — internal building block of auraFlashAttention. Routed inside the composite based on bit-width."),
        UncoveredKernel(familyName: "aura_value_int{2,3,4,6,8}",
                        rationale: "AURA per-token value kernel — internal building block of auraFlashAttention."),

        // ── Bit-width / tile-shape variants of an already-wrapped op ─
        UncoveredKernel(familyName: "mt_qmm_b{3,4,5,6,8} / mt_qmv_b{3,4,5,6,8} / mt_qvm_b{3,4,5,6,8}",
                        rationale: "MLX-style quantized matmul variants. FFAI's production path uses ffai_* / dequant_gemv_* kernels (different geometry); these MLX-style siblings are kept for `tile bench` side-by-side comparisons only."),
        UncoveredKernel(familyName: "mt_qmm_bm2_int8_fast / mt_qmm_bm4_int8_fast / mt_qmm_int8_fast / mt_qmv_int8_fast / mt_qvm_int4_fast / mt_qmm_mma_int8 / mt_qmm_mma_m16_int8 / mt_qmm_mma_mpp_int8 / mt_qmm_nax_int8 / mt_qmm_nax",
                        rationale: "MLX-style int8/int4 GEMM variants. Not on the FFAI hot path (which uses ffai_gemm + dequant_gemv_int*)."),
        UncoveredKernel(familyName: "mt_moe_gather_qmm_b{3,5,6,8} / *_mma_b{3,5,6,8} / *_int8 / *_int8_bm{8,16,64}_mpp / *_int4_m{8,16,32}",
                        rationale: "MoE gather quantized matmul bit-width / tile-shape variants. Routed inside Ops.moeGatherDequantGemm* by bits + BM."),
        UncoveredKernel(familyName: "mt_steel_gemm_{32x32,64x64,…}_{1x2,2x2,4x2,…}",
                        rationale: "Tile-shape variants of the base steel GEMM. Internal to Ops.gemm dispatch; user-facing wrapper picks the right tile."),
        UncoveredKernel(familyName: "mt_steel_gemm_{gather,masked,segmented,splitk,splitk_accum,fused,nax}*",
                        rationale: "Specialized steel GEMM variants (gather / masked / split-K / NAX). Not yet on the production hot path; reserved for future Ops.gemm specializations."),
        UncoveredKernel(familyName: "sdpa_decode_2pass_pass{1,2}_d{64,96,128,256}",
                        rationale: "Two-pass SDPA decode used for very long contexts. FFAI currently routes through the single-pass variants (Ops.sdpaDecode); promotion path tracked in the perf backlog."),
        UncoveredKernel(familyName: "sdpa_decode_batched_q{2,4,8}",
                        rationale: "Quantized-KV batched-prefill SDPA. Composed inside AURAQuantizedKVCache; not exposed standalone."),

        // ── _record / state_replay infrastructure ──────────────────
        UncoveredKernel(familyName: "*_record",
                        rationale: "dispatch_chain indirect-command-buffer recording variants. Not standalone-callable; encoded into MTLIndirectComputeCommand by metaltile's chain replay system."),
        UncoveredKernel(familyName: "state_replay_d* / ssm_step_record_d* / gated_delta_step_record_d*",
                        rationale: "Dispatch-chain replay specializations of recurrence kernels (one PSO per state shape). Used by the indirect-cmdbuf prefill replay; not exposed standalone."),
        UncoveredKernel(familyName: "ssm_replay_d{16_64_4, 128_128_32}",
                        rationale: "SSM replay specializations for the Mamba2 prefill chain. Composed inside Mamba2LayerCache replay."),

        // ── Probe / bench / test kernels ───────────────────────────
        UncoveredKernel(familyName: "mt_sgload_smoke",
                        rationale: "Subgroup-load smoke test kernel — used only by metaltile's own GPU probe."),

        // ── Bit-stream affine quant variants (bits 3 / 5 / 6) ───────
        UncoveredKernel(familyName: "mt_affine_{quantize,dequantize}_int{3,5,6}",
                        rationale: "Bit-stream packed affine quant for odd bit-widths. Not yet exposed at the QuantizedOps layer — caller-side production callers either use Ops.dequantGemv (which has int{3,5,6} arms via the dequant_gemv_int* kernels) or QuantizedOps.{quantize,dequantize}Affine for bits ∈ {2,4,8}. Promotion path: add the bit-stream packing arm to QuantizedOps if a per-buffer int{3,5,6} need arises."),

        // ── Vision: MMA / Winograd / grouped / patch variants ───────
        UncoveredKernel(familyName: "conv2d_mma / conv2d_grouped / conv2d_patch{14,16}",
                        rationale: "Specialized conv2d variants for vision towers (MMA / depthwise-grouped / fixed-patch-stride). Future Ops.conv2d specializations once any production VLM benefits."),
        UncoveredKernel(familyName: "conv3d_{generic,grouped,mma}",
                        rationale: "3D convolution variants. No production caller yet; video VLMs that need them will be added with their integration."),
        UncoveredKernel(familyName: "patch_embed_mma",
                        rationale: "MMA-tile patch-embed variant of patch_embed. Routed automatically inside Ops.patchEmbed when the input geometry matches the MMA tile."),
        UncoveredKernel(familyName: "winograd_conv2d_3x3 / winograd_conv2d_3x3_split / winograd_filter_transform_3x3",
                        rationale: "Winograd-accelerated 3x3 convolution. No production caller routes through this yet; the standard conv2d path is sufficient at the production model shapes we ship."),

        // ── FP4 / FP8 / MXFP ───────────────────────────────────────
        UncoveredKernel(familyName: "mt_fp4_{quant_dequant,qmm_mma} / mt_fp8_{e4m3,e5m2}_{quant_dequant,qmm_mma} / mt_fp_qmm_nax",
                        rationale: "FP4 / FP8 / MXFP quantization. FFAI's DType enum does not yet expose fp4/fp8 cases — wrapping these would require a DType expansion first. Tracked as a follow-up; the existing GPT-OSS MXFP4 transcode goes through a model-specific Loader path."),
        UncoveredKernel(familyName: "{quantize,bulk_dequant}_kv_fp8_e{4m3,5m2}",
                        rationale: "FP8 KV-cache quant siblings of int4/int8 (which Ops.quantizeKVInt{4,8} wraps). Blocked on FFAI DType.fp8 (see above)."),

        // ── FFT / Hadamard / scan / sort / strided ─────────────────
        UncoveredKernel(familyName: "mt_fft_n{32,64,…,1024} / mt_fft_bluestein_*",
                        rationale: "FFT building blocks composed inside mel_spectrogram / vocoder_istft. Not exposed standalone — the audio pipeline is the only caller."),
        UncoveredKernel(familyName: "mt_hadamard_n{64,128,…,1024}",
                        rationale: "Walsh-Hadamard transform — composed inside AURA encode (the SRHT rotation step). Direct standalone use is uncommon enough we route through auraEncode."),
        UncoveredKernel(familyName: "mt_scan / mt_scan_exclusive / mt_scan_{max,min,prod}*",
                        rationale: "Inclusive / exclusive prefix-scan family. No production caller at the Ops layer yet."),
        UncoveredKernel(familyName: "mt_sort / mt_sort_segmented",
                        rationale: "Sort kernels. No on-GPU sort production caller; logits sampling uses GPU mask + CPU-side argmax."),
        UncoveredKernel(familyName: "mt_strided_copy / mt_strided_copy_nd",
                        rationale: "Strided / N-D copy. Production data movement uses MTLBlit (Ops.copy) or contiguous mt_copy (Ops.copyKernel). Strided variant is reserved for the future when slicing semantics land."),
        UncoveredKernel(familyName: "mt_masked_scatter / mt_scatter / mt_scatter_axis",
                        rationale: "Scatter / masked-scatter primitives. No production caller; the embedding write path uses kv_cache_update for KV and direct buffer-row writes for embeddings."),
        UncoveredKernel(familyName: "mt_gather_axis / mt_gather_front",
                        rationale: "Gather primitives. Embedding gather uses Ops.gather / Ops.dequantGather; axis-gather has no production caller."),
        UncoveredKernel(familyName: "mt_merge / mt_select / mt_binary_two",
                        rationale: "Element-wise primitives — merge (two-input ternary), select, binary_two. No production caller at the Ops layer."),
        UncoveredKernel(familyName: "mt_seg_reduce / mt_seg_reduce_{max,min,prod} / mt_col_reduce* / mt_all_reduce*",
                        rationale: "Segmented + column + cross-row reduction primitives. mt_row_reduce variants are also unwrapped — Ops exposes the higher-level argmax / softmax / logsumexp instead, which cover the production decoder needs."),
        UncoveredKernel(familyName: "mt_row_reduce / mt_row_reduce_{max,min,prod}",
                        rationale: "Per-row sum/max/min/prod. Sampling pipeline uses argmax / argmin / softmax instead; raw row-reductions have no production caller yet."),
        UncoveredKernel(familyName: "mt_rms_inv_scalar",
                        rationale: "Bench-only helper used by RMSNorm correctness tests on the metaltile side."),

        // ── Logits processors (the chain-replay variants) ──────────
        // The non-record logits_* are wrapped in Ops.swift via OpsLogits.swift.

        // ── Scalar math primitives ─────────────────────────────────
        UncoveredKernel(familyName: "mt_acos / mt_acosh / mt_asin / mt_asinh / mt_atan / mt_atan2 / mt_atanh / mt_cos / mt_cosh / mt_sin / mt_sinh / mt_tan / mt_tanh_op / mt_erf / mt_erfinv / mt_exp2 / mt_expm1 / mt_log10 / mt_log1p / mt_log2 / mt_logaddexp / mt_remainder / mt_sign / mt_trunc / mt_random_hash",
                        rationale: "Trig / hyperbolic / specialised log / random scalar primitives. No production caller; covered by torch's CPU-side preprocessing for the rare callers that need them. Wrappers can be added incrementally if a model family adopts one."),
        UncoveredKernel(familyName: "mt_fft_bluestein_chirp_filter",
                        rationale: "Bluestein-FFT chirp-filter helper. Composed inside mel_spectrogram; not a standalone caller."),
        UncoveredKernel(familyName: "mt_scalar_fma_chain8",
                        rationale: "Chain-of-8 scalar FMA used by AURA encode rotation. Internal building block."),

        // ── Other ──────────────────────────────────────────────────
        UncoveredKernel(familyName: "mt_gemv_masked",
                        rationale: "Masked GEMV for sparse attention. No production caller yet — sparse attention variants use sdpa_multi/decode instead."),
        UncoveredKernel(familyName: "flash_quantized_sdpa_{b4,b8}_d{64,96,128,256,512} / flash_quantized_sdpa_{bool,float}_mask_*",
                        rationale: "Quantized-KV flash SDPA siblings of aura_flash_sdpa. Composed inside AURAQuantizedKVCache decode; not exposed standalone."),
        UncoveredKernel(familyName: "ssm_step_a2d",
                        rationale: "A-2D variant of ssm_step for models like Jamba whose A_log is 2D rather than 1D. Routed inside Mamba2LayerCache based on the A_log shape — not exposed as a separate wrapper."),

        // ── Future per-headDim sdpa specializations ────────────────
        UncoveredKernel(familyName: "mt_sdpa / mt_sdpa_prefill / mt_sdpa_vector / mt_sdpa_prefill_mma_bf16",
                        rationale: "MLX-side base SDPA kernels — FFAI routes through ffai_sdpa_decode / ffai_sdpa_multi / ffai_sdpa_bidirectional which carry the dispatch-shape contracts and DISPATCH-INVARIANTS docs that prevent the 2026-05-19 GPU-pin class of bug. The MLX siblings are kept for bench comparisons."),
    ]

    /// Count of intentionally-unwrapped kernel families. Used by the
    /// coverage manifest test to assert the explanation list isn't
    /// silently truncated.
    public static var count: Int { intentionallyUnwrapped.count }
}
