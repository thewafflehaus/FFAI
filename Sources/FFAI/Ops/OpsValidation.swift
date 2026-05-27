// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// OpsValidation — pure, testable wrapper precondition logic.
//
// Background: every `Ops.*` wrapper around a reduction-mode metaltile
// kernel has dispatch-shape invariants that, if violated, range from
// silent miscompute to a non-preemptive GPU pin (system freeze).
// Before this file existed, the invariants lived inline in each
// wrapper as `precondition` calls. That was load-bearing but
// untestable in CI — a `precondition` failure traps the entire test
// process, so the only "test" of a precondition was producing it in
// production.
//
// Each validation function below returns:
//   * `nil` — dispatch shape is valid, wrapper proceeds.
//   * `String` — human-readable reason. Wrapper calls
//     `preconditionFailure("Ops.<fn>: \(reason)")` to halt.
//
// Tests in `Tests/FFAITests/OpsValidationTests.swift` exercise good +
// bad inputs without producing a trap. New reduction-mode wrappers
// should add a `validate*` function here in the same commit as the
// wrapper.
//
// See `papers/post-mortem-2026-05-19-dispatch-shape-gpu-freeze.md`
// for the full story behind why this file exists.

import Foundation

public enum OpsValidation {

    // ─── rmsNorm + rmsNormRows ─────────────────────────────────────
    //
    // `Ops.dispatchRmsNorm` routes by row width across two metaltile
    // kernels (`crates/metaltile-std/src/mlx/rms_norm.rs`):
    //   • `mt_rms_norm` (fast path). `N = TPG * 4` (4 elements per
    //     thread), `TPG = n / 4`, `TPG` a multiple of 32 ⇒ `n` a
    //     multiple of 128, `TPG ≤ 1024` ⇒ `n ≤ 4096`.
    //   • `mt_rms_norm_wide` (any width). Each thread strides over the
    //     row with `TPG = 1024`; no 128-alignment or upper bound. Used
    //     when `n > 4096` (Gemma 4 27B+ hidden 5376) AND when `n` is
    //     not a multiple of 128 (SmolVLM2 vision tower d=960 etc).
    //
    // Caller-facing invariant: just `n > 0`. The wrapper picks the
    // right kernel.

    /// Validate the row-width parameter for `Ops.rmsNorm` and
    /// `Ops.rmsNormRows`. The single-row and multi-row dispatches
    /// share the per-row invariant. Any positive `n` is accepted —
    /// `dispatchRmsNorm` routes 128-aligned rows ≤ 4096 to the fast
    /// kernel and everything else to the always-correct wide kernel.
    public static func validateRmsNorm(n: Int) -> String? {
        if n <= 0 {
            return "n must be positive (got \(n))"
        }
        return nil
    }

    // ─── sdpaDecode ────────────────────────────────────────────────
    //
    // Kernel invariants (from `crates/metaltile-std/src/ffai/sdpa_decode.rs`
    // + `sdpa_decode_d64.rs` §"DISPATCH INVARIANTS"):
    //   1. `head_dim ∈ {64, 128, 256, 512}`. Each lane owns
    //      `head_dim / 32` consecutive Q/K/V elements; loads are
    //      unconditional. Wrong head_dim → wrong TPG → infinite loop in
    //      `for _t in range(sg, n_kv, ns=TPG/32)` when `TPG < 32`
    //      makes `ns = 0`. (See FFAI post-mortem 2026-05-19.)
    //      head_dim=128 covers Llama 3.2 3B+ / Qwen3 / GPT-OSS full
    //      layers, head_dim=64 covers Llama 3.2 1B and GPT-OSS
    //      sliding-window layers, head_dim=256 covers Gemma 3 / Gemma 4
    //      sliding layers, head_dim=512 covers Gemma 4 global layers.
    //   2. `nQHeads % nKVHeads == 0` so GQA fan-out is integer.
    //   3. `n_kv ≤ kv_stride`. The kernel walks `[0, n_kv)` only;
    //      `kv_stride` is the pre-allocated maxSeq capacity.
    //   4. Sliding-window / sink fast path (head_dim=128 variants only,
    //      added in metaltile PR #50): the kernel attends the sink
    //      range `[0, sink_end)` plus the window range
    //      `[window_start, n_kv)` and skips `[sink_end, window_start)`
    //      at the loop-bound level. Caller contract:
    //        * `0 ≤ sink_end ≤ window_start` — otherwise the sink and
    //          window passes overlap and the online softmax
    //          double-counts the intersection.
    //        * `window_start ≤ n_kv` — the window pass walks
    //          `[window_start, n_kv)`; a larger start would make the
    //          range empty (harmless) but signals a caller bug.
    //      Both default to 0, which is exactly dense full attention.

    /// head_dim values for which a kernel specialization currently
    /// exists. Caller is responsible for routing to the matching
    /// kernel; see `Ops.sdpaDecode`.
    public static let supportedSdpaHeadDims: Set<Int> = [64, 128, 256, 512]

    /// head_dim values whose kernel variant carries the `sink_end` /
    /// `window_start` constexprs. The d64 / d256 variants are dense-only
    /// — passing a non-zero sink/window to `Ops.sdpaDecode` with those
    /// head dims is rejected.
    public static let slidingWindowSdpaHeadDims: Set<Int> = [128]

    public static func validateSdpaDecode(
        headDim: Int, nQHeads: Int, nKVHeads: Int,
        nKV: Int, kvStride: Int,
        sinkEnd: Int = 0, windowStart: Int = 0
    ) -> String? {
        if !supportedSdpaHeadDims.contains(headDim) {
            let supported = supportedSdpaHeadDims.sorted()
                .map(String.init).joined(separator: ", ")
            return
                "head_dim must be one of {\(supported)} (got \(headDim)); other specializations not yet emitted"
        }
        if nQHeads <= 0 {
            return "nQHeads must be positive (got \(nQHeads))"
        }
        if nKVHeads <= 0 {
            return "nKVHeads must be positive (got \(nKVHeads))"
        }
        if !nQHeads.isMultiple(of: nKVHeads) {
            return "nQHeads (\(nQHeads)) must be a multiple of nKVHeads (\(nKVHeads))"
        }
        if nKV < 0 {
            return "nKV must be non-negative (got \(nKV))"
        }
        if nKV > kvStride {
            return
                "nKV (\(nKV)) must not exceed kvStride (\(kvStride)) — kernel would read past cache"
        }
        // Sliding-window / sink fast-path contract (invariant 4).
        if sinkEnd != 0 || windowStart != 0 {
            if !slidingWindowSdpaHeadDims.contains(headDim) {
                let supported = slidingWindowSdpaHeadDims.sorted()
                    .map(String.init).joined(separator: ", ")
                return
                    "sinkEnd/windowStart are only supported for head_dim ∈ {\(supported)} (got \(headDim)); the d64/d256 kernels are dense-only"
            }
            if sinkEnd < 0 {
                return "sinkEnd must be non-negative (got \(sinkEnd))"
            }
            if windowStart < 0 {
                return "windowStart must be non-negative (got \(windowStart))"
            }
            if sinkEnd > windowStart {
                return
                    "sinkEnd (\(sinkEnd)) must not exceed windowStart (\(windowStart)) — overlapping sink + window ranges double-count in the online softmax"
            }
            if windowStart > nKV {
                return
                    "windowStart (\(windowStart)) must not exceed nKV (\(nKV)) — window pass walks [windowStart, nKV)"
            }
        }
        return nil
    }

    // ─── sdpa_multi ────────────────────────────────────────────────
    //
    // `ffai_sdpa_multi` is a reduction kernel — same machine-freeze
    // hazard as `sdpa_decode` (TPG below 32 → infinite GPU loop). The
    // wrapper hard-fixes TPG = 1024 so the geometry can't reach the
    // freeze condition; what this validates is the shape contract the
    // kernel indexes against. head_dim is 128-only: each lane owns
    // 128/32 = 4 consecutive elements, indexed unconditionally.

    public static func validateSdpaMulti(
        headDim: Int, nQHeads: Int, nKVHeads: Int,
        baseKV: Int, nQuery: Int, kvStride: Int
    ) -> String? {
        if headDim != 128 && headDim != 256 {
            return "head_dim must be 128 or 256 (got \(headDim)); "
                + "ffai_sdpa_multi has dedicated d128 / d256 kernel variants only"
        }
        if nQHeads <= 0 {
            return "nQHeads must be positive (got \(nQHeads))"
        }
        if nKVHeads <= 0 {
            return "nKVHeads must be positive (got \(nKVHeads))"
        }
        if !nQHeads.isMultiple(of: nKVHeads) {
            return "nQHeads (\(nQHeads)) must be a multiple of nKVHeads (\(nKVHeads))"
        }
        if nQuery < 1 {
            return "nQuery must be ≥ 1 (got \(nQuery))"
        }
        if baseKV < 0 {
            return "baseKV must be non-negative (got \(baseKV))"
        }
        if baseKV + nQuery > kvStride {
            return "baseKV+nQuery (\(baseKV + nQuery)) must not exceed kvStride "
                + "(\(kvStride)) — kernel would read past cache"
        }
        return nil
    }

    // ─── add_rms_norm (fused residual + RMSNorm) ───────────────────
    //
    // `mt_add_rms_norm` fuses the transformer block's residual stream
    // and post-residual RMSNorm into one kernel. Each row uses TPG =
    // n / 4 (4 elements per thread, same pattern as `mt_rms_norm`),
    // so n must be a multiple of 4 AND n / 4 must stay within Apple
    // Silicon's 1024-thread-per-group cap → **n ≤ 4096**. Wider
    // hidden sizes (Gemma 4 27B+ at hidden 5376) MUST fall back to
    // separate `add` + `rmsNorm` calls.

    /// Largest row width the fused `mt_add_rms_norm` kernel can take.
    /// Above this, callers should use separate `Ops.add` + `Ops.rmsNorm`.
    public static let addRmsNormMaxRowSize = 4096

    public static func validateAddRmsNorm(n: Int) -> String? {
        if n <= 0 {
            return "n must be positive (got \(n))"
        }
        if !n.isMultiple(of: 4) {
            return "n must be a multiple of 4 (got \(n)); kernel vectorises "
                + "the row in 4-element chunks"
        }
        if n > addRmsNormMaxRowSize {
            return "n must be ≤ \(addRmsNormMaxRowSize) (got \(n)); larger "
                + "hidden sizes exceed the 1024-thread group cap (TPG = n / 4)"
        }
        if n / 4 < 32 {
            return "n / 4 = \(n / 4) must be ≥ 32 (TPG below a simdgroup width "
                + "would make the reduction degenerate)"
        }
        return nil
    }

    // ─── sdpa_bidirectional_d{32,64,72} ────────────────────────────
    //
    // `ffai_sdpa_bidirectional_dN` is the always-bidirectional sibling
    // of `sdpa_multi`, exposed at N ∈ {32, 64, 72} for vision-tower
    // head dimensions (FastViT-HD, SigLIP / CLIP, PaliGemma SigLIP-So400m
    // respectively). Same reduction-mode hazard as `sdpa_multi`: TPG
    // below 32 → infinite GPU loop. Wrapper hard-fixes TPG = 1024.
    // Caller MUST supply one of the three supported head_dims.

    /// Whether `Ops.sdpaBidirectional` has a kernel for this head_dim.
    /// d32: FastViT-HD. d64: SigLIP-base / CLIP-L / Mistral3 vision /
    /// Gemma4-E2B/E4B / Qwen3-VL 2B/4B. d72: SigLIP-So400m
    /// (Paligemma, Gemma3VL, Gemma4-26B/31B, Idefics3, Qwen3-VL 30B-A3B).
    /// d80: Qwen2.5-VL. d96: Qwen2-VL. (d128 lives in sdpa_multi.)
    public static let sdpaBidirectionalSupportedHeadDims: Set<Int> = [32, 64, 72, 80, 96]

    public static func validateSdpaBidirectional(
        headDim: Int, nQHeads: Int, nKVHeads: Int,
        baseKV: Int, nQuery: Int, kvStride: Int
    ) -> String? {
        if !sdpaBidirectionalSupportedHeadDims.contains(headDim) {
            return "head_dim must be one of {32, 64, 72, 80, 96} (got "
                + "\(headDim)); for d128 use Ops.sdpaMulti(causal: false)"
        }
        if nQHeads <= 0 {
            return "nQHeads must be positive (got \(nQHeads))"
        }
        if nKVHeads <= 0 {
            return "nKVHeads must be positive (got \(nKVHeads))"
        }
        if !nQHeads.isMultiple(of: nKVHeads) {
            return "nQHeads (\(nQHeads)) must be a multiple of nKVHeads (\(nKVHeads))"
        }
        if nQuery < 1 {
            return "nQuery must be ≥ 1 (got \(nQuery))"
        }
        if baseKV < 0 {
            return "baseKV must be non-negative (got \(baseKV))"
        }
        if baseKV + nQuery > kvStride {
            return "baseKV+nQuery (\(baseKV + nQuery)) must not exceed kvStride "
                + "(\(kvStride)) — kernel would read past cache"
        }
        return nil
    }

    // ─── gemv ──────────────────────────────────────────────────────
    //
    // `mt_gemv` (MLX-derived). Adaptive `lsize` reduction → no GPU-pin
    // risk regardless of TPG. Caller-controllable shape (outDim, inDim);
    // the wrapper already checks shape consistency via the `weight`
    // tensor. Validation here just pins the basic positivity contract.

    public static func validateGemv(outDim: Int, inDim: Int) -> String? {
        if outDim <= 0 {
            return "outDim must be positive (got \(outDim))"
        }
        if inDim <= 0 {
            return "inDim must be positive (got \(inDim))"
        }
        return nil
    }

    // ─── gemm (multi-row) ──────────────────────────────────────────
    //
    // `ffai_gemm` — tiled `out[r,:] = weight · input[r,:]` over a block
    // of rows. Reduction-mode (threadgroup tiles + barriers), TPG hard-
    // fixed at 1024 by the wrapper. The one shape contract the kernel
    // can't check: `inDim % 16 == 0` — the K loop strides by the
    // 16-wide tile with no remainder handling, so an unaligned inDim
    // silently drops the trailing partial tile.

    public static func validateGemm(inDim: Int, outDim: Int, nRows: Int) -> String? {
        if inDim <= 0 {
            return "inDim must be positive (got \(inDim))"
        }
        if outDim <= 0 {
            return "outDim must be positive (got \(outDim))"
        }
        if nRows <= 0 {
            return "nRows must be positive (got \(nRows))"
        }
        if !inDim.isMultiple(of: 16) {
            return "inDim (\(inDim)) must be a multiple of 16 — the K tile width"
        }
        return nil
    }

    // ─── dequantGemv ───────────────────────────────────────────────
    //
    // Affine-dequant + matvec for int{3,4,5,6,8} weights. Three silent-
    // miscompute footguns the wrapper didn't previously catch:
    //
    //   1. `inDim % groupSize != 0` → kernel's `n_groups = inDim/groupSize`
    //      rounds down, silently dropping the partial trailing group.
    //   2. For pack-strided bit-widths (int4, int8), `inDim % vals_per_pack`
    //      must be 0 — kernel's `n_packs_per_row = inDim/vals_per_pack`
    //      rounds down, silently dropping unaligned tail elements.
    //   3. `scales`/`biases` element counts must be `outDim × n_groups`.
    //      Smaller → OOB reads → garbage output. Larger → no harm.
    //
    // Element-strided bit-widths {3, 5, 6} don't have the pack-alignment
    // constraint — the kernel walks individual elements.

    public static func validateDequantGemv(
        outDim: Int, inDim: Int, bits: Int, groupSize: Int,
        scalesCount: Int, biasesCount: Int
    ) -> String? {
        // Supported bit-widths (mirrors the wrapper's switch arms).
        if !(bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8) {
            return "bits=\(bits) unsupported — must be one of 2, 3, 4, 5, 6, or 8"
        }
        if outDim <= 0 {
            return "outDim must be positive (got \(outDim))"
        }
        if inDim <= 0 {
            return "inDim must be positive (got \(inDim))"
        }
        if groupSize <= 0 {
            return "groupSize must be positive (got \(groupSize))"
        }
        // Footgun 1: partial trailing group.
        if !inDim.isMultiple(of: groupSize) {
            return
                "inDim=\(inDim) must be a multiple of groupSize=\(groupSize) — partial trailing group would be silently dropped"
        }
        // Footgun 2: pack-alignment for pack-strided variants.
        if bits == 2 || bits == 4 || bits == 8 {
            let valsPerPack = 32 / bits  // 16 for int2, 8 for int4, 4 for int8
            if !inDim.isMultiple(of: valsPerPack) {
                return
                    "inDim=\(inDim) must be a multiple of \(valsPerPack) for bits=\(bits) (pack-strided kernel — unaligned tail elements silently dropped)"
            }
            if !groupSize.isMultiple(of: valsPerPack) {
                return
                    "groupSize=\(groupSize) must be a multiple of \(valsPerPack) for bits=\(bits) (packs_per_group must be exact)"
            }
        }
        // Footgun 3: scales/biases sizing.
        let nGroups = inDim / groupSize
        let expected = outDim * nGroups
        if scalesCount != expected {
            return
                "scales must have outDim × n_groups = \(outDim) × \(nGroups) = \(expected) elements, got \(scalesCount)"
        }
        if biasesCount != expected {
            return
                "biases must have outDim × n_groups = \(outDim) × \(nGroups) = \(expected) elements, got \(biasesCount)"
        }
        return nil
    }

    // ─── auraEncode ────────────────────────────────────────────────
    //
    // Kernel invariants (from `crates/metaltile-std/src/ffai/aura_encode.rs`
    // §"DISPATCH INVARIANTS"):
    //   1. `TPG = dim` — one thread per rotated coordinate.
    //   2. `dim` must be a multiple of 32 (`simd_sum` reduction).
    //   3. `dim ≤ 1024` (`threadgroup_alloc("shared_unit", 1024)`).
    //   4. `bits ∈ {2, 3, 4, 8}` — encode kernel only emits these.

    public static func validateAuraEncode(
        rows: Int, dim: Int, bits: Int
    ) -> String? {
        if rows <= 0 {
            return "rows must be positive (got \(rows))"
        }
        if dim <= 0 {
            return "dim must be positive (got \(dim))"
        }
        if !dim.isMultiple(of: 32) {
            return
                "dim=\(dim) must be a multiple of 32 (one Apple simdgroup); simd_sum reduction is undefined otherwise"
        }
        if dim > 1024 {
            return
                "dim=\(dim) > 1024 — exceeds the kernel's TPG cap (shared_unit allocates 1024 slots)"
        }
        if bits != 2 && bits != 3 && bits != 4 && bits != 8 {
            return
                "bits=\(bits) unsupported — encode kernel emits only int2/int3/int4/int8 variants"
        }
        return nil
    }

    // ─── quantizeKV / bulkDequantKV (int4 / int8) ──────────────────
    //
    // Affine KV-cache quant/dequant (`crates/metaltile-std/src/ffai/kv_cache.rs`).
    // Grid3D mode — one thread per group (quantize) or per output
    // element (bulk dequant), so no GPU-pin risk. The footguns are
    // silent miscompute from integer-truncating divisions baked into
    // the kernel's offset arithmetic:
    //
    //   1. `groups_per_head = head_dim / group_size`. If `head_dim`
    //      isn't a multiple of `group_size` the trailing partial group
    //      is silently dropped — its head_dim slots never get written.
    //   2. `head_dim / vals_per_pack` is the per-row packed stride
    //      (`vals_per_pack = 32/bits` = 8 for int4, 4 for int8). A
    //      non-multiple `head_dim` truncates the stride → packed words
    //      for the tail are written/read at the wrong offset.
    //   3. `group_size / vals_per_pack` is the per-group pack count.
    //      A non-multiple `group_size` drops the group's tail packs.
    //
    // Only int4/int8 are emitted (`bits` switch in the wrapper).

    public static func validateQuantizeKV(
        nKVHeads: Int, headDim: Int, groupSize: Int, bits: Int
    ) -> String? {
        if bits != 4 && bits != 8 {
            return "bits=\(bits) unsupported — KV quant emits only int4/int8 variants"
        }
        if nKVHeads <= 0 {
            return "nKVHeads must be positive (got \(nKVHeads))"
        }
        if headDim <= 0 {
            return "headDim must be positive (got \(headDim))"
        }
        if groupSize <= 0 {
            return "groupSize must be positive (got \(groupSize))"
        }
        // Footgun 1: partial trailing group.
        if !headDim.isMultiple(of: groupSize) {
            return
                "headDim=\(headDim) must be a multiple of groupSize=\(groupSize) — partial trailing group would be silently dropped"
        }
        // Footguns 2 + 3: pack alignment. vals_per_pack = 32/bits.
        let valsPerPack = 32 / bits  // 8 for int4, 4 for int8
        if !headDim.isMultiple(of: valsPerPack) {
            return
                "headDim=\(headDim) must be a multiple of \(valsPerPack) for bits=\(bits) (pack-strided kernel — unaligned tail packed at the wrong offset)"
        }
        if !groupSize.isMultiple(of: valsPerPack) {
            return
                "groupSize=\(groupSize) must be a multiple of \(valsPerPack) for bits=\(bits) (packs_per_group must be exact)"
        }
        return nil
    }

    // ─── dequantGather ─────────────────────────────────────────────
    //
    // Affine-dequantizing embedding gather (`ffai/dequant_gather.rs`).
    // Grid3D mode — one thread per output element, no GPU-pin risk.
    // The silent-miscompute footgun:
    //
    //   * `groups_per_row = hidden / group_size`. A non-multiple
    //     `hidden` truncates the group count, so the trailing partial
    //     group's scale/bias is read from the wrong index.
    //
    // Unlike `dequantGemv` there is no pack-alignment constraint — the
    // kernel walks the bit-stream per individual element (`bit_off =
    // d * bits`), so any `hidden` is bit-addressable. Supported bit
    // widths mirror the wrapper's switch: {3, 4, 5, 6, 8}.

    public static func validateDequantGather(
        hidden: Int, bits: Int, groupSize: Int
    ) -> String? {
        if !(bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8) {
            return "bits=\(bits) unsupported — must be one of 2, 3, 4, 5, 6, or 8"
        }
        if hidden <= 0 {
            return "hidden must be positive (got \(hidden))"
        }
        if groupSize <= 0 {
            return "groupSize must be positive (got \(groupSize))"
        }
        if !hidden.isMultiple(of: groupSize) {
            return
                "hidden=\(hidden) must be a multiple of groupSize=\(groupSize) — partial trailing group would read scale/bias at the wrong index"
        }
        return nil
    }

    // ─── auraDequantRotated ────────────────────────────────────────
    //
    // AURA bulk dequant (`ffai/aura_dequant_rotated.rs`). Grid3D mode
    // — one thread per packed word, no GPU-pin risk. Two contracts:
    //
    //   1. `cacheStride >= tokens`. The kernel's `tokens` constexpr
    //      doubles as the per-head row stride of the packed/norms/out
    //      buffers. For a `[nKVHeads, maxSeq, …]` buffer the wrapper
    //      must pass `cacheStride = maxSeq`; passing the fill count
    //      makes heads 1…n address the wrong rows — the AURA "coherent
    //      then collapse" bug.
    //   2. `packedWidth` must cover every dim: `packedWidth >=
    //      ceil(dim / dims_per_word)` where `dims_per_word = 32/bits`
    //      (clean path) or `ceil(32/bits)` (odd-width spill path).
    //      Too small → trailing dims never written (zeros).
    //
    // Bit widths {2, 3, 4, 8} are emitted by the encode kernel; the
    // dequant kernel additionally has {3, 5, 6} odd-width arms, but the
    // wrapper's switch only routes {2, 3, 4, 8}.

    public static func validateAuraDequantRotated(
        dim: Int, packedWidth: Int, tokens: Int, bits: Int,
        cacheStride: Int
    ) -> String? {
        if bits != 2 && bits != 3 && bits != 4 && bits != 8 {
            return "bits=\(bits) unsupported — auraDequantRotated routes only int2/int3/int4/int8"
        }
        if dim <= 0 {
            return "dim must be positive (got \(dim))"
        }
        if tokens <= 0 {
            return "tokens must be positive (got \(tokens))"
        }
        if packedWidth <= 0 {
            return "packedWidth must be positive (got \(packedWidth))"
        }
        // Contract 1: per-head row stride.
        if cacheStride < tokens {
            return
                "cacheStride (\(cacheStride)) must be >= tokens (\(tokens)) — smaller stride makes heads 1…n read/write the wrong rows (AURA coherent-then-collapse bug)"
        }
        // Contract 2: packedWidth must cover every dim. The clean path
        // packs `32/bits` dims per word; the odd-width path packs
        // `ceil(32/bits)`. Both need `packedWidth * dims_per_word >= dim`;
        // the clean (smaller) `dims_per_word` is the binding lower bound.
        let dimsPerWord = 32 / bits  // floor; clean path. ≥ odd-path stride.
        let minPackedWidth = (dim + dimsPerWord - 1) / dimsPerWord
        if packedWidth < minPackedWidth {
            return
                "packedWidth=\(packedWidth) too small for dim=\(dim) at bits=\(bits) — need >= ceil(dim / \(dimsPerWord)) = \(minPackedWidth); trailing dims would be left as zeros"
        }
        return nil
    }

    // ─── auraRotatePerHead ─────────────────────────────────────────
    //
    // Per-head SRHT rotation (`Ops.auraRotatePerHead`). Not a kernel
    // wrapper itself — fans out one `Ops.gemv` per head — so there is
    // no dispatch-shape hazard. The preconditions are pure shape /
    // dtype contracts the gemv fan-out relies on:
    //
    //   1. `x` is a flat `[nHeads * headDim]` tensor.
    //   2. `rotation` is a square `[headDim, headDim]` matrix.
    //   3. `rotation.dtype == x.dtype` — gemv requires matched dtypes.

    public static func validateAuraRotatePerHead(
        xElementCount: Int, rotationShape: [Int],
        rotationDtypeMatchesX: Bool,
        nHeads: Int, headDim: Int
    ) -> String? {
        if nHeads <= 0 {
            return "nHeads must be positive (got \(nHeads))"
        }
        if headDim <= 0 {
            return "headDim must be positive (got \(headDim))"
        }
        if xElementCount != nHeads * headDim {
            return "x has \(xElementCount) elements, expected nHeads*headDim=\(nHeads * headDim)"
        }
        if rotationShape != [headDim, headDim] {
            return "rotation shape \(rotationShape) must be [\(headDim), \(headDim)]"
        }
        if !rotationDtypeMatchesX {
            return "rotation dtype must match x dtype (gemv requires matched dtypes)"
        }
        return nil
    }

    // ─── gatedDeltaStep ────────────────────────────────────────────
    //
    // Kernel invariants (from
    // `crates/metaltile-std/src/ffai/gated_delta.rs`
    // §"DISPATCH INVARIANTS"):
    //   1. TPG = 32 (exactly one simdgroup). The wrapper hard-codes
    //      this; the only caller-facing risk is the grid dimension.
    //   2. `Dk % 32 == 0` — each lane owns `Dk / 32` state columns and
    //      the per-lane state register array is capped at 8 (Dk ≤ 256).
    //   3. `Hv % Hk == 0` — the `hv → hk` GQA fan-out must be integer.
    //
    // `mt_gated_delta_step` takes `(Dk, Dv, Hv, Hk)` as runtime
    // `#[constexpr]` scalars: a single PSO serves every model config,
    // so there is no per-tuple emission list to gate on. The checks
    // below are the genuine dispatch-geometry invariants.

    /// Per-lane state register array cap in the kernel (`decayed[8]` /
    /// `k_cache[8]`) — `Dk / 32` columns per lane must not exceed it.
    public static let gatedDeltaMaxStateColumns = 8

    /// Validate the dimensions for `Ops.gatedDeltaStep`. The kernel is
    /// reduction-mode with a fixed 32-lane threadgroup; the dimensions
    /// are runtime constexprs so any geometry satisfying the simdgroup
    /// and GQA-fan-out invariants is dispatchable.
    public static func validateGatedDeltaStep(
        keyHeadDim: Int, valueHeadDim: Int,
        numKeyHeads: Int, numValueHeads: Int
    ) -> String? {
        if keyHeadDim <= 0 || valueHeadDim <= 0 {
            return "head dims must be positive (got Dk=\(keyHeadDim), Dv=\(valueHeadDim))"
        }
        if numKeyHeads <= 0 || numValueHeads <= 0 {
            return "head counts must be positive (got Hk=\(numKeyHeads), Hv=\(numValueHeads))"
        }
        if !keyHeadDim.isMultiple(of: 32) {
            return
                "keyHeadDim (\(keyHeadDim)) must be a multiple of 32 — each of the 32 lanes owns Dk/32 state columns"
        }
        if keyHeadDim / 32 > gatedDeltaMaxStateColumns {
            return
                "keyHeadDim (\(keyHeadDim)) exceeds the kernel's per-lane state cap — Dk/32 must be ≤ \(gatedDeltaMaxStateColumns) (Dk ≤ \(gatedDeltaMaxStateColumns * 32))"
        }
        if !numValueHeads.isMultiple(of: numKeyHeads) {
            return
                "numValueHeads (\(numValueHeads)) must be a multiple of numKeyHeads (\(numKeyHeads)) for integer GQA fan-out"
        }
        return nil
    }

    // ─── ffai_rms_norm_residual + ffai_gated_rmsnorm ───────────────
    //
    // Both kernels share `mt_add_rms_norm`'s row-width contract:
    //   * `N = TPG * 4` → caller picks `TPG = n / 4`.
    //   * `TPG` must be a multiple of 32 (simdgroup floor) and ≤ 1024.
    //   * Combined: `n` multiple of 128 and `n ≤ 4096`.
    // Wider rows must fall back to unfused `add` + `rmsNorm` /
    // `silu(z) * w * rmsNorm(y)`.

    /// Validate the row-width parameter for `Ops.rmsNormResidual`.
    public static func validateRmsNormResidual(n: Int) -> String? {
        if n <= 0 {
            return "n must be positive (got \(n))"
        }
        if !n.isMultiple(of: 128) {
            return "n=\(n) must be a multiple of 128 (32-lane simdgroup × 4 elements/thread)"
        }
        if n > 4096 {
            return
                "n must be ≤ 4096 (got \(n)); the kernel's TPG = n/4 exceeds the 1024-thread group cap"
        }
        return nil
    }

    /// Validate the row-width parameter for `Ops.gatedRmsNorm`.
    public static func validateGatedRmsNorm(n: Int) -> String? {
        // Same invariants as `mt_add_rms_norm` / `rms_norm_residual` —
        // TPG = n/4, multiple of 32, ≤ 1024.
        if n <= 0 {
            return "n must be positive (got \(n))"
        }
        if !n.isMultiple(of: 128) {
            return "n=\(n) must be a multiple of 128 (32-lane simdgroup × 4 elements/thread)"
        }
        if n > 4096 {
            return
                "n must be ≤ 4096 (got \(n)); the kernel's TPG = n/4 exceeds the 1024-thread group cap"
        }
        return nil
    }

    // ─── mt_rms_norm_small ─────────────────────────────────────────
    //
    // Per-head RMSNorm specialisation for n < 128 (the default
    // `mt_rms_norm` floor). 2 elements/thread → TPG = n/2:
    //   * TPG ≥ 32 (one full simdgroup) → n ≥ 64.
    //   * TPG ≤ 1024 (Apple cap) → n ≤ 2048.
    //   * n even (each thread loads 2 contiguous elements).

    public static func validateRmsNormSmall(n: Int) -> String? {
        if n <= 0 {
            return "n must be positive (got \(n))"
        }
        if !n.isMultiple(of: 2) {
            return "n=\(n) must be even (2 elements per thread)"
        }
        if n < 64 {
            return
                "n=\(n) must be ≥ 64 — TPG = n/2 below 32 makes the simdgroup reduction degenerate"
        }
        if n > 2048 {
            return
                "n=\(n) must be ≤ 2048 — TPG = n/2 exceeds the 1024-thread group cap; use rmsNorm or rmsNormWide for larger rows"
        }
        return nil
    }

    // ─── logits processors ─────────────────────────────────────────
    //
    // Five sampling-pipeline kernels:
    //   * temperature / topk-mask / repetition-penalty: Grid3D, pure
    //     elementwise. Only contracts are scalar parameter sanity and
    //     n > 0 (vocab non-empty).
    //   * min-p / top-p mask: Reduction-mode, one threadgroup per row,
    //     TPG = 256. Contract is row-width `n > 0` + rows > 0 +
    //     `0 < p < 1`.

    public static func validateLogitsTemperature(n: Int, temperature: Float) -> String? {
        if n <= 0 {
            return "n must be positive (got \(n))"
        }
        if !temperature.isFinite || temperature <= 0 {
            return
                "temperature must be a positive finite float (got \(temperature)); use Ops.argmax for greedy/zero-temperature sampling"
        }
        return nil
    }

    public static func validateLogitsRepetitionPenalty(
        vocab: Int, nTokenIds: Int, penalty: Float
    ) -> String? {
        if vocab <= 0 {
            return "vocab size must be positive (got \(vocab))"
        }
        if nTokenIds < 0 {
            return "tokenIds count must be non-negative (got \(nTokenIds))"
        }
        if !penalty.isFinite || penalty <= 0 {
            return "penalty must be a positive finite float (got \(penalty)); 1.0 disables"
        }
        return nil
    }

    public static func validateLogitsTopKMask(n: Int) -> String? {
        if n <= 0 {
            return "n must be positive (got \(n))"
        }
        return nil
    }

    public static func validateLogitsMinPMask(n: Int, rows: Int, minP: Float) -> String? {
        if n <= 0 {
            return "n must be positive (got \(n))"
        }
        if rows <= 0 {
            return "rows must be positive (got \(rows))"
        }
        if !minP.isFinite || minP <= 0 || minP >= 1 {
            return "minP must satisfy 0 < minP < 1 (got \(minP))"
        }
        return nil
    }

    public static func validateLogitsTopPMask(n: Int, rows: Int, topP: Float) -> String? {
        if n <= 0 {
            return "n must be positive (got \(n))"
        }
        if rows <= 0 {
            return "rows must be positive (got \(rows))"
        }
        if !topP.isFinite || topP <= 0 || topP >= 1 {
            return "topP must satisfy 0 < topP < 1 (got \(topP))"
        }
        return nil
    }
}
