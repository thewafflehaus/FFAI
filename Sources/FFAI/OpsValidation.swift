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
    // Kernel invariants (from `crates/metaltile-std/src/mlx/rms_norm.rs`
    // §"DISPATCH INVARIANTS"):
    //   1. `N = TPG * 4` — each thread owns 4 consecutive elements.
    //      The wrapper computes `TPG = n / 4`.
    //   2. `TPG` must be a multiple of 32 (cross-simdgroup reduction).
    //      Combined with (1): `n` must be a multiple of 128.
    //   3. `TPG ≤ 1024` (Apple's max-threads-per-threadgroup cap).
    //      Combined with (1): `n ≤ 4096`. Larger rows need
    //      `rmsNormRows` chunked dispatch (forthcoming).

    /// Validate the row-width parameter for `Ops.rmsNorm` and
    /// `Ops.rmsNormRows`. The single-row and multi-row dispatches
    /// share the per-row invariant.
    public static func validateRmsNorm(n: Int) -> String? {
        if n <= 0 {
            return "n must be positive (got \(n))"
        }
        if !n.isMultiple(of: 128) {
            return "n=\(n) must be a multiple of 128 (32-lane simdgroup × 4 elements/thread)"
        }
        if n / 4 > 1024 {
            return "n=\(n) > 4096 — exceeds the 1024-thread cap of this kernel; use rmsNormRows or a chunked variant for larger rows"
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
            return "head_dim must be one of {\(supported)} (got \(headDim)); other specializations not yet emitted"
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
            return "nKV (\(nKV)) must not exceed kvStride (\(kvStride)) — kernel would read past cache"
        }
        // Sliding-window / sink fast-path contract (invariant 4).
        if sinkEnd != 0 || windowStart != 0 {
            if !slidingWindowSdpaHeadDims.contains(headDim) {
                let supported = slidingWindowSdpaHeadDims.sorted()
                    .map(String.init).joined(separator: ", ")
                return "sinkEnd/windowStart are only supported for head_dim ∈ {\(supported)} (got \(headDim)); the d64/d256 kernels are dense-only"
            }
            if sinkEnd < 0 {
                return "sinkEnd must be non-negative (got \(sinkEnd))"
            }
            if windowStart < 0 {
                return "windowStart must be non-negative (got \(windowStart))"
            }
            if sinkEnd > windowStart {
                return "sinkEnd (\(sinkEnd)) must not exceed windowStart (\(windowStart)) — overlapping sink + window ranges double-count in the online softmax"
            }
            if windowStart > nKV {
                return "windowStart (\(windowStart)) must not exceed nKV (\(nKV)) — window pass walks [windowStart, nKV)"
            }
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
        if !(bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8) {
            return "bits=\(bits) unsupported — must be one of 3, 4, 5, 6, or 8"
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
            return "inDim=\(inDim) must be a multiple of groupSize=\(groupSize) — partial trailing group would be silently dropped"
        }
        // Footgun 2: pack-alignment for pack-strided variants.
        if bits == 4 || bits == 8 {
            let valsPerPack = 32 / bits  // 8 for int4, 4 for int8
            if !inDim.isMultiple(of: valsPerPack) {
                return "inDim=\(inDim) must be a multiple of \(valsPerPack) for bits=\(bits) (pack-strided kernel — unaligned tail elements silently dropped)"
            }
            if !groupSize.isMultiple(of: valsPerPack) {
                return "groupSize=\(groupSize) must be a multiple of \(valsPerPack) for bits=\(bits) (packs_per_group must be exact)"
            }
        }
        // Footgun 3: scales/biases sizing.
        let nGroups = inDim / groupSize
        let expected = outDim * nGroups
        if scalesCount != expected {
            return "scales must have outDim × n_groups = \(outDim) × \(nGroups) = \(expected) elements, got \(scalesCount)"
        }
        if biasesCount != expected {
            return "biases must have outDim × n_groups = \(outDim) × \(nGroups) = \(expected) elements, got \(biasesCount)"
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
            return "dim=\(dim) must be a multiple of 32 (one Apple simdgroup); simd_sum reduction is undefined otherwise"
        }
        if dim > 1024 {
            return "dim=\(dim) > 1024 — exceeds the kernel's TPG cap (shared_unit allocates 1024 slots)"
        }
        if bits != 2 && bits != 3 && bits != 4 && bits != 8 {
            return "bits=\(bits) unsupported — encode kernel emits only int2/int3/int4/int8 variants"
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
            return "headDim=\(headDim) must be a multiple of groupSize=\(groupSize) — partial trailing group would be silently dropped"
        }
        // Footguns 2 + 3: pack alignment. vals_per_pack = 32/bits.
        let valsPerPack = 32 / bits  // 8 for int4, 4 for int8
        if !headDim.isMultiple(of: valsPerPack) {
            return "headDim=\(headDim) must be a multiple of \(valsPerPack) for bits=\(bits) (pack-strided kernel — unaligned tail packed at the wrong offset)"
        }
        if !groupSize.isMultiple(of: valsPerPack) {
            return "groupSize=\(groupSize) must be a multiple of \(valsPerPack) for bits=\(bits) (packs_per_group must be exact)"
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
        if !(bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8) {
            return "bits=\(bits) unsupported — must be one of 3, 4, 5, 6, or 8"
        }
        if hidden <= 0 {
            return "hidden must be positive (got \(hidden))"
        }
        if groupSize <= 0 {
            return "groupSize must be positive (got \(groupSize))"
        }
        if !hidden.isMultiple(of: groupSize) {
            return "hidden=\(hidden) must be a multiple of groupSize=\(groupSize) — partial trailing group would read scale/bias at the wrong index"
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
            return "cacheStride (\(cacheStride)) must be >= tokens (\(tokens)) — smaller stride makes heads 1…n read/write the wrong rows (AURA coherent-then-collapse bug)"
        }
        // Contract 2: packedWidth must cover every dim. The clean path
        // packs `32/bits` dims per word; the odd-width path packs
        // `ceil(32/bits)`. Both need `packedWidth * dims_per_word >= dim`;
        // the clean (smaller) `dims_per_word` is the binding lower bound.
        let dimsPerWord = 32 / bits  // floor; clean path. ≥ odd-path stride.
        let minPackedWidth = (dim + dimsPerWord - 1) / dimsPerWord
        if packedWidth < minPackedWidth {
            return "packedWidth=\(packedWidth) too small for dim=\(dim) at bits=\(bits) — need >= ceil(dim / \(dimsPerWord)) = \(minPackedWidth); trailing dims would be left as zeros"
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
    // `crates/metaltile-std/src/ffai/gated_delta_step.rs`
    // §"DISPATCH INVARIANTS"):
    //   1. TPG = 32 (exactly one simdgroup). The wrapper hard-codes
    //      this; the only caller-facing risk is the grid dimension.
    //   2. `Dk % 32 == 0` — each lane owns `Dk / 32` state columns.
    //   3. `Hv % Hk == 0` — the `hv → hk` GQA fan-out must be integer.
    //   4. `(Dk, Dv, Hk, Hv)` baked into the kernel as compile-time
    //      constants → only the emitted instantiations are dispatchable.
    //
    // The `(Dk, Dv, Hk, Hv)` tuples for which a kernel is emitted, as
    // `keyHeadDim_valueHeadDim_numKeyHeads_numValueHeads`. Mirrors the
    // `gated_delta_step_kernel!` instantiations in the kernel source.
    public static let supportedGatedDeltaConfigs: Set<[Int]> = [
        [192, 128, 4, 4],    // Qwen3.5-A3B
        [128, 128, 8, 8],
        [128, 128, 16, 16],  // Qwen3.5 dense 0.8B-9B
        [128, 128, 16, 32],  // Qwen3.5-35B
        [128, 128, 16, 48],  // Qwen3.5 dense 27B
        [64, 64, 8, 8],      // Qwen3.5 small
    ]

    /// Validate the dimensions for `Ops.gatedDeltaStep`. The kernel is
    /// reduction-mode and only the emitted `(Dk, Dv, Hk, Hv)` tuples
    /// have a dispatchable specialization.
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
            return "keyHeadDim (\(keyHeadDim)) must be a multiple of 32 — each of the 32 lanes owns Dk/32 state columns"
        }
        if !numValueHeads.isMultiple(of: numKeyHeads) {
            return "numValueHeads (\(numValueHeads)) must be a multiple of numKeyHeads (\(numKeyHeads)) for integer GQA fan-out"
        }
        let config = [keyHeadDim, valueHeadDim, numKeyHeads, numValueHeads]
        if !supportedGatedDeltaConfigs.contains(config) {
            return "no gated_delta_step kernel emitted for (Dk,Dv,Hk,Hv)=\(config); add the instantiation in gated_delta_step.rs"
        }
        return nil
    }
}
