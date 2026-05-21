// QuantizedOpsValidation tests — verify the wrapper preconditions for
// the quantized Ops (KV cache quant/dequant, dequant-gather, AURA
// dequant + per-head rotation) reject bad inputs and accept good ones,
// without producing a process trap.
//
// Background: each `Ops.*` wrapper around a quantized metaltile kernel
// calls `preconditionFailure` when its dispatch contract is violated.
// Those traps halt the whole test process, so the pure-function
// `OpsValidation.validate*` shape (returning `String?`) lets us
// exercise the same logic in CI.
//
// These quantized kernels are all Grid3D mode — no GPU-pin hazard —
// but they carry silent-miscompute footguns: integer-truncating
// divisions (`head_dim / group_size`, `head_dim / vals_per_pack`,
// `hidden / group_size`) baked into the kernel's offset arithmetic,
// and the AURA `cacheStride` row-stride contract whose violation is
// the "coherent then collapse" bug. See `Sources/FFAI/OpsValidation.swift`
// and the post-mortem `papers/post-mortem-2026-05-19-dispatch-shape-gpu-freeze.md`.

import Testing
@testable import FFAI

@Suite("QuantizedOpsValidation — quantized wrapper preconditions")
struct QuantizedOpsValidationTests {

    // ─── quantizeKV / bulkDequantKV (int4 / int8) ──────────────────

    @Test("validateQuantizeKV accepts production shapes")
    func quantizeKVAcceptsLegal() {
        // Llama 3.2 KV: 8 kv-heads, head_dim=128, group_size=64.
        // int4: vals_per_pack=8 — 128 % 8 == 0, 64 % 8 == 0.
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 8, headDim: 128, groupSize: 64, bits: 4) == nil)
        // int8: vals_per_pack=4 — 128 % 4 == 0, 64 % 4 == 0.
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 8, headDim: 128, groupSize: 64, bits: 8) == nil)
        // head_dim == group_size (one group per head).
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 4, headDim: 64, groupSize: 64, bits: 4) == nil)
        // group_size = 32 — multiple of both 8 and 4.
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 1, headDim: 256, groupSize: 32, bits: 4) == nil)
    }

    @Test("validateQuantizeKV rejects unsupported bit-widths")
    func quantizeKVRejectsBadBits() {
        // KV quant only emits int4/int8.
        for badBits in [0, 1, 2, 3, 5, 6, 16, -4] {
            #expect(OpsValidation.validateQuantizeKV(
                nKVHeads: 8, headDim: 128, groupSize: 64, bits: badBits) != nil,
                "bits=\(badBits) should be rejected")
        }
    }

    @Test("validateQuantizeKV rejects non-positive dims")
    func quantizeKVRejectsNonPositive() {
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 0, headDim: 128, groupSize: 64, bits: 4) != nil)
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 8, headDim: 0, groupSize: 64, bits: 4) != nil)
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 8, headDim: 128, groupSize: 0, bits: 4) != nil)
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: -1, headDim: 128, groupSize: 64, bits: 4) != nil)
    }

    @Test("validateQuantizeKV rejects partial trailing group (silent-miscompute footgun)")
    func quantizeKVRejectsPartialGroup() {
        // head_dim=100, group_size=64 → groups_per_head=1 (truncated
        // from 1.56), tail 36 slots never written.
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 8, headDim: 100, groupSize: 64, bits: 4) != nil)
        // head_dim=192, group_size=128 → 1 group, tail 64 dropped.
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 8, headDim: 192, groupSize: 128, bits: 8) != nil)
    }

    @Test("validateQuantizeKV rejects pack-unaligned head_dim / group_size")
    func quantizeKVRejectsUnalignedPacks() {
        // int8: vals_per_pack=4. group_size=6 divides head_dim=132 and
        // 132 % 4 == 0, but group_size 6 isn't a multiple of 4 →
        // packs_per_group is inexact.
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 8, headDim: 132, groupSize: 6, bits: 8) != nil)
        // int4: vals_per_pack=8. group_size=12 (not multiple of 8)
        // divides head_dim=96 but fails the pack-alignment check.
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 8, headDim: 96, groupSize: 12, bits: 4) != nil)
        // head_dim not a multiple of vals_per_pack: int4 needs % 8.
        // head_dim=12, group_size=12 → group divides, but 12 % 8 != 0.
        #expect(OpsValidation.validateQuantizeKV(
            nKVHeads: 1, headDim: 12, groupSize: 12, bits: 4) != nil)
    }

    // ─── dequantGather ─────────────────────────────────────────────

    @Test("validateDequantGather accepts production shapes")
    func dequantGatherAcceptsLegal() {
        // Embedding gather: hidden=4096, group_size=64.
        for bits in [3, 4, 5, 6, 8] {
            #expect(OpsValidation.validateDequantGather(
                hidden: 4096, bits: bits, groupSize: 64) == nil,
                "bits=\(bits) should be legal at hidden=4096")
        }
        // hidden == group_size (one group).
        #expect(OpsValidation.validateDequantGather(
            hidden: 64, bits: 4, groupSize: 64) == nil)
        // No pack-alignment constraint — odd hidden is fine as long as
        // it's group-divisible (kernel walks the bit-stream per element).
        #expect(OpsValidation.validateDequantGather(
            hidden: 96, bits: 4, groupSize: 24) == nil)
    }

    @Test("validateDequantGather rejects unsupported bit-widths")
    func dequantGatherRejectsBadBits() {
        for badBits in [0, 1, 2, 7, 9, 16, -4] {
            #expect(OpsValidation.validateDequantGather(
                hidden: 4096, bits: badBits, groupSize: 64) != nil,
                "bits=\(badBits) should be rejected")
        }
    }

    @Test("validateDequantGather rejects non-positive dims")
    func dequantGatherRejectsNonPositive() {
        #expect(OpsValidation.validateDequantGather(
            hidden: 0, bits: 4, groupSize: 64) != nil)
        #expect(OpsValidation.validateDequantGather(
            hidden: 4096, bits: 4, groupSize: 0) != nil)
        #expect(OpsValidation.validateDequantGather(
            hidden: -64, bits: 4, groupSize: 64) != nil)
    }

    @Test("validateDequantGather rejects partial trailing group (silent-miscompute footgun)")
    func dequantGatherRejectsPartialGroup() {
        // hidden=100, group_size=64 → groups_per_row=1 (truncated),
        // trailing 36 elements read scale/bias at the wrong index.
        #expect(OpsValidation.validateDequantGather(
            hidden: 100, bits: 4, groupSize: 64) != nil)
        // hidden=130, group_size=64 → 2 groups exact, but 2 elements drop.
        #expect(OpsValidation.validateDequantGather(
            hidden: 130, bits: 4, groupSize: 64) != nil)
    }

    // ─── auraDequantRotated ────────────────────────────────────────

    @Test("validateAuraDequantRotated accepts production shapes")
    func auraDequantRotatedAcceptsLegal() {
        // dim=128, int4 → dims_per_word=8, packed_width=ceil(128/8)=16.
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 128, packedWidth: 16, tokens: 64, bits: 4,
            cacheStride: 4096) == nil)
        // int8 → dims_per_word=4, packed_width=ceil(128/4)=32.
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 128, packedWidth: 32, tokens: 1, bits: 8,
            cacheStride: 4096) == nil)
        // int2 → dims_per_word=16, packed_width=ceil(128/16)=8.
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 128, packedWidth: 8, tokens: 10, bits: 2,
            cacheStride: 10) == nil)
        // cacheStride == tokens (buffer exactly [nKVHeads, tokens, …]).
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 64, packedWidth: 8, tokens: 16, bits: 4,
            cacheStride: 16) == nil)
        // Over-wide packedWidth is harmless (trailing words unused).
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 128, packedWidth: 20, tokens: 4, bits: 4,
            cacheStride: 4) == nil)
    }

    @Test("validateAuraDequantRotated rejects unsupported bit-widths")
    func auraDequantRotatedRejectsBadBits() {
        // Wrapper routes only int2/int3/int4/int8.
        for badBits in [0, 1, 5, 6, 7, 9, 16, -2] {
            #expect(OpsValidation.validateAuraDequantRotated(
                dim: 128, packedWidth: 64, tokens: 1, bits: badBits,
                cacheStride: 1) != nil,
                "bits=\(badBits) should be rejected")
        }
    }

    @Test("validateAuraDequantRotated rejects non-positive dims")
    func auraDequantRotatedRejectsNonPositive() {
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 0, packedWidth: 16, tokens: 1, bits: 4, cacheStride: 1) != nil)
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 128, packedWidth: 16, tokens: 0, bits: 4, cacheStride: 1) != nil)
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 128, packedWidth: 0, tokens: 1, bits: 4, cacheStride: 1) != nil)
    }

    @Test("validateAuraDequantRotated rejects cacheStride < tokens (coherent-then-collapse bug)")
    func auraDequantRotatedRejectsBadStride() {
        // The AURA "coherent then collapse" footgun: passing the fill
        // count instead of maxSeq makes heads 1…n address wrong rows.
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 128, packedWidth: 16, tokens: 100, bits: 4,
            cacheStride: 50) != nil)
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 128, packedWidth: 16, tokens: 64, bits: 4,
            cacheStride: 63) != nil)
    }

    @Test("validateAuraDequantRotated rejects packedWidth too small for dim")
    func auraDequantRotatedRejectsUndersizedPackedWidth() {
        // dim=128, int4 → need packed_width >= ceil(128/8)=16.
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 128, packedWidth: 15, tokens: 1, bits: 4,
            cacheStride: 1) != nil)
        // dim=128, int8 → need packed_width >= ceil(128/4)=32.
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 128, packedWidth: 31, tokens: 1, bits: 8,
            cacheStride: 1) != nil)
        // dim=130, int4 → ceil(130/8)=17; packed_width=16 leaves dims uncovered.
        #expect(OpsValidation.validateAuraDequantRotated(
            dim: 130, packedWidth: 16, tokens: 1, bits: 4,
            cacheStride: 1) != nil)
    }

    // ─── auraRotatePerHead ─────────────────────────────────────────

    @Test("validateAuraRotatePerHead accepts production shapes")
    func auraRotatePerHeadAcceptsLegal() {
        // Qwen3 1.7B: 16 q-heads, head_dim=128 → x is [16*128].
        #expect(OpsValidation.validateAuraRotatePerHead(
            xElementCount: 16 * 128, rotationShape: [128, 128],
            rotationDtypeMatchesX: true,
            nHeads: 16, headDim: 128) == nil)
        // Single head.
        #expect(OpsValidation.validateAuraRotatePerHead(
            xElementCount: 64, rotationShape: [64, 64],
            rotationDtypeMatchesX: true,
            nHeads: 1, headDim: 64) == nil)
    }

    @Test("validateAuraRotatePerHead rejects non-positive head params")
    func auraRotatePerHeadRejectsNonPositive() {
        #expect(OpsValidation.validateAuraRotatePerHead(
            xElementCount: 0, rotationShape: [128, 128],
            rotationDtypeMatchesX: true, nHeads: 0, headDim: 128) != nil)
        #expect(OpsValidation.validateAuraRotatePerHead(
            xElementCount: 0, rotationShape: [0, 0],
            rotationDtypeMatchesX: true, nHeads: 16, headDim: 0) != nil)
    }

    @Test("validateAuraRotatePerHead rejects x element-count mismatch")
    func auraRotatePerHeadRejectsBadXCount() {
        // x has 2000 elements but nHeads*headDim = 16*128 = 2048.
        #expect(OpsValidation.validateAuraRotatePerHead(
            xElementCount: 2000, rotationShape: [128, 128],
            rotationDtypeMatchesX: true,
            nHeads: 16, headDim: 128) != nil)
    }

    @Test("validateAuraRotatePerHead rejects non-square / wrong rotation shape")
    func auraRotatePerHeadRejectsBadRotationShape() {
        // Rotation must be [headDim, headDim].
        #expect(OpsValidation.validateAuraRotatePerHead(
            xElementCount: 16 * 128, rotationShape: [128, 64],
            rotationDtypeMatchesX: true,
            nHeads: 16, headDim: 128) != nil)
        // 1D rotation.
        #expect(OpsValidation.validateAuraRotatePerHead(
            xElementCount: 16 * 128, rotationShape: [128],
            rotationDtypeMatchesX: true,
            nHeads: 16, headDim: 128) != nil)
        // Wrong size square.
        #expect(OpsValidation.validateAuraRotatePerHead(
            xElementCount: 16 * 128, rotationShape: [64, 64],
            rotationDtypeMatchesX: true,
            nHeads: 16, headDim: 128) != nil)
    }

    @Test("validateAuraRotatePerHead rejects rotation/x dtype mismatch")
    func auraRotatePerHeadRejectsDtypeMismatch() {
        // gemv requires matched dtypes.
        #expect(OpsValidation.validateAuraRotatePerHead(
            xElementCount: 16 * 128, rotationShape: [128, 128],
            rotationDtypeMatchesX: false,
            nHeads: 16, headDim: 128) != nil)
    }

    // ─── Failure messages are useful ───────────────────────────────

    @Test("Quantized-Op failure messages reference the offending value")
    func failureMessagesAreUseful() {
        // Same contract as OpsValidationTests: returning String? means
        // the caller sees WHY a dispatch was rejected.
        let kvMsg = OpsValidation.validateQuantizeKV(
            nKVHeads: 8, headDim: 100, groupSize: 64, bits: 4)
        #expect(kvMsg?.contains("100") == true)
        #expect(kvMsg?.contains("64") == true)

        let gatherMsg = OpsValidation.validateDequantGather(
            hidden: 100, bits: 4, groupSize: 64)
        #expect(gatherMsg?.contains("100") == true)

        let auraMsg = OpsValidation.validateAuraDequantRotated(
            dim: 128, packedWidth: 16, tokens: 100, bits: 4, cacheStride: 50)
        #expect(auraMsg?.contains("50") == true)
        #expect(auraMsg?.contains("100") == true)

        let rotMsg = OpsValidation.validateAuraRotatePerHead(
            xElementCount: 2000, rotationShape: [128, 128],
            rotationDtypeMatchesX: true, nHeads: 16, headDim: 128)
        #expect(rotMsg?.contains("2000") == true)
        #expect(rotMsg?.contains("2048") == true)
    }
}
