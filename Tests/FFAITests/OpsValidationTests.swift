// OpsValidation tests — verify every wrapper precondition rejects
// bad inputs and accepts good ones, without producing a process trap.
//
// Background: each `Ops.*` wrapper around a reduction-mode metaltile
// kernel calls `preconditionFailure` when dispatch invariants are
// violated. Those traps halt the entire test process, so the
// pure-function `OpsValidation.validate*` shape (returning
// `String?`) lets us exercise the same logic in CI.
//
// New reduction-mode wrappers should add a test case here in the
// same commit as the wrapper. See `Sources/FFAI/OpsValidation.swift`
// for the validation surface and the post-mortem
// `papers/post-mortem-2026-05-19-dispatch-shape-gpu-freeze.md` for
// why this exists.

import Testing
@testable import FFAI

@Suite("OpsValidation — wrapper preconditions")
struct OpsValidationTests {

    // ─── rmsNorm / rmsNormRows ─────────────────────────────────────

    @Test("rmsNorm accepts legal sizes")
    func rmsNormAcceptsLegal() {
        // Smallest legal n: 128 (one full simdgroup × 4 elts/thread).
        #expect(OpsValidation.validateRmsNorm(n: 128) == nil)
        // Common Llama hidden dims, all multiples of 128 ≤ 4096.
        #expect(OpsValidation.validateRmsNorm(n: 256) == nil)
        #expect(OpsValidation.validateRmsNorm(n: 1024) == nil)
        #expect(OpsValidation.validateRmsNorm(n: 2048) == nil)
        #expect(OpsValidation.validateRmsNorm(n: 4096) == nil)
    }

    @Test("rmsNorm rejects n=0 and negative")
    func rmsNormRejectsNonPositive() {
        #expect(OpsValidation.validateRmsNorm(n: 0) != nil)
        #expect(OpsValidation.validateRmsNorm(n: -128) != nil)
    }

    @Test("rmsNorm rejects n not multiple of 128")
    func rmsNormRejectsNonMultipleOf128() {
        // The 2026-05-19 GPU freeze precursor: this was n=4 + the wrapper
        // computed tgWidth=1, generated MSL with n_simd=0 → tg_ssq=0.
        #expect(OpsValidation.validateRmsNorm(n: 4) != nil)
        #expect(OpsValidation.validateRmsNorm(n: 32) != nil)
        #expect(OpsValidation.validateRmsNorm(n: 100) != nil)
        // Right next to 128 — these all fail the mod check.
        #expect(OpsValidation.validateRmsNorm(n: 127) != nil)
        #expect(OpsValidation.validateRmsNorm(n: 129) != nil)
    }

    @Test("rmsNorm rejects n above 4096 (TPG cap)")
    func rmsNormRejectsTooLargeN() {
        // n / 4 > 1024 → TPG exceeds Apple's max-threads-per-threadgroup.
        #expect(OpsValidation.validateRmsNorm(n: 4224) != nil)  // 4224/4 = 1056
        #expect(OpsValidation.validateRmsNorm(n: 8192) != nil)
    }

    // ─── sdpaDecode ────────────────────────────────────────────────

    @Test("sdpaDecode accepts production shapes")
    func sdpaDecodeAcceptsLegal() {
        // Llama 3.1 (32 q-heads, 8 kv-heads = 4x GQA, head_dim=128).
        #expect(OpsValidation.validateSdpaDecode(
            headDim: 128, nQHeads: 32, nKVHeads: 8,
            nKV: 100, kvStride: 4096) == nil)
        // Qwen3 0.6B (16 q-heads, 8 kv-heads = 2x GQA, head_dim=128).
        #expect(OpsValidation.validateSdpaDecode(
            headDim: 128, nQHeads: 16, nKVHeads: 8,
            nKV: 0, kvStride: 1) == nil)
        // No GQA, head_dim=128.
        #expect(OpsValidation.validateSdpaDecode(
            headDim: 128, nQHeads: 8, nKVHeads: 8,
            nKV: 1024, kvStride: 4096) == nil)
        // Llama 3.2 1B: head_dim=64, 32 q-heads, 8 kv-heads.
        #expect(OpsValidation.validateSdpaDecode(
            headDim: 64, nQHeads: 32, nKVHeads: 8,
            nKV: 256, kvStride: 4096) == nil)
        // GPT-OSS-20B sliding-window layers: head_dim=64, 64 q-heads,
        // 8 kv-heads.
        #expect(OpsValidation.validateSdpaDecode(
            headDim: 64, nQHeads: 64, nKVHeads: 8,
            nKV: 1, kvStride: 128) == nil)
    }

    @Test("sdpaDecode rejects head_dim outside {64, 128, 256}")
    func sdpaDecodeRejectsBadHeadDim() {
        // 2026-05-19 GPU freeze trigger: head_dim=4 with the elementwise
        // sizing helper → 4 threads → n_simd=0 → infinite loop. The set
        // also covers values we've never specialized.
        for badHeadDim in [4, 32, 96, 127, 129, 192, 512] {
            #expect(OpsValidation.validateSdpaDecode(
                headDim: badHeadDim, nQHeads: 8, nKVHeads: 8,
                nKV: 1, kvStride: 4) != nil,
                "head_dim=\(badHeadDim) should be rejected")
        }
    }

    @Test("supportedSdpaHeadDims is what wrappers route to")
    func supportedHeadDimsMatchKernels() {
        // Pin the set so adding a new head_dim variant in metaltile
        // requires updating both this set and the dispatch switch in
        // Ops.sdpaDecode in the same commit. Gemma 3 + Gemma 4 added
        // head_dim=256 in the Phase 6 wave.
        #expect(OpsValidation.supportedSdpaHeadDims == [64, 128, 256])
    }

    @Test("sdpaDecode rejects non-integer GQA fan-out")
    func sdpaDecodeRejectsBadGQA() {
        #expect(OpsValidation.validateSdpaDecode(
            headDim: 128, nQHeads: 7, nKVHeads: 4,
            nKV: 1, kvStride: 4) != nil)
        #expect(OpsValidation.validateSdpaDecode(
            headDim: 128, nQHeads: 32, nKVHeads: 5,
            nKV: 1, kvStride: 4) != nil)
    }

    @Test("sdpaDecode rejects zero / negative head counts")
    func sdpaDecodeRejectsBadHeadCounts() {
        #expect(OpsValidation.validateSdpaDecode(
            headDim: 128, nQHeads: 0, nKVHeads: 1,
            nKV: 1, kvStride: 4) != nil)
        #expect(OpsValidation.validateSdpaDecode(
            headDim: 128, nQHeads: 8, nKVHeads: 0,
            nKV: 1, kvStride: 4) != nil)
    }

    @Test("sdpaDecode rejects n_kv > kv_stride")
    func sdpaDecodeRejectsOverflowingNKV() {
        // Walking past the pre-allocated cache → OOB reads on K/V.
        #expect(OpsValidation.validateSdpaDecode(
            headDim: 128, nQHeads: 8, nKVHeads: 8,
            nKV: 10, kvStride: 4) != nil)
        // n_kv == kv_stride is fine (cache exactly full).
        #expect(OpsValidation.validateSdpaDecode(
            headDim: 128, nQHeads: 8, nKVHeads: 8,
            nKV: 4, kvStride: 4) == nil)
    }

    // ─── auraEncode ────────────────────────────────────────────────

    @Test("auraEncode accepts production shapes")
    func auraEncodeAcceptsLegal() {
        // AURA scheme dims from the planning doc — all multiples of 32
        // and ≤ 1024.
        for d in [64, 96, 128, 192, 256, 512, 1024] {
            #expect(OpsValidation.validateAuraEncode(rows: 1, dim: d, bits: 4) == nil,
                    "dim=\(d) should be legal")
        }
        // All supported bit-widths at production dim=128.
        for b in [2, 3, 4, 8] {
            #expect(OpsValidation.validateAuraEncode(rows: 1, dim: 128, bits: b) == nil,
                    "bits=\(b) should be legal")
        }
    }

    @Test("auraEncode rejects bad dim")
    func auraEncodeRejectsBadDim() {
        #expect(OpsValidation.validateAuraEncode(rows: 1, dim: 0, bits: 4) != nil)
        #expect(OpsValidation.validateAuraEncode(rows: 1, dim: -32, bits: 4) != nil)
        // dim must be multiple of 32 (simdgroup width).
        #expect(OpsValidation.validateAuraEncode(rows: 1, dim: 16, bits: 4) != nil)
        #expect(OpsValidation.validateAuraEncode(rows: 1, dim: 33, bits: 4) != nil)
        // dim > 1024 exceeds shared_unit alloc.
        #expect(OpsValidation.validateAuraEncode(rows: 1, dim: 2048, bits: 4) != nil)
    }

    @Test("auraEncode rejects unsupported bit-widths")
    func auraEncodeRejectsBadBits() {
        // Only int2/3/4/8 emitted; everything else should trap.
        for badBits in [0, 1, 5, 6, 7, 9, 16] {
            #expect(OpsValidation.validateAuraEncode(rows: 1, dim: 128, bits: badBits) != nil,
                    "bits=\(badBits) should be rejected")
        }
    }

    @Test("auraEncode rejects zero / negative rows")
    func auraEncodeRejectsBadRows() {
        #expect(OpsValidation.validateAuraEncode(rows: 0, dim: 128, bits: 4) != nil)
        #expect(OpsValidation.validateAuraEncode(rows: -1, dim: 128, bits: 4) != nil)
    }

    // ─── gemv ──────────────────────────────────────────────────────

    @Test("gemv accepts legal shapes")
    func gemvAcceptsLegal() {
        #expect(OpsValidation.validateGemv(outDim: 4096, inDim: 4096) == nil)
        #expect(OpsValidation.validateGemv(outDim: 1, inDim: 1) == nil)
    }

    @Test("gemv rejects zero / negative dims")
    func gemvRejectsBadDims() {
        #expect(OpsValidation.validateGemv(outDim: 0, inDim: 256) != nil)
        #expect(OpsValidation.validateGemv(outDim: 256, inDim: 0) != nil)
        #expect(OpsValidation.validateGemv(outDim: -1, inDim: 256) != nil)
    }

    // ─── dequantGemv ───────────────────────────────────────────────

    @Test("dequantGemv accepts production shapes")
    func dequantGemvAcceptsLegal() {
        // int4 with groupSize=64. inDim=4096, outDim=4096 (Llama hidden).
        // n_groups = 4096/64 = 64. scales/biases = 4096*64 = 262144.
        #expect(OpsValidation.validateDequantGemv(
            outDim: 4096, inDim: 4096, bits: 4, groupSize: 64,
            scalesCount: 4096 * 64, biasesCount: 4096 * 64) == nil)
        // int8 with groupSize=32. inDim=128, outDim=64.
        // n_groups = 128/32 = 4. scales/biases = 64*4 = 256.
        #expect(OpsValidation.validateDequantGemv(
            outDim: 64, inDim: 128, bits: 8, groupSize: 32,
            scalesCount: 64 * 4, biasesCount: 64 * 4) == nil)
        // int6 (element-strided, no pack alignment).
        #expect(OpsValidation.validateDequantGemv(
            outDim: 2, inDim: 64, bits: 6, groupSize: 64,
            scalesCount: 2, biasesCount: 2) == nil)
    }

    @Test("dequantGemv rejects unsupported bits")
    func dequantGemvRejectsBadBits() {
        for badBits in [0, 1, 2, 7, 9, 16, -4] {
            #expect(OpsValidation.validateDequantGemv(
                outDim: 64, inDim: 128, bits: badBits, groupSize: 32,
                scalesCount: 0, biasesCount: 0) != nil,
                "bits=\(badBits) should be rejected")
        }
    }

    @Test("dequantGemv rejects partial trailing group (silent-miscompute footgun)")
    func dequantGemvRejectsPartialGroup() {
        // inDim=130, groupSize=64 → n_groups=2 (integer), but kernel
        // walks 128 elements only, dropping 2. Silent miscompute.
        #expect(OpsValidation.validateDequantGemv(
            outDim: 4, inDim: 130, bits: 4, groupSize: 64,
            scalesCount: 0, biasesCount: 0) != nil)
        // inDim=100, groupSize=32 → n_groups=3 (truncated from 3.125).
        #expect(OpsValidation.validateDequantGemv(
            outDim: 4, inDim: 100, bits: 4, groupSize: 32,
            scalesCount: 0, biasesCount: 0) != nil)
    }

    @Test("dequantGemv rejects unaligned pack tail for int4/int8")
    func dequantGemvRejectsUnalignedPacks() {
        // int4: vals_per_pack=8. inDim must be multiple of 8.
        // inDim=72, groupSize=8 → passes group-divisibility but fails pack alignment? Actually 72%8==0.
        // Need inDim that's group-divisible but not pack-divisible. group_size=72/9=8 doesn't work since 72%8==0.
        // Hmm — for int4, both group and pack constraints are multiples of 8, so any group-divisible inDim
        // is also pack-divisible. The pack check only bites when groupSize is bigger than valsPerPack.
        // Example: inDim=48, bits=4, groupSize=48 → n_groups=1 ✓; vals_per_pack=8, 48%8==0 ✓. Passes.
        // Need a case where group_size isn't pack-aligned: e.g. groupSize=6 (not multiple of 8).
        // But groupSize=6 isn't typical. Let me use groupSize=24, bits=4 (24 not multiple of 8).
        #expect(OpsValidation.validateDequantGemv(
            outDim: 4, inDim: 96, bits: 4, groupSize: 24,
            scalesCount: 0, biasesCount: 0) != nil)
        // int8: vals_per_pack=4. groupSize=6 fails pack alignment.
        #expect(OpsValidation.validateDequantGemv(
            outDim: 4, inDim: 96, bits: 8, groupSize: 6,
            scalesCount: 0, biasesCount: 0) != nil)
    }

    @Test("dequantGemv rejects scales/biases sizing mismatch")
    func dequantGemvRejectsBadScalesBiases() {
        // outDim=4, inDim=128, groupSize=64 → n_groups=2, expected scales=8.
        // Too few:
        #expect(OpsValidation.validateDequantGemv(
            outDim: 4, inDim: 128, bits: 4, groupSize: 64,
            scalesCount: 4, biasesCount: 8) != nil)
        // Biases wrong:
        #expect(OpsValidation.validateDequantGemv(
            outDim: 4, inDim: 128, bits: 4, groupSize: 64,
            scalesCount: 8, biasesCount: 4) != nil)
        // Exactly right:
        #expect(OpsValidation.validateDequantGemv(
            outDim: 4, inDim: 128, bits: 4, groupSize: 64,
            scalesCount: 8, biasesCount: 8) == nil)
    }

    @Test("dequantGemv rejects non-positive dims")
    func dequantGemvRejectsNonPositive() {
        #expect(OpsValidation.validateDequantGemv(
            outDim: 0, inDim: 128, bits: 4, groupSize: 64,
            scalesCount: 0, biasesCount: 0) != nil)
        #expect(OpsValidation.validateDequantGemv(
            outDim: 4, inDim: 0, bits: 4, groupSize: 64,
            scalesCount: 0, biasesCount: 0) != nil)
        #expect(OpsValidation.validateDequantGemv(
            outDim: 4, inDim: 128, bits: 4, groupSize: 0,
            scalesCount: 0, biasesCount: 0) != nil)
    }

    // ─── Failure messages are useful ───────────────────────────────

    @Test("Failure messages reference the offending value")
    func failureMessagesAreUseful() {
        // The point of returning String? instead of Bool: callers and
        // the user see WHY the dispatch was rejected, not just THAT it
        // was. This pins the contract.
        let rmsMsg = OpsValidation.validateRmsNorm(n: 100)
        #expect(rmsMsg?.contains("100") == true)
        #expect(rmsMsg?.contains("128") == true)

        // head_dim=192 is unsupported (no specialization). The message
        // mentions the offending value + the supported set.
        let sdpaMsg = OpsValidation.validateSdpaDecode(
            headDim: 192, nQHeads: 1, nKVHeads: 1, nKV: 0, kvStride: 1)
        #expect(sdpaMsg?.contains("192") == true)
        #expect(sdpaMsg?.contains("128") == true)

        let auraMsg = OpsValidation.validateAuraEncode(rows: 1, dim: 33, bits: 4)
        #expect(auraMsg?.contains("33") == true)
        #expect(auraMsg?.contains("32") == true)
    }
}
