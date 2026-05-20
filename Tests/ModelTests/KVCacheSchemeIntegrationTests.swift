// KV-cache-scheme integration: load the same Qwen3 1.7B bf16 model
// under every KV cache scheme FFAI exposes and assert each one
// produces coherent generated text. Pins the contract that every
// KVCacheKind round-trips real K/V values well enough that the
// downstream SDPA produces good tokens.
//
// One @Test per scheme so Swift Testing's `.serialized` trait lets
// the previous Model go out of scope before the next one loads.
// Each scheme loads + decodes ~64 tokens; total wall-clock is the
// load time × N schemes (no KV write/decode amortization across
// schemes). Skipped if network/checkpoint isn't available.

import Foundation
import Testing
@testable import FFAI

@Suite("KV cache schemes — coherent output", .serialized)
struct KVCacheSchemeIntegrationTests {

    /// Common test fixture — load Qwen3 1.7B bf16 under `kvCache`
    /// and run `Once upon a time…` → 64 greedy-decoded tokens.
    /// Returns the GenerationResult; caller asserts coherence on
    /// the token stream and prints the decoded text.
    private func decode(_ kvCache: KVCacheKind) async throws -> GenerationResult? {
        let modelId = "mlx-community/Qwen3-1.7B-bf16"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 64

        let m: Model
        do {
            let opts = LoadOptions(kvCache: kvCache)
            m = try await ModelLoadLock.shared.loadSerially {
                try await Model.load(modelId, options: opts)
            }
        } catch {
            print("KV-cache-scheme test (\(kvCache)) skipped: \(error)")
            return nil
        }

        return try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
    }

    @Test("raw bf16 KV cache produces coherent output")
    func rawScheme() async throws {
        guard let result = try await decode(.raw) else { return }
        print("[KV=raw] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "raw")
    }

    @Test("affineQuantized int8 KV cache produces coherent output")
    func affineInt8Scheme() async throws {
        guard let result = try await decode(.affineQuantized(bits: 8, groupSize: 64)) else { return }
        print("[KV=affine8] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "affine8")
    }

    @Test("affineQuantized int4 KV cache produces coherent output")
    func affineInt4Scheme() async throws {
        guard let result = try await decode(.affineQuantized(bits: 4, groupSize: 64)) else { return }
        print("[KV=affine4] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "affine4")
    }

    // ─── AURA + per-layer SRHT rotation ────────────────────────────────
    //
    // Every AURA recipe below now runs with a per-layer SRHT rotation
    // Π_l (deterministic seed = layer index). The model applies Π_l to
    // Q after RoPE so SDPA scores cancel out, and applies Π_l^T to the
    // SDPA output so the residual stream stays in the original
    // activation space. See `AURAQuantizedKVCache` header for the math.
    //
    // Phase 5d.E Stage 1a landed the infrastructure (Ops.auraRotatePerHead
    // + per-layer Π_l + Q/output rotation in Qwen3/Llama forward).
    //
    // The earlier "coherent then collapse around index 50" failure was a
    // dequant-kernel stride bug, not a codec / DC-bias / Stage-2 issue:
    // `aura_dequant_rotated` keys all per-head offset arithmetic off its
    // `tokens` constexpr, but `AURAQuantizedKVCache`'s buffers are laid
    // out `[nKVHeads, maxSeq, …]`. `prepareForAttention` passed the fill
    // count (`length`) as that constexpr, so every head past head 0 was
    // dequanted at the wrong offset, with the error growing as the cache
    // filled. Fix: `Ops.auraDequantRotated` now takes an explicit
    // `cacheStride` (= `maxSeq`) for the kernel's stride arithmetic while
    // the grid height stays the row count to process.

    @Test(
        "auraQuantized aura4v4 (symmetric, SRHT rotation) produces coherent output"
    )
    func auraSymmetric4v4() async throws {
        guard let result = try await decode(.auraQuantized(scheme: .default)) else { return }
        print("[KV=aura4v4] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "aura4v4")
    }

    @Test(
        "auraQuantized aura4v2 (asymmetric K/V, SRHT rotation) produces coherent output"
    )
    func auraAsymmetric4v2() async throws {
        guard let result = try await decode(.auraQuantized(scheme: .aura4v2)) else { return }
        print("[KV=aura4v2] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "aura4v2")
    }

    @Test("auraQuantized aura8v4 (asymmetric K/V, 8-bit K + 4-bit V) produces coherent output")
    func auraAsymmetric8v4() async throws {
        // aura8v4 — exercises the kb=8 path in aura_score and the
        // vb=4 path in aura_value / aura_flash_p1 in one config.
        let scheme = AURAScheme(keyBits: 8, valueBits: 4)
        guard let result = try await decode(.auraQuantized(scheme: scheme)) else { return }
        print("[KV=aura8v4] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "aura8v4")
    }

    @Test(
        "auraQuantized aura8v8 (symmetric 8-bit K/V, SRHT rotation) produces coherent output"
    )
    func auraSymmetric8v8() async throws {
        // aura8v8 — highest-precision AURA recipe; exercises kb=8 +
        // vb=8 through every codec kernel.
        let scheme = AURAScheme(keyBits: 8, valueBits: 8)
        guard let result = try await decode(.auraQuantized(scheme: scheme)) else { return }
        print("[KV=aura8v8] \(result.text)")
        expectCoherentOutput(result.generatedTokens, minTokens: 8, label: "aura8v8")
    }
}
