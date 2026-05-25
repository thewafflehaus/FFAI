// Slow integration test for Gemma 4. First-light verification that the
// family's architectural quirks compose into coherent generated text on
// a real checkpoint:
//
//   • two attention geometries (sliding head_dim 256 + global 512)
//   • ProportionalRoPE on the global layers
//   • value RMSNorm, per-head q/k norms, plain RMSNorm (no Gemma
//     `(1 + weight)` fold — Gemma 4 dropped that convention)
//   • leading `<bos>` prefix (Gemma 4's tokenizer post-processor
//     does not add one — `Gemma4Model.requiresLeadingBOS`)
//   • Per-Layer Embeddings (PLE) — E2B / E4B are PLE variants
//   • per-layer learned scalar
//   • sqrt(hidden) embed scale, GELU MLP, tied embeddings
//   • final logit soft-capping
//
// The smallest Gemma 4 checkpoint on mlx-community is the E2B size,
// which is itself a PLE (Gemma4E) variant — there is no PLE-free dense
// checkpoint below 31B and no MoE checkpoint below 26B, so the dense
// and MoE variants are guard-skipped here (no small/raw checkpoint).
// Tests are skipped entirely if the checkpoint is not available
// locally.

import Foundation
import Testing
@testable import FFAI

@Suite("Gemma 4 integration", .serialized)
struct Gemma4IntegrationTests {

    @Test("Gemma4E (E2B): load + greedy generate produces coherent output")
    func loadAndGenerateE2B() async throws {
        let modelId = "mlx-community/gemma-4-e2b-it-bf16"
        // Open-ended prompt: a short factual question ("The capital of
        // France is") elicits a 2-token answer ("Paris.") and the model
        // correctly stops, which can't exercise the coherence checker's
        // `minTokens` budget. A story opener keeps the model generating
        // sustained prose across every architectural path (sliding +
        // global attention, PLE, KV sharing, soft-capping).
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Gemma 4 E2B integration test skipped: \(error)")
            return
        }

        // E2B canonical shapes (from HF text_config):
        //   hidden 1536, 35 layers, 8 heads, sliding head_dim 256,
        //   1 sliding KV head.
        #expect(m.engine.hidden == 1536)
        #expect(m.engine.nLayers == 35)
        #expect(m.engine.nHeads == 8)
        #expect(m.engine.headDim == 256)

        // Resolved as the PLE (Gemma4E) variant.
        let gemma4 = m.engine as? Gemma4Model
        #expect(gemma4 != nil, "expected Gemma4Model engine")
        #expect(gemma4?.ple != nil, "E2B is a PLE variant — Gemma4PLE must be present")

        // Per-layer cache geometry: sliding layers windowed, global
        // layers unbounded. E2B layer_types: every 5th layer (idx 4, 9,
        // 14, 19, 24, 29, 34) is full_attention ⇒ 7 global, 28 sliding.
        let caches = m.engine.makeLayerCaches()
        var slidingCount = 0
        var globalCount = 0
        for (i, c) in caches.enumerated() {
            guard let kv = c as? any KVCacheProtocol else { continue }
            let isGlobal = (i + 1) % 5 == 0
            switch kv.eviction {
            case .window:
                #expect(!isGlobal, "layer \(i): expected global (.unbounded)")
                slidingCount += 1
            case .unbounded:
                #expect(isGlobal, "layer \(i): expected sliding (.window)")
                globalCount += 1
            }
        }
        #expect(globalCount == 7)
        #expect(slidingCount == 28)

        // Greedy decode must produce coherent text. Two first-light bugs
        // were fixed: (1) a missing leading `<bos>` — Gemma 4 is
        // BOS-critical, but unlike Gemma 3 its `tokenizer.json`
        // post-processor's `single` template is bare (no `<bos>`), so
        // `Tokenizer.encode` returned none and the residual stream was
        // wrong from token 0; `Gemma4Model.requiresLeadingBOS` now
        // drives `Generate.encodePrompt` to prepend it. (2) a Gemma
        // 3-style `(1 + weight)` RMSNorm fold applied at load — Gemma 4
        // dropped that convention, so the fold doubled every norm scale.
        // Global-layer attention runs the d512 `Ops.sdpaDecode` GPU
        // kernel (no host readback).
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        print("Gemma 4 E2B decoded: \"\(prompt)\(result.text)\"")
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, minTokens: maxTokens,
                             label: "Gemma 4 E2B-it bf16")
    }

    @Test("Gemma4Dense (31B): load + greedy generate produces coherent output")
    func loadAndGenerateDense() async throws {
        // The smallest non-PLE dense Gemma 4 is 31B — only available as
        // a 4-/8-bit quantization. Guard-skip if not present locally.
        let modelId = "mlx-community/gemma-4-31b-it-4bit"
        let prompt = "Once upon a time, in a quiet village"

        // Guard-skip if the 31B checkpoint isn't cached locally — it is
        // large (4-bit ~17 GB) and not in the default test set. The
        // 5376-wide hidden state exercises the wide-row RMSNorm kernel
        // during the prewarm forward, so a successful load already
        // proves that path.
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Gemma 4 31B integration test skipped (checkpoint not available): \(error)")
            return
        }
        let gemma4 = m.engine as? Gemma4Model
        #expect(gemma4 != nil)
        #expect(gemma4?.ple == nil, "31B is a dense (non-PLE) variant")

        // Generation gated behind `FFAI_BUILD_MACHINE`: the 31B
        // checkpoint is large and greedy-decoding 48 tokens is slow for
        // a routine run. The 5376-wide hidden state routes through the
        // wide-row RMSNorm kernel (the old 4096-row cap is gone). The
        // E2B test above covers the same code paths unconditionally;
        // this is the dense-variant spot check.
        guard ProcessInfo.processInfo.environment["FFAI_BUILD_MACHINE"] != nil else {
            print("Gemma 4 31B generation check skipped " +
                  "(set FFAI_BUILD_MACHINE to run — large checkpoint).")
            return
        }
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, minTokens: 48,
                             label: "Gemma 4 31B-it 4bit")
    }

    @Test("Gemma4MoE (26B-A4B): load + greedy generate produces coherent output")
    func loadAndGenerateMoE() async throws {
        // 26B-A4B is the mixture-of-experts variant: every layer runs a
        // shared dense MLP and an 8-of-128 routed expert mixture in
        // parallel, each branch with its own pre/post norm, plus the
        // Gemma 4 router (input RMSNorm + per-expert scale). This test
        // uses the uniformly-8-bit checkpoint; the 4-bit checkpoint
        // mixes 4-/8-bit per layer, now handled by the per-tensor
        // `deriveAffineQuantBits` bit-width derivation.
        let modelId = "mlx-community/gemma-4-26b-a4b-it-8bit"
        let prompt = "The capital of France is"

        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Gemma 4 26B-A4B integration test skipped (checkpoint not available): \(error)")
            return
        }
        let gemma4 = m.engine as? Gemma4Model
        #expect(gemma4 != nil, "expected Gemma4Model engine")
        // 26B-A4B has hidden_size_per_layer_input = 0 → MoE, not PLE.
        #expect(gemma4?.ple == nil, "26B-A4B is the MoE (non-PLE) variant")

        // Generation gated behind `FFAI_BUILD_MACHINE`: the 26B
        // checkpoint is large (8-bit ~28 GB) and slow to greedy-decode.
        guard ProcessInfo.processInfo.environment["FFAI_BUILD_MACHINE"] != nil else {
            print("Gemma 4 26B-A4B generation coherence check skipped " +
                  "(set FFAI_BUILD_MACHINE to run).")
            return
        }
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, minTokens: 24,
                             label: "Gemma 4 26B-A4B-it 8bit")
    }
}
