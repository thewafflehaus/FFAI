// Slow integration test: downloads (or hits cache) the smallest
// FalconH1 checkpoint and runs end-to-end greedy generation. Skipped
// automatically if the network or checkpoint isn't available.
//
// FalconH1 is the first Phase 5e *hybrid* model — every decoder layer
// runs BOTH a Mamba 2 SSM mixer AND a grouped-query attention path on
// the same normalized input. This test exercises the full hybrid
// decode loop: the `DecoderLayer` protocol stack, the dual-cache
// `FalconH1LayerCache` (Mamba state + KV), and the scalar-multiplier
// weight folding.
//
// Falcon-H1-Tiny-90M-Instruct (~173MB bf16) is the smallest published
// FalconH1 family checkpoint, so the integration suite stays fast. The
// architecture (FalconH1Hybrid) is identical in shape across the
// 0.5B / 1.5B / 3B / 7B variants — those are drop-in swaps.

import Foundation
import Testing
@testable import FFAI

@Suite("FalconH1 Tiny-90M integration", .serialized)
struct FalconH1IntegrationTests {

    @Test("load + greedy generate produces coherent hybrid output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/Falcon-H1-Tiny-90M-Instruct-bf16"
        let prompt = "Once upon a time, in a quiet village"

        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("FalconH1 integration test skipped: \(error)")
            return
        }

        // Engine should be FalconH1 (not Llama / Qwen3 / Mamba 2).
        #expect(m.falconH1 != nil)
        #expect(m.llama == nil)
        #expect(m.qwen3 == nil)
        #expect(m.mamba2 == nil)

        // Shapes from the published Tiny-90M config.
        #expect(m.engine.hidden == 512)
        #expect(m.engine.nLayers == 24)
        #expect(m.engine.nHeads == 8)
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.headDim == 64)
        #expect(m.engine.vocab == 32_768)
        if let fh1 = m.falconH1 {
            // Mamba 2 mixer geometry — d_ssm = mamba_n_heads * mamba_d_head.
            #expect(fh1.dSSM == 768)
            #expect(fh1.mambaNHeads == 24)
            #expect(fh1.mambaHeadDim == 32)
            #expect(fh1.stateDim == 64)
            #expect(fh1.convKernel == 4)
            // conv_dim = d_ssm + 2 * n_groups * state_dim = 768 + 128 = 896
            #expect(fh1.convDim == 896)
            // Every layer is a parallel-hybrid FalconH1DecoderLayer.
            #expect(fh1.layers.count == 24)
            #expect(fh1.layers.allSatisfy { $0 is FalconH1DecoderLayer })
        }

        // ── Forward-shape smoke test ──────────────────────────────────
        // One BOS-style token through the full hybrid stack. Logits
        // should be finite and non-degenerate (the model is trained).
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 24)
        #expect(caches.allSatisfy { $0 is FalconH1LayerCache })
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 32_768)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        // Top logit strictly greater than the 5th — degenerate
        // (all-equal) logits indicate a forward-pass numerical bug.
        #expect(top[0].1 > top[4].1)

        // ── End-to-end greedy generation ──────────────────────────────
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 64, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)

        // Print the actual decoded text for manual inspection.
        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("FalconH1 Tiny-90M decoded output: \(decoded)")

        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "FalconH1 Tiny-90M bf16"
        )
    }
}
