// Slow integration test: downloads (or hits cache) the smallest
// published LFM2 checkpoint and runs end-to-end greedy generation.
// Skipped automatically if the network or checkpoint isn't available.
//
// LFM2 (LiquidAI's Liquid Foundation Models 2 — and LFM2.5, which is
// architecturally identical) is a stack-interleaved hybrid: a
// `layer_types` / `full_attn_idxs` schedule assigns each decoder layer
// one mixer kind — `conv` (LFM2's double-gated short convolution) or
// `full_attention` (GQA + RoPE, with a host-side per-head Q/K RMSNorm).
// Every layer carries a feed-forward half: a dense SwiGLU MLP (`lfm2`)
// or a block-sparse MoE block (`lfm2_moe`). This test exercises the
// heterogeneous `[any DecoderLayer]` decode loop, the per-index cache
// array (`LFM2ConvCache` / `KVCache`), the gated short-conv mixer over
// the shipped `conv1d_causal_step` kernel, and the Granite4
// command-buffer discipline.
//
// LiquidAI/LFM2-350M (~350M params, bf16) is the smallest published
// LFM2 checkpoint.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("LFM2 Text Integration", .serialized)
struct LFM2TextIntegrationTests {

    @Test("load + greedy generate produces coherent hybrid output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/LFM2-350M-4bit"
        let prompt = "The history of the printing press began when"

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Engine should be LFM2 (not Llama / a hybrid family).
        #expect(m.lfm2 != nil)
        #expect(m.graniteMoeHybrid == nil)
        #expect(m.falconH1 == nil)

        // head_dim 64, GQA, hybrid conv/attention stack.
        #expect(m.engine.headDim == 64)
        #expect(m.engine.nLayers > 0)
        #expect(m.engine.nKVHeads <= m.engine.nHeads)

        if let l = m.lfm2 {
            // Stack is a genuine mix of conv and attention layers.
            let convCount = l.layers.filter {
                ($0 as? LFM2Layer)?.kind == .conv
            }.count
            let attnCount = l.layers.filter {
                ($0 as? LFM2Layer)?.kind == .attention
            }.count
            #expect(convCount > 0)
            #expect(attnCount > 0)
            #expect(convCount + attnCount == l.layers.count)
            // LFM2 is conv-heavy — attention layers are the sparse subset.
            #expect(convCount > attnCount)
        }

        // ── Cache kinds match the layer kinds, index-for-index ────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == m.engine.nLayers)
        if let l = m.lfm2 {
            for (i, layer) in l.layers.enumerated() {
                switch (layer as? LFM2Layer)?.kind {
                case .conv:
                    #expect(caches[i] is LFM2ConvCache)
                case .attention:
                    #expect(caches[i] is KVCache)
                case .none:
                    Issue.record("unexpected layer kind at index \(i)")
                }
            }
        }

        // ── Forward-shape smoke test ──────────────────────────────────
        // One token through the full heterogeneous stack. Logits should
        // be finite and non-degenerate (the model is trained).
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == m.engine.vocab)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 > top[4].1)

        // ── End-to-end greedy generation ──────────────────────────────
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)

        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("LFM2-350M decoded output: \(decoded)")

        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "LFM2-350M bf16"
        )
    }

    // LFM2-MoE (`Lfm2MoeForCausalLM`) — the LFM2 conv/attention backbone
    // with a block-sparse MoE feed-forward on every layer at index ≥
    // `num_dense_layers`. `LiquidAI/LFM2-8B-A1B` (~8B params, bf16) is
    // the smallest published LFM2-MoE checkpoint — large, so this test
    // skips unless it is already cached locally.
    @Test("LFM2-MoE load + greedy generate produces coherent output")
    func loadAndGenerateMoE() async throws {
        let modelId = "mlx-community/LFM2-8B-A1B-4bit"
        let prompt = "The history of the printing press began when"

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        #expect(m.lfm2 != nil)
        #expect(m.engine.headDim == 64)

        if let l = m.lfm2 {
            // The MoE checkpoint carries at least one MoE feed-forward.
            #expect(l.hasMoE)
            let moeCount = l.layers.filter {
                ($0 as? LFM2Layer)?.isMoELayer == true
            }.count
            #expect(moeCount > 0)
            // First `num_dense_layers` layers keep a dense FFN.
            #expect(moeCount < l.layers.count)
        }

        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == m.engine.vocab)
        let top = Sampling.topN(logits, n: 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 > top[4].1)

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("LFM2-8B-A1B decoded output: \(decoded)")

        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "LFM2-8B-A1B MoE bf16"
        )
    }
}
