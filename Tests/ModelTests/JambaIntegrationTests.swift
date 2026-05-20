// Slow integration test: downloads (or hits cache) the smallest
// FFAI-runnable Jamba checkpoint and runs end-to-end greedy generation.
// Skipped automatically if the network or checkpoint isn't available.
//
// Jamba (AI21) is a stack-interleaved hybrid — a `layers_block_type`
// schedule (here derived from attn_layer_period / attn_layer_offset)
// assigns each decoder layer one mixer kind: a Mamba 1 selective SSM
// ("mamba") or multi-head attention ("attention"). Every layer also
// carries a feed-forward half: a dense SwiGLU MLP (`num_experts == 1`)
// or a block-sparse MoE block.
//
// This test exercises the heterogeneous `[any DecoderLayer]` decode
// loop, the per-index cache array (`JambaMambaLayerCache` / `KVCache`),
// the host-side Mamba 1 selective scan (the shipped Mamba 2 `ssm_step`
// kernel cannot express Mamba 1's per-(channel, state) decay — see
// Jamba.swift), the 2-D `A_log` handling, and no-RoPE attention.
//
// mlx-community/AI21-Jamba-Reasoning-3B-bf16 (~3B params, bf16) is the
// smallest published Jamba checkpoint that runs end-to-end without
// quantized-MoE expert slicing: 28 layers (26 Mamba + 2 attention),
// num_experts = 1 → a dense SwiGLU FFN on every layer. The 4bit / 6bit
// conversions are quantized; the MoE feed-forward path is implemented
// (and unit-covered via MoELayerTests) but no small raw MoE Jamba
// checkpoint exists to integration-test it here.

import Foundation
import Testing
@testable import FFAI

@Suite("Jamba integration", .serialized)
struct JambaIntegrationTests {

    @Test("load + greedy generate produces coherent hybrid output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/AI21-Jamba-Reasoning-3B-bf16"
        let prompt = "The history of the printing press began when"

        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Jamba integration test skipped: \(error)")
            return
        }

        // Engine should be Jamba (not Llama / FalconH1 / NemotronH /
        // GraniteMoeHybrid / Mamba 2).
        #expect(m.jamba != nil)
        #expect(m.falconH1 == nil)
        #expect(m.nemotronH == nil)
        #expect(m.graniteMoeHybrid == nil)
        #expect(m.mamba2 == nil)

        // Shapes from the published AI21-Jamba-Reasoning-3B config.
        #expect(m.engine.hidden == 2560)
        #expect(m.engine.nLayers == 28)
        #expect(m.engine.nHeads == 20)
        #expect(m.engine.nKVHeads == 1)
        #expect(m.engine.vocab == 65_536)
        if let j = m.jamba {
            // Mamba 1 mixer geometry — d_inner = mamba_expand * hidden.
            #expect(j.dInner == 5120)
            #expect(j.dState == 16)
            #expect(j.dtRank == 160)
            #expect(j.convKernel == 4)
            // Heterogeneous stack: attn_layer_period=14, offset=7 →
            // attention at layers 7 and 21, Mamba elsewhere.
            #expect(j.layers.count == 28)
            let mambaCount = j.layers.filter { $0 is JambaMambaLayer }.count
            let attnCount = j.layers.filter { $0 is JambaAttentionLayer }.count
            #expect(mambaCount == 26)
            #expect(attnCount == 2)
            // num_experts = 1 → dense SwiGLU FFN, no MoE.
            #expect(j.hasMoE == false)
        }

        // ── Forward-shape smoke test ──────────────────────────────────
        // One token through the full heterogeneous stack. Logits should
        // be finite and non-degenerate (the model is trained).
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 28)
        // Cache kinds match the layer kinds, index-for-index.
        if let j = m.jamba {
            for (i, layer) in j.layers.enumerated() {
                switch layer {
                case is JambaMambaLayer:
                    #expect(caches[i] is JambaMambaLayerCache)
                case is JambaAttentionLayer:
                    #expect(caches[i] is KVCache)
                default:
                    Issue.record("unexpected layer kind at index \(i)")
                }
            }
        }
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 65_536)
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
        print("Jamba-3B decoded output: \(decoded)")

        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "AI21-Jamba-Reasoning-3B bf16"
        )
    }
}
