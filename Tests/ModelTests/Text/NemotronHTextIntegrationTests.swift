// Slow integration test: downloads (or hits cache) the smallest
// FFAI-runnable NemotronH checkpoint and runs end-to-end greedy
// generation. Skipped automatically if the network or checkpoint isn't
// available.
//
// NemotronH is FFAI's first *stack-interleaved* hybrid — a
// `hybrid_override_pattern` string assigns each decoder layer one mixer
// kind (Mamba 2 "M", attention "*", dense squared-ReLU MLP "-"), and
// the kinds vary down the stack. This test exercises the heterogeneous
// `[any DecoderLayer]` decode loop, the per-index cache array
// (`Mamba2LayerCache` / `KVCache` / `StatelessLayerCache`), the
// grouped-B/C Mamba path (`n_groups = 8`), the gated mixer RMSNorm, and
// no-RoPE attention.
//
// ekryski/Nemotron-H-4B-Base-8K-4bit (~4B params, bf16) is the smallest
// published NemotronH checkpoint whose config the shipped FFAI path can
// run end-to-end: its Mamba layers have d_inner/n_groups = 896, a valid
// rmsNormRows row size, and the pattern carries no MoE ("E") layers.
// The MoE-bearing Cascade-2 / Nemotron-3 checkpoints are rejected at
// load (see NemotronH.swift) and are 30B+ regardless.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("NemotronH Text Integration", .serialized)
struct NemotronHTextIntegrationTests {

    @Test("load + greedy generate produces coherent stack-interleaved hybrid output")
    func loadAndGenerate() async throws {
        let modelId = "ekryski/Nemotron-H-4B-Base-8K-4bit"
        let prompt = "The history of the printing press began when"

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Engine should be NemotronH (not Llama / FalconH1 / Mamba 2).
        #expect(m.nemotronH != nil)
        #expect(m.falconH1 == nil)
        #expect(m.mamba2 == nil)

        // Shapes from the published Nemotron-H-4B-Base config.
        #expect(m.engine.hidden == 3072)
        #expect(m.engine.nLayers == 52)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)
        #expect(m.engine.vocab == 131_072)
        if let nh = m.nemotronH {
            // Mamba 2 mixer geometry — d_inner = mamba_num_heads * mamba_head_dim.
            #expect(nh.mambaNHeads == 112)
            #expect(nh.mambaHeadDim == 64)
            #expect(nh.dInner == 7168)
            #expect(nh.stateDim == 128)
            #expect(nh.nGroups == 8)
            #expect(nh.convKernel == 4)
            // conv_dim = d_inner + 2 * n_groups * state_dim = 7168 + 2048.
            #expect(nh.convDim == 9216)
            // Heterogeneous stack: 24 Mamba, 4 attention, 24 dense MLP.
            #expect(nh.layers.count == 52)
            let mambaCount = nh.layers.filter { $0 is NemotronHMambaLayer }.count
            let attnCount = nh.layers.filter { $0 is NemotronHAttentionLayer }.count
            let mlpCount = nh.layers.filter { $0 is NemotronHMLPLayer }.count
            #expect(mambaCount == 24)
            #expect(attnCount == 4)
            #expect(mlpCount == 24)
        }

        // ── Forward-shape smoke test ──────────────────────────────────
        // One token through the full heterogeneous stack. Logits should
        // be finite and non-degenerate (the model is trained).
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 52)
        // Cache kinds match the layer kinds, index-for-index.
        if let nh = m.nemotronH {
            for (i, layer) in nh.layers.enumerated() {
                switch layer {
                case is NemotronHMambaLayer:
                    #expect(caches[i] is Mamba2LayerCache)
                case is NemotronHAttentionLayer:
                    #expect(caches[i] is KVCache)
                case is NemotronHMLPLayer:
                    #expect(caches[i] is StatelessLayerCache)
                default:
                    Issue.record("unexpected layer kind at index \(i)")
                }
            }
        }
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 131_072)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        // Top logit strictly greater than the 5th — degenerate
        // (all-equal) logits indicate a forward-pass numerical bug.
        #expect(top[0].1 > top[4].1)

        // ── End-to-end greedy generation ──────────────────────────────
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)

        // Print the actual decoded text for manual inspection.
        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("NemotronH-4B decoded output: \(decoded)")

        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "NemotronH-4B Base bf16"
        )
    }
}
