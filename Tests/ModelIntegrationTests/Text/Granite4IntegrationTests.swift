// Copyright 2026 Eric Kryski (@ekryski)
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
// Slow integration test: downloads (or hits cache) the smallest
// FFAI-runnable Granite4 checkpoint and runs end-to-end greedy
// generation. Skipped automatically if the network or checkpoint isn't
// available.
//
// Granite4 (IBM Granite 4.0 "-H") is a stack-interleaved hybrid
// — a `layer_types` array assigns each decoder layer one mixer kind
// (Mamba 2 "mamba", attention "attention"), and the FFN half of every
// layer is either an MoE block (+ shared expert) or a dense SwiGLU MLP.
// This test exercises the heterogeneous `[any DecoderLayer]` decode
// loop, the per-index cache array (`Mamba2LayerCache` / `KVCache`), the
// no-RoPE attention path, and the four Granite scalar multipliers
// (embedding / residual / attention / logits scaling).
//
// mlx-community/granite-4.0-h-350m-4bit (~350M params, bf16) is the
// smallest published Granite4 checkpoint: 32 layers, 28 Mamba +
// 4 attention, num_local_experts = 0 (a dense SwiGLU FFN). It exercises
// the full hybrid stack end-to-end. The 64-expert MoE checkpoints
// (H-Tiny / H-Small) are 7B+ and ship only quantized on mlx-community;
// the MoE feed-forward path is implemented and unit-covered via
// `MoELayerTests`, but no small raw MoE checkpoint exists to integration
// test it here.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Granite4 Integration", .serialized)
struct Granite4IntegrationTests {

    @Test("load + greedy generate produces coherent hybrid output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/granite-4.0-h-350m-4bit"
        let prompt = "The history of the printing press began when"

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Engine should be Granite4 (not Llama / FalconH1 /
        // NemotronH / Mamba 2).
        #expect(m.graniteMoeHybrid != nil)
        #expect(m.falconH1 == nil)
        #expect(m.nemotronH == nil)
        #expect(m.mamba2 == nil)

        // Shapes from the published granite-4.0-h-350m config.
        #expect(m.engine.hidden == 768)
        #expect(m.engine.nLayers == 32)
        #expect(m.engine.nHeads == 12)
        #expect(m.engine.nKVHeads == 4)
        #expect(m.engine.headDim == 64)
        #expect(m.engine.vocab == 100_352)
        if let g = m.graniteMoeHybrid {
            // Mamba 2 mixer geometry — d_inner = mamba_n_heads * mamba_d_head.
            #expect(g.mambaNHeads == 48)
            #expect(g.mambaHeadDim == 32)
            #expect(g.dInner == 1536)
            #expect(g.stateDim == 128)
            #expect(g.nGroups == 1)
            #expect(g.convKernel == 4)
            // conv_dim = d_inner + 2 * n_groups * state_dim = 1536 + 256.
            #expect(g.convDim == 1792)
            // Heterogeneous stack: 28 Mamba + 4 attention.
            #expect(g.layers.count == 32)
            let mambaCount = g.layers.filter {
                ($0 as? Granite4Layer)?.kind == .mamba
            }.count
            let attnCount = g.layers.filter {
                ($0 as? Granite4Layer)?.kind == .attention
            }.count
            #expect(mambaCount == 28)
            #expect(attnCount == 4)
            // H-350M is dense (num_local_experts = 0): no MoE layer.
            #expect(g.hasMoE == false)
        }

        // ── Forward-shape smoke test ──────────────────────────────────
        // One token through the full heterogeneous stack. Logits should
        // be finite and non-degenerate (the model is trained).
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 32)
        // Cache kinds match the layer kinds, index-for-index.
        if let g = m.graniteMoeHybrid {
            for (i, layer) in g.layers.enumerated() {
                switch (layer as? Granite4Layer)?.kind {
                case .mamba:
                    #expect(caches[i] is Mamba2LayerCache)
                case .attention:
                    #expect(caches[i] is KVCache)
                case .none:
                    Issue.record("unexpected layer kind at index \(i)")
                }
            }
        }
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 100_352)
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
        print("Granite4-350M decoded output: \(decoded)")

        // Granite4-350M is a tiny hybrid (Mamba 2 + MoE + attn)
        // and at 200 tokens / temperature=0 it bottoms out around 18%
        // unique-token ratio — coherent for the first ~60 tokens, then a
        // "the analysis center" style cycle. Same small-model quality
        // ceiling as Jamba 3B; the run-length floor still catches real
        // empty-kernel / stuck-argmax regressions.
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            minUniqueRatio: 0.12,
            label: "Granite4-350M H bf16"
        )
    }
}
