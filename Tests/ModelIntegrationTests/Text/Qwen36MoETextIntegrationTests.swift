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
// FFAI-runnable Qwen3.6 MoE checkpoint and runs end-to-end greedy
// generation. Mirrors `Qwen35MoETextIntegrationTests` exactly —
// Qwen3.6 ships under the same `qwen3_5*` model_type strings and
// reuses the Qwen3.5 GDN-hybrid engine wholesale (`m.qwen35`), so
// only the published per-checkpoint shapes differ from the 3.5 MoE
// sibling.
//
// mlx-community/Qwen3.6-35B-A3B-4bit (~35B params, 4-bit affine
// quant, ~3B active per token) is the smallest published Qwen3.6
// MoE conversion: 40 layers (30 GDN + 10 attention), 256 experts
// top-8, moe_intermediate=512, with an always-on shared expert
// (shared_intermediate=512). Same shape as the 3.5 MoE sibling —
// only the training run differs.
//
// Dense + MoE are in separate files (see `Qwen36TextIntegrationTests`
// for dense) so the two large checkpoints don't try to load in the
// same test.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Qwen3.6 MoE Text Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct Qwen36MoETextIntegrationTests {

    @Test("MoE GDN hybrid: load + greedy generate produces coherent output")
    func loadAndGenerateMoE() async throws {
        let modelId = "mlx-community/Qwen3.6-35B-A3B-4bit"
        let prompt = "The history of the printing press began when"

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Engine should be Qwen3.6 MoE (routes through the same Qwen3.5
        // dense GDN-hybrid engine; per-checkpoint dense-vs-MoE is
        // decided from `num_experts` inside `loadModel`).
        #expect(m.qwen35 != nil)
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)
        #expect(m.falconH1 == nil)
        #expect(m.nemotronH == nil)
        #expect(m.graniteMoeHybrid == nil)

        // Shapes from the published Qwen3.6-35B-A3B-4bit `text_config`.
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 40)
        #expect(m.engine.nHeads == 16)
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.vocab == 248_320)
        if let q = m.qwen35 {
            // GDN mixer geometry — Qwen3.6 inherits head_dim 256 and
            // full_attention_interval = 4 from Qwen3.5.
            #expect(q.convKernel == 4)
            #expect(q.headDim == 256)
            // Heterogeneous stack: full_attention_interval = 4 → every
            // 4th layer is attention (10 of 40).
            #expect(q.layers.count == 40)
            let gdnCount = q.layers.filter { $0 is Qwen35GDNLayer }.count
            let attnCount = q.layers.filter { $0 is Qwen35AttentionLayer }.count
            #expect(gdnCount == 30)
            #expect(attnCount == 10)
            // num_experts = 256 → MoE feed-forward + always-on shared.
            #expect(q.hasMoE == true)
        }

        // ── Cache-kind alignment ──────────────────────────────────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 40)
        if let q = m.qwen35 {
            for (i, layer) in q.layers.enumerated() {
                switch layer {
                case is Qwen35GDNLayer:
                    #expect(caches[i] is Qwen35GDNLayerCache)
                case is Qwen35AttentionLayer:
                    #expect(caches[i] is KVCache)
                default:
                    Issue.record("unexpected layer kind at index \(i)")
                }
            }
        }

        // ── Forward-shape smoke test ──────────────────────────────────
        // One token through the full heterogeneous MoE stack. Logits
        // should be finite and non-degenerate (the model is trained).
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 248_320)
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
        print("Qwen3.6-35B-A3B decoded output: \(decoded)")

        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "Qwen3.6-35B-A3B 4bit MoE GDN hybrid"
        )
    }
}
