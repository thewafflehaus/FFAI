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
// FFAI-runnable Qwen3.6 dense checkpoint and runs end-to-end greedy
// generation. Mirrors Qwen35TextIntegrationTests.swift — Qwen3.6
// ships under the same `qwen3_5*` model_type strings and reuses the
// Qwen3.5 GDN-hybrid engine wholesale (`m.qwen35`), so the test
// shape is identical to its Qwen3.5 sibling and only the published
// per-checkpoint shapes differ.
//
// mlx-community/Qwen3.6-27B-4bit (~27B params, 4-bit affine quant)
// is the smallest published Qwen3.6 dense checkpoint: 64 layers
// (48 GDN + 16 attention), num_experts absent → a dense SwiGLU FFN
// on every layer. GDN dims are (Dk,Dv,Hk,Hv) = (128,128,24,24) —
// see the family root for the Qwen3.5 ↔ 3.6 dispatch.
//
// The MoE Qwen3.6-35B-A3B variant lives in
// `Qwen36MoETextIntegrationTests.swift` (separate file so the two
// large checkpoints don't try to load in the same test).

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Qwen3.6 Text Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct Qwen36TextIntegrationTests {

    @Test("dense GDN hybrid: load + greedy generate produces coherent output")
    func loadAndGenerateDense() async throws {
        let modelId = "mlx-community/Qwen3.6-27B-4bit"
        let prompt = "The history of the printing press began when"

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Engine should be Qwen3.6 (routes through the same Qwen3.5
        // dense GDN-hybrid engine; every other family slot stays nil).
        #expect(m.qwen35 != nil)
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)
        #expect(m.falconH1 == nil)
        #expect(m.nemotronH == nil)
        #expect(m.graniteMoeHybrid == nil)

        // Shapes from the published Qwen3.6-27B-4bit `text_config`.
        #expect(m.engine.hidden == 5120)
        #expect(m.engine.nLayers == 64)
        #expect(m.engine.nHeads == 24)
        #expect(m.engine.nKVHeads == 4)
        #expect(m.engine.vocab == 248_320)
        if let q = m.qwen35 {
            // GDN mixer geometry — Qwen3.6 inherits the same head_dim 256
            // and full_attention_interval = 4 as Qwen3.5.
            #expect(q.convKernel == 4)
            #expect(q.headDim == 256)
            // Heterogeneous stack: full_attention_interval = 4 → every
            // 4th layer is attention (16 of 64).
            #expect(q.layers.count == 64)
            let gdnCount = q.layers.filter { $0 is Qwen35GDNLayer }.count
            let attnCount = q.layers.filter { $0 is Qwen35AttentionLayer }.count
            #expect(gdnCount == 48)
            #expect(attnCount == 16)
            // num_experts absent → dense SwiGLU FFN, no MoE.
            #expect(q.hasMoE == false)
        }

        // ── Cache-kind alignment ──────────────────────────────────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 64)
        if let q = m.qwen35 {
            for (i, layer) in q.layers.enumerated() {
                switch layer {
                case is Qwen35GDNLayer:
                    #expect(caches[i] is GDNLayerCache)
                case is Qwen35AttentionLayer:
                    #expect(caches[i] is KVCache)
                default:
                    Issue.record("unexpected layer kind at index \(i)")
                }
            }
        }

        // ── Forward-shape smoke test ──────────────────────────────────
        // One token through the full heterogeneous stack. Logits should
        // be finite and non-degenerate (the model is trained).
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

        // Print the actual decoded text for manual inspection.
        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("Qwen3.6-27B decoded output: \(decoded)")

        // KNOWN FAILURE (2026-05-27): the 27B-4bit checkpoint currently
        // emits token 0 ("!") every step. Same Qwen3.5 engine produces
        // coherent output on Qwen3.5-0.8B-4bit, so the regression is
        // 27B-scale-specific (likely in the loader's handling of one
        // of the 27B-specific config fields — wider hidden=5120,
        // intermediate=17408, or 6:1 GQA). Suite stays enabled so the
        // failure is visible in every bisect run rather than
        // disappearing behind a skip-reason.
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "Qwen3.6-27B 4bit dense GDN hybrid"
        )
    }
}
