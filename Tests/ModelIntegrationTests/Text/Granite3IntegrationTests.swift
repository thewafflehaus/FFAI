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
// Granite 3 family integration coverage — IBM's Granite v3 dense text
// models (granite-3.0, granite-3.1, granite-3.2). Llama-3-shaped
// weights routed through `LlamaDense`.
//
// Granite 4 (granite-4.0-h, GraniteMoeHybrid) is a different
// architecture (Mamba 2 / attention / MoE hybrid) and has its own
// integration test under `Granite4IntegrationTests.swift`.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Granite3 Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct Granite3IntegrationTests {

    @Test("Granite-3.2-2B-Instruct (GraniteForCausalLM) decodes coherently")
    func granite3() async throws {
        // ── 1. Load ──────────────────────────────────────────────────────
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/IBM-granite-3.2-2b-instruct-4bit")
        }

        // ── 2. Interfaces we expect ──────────────────────────────────────
        // Granite 3 is Llama-shaped — engine routes through `m.llama`.
        // Granite 4 (hybrid) is a different engine and stays nil here.
        #expect(m.llama != nil)
        #expect(m.graniteMoeHybrid == nil)
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)

        // Granite 3.2 2B canonical (same dim as 3.0/3.1 — only the
        // training data + alignment changed). head_dim is derived as
        // hidden / num_attention_heads (config field is absent).
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 40)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 64)  // 2048 / 32
        #expect(m.engine.vocab == 49155)

        // ── 3. Errors we expect ─────────────────────────────────────────
        // Cache-kind alignment — every Llama layer expects a KVCache.
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 40)
        for (i, c) in caches.enumerated() {
            if !(c is KVCache) {
                Issue.record("Granite3: layer \(i) cache is \(type(of: c)), expected KVCache")
            }
        }

        // ── 4. Forward pass produces expected output ────────────────────
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 49155)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        // Strict ordering — all-equal logits indicate a forward-pass bug.
        #expect(top[0].1 > top[4].1)

        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "Granite 3.2 2B 4bit")
    }
}
