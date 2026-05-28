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
// Slow integration test for Qwen 2.5 0.5B Instruct. The 2.x and 2.5
// series share the same architecture; this test exercises the
// bias-bearing QKV projection path that distinguishes them from the
// 3-series + Mistral7B (which use no biases).
//
// 0.5B picked over larger sizes so the test downloads in seconds rather
// than minutes when not cached. Skipped automatically if the
// checkpoint isn't reachable.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Qwen2 Text Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct Qwen2TextIntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        // ── 1. Load ──────────────────────────────────────────────────────
        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // ── 2. Interfaces we expect ──────────────────────────────────────
        // Qwen 2.5 is Llama-shaped with bias-bearing QKV projections —
        // routes through the bias-aware loadLinear path of the Llama
        // engine (`m.llama`), NOT the bias-less 3-series engine.
        #expect(
            m.llama != nil,
            "Qwen 2.5 should load through the Llama engine after bias-aware Linear")
        #expect(m.qwen3 == nil)
        #expect(m.qwen35 == nil)
        // Qwen 2.5 0.5B canonical shapes. GQA: 2 KV heads.
        #expect(m.engine.hidden == 896)
        #expect(m.engine.nLayers == 24)
        #expect(m.engine.nHeads == 14)
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.headDim == 64)
        #expect(m.engine.vocab == 151_936)

        // ── 3. Errors we expect ─────────────────────────────────────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 24)
        for (i, c) in caches.enumerated() {
            if !(c is KVCache) {
                Issue.record("Qwen2.5: layer \(i) cache is \(type(of: c)), expected KVCache")
            }
        }

        // ── 4. Forward pass produces expected output ────────────────────
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 151_936)
        let top = Sampling.topN(logits, n: 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 > top[4].1)

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "Qwen 2.5 0.5B 4bit"
        )
    }
}
