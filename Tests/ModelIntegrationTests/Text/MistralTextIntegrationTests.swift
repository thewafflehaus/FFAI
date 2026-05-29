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
// Slow integration test for Mistral 7B Instruct. Mistral 7B uses the
// Llama 3 architecture verbatim, so the FFAI engine returned is a
// `LlamaModel`; this test just confirms that the dispatch path through
// the Mistral family enum loads the checkpoint and decodes coherent
// output.
//
// Skipped automatically if the checkpoint isn't available.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Mistral Integration", .serialized
)
struct MistralTextIntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        // mlx-community 4-bit pack — keeps the download manageable
        // while still exercising the production Mistral weight layout
        // + Llama-loader path.
        let modelId = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        // ── 1. Load ──────────────────────────────────────────────────────
        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // ── 2. Interfaces we expect ──────────────────────────────────────
        #expect(m.llama != nil, "Mistral 7B should load through the Llama engine")
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)
        // Mistral 7B canonical shapes. GQA: nKVHeads = 8.
        #expect(m.engine.hidden == 4096)
        #expect(m.engine.nLayers == 32)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)
        #expect(m.engine.vocab == 32_768)

        // ── 3. Errors we expect ─────────────────────────────────────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 32)
        for (i, c) in caches.enumerated() {
            if !(c is KVCache) {
                Issue.record("Mistral: layer \(i) cache is \(type(of: c)), expected KVCache")
            }
        }

        // ── 4. Forward pass produces expected output ────────────────────
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 32_768)
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
            label: "Mistral 7B 4bit"
        )
    }
}
