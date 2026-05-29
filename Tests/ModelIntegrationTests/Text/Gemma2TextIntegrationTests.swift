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
// Slow integration test for Gemma 2 2B-it. Verifies the family file
// loads from a real checkpoint, instantiates a per-layer KV cache with
// the right sliding / unbounded split (sliding_window_pattern=2 means
// alternating), and decodes coherent text through the 26-layer stack.
//
// Uses the mlx-community 2B-it bf16 conversion (smallest published
// Gemma 2 variant). Skipped automatically if the checkpoint isn't
// reachable — the bf16 conversion is ~5 GB so the test downloads in a
// few minutes on first run and is cached after.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Gemma2 Integration", .serialized
)
struct Gemma2TextIntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/gemma-2-2b-it-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        // ── 1. Load ──────────────────────────────────────────────────────
        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // ── 2. Interfaces we expect ──────────────────────────────────────
        #expect(m.gemma2 != nil, "Gemma 2 should load through the Gemma2 family")
        #expect(m.llama == nil)
        #expect(m.qwen3 == nil)
        // Gemma 2 2B canonical shapes (from the published config).
        #expect(m.engine.hidden == 2304)
        #expect(m.engine.nLayers == 26)
        #expect(m.engine.nHeads == 8)
        #expect(m.engine.nKVHeads == 4)
        #expect(m.engine.headDim == 256)
        #expect(m.engine.vocab == 256_000)

        // Sliding-window alternation: odd layers (0, 2, 4, …) are sliding;
        // even layers (1, 3, 5, …) are full attention. Spot-check both.
        if let g2 = m.gemma2 {
            #expect(g2.slidingWindow == 4096)
            #expect(g2.slidingWindowPattern == 2)
            #expect(g2.layers[0].isSliding == true)
            #expect(g2.layers[1].isSliding == false)
            #expect(g2.layers[2].isSliding == true)
        }

        // ── 3. Errors we expect ─────────────────────────────────────────
        // Cache shape — Gemma 2 uses a mix of full + sliding KVCaches.
        // Both subtypes conform to `LayerCacheProtocol`; we just assert
        // the count matches nLayers (a mis-sized cache array is a
        // forward-time crash, so check up-front).
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 26)

        // ── 4. Forward pass produces expected output ────────────────────
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 256_000)
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
            label: "Gemma 2 2B-it 4bit"
        )
        let text = m.tokenizer.decode(
            tokens: result.generatedTokens,
            skipSpecialTokens: true)
        print("Gemma 2 2B-it generated: \(text)")
    }
}
