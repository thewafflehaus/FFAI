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
// Llama 3 integration coverage — Meta's Llama 3.x dense decoder. The
// pinned target is the smallest published quantized variant
// (`mlx-community/Llama-3.2-1B-Instruct-4bit`); shape assertions land
// on its published config: hidden=2048, nLayers=16, nHeads=32,
// nKVHeads=8 (4:1 GQA), head_dim=64, vocab=128_256.
//
// File is named `Llama3TextIntegrationTests` (vs the family root's
// `Llama` namespace) because the Llama 2 and Llama 4 lines have their
// own scheduled coverage:
//   - Llama 2: `Llama2TextIntegrationTests` (planned — see
//     `planning/planned-model-support.md`)
//   - Llama 4: `Llama4TextIntegrationTests` (multi-modal MoE; planned)
// Each gets its own suite so a regression specific to one Llama major
// doesn't hide behind a shared shape-assertion list.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Llama3 Text Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct Llama3TextIntegrationTests {

    @Test("Llama-3.2-1B-Instruct (LlamaForCausalLM, 4-bit) decodes coherently")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200
        let bosTokenId = 128_000

        // ── 1. Load ──────────────────────────────────────────────────────
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(modelId)
        }

        // ── 2. Interfaces we expect ──────────────────────────────────────
        #expect(m.llama != nil, "Llama 3.2 1B should load through the Llama engine")
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)
        #expect(m.starcoder2 == nil)
        // Sanity: shapes match the published Llama 3.2 1B config.
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 16)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 64)
        #expect(m.engine.vocab == 128_256)

        // ── 3. Errors we expect ─────────────────────────────────────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 16)
        for (i, c) in caches.enumerated() {
            if !(c is KVCache) {
                Issue.record("Llama 3.2: layer \(i) cache is \(type(of: c)), expected KVCache")
            }
        }

        // ── 4. Forward pass produces expected output ────────────────────
        // Single-token forward: BOS → finite, non-zero, ordered logits.
        let logits = m.engine.forward(tokenId: bosTokenId, position: 0, caches: caches)
        #expect(logits.elementCount == 128_256)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 != 0)
        #expect(top[0].1 > top[4].1)

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "Llama 3.2 1B 4bit"
        )
    }
}
