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
// OLMo family integration coverage — Allen AI's Olmo / Olmo 2 Llama-
// shaped dense decoder. We exercise OLMo 2 (the largest / most-cited
// release) so loader refactors can't silently break the
// architecture-string routing or forward path.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "OLMo Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct OLMoTextIntegrationTests {

    @Test("OLMo-2-1124-7B-Instruct (Olmo2ForCausalLM) decodes coherently")
    func olmo2() async throws {
        // ── 1. Load ──────────────────────────────────────────────────────
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/OLMo-2-1124-7B-Instruct-4bit")
        }

        // ── 2. Interfaces we expect ──────────────────────────────────────
        // OLMo 2 is post-norm + q/k-norm — its own decoder, NOT the Llama
        // path, so `m.olmo2` is set and `m.llama` is nil.
        #expect(m.olmo2 != nil)
        #expect(m.llama == nil)
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)
        #expect(m.falconH1 == nil)

        // OLMo 2 7B canonical: hidden=4096, nLayers=32, nHeads=32,
        // nKVHeads=32 (MHA, not GQA), head_dim=128 (4096/32),
        // vocab=100_352.
        #expect(m.engine.hidden == 4096)
        #expect(m.engine.nLayers == 32)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 32)
        #expect(m.engine.headDim == 128)
        #expect(m.engine.vocab == 100_352)

        // ── 3. Errors we expect ─────────────────────────────────────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 32)
        for (i, c) in caches.enumerated() {
            if !(c is KVCache) {
                Issue.record("OLMo2: layer \(i) cache is \(type(of: c)), expected KVCache")
            }
        }

        // ── 4. Forward pass produces expected output ────────────────────
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 100_352)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 > top[4].1)

        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "OLMo 2 7B 4bit"
        )
    }
}
