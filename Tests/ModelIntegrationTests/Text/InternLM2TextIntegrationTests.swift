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
// InternLM 2 family integration coverage — Shanghai AI Lab's
// InternLM v2 Llama-shaped dense decoder. Some checkpoints use a
// fused `wqkv` projection that `loadLinear` handles transparently.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "InternLM2 Integration", .serialized
)
struct InternLM2TextIntegrationTests {

    @Test("InternLM2.5-7B-Chat (InternLM2ForCausalLM) decodes coherently")
    func internLM2() async throws {
        // ── 1. Load ──────────────────────────────────────────────────────
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/internlm2_5-7b-chat-4bit")
        }

        // ── 2. Interfaces we expect ──────────────────────────────────────
        // InternLM 2 is Llama-shaped — routes through `m.llama`.
        #expect(m.llama != nil)
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)
        #expect(m.falconH1 == nil)

        // InternLM 2.5 7B canonical: hidden=4096, nLayers=32, nHeads=32,
        // nKVHeads=8 (4:1 GQA), head_dim=128 (4096/32), vocab=92_544.
        #expect(m.engine.hidden == 4096)
        #expect(m.engine.nLayers == 32)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)
        #expect(m.engine.vocab == 92_544)

        // ── 3. Errors we expect ─────────────────────────────────────────
        // Cache-kind alignment + fused-`wqkv` projection split happens
        // inside `loadLinear`'s auto-detection on load; if that ever
        // regressed we'd see a load-time throw rather than a forward-time
        // shape mismatch. The forward smoke test below is the
        // belt-and-braces check.
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 32)
        for (i, c) in caches.enumerated() {
            if !(c is KVCache) {
                Issue.record("InternLM2: layer \(i) cache is \(type(of: c)), expected KVCache")
            }
        }

        // ── 4. Forward pass produces expected output ────────────────────
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 92_544)
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
            label: "InternLM2.5 7B 4bit"
        )
    }
}
