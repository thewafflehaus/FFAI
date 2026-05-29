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
// Starcoder 2 family integration coverage — BigCode's code dense
// decoder. Structurally distinct from the Llama dense family in three
// ways that the Llama loader can't accommodate:
//   - LayerNorm with `.bias` (NOT RMSNorm) for input_layernorm /
//     post_attention_layernorm / model.norm
//   - Single-projection GELU-tanh MLP with `c_fc` (up) + `c_proj`
//     (down) names — NOT the SwiGLU triad
//   - `norm_epsilon` config field (NOT `rms_norm_eps`)
//
// Routes through the dedicated `Starcoder2.variant(for:)` →
// `Starcoder2Dense.loadModel` path; the loader landed in
// `Sources/FFAI/Models/Text/Starcoder2Text.swift`. Prior to that
// dedicated loader, the test surfaced a misleading
// `Llama: required config field missing` error because Starcoder2
// was misrouted through `llamaCompatibleArchs` in
// `Loader/Model.swift`.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("Starcoder2 Integration", .serialized)
struct Starcoder2TextIntegrationTests {

    @Test("Starcoder2-3B (Starcoder2ForCausalLM, LayerNorm + GELU MLP) decodes coherently")
    func starcoder2() async throws {
        let modelId = "mlx-community/starcoder2-3b-4bit"
        let prompt = "def fibonacci(n):\n"

        // ── 1. Load ──────────────────────────────────────────────────────
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(modelId)
        }

        // ── 2. Interfaces we expect ──────────────────────────────────────
        // Engine should be Starcoder2 (NOT Llama — Starcoder2 was
        // previously misrouted through the Llama loader and this
        // accessor would have returned nil, with `m.llama` returning
        // a broken wrapper that threw at first config read).
        #expect(m.starcoder2 != nil, "Starcoder2-3B should load through the Starcoder2 engine")
        #expect(m.llama == nil)
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)

        // Starcoder2 3B canonical shapes (from the published config):
        //   hidden = 3072, nLayers = 30, nHeads = 24, nKVHeads = 2
        //   (12:1 GQA), head_dim = 128 (derived: 3072/24), vocab =
        //   49_152, intermediate = 12_288 (= 4 · hidden), tied embed.
        #expect(m.engine.hidden == 3072)
        #expect(m.engine.nLayers == 30)
        #expect(m.engine.nHeads == 24)
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.headDim == 128)
        #expect(m.engine.vocab == 49_152)

        // ── 3. Errors we expect ─────────────────────────────────────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 30)
        for (i, c) in caches.enumerated() {
            if !(c is KVCache) {
                Issue.record("Starcoder2: layer \(i) cache is \(type(of: c)), expected KVCache")
            }
        }

        // ── 4. Forward pass produces expected output ────────────────────
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 49_152)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        // Strict ordering — all-equal logits would indicate a forward
        // bug (e.g. mis-loaded LayerNorm bias, wrong GELU formula,
        // c_fc/c_proj swap).
        #expect(top[0].1 > top[4].1)

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)

        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("Starcoder2-3B decoded output: \(decoded)")

        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "Starcoder2 3B 4bit"
        )
    }
}
