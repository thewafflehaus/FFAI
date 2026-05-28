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
// MiniCPM5-1B — OpenBMB's 1B base text model. Declares
// `architectures: ["LlamaForCausalLM"]` + `model_type: "llama"`, so it
// routes through FFAI's existing Llama dense loader with no family-
// specific code. Pinned target is the mlx affine int4 packing
// (`openbmb/MiniCPM5-1B-MLX`) — exercises the QuantizedLinear /
// QuantizedEmbedding load path on a Llama-shaped backbone with the
// head_dim=128 SDPA kernel Llama 3.2 / Qwen 3 use.
//
// Canonical shape (from the published config): hidden=1536, nLayers=24,
// nHeads=16, nKVHeads=2 (8:1 GQA), head_dim=128, vocab=130_560,
// intermediate=4608, rope_theta=5e6, max_position=131_072. Untied
// embed (the MLX repack keeps the explicit `lm_head.weight`).

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "MiniCPM5 Text Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct MiniCPM5TextIntegrationTests {

    @Test("MiniCPM5-1B-MLX (int4, Llama dispatch) decodes coherently")
    func loadAndGenerate() async throws {
        let modelId = "openbmb/MiniCPM5-1B-MLX"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        // ── 1. Load ──────────────────────────────────────────────────────
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(modelId)
        }

        // ── 2. Interfaces we expect ──────────────────────────────────────
        // MiniCPM5 declares LlamaForCausalLM + model_type "llama" so it
        // routes through `m.llama`. Every other family slot stays nil.
        #expect(m.llama != nil, "MiniCPM5-1B should load through the Llama engine")
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)
        #expect(m.falconH1 == nil)
        #expect(m.starcoder2 == nil)

        // Shapes from the published MiniCPM5-1B-MLX config.
        #expect(m.engine.hidden == 1536)
        #expect(m.engine.nLayers == 24)
        #expect(m.engine.nHeads == 16)
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.headDim == 128)
        #expect(m.engine.vocab == 130_560)

        // ── 3. Errors we expect ─────────────────────────────────────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 24)
        for (i, c) in caches.enumerated() {
            if !(c is KVCache) {
                Issue.record("MiniCPM5: layer \(i) cache is \(type(of: c)), expected KVCache")
            }
        }

        // ── 4. Forward pass produces expected output ────────────────────
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 130_560)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 > top[4].1)

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)

        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("MiniCPM5-1B-MLX decoded output: \(decoded)")

        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "MiniCPM5-1B int4"
        )
    }
}
