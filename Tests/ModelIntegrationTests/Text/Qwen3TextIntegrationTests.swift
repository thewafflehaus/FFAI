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
// Slow integration test: downloads Qwen3 1.7B bf16 and asserts the
// model loads + the engine produces coherent text via greedy decode.
//
// 1.7B (vs 4B) keeps the integration suite fast — same architecture,
// smaller weights. Skipped automatically if checkpoint isn't available.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("Qwen3 Text Integration", .serialized)
struct Qwen3TextIntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/Qwen3-1.7B-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Engine should be Qwen3 (not Llama).
        #expect(m.qwen3 != nil)
        #expect(m.llama == nil)

        // Shapes match the published config (Qwen3 1.7B).
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 28)
        #expect(m.engine.nHeads == 16)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)
        #expect(m.engine.vocab == 151_936)

        // First-token forward: finite logits.
        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)

        // Greedy decode of the prompt — exercises Qwen3's q_norm/k_norm
        // path through the full per-token forward.
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "Qwen3 1.7B bf16")
    }
}
