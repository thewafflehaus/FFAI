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
// Slow integration test for Llama 3.2 1B. Asserts:
//   1. Model loads + shapes match the published config.
//   2. A single-token forward pass yields finite, non-zero logits.
//   3. Greedy decode produces coherent output (not stuck / not
//      degenerate — see CoherentOutput.swift for the contract).
//
// Skipped automatically if the network/checkpoint isn't available.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("Llama Integration", .serialized)
struct LlamaIntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200
        let bosTokenId = 128_000

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Sanity: shapes match the published config.
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 16)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 64)
        #expect(m.engine.vocab == 128_256)
        #expect(m.llama != nil, "expected engine to be a LlamaModel")

        // Single-token forward: BOS → finite, non-zero logits.
        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: bosTokenId, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 != 0)

        // Greedy decode of the prompt → coherent output.
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "Llama 3.2 1B 4bit")
    }
}
