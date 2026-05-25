// Slow integration test for Qwen 2.5 0.5B Instruct. The 2.x and 2.5
// series share the same architecture; this test exercises the
// bias-bearing QKV projection path that distinguishes them from the
// 3-series + Mistral7B (which use no biases).
//
// 0.5B picked over larger sizes so the test downloads in seconds rather
// than minutes when not cached. Skipped automatically if the
// checkpoint isn't reachable.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen 2.5 0.5B integration", .serialized)
struct Qwen2IntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Qwen 2.5 0.5B canonical shapes.
        #expect(m.engine.hidden == 896)
        #expect(m.engine.nLayers == 24)
        #expect(m.engine.nHeads == 14)
        // GQA: 2 KV heads.
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.headDim == 64)
        #expect(m.llama != nil, "Qwen 2.5 should load through the 3-series engine after bias-aware Linear")

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "Qwen 2.5 0.5B")
    }
}
