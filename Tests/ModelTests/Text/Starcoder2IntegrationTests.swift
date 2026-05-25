// Starcoder 2 family integration coverage — BigCode's Llama-shaped
// code dense decoder. Attention biases ride the same loadLinear
// auto-detection path as Qwen 2.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Starcoder2 Integration", .serialized)
struct Starcoder2IntegrationTests {

    @Test("Starcoder2-3B (Starcoder2ForCausalLM, attention biases) decodes coherently")
    func starcoder2() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/starcoder2-3b-4bit")
        }
        // Starcoder2 3B canonical: hidden=3072, nLayers=30, nHeads=24,
        // nKVHeads=2, headDim=128. The attention biases pass through
        // loadLinear's auto-detection — same path as Qwen 2.
        #expect(m.engine.hidden == 3072)
        #expect(m.engine.nLayers == 30)
        #expect(m.engine.headDim == 128)
        let result = try await m.generate(
            prompt: "def fibonacci(n):\n",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "Starcoder2 3B 4bit")
    }
}
