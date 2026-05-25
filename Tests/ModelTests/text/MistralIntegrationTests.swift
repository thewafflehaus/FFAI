// Slow integration test for Mistral 7B Instruct. Mistral 7B uses the
// Llama 3 architecture verbatim, so the FFAI engine returned is a
// `LlamaModel`; this test just confirms that the dispatch path through
// the Mistral family enum loads the checkpoint and decodes coherent
// output.
//
// Skipped automatically if the checkpoint isn't available.

import Foundation
import Testing
@testable import FFAI

@Suite("Mistral 7B integration", .serialized)
struct MistralIntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        // mlx-community 4-bit pack — keeps the download manageable
        // while still exercising the production Mistral weight layout
        // + Llama-loader path.
        let modelId = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 64

        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Mistral integration test skipped: \(error)")
            return
        }

        // Mistral 7B canonical shapes.
        #expect(m.engine.hidden == 4096)
        #expect(m.engine.nLayers == 32)
        #expect(m.engine.nHeads == 32)
        // GQA: nKVHeads = 8.
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)
        #expect(m.llama != nil, "Mistral 7B should load through the Llama engine")

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "Mistral 7B 4-bit")
    }
}
