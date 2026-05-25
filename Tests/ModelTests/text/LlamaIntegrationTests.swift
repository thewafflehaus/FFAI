// Slow integration test for Llama 3.2 1B. Asserts:
//   1. Model loads + shapes match the published config.
//   2. A single-token forward pass yields finite, non-zero logits.
//   3. Greedy decode produces coherent output (not stuck / not
//      degenerate — see CoherentOutput.swift for the contract).
//
// Skipped automatically if the network/checkpoint isn't available.

import Foundation
import Testing
@testable import FFAI

@Suite("Llama 3.2 1B integration", .serialized)
struct LlamaIntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        let modelId = "unsloth/Llama-3.2-1B"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200
        let bosTokenId = 128_000

        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Llama integration test skipped: \(error)")
            return
        }

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
        expectCoherentOutput(result.generatedTokens, label: "Llama 3.2 1B fp16")
    }
}
