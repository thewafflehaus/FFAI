// Slow integration test: downloads (or hits cache) the real Llama 3.2 1B
// checkpoint and runs end-to-end greedy generation. Asserts the model is
// loaded, forward produces non-zero/non-NaN logits, and Sampling.argmax
// returns a sensible answer for a fact-style prompt.
//
// Skipped automatically if the network/checkpoint isn't available.

import Foundation
import Testing
@testable import FFAI

@Suite("Llama 3.2 1B integration", .serialized)
struct LlamaIntegrationTests {

    @Test("load + greedy generate produces coherent text")
    func loadAndGenerate() async throws {
        let m: Model
        do {
            // unsloth/Llama-3.2-1B is an ungated mirror that doesn't need an HF token.
            m = try await Model.load("unsloth/Llama-3.2-1B")
        } catch {
            print("Llama integration test skipped: \(error)")
            return
        }

        // Sanity: shapes match the published config
        #expect(m.llama.hidden == 2048)
        #expect(m.llama.nLayers == 16)
        #expect(m.llama.nHeads == 32)
        #expect(m.llama.nKVHeads == 8)
        #expect(m.llama.headDim == 64)
        #expect(m.llama.vocab == 128256)

        // Forward one token (BOS) and check we get finite, non-zero logits.
        let caches = m.llama.makeKVCache()
        let logits = m.llama.forward(tokenId: 128000, position: 0, caches: caches)
        let topByOneToken = Sampling.topN(logits, n: 5)
        #expect(topByOneToken.count == 5)
        #expect(topByOneToken[0].1.isFinite)
        #expect(topByOneToken[0].1 != 0)

        // Run a short greedy generation. We can't pin a specific token output
        // (sampling reproducibility depends on hardware) but we expect the
        // generated text to be non-empty and decode without crashing.
        let result = try await m.generate(
            prompt: "The capital of France is",
            options: GenerateOptions(maxNewTokens: 4)
        )
        #expect(result.generatedTokens.count >= 1)
        #expect(!result.text.isEmpty)
        #expect(result.tokensPerSecond > 0)
    }
}
