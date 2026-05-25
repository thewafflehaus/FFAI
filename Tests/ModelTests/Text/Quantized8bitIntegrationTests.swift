// End-to-end test: download mlx-community/Qwen3-1.7B-8bit and assert
// FFAI's greedy decode produces coherent output. Exercises the
// dequant_gemv_int8 kernel end-to-end on Qwen3 1.7B.
//
// Skipped if network/checkpoint isn't available.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Qwen3 8-bit Integration", .serialized)
struct Quantized8bitIntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/Qwen3-1.7B-8bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        #expect(m.config.quantization?.bits == 8)
        #expect(m.config.quantization?.groupSize == 64)
        #expect(m.qwen3 != nil)

        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 28)
        #expect(m.engine.nHeads == 16)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)

        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "Qwen3 1.7B 8-bit")
    }
}
