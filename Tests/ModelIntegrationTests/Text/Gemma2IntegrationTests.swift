// Slow integration test for Gemma 2 2B-it. Verifies the family file
// loads from a real checkpoint, instantiates a per-layer KV cache with
// the right sliding / unbounded split (sliding_window_pattern=2 means
// alternating), and decodes coherent text through the 26-layer stack.
//
// Uses the mlx-community 2B-it bf16 conversion (smallest published
// Gemma 2 variant). Skipped automatically if the checkpoint isn't
// reachable — the bf16 conversion is ~5 GB so the test downloads in a
// few minutes on first run and is cached after.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Gemma2 Integration", .serialized)
struct Gemma2IntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/gemma-2-2b-it-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Gemma 2 2B canonical shapes (from the published config).
        #expect(m.engine.hidden == 2304)
        #expect(m.engine.nLayers == 26)
        #expect(m.engine.nHeads == 8)
        #expect(m.engine.nKVHeads == 4)
        #expect(m.engine.headDim == 256)
        #expect(m.gemma2 != nil, "Gemma 2 should load through the Gemma2 family")

        // Sliding-window alternation: odd layers (0, 2, 4, …) are sliding;
        // even layers (1, 3, 5, …) are full attention. Spot-check both.
        if let g2 = m.gemma2 {
            #expect(g2.slidingWindow == 4096)
            #expect(g2.slidingWindowPattern == 2)
            #expect(g2.layers[0].isSliding == true)
            #expect(g2.layers[1].isSliding == false)
            #expect(g2.layers[2].isSliding == true)
        }

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "Gemma 2 2B-it")
        let text = m.tokenizer.decode(tokens: result.generatedTokens,
                                       skipSpecialTokens: true)
        print("Gemma 2 2B-it generated: \(text)")
    }
}
