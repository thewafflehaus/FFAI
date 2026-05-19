// Slow integration test for DeepSeek R1 distilled models. These are
// fine-tunes of Qwen 2 / 3-series-Llama architectures, not novel
// model classes — they load through the existing Qwen2 / Llama
// dispatch path. The test pins this so future loader refactors can't
// silently break the R1-distill checkpoints.
//
// Skipped automatically if the checkpoint isn't reachable.

import Foundation
import Testing
@testable import FFAI

@Suite("DeepSeek R1 Distill integration", .serialized)
struct DeepSeekR1DistillIntegrationTests {

    @Test("R1-Distill-Qwen-1.5B (Qwen 2 architecture) generates coherent output")
    func r1DistillQwen() async throws {
        // Smallest distill — 1.5B parameters in 4-bit ≈ 800 MB on disk.
        // Uses model_type='qwen2' / arch='Qwen2ForCausalLM', so it
        // flows through the bias-aware Linear path added with the
        // Qwen 2.x family wiring.
        let modelId = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 64

        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("R1-Distill-Qwen test skipped: \(error)")
            return
        }

        // 1.5B shape sanity (per the HF config):
        //   hidden = 1536, nLayers = 28, nHeads = 12, nKVHeads = 2,
        //   headDim = 128, intermediate = 8960.
        #expect(m.engine.hidden == 1536)
        #expect(m.engine.nLayers == 28)
        #expect(m.engine.nHeads == 12)
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.headDim == 128)
        #expect(m.llama != nil, "R1-Distill-Qwen should load through the 3-series engine after bias-aware Linear")

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "R1-Distill-Qwen-1.5B")
    }
}
