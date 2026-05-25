// OLMo family integration coverage — Allen AI's Olmo / Olmo 2 Llama-
// shaped dense decoder. We exercise OLMo 2 (the largest / most-cited
// release) so loader refactors can't silently break the
// architecture-string routing or forward path.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("OLMo Integration", .serialized)
struct OLMoIntegrationTests {

    @Test("OLMo-2-1124-7B-Instruct (Olmo2ForCausalLM) decodes coherently")
    func olmo2() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/OLMo-2-1124-7B-Instruct-4bit")
        }
        // OLMo 2 7B canonical: hidden=4096, headDim=128.
        #expect(m.engine.hidden == 4096)
        #expect(m.engine.headDim == 128)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "OLMo 2 7B 4bit")
    }
}
