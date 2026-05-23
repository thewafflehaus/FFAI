// GenerationParametersTests — defaults, the with(_:) copy-mutator,
// per-family value table, and equality.

import Foundation
import Testing
@testable import FFAI

@Suite("GenerationParameters")
struct GenerationParametersTests {

    @Test("Init defaults match the documented baseline")
    func initDefaults() {
        let p = GenerationParameters()
        #expect(p.maxTokens == 256)
        #expect(p.stopOnEOS == true)
        #expect(p.extraStopTokens.isEmpty)
        // prefillStepSize is now Int? — nil means "use the engine's
        // tuned default" (Phase 6.6 chunked-prefill wiring). Generic
        // engines still resolve to 1024 inside Generate.driveGeneration.
        #expect(p.prefillStepSize == nil)
        #expect(p.temperature == 0.6)
        #expect(p.topP == 1.0)
        #expect(p.topK == 0)
        #expect(p.minP == 0.0)
        #expect(p.repetitionPenalty == 1.0)
        #expect(p.presencePenalty == 0.0)
        #expect(p.seed == nil)
    }

    @Test("Custom init sets every field")
    func customInit() {
        let p = GenerationParameters(
            maxTokens: 64, stopOnEOS: false, extraStopTokens: [42, 99],
            prefillStepSize: 4096, temperature: 0.0, topP: 0.5, topK: 40,
            minP: 0.05, repetitionPenalty: 1.1, presencePenalty: 0.5,
            seed: 12345
        )
        #expect(p.maxTokens == 64)
        #expect(p.stopOnEOS == false)
        #expect(p.extraStopTokens == [42, 99])
        #expect(p.prefillStepSize == 4096)
        #expect(p.temperature == 0.0)
        #expect(p.topP == 0.5)
        #expect(p.topK == 40)
        #expect(p.minP == 0.05)
        #expect(p.repetitionPenalty == 1.1)
        #expect(p.presencePenalty == 0.5)
        #expect(p.seed == 12345)
    }

    @Test("with(_:) is a copy-mutator")
    func withCopyMutator() {
        let base = GenerationParameters(maxTokens: 64, temperature: 0.6)
        let edited = base.with { $0.maxTokens = 128 }
        #expect(edited.maxTokens == 128)
        #expect(edited.temperature == base.temperature)
        #expect(base.maxTokens == 64)   // base immutable
    }

    @Test("Equatable")
    func equatable() {
        let a = GenerationParameters(maxTokens: 128)
        let b = GenerationParameters(maxTokens: 128)
        let c = GenerationParameters(maxTokens: 64)
        #expect(a == b)
        #expect(a != c)
    }

    @Test("Per-family defaults table")
    func familyDefaults() {
        // Llama 3.x dense
        let llama = LlamaDense.defaultGenerationParameters
        #expect(llama.temperature == 0.6)
        #expect(llama.topP == 1.0)
        #expect(llama.topK == 0)
        #expect(llama.repetitionPenalty == 1.0)
        #expect(llama.prefillStepSize == 1024)
        #expect(llama.maxTokens == 256)

        // Qwen 3 dense
        let qwen = Qwen3Dense.defaultGenerationParameters
        #expect(qwen.temperature == 0.6)
        #expect(qwen.topP == 0.95)
        #expect(qwen.topK == 20)
        #expect(qwen.minP == 0.0)
        #expect(qwen.repetitionPenalty == 1.0)
        #expect(qwen.prefillStepSize == 1024)
        #expect(qwen.maxTokens == 256)

        // Family defaults must differ.
        #expect(llama != qwen)
    }
}
