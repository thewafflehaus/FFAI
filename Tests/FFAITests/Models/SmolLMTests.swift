// SmolLMTests — root-file unit tests for `Sources/FFAI/Models/SmolLM.swift`.
//
// Offline. SmolLM 1/2/3 are Llama-shaped pass-throughs; the root file
// just declares dispatch metadata. These tests guard that all three
// generations' model_type / architecture labels stay advertised.

import Foundation
import Testing
@testable import FFAI

@Suite("SmolLM Family Root")
struct SmolLMRootTests {

    @Test("modelTypes covers all three SmolLM generations")
    func modelTypes() {
        #expect(SmolLM.modelTypes.contains("smollm"))
        #expect(SmolLM.modelTypes.contains("smollm2"))
        #expect(SmolLM.modelTypes.contains("smollm3"))
    }

    @Test("architectures covers all three SmolLM generations")
    func architectures() {
        #expect(SmolLM.architectures.contains("SmolLMForCausalLM"))
        #expect(SmolLM.architectures.contains("SmolLM2ForCausalLM"))
        #expect(SmolLM.architectures.contains("SmolLM3ForCausalLM"))
    }
}
