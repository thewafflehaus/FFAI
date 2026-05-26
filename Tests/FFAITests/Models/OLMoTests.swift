// OLMoTests — root-file unit tests for `Sources/FFAI/Models/OLMo.swift`.
//
// Offline. OLMo 1 / OLMo 2 are Llama-shaped pass-throughs; the root
// file just declares dispatch metadata. These tests guard the
// model_type / architecture constants for both generations.

import Foundation
import Testing
@testable import FFAI

@Suite("OLMo Family Root")
struct OLMoRootTests {

    @Test("modelTypes covers both olmo and olmo2")
    func modelTypes() {
        #expect(OLMo.modelTypes.contains("olmo"))
        #expect(OLMo.modelTypes.contains("olmo2"))
    }

    @Test("architectures covers OlmoForCausalLM and Olmo2ForCausalLM")
    func architectures() {
        #expect(OLMo.architectures.contains("OlmoForCausalLM"))
        #expect(OLMo.architectures.contains("Olmo2ForCausalLM"))
    }
}
