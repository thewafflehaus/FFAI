// InternLM2Tests — root-file unit tests for `Sources/FFAI/Models/InternLM2.swift`.
//
// Offline. InternLM 2 is a Llama-shaped pass-through; the root file
// just declares dispatch metadata. These tests guard the model_type /
// architecture constants the registry sniffs.

import Foundation
import Testing
@testable import FFAI

@Suite("InternLM2 Family Root")
struct InternLM2RootTests {

    @Test("modelTypes advertises the internlm2 label")
    func modelTypes() {
        #expect(InternLM2.modelTypes.contains("internlm2"))
        #expect(!InternLM2.modelTypes.isEmpty)
    }

    @Test("architectures advertises InternLM2ForCausalLM")
    func architectures() {
        #expect(InternLM2.architectures.contains("InternLM2ForCausalLM"))
        #expect(!InternLM2.architectures.isEmpty)
    }
}
