// Starcoder2Tests — root-file unit tests for `Sources/FFAI/Models/Starcoder2.swift`.
//
// Offline. Starcoder 2 is a Llama-shaped pass-through; the root file
// just declares dispatch metadata. These tests guard the model_type /
// architecture constants the registry sniffs.

import Foundation
import Testing
@testable import FFAI

@Suite("Starcoder2 Family Root")
struct Starcoder2RootTests {

    @Test("modelTypes advertises the starcoder2 label")
    func modelTypes() {
        #expect(Starcoder2.modelTypes.contains("starcoder2"))
        #expect(!Starcoder2.modelTypes.isEmpty)
    }

    @Test("architectures advertises Starcoder2ForCausalLM")
    func architectures() {
        #expect(Starcoder2.architectures.contains("Starcoder2ForCausalLM"))
        #expect(!Starcoder2.architectures.isEmpty)
    }
}
