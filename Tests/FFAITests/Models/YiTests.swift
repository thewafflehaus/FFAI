// YiTests — root-file unit tests for `Sources/FFAI/Models/Yi.swift`.
//
// Offline. Yi is a Llama-shaped pass-through; the root file just
// declares dispatch metadata. These tests guard the model_type /
// architecture constants the registry sniffs.

import Foundation
import Testing
@testable import FFAI

@Suite("Yi Family Root")
struct YiRootTests {

    @Test("modelTypes advertises the yi label")
    func modelTypes() {
        #expect(Yi.modelTypes.contains("yi"))
        #expect(!Yi.modelTypes.isEmpty)
    }

    @Test("architectures advertises YiForCausalLM")
    func architectures() {
        #expect(Yi.architectures.contains("YiForCausalLM"))
        #expect(!Yi.architectures.isEmpty)
    }
}
