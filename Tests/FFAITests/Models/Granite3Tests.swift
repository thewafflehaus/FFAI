// Granite3Tests — root-file unit tests for `Sources/FFAI/Models/Granite3.swift`.
//
// Offline. Granite v3 is a Llama-shaped pass-through; the root file
// only declares dispatch metadata (no variant protocol, no error
// type). These tests guard those constants so the registry sniff
// stays anchored.

import Foundation
import Testing
@testable import FFAI

@Suite("Granite3 Family Root")
struct Granite3RootTests {

    @Test("modelTypes advertises the granite label")
    func modelTypes() {
        #expect(Granite3.modelTypes.contains("granite"))
        #expect(!Granite3.modelTypes.isEmpty)
    }

    @Test("architectures advertises GraniteForCausalLM")
    func architectures() {
        #expect(Granite3.architectures.contains("GraniteForCausalLM"))
        #expect(!Granite3.architectures.isEmpty)
    }
}
