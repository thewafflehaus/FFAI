// Mamba2Tests — root-file unit tests for `Sources/FFAI/Models/Mamba2.swift`.
//
// Offline. Covers the family enum's metadata + variant dispatch and
// each `Mamba2Error` case's description. The concrete `Mamba2Dense`
// + SSM layer impl live under `Models/Text/Mamba2Text.swift` and are
// exercised by the integration suite.

import Foundation
import Testing
@testable import FFAI

@Suite("Mamba2 Family Root")
struct Mamba2RootTests {

    @Test("modelTypes advertises the mamba2 label")
    func modelTypes() {
        #expect(Mamba2.modelTypes.contains("mamba2"))
    }

    @Test("architectures advertises Mamba2ForCausalLM")
    func architectures() {
        #expect(Mamba2.architectures.contains("Mamba2ForCausalLM"))
    }

    @Test("variant(for:) returns Mamba2Dense (the only variant today)")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "Mamba2ForCausalLM",
                              modelType: "mamba2", raw: [:])
        let v = try Mamba2.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Mamba2Dense.self))
    }

    @Test("Mamba2Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(Mamba2Error.missingConfig("hidden_size").description
            .contains("hidden_size"))
        #expect(Mamba2Error.unsupportedConfig("bad").description.contains("bad"))
    }
}
