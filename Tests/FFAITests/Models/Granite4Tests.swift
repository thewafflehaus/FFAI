// Granite4Tests — root-file unit tests for `Sources/FFAI/Models/Granite4.swift`.
//
// Offline. Covers Granite 4 family metadata + variant dispatch
// (`Granite4Hybrid` is the only variant) + `Granite4Error`
// descriptions.

import Foundation
import Testing
@testable import FFAI

@Suite("Granite4 Family Root")
struct Granite4RootTests {

    @Test("modelTypes advertises granitemoehybrid")
    func modelTypes() {
        #expect(Granite4.modelTypes.contains("granitemoehybrid"))
    }

    @Test("architectures advertises Granite4ForCausalLM")
    func architectures() {
        #expect(Granite4.architectures.contains("Granite4ForCausalLM"))
    }

    @Test("variant(for:) returns Granite4Hybrid")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "Granite4ForCausalLM",
                              modelType: "granitemoehybrid", raw: [:])
        let v = try Granite4.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Granite4Hybrid.self))
    }

    @Test("Granite4Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(Granite4Error.missingConfig("layer_types").description
            .contains("layer_types"))
        #expect(Granite4Error.unsupportedConfig("bad").description.contains("bad"))
        #expect(Granite4Error.missingConfig("x").description.contains("Granite4"))
    }
}
