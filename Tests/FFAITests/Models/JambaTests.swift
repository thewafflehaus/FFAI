// JambaTests — root-file unit tests for `Sources/FFAI/Models/Jamba.swift`.
//
// Offline. Covers Jamba family enum metadata + variant dispatch
// (`JambaHybrid` is the only variant) + `JambaError` cases.

import Foundation
import Testing
@testable import FFAI

@Suite("Jamba Family Root")
struct JambaRootTests {

    @Test("modelTypes advertises jamba")
    func modelTypes() {
        #expect(Jamba.modelTypes.contains("jamba"))
    }

    @Test("architectures advertises JambaForCausalLM")
    func architectures() {
        #expect(Jamba.architectures.contains("JambaForCausalLM"))
    }

    @Test("variant(for:) returns JambaHybrid")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "JambaForCausalLM",
                              modelType: "jamba", raw: [:])
        let v = try Jamba.variant(for: cfg)
        #expect(String(describing: v) == String(describing: JambaHybrid.self))
    }

    @Test("JambaError stringifies every case with its payload")
    func errorDescriptions() {
        #expect(JambaError.missingConfig("hidden_size").description
            .contains("hidden_size"))
        #expect(JambaError.unsupportedConfig("bad").description.contains("bad"))
        #expect(JambaError.missingConfig("x").description.contains("Jamba"))
    }
}
