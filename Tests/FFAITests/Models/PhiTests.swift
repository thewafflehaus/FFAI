// PhiTests — root-file unit tests for `Sources/FFAI/Models/Phi.swift`.
//
// Offline. Covers Phi-3 family metadata + variant dispatch + every
// `PhiError` case.

import Foundation
import Testing
@testable import FFAI

@Suite("Phi Family Root")
struct PhiRootTests {

    @Test("modelTypes advertises phi3")
    func modelTypes() {
        #expect(Phi.modelTypes.contains("phi3"))
    }

    @Test("architectures advertises Phi3ForCausalLM")
    func architectures() {
        #expect(Phi.architectures.contains("Phi3ForCausalLM"))
    }

    @Test("variant(for:) returns Phi3Dense")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "Phi3ForCausalLM",
                              modelType: "phi3", raw: [:])
        let v = try Phi.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Phi3Dense.self))
    }

    @Test("PhiError stringifies every case")
    func errorDescriptions() {
        #expect(PhiError.missingConfig.description.contains("Phi"))
        #expect(PhiError.unsupportedRopeScaling("yarn").description.contains("yarn"))
        #expect(PhiError.quantizedFusedNotSupported.description.contains("quantized"))
    }
}
