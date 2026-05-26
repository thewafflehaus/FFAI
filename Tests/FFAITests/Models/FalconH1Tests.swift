// FalconH1Tests — root-file unit tests for `Sources/FFAI/Models/FalconH1.swift`.
//
// Offline. FalconH1 root declares family metadata + variant dispatch
// (only `FalconH1Hybrid` ships) + the `FalconH1Error` cases. Tests
// guard the registry surface + error message shape.

import Foundation
import Testing
@testable import FFAI

@Suite("FalconH1 Family Root")
struct FalconH1RootTests {

    @Test("modelTypes advertises falcon_h1")
    func modelTypes() {
        #expect(FalconH1.modelTypes.contains("falcon_h1"))
    }

    @Test("architectures advertises FalconH1ForCausalLM")
    func architectures() {
        #expect(FalconH1.architectures.contains("FalconH1ForCausalLM"))
    }

    @Test("variant(for:) returns FalconH1Hybrid")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "FalconH1ForCausalLM",
                              modelType: "falcon_h1", raw: [:])
        let v = try FalconH1.variant(for: cfg)
        #expect(String(describing: v) == String(describing: FalconH1Hybrid.self))
    }

    @Test("FalconH1Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(FalconH1Error.missingConfig("hidden").description.contains("hidden"))
        #expect(FalconH1Error.unsupportedConfig("bad").description.contains("bad"))
        #expect(FalconH1Error.missingConfig("x").description.contains("FalconH1"))
    }
}
