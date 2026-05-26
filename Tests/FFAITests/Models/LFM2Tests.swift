// LFM2Tests — root-file unit tests for `Sources/FFAI/Models/LFM2.swift`.
//
// Offline. The LFM2 text-family + layer-schedule + expert-bias routing
// is covered in `Tests/FFAITests/Models/Text/LFM2TextTests.swift`; this
// file focuses on `isMoE(_)` dispatch, `LFM2Error` + `LFM2VLError` shape,
// and the `LFM2VL` orchestrator's family constants.

import Foundation
import Testing
@testable import FFAI

@Suite("LFM2 Family Root — isMoE dispatch")
struct LFM2RootDispatchTests {

    @Test("isMoE recognises model_type == lfm2_moe")
    func moeByModelType() {
        let cfg = ModelConfig(architecture: nil, modelType: "lfm2_moe", raw: [:])
        #expect(LFM2.isMoE(cfg))
    }

    @Test("isMoE recognises architecture == Lfm2MoeForCausalLM")
    func moeByArchitecture() {
        let cfg = ModelConfig(architecture: "Lfm2MoeForCausalLM",
                              modelType: "lfm2", raw: [:])
        #expect(LFM2.isMoE(cfg))
    }

    @Test("isMoE is false for the dense LFM2 model_type")
    func denseConfig() {
        let cfg = ModelConfig(architecture: "Lfm2ForCausalLM",
                              modelType: "lfm2", raw: [:])
        #expect(!LFM2.isMoE(cfg))
    }

    @Test("variant(for:) returns LFM2MoE for MoE configs")
    func variantMoE() throws {
        let cfg = ModelConfig(architecture: nil, modelType: "lfm2_moe", raw: [:])
        let v = try LFM2.variant(for: cfg)
        #expect(String(describing: v) == String(describing: LFM2MoE.self))
    }

    @Test("variant(for:) returns LFM2Dense for dense configs")
    func variantDense() throws {
        let cfg = ModelConfig(architecture: "Lfm2ForCausalLM",
                              modelType: "lfm2", raw: [:])
        let v = try LFM2.variant(for: cfg)
        #expect(String(describing: v) == String(describing: LFM2Dense.self))
    }

    @Test("LFM2Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(LFM2Error.missingConfig("hidden").description.contains("hidden"))
        #expect(LFM2Error.unsupportedConfig("bad").description.contains("bad"))
        #expect(LFM2Error.missingConfig("x").description.contains("LFM2"))
    }
}

// ─── LFM2VL orchestrator constants ───────────────────────────────────

@Suite("LFM2VL Orchestrator — registration + error")
struct LFM2VLRootTests {

    @Test("LFM2VL advertises the Lfm2Vl architecture")
    func registration() {
        #expect(LFM2VL.architectures.contains("Lfm2VlForConditionalGeneration"))
    }

    @Test("LFM2VL exposes the canonical image_token_index default")
    func imageTokenId() {
        #expect(LFM2VL.defaultImageTokenId == 396)
    }

    @Test("LFM2VLError stringifies every case with its payload")
    func errorDescriptions() {
        #expect(LFM2VLError.missingConfig.description.contains("LFM2VL"))
        #expect(LFM2VLError.missingTensor("foo").description.contains("foo"))
        #expect(LFM2VLError.unsupportedConfig("bad").description.contains("bad"))
    }
}
