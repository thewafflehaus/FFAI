// Gemma4Tests — root-file unit tests for `Sources/FFAI/Models/Gemma4.swift`.
//
// Offline. Covers the family enum's metadata, the three-arm
// `variant(for:)` dispatch (Dense vs E vs MoE), the default
// GenerationParameters declared inline at the root, and each
// `Gemma4Error` case's description shape.

import Foundation
import Testing
@testable import FFAI

@Suite("Gemma4 Family Root")
struct Gemma4RootTests {

    @Test("modelTypes covers gemma4 and gemma4_text")
    func modelTypes() {
        #expect(Gemma4.modelTypes.contains("gemma4"))
        #expect(Gemma4.modelTypes.contains("gemma4_text"))
    }

    @Test("architectures covers ForCausalLM + ForConditionalGeneration")
    func architectures() {
        #expect(Gemma4.architectures.contains("Gemma4ForCausalLM"))
        #expect(Gemma4.architectures.contains("Gemma4TextForCausalLM"))
        #expect(Gemma4.architectures.contains("Gemma4ForConditionalGeneration"))
    }

    @Test("variant(for:) returns Gemma4Dense by default (no MoE / no PLE)")
    func variantDense() throws {
        let cfg = ModelConfig(architecture: "Gemma4ForCausalLM",
                              modelType: "gemma4", raw: [:])
        let v = try Gemma4.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Gemma4Dense.self))
    }

    @Test("variant(for:) returns Gemma4E when hidden_size_per_layer_input > 0")
    func variantE() throws {
        let cfg = ModelConfig(
            architecture: "Gemma4ForCausalLM", modelType: "gemma4",
            raw: ["hidden_size_per_layer_input": 256])
        let v = try Gemma4.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Gemma4E.self))
    }

    @Test("variant(for:) returns Gemma4MoE when enable_moe_block == true")
    func variantMoE() throws {
        let cfg = ModelConfig(
            architecture: "Gemma4ForCausalLM", modelType: "gemma4",
            raw: ["enable_moe_block": true])
        let v = try Gemma4.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Gemma4MoE.self))
    }

    @Test("variant(for:) reads nested text_config — MoE wins over PLE")
    func variantNestedTextConfig() throws {
        let tc: [String: Any] = [
            "enable_moe_block": true,
            "hidden_size_per_layer_input": 256,
        ]
        let cfg = ModelConfig(
            architecture: "Gemma4ForConditionalGeneration", modelType: "gemma4",
            raw: ["text_config": tc])
        let v = try Gemma4.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Gemma4MoE.self))
    }

    @Test("Default GenerationParameters from the root extension are sane")
    func defaultGenParams() {
        // The default ships at the root via the protocol extension — pick
        // an arbitrary concrete variant; they all inherit the same value.
        let gp = Gemma4Dense.defaultGenerationParameters
        #expect(gp.maxTokens > 0)
        #expect(gp.temperature >= 0)
        #expect(gp.prefillStepSize == 4096)   // documented family optimum
    }

    @Test("Gemma4Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(Gemma4Error.missingConfig("text_config").description
            .contains("text_config"))
        #expect(Gemma4Error.missingTensor("model.embed_tokens.weight").description
            .contains("model.embed_tokens.weight"))
        #expect(Gemma4Error.unsupportedHeadDim(99).description.contains("99"))
        #expect(Gemma4Error.unalignedNorm(193).description.contains("193"))
    }
}
