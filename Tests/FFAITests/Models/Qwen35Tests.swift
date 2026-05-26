// Qwen35Tests — root-file unit tests for `Sources/FFAI/Models/Qwen35.swift`.
//
// Offline. Covers Qwen 3.5 family metadata + variant dispatch
// (`Qwen35Hybrid` is the only variant — dense-vs-MoE is decided inside
// the loader from `num_experts`) + every `Qwen35Error` case.
//
// VL surface lives at `Tests/FFAITests/Models/Vision/Qwen35VisionConfigTests.swift`.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen35 Family Root")
struct Qwen35RootTests {

    @Test("modelTypes covers all four qwen3_5 variants")
    func modelTypes() {
        #expect(Qwen35.modelTypes.contains("qwen3_5"))
        #expect(Qwen35.modelTypes.contains("qwen3_5_text"))
        #expect(Qwen35.modelTypes.contains("qwen3_5_moe"))
        #expect(Qwen35.modelTypes.contains("qwen3_5_moe_text"))
    }

    @Test("architectures covers ForCausalLM + ForConditionalGeneration for dense and MoE")
    func architectures() {
        #expect(Qwen35.architectures.contains("Qwen3_5ForCausalLM"))
        #expect(Qwen35.architectures.contains("Qwen3_5ForConditionalGeneration"))
        #expect(Qwen35.architectures.contains("Qwen3_5MoeForCausalLM"))
        #expect(Qwen35.architectures.contains("Qwen3_5MoeForConditionalGeneration"))
    }

    @Test("variant(for:) returns Qwen35Hybrid (dense/MoE decided at load time)")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "Qwen3_5ForCausalLM",
                              modelType: "qwen3_5", raw: [:])
        let v = try Qwen35.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Qwen35Hybrid.self))
    }

    @Test("Qwen35Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(Qwen35Error.missingConfig("hidden_size").description
            .contains("hidden_size"))
        #expect(Qwen35Error.missingConfig("x").description.contains("Qwen3.5"))
        #expect(Qwen35Error.unsupportedConfig("bad").description.contains("bad"))
    }
}
