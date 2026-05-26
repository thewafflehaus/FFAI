// Qwen2TextTests — unit coverage for `Sources/FFAI/Models/Text/Qwen2Text.swift`.
//
// Offline. Qwen 2 / 2.5 dense reuses the Llama dense engine (loadLinear
// auto-detects the QKV projection biases the Qwen 2 line carries). The
// file ships only the family enum dispatch entry — no concrete
// variant, layer, or error type. Tests guard the family-metadata
// surface and the `variant(for:)` contract: every Qwen 2 / 2.5 config
// resolves to `LlamaDense`.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen2 Text Family Routing")
struct Qwen2TextTests {

    @Test("Qwen2.modelTypes advertises qwen2 (covers 2.x + 2.5)")
    func modelTypes() {
        #expect(Qwen2.modelTypes.contains("qwen2"))
    }

    @Test("Qwen2.architectures advertises Qwen2ForCausalLM")
    func architectures() {
        #expect(Qwen2.architectures.contains("Qwen2ForCausalLM"))
    }

    @Test("Qwen2.variant(for:) returns LlamaDense for any qwen2 config")
    func routesThroughLlamaDense() throws {
        let cfg = ModelConfig(architecture: "Qwen2ForCausalLM",
                              modelType: "qwen2", raw: [:])
        let v = try Qwen2.variant(for: cfg)
        #expect(String(describing: v) == String(describing: LlamaDense.self))
    }
}
