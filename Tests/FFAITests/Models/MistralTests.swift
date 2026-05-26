// MistralTests — root-file unit tests for `Sources/FFAI/Models/Mistral.swift`.
//
// Offline. Mistral 7B / Nemo are byte-identical to Llama 3 dense; the
// root file declares the dispatch metadata and routes
// `variant(for:)` to `LlamaDense`. These tests guard the registry
// surface + the variant dispatch shape.

import Foundation
import Testing
@testable import FFAI

@Suite("Mistral Family Root")
struct MistralRootTests {

    @Test("modelTypes advertises the mistral label")
    func modelTypes() {
        #expect(Mistral.modelTypes.contains("mistral"))
        #expect(!Mistral.modelTypes.isEmpty)
    }

    @Test("architectures advertises MistralForCausalLM")
    func architectures() {
        #expect(Mistral.architectures.contains("MistralForCausalLM"))
        #expect(!Mistral.architectures.isEmpty)
    }

    @Test("variant(for:) routes to LlamaDense (Mistral has no variants of its own)")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "MistralForCausalLM",
                              modelType: "mistral", raw: [:])
        let v = try Mistral.variant(for: cfg)
        #expect(String(describing: v) == String(describing: LlamaDense.self))
    }
}
