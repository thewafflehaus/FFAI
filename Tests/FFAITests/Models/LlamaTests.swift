// LlamaTests — root-file unit tests for `Sources/FFAI/Models/Llama.swift`.
//
// Offline. Covers the family enum's modelTypes / architectures
// metadata, the `variant(for:)` dispatch (`LlamaDense` is the only
// shipped variant), and the `LlamaError` description shape. The dense
// decoder itself + its weight-loading paths are exercised by the
// model integration suite — this file is the lightweight surface
// guard for the root anchor.

import Foundation
import Testing
@testable import FFAI

@Suite("Llama Family Root")
struct LlamaRootTests {

    @Test("modelTypes / architectures advertise the canonical entries")
    func familyMetadata() {
        #expect(Llama.modelTypes.contains("llama"))
        #expect(Llama.architectures.contains("LlamaForCausalLM"))
        #expect(!Llama.modelTypes.isEmpty)
        #expect(!Llama.architectures.isEmpty)
    }

    @Test("variant(for:) returns LlamaDense for any llama config")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "LlamaForCausalLM",
                              modelType: "llama", raw: [:])
        let v = try Llama.variant(for: cfg)
        // Metatype comparison via String(describing:) — value-type
        // metatypes don't conform to Equatable.
        #expect(String(describing: v) == String(describing: LlamaDense.self))
    }

    @Test("LlamaError.missingConfig description names the family + cause")
    func errorDescription() {
        let desc = LlamaError.missingConfig.description
        #expect(desc.contains("Llama"))
        #expect(desc.contains("missing"))
    }
}
