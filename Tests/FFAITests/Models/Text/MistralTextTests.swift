// MistralTextTests — unit coverage for `Sources/FFAI/Models/Text/MistralText.swift`.
//
// Offline. `MistralText.swift` ships no Swift declarations — Mistral 7B
// / Nemo / Small route their dense-text path through `LlamaDense` from
// the Llama family. This file exists so the source-to-test mirror is
// complete (every Models/Text/<X>Text.swift has a corresponding
// Tests/FFAITests/Models/Text/<X>TextTests.swift) and asserts the
// routing contract: `Mistral.variant(for:)` returns `LlamaDense.self`.

import Foundation
import Testing
@testable import FFAI

@Suite("MistralText Routes Through LlamaDense")
struct MistralTextTests {

    @Test("Mistral.variant(for:) returns LlamaDense — Mistral has no own variant")
    func routesThroughLlamaDense() throws {
        let cfg = ModelConfig(architecture: "MistralForCausalLM",
                              modelType: "mistral", raw: [:])
        let v = try Mistral.variant(for: cfg)
        #expect(String(describing: v) == String(describing: LlamaDense.self))
    }
}
