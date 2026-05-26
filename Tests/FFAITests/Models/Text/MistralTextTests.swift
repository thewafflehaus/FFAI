// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
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
        let cfg = ModelConfig(
            architecture: "MistralForCausalLM",
            modelType: "mistral", raw: [:])
        let v = try Mistral.variant(for: cfg)
        #expect(String(describing: v) == String(describing: LlamaDense.self))
    }
}
