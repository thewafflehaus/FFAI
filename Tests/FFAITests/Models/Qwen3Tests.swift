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
// Qwen3Tests — root-file unit tests for `Sources/FFAI/Models/Qwen3.swift`.
//
// Offline. Covers Qwen 3 family metadata + variant dispatch
// (`Qwen3Dense`) + the unified `Qwen3Error` cases. The VL surface is
// already covered by `Tests/FFAITests/Models/Vision/Qwen3VisionConfigTests.swift`
// — this file focuses on the family-root anchor.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3 Family Root")
struct Qwen3RootTests {

    @Test("modelTypes advertises the qwen3 label")
    func modelTypes() {
        #expect(Qwen3.modelTypes.contains("qwen3"))
    }

    @Test("architectures covers both ForCausalLM and Qwen3VL")
    func architectures() {
        #expect(Qwen3.architectures.contains("Qwen3ForCausalLM"))
        #expect(Qwen3.architectures.contains("Qwen3VLForConditionalGeneration"))
    }

    @Test("variant(for:) returns Qwen3Dense")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "Qwen3ForCausalLM",
                              modelType: "qwen3", raw: [:])
        let v = try Qwen3.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Qwen3Dense.self))
    }

    @Test("Qwen3Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(Qwen3Error.missingConfig.description.contains("Qwen3"))
        #expect(Qwen3Error.missingTensor("model.embed_tokens.weight").description
            .contains("model.embed_tokens.weight"))
    }
}
