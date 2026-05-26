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
// Gemma3Tests — root-file unit tests for `Sources/FFAI/Models/Gemma3.swift`.
//
// Offline. Covers the family enum's metadata + variant dispatch to
// `Gemma3Dense` and the unified `Gemma3Error` description (raised by
// both the text loader AND the Gemma 3 VL orchestrator in
// `Models/Vision/Gemma3Vision.swift`).

import Foundation
import Testing

@testable import FFAI

@Suite("Gemma3 Family Root")
struct Gemma3RootTests {

    @Test("modelTypes covers both gemma3 and gemma3_text labels")
    func modelTypes() {
        #expect(Gemma3.modelTypes.contains("gemma3"))
        #expect(Gemma3.modelTypes.contains("gemma3_text"))
    }

    @Test("architectures covers both ForCausalLM variants")
    func architectures() {
        #expect(Gemma3.architectures.contains("Gemma3ForCausalLM"))
        #expect(Gemma3.architectures.contains("Gemma3TextForCausalLM"))
    }

    @Test("variant(for:) returns Gemma3Dense (the only variant today)")
    func variantDispatch() throws {
        let cfg = ModelConfig(
            architecture: "Gemma3ForCausalLM",
            modelType: "gemma3", raw: [:])
        let v = try Gemma3.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Gemma3Dense.self))
    }

    @Test("Gemma3Error.missingConfig description names the family")
    func errorDescription() {
        #expect(Gemma3Error.missingConfig.description.contains("Gemma3"))
        #expect(Gemma3Error.missingConfig.description.contains("missing"))
    }
}
