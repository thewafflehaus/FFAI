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
// Gemma2Tests — root-file unit tests for `Sources/FFAI/Models/Gemma2.swift`.
//
// Offline. Covers the family enum's metadata + variant dispatch to
// `Gemma2Dense` and the `Gemma2Error` description. The concrete dense
// loader lives in `Models/Text/Gemma2Text.swift` and is exercised by
// the integration suite — this file is the lightweight root anchor.

import Foundation
import Testing
@testable import FFAI

@Suite("Gemma2 Family Root")
struct Gemma2RootTests {

    @Test("modelTypes covers both gemma2 and gemma2_text labels")
    func modelTypes() {
        #expect(Gemma2.modelTypes.contains("gemma2"))
        #expect(Gemma2.modelTypes.contains("gemma2_text"))
    }

    @Test("architectures advertises Gemma2ForCausalLM")
    func architectures() {
        #expect(Gemma2.architectures.contains("Gemma2ForCausalLM"))
        #expect(!Gemma2.architectures.isEmpty)
    }

    @Test("variant(for:) returns Gemma2Dense (the only variant today)")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "Gemma2ForCausalLM",
                              modelType: "gemma2", raw: [:])
        let v = try Gemma2.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Gemma2Dense.self))
    }

    @Test("Gemma2Error.missingConfig description names the family")
    func errorDescription() {
        #expect(Gemma2Error.missingConfig.description.contains("Gemma2"))
        #expect(Gemma2Error.missingConfig.description.contains("missing"))
    }
}
