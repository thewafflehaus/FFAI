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
