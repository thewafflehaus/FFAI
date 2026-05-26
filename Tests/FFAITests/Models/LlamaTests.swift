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
        let cfg = ModelConfig(
            architecture: "LlamaForCausalLM",
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
