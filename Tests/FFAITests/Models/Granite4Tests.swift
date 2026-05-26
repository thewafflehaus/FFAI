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
// Granite4Tests — root-file unit tests for `Sources/FFAI/Models/Granite4.swift`.
//
// Offline. Covers Granite 4 family metadata + variant dispatch
// (`Granite4Hybrid` is the only variant) + `Granite4Error`
// descriptions.

import Foundation
import Testing
@testable import FFAI

@Suite("Granite4 Family Root")
struct Granite4RootTests {

    @Test("modelTypes advertises granitemoehybrid")
    func modelTypes() {
        #expect(Granite4.modelTypes.contains("granitemoehybrid"))
    }

    @Test("architectures advertises Granite4ForCausalLM")
    func architectures() {
        #expect(Granite4.architectures.contains("Granite4ForCausalLM"))
    }

    @Test("variant(for:) returns Granite4Hybrid")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "Granite4ForCausalLM",
                              modelType: "granitemoehybrid", raw: [:])
        let v = try Granite4.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Granite4Hybrid.self))
    }

    @Test("Granite4Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(Granite4Error.missingConfig("layer_types").description
            .contains("layer_types"))
        #expect(Granite4Error.unsupportedConfig("bad").description.contains("bad"))
        #expect(Granite4Error.missingConfig("x").description.contains("Granite4"))
    }
}
