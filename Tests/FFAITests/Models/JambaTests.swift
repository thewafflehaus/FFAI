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
// JambaTests — root-file unit tests for `Sources/FFAI/Models/Jamba.swift`.
//
// Offline. Covers Jamba family enum metadata + variant dispatch
// (`JambaHybrid` is the only variant) + `JambaError` cases.

import Foundation
import Testing
@testable import FFAI

@Suite("Jamba Family Root")
struct JambaRootTests {

    @Test("modelTypes advertises jamba")
    func modelTypes() {
        #expect(Jamba.modelTypes.contains("jamba"))
    }

    @Test("architectures advertises JambaForCausalLM")
    func architectures() {
        #expect(Jamba.architectures.contains("JambaForCausalLM"))
    }

    @Test("variant(for:) returns JambaHybrid")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "JambaForCausalLM",
                              modelType: "jamba", raw: [:])
        let v = try Jamba.variant(for: cfg)
        #expect(String(describing: v) == String(describing: JambaHybrid.self))
    }

    @Test("JambaError stringifies every case with its payload")
    func errorDescriptions() {
        #expect(JambaError.missingConfig("hidden_size").description
            .contains("hidden_size"))
        #expect(JambaError.unsupportedConfig("bad").description.contains("bad"))
        #expect(JambaError.missingConfig("x").description.contains("Jamba"))
    }
}
