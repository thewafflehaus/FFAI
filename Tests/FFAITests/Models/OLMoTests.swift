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
// OLMoTests — root-file unit tests for `Sources/FFAI/Models/OLMo.swift`.
//
// Offline. OLMo 1 / OLMo 2 are Llama-shaped pass-throughs; the root
// file just declares dispatch metadata. These tests guard the
// model_type / architecture constants for both generations.

import Foundation
import Testing
@testable import FFAI

@Suite("OLMo Family Root")
struct OLMoRootTests {

    @Test("modelTypes covers both olmo and olmo2")
    func modelTypes() {
        #expect(OLMo.modelTypes.contains("olmo"))
        #expect(OLMo.modelTypes.contains("olmo2"))
    }

    @Test("architectures covers OlmoForCausalLM and Olmo2ForCausalLM")
    func architectures() {
        #expect(OLMo.architectures.contains("OlmoForCausalLM"))
        #expect(OLMo.architectures.contains("Olmo2ForCausalLM"))
    }
}
