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
// SmolLMTests — root-file unit tests for `Sources/FFAI/Models/SmolLM.swift`.
//
// Offline. SmolLM 1/2/3 are Llama-shaped pass-throughs; the root file
// just declares dispatch metadata. These tests guard that all three
// generations' model_type / architecture labels stay advertised.

import Foundation
import Testing
@testable import FFAI

@Suite("SmolLM Family Root")
struct SmolLMRootTests {

    @Test("modelTypes covers all three SmolLM generations")
    func modelTypes() {
        #expect(SmolLM.modelTypes.contains("smollm"))
        #expect(SmolLM.modelTypes.contains("smollm2"))
        #expect(SmolLM.modelTypes.contains("smollm3"))
    }

    @Test("architectures covers all three SmolLM generations")
    func architectures() {
        #expect(SmolLM.architectures.contains("SmolLMForCausalLM"))
        #expect(SmolLM.architectures.contains("SmolLM2ForCausalLM"))
        #expect(SmolLM.architectures.contains("SmolLM3ForCausalLM"))
    }
}
