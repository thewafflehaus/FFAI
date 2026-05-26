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
// InternLM2Tests — root-file unit tests for `Sources/FFAI/Models/InternLM2.swift`.
//
// Offline. InternLM 2 is a Llama-shaped pass-through; the root file
// just declares dispatch metadata. These tests guard the model_type /
// architecture constants the registry sniffs.

import Foundation
import Testing

@testable import FFAI

@Suite("InternLM2 Family Root")
struct InternLM2RootTests {

    @Test("modelTypes advertises the internlm2 label")
    func modelTypes() {
        #expect(InternLM2.modelTypes.contains("internlm2"))
        #expect(!InternLM2.modelTypes.isEmpty)
    }

    @Test("architectures advertises InternLM2ForCausalLM")
    func architectures() {
        #expect(InternLM2.architectures.contains("InternLM2ForCausalLM"))
        #expect(!InternLM2.architectures.isEmpty)
    }
}
