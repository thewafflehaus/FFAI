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
// Offline. OLMo 1 is a pre-norm Llama-shaped pass-through; OLMo 2 is a
// post-norm + q/k-norm decoder with its own loader. These tests guard
// the dispatch metadata and the olmo1/olmo2 split the loader relies on.

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

    @Test("olmo1 / olmo2 dispatch sets are disjoint and correct")
    func generationSplit() {
        // OLMo 1 (pre-norm) → Llama path; OLMo 2 (post-norm) → dedicated.
        #expect(OLMo.olmo1Architectures == ["OlmoForCausalLM"])
        #expect(OLMo.olmo1ModelTypes == ["olmo"])
        #expect(OLMo.olmo2Architectures == ["Olmo2ForCausalLM"])
        #expect(OLMo.olmo2ModelTypes == ["olmo2"])
        // No architecture is claimed by both paths.
        #expect(OLMo.olmo1Architectures.isDisjoint(with: OLMo.olmo2Architectures))
        #expect(OLMo.olmo1ModelTypes.isDisjoint(with: OLMo.olmo2ModelTypes))
        // Together they reconstruct the full advertised set.
        #expect(
            OLMo.olmo1Architectures.union(OLMo.olmo2Architectures) == OLMo.architectures)
        #expect(OLMo.olmo1ModelTypes.union(OLMo.olmo2ModelTypes) == OLMo.modelTypes)
    }
}
