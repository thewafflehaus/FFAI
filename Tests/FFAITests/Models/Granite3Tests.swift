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
// Granite3Tests — root-file unit tests for `Sources/FFAI/Models/Granite3.swift`.
//
// Offline. Granite v3 is a Llama-shaped pass-through; the root file
// only declares dispatch metadata (no variant protocol, no error
// type). These tests guard those constants so the registry sniff
// stays anchored.

import Foundation
import Testing

@testable import FFAI

@Suite("Granite3 Family Root")
struct Granite3RootTests {

    @Test("modelTypes advertises the granite label")
    func modelTypes() {
        #expect(Granite3.modelTypes.contains("granite"))
        #expect(!Granite3.modelTypes.isEmpty)
    }

    @Test("architectures advertises GraniteForCausalLM")
    func architectures() {
        #expect(Granite3.architectures.contains("GraniteForCausalLM"))
        #expect(!Granite3.architectures.isEmpty)
    }

    @Test("muP(from:) parses the four µP multipliers from config")
    func muPParsing() {
        // Values in the ballpark of the published granite-3.x configs
        // (embedding_multiplier 12, attention_multiplier 0.0078125,
        // residual_multiplier 0.22, logits_scaling 8).
        let cfg = ModelConfig(
            architecture: "GraniteForCausalLM", modelType: "granite",
            raw: [
                "embedding_multiplier": 12.0,
                "attention_multiplier": 0.0078125,
                "residual_multiplier": 0.22,
                "logits_scaling": 8.0,
            ])
        let muP = Granite3.muP(from: cfg)
        #expect(muP.embedding == 12.0)
        #expect(muP.attention == Float(0.0078125))
        #expect(muP.residual == Float(0.22))
        #expect(muP.logits == 8.0)
        #expect(!muP.isIdentity)
    }

    @Test("muP(from:) is identity when the multipliers are absent")
    func muPDefaults() {
        let cfg = ModelConfig(
            architecture: "GraniteForCausalLM", modelType: "granite", raw: [:])
        let muP = Granite3.muP(from: cfg)
        #expect(muP.embedding == 1.0)
        #expect(muP.attention == nil)
        #expect(muP.residual == 1.0)
        #expect(muP.logits == 1.0)
        #expect(muP.isIdentity)
    }
}
