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
}
