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
// Idefics3Tests — root-file unit tests for `Sources/FFAI/Models/Idefics3.swift`.
//
// Offline. The Idefics3 config decode + registry routing surface is
// already covered by `Tests/FFAITests/Models/Vision/Idefics3Tests.swift`;
// this file focuses on the variant-dispatch shape + `Idefics3Error`
// descriptions + the variant's defaultGenerationParameters sanity.

import Foundation
import Testing

@testable import FFAI

@Suite("Idefics3 Family Root — variant + defaults")
struct Idefics3RootTests {

    @Test("variant(for:) returns Idefics3Dense (the only variant today)")
    func variantDispatch() throws {
        let cfg = ModelConfig(
            architecture: "Idefics3ForConditionalGeneration",
            modelType: "idefics3", raw: [:])
        let v = try Idefics3.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Idefics3Dense.self))
    }

    @Test("Idefics3Dense exposes vision-in capability")
    func capabilities() {
        let caps = Idefics3Dense.availableCapabilities
        #expect(caps.contains(.textIn))
        #expect(caps.contains(.textOut))
        #expect(caps.contains(.imageIn))
    }

    @Test("Idefics3Dense defaultGenerationParameters are sane")
    func defaultGenParams() {
        let gp = Idefics3Dense.defaultGenerationParameters
        #expect(gp.maxTokens > 0)
        #expect(gp.temperature >= 0)
    }

    @Test("Idefics3Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(
            Idefics3Error.missingConfig("text_config").description
                .contains("text_config"))
        #expect(
            Idefics3Error.missingVisionConfig("hidden_size").description
                .contains("hidden_size"))
        #expect(
            Idefics3Error.missingTextConfig("vocab_size").description
                .contains("vocab_size"))
    }
}
