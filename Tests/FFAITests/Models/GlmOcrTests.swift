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
// GlmOcrTests — root-file unit tests for `Sources/FFAI/Models/GlmOcr.swift`.
//
// Offline. The vision-config + dispatch routing surface is already
// covered by `Tests/FFAITests/Models/Vision/GlmOcrTests.swift`; this
// file focuses on the family-root constants + `GlmOcrError`
// descriptions that the Vision test doesn't cover.

import Foundation
import Testing
@testable import FFAI

@Suite("GlmOcr Family Root — error + constants")
struct GlmOcrRootTests {

    @Test("GlmOcr advertises the canonical model_type + architecture")
    func registration() {
        #expect(GlmOcr.modelTypes.contains("glm_ocr"))
        #expect(GlmOcr.architectures.contains("GlmOcrForConditionalGeneration"))
    }

    @Test("token id defaults match the shipped checkpoint")
    func tokenIdDefaults() {
        #expect(GlmOcr.defaultImageTokenId == 59_280)
        #expect(GlmOcr.defaultEosTokenId == 59_246)
    }

    @Test("GlmOcrError stringifies every case with its payload")
    func errorDescriptions() {
        #expect(GlmOcrError.missingConfig.description.contains("GlmOcr"))
        #expect(GlmOcrError.missingTensor("foo").description.contains("foo"))
    }
}
