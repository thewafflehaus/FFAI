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
