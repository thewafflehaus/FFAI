// PixtralTests — root-file unit tests for `Sources/FFAI/Models/Pixtral.swift`.
//
// Offline. The Pixtral vision-config decode + dispatch routing is
// already covered by
// `Tests/FFAITests/Models/Vision/PixtralVisionConfigTests.swift`.
// This file focuses on the family-root constants + `PixtralError`
// descriptions.

import Foundation
import Testing
@testable import FFAI

@Suite("Pixtral Family Root — error + constants")
struct PixtralRootTests {

    @Test("Pixtral advertises the canonical model_type + architecture")
    func registration() {
        #expect(Pixtral.modelTypes.contains("pixtral"))
        #expect(Pixtral.architectures.contains("LlavaForConditionalGeneration"))
    }

    @Test("default image_token_id is the documented Pixtral 10 sentinel")
    func imageTokenId() {
        #expect(Pixtral.defaultImageTokenId == 10)
    }

    @Test("PixtralError stringifies every case with its payload")
    func errorDescriptions() {
        #expect(PixtralError.missingConfig.description.contains("Pixtral"))
        #expect(PixtralError.missingTensor("foo").description.contains("foo"))
    }
}
