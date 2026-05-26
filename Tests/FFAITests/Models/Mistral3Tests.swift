// Mistral3Tests — root-file unit tests for `Sources/FFAI/Models/Mistral3.swift`.
//
// Offline. The Mistral3 vision-config + dispatch routing surface is
// already covered by `Tests/FFAITests/Models/Vision/Mistral3Tests.swift`.
// This file focuses on the family-root constants + `Mistral3Error`
// descriptions.

import Foundation
import Testing
@testable import FFAI

@Suite("Mistral3 Family Root — error + constants")
struct Mistral3RootTests {

    @Test("Mistral3 advertises the canonical model_type + architecture")
    func registration() {
        #expect(Mistral3.modelTypes.contains("mistral3"))
        #expect(Mistral3.architectures
            .contains("Mistral3ForConditionalGeneration"))
    }

    @Test("Mistral3 exposes the canonical defaults")
    func defaults() {
        #expect(Mistral3.defaultImageTokenId == 10)
        #expect(Mistral3.defaultSpatialMergeSize == 2)
    }

    @Test("Mistral3Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(Mistral3Error.missingConfig.description.contains("Mistral3"))
        #expect(Mistral3Error.missingTensor("foo").description.contains("foo"))
    }
}
