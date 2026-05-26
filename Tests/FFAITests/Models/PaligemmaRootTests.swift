// PaligemmaTests — root-file unit tests for `Sources/FFAI/Models/Paligemma.swift`.
//
// Offline. The Paligemma registry routing + family enum membership is
// already covered by `Tests/FFAITests/Models/Vision/PaligemmaTests.swift`.
// This file focuses on the variant dispatch shape +
// `PaligemmaStandard` defaults + `PaligemmaError` descriptions.

import Foundation
import Testing
@testable import FFAI

@Suite("Paligemma Family Root — variant + defaults")
struct PaligemmaRootTests {

    @Test("variant(for:) returns PaligemmaStandard")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "PaliGemmaForConditionalGeneration",
                              modelType: "paligemma", raw: [:])
        let v = try Paligemma.variant(for: cfg)
        #expect(String(describing: v)
            == String(describing: PaligemmaStandard.self))
    }

    @Test("PaligemmaStandard advertises text + visionIn")
    func capabilities() {
        let caps = PaligemmaStandard.availableCapabilities
        #expect(caps.contains(.textIn))
        #expect(caps.contains(.textOut))
        #expect(caps.contains(.visionIn))
    }

    @Test("PaligemmaStandard.defaultGenerationParameters are greedy by default")
    func defaultGenParams() {
        let gp = PaligemmaStandard.defaultGenerationParameters
        #expect(gp.maxTokens > 0)
        // The reference recommends greedy for VQA/caption tasks.
        #expect(gp.temperature == 0)
        #expect(gp.topP == 1.0)
    }

    @Test("PaligemmaError stringifies every case with its payload")
    func errorDescriptions() {
        #expect(PaligemmaError.missingConfig("vision_config").description
            .contains("vision_config"))
        #expect(PaligemmaError.imageNotSet.description.contains("setImagePixels"))
        #expect(PaligemmaError.missingConfig("x").description.contains("Paligemma"))
    }
}
