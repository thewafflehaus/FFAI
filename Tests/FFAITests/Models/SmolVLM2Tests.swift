// SmolVLM2Tests — root-file unit tests for `Sources/FFAI/Models/SmolVLM2.swift`.
//
// Offline. The SmolVLM2 config + registry routing surface is already
// covered by `Tests/FFAITests/Models/Vision/SmolVLM2VisionConfigTests.swift`.
// This file focuses on the variant-dispatch shape +
// `SmolVLM2Dense` defaults + `SmolVLM2Error` descriptions.

import Foundation
import Testing
@testable import FFAI

@Suite("SmolVLM2 Family Root — variant + defaults")
struct SmolVLM2RootTests {

    @Test("variant(for:) returns SmolVLM2Dense (the only variant today)")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "SmolVLMForConditionalGeneration",
                              modelType: "smolvlm", raw: [:])
        let v = try SmolVLM2.variant(for: cfg)
        #expect(String(describing: v) == String(describing: SmolVLM2Dense.self))
    }

    @Test("SmolVLM2Dense advertises text + image + video capabilities")
    func capabilities() {
        let caps = SmolVLM2Dense.availableCapabilities
        #expect(caps.contains(.textIn))
        #expect(caps.contains(.textOut))
        #expect(caps.contains(.visionIn))
        #expect(caps.contains(.videoIn))
    }

    @Test("SmolVLM2Dense.defaultGenerationParameters are sane")
    func defaultGenParams() {
        let gp = SmolVLM2Dense.defaultGenerationParameters
        #expect(gp.maxTokens > 0)
        #expect(gp.temperature >= 0)
    }

    @Test("SmolVLM2Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(SmolVLM2Error.missingConfig("vision_config").description
            .contains("vision_config"))
        #expect(SmolVLM2Error.missingVisionConfig("hidden_size").description
            .contains("hidden_size"))
        #expect(SmolVLM2Error.missingTextConfig("vocab_size").description
            .contains("vocab_size"))
    }
}
