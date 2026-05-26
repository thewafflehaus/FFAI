// Qwen25Tests — root-file unit tests for `Sources/FFAI/Models/Qwen25.swift`.
//
// Offline. `Qwen25VL` is a VL-only orchestrator (no `model_type` set);
// the file declares the image / video token defaults + capability set
// + the `Qwen25VLError` cases. The `vision_config` decode surface is
// already covered by `Tests/FFAITests/Models/Vision/Qwen25VisionConfigTests.swift`.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen25VL Family Root")
struct Qwen25VLRootTests {

    @Test("Qwen25VL exposes the canonical image / video placeholder ids")
    func tokenIdDefaults() {
        #expect(Qwen25VL.defaultImageTokenId == 151_655)
        #expect(Qwen25VL.defaultVideoTokenId == 151_656)
    }

    @Test("availableCapabilities advertise text + image + video in")
    func capabilities() {
        let caps = Qwen25VL.availableCapabilities
        #expect(caps.contains(.textIn))
        #expect(caps.contains(.textOut))
        #expect(caps.contains(.visionIn))
        #expect(caps.contains(.videoIn))
    }

    @Test("Qwen25VLError stringifies every case with its payload")
    func errorDescriptions() {
        #expect(Qwen25VLError.missingConfig.description.contains("Qwen25VL"))
        #expect(Qwen25VLError.missingTensor("vision_tower.x").description
            .contains("vision_tower.x"))
    }
}
