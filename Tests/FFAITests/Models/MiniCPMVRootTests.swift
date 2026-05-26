// MiniCPMVTests — root-file unit tests for `Sources/FFAI/Models/MiniCPMV.swift`.
//
// Offline. MiniCPM-V-4.6 registration + the bilinear position-embed
// resample helper are already covered by
// `Tests/FFAITests/Models/Vision/MiniCPMVTests.swift`. This file
// focuses on the family-root constants + `MiniCPMVError`
// descriptions + the multi-modality capability advertisement.

import Foundation
import Testing
@testable import FFAI

@Suite("MiniCPM-V Family Root — error + multi-modality capabilities")
struct MiniCPMVRootTests {

    @Test("MiniCPMV4_6 advertises the canonical model_type + architecture")
    func registration() {
        #expect(MiniCPMV4_6.modelTypes.contains("minicpmv4_6"))
        #expect(MiniCPMV4_6.architectures
            .contains("MiniCPMV4_6ForConditionalGeneration"))
    }

    @Test("token id defaults match the shipped checkpoint")
    func tokenIdDefaults() {
        #expect(MiniCPMV4_6.defaultImageTokenId == 248_056)
        #expect(MiniCPMV4_6.defaultVideoTokenId == 248_057)
    }

    @Test("availableCapabilities advertise text + image + video in")
    func capabilities() {
        let caps = MiniCPMV4_6.availableCapabilities
        #expect(caps.contains(.textIn))
        #expect(caps.contains(.textOut))
        #expect(caps.contains(.visionIn))
        #expect(caps.contains(.videoIn))
    }

    @Test("runtimeImageSize is the 448px tile the chat template emits")
    func runtimeImageSize() {
        #expect(MiniCPMV4_6.runtimeImageSize == 448)
    }

    @Test("MiniCPMVError stringifies every case with its payload")
    func errorDescriptions() {
        #expect(MiniCPMVError.missingConfig("vision_config").description
            .contains("vision_config"))
        #expect(MiniCPMVError.unsupportedConfig("bad").description.contains("bad"))
        #expect(MiniCPMVError.missingConfig("x").description.contains("MiniCPM-V"))
    }
}
