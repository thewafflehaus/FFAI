// NemotronDiffusionTextTests — unit coverage for
// `Sources/FFAI/Models/Text/NemotronDiffusionText.swift`.
//
// Offline. Covers the `NemotronDiffusionDense` variant surface
// (capabilities + greedy generation defaults) and the
// `NemotronDiffusionError.missingConfig` description. The full
// tri-mode (AR / block-diffusion / self-speculation) decoder + the
// YaRN RoPE loader path are exercised by
// Tests/ModelTests/NemotronDiffusionIntegrationTests.swift and by the
// existing `NemotronLabsDiffusionTests.swift` (kept under its
// pre-rename basename as a focused supplementary suite). This file is
// the lightweight surface guard.

import Foundation
import Testing
@testable import FFAI

@Suite("NemotronDiffusionDense Variant Surface")
struct NemotronDiffusionTextTests {

    @Test("NemotronDiffusionDense advertises text in/out capabilities")
    func capabilities() {
        #expect(NemotronDiffusionDense.availableCapabilities.contains(.textIn))
        #expect(NemotronDiffusionDense.availableCapabilities.contains(.textOut))
        #expect(!NemotronDiffusionDense.availableCapabilities.contains(.visionIn))
    }

    @Test("NemotronDiffusionDense default generation parameters are greedy")
    func defaultGenerationParameters() {
        let p = NemotronDiffusionDense.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature == 0.0)
        #expect(p.topP == 1.0)
        #expect((p.prefillStepSize ?? 0) >= 256)
    }

    @Test("NemotronDiffusionError.missingConfig description names the family")
    func errorDescription() {
        let desc = NemotronDiffusionError.missingConfig.description
        #expect(desc.contains("NemotronDiffusion"))
        #expect(desc.contains("missing"))
    }
}
