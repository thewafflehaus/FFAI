// NemotronDiffusionTests — root-file unit tests for
// `Sources/FFAI/Models/NemotronDiffusion.swift`.
//
// Offline. Covers the diffusion text family entry point + variant
// dispatch (`NemotronDiffusionDense`) + the `NemotronDiffusionError`
// case. The shared root `enum Nemotron` (modelTypes / architectures
// union) is covered in `NemotronTests.swift`.

import Foundation
import Testing
@testable import FFAI

@Suite("NemotronDiffusion Family Root")
struct NemotronDiffusionRootTests {

    @Test("modelTypes covers text + VLM diffusion checkpoints")
    func modelTypes() {
        #expect(NemotronDiffusion.modelTypes.contains("nemotron_labs_diffusion"))
        #expect(NemotronDiffusion.modelTypes.contains("nemotron_labs_diffusion_vlm"))
    }

    @Test("architectures covers text + VLM diffusion checkpoints")
    func architectures() {
        #expect(NemotronDiffusion.architectures.contains("NemotronDiffusionModel"))
        #expect(NemotronDiffusion.architectures.contains("NemotronDiffusionVLMModel"))
    }

    @Test("variant(for:) returns NemotronDiffusionDense")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "NemotronDiffusionModel",
                              modelType: "nemotron_labs_diffusion", raw: [:])
        let v = try NemotronDiffusion.variant(for: cfg)
        #expect(String(describing: v)
            == String(describing: NemotronDiffusionDense.self))
    }

    @Test("NemotronDiffusionError.missingConfig description names the family")
    func errorDescription() {
        #expect(NemotronDiffusionError.missingConfig.description
            .contains("NemotronDiffusion"))
        #expect(NemotronDiffusionError.missingConfig.description
            .contains("missing"))
    }
}
