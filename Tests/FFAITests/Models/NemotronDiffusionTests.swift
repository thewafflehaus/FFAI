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
        let cfg = ModelConfig(
            architecture: "NemotronDiffusionModel",
            modelType: "nemotron_labs_diffusion", raw: [:])
        let v = try NemotronDiffusion.variant(for: cfg)
        #expect(
            String(describing: v)
                == String(describing: NemotronDiffusionDense.self))
    }

    @Test("NemotronDiffusionError.missingConfig description names the family")
    func errorDescription() {
        #expect(
            NemotronDiffusionError.missingConfig.description
                .contains("NemotronDiffusion"))
        #expect(
            NemotronDiffusionError.missingConfig.description
                .contains("missing"))
    }
}
