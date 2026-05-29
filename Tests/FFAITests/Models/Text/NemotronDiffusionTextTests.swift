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
// NemotronDiffusionTextTests — unit coverage for
// `Sources/FFAI/Models/Text/NemotronDiffusionText.swift`.
//
// Offline. Covers:
//   • the `NemotronDiffusionDense` variant surface (capabilities +
//     greedy generation defaults),
//   • the `NemotronDiffusionError.missingConfig` description,
//   • the family registry constants (`modelTypes` / `architectures`),
//   • the diffusion decode helpers (`numTransferTokens`,
//     `transferIndex`) — pure-CPU coverage.
//
// The full tri-mode (AR / block-diffusion / self-speculation) decoder
// + the YaRN RoPE loader path are exercised by
// Tests/ModelTests/NemotronDiffusionIntegrationTests.swift.

import Foundation
import Testing

@testable import FFAI

@Suite("NemotronDiffusionDense Variant Surface")
struct NemotronDiffusionTextTests {

    @Test("NemotronDiffusionDense advertises text in/out capabilities")
    func capabilities() {
        #expect(NemotronDiffusionDense.availableCapabilities.contains(.textIn))
        #expect(NemotronDiffusionDense.availableCapabilities.contains(.textOut))
        #expect(!NemotronDiffusionDense.availableCapabilities.contains(.imageIn))
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

    // ─── Family registry (was NemotronDiffusionRegistryTests) ─────────

    @Test("family declares the expected model_type / architecture keys")
    func registryKeys() {
        #expect(NemotronDiffusion.modelTypes.contains("nemotron_labs_diffusion"))
        #expect(NemotronDiffusion.architectures.contains("NemotronDiffusionModel"))
        // Must not collide with the planned NemotronH hybrid family.
        #expect(!NemotronDiffusion.modelTypes.contains("nemotron_h"))
    }

    @Test("dense variant declares text-only capabilities")
    func variantCapabilities() throws {
        let config = ModelConfig(
            architecture: "NemotronDiffusionModel",
            modelType: "nemotron_labs_diffusion", raw: [:])
        let variant = try NemotronDiffusion.variant(for: config)
        #expect(variant.availableCapabilities == [.textIn, .textOut])
    }

    // ─── Confidence transfer (was NemotronDiffusionTransferTests) ─────────

    @Test("numTransferTokens splits evenly with the remainder front-loaded")
    func numTransferEvenSplit() {
        // 10 masked positions over 4 steps → 3,3,2,2.
        #expect(Model.numTransferTokens(maskCount: 10, steps: 4) == [3, 3, 2, 2])
        // Exact division → uniform.
        #expect(Model.numTransferTokens(maskCount: 8, steps: 4) == [2, 2, 2, 2])
        // Fewer masked than steps → remainder front-loads single tokens.
        #expect(Model.numTransferTokens(maskCount: 3, steps: 5) == [1, 1, 1, 0, 0])
        // Sum always equals the mask count.
        #expect(Model.numTransferTokens(maskCount: 31, steps: 32).reduce(0, +) == 31)
    }

    private func logitsTensor(_ values: [Float]) -> Tensor {
        let t = Tensor.empty(shape: [values.count], dtype: .f32)
        t.copyIn(from: values)
        return t
    }

    @Test("threshold commits high-confidence positions, always at least one")
    func transferWithThreshold() {
        // pos0 non-mask; pos1 mask, very confident; pos2 mask, diffuse.
        let blockLogits = [
            logitsTensor([0, 0, 5, 0]),  // non-mask — ignored for confidence
            logitsTensor([0, 0, 12, 0]),  // mask — softmax peak ≈ 1.0
            logitsTensor([1.0, 1.0, 1.0, 1.05]),  // mask — diffuse, low confidence
        ]
        let isMask = [false, true, true]
        let (x0, transfer) = Model.transferIndex(
            blockLogits: blockLogits, isMask: isMask,
            numTransfer: 99, threshold: 0.9)

        #expect(x0[1] == 2)  // argmax of the confident position
        #expect(x0[2] == 3)  // argmax of the diffuse position
        // Only the confident masked position clears the 0.9 threshold;
        // the diffuse one is rank 1 and below threshold → not committed.
        #expect(transfer == [1])
    }

    @Test("threshold always commits the single best masked position")
    func transferThresholdForcesProgress() {
        // Both masked positions are diffuse / below threshold.
        let blockLogits = [
            logitsTensor([1.0, 1.05, 1.0, 1.0]),
            logitsTensor([1.0, 1.0, 1.1, 1.0]),
        ]
        let (_, transfer) = Model.transferIndex(
            blockLogits: blockLogits, isMask: [true, true],
            numTransfer: 99, threshold: 0.95)
        // Rank-0 (highest confidence) always commits so the block
        // never stalls — exactly one position transfers.
        #expect(transfer.count == 1)
    }

    @Test("without a threshold the top-numTransfer positions commit")
    func transferTopK() {
        let blockLogits = [
            logitsTensor([0, 0, 8, 0]),  // confident
            logitsTensor([1.0, 1.0, 1.0, 1.02]),  // diffuse
            logitsTensor([0, 6, 0, 0]),  // medium
        ]
        let (_, transfer) = Model.transferIndex(
            blockLogits: blockLogits, isMask: [true, true, true],
            numTransfer: 2, threshold: nil)
        #expect(transfer.count == 2)
        // The two highest-confidence positions are 0 and 2.
        #expect(Set(transfer) == Set([0, 2]))
    }

    @Test("family registers both the text-only and VLM checkpoints")
    func familyRegistersTextAndVLM() {
        // The VLM checkpoint's text backbone shares the `encoder.*`
        // layout, so both route through this family.
        #expect(NemotronDiffusion.modelTypes.contains("nemotron_labs_diffusion"))
        #expect(NemotronDiffusion.modelTypes.contains("nemotron_labs_diffusion_vlm"))
        #expect(NemotronDiffusion.architectures.contains("NemotronDiffusionModel"))
        #expect(NemotronDiffusion.architectures.contains("NemotronDiffusionVLMModel"))
    }

    @Test("DiffusionMode covers all three decode strategies")
    func diffusionModeCases() {
        // The unified `generate(prompt:mode:)` selector dispatches on
        // these; if a strategy is added the switch must be updated too.
        #expect(
            Set(DiffusionMode.allCases)
                == Set([.autoregressive, .diffusion, .selfSpeculative]))
        // Raw values are stable (callers / CLI may parse them).
        #expect(DiffusionMode.selfSpeculative.rawValue == "selfSpeculative")
        #expect(DiffusionMode.diffusion.rawValue == "diffusion")
        #expect(DiffusionMode.autoregressive.rawValue == "autoregressive")
    }

    @Test("LoadOptions.diffusionMode defaults to self-speculation, honors override")
    func loadOptionsDiffusionModeDefault() {
        // Bare load → self-speculation (matches generate(mode:nil) fallback).
        #expect(LoadOptions().diffusionMode == .selfSpeculative)
        // Explicit init-time selection is preserved on the struct.
        #expect(LoadOptions(diffusionMode: .diffusion).diffusionMode == .diffusion)
        #expect(LoadOptions(diffusionMode: .autoregressive).diffusionMode == .autoregressive)
    }
}
