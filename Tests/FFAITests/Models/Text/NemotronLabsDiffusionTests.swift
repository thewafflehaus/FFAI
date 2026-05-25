// NemotronLabsDiffusionTests — pure-CPU coverage of the diffusion
// decode helpers and the family registry constants. The end-to-end
// tri-mode generation is exercised by the integration test under
// Tests/ModelTests/.

import Foundation
import Testing
@testable import FFAI

@Suite("NemotronLabsDiffusion Family Registry")
struct NemotronLabsDiffusionRegistryTests {

    @Test("family declares the expected model_type / architecture keys")
    func registryKeys() {
        #expect(NemotronLabsDiffusion.modelTypes.contains("nemotron_labs_diffusion"))
        #expect(NemotronLabsDiffusion.architectures.contains("NemotronLabsDiffusionModel"))
        // Must not collide with the planned NemotronH hybrid family.
        #expect(!NemotronLabsDiffusion.modelTypes.contains("nemotron_h"))
    }

    @Test("dense variant declares text-only capabilities")
    func variantCapabilities() throws {
        let config = ModelConfig(architecture: "NemotronLabsDiffusionModel",
                                 modelType: "nemotron_labs_diffusion", raw: [:])
        let variant = try NemotronLabsDiffusion.variant(for: config)
        #expect(variant.availableCapabilities == [.textIn, .textOut])
    }
}

@Suite("NemotronLabsDiffusion Confidence Transfer")
struct NemotronLabsDiffusionTransferTests {

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
            logitsTensor([0, 0, 5, 0]),       // non-mask — ignored for confidence
            logitsTensor([0, 0, 12, 0]),      // mask — softmax peak ≈ 1.0
            logitsTensor([1.0, 1.0, 1.0, 1.05]),  // mask — diffuse, low confidence
        ]
        let isMask = [false, true, true]
        let (x0, transfer) = Model.transferIndex(
            blockLogits: blockLogits, isMask: isMask,
            numTransfer: 99, threshold: 0.9)

        #expect(x0[1] == 2)              // argmax of the confident position
        #expect(x0[2] == 3)              // argmax of the diffuse position
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
            logitsTensor([0, 0, 8, 0]),       // confident
            logitsTensor([1.0, 1.0, 1.0, 1.02]),  // diffuse
            logitsTensor([0, 6, 0, 0]),       // medium
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
        #expect(NemotronLabsDiffusion.modelTypes.contains("nemotron_labs_diffusion"))
        #expect(NemotronLabsDiffusion.modelTypes.contains("nemotron_labs_diffusion_vlm"))
        #expect(NemotronLabsDiffusion.architectures.contains("NemotronLabsDiffusionModel"))
        #expect(NemotronLabsDiffusion.architectures.contains("NemotronLabsDiffusionVLMModel"))
    }
}
