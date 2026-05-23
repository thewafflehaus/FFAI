// Slow integration test for Gemma 4 VL (the bespoke Gemma 4 ViT tower +
// multi-modal embedder + Gemma 4 text backbone, the
// `Gemma4ForConditionalGeneration` checkpoint).
//
// Verifies the vision path end-to-end on a real checkpoint: the Gemma 4
// vision tower loads, runs its RoPE-attention forward and
// attention-pooling head, the multi-modal embedder projects the pooled
// soft tokens into the text hidden dim, the cross-modal splice injects
// them, and the fused stream decodes coherent text through the Gemma 4
// backbone (which now supports embedding-input forward for the splice).
//
// Uses the mlx-community gemma-4-e2b-it 4-bit conversion — the smallest
// published Gemma 4 VL (its `config.json` carries a `vision_config`, so
// it routes through the Gemma4VL loader). The checkpoint MUST load — a
// load failure fails the test.

import Foundation
import Testing
@testable import FFAI

@Suite("Gemma 4 VL integration", .serialized)
struct Gemma4VLIntegrationTests {

    static let modelId = "mlx-community/gemma-4-e2b-it-4bit"

    @Test("load — Gemma 4 VL checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        // The checkpoint is a VLM — vlModel is present, .visionIn is
        // available, and the text backbone supports the splice.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.visionIn))
        #expect(m.engine.supportsEmbeddingInput)  // VLM splice prerequisite

        let vlm = try #require(m.vlModel)
        // The vision tower contributes a positive run of soft tokens.
        #expect(vlm.imageTokenCount > 0)
    }

    @Test("enable / disable .visionIn — runtime capability flip")
    func capabilityFlip() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        #expect(m.availableCapabilities.contains(.visionIn))
        m.disable(.visionIn)
        #expect(!m.isEnabled(.visionIn))
        m.enable(.visionIn)
        #expect(m.isEnabled(.visionIn))
    }

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "Gemma 4 VL checkpoint is not a VLM")

        // Build an image+text prompt: a run of `imageTokenCount`
        // <image_soft_token> placeholders followed by a text question.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "<start_of_turn>user\nDescribe this image.<end_of_turn>\n"
                + "<start_of_turn>model\n")
        let promptTokens = Array(repeating: imageTokenId,
                                 count: vlm.imageTokenCount) + questionTokens

        // A real photograph — the golden-retriever fixture.
        let image = try VLMTestSupport.dogImage()

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 64, eosTokenId: m.config.eosTokenId)

        // Coherence first, then the content check: the caption should
        // mention a dog.
        expectCoherentOutput(generated, minTokens: 8,
                             label: "Gemma 4 VL image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Gemma 4 VL generated: \(text)")
        VLMTestSupport.expectMentionsDog(text, label: "Gemma 4 VL")
    }
}
