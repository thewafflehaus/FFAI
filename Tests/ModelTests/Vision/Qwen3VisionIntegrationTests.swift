// Slow integration test for Qwen 3-VL (the dynamic-resolution
// full-attention ViT + Qwen 3 dense text backbone
// `Qwen3VLForConditionalGeneration` checkpoint).
//
// Verifies the vision path end-to-end on a real checkpoint: the
// Qwen 3-VL vision tower loads, runs its full-attention + M-RoPE forward
// and patch-merger, the cross-modal splice injects the merged image
// tokens, and the fused stream decodes coherent text through the Qwen 3
// backbone (which supports embedding-input forward for the splice).
//
// Uses the mlx-community 2B-Instruct 4-bit conversion — the smallest
// published Qwen3-VL with a complete local snapshot. The checkpoint MUST
// load — a load failure fails the test.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Qwen3 Vision Integration", .serialized)
struct Qwen3VisionIntegrationTests {

    static let modelId = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

    @Test("load — Qwen 3-VL checkpoint loads with vision capability")
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
        // The vision tower contributes a positive run of merged tokens.
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
        let vlm = try #require(m.vlModel, "Qwen 3-VL checkpoint is not a VLM")

        // Build an image+text prompt: a run of `imageTokenCount`
        // <|image_pad|> placeholders followed by a text question.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "<|im_start|>user\nDescribe this image.<|im_end|>\n"
                + "<|im_start|>assistant\n")
        let promptTokens = Array(repeating: imageTokenId,
                                 count: vlm.imageTokenCount) + questionTokens

        // A real photograph — the golden-retriever fixture.
        let image = try VisionTestHelpers.dogImage()

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence first, then the content check: the caption should
        // mention a dog.
        expectCoherentOutput(generated, minTokens: 8,
                             label: "Qwen 3-VL image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 3-VL generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "Qwen 3-VL")
    }
}
