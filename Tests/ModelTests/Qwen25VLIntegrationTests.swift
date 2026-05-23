// Slow integration test for Qwen 2.5-VL (the dynamic-resolution
// windowed-attention ViT + Qwen 2.x text backbone
// `Qwen2_5_VLForConditionalGeneration` checkpoint).
//
// Verifies the vision path end-to-end on a real checkpoint: the
// Qwen 2.5-VL vision tower loads, runs its windowed-attention + M-RoPE
// forward and patch-merger, the cross-modal splice injects the merged
// image tokens, and the fused stream decodes coherent text through the
// Qwen 2.x backbone (routed through the Llama dense engine, which now
// supports embedding-input forward for the splice).
//
// Uses the mlx-community 3B-Instruct conversion (smallest published
// Qwen 2.5-VL). The checkpoint MUST load — a load failure fails the test.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen 2.5-VL integration", .serialized)
struct Qwen25VLIntegrationTests {

    static let modelId = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

    @Test("load — Qwen 2.5-VL checkpoint loads with vision capability")
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
        let vlm = try #require(m.vlModel, "Qwen 2.5-VL checkpoint is not a VLM")

        // Build an image+text prompt: a run of `imageTokenCount`
        // <|image_pad|> placeholders wrapped in vision-start / vision-end
        // markers, followed by a text question. Qwen's chat template
        // normally expands the placeholder; here we assemble it directly.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "<|im_start|>user\nDescribe this image.<|im_end|>\n"
                + "<|im_start|>assistant\n")
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
                             label: "Qwen 2.5-VL image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 2.5-VL generated: \(text)")
        VLMTestSupport.expectMentionsDog(text, label: "Qwen 2.5-VL")
    }
}
