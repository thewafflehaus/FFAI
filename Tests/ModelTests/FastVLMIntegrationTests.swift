// Slow integration test for FastVLM (Apple's FastViTHD + mlp2x_gelu
// projector + Qwen2 text backbone, the `LlavaQwen2ForCausalLM` /
// `llava_qwen2` checkpoint).
//
// Verifies the FastVLM vision path end-to-end on a real checkpoint:
// the FastViTHD tower loads its reparameterized conv weights (including
// BN folding), the mlp2x_gelu projector maps 3072-dim feature tokens to
// the Qwen2 text hidden dim (896), the cross-modal splice injects the
// image tokens, and the fused stream decodes coherent text through the
// Qwen2-0.5B backbone.
//
// Uses the mlx-community FastVLM-0.5B-bf16 conversion. The checkpoint
// MUST be available at the standard HuggingFace cache path.
//
// DO NOT run this test directly with `swift test` — use `make test-integration`
// to serialize model loads and avoid GPU memory pressure.

import Foundation
import Testing
@testable import FFAI

@Suite("FastVLM integration", .serialized)
struct FastVLMIntegrationTests {

    static let modelId = "mlx-community/FastVLM-0.5B-bf16"

    @Test("load — FastVLM checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        // The checkpoint is a VLM — vlModel is present, .visionIn is
        // available, and the text backbone is a Qwen2-0.5B engine.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.visionIn))
        #expect(m.engine.hidden == 896)          // Qwen2-0.5B text hidden
        #expect(m.engine.supportsEmbeddingInput) // VLM splice prerequisite

        let vlm = try #require(m.vlModel)
        // 1024px input: 4× stem + 4 stride-2 PEs → 16×16 = 256 tokens.
        #expect(vlm.imageTokenCount == 256)
        // Image placeholder token id is -200 (FastVLM processor convention).
        #expect(vlm.imageTokenId == -200)
    }

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "FastVLM checkpoint is not a VLM")

        // Build an image+text prompt.
        // FastVLM's chat template puts <image> at the start of the user
        // message; the processor replaces <image> with token id -200.
        // Here we assemble the token stream directly: 256 image placeholders
        // followed by a tokenized question.
        let imageTokenId = vlm.imageTokenId  // -200
        let questionTokens = m.tokenizer.encode(
            text: "What animal is in the image?\nAssistant:")
        let promptTokens = Array(repeating: imageTokenId,
                                 count: vlm.imageTokenCount) + questionTokens

        // A real photograph — the golden-retriever fixture.
        let image = try VLMTestSupport.dogImage()

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 64, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence first, then the content check.
        expectCoherentOutput(generated, minTokens: 4,
                             label: "FastVLM image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("FastVLM generated: \(text)")
        VLMTestSupport.expectMentionsDog(text, label: "FastVLM")
    }
}
