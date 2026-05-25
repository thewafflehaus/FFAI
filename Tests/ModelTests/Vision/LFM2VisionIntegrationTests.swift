// Slow integration test for LFM2-VL (the SigLIP2 + LFM2
// `Lfm2VlForConditionalGeneration` checkpoint).
//
// Verifies the end-to-end vision path on a real checkpoint:
// the SigLIP2 vision tower loads into the shared `VisionEncoder`, the
// pixel-unshuffle + MLP projector reduces and projects the 256 ViT tokens
// to 64 super-patch embeddings, the cross-modal splice injects the image
// tokens, and the fused stream decodes coherent text through the LFM2
// hybrid text backbone (short-conv / attention layers).
//
// Uses the mlx-community 4-bit conversion (LFM2-VL-1.6B-4bit). The
// checkpoint MUST load — a load failure fails the test.
//
// DO NOT RUN — heavyweight test, requires the HuggingFace checkpoint at
// ~/.cache/huggingface/hub/models--mlx-community--LFM2-VL-1.6B-4bit/.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("LFM2 Vision Integration", .serialized)
struct LFM2VisionIntegrationTests {

    static let modelId = "mlx-community/LFM2-VL-1.6B-4bit"

    @Test("load — LFM2-VL checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        // The checkpoint is a VLM — vlModel is present, .visionIn is
        // available, and the text backbone is an LFM2 hybrid engine.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.visionIn))
        #expect(m.engine.hidden == 2048)         // LFM2-1.6B text hidden
        #expect(m.engine.supportsEmbeddingInput) // VLM splice prerequisite

        let vlm = try #require(m.vlModel)
        // SigLIP2-256-patch-16 → 256 patches, pixel-unshuffle(2) → 64 tokens.
        #expect(vlm.imageTokenCount == 64)
        #expect(vlm.imageTokenId == 396)
    }

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "LFM2-VL checkpoint is not a VLM")

        // Build a minimal image+text prompt. 64 image-placeholder tokens
        // followed by a short instruction (no chat template — keeping it
        // simple for the coherence-first integration test).
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "Describe this image.")
        let promptTokens = Array(repeating: imageTokenId,
                                 count: vlm.imageTokenCount) + questionTokens

        // The shared golden-retriever fixture.
        let image = try VisionTestHelpers.dogImage()

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence first: at least 8 tokens generated, no degenerate
        // repeats — then the content check.
        expectCoherentOutput(generated, minTokens: 8,
                             label: "LFM2-VL image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("LFM2-VL generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "LFM2-VL")
    }
}
