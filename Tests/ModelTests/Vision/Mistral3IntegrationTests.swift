// Mistral3 integration test — Mistral AI's Mistral Small 3.1
// vision-language model (`mistral3` model_type,
// `Mistral3ForConditionalGeneration` architecture).
//
// Uses the mlx-community 4-bit quantized conversion:
//   `mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit`
//
// The checkpoint is cached locally at the time this test was written
// (2026-05-23) — the suite runs without the `.disabled` tag.
//
// DO NOT RUN `swift test` on this target — individual ModelTests load
// multi-GB checkpoints and must be run via `make test-integration`
// (serialized, --num-workers 1).

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Mistral3 Vision Integration", .serialized)
struct Mistral3IntegrationTests {

    static let modelId = "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit"

    @Test("load — Mistral3 checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        // The checkpoint is a VLM — vlModel is present, .visionIn is
        // available, and the text backbone supports the splice.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.visionIn))
        #expect(m.engine.supportsEmbeddingInput)

        let vlm = try #require(m.vlModel)
        // The vision tower contributes a positive run of merged patch tokens.
        #expect(vlm.imageTokenCount > 0)
        // For the mlx-community Mistral-Small-3.1 4bit conversion with
        // image_size=1540, patchSize=14, spatialMergeSize=2:
        //   110×110 patches → 55×55 = 3025 merged tokens.
        #expect(vlm.imageTokenCount == 3025)
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
        let vlm = try #require(m.vlModel, "Mistral3 checkpoint is not a VLM")

        // Build an image+text prompt. Mistral3 uses the `[IMG]` special
        // token (id = 10 by default) as the image placeholder in its
        // prompt format; one placeholder per merged patch.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "\nDescribe this image.")
        let promptTokens = Array(repeating: imageTokenId,
                                 count: vlm.imageTokenCount) + questionTokens

        // The golden-retriever fixture shared by all VLM tests.
        let image = try VisionTestHelpers.dogImage()

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId)

        // Coherence gate: the model must produce at least 8 tokens of
        // non-trivial text.
        expectCoherentOutput(generated, minTokens: 8,
                             label: "Mistral3 image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Mistral3 generated: \(text)")
        // Content gate: the caption should mention a dog.
        VisionTestHelpers.expectMentionsDog(text, label: "Mistral3")
    }
}
