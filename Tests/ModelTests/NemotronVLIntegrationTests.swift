// Slow integration test for Nemotron-VLM (NVIDIA's Nemotron Nano VL —
// a ViT tower + multi-modal projector + the NemotronH stack-interleaved
// hybrid text backbone).
//
// Verifies the vision path end-to-end on a real checkpoint: the ViT
// vision tower loads into the shared `VisionEncoder`, runs its
// bidirectional-attention forward, the multi-modal projector maps the
// patch tokens into the NemotronH text hidden dim, the cross-modal
// splice injects them, and the fused stream decodes coherent text
// through the NemotronH backbone (which now supports embedding-input
// forward for the splice).
//
// DISABLED — no loadable checkpoint exists. The FFAI `NemotronVL` loader
// expects an mlx-style conversion: a top-level `vision_config` plus a
// `text_config` whose `model_type` is `nemotron_h`, with text weights
// under the `language_model.*` prefix (the layout `dispatchAndLoad`'s
// `isNemotronVisionLanguage` detects). As of the May 2026 HF index:
//   * `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` ships only a raw
//     PyTorch checkpoint driven by a custom remote-code `configuration.py`
//     / `modeling` pair (RADIO vision tower) — its `config.json` does not
//     expose the standard `text_config.model_type == nemotron_h` fields,
//     so FFAI cannot route or load it.
//   * No `mlx-community` (or other) MLX conversion of any Nemotron Nano
//     VL / Nemotron-12B-v2-VL checkpoint has been published.
// Re-enable once an mlx-style Nemotron-VL conversion exists in the HF
// cache (and update `modelId` to it).

import Foundation
import Testing
@testable import FFAI

// Reason the suite is disabled — kept as a named `Comment` constant so
// the long explanation does not blow up the `@Suite` macro's
// type-checker.
private let nemotronVLDisabledReason = Comment(rawValue:
    "No mlx-style Nemotron-VL checkpoint exists: the only published "
    + "Nemotron Nano VL is nvidia's raw PyTorch remote-code checkpoint, "
    + "which lacks the text_config.model_type == nemotron_h layout the "
    + "FFAI NemotronVL loader requires; no mlx-community conversion has "
    + "been released.")

@Suite("Nemotron-VLM integration", .serialized,
       .disabled(nemotronVLDisabledReason))
struct NemotronVLIntegrationTests {

    static let modelId = "mlx-community/Llama-3.1-Nemotron-Nano-VL-8B-V1-4bit"

    @Test("load — Nemotron-VLM checkpoint loads with vision capability")
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
        // The vision tower contributes a positive run of projected tokens.
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
        let vlm = try #require(m.vlModel, "Nemotron-VLM checkpoint is not a VLM")

        // Build an image+text prompt: a run of `imageTokenCount`
        // image-placeholder tokens followed by a text question.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(text: "Describe this image.")
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
                             label: "Nemotron-VLM image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Nemotron-VLM generated: \(text)")
        VLMTestSupport.expectMentionsDog(text, label: "Nemotron-VLM")
    }
}
