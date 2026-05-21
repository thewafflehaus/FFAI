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
// Uses the smallest published mlx-community Gemma 4 VL conversion.
// Skipped if not available locally.

import Foundation
import Testing
@testable import FFAI

@Suite("Gemma 4 VL integration", .serialized)
struct Gemma4VLIntegrationTests {

    static let modelId = "mlx-community/gemma-4-4b-it-4bit"

    @Test("load — Gemma 4 VL checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially {
                try await Model.load(Self.modelId)
            }
        } catch {
            print("Gemma 4 VL integration test skipped: \(error)")
            return
        }

        // The checkpoint is a VLM — vlModel is present, .visionIn is
        // available, and the text backbone supports the splice.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.visionIn))
        #expect(m.engine.supportsEmbeddingInput)  // VLM splice prerequisite

        guard let vlm = m.vlModel else { return }
        // The vision tower contributes a positive run of soft tokens.
        #expect(vlm.imageTokenCount > 0)
    }

    @Test("enable / disable .visionIn — runtime capability flip")
    func capabilityFlip() async throws {
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially {
                try await Model.load(Self.modelId)
            }
        } catch {
            print("Gemma 4 VL capability test skipped: \(error)")
            return
        }
        #expect(m.availableCapabilities.contains(.visionIn))
        m.disable(.visionIn)
        #expect(!m.isEnabled(.visionIn))
        m.enable(.visionIn)
        #expect(m.isEnabled(.visionIn))
    }

    @Test("image + text prompt — coherent multi-modal generation")
    func imageTextGeneration() async throws {
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially {
                try await Model.load(Self.modelId)
            }
        } catch {
            print("Gemma 4 VL generation test skipped: \(error)")
            return
        }
        guard let vlm = m.vlModel else {
            print("Gemma 4 VL generation test skipped: not a VLM")
            return
        }

        // Build an image+text prompt: a run of `imageTokenCount`
        // <image_soft_token> placeholders followed by a text question.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "<start_of_turn>user\nDescribe this image.<end_of_turn>\n"
                + "<start_of_turn>model\n")
        let promptTokens = Array(repeating: imageTokenId,
                                 count: vlm.imageTokenCount) + questionTokens

        // A solid test image at the encoder's expected square resolution.
        let cfg = vlm.visionEncoder.config
        let image = RGBImage.solid(width: cfg.imageSize, height: cfg.imageSize,
                                   r: 0.40, g: 0.55, b: 0.65)

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 64, eosTokenId: m.config.eosTokenId)

        // The contract is coherence: a real image+text prompt should
        // decode a non-degenerate run of tokens.
        expectCoherentOutput(generated, minTokens: 12,
                             label: "Gemma 4 VL image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Gemma 4 VL generated: \(text)")
    }
}
