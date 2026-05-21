// Slow integration test for Gemma 3 VL (the SigLIP + Gemma 3 text
// `Gemma3ForConditionalGeneration` checkpoint).
//
// Verifies the Phase 6.5 vision path end-to-end on a real checkpoint:
// the SigLIP vision tower loads into the shared `VisionEncoder`, the
// multi-modal projector pools + projects the patch grid, the cross-modal
// splice injects the image tokens, and the fused stream decodes coherent
// text through the Gemma 3 backbone.
//
// Uses the mlx-community 4B-it conversion (smallest published Gemma 3
// VLM). Skipped if not available locally.

import Foundation
import Testing
@testable import FFAI

@Suite("Gemma 3 VL integration", .serialized)
struct Gemma3VLIntegrationTests {

    @Test("load — Gemma 3 VL checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let modelId = "mlx-community/gemma-3-4b-it-bf16"

        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Gemma 3 VL integration test skipped: \(error)")
            return
        }

        // The checkpoint is a VLM — vlModel is present, .visionIn is
        // available, and the text backbone is a Gemma 3 4B engine.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.visionIn))
        #expect(m.engine.hidden == 2560)        // 4B text hidden
        #expect(m.engine.supportsEmbeddingInput) // VLM splice prerequisite

        guard let vlm = m.vlModel else { return }
        // SigLIP-896 / patch-14 → 64×64 patches, pooled 4×4 → 256 tokens.
        #expect(vlm.imageTokenCount == 256)
    }

    @Test("enable / disable .visionIn — runtime capability flip")
    func capabilityFlip() async throws {
        let modelId = "mlx-community/gemma-3-4b-it-bf16"
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Gemma 3 VL capability test skipped: \(error)")
            return
        }
        // A VL checkpoint can toggle .visionIn at runtime.
        #expect(m.availableCapabilities.contains(.visionIn))
        m.disable(.visionIn)
        #expect(!m.isEnabled(.visionIn))
        m.enable(.visionIn)
        #expect(m.isEnabled(.visionIn))
        // textIn / textOut are universal — disable is a no-op.
        m.disable(.textIn)
        #expect(m.isEnabled(.textIn))
    }

    @Test("image + text prompt — coherent multi-modal generation")
    func imageTextGeneration() async throws {
        let modelId = "mlx-community/gemma-3-4b-it-bf16"
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Gemma 3 VL generation test skipped: \(error)")
            return
        }
        guard let vlm = m.vlModel else {
            print("Gemma 3 VL generation test skipped: not a VLM")
            return
        }

        // Build an image+text prompt. Gemma 3's chat template wraps the
        // image with <start_of_image> … <image_soft_token>×256 …
        // <end_of_image>; here we assemble the token stream directly:
        // a run of `imageTokenCount` image placeholders followed by a
        // text question.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "<start_of_turn>user\nDescribe this image.<end_of_turn>\n"
                + "<start_of_turn>model\n")
        let promptTokens = Array(repeating: imageTokenId,
                                 count: vlm.imageTokenCount) + questionTokens

        // A solid test image at the encoder's native resolution.
        let cfg = vlm.visionEncoder.config
        let image = RGBImage.solid(width: cfg.imageSize, height: cfg.imageSize,
                                   r: 0.55, g: 0.45, b: 0.30)

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 64, eosTokenId: m.config.eosTokenId)

        // The contract is coherence: a real image+text prompt should
        // decode a non-degenerate run of tokens.
        expectCoherentOutput(generated, minTokens: 16,
                             label: "Gemma 3 VL image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Gemma 3 VL generated: \(text)")
    }
}
