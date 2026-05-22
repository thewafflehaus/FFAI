// Slow integration test for Qwen 3-VL-MoE (the Qwen3-VL ViT tower + the
// Qwen 3.5 mixture-of-experts hybrid text backbone, the
// `Qwen3VLMoeForConditionalGeneration` checkpoint).
//
// Verifies the vision path end-to-end on a real checkpoint: the
// Qwen3-VL vision tower loads, runs its full-attention + M-RoPE forward
// and patch-merger, the cross-modal splice injects the merged image
// tokens, and the fused stream decodes coherent text through the
// Qwen 3.5-MoE backbone (which now supports embedding-input forward for
// the splice).
//
// Uses the smallest published mlx-community Qwen3-VL-MoE conversion.
// Skipped if not available locally.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen 3-VL-MoE integration", .serialized)
struct Qwen3VLMoeIntegrationTests {

    static let modelId = "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"

    @Test("load — Qwen 3-VL-MoE checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially {
                try await Model.load(Self.modelId)
            }
        } catch {
            print("Qwen 3-VL-MoE integration test skipped: \(error)")
            return
        }

        // The checkpoint is a VLM — vlModel is present, .visionIn is
        // available, and the text backbone supports the splice.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.visionIn))
        #expect(m.engine.supportsEmbeddingInput)  // VLM splice prerequisite

        guard let vlm = m.vlModel else { return }
        // The vision tower contributes a positive run of merged tokens.
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
            print("Qwen 3-VL-MoE capability test skipped: \(error)")
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
            print("Qwen 3-VL-MoE generation test skipped: \(error)")
            return
        }
        guard let vlm = m.vlModel else {
            print("Qwen 3-VL-MoE generation test skipped: not a VLM")
            return
        }

        // Build an image+text prompt: a run of `imageTokenCount`
        // <|image_pad|> placeholders followed by a text question.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "<|im_start|>user\nDescribe this image.<|im_end|>\n"
                + "<|im_start|>assistant\n")
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
                             label: "Qwen 3-VL-MoE image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 3-VL-MoE generated: \(text)")
    }
}
