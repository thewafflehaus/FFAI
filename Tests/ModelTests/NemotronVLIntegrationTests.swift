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
// Uses the smallest published mlx-community Nemotron Nano VL conversion.
// Skipped if not available locally.

import Foundation
import Testing
@testable import FFAI

@Suite("Nemotron-VLM integration", .serialized)
struct NemotronVLIntegrationTests {

    static let modelId = "mlx-community/Llama-3.1-Nemotron-Nano-VL-8B-V1-4bit"

    @Test("load — Nemotron-VLM checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially {
                try await Model.load(Self.modelId)
            }
        } catch {
            print("Nemotron-VLM integration test skipped: \(error)")
            return
        }

        // The checkpoint is a VLM — vlModel is present, .visionIn is
        // available, and the text backbone supports the splice.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.visionIn))
        #expect(m.engine.supportsEmbeddingInput)  // VLM splice prerequisite

        guard let vlm = m.vlModel else { return }
        // The vision tower contributes a positive run of projected tokens.
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
            print("Nemotron-VLM capability test skipped: \(error)")
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
            print("Nemotron-VLM generation test skipped: \(error)")
            return
        }
        guard let vlm = m.vlModel else {
            print("Nemotron-VLM generation test skipped: not a VLM")
            return
        }

        // Build an image+text prompt: a run of `imageTokenCount`
        // image-placeholder tokens followed by a text question.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(text: "Describe this image.")
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
                             label: "Nemotron-VLM image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Nemotron-VLM generated: \(text)")
    }
}
