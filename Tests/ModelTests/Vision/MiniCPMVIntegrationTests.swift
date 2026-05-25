// Slow integration test for MiniCPM-V 4.6 (the SigLIP2-400M + Qwen3.5
// `MiniCPMV4_6ForConditionalGeneration` checkpoint).
//
// Verifies the v1 vision path end-to-end on the real `mlx-community/MiniCPM-V-4.6-4bit`
// checkpoint: SigLIP2 vision tower loads, position embedding is
// bilinearly resampled to the 32×32 runtime grid, `vit_merger` runs
// after encoder layer 6 (16× mode), the final `merger` projects into
// the Qwen 3.5 text hidden dim, the cross-modal splice injects 64
// vision tokens, and the fused stream decodes coherent text through the
// Qwen 3.5 backbone.
//
// The base checkpoint is ~9B params (~18 GB bf16) — skipped unless
// already cached locally.

import Foundation
import Testing
@testable import FFAI

@Suite("MiniCPMV Vision Integration", .serialized)
struct MiniCPMVIntegrationTests {

    @Test("load — MiniCPM-V 4.6 loads with vision capability")
    func loadVLCheckpoint() async throws {
        let modelId = "mlx-community/MiniCPM-V-4.6-4bit"
        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // The checkpoint is a VLM — vlModel is present, .visionIn is
        // available, and the text backbone is Qwen 3.5 (hidden 1024).
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.visionIn))
        #expect(m.engine.hidden == 1024)
        #expect(m.engine.supportsEmbeddingInput)
        // Qwen 3.5 is the text backbone (`qwen3_5_text` text_config).
        #expect(m.qwen35 != nil)

        let vlm = try #require(m.vlModel, "MiniCPM-V 4.6 checkpoint is not a VLM")
        // v1: 448×448 / patch-14 → 32×32 patches → vit_merger 16×16 →
        // merger 8×8 = 64 tokens per image (matches `query_num: 64`).
        #expect(vlm.imageTokenCount == 64)
        #expect(vlm.imageTokenId == MiniCPMV4_6.defaultImageTokenId)
        #expect(vlm.visionEncoder.config.imageSize == MiniCPMV4_6.runtimeImageSize)
        #expect(vlm.visionEncoder.config.patchSize == 14)
        #expect(vlm.visionEncoder.config.hidden == 1152)
        #expect(vlm.visionEncoder.config.nLayers == 27)
    }

    @Test("image + text prompt — coherent multi-modal generation")
    func imageTextGeneration() async throws {
        let modelId = "mlx-community/MiniCPM-V-4.6-4bit"
        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        let vlm = try #require(m.vlModel, "MiniCPM-V 4.6 checkpoint is not a VLM")

        // Build an image+text prompt. MiniCPM-V uses the Qwen-style chat
        // wrapper (<|im_start|>user … <|im_end|> <|im_start|>assistant)
        // around a run of <|image_pad|> placeholders, NO <|vision_start|>
        // / <|vision_end|> markers (the MiniCPM-V chat template only
        // emits <|image_pad|>; checkpoint config + chat_template.jinja
        // confirm this). Without the chat wrapping the model has no
        // signal that it's in assistant-response mode and falls into a
        // degenerate emit-only-space loop (token 220 = `Ġ`), which is
        // exactly how this test had been failing.
        let imageTokenId = vlm.imageTokenId
        let header = m.tokenizer.encode(text: "<|im_start|>user\n")
        let trailer = m.tokenizer.encode(
            text: "\nDescribe this image briefly.<|im_end|>\n"
                + "<|im_start|>assistant\n")
        let promptTokens = header
            + Array(repeating: imageTokenId, count: vlm.imageTokenCount)
            + trailer

        // A solid test image at the encoder's runtime resolution (448).
        let cfg = vlm.visionEncoder.config
        let image = RGBImage.solid(width: cfg.imageSize, height: cfg.imageSize,
                                   r: 0.55, g: 0.45, b: 0.30)

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence-only contract: a real image+text prompt should
        // decode a non-degenerate run of tokens.
        expectCoherentOutput(generated, minTokens: 16,
                             label: "MiniCPM-V 4.6 image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("MiniCPM-V 4.6 generated: \(text)")
    }
}
