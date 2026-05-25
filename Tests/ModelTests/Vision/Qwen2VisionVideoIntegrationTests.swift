// Slow integration test for Qwen 2-VL's *video* path
// (`Qwen2VLForConditionalGeneration` checkpoint, 2B-Instruct-4bit).
//
// Verifies the end-to-end multi-frame video-inference path on a real
// checkpoint: the Qwen 2-VL vision tower loads, runs its full-attention
// + M-RoPE forward over a stack of preprocessed frames with temporal-
// patch folding (each `temporal_patch_size` consecutive frames fold into
// one temporal patch), the cross-modal splice injects the merged video
// tokens at every `<|video_pad|>` placeholder, and the fused stream
// decodes coherent text through the Qwen 2 backbone.
//
// The checkpoint download is ~1.5 GB; this test is gated by the standard
// `ModelLoadLock` serialization and run only via `make test-integration`.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Qwen2 Vision Video Integration", .serialized)
struct Qwen2VisionVideoIntegrationTests {

    /// The smallest published Qwen 2-VL conversion — 2B-Instruct-4bit
    /// from mlx-community. Same checkpoint used by the image integration
    /// test.
    static let modelId = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

    /// Number of evenly-spaced frames to pull out of `cat.mp4`. Must be
    /// a multiple of `temporal_patch_size` (2) so the vision tower
    /// doesn't have to pad with the last frame — keeps the placeholder
    /// arithmetic exact.
    static let frameCount = 4

    @Test("load — checkpoint reports .videoIn capability")
    func loadVLCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        // The Qwen 2-VL family now declares text + vision + video.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.visionIn))
        #expect(m.availableCapabilities.contains(.videoIn))

        let vlm = try #require(m.vlModel)
        // The video splice needs the family loader to thread
        // `video_token_id` through to VLModel.init.
        #expect(vlm.videoTokenId != nil)
    }

    @Test("video + text prompt — describes the cat clip")
    func videoTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "Qwen 2-VL checkpoint is not a VLM")
        let videoTokenId = try #require(vlm.videoTokenId)

        // Pull `frameCount` evenly-spaced frames from cat.mp4.
        let frames = try VisionTestHelpers.catVideoFrames(maxFrames: Self.frameCount)

        // The merged-token-per-temporal-patch count is the same as one
        // image's merged token count. With `temporal_patch_size` = 2 and
        // 4 frames, the splice substitutes 2 × `imageTokenCount` rows.
        let temporalPatchSize = 2
        let videoTokenCount = (frames.count / temporalPatchSize) * vlm.imageTokenCount

        // Build the standard Qwen 2-VL video prompt:
        //   <|im_start|>user\n<|vision_start|><|video_pad|>...<|vision_end|>What's in this video?<|im_end|>\n
        //   <|im_start|>assistant\n
        let preTokens = m.tokenizer.encode(
            text: "<|im_start|>user\n<|vision_start|>")
        let postTokens = m.tokenizer.encode(
            text: "<|vision_end|>What's in this video?<|im_end|>\n"
                + "<|im_start|>assistant\n")
        let promptTokens = preTokens
            + Array(repeating: videoTokenId, count: videoTokenCount)
            + postTokens

        let generated = try vlm.generate(
            promptTokens: promptTokens, videoFrames: frames,
            maxTokens: 200,
            eosTokenId: m.config.eosTokenId,
            eosTokenIds: m.config.eosTokenIds)

        // Coherence first, then the content check: the caption should
        // mention a cat (or kitten — model verbosity varies).
        expectCoherentOutput(generated, minTokens: 8,
                             label: "Qwen 2-VL video+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 2-VL video generated: \(text)")
        VisionTestHelpers.expectMentionsCat(text, label: "Qwen 2-VL video")
    }
}
