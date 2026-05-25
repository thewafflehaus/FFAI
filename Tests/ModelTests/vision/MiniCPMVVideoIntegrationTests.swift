// Slow integration test for MiniCPM-V 4.6's *video* path
// (`MiniCPMV4_6ForConditionalGeneration` checkpoint, mlx-community/MiniCPM-V-4.6-4bit).
//
// Architecture: each video frame is encoded independently through the
// same SigLIP2-400M + vit_merger + merger stack as a single image,
// producing `outputTokenCount` (64) merged tokens per frame. The
// per-frame token runs are concatenated and substituted into the
// prompt at every `<|video_pad|>` placeholder (token id 248057 —
// distinct from the image placeholder 248056).
//
// Unlike Qwen 2/2.5/3 VL (which fold frames into a temporal-patch
// axis at the conv stem), MiniCPM-V 4.6 has no temporal folding
// step — video token count grows linearly with frame count.
//
// The checkpoint is ~9B params (~18 GB bf16) — skipped unless already
// cached locally. DO NOT run via `make test-integration` unless the
// checkpoint is available: it loads multi-GB weights and runs the full
// vision encoder on every invocation.

import Foundation
import Testing
@testable import FFAI

@Suite("MiniCPM-V 4.6 video integration", .serialized)
struct MiniCPMVVideoIntegrationTests {

    /// The base openbmb checkpoint (non-quantized bf16). A future pass can
    /// add an mlx-community 4-bit conversion entry here once one ships.
    static let modelId = "mlx-community/MiniCPM-V-4.6-4bit"

    /// Number of evenly-spaced frames to pull from cat.mp4.  MiniCPM-V 4.6
    /// has no temporal-patch requirement (no folding step), so any positive
    /// frame count is valid. 4 frames is a lightweight integration sweep.
    static let frameCount = 4

    @Test("load — checkpoint reports video capability")
    func loadVLCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        // The MiniCPM-V 4.6 family loader threads video_token_id (248057)
        // through to VLModel.init; confirm it landed.
        let vlm = try #require(m.vlModel,
                               "MiniCPM-V 4.6 checkpoint is not a VLM")
        #expect(vlm.videoTokenId != nil,
                "video_token_id should be threaded through from config")
        #expect(vlm.videoTokenId == MiniCPMV4_6.defaultVideoTokenId,
                "video_token_id should be 248057 (<|video_pad|>)")
        #expect(vlm.imageTokenId == MiniCPMV4_6.defaultImageTokenId,
                "image_token_id should be 248056 (<|image_pad|>)")
        // Each frame produces 64 merged tokens (32×32 → vit_merger 16×16
        // → merger 8×8 = 64) for the v1 448×448 path.
        #expect(vlm.imageTokenCount == 64,
                "imageTokenCount should be 64 for the v1 single-tile path")
    }

    @Test("video + text prompt — describes the cat clip")
    func videoTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel,
                               "MiniCPM-V 4.6 checkpoint is not a VLM")
        let videoTokenId = try #require(vlm.videoTokenId,
                                        "video_token_id must be set for video generation")

        // Decode `frameCount` evenly-spaced frames from cat.mp4.
        let frames = try VLMTestSupport.catVideoFrames(maxFrames: Self.frameCount)

        // MiniCPM-V 4.6 video token count: frameCount × outputTokenCount
        // (no temporal folding — each frame maps 1:1 to a token run).
        // The encoder's imageTokenCount is the per-frame merged count.
        let videoTokenCount = frames.count * vlm.imageTokenCount

        // Build the MiniCPM-V chat-template prompt for a video query.
        // The chat template emits one <|video_pad|> placeholder per token;
        // we inline the expanded run directly (the Python side does the
        // same expansion before passing to the tokenizer).
        // Prompt structure:
        //   <|im_start|>user\n
        //   <|video_pad|> × videoTokenCount
        //   \nWhat's in this video?<|im_end|>\n
        //   <|im_start|>assistant\n
        let header = m.tokenizer.encode(text: "<|im_start|>user\n")
        let trailer = m.tokenizer.encode(
            text: "\nWhat's in this video?<|im_end|>\n"
                + "<|im_start|>assistant\n")
        let promptTokens = header
            + Array(repeating: videoTokenId, count: videoTokenCount)
            + trailer

        let generated = try vlm.generate(
            promptTokens: promptTokens,
            videoFrames: frames,
            maxTokens: 200,
            eosTokenId: m.config.eosTokenId,
            eosTokenIds: m.config.eosTokenIds)

        // Coherence first — the model should produce a non-degenerate run.
        expectCoherentOutput(generated, minTokens: 8,
                             label: "MiniCPM-V 4.6 video+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("MiniCPM-V 4.6 video generated: \(text)")
        VLMTestSupport.expectMentionsCat(text, label: "MiniCPM-V 4.6 video")
    }
}
