// Slow integration test for SmolVLM2's *video* path
// (`SmolVLMForConditionalGeneration` checkpoint,
// HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx).
//
// Architecture: each video frame is encoded independently through the
// same SigLIP ViT + pixel-shuffle connector as a single image. SmolVLM2
// does NOT declare a separate `video_token_id` — the HF checkpoint
// config contains only `image_token_id` (49190 / `<image>`). Each frame
// uses `imageTokensPerFrame` (= nPatches / scaleFactor²) `<image>`
// placeholder tokens. The video prompt therefore contains
// `frameCount × imageTokensPerFrame` consecutive image placeholders,
// and `SmolVLM2Model.prefillWithImage` consumes the concatenated
// per-frame embeddings in order.
//
// The checkpoint is ~500M params (~1 GB). The snapshot must already be
// in the local HF cache — the test skips rather than downloading.
// DO NOT run via `make test-integration` unless the snapshot is cached.

import CoreImage
import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("SmolVLM2 Vision Video Integration", .serialized)
struct SmolVLM2VideoIntegrationTests {

    /// Resolve the local HF cache snapshot directory for SmolVLM2-500M.
    /// Returns nil if not found — the test is skipped when the snapshot
    /// is absent.
    static func snapshotDir() -> URL? {
        let base = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent(
                "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct-mlx")
            .appendingPathComponent("snapshots")
        return (try? FileManager.default.contentsOfDirectory(
            at: base, includingPropertiesForKeys: nil
        ))?.first
    }

    /// Number of evenly-spaced frames to pull from cat.mp4.
    /// SmolVLM2 has no temporal-patch folding requirement (each frame
    /// maps directly to a fixed token run), so any positive count is
    /// valid. 4 frames is a lightweight integration sweep.
    static let frameCount = 4

    /// Resize an `RGBImage` to `size × size` and apply SmolVLM2
    /// normalization (mean=0.5, std=0.5 → `pixel * 2 - 1`).
    /// Returns a flat HWC Float32 array (`size * size * 3` elements).
    private static func preprocessFrame(_ image: RGBImage,
                                         size: Int) -> [Float] {
        let resized = ImagePreprocessing.resize(
            image, targetW: size, targetH: size)
        // Normalize: (x - 0.5) / 0.5 = 2*x - 1
        return resized.pixels.map { $0 * 2.0 - 1.0 }
    }

    @Test("load — SmolVLM2 family declares videoIn capability")
    func loadCheckpoint() async throws {
        let dir = try #require(Self.snapshotDir(),
                               "SmolVLM2 snapshot not found in HF cache — skip")
        let m = try await Model.load(dir.path)
        // SmolVLM2Dense.availableCapabilities now includes .videoIn.
        // Note: Model.swift currently hardcodes .visionIn for SmolVLM2;
        // the family declaration is the ground truth checked here.
        #expect(SmolVLM2Dense.availableCapabilities.contains(.visionIn),
                "SmolVLM2 family should declare .visionIn")
        #expect(SmolVLM2Dense.availableCapabilities.contains(.videoIn),
                "SmolVLM2 family should declare .videoIn")
        let smol = try #require(m.smolVLM2,
                                "engine should be SmolVLM2Model")
        // imageTokensPerFrame = (512/16)² / 4² = 1024 / 16 = 64.
        #expect(smol.imageTokensPerFrame == 64,
                "imageTokensPerFrame should be 64 for the 500M config")
    }

    @Test("video + text prompt — describes the cat clip")
    func videoTextGeneration() async throws {
        let dir = try #require(Self.snapshotDir(),
                               "SmolVLM2 snapshot not found in HF cache — skip")
        let m = try await Model.load(dir.path)
        let smol = try #require(m.smolVLM2,
                                "engine should be SmolVLM2Model for this checkpoint")

        let vc = smol.cfg.visionConfig

        // Decode `frameCount` evenly-spaced frames from cat.mp4.
        let rawFrames = try VisionTestHelpers.catVideoFrames(maxFrames: Self.frameCount)

        // Convert each RGBImage → SmolVLM2 HWC Float32 (normalized).
        let pixelFrames: [[Float]] = rawFrames.map {
            Self.preprocessFrame($0, size: vc.imageSize)
        }

        // Encode all frames, producing a concatenated flat embedding
        // of shape [frameCount × imageTokensPerFrame × textHidden].
        let videoEmbeds = smol.encodeVideoFrames(
            frames: pixelFrames,
            height: vc.imageSize,
            width: vc.imageSize)

        let imageTokenId = smol.cfg.imageTokenId
        let tokensPerFrame = smol.imageTokensPerFrame
        let totalImageTokens = rawFrames.count * tokensPerFrame

        // SmolVLM2 does not use a separate video_token_id — each frame
        // contributes `tokensPerFrame` `<image>` (49190) placeholders.
        // Prompt structure mirrors the SmolVLM2 chat template:
        //   User: <image>…(×N) What's in this video?
        //   Assistant:
        let textBefore = m.tokenizer.encode(
            text: "User: ")
        let textAfter = m.tokenizer.encode(
            text: " What's in this video?\nAssistant:")

        let promptTokens = textBefore
            + Array(repeating: imageTokenId, count: totalImageTokens)
            + textAfter

        // Run prefill with the concatenated video frame embeddings,
        // then greedy-decode up to 200 tokens.
        let freshCaches = m.engine.makeLayerCaches()
        let lastLogits = smol.prefillWithImage(
            tokenIds: promptTokens,
            imageEmbeds: videoEmbeds,
            caches: freshCaches,
            device: .shared)

        // Verify the prefill produced finite, non-zero logits.
        let topK = Sampling.topN(lastLogits, n: 5)
        #expect(!topK.isEmpty, "prefill should produce logits")
        #expect(topK[0].1.isFinite, "top logit should be finite")
        #expect(topK[0].1 != 0, "top logit should be non-zero")

        // Greedy decode.
        var generatedTokens: [Int] = []
        var nextToken = Sampling.argmax(lastLogits)
        let eosId = m.config.eosTokenId ?? 2
        let stopSet: Set<Int> = Set(m.config.eosTokenIds)
            .union([eosId])

        for step in 0..<200 {
            if stopSet.contains(nextToken) { break }
            generatedTokens.append(nextToken)
            nextToken = m.engine.forwardSample(
                tokenId: nextToken,
                position: promptTokens.count + step,
                caches: freshCaches)
        }

        let text = m.tokenizer.decode(tokens: generatedTokens,
                                       skipSpecialTokens: true)
        print("SmolVLM2 video generated: \(text)")
        #expect(!text.isEmpty, "generated text should be non-empty")
        VisionTestHelpers.expectMentionsCat(text, label: "SmolVLM2 video")
    }
}
