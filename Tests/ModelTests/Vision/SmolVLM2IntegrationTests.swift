// SmolVLM2IntegrationTests — load the cached SmolVLM2-500M checkpoint and
// run end-to-end image captioning.
//
// Skipped automatically if the checkpoint is not available in the HF cache.
// DO NOT run this test in CI unless the 500M checkpoint is cached; it loads
// multi-GB weights and runs the full vision encoder on every invocation.
//
// Pattern: load → check shapes → encode a test image → generate a caption →
// assert the output mentions "dog" (using dog.jpeg from the Fixtures dir).
//
// The checkpoint path used is the MLX-format snapshot:
//   HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx
// Now also covers the video path for the same checkpoint (merged from the
// prior SmolVLM2VideoIntegrationTests.swift).

import CoreImage
import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("SmolVLM2 Vision Integration (image + video)", .serialized)
struct SmolVLM2IntegrationTests {

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

    /// Resolve the HF cache snapshot directory for SmolVLM2-500M.
    /// Returns nil if not found locally (test will be skipped).
    static func snapshotDir() -> URL? {
        // Primary: the MLX-format snapshot downloaded during development
        let primary = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent("models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct-mlx")
            .appendingPathComponent("snapshots")
        if let snap = try? FileManager.default.contentsOfDirectory(
            at: primary, includingPropertiesForKeys: nil
        ).first {
            return snap
        }
        return nil
    }

    /// Load dog.jpeg from the test Fixtures directory.
    static func dogImagePixels(imageSize: Int) -> [Float]? {
        // The Fixtures directory is bundled alongside ModelTests via .copy("../Fixtures")
        guard let bundle = Bundle.allBundles.first(where: {
            $0.bundlePath.contains("ModelTests")
        }) else { return nil }
        let fixturesURL = bundle.resourceURL?
            .appendingPathComponent("Fixtures")
            .appendingPathComponent("dog.jpeg")
            ?? URL(fileURLWithPath: NSHomeDirectory())
                .appendingPathComponent("Development/personal/ai/FFAI/Tests/Fixtures/dog.jpeg")

        guard FileManager.default.fileExists(atPath: fixturesURL.path),
              let ciImage = CIImage(contentsOf: fixturesURL) else {
            // Fallback: look for the fixture relative to the project root
            let fallback = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()   // ModelTests/
                .deletingLastPathComponent()   // Tests/
                .appendingPathComponent("Fixtures/dog.jpeg")
            guard FileManager.default.fileExists(atPath: fallback.path),
                  let ci = CIImage(contentsOf: fallback) else {
                return nil
            }
            return normalizeAndResize(ci, to: imageSize)
        }
        return normalizeAndResize(ciImage, to: imageSize)
    }

    /// Resize a CIImage to [imageSize x imageSize] and normalize with ImageNet-like stats.
    /// Returns [height, width, channels] Float32 array.
    ///
    /// SmolVLM2 normalization: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
    private static func normalizeAndResize(_ ci: CIImage, to size: Int) -> [Float]? {
        let context = CIContext()
        let targetSize = CGSize(width: size, height: size)
        let scaleX = targetSize.width / ci.extent.width
        let scaleY = targetSize.height / ci.extent.height
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        let cropped = scaled.cropped(to: CGRect(origin: .zero, size: targetSize))

        // Render to a Float32 RGBA bitmap
        var rgba = [Float](repeating: 0, count: size * size * 4)
        rgba.withUnsafeMutableBytes { ptr in
            context.render(cropped,
                           toBitmap: ptr.baseAddress!,
                           rowBytes: size * 4 * MemoryLayout<Float>.size,
                           bounds: CGRect(origin: .zero, size: targetSize),
                           format: .RGBAf,
                           colorSpace: CGColorSpaceCreateDeviceRGB())
        }

        // Extract RGB, normalize: (x - 0.5) / 0.5 = 2*x - 1
        var rgb = [Float](repeating: 0, count: size * size * 3)
        for i in 0..<(size * size) {
            rgb[i * 3 + 0] = rgba[i * 4 + 0] * 2.0 - 1.0  // R
            rgb[i * 3 + 1] = rgba[i * 4 + 1] * 2.0 - 1.0  // G
            rgb[i * 3 + 2] = rgba[i * 4 + 2] * 2.0 - 1.0  // B
        }
        return rgb
    }

    @Test("load SmolVLM2-500M + shape check + image encode + generate")
    func loadAndGenerate() async throws {
        let dir = try #require(Self.snapshotDir(),
                               "SmolVLM2 checkpoint not found in HF cache")

        // ─── Load model ──────────────────────────────────────────────────────
        let m = try await Model.load(dir.path)

        // Verify the engine is a SmolVLM2Model
        let smolVLM2 = try #require(m.smolVLM2,
                                    "expected engine to be SmolVLM2Model, got \(type(of: m.engine))")

        // ─── Shape checks (500M config values) ───────────────────────────────
        let tc = smolVLM2.cfg.textConfig
        let vc = smolVLM2.cfg.visionConfig

        #expect(tc.hiddenSize        == 960,   "text hidden_size")
        #expect(tc.numHiddenLayers   == 32,    "text num_hidden_layers")
        #expect(tc.numAttentionHeads == 15,    "text num_attention_heads")
        #expect(tc.numKeyValueHeads  == 5,     "text num_kv_heads")
        #expect(tc.headDim           == 64,    "text head_dim")
        #expect(tc.vocabSize         == 49280, "text vocab_size")

        #expect(vc.hiddenSize        == 768,  "vision hidden_size")
        #expect(vc.numHiddenLayers   == 12,   "vision num_hidden_layers")
        #expect(vc.numAttentionHeads == 12,   "vision num_attention_heads")
        #expect(vc.patchSize         == 16,   "vision patch_size")
        #expect(vc.imageSize         == 512,  "vision image_size")

        #expect(smolVLM2.cfg.scaleFactor   == 4,     "scale_factor")
        #expect(smolVLM2.cfg.imageTokenId  == 49190, "image_token_id")

        // ─── Pure text forward (sanity check without image) ──────────────────
        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        let topTokens = Sampling.topN(logits, n: 5)
        #expect(topTokens.count == 5)
        #expect(topTokens[0].1.isFinite)
        #expect(topTokens[0].1 != 0)

        // ─── Image encoding ───────────────────────────────────────────────────
        let pixels = try #require(Self.dogImagePixels(imageSize: vc.imageSize),
                                  "SmolVLM2 integration test: dog.jpeg fixture not found")

        let imageEmbeds = smolVLM2.encodeImage(
            pixels: pixels,
            height: vc.imageSize,
            width: vc.imageSize
        )

        // Scale factor 4 → numImageTokens = numPatches / 16
        // numPatches = (512/16)² = 1024, numImageTokens = 64
        let expectedImageTokens = (vc.imageSize / vc.patchSize) * (vc.imageSize / vc.patchSize)
            / (smolVLM2.cfg.scaleFactor * smolVLM2.cfg.scaleFactor)
        #expect(imageEmbeds.count == expectedImageTokens * tc.hiddenSize,
                "image embeds count should be nImageTokens * textHidden")

        // ─── Generate a caption ───────────────────────────────────────────────
        // Build a minimal prompt: [BOS] <image...×64> "Describe this image."
        // We use image_token_id (49190) as placeholders for the image patches.
        // The tokenizer is needed for a proper chat template; here we use raw
        // token encoding as a minimal integration check.
        let imageTokenId = smolVLM2.cfg.imageTokenId
        let imageTokenCount = expectedImageTokens

        // Construct token sequence: BOS + imageTokenId×N + text tokens
        // "Describe this image." — we just run a short prompt for coherence check
        let textTokens = m.tokenizer.encode(text: "Describe this image.")
        var promptTokenIds = [1]  // BOS
        promptTokenIds += [Int](repeating: imageTokenId, count: imageTokenCount)
        promptTokenIds += textTokens

        // Prefill with image embeddings + generate short sequence
        let freshCaches = m.engine.makeLayerCaches()
        let lastLogits = smolVLM2.prefillWithImage(
            tokenIds: promptTokenIds,
            imageEmbeds: imageEmbeds,
            caches: freshCaches,
            device: .shared
        )

        // Verify we get finite, non-zero logits from the prefill
        let prefillTop = Sampling.topN(lastLogits, n: 5)
        #expect(prefillTop.count == 5)
        #expect(prefillTop[0].1.isFinite, "prefill logits should be finite")
        #expect(prefillTop[0].1 != 0, "prefill logits should be non-zero")

        // Decode a few tokens
        var generatedTokens: [Int] = []
        var nextToken = Sampling.argmax(lastLogits)
        let eosId = m.config.eosTokenId ?? 2
        for step in 0..<32 {
            if nextToken == eosId { break }
            generatedTokens.append(nextToken)
            nextToken = m.engine.forwardSample(
                tokenId: nextToken,
                position: promptTokenIds.count + step,
                caches: freshCaches
            )
        }

        let generatedText = m.tokenizer.decode(tokens: generatedTokens,
                                                skipSpecialTokens: true)

        // Verify the generated text is non-empty and contains something coherent.
        // For a dog image with "Describe this image." we expect mention of "dog"
        // or at least some animal-related content.
        #expect(!generatedText.isEmpty, "generated text should be non-empty")
        let lower = generatedText.lowercased()
        let containsDog = lower.contains("dog") || lower.contains("animal")
            || lower.contains("puppy") || lower.contains("canine")
            || lower.contains("pet")
        #expect(containsDog,
                "expected generated text to contain 'dog' or related term, got: \(generatedText)")
    }

    // ─── Video ───────────────────────────────────────────────────────────

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
