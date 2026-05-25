// Idefics3IntegrationTests — load the cached Idefics3-8B checkpoint and run
// end-to-end image captioning.
//
// Skipped automatically if the checkpoint is not available in the HF cache.
// DO NOT run this test in CI unless the Idefics3-8B checkpoint is cached; it
// loads multi-GB weights and runs the full vision encoder on every invocation.
//
// Pattern: load → check shapes → encode a test image → generate a caption →
// assert the output mentions "dog" (using dog.jpeg from the test Fixtures dir).
//
// The checkpoint path used is the mlx-community conversion:
//   mlx-community/Idefics3-8B-Llama3-bf16

import CoreImage
import Foundation
import Testing
@testable import FFAI

@Suite("Idefics3 8B integration", .serialized)
struct Idefics3IntegrationTests {

    /// Load dog.jpeg from the test Fixtures directory.
    /// Returns nil if the file cannot be found or decoded.
    /// Output format: [height, width, channels] Float32, normalized with
    /// mean=[0.5,0.5,0.5] std=[0.5,0.5,0.5] (SigLIP / CLIP convention).
    static func dogImagePixels(imageSize: Int) -> [Float]? {
        // Search using #filePath so the test can locate the fixture without a
        // bundle resource rule — the fixture may live under Tests/Fixtures/ or
        // Tests/ModelTests/Resources/ depending on the build.
        let candidates: [URL] = [
            // Tests/Fixtures/dog.jpeg (Package.swift copies this for ModelTests)
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()        // ModelTests/
                .deletingLastPathComponent()        // Tests/
                .appendingPathComponent("Fixtures/dog.jpeg"),
            // Tests/ModelTests/Resources/dog.jpeg (other worktrees)
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .appendingPathComponent("Resources/dog.jpeg"),
            // Absolute path for local development convenience
            URL(fileURLWithPath: NSHomeDirectory())
                .appendingPathComponent("Development/personal/ai/FFAI/Tests/ModelTests/Resources/dog.jpeg"),
        ]

        for url in candidates {
            guard FileManager.default.fileExists(atPath: url.path),
                  let ciImage = CIImage(contentsOf: url) else { continue }
            return normalizeAndResize(ciImage, to: imageSize)
        }
        return nil
    }

    /// Resize a CIImage to [imageSize × imageSize] and normalize to
    /// [height, width, channels] Float32 with (x - 0.5) / 0.5 = 2x - 1.
    private static func normalizeAndResize(_ ci: CIImage, to size: Int) -> [Float]? {
        let context    = CIContext()
        let targetSize = CGSize(width: size, height: size)
        let scaleX     = targetSize.width  / ci.extent.width
        let scaleY     = targetSize.height / ci.extent.height
        let scaled     = ci.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        let cropped    = scaled.cropped(to: CGRect(origin: .zero, size: targetSize))

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

        // Extract RGB channels, normalize to [-1, 1]: (x - 0.5) / 0.5 = 2x - 1
        // Layout: [height, width, channels] — HWC as expected by Idefics3Model.encodeImage
        var rgb = [Float](repeating: 0, count: size * size * 3)
        for i in 0..<(size * size) {
            rgb[i * 3 + 0] = rgba[i * 4 + 0] * 2.0 - 1.0  // R
            rgb[i * 3 + 1] = rgba[i * 4 + 1] * 2.0 - 1.0  // G
            rgb[i * 3 + 2] = rgba[i * 4 + 2] * 2.0 - 1.0  // B
        }
        return rgb
    }

    @Test("load Idefics3-8B + shape check + image encode + generate")
    func loadAndGenerate() async throws {
        // Match the rest of the VLM test suite: load by HF id so the
        // test auto-resolves the snapshot via HF (download on miss).
        // The previous `snapshotDir() / #require` pattern recorded a
        // failure when the local cache lacked the snapshot — but the
        // integration runner can pull it on demand. Falling back to
        // the local path resolver only when the HF load itself fails
        // would mask real load bugs; just use the id directly.
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/Idefics3-8B-Llama3-bf16")
        }

        // Verify the engine is an Idefics3Model
        let idefics3 = try #require(m.engine as? Idefics3Model,
                                    "expected engine to be Idefics3Model, got \(type(of: m.engine))")

        // ─── Shape checks (Idefics3-8B config values) ────────────────────────
        let tc = idefics3.cfg.textConfig
        let vc = idefics3.cfg.visionConfig

        // Llama-3-8B text backbone
        #expect(tc.hiddenSize        == 4096,   "text hidden_size")
        #expect(tc.numHiddenLayers   == 32,     "text num_hidden_layers")
        #expect(tc.numAttentionHeads == 32,     "text num_attention_heads")
        #expect(tc.numKeyValueHeads  == 8,      "text num_kv_heads")
        #expect(tc.headDim           == 128,    "text head_dim")
        #expect(tc.vocabSize         == 128259, "text vocab_size")

        // SigLIP-400M vision backbone
        #expect(vc.hiddenSize        == 1152, "vision hidden_size")
        #expect(vc.numHiddenLayers   == 27,   "vision num_hidden_layers")
        #expect(vc.numAttentionHeads == 16,   "vision num_attention_heads")
        #expect(vc.patchSize         == 14,   "vision patch_size")
        #expect(vc.imageSize         == 364,  "vision image_size")

        // Connector config
        #expect(idefics3.cfg.scaleFactor  == 2,     "scale_factor")
        #expect(idefics3.cfg.imageTokenId == 49153, "image_token_id")

        // ─── Pure text forward (sanity check without image) ──────────────────
        let caches   = m.engine.makeLayerCaches()
        let logits   = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        let topTokens = Sampling.topN(logits, n: 5)
        #expect(topTokens.count == 5)
        #expect(topTokens[0].1.isFinite)
        #expect(topTokens[0].1 != 0)

        // ─── Image encoding ───────────────────────────────────────────────────
        let pixels = try #require(Self.dogImagePixels(imageSize: vc.imageSize),
                                  "Idefics3 integration test: dog.jpeg fixture not found")

        let imageEmbeds = idefics3.encodeImage(
            pixels: pixels,
            height: vc.imageSize,
            width: vc.imageSize
        )

        // With scale_factor=2:
        //   nPatches = (364/14)² = 26² = 676
        //   nImageTokens = 676 / (2*2) = 169
        let nPatches      = (vc.imageSize / vc.patchSize) * (vc.imageSize / vc.patchSize)
        let sf2           = idefics3.cfg.scaleFactor * idefics3.cfg.scaleFactor
        let expectedImageTokens = nPatches / sf2
        #expect(imageEmbeds.count == expectedImageTokens * tc.hiddenSize,
                "image embeds count should be nImageTokens * textHidden")

        // ─── Generate a caption ───────────────────────────────────────────────
        // Build a minimal prompt: [BOS] <image×N> "Describe this image."
        // We use image_token_id (49153) as placeholders for the image patches.
        let imageTokenId    = idefics3.cfg.imageTokenId
        let imageTokenCount = expectedImageTokens

        // Construct token sequence: BOS + imageTokenId×N + text tokens
        let textTokens  = m.tokenizer.encode(text: "Describe this image.")
        var promptTokenIds = [1]  // BOS
        promptTokenIds += [Int](repeating: imageTokenId, count: imageTokenCount)
        promptTokenIds += textTokens

        // Prefill with image embeddings then check logits
        let freshCaches = m.engine.makeLayerCaches()
        let lastLogits  = idefics3.prefillWithImage(
            tokenIds: promptTokenIds,
            imageEmbeds: imageEmbeds,
            caches: freshCaches,
            device: Device.shared
        )

        // Verify we get finite, non-zero logits from the prefill
        let prefillTop = Sampling.topN(lastLogits, n: 5)
        #expect(prefillTop.count == 5)
        #expect(prefillTop[0].1.isFinite, "prefill logits should be finite")
        #expect(prefillTop[0].1 != 0, "prefill logits should be non-zero")

        // Decode a few tokens greedily
        var generatedTokens: [Int] = []
        var nextToken = Sampling.argmax(lastLogits)
        let eosId     = m.config.eosTokenId ?? 2
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

        // Verify the generated text is non-empty and semantically plausible.
        // For a dog image with "Describe this image." we expect some mention of
        // "dog" or related terms.
        #expect(!generatedText.isEmpty, "generated text should be non-empty")
        let lower = generatedText.lowercased()
        let containsDog = lower.contains("dog") || lower.contains("animal")
            || lower.contains("puppy") || lower.contains("canine")
            || lower.contains("pet")  || lower.contains("fur")
        #expect(containsDog,
                "expected generated text to contain 'dog' or related term, got: \(generatedText)")
    }
}
