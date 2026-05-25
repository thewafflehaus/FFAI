// GlmOcr integration test — THUDM's GLM-OCR vision-language model.
//
// Uses the mlx-community 4-bit quantized conversion:
//   `mlx-community/GLM-OCR-4bit`
//
// This test exercises the complete multi-modal path: loading the checkpoint,
// encoding the dog-photo fixture through the vision tower (CPU ViT), injecting
// the merged image tokens into the text-decode stream, and asserting the model
// outputs text that mentions a dog.
//
// The test is skipped automatically if the checkpoint is not cached locally
// (the Model.load catch path). It is never skipped on a hard infrastructure
// error — a load failure that is not a missing-checkpoint error will fail.
//
// DO NOT RUN `swift test` on this target — individual ModelTests load
// multi-GB checkpoints and must be run via `make test-integration`
// (serialized, --num-workers 1).

import Foundation
import Testing
#if canImport(CoreImage)
import CoreImage
import CoreGraphics
#endif
@testable import FFAI
import TestHelpers

@Suite("GlmOcr Vision Integration", .serialized)
struct GlmOcrIntegrationTests {

    static let modelId = "mlx-community/GLM-OCR-4bit"

    // ── Load the dog-photo fixture using CoreImage ────────────────────

    /// Load `Resources/dog.jpeg` into a `GlmOcrRGBImage`.
    ///
    /// The image is resized to 336×336 (GLM-OCR's standard input resolution)
    /// and normalized with the CLIP mean/std used by the GLM-OCR tower
    /// (mean=[0.48145466, 0.4578275, 0.40821073],
    ///  std=[0.26862954, 0.26130258, 0.27577711]).
    ///
    /// Throws `XCTSkip` if the fixture is missing (should never happen since
    /// the image is checked in, but guards against accidental deletion).
    static func dogImage(file: StaticString = #filePath) throws -> GlmOcrRGBImage {
        let url = URL(fileURLWithPath: "\(file)")
            .deletingLastPathComponent()
            .appendingPathComponent("Resources")
            .appendingPathComponent("dog.jpeg")

        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GlmOcrTestError.missingFixture(url.path)
        }

        // Decode the JPEG using CoreImage.
#if canImport(CoreImage)
        guard let ciImage = CIImage(contentsOf: url) else {
            throw GlmOcrTestError.decodeFailed(url.path)
        }
        let targetW = 336
        let targetH = 336
        // Scale to fill 336×336.
        let extent = ciImage.extent
        let scaleX = CGFloat(targetW) / extent.width
        let scaleY = CGFloat(targetH) / extent.height
        let scaled = ciImage
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        // Render via a CGContext to get raw RGBA bytes.
        let bytesPerRow = targetW * 4
        var rawBytes = [UInt8](repeating: 0, count: targetH * bytesPerRow)
        guard let ctx = CGContext(
            data: &rawBytes,
            width: targetW, height: targetH,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw GlmOcrTestError.decodeFailed(url.path)
        }
        let ciCtx = CIContext()
        guard let cgImg = ciCtx.createCGImage(scaled,
                                              from: CGRect(x: 0, y: 0,
                                                           width: targetW, height: targetH))
        else {
            throw GlmOcrTestError.decodeFailed(url.path)
        }
        ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: targetW, height: targetH))

        // CLIP normalization constants for GLM-OCR.
        let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
        let std:  [Float] = [0.26862954, 0.26130258, 0.27577711]

        // Convert RGBA UInt8 → normalized HWC Float32 (drop alpha channel).
        var pixels = [Float](repeating: 0, count: targetH * targetW * 3)
        for y in 0..<targetH {
            // CoreImage renders bottom-up but CGContext draws top-down here;
            // the row is already in the correct top-down order.
            let srcRow = y * bytesPerRow
            let dstRow = y * targetW * 3
            for x in 0..<targetW {
                let r = Float(rawBytes[srcRow + x * 4 + 0]) / 255.0
                let g = Float(rawBytes[srcRow + x * 4 + 1]) / 255.0
                let b = Float(rawBytes[srcRow + x * 4 + 2]) / 255.0
                pixels[dstRow + x * 3 + 0] = (r - mean[0]) / std[0]
                pixels[dstRow + x * 3 + 1] = (g - mean[1]) / std[1]
                pixels[dstRow + x * 3 + 2] = (b - mean[2]) / std[2]
            }
        }
        return GlmOcrRGBImage(data: pixels, height: targetH, width: targetW)
#else
        // Non-Apple platform: fall back to a solid-color stand-in.
        // (The test will still exercise the pipeline but won't produce
        // a meaningful caption.)
        return GlmOcrRGBImage.solid(width: 336, height: 336,
                                    r: 0.0, g: 0.0, b: 0.0)
#endif
    }

    // ── Tests ─────────────────────────────────────────────────────────

    @Test("load — GLM-OCR checkpoint loads and is recognised as GlmOcrModel")
    func loadGlmOcrCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        // Must be dispatched to GlmOcrModel.
        #expect(m.engine is GlmOcrModel, "expected engine to be GlmOcrModel")

        // Sanity-check text backbone shapes (from the cached config).
        #expect(m.engine.hidden == 1536)
        #expect(m.engine.nLayers == 16)
        #expect(m.engine.nHeads == 16)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)
        #expect(m.engine.vocab == 59392)

        // Checkpoint should declare 4-bit quantization.
        #expect(m.config.quantization?.bits == 4)
        #expect(m.config.quantization?.groupSize == 64)

        // Single-token forward pass should produce finite logits.
        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
    }

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        let model = try #require(m.engine as? GlmOcrModel, "expected GlmOcrModel engine")

        // Load the dog-photo fixture.
        let image = try GlmOcrIntegrationTests.dogImage()

        // Build the prompt. GLM-OCR expects the image tokens as sequential
        // `imageTokenId` placeholders, followed by a text instruction.
        // We use one image token per merged vision token; the tower will
        // expand each one with the encoded patch.
        //
        // The typical input format for a single-image describe task:
        //   <image_start><image_tokens…><image_end><user_request>
        //
        // For a coherence-first test we just use a flat sequence of image
        // token placeholders followed by the question tokens.
        let imageTokenId = model.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "\nDescribe this image in detail.")

        // How many image placeholders? Each vision patch is one token;
        // with 336×336, patch=14, merge=2 we get:
        //   gridH = gridW = 336/14 = 24 → 24×24 patches
        //   mergedH = mergedW = 24/2 = 12 → 144 merged tokens
        let gridHW = image.height / 14   // 24
        let mergedHW = gridHW / 2        // 12
        let numImageTokens = mergedHW * mergedHW  // 144
        let promptTokens = Array(repeating: imageTokenId,
                                 count: numImageTokens) + questionTokens

        let generated = model.generate(
            image: image, promptTokens: promptTokens,
            maxTokens: 200, device: Device.shared)

        // The model must produce at least one token.
        #expect(!generated.isEmpty)

        // Decode and print for human inspection.
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("GlmOcr generated: \(text)")

        // Coherence first — not stuck / degenerate.
        expectCoherentOutput(generated, minTokens: 4, label: "GlmOcr image+text")

        // Content check: the caption should mention a dog.
        let lowered = text.lowercased()
        #expect(lowered.contains("dog"),
                "GlmOcr caption should mention a dog — got: \(text)")
    }
}

// ── Local test error ─────────────────────────────────────────────────

private enum GlmOcrTestError: Error, CustomStringConvertible {
    case missingFixture(String)
    case decodeFailed(String)

    var description: String {
        switch self {
        case .missingFixture(let p): return "GlmOcrTests: fixture not found at \(p)"
        case .decodeFailed(let p):   return "GlmOcrTests: failed to decode image at \(p)"
        }
    }
}
