// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// GlmOcr integration test — THUDM's GLM-OCR vision-language model.
//
// Uses the mlx-community 4-bit quantized conversion:
//   `mlx-community/GLM-OCR-4bit`
//
// This test exercises the complete multi-modal path: loading the checkpoint,
// encoding the OCR fixture (`Tests/Resources/testocr.png`) through the
// vision tower (CPU ViT), injecting the merged image tokens into the
// text-decode stream, and asserting the model recovers the printed
// passage from the image. GLM-OCR is purpose-built for OCR; this is
// the right shape of correctness signal for the family.
//
// The test is skipped automatically if the checkpoint is not cached locally
// (the Model.load catch path). It is never skipped on a hard infrastructure
// error — a load failure that is not a missing-checkpoint error will fail.
//
// DO NOT RUN `swift test` on this target — individual ModelTests load
// multi-GB checkpoints and must be run via `make test-integration`
// (serialized, --num-workers 1).

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "GlmOcr Vision Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct GlmOcrIntegrationTests {

    static let modelId = "mlx-community/GLM-OCR-4bit"

    // ── Load the OCR fixture (testocr.png) ────────────────────────────

    /// Load `Resources/testocr.png` resized to 336×336 (GLM-OCR's
    /// standard input resolution) and CLIP-normalised, returned in the
    /// HWC layout `GlmOcrRGBImage` expects.
    static func ocrImage() throws -> GlmOcrRGBImage {
        let pixels = try VisionTestHelpers.ocrTestImageHWCNormalized(
            targetSize: 336, normalization: .clip)
        return GlmOcrRGBImage(data: pixels, height: 336, width: 336)
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

    @Test("image + text prompt — transcribes the testocr.png passage")
    func imageTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        let model = try #require(m.engine as? GlmOcrModel, "expected GlmOcrModel engine")

        // Load the OCR test fixture (printed black-on-white passage).
        let image = try GlmOcrIntegrationTests.ocrImage()

        // Build the prompt. GLM-OCR expects the image tokens as sequential
        // `imageTokenId` placeholders, followed by a text instruction.
        // We use one image token per merged vision token; the tower will
        // expand each one with the encoded patch.
        //
        // The typical input format for a single-image OCR task:
        //   <image_start><image_tokens…><image_end><user_request>
        //
        // For a coherence-first test we just use a flat sequence of image
        // token placeholders followed by the OCR instruction.
        let imageTokenId = model.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "\nTranscribe the text in this image exactly.")

        // How many image placeholders? Each vision patch is one token;
        // with 336×336, patch=14, merge=2 we get:
        //   gridH = gridW = 336/14 = 24 → 24×24 patches
        //   mergedH = mergedW = 24/2 = 12 → 144 merged tokens
        let gridHW = image.height / 14  // 24
        let mergedHW = gridHW / 2  // 12
        let numImageTokens = mergedHW * mergedHW  // 144
        let promptTokens =
            Array(
                repeating: imageTokenId,
                count: numImageTokens) + questionTokens

        // 80 tokens is enough to recover the leading "This is a lot of
        // 12 point text" phrase from `testocr.png` — the OCR signal we
        // assert on. Capping at 80 keeps the bisect under the 500 s
        // per-suite timeout (the CPU SigLIP-style vision tower + a
        // 200-token text decode together ran past the timeout).
        let generated = model.generate(
            image: image, promptTokens: promptTokens,
            maxTokens: 80, device: Device.shared)

        // The model must produce at least one token.
        #expect(!generated.isEmpty)

        // Decode and print for human inspection.
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("GlmOcr generated: \(text)")

        // Coherence first — not stuck / degenerate.
        expectCoherentOutput(generated, minTokens: 4, label: "GlmOcr image+text")

        // Content check: recover at least two phrases from the printed
        // passage. testocr.png contains "This is a lot of 12 point text…
        // The quick brown dog jumped over the lazy fox." — see
        // `VisionTestHelpers.ocrCandidatePhrases` for the full list.
        VisionTestHelpers.expectRecognizesOCRText(text, label: "GlmOcr OCR")
    }
}
