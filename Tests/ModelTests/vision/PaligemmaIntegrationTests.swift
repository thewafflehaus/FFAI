// PaliGemma 2 integration test: loads the
// mlx-community/paligemma2-3b-mix-448-8bit checkpoint (PaliGemma 2 =
// SigLIP-So400m + Gemma 2 2B text backbone) and runs end-to-end image
// captioning over the dog fixture.
//
// PaliGemma's vision substitution happens INSIDE its `forward(tokenId:...)`:
// the model precomputes image features via setImagePixels(_:) and, at every
// image-token position, swaps the text embedding for the corresponding
// vision feature. The integration path therefore:
//   1. Load Model → engine downcasts to PaligemmaModel.
//   2. Compute resize+CHW pixels via VLMTestSupport.dogImageCHW(targetSize:)
//      using SigLIP-style mean 0.5 / std 0.5 (PaliGemma normalization).
//   3. Call pg.setImagePixels(pixels) once.
//   4. Build promptTokens = [imageTokenId × 1024] + textTokens.
//   5. Drive the standard greedy decode via m.engine.forward + argmax.
//   6. Assert the decoded caption mentions "dog".
//
// This test is NOT run automatically (see CLAUDE.md → make test-integration).

import Foundation
import Testing
@testable import FFAI

@Suite("PaliGemma 3B integration", .serialized)
struct PaligemmaIntegrationTests {

    @Test("load + image+text generation mentions dog")
    func loadAndGenerate() async throws {
        // PaliGemma 2 — the 2024 refresh using the Gemma 2 text backbone.
        // The original PaliGemma 1 (mlx-community/paligemma-3b-mix-448-8bit)
        // is 2 years old and the test moves with the supported lineage.
        let modelId = "mlx-community/paligemma2-3b-mix-448-8bit"

        let m = try await Model.load(modelId)

        // Verify basic shapes from the PaliGemma 2 3B (Gemma 2 2B backbone)
        // published config. Gemma 2 2B = hidden 2304, 26 layers, 8 heads,
        // 4 KV heads (GQA 2×), head_dim 256.
        #expect(m.engine.hidden == 2304)
        #expect(m.engine.nLayers == 26)
        #expect(m.engine.nHeads == 8)
        #expect(m.engine.nKVHeads == 4)
        #expect(m.engine.headDim == 256)
        #expect(m.engine.vocab == 257216)

        let pg = try #require(m.engine as? PaligemmaModel,
                              "Expected a PaligemmaModel engine")

        // Load + preprocess the dog fixture at PaliGemma's 448 resolution
        // with SigLIP normalization (mean 0.5 / std 0.5 per channel).
        let pixels = try VLMTestSupport.dogImageCHWNormalized(
            targetSize: 448, normalization: .siglip)
        #expect(pixels.count == 3 * 448 * 448)
        pg.setImagePixels(pixels)

        // Build the canonical PaliGemma prompt — matches HF's
        // `PaliGemmaProcessor`:
        //
        //   <image>×N <bos> <prompt> \n
        //
        // Where:
        //   • <image>×N is the per-image placeholder run that
        //     PaligemmaModel.forward swaps with vision features.
        //   • <bos> (id 2) delimits images from the text prompt — the
        //     model was trained to treat everything after <bos> as the
        //     instruction. Skipping it leaves the model in an undefined
        //     state and it immediately samples EOS.
        //   • <prompt> is free-form English for the mix-* checkpoints;
        //     we use the canonical "caption en" task prefix.
        //   • Trailing newline (id 108) terminates the prompt and is
        //     the model's signal to start generating the answer.
        let imageTokenId   = pg.imageTokenIndex
        let numImageTokens = pg.numImageTokens
        let bosId = 2
        // tokenizer.encode does NOT auto-prepend BOS on its own here —
        // we splice it in explicitly so the prompt structure is right
        // regardless of how the tokenizer is configured.
        let textTokens = m.tokenizer.encode(text: "caption en\n")
        let promptTokens = Array(repeating: imageTokenId, count: numImageTokens)
            + [bosId] + textTokens.filter { $0 != bosId }

        // Greedy decode up to 200 tokens — the shared VLM integration
        // ceiling. PaligemmaModel.forward injects the precomputed image
        // features at imageTokenIndex positions; non-image positions
        // route to Gemma2Model's embed lookup.
        let caches = m.engine.makeLayerCaches()
        var generated: [Int] = []
        var nextToken = 0
        for (pos, tok) in promptTokens.enumerated() {
            let logits = m.engine.forward(tokenId: tok, position: pos,
                                          caches: caches,
                                          device: Device.shared)
            if pos == promptTokens.count - 1 {
                nextToken = argmaxOnHost(logits)
            }
        }
        let stopSet: Set<Int> = Set(m.config.eosTokenIds)
        var pos = promptTokens.count
        for _ in 0..<200 {
            if stopSet.contains(nextToken) { break }
            generated.append(nextToken)
            nextToken = m.engine.forwardSample(
                tokenId: nextToken, position: pos,
                caches: caches, device: Device.shared)
            pos += 1
        }

        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("PaliGemma generated (\(generated.count) tokens): \(text)")
        VLMTestSupport.expectMentionsDog(text, label: "PaliGemma")
    }
}

/// Host-side argmax for the final-position logits. Used by the
/// PaliGemma integration test for the first decoded token (prefill tail);
/// every subsequent step uses `engine.forwardSample` which keeps the
/// argmax on-GPU.
private func argmaxOnHost(_ logits: Tensor) -> Int {
    let arr = logits.toFloatArray()
    var bestIdx = 0
    var bestVal: Float = -.infinity
    for (i, v) in arr.enumerated() where v > bestVal {
        bestIdx = i; bestVal = v
    }
    return bestIdx
}
