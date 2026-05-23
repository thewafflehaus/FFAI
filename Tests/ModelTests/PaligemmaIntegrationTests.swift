// PaliGemma integration test: loads the cached mlx-community/paligemma-3b-mix-448-8bit
// checkpoint and runs end-to-end image captioning over the dog fixture.
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
        let modelId = "mlx-community/paligemma-3b-mix-448-8bit"

        let m: Model
        do {
            m = try await Model.load(modelId)
        } catch {
            print("PaliGemma integration test skipped: \(error)")
            return
        }

        // Verify basic shapes from the published config.
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.nLayers == 18)
        #expect(m.engine.nHeads == 8)
        #expect(m.engine.nKVHeads == 1)
        #expect(m.engine.headDim == 256)
        #expect(m.engine.vocab == 257216)

        guard let pg = m.engine as? PaligemmaModel else {
            Issue.record("Expected a PaligemmaModel engine")
            return
        }

        // Load + preprocess the dog fixture at PaliGemma's 448 resolution
        // with SigLIP normalization (mean 0.5 / std 0.5 per channel).
        let pixels = try VLMTestSupport.dogImageCHWNormalized(
            targetSize: 448, normalization: .siglip)
        #expect(pixels.count == 3 * 448 * 448)
        pg.setImagePixels(pixels)

        // Build the prompt: 1024 image tokens followed by "describe image".
        let imageTokenId   = pg.imageTokenIndex
        let numImageTokens = pg.numImageTokens
        let textTokens = m.tokenizer.encode(text: "describe image")
        let promptTokens = Array(repeating: imageTokenId, count: numImageTokens)
            + textTokens

        // Greedy decode 32 tokens. PaligemmaModel.forward injects the
        // precomputed image features at imageTokenIndex positions.
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
        for _ in 0..<32 {
            if stopSet.contains(nextToken) { break }
            generated.append(nextToken)
            nextToken = m.engine.forwardSample(
                tokenId: nextToken, position: pos,
                caches: caches, device: Device.shared)
            pos += 1
        }

        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("PaliGemma generated: \(text)")
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
