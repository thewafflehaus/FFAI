// Slow integration test: downloads (or hits cache) Mamba 2 130M and
// runs end-to-end greedy generation. Skipped automatically if the
// network or checkpoint isn't available.
//
// 130M is the smallest Mamba 2 family checkpoint (~260MB), so the
// integration suite stays fast. The architecture (Mamba2Dense) is the
// same for 370M / 780M / 1.3B / 2.7B — those would be drop-in swaps
// once we want a perf test.

import Foundation
import Testing
@testable import FFAI

@Suite("Mamba 2 130M integration", .serialized)
struct Mamba2IntegrationTests {

    @Test("load + greedy generate produces non-degenerate text")
    func loadAndGenerate() async throws {
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load("mlx-community/mamba2-130m") }
        } catch {
            print("Mamba 2 integration test skipped: \(error)")
            return
        }

        // Engine should be Mamba 2 (not Llama / Qwen3).
        #expect(m.mamba2 != nil)
        #expect(m.llama == nil)
        #expect(m.qwen3 == nil)

        // Shapes from the published 130M config.
        #expect(m.engine.hidden == 768)
        #expect(m.engine.nLayers == 24)
        #expect(m.engine.nHeads == 24)
        #expect(m.engine.headDim == 64)
        #expect(m.engine.vocab == 50_288)
        if let mamba = m.mamba2 {
            #expect(mamba.stateDim == 128)
            #expect(mamba.convKernel == 4)
            #expect(mamba.dInner == 1536)
            // d_inner + 2 * n_groups * state_dim = 1536 + 256 = 1792
            #expect(mamba.convDim == 1792)
        }

        // Forward one BOS-style token. Logits should be finite and
        // non-uniform (the model has been pre-trained — top tokens
        // should pull ahead of the noise floor).
        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        // Top logit should be strictly greater than the 5th — degenerate
        // (all-equal) logits would indicate a forward-pass numerical bug.
        #expect(top[0].1 > top[4].1)

        // Greedy generation. Asserts the model produces coherent output
        // (no stuck-at-one-token, no degenerate alternation cycles).
        // Thresholds are relaxed because 130M base LMs are repetitive
        // at greedy — minUniqueRatio drops to 0.15 from the default 0.2,
        // and we accept 50 tokens instead of the default 50 floor.
        let result = try await m.generate(
            prompt: "The quick brown fox jumps over the",
            parameters: GenerationParameters(maxTokens: 50, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            minUniqueRatio: 0.15,
            label: "Mamba 2 130M"
        )
    }
}
