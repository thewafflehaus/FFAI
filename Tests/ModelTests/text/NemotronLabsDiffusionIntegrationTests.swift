// Slow integration test: downloads Nemotron-Labs-Diffusion 3B and
// asserts the model loads + produces coherent text in all three
// inference modes — autoregressive, block-wise diffusion, and linear
// self-speculation.
//
// 3B is the smallest checkpoint in the family. Skipped automatically if
// the checkpoint isn't available (offline CI).

import Foundation
import Testing
@testable import FFAI

@Suite("Nemotron-Labs-Diffusion 3B integration", .serialized)
struct NemotronLabsDiffusionIntegrationTests {

    @Test("load + tri-mode generation produces coherent output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/Nemotron-Labs-Diffusion-3B-4bit"
        let prompt = "Once upon a time, in a quiet village"

        // 4096-token context is plenty for the test prompts and keeps
        // the KV cache small; the checkpoint's YaRN window is 262144.
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(modelId, options: LoadOptions(maxContextLength: 4096))
        }

        // Engine should be the tri-mode diffusion model, not Llama.
        #expect(m.nemotronLabsDiffusion != nil)
        #expect(m.llama == nil)

        // Shapes match the published 3B config.
        #expect(m.engine.hidden == 3072)
        #expect(m.engine.nLayers == 26)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)
        #expect(m.engine.vocab == 131_072)
        if let nd = m.nemotronLabsDiffusion {
            #expect(nd.maskTokenId == 100)
            #expect(nd.blockSize == 32)
            // The linear_spec_lora adapter ships in the checkpoint and
            // should auto-attach for the self-speculation drafter.
            #expect(nd.hasLoRA)
        }

        // LoRA hot unload / reload — the adapter can be swapped at
        // runtime without reloading the model.
        #expect(m.hasLoRA)
        m.unloadLoRA()
        #expect(m.hasLoRA == false)
        m.loadLoRA(from: m.modelDirectory)
        #expect(m.hasLoRA, "LoRA should reattach from the model directory")

        // Single-token forward: finite, non-degenerate logits.
        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 > top[4].1)

        // Mode 1 — autoregressive decoding via the standard loop.
        let ar = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 200, temperature: 0))
        #expect(ar.tokensPerSecond > 0)
        expectCoherentOutput(ar.generatedTokens, label: "Nemotron-Labs-Diffusion 3B AR")

        // Mode 2 — block-wise diffusion decoding.
        let diff = m.generateDiffusion(
            prompt: prompt,
            parameters: DiffusionParameters(maxNewTokens: 64, blockLength: 32,
                                            confidenceThreshold: 0.9))
        #expect(diff.forwardPasses > 0)
        expectCoherentOutput(diff.generatedTokens,
                             label: "Nemotron-Labs-Diffusion 3B diffusion")

        // Mode 3 — linear self-speculation (diffusion draft + AR verify).
        let ss = m.generateSelfSpeculative(
            prompt: prompt,
            parameters: DiffusionParameters(maxNewTokens: 64, blockLength: 32,
                                            confidenceThreshold: nil))
        #expect(ss.forwardPasses > 0)
        expectCoherentOutput(ss.generatedTokens,
                             label: "Nemotron-Labs-Diffusion 3B self-spec")
    }
}
