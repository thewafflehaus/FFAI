// MiniCPM5-1B — OpenBMB's 1B base text model. Declares
// `architectures: ["LlamaForCausalLM"]` + `model_type: "llama"`, so it
// routes through FFAI's existing Llama dense loader with no family-
// specific code. Two integration variants are pinned:
//
//   - `openbmb/MiniCPM5-1B` — upstream bf16 (24 layers, hidden=1536,
//     16 heads / 2 KV heads (GQA fan-out 8), head_dim=128, vocab=130560,
//     rope_theta=5e6, 131k context).
//   - `openbmb/MiniCPM5-1B-MLX` — same shape, mlx affine int4 packing
//     (group_size=64). Exercises the QuantizedLinear / QuantizedEmbedding
//     load path on a Llama-shaped backbone.
//
// Both target the head_dim=128 SDPA path that Llama 3.2 3B / Qwen 3
// exercise, so a load + greedy-decode run validates the standard
// transformer forward against a fresh checkpoint family.

import Foundation
import Testing
@testable import FFAI

@Suite("MiniCPM5-1B integration", .serialized)
struct MiniCPM5IntegrationTests {

    @Test("MiniCPM5-1B (bf16) — Llama dispatch + coherent decode")
    func miniCPM5Bf16() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("openbmb/MiniCPM5-1B")
        }
        // Routes through the Llama dispatch — verify the shape pins.
        #expect(m.engine.hidden == 1536)
        #expect(m.engine.nLayers == 24)
        #expect(m.engine.nHeads == 16)
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.headDim == 128)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "MiniCPM5-1B bf16")
    }

    @Test("MiniCPM5-1B-MLX (int4) — Llama dispatch + QuantizedLinear forward")
    func miniCPM5MLX() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("openbmb/MiniCPM5-1B-MLX")
        }
        // Same canonical shape as the bf16 variant — quantization changes
        // weight storage but not the hyperparameters.
        #expect(m.engine.hidden == 1536)
        #expect(m.engine.nLayers == 24)
        #expect(m.engine.headDim == 128)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "MiniCPM5-1B int4")
    }
}
