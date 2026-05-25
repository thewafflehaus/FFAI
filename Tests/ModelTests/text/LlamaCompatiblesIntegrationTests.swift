// Integration coverage for the LlamaCompatibles cluster — models
// whose config.json declares a non-"LlamaForCausalLM" arch but whose
// weight layout + forward shape is byte-for-byte compatible with
// our Llama loader. We test one representative from each cluster so
// future loader refactors can't silently break these checkpoints.
//
// Load failures propagate to the test runner — these tests exist to
// pin coverage; a missing checkpoint is a real failure, not a silent
// pass.

import Foundation
import Testing
@testable import FFAI

@Suite("LlamaCompatibles integration", .serialized)
struct LlamaCompatiblesIntegrationTests {

    @Test("SmolLM2-360M-Instruct (LlamaForCausalLM, no biases) decodes coherently")
    func smolLM2() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/SmolLM2-360M-Instruct-bf16")
        }
        // SmolLM2 360M canonical: hidden=960, nLayers=32, nHeads=15,
        // nKVHeads=5, headDim=64. Verifies the head_dim=64 SDPA path
        // (same kernel Llama 3.2 1B uses).
        #expect(m.engine.hidden == 960)
        #expect(m.engine.nLayers == 32)
        #expect(m.engine.headDim == 64)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "SmolLM2 360M bf16")
    }

    @Test("Starcoder2-3B (Starcoder2ForCausalLM, attention biases) decodes coherently")
    func starcoder2() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/Starcoder2-3B-bf16")
        }
        // Starcoder2 3B canonical: hidden=3072, nLayers=30, nHeads=24,
        // nKVHeads=2, headDim=128. The attention biases pass through
        // loadLinear's auto-detection — same path as Qwen 2.
        #expect(m.engine.hidden == 3072)
        #expect(m.engine.nLayers == 30)
        #expect(m.engine.headDim == 128)
        let result = try await m.generate(
            prompt: "def fibonacci(n):\n",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "Starcoder2 3B bf16")
    }

    @Test("OLMo-2-0425-1B-Instruct (Olmo2ForCausalLM) decodes coherently")
    func olmo2() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/OLMo-2-0425-1B-Instruct-bf16")
        }
        // OLMo 2 1B canonical: hidden=2048, nLayers=16, nHeads=16,
        // nKVHeads=16 (MHA), headDim=128.
        #expect(m.engine.hidden == 2048)
        #expect(m.engine.headDim == 128)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "OLMo 2 1B bf16")
    }

    @Test("SmolLM3-3B (SmolLM3ForCausalLM, every-Nth attention layer) decodes coherently")
    func smolLM3() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/SmolLM3-3B-bf16")
        }
        // SmolLM3 3B: hidden=2048, nLayers=36, nHeads=16, nKVHeads=4.
        #expect(m.engine.hidden == 2048)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "SmolLM3 3B bf16")
    }

    @Test("Granite-3-2B-Instruct (GraniteForCausalLM) decodes coherently")
    func granite3() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/granite-3.0-2b-instruct-bf16")
        }
        // Granite 3 2B canonical: hidden=2048, nLayers=40, nHeads=32,
        // nKVHeads=8, headDim=64.
        #expect(m.engine.hidden == 2048)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "Granite 3 2B bf16")
    }

    @Test("InternLM2-1.8B-Chat (InternLM2ForCausalLM) decodes coherently")
    func internLM2() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/internlm2-chat-1_8b-bf16")
        }
        // InternLM 2 1.8B canonical: hidden=2048, nLayers=24, nHeads=16,
        // nKVHeads=8, headDim=128.
        #expect(m.engine.hidden == 2048)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "InternLM2 1.8B bf16")
    }

    @Test("SmolLM-360M (SmolLMForCausalLM, original family) decodes coherently")
    func smolLM1() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/SmolLM-360M-Instruct-bf16")
        }
        // SmolLM 1 360M canonical: hidden=960, nLayers=32, nHeads=15,
        // nKVHeads=5, headDim=64.
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "SmolLM 360M bf16")
    }
}
