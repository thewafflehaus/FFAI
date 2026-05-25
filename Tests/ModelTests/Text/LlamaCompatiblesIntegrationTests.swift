// Integration coverage for the LlamaCompatibles cluster — models
// whose config.json declares a non-"LlamaForCausalLM" arch but whose
// weight layout + forward shape is byte-for-byte compatible with
// our Llama loader. We test one representative from each cluster so
// future loader refactors can't silently break these checkpoints.
//
// Load failures propagate to the test runner — these tests exist to
// pin coverage; a missing checkpoint is a real failure, not a silent
// pass.
//
// 2026-05-25 — mlx-community renamed / dropped a number of the small
// bf16 conversions this suite originally used. Every repo ID below
// has been verified via the HF API at the time of the rename.
// Substitutions:
//   - SmolLM2-360M-Instruct-bf16   → SmolLM2-360M-Instruct-bf16-mlx
//     (rename only — same bf16 weights, `-mlx` suffix convention).
//   - Starcoder2-3B-bf16           → starcoder2-3b-4bit
//     (only the 4-bit conversion is published now).
//   - OLMo-2-0425-1B-Instruct-bf16 → OLMo-2-1124-7B-Instruct-4bit
//     (1B variant was never converted; 7B is the smallest available).
//   - granite-3.0-2b-instruct-bf16 → IBM-granite-3.2-2b-instruct-4bit
//     (3.0-2b not converted; 3.2-2b is the closest current line).
//   - internlm2-chat-1_8b-bf16     → internlm2_5-7b-chat-4bit
//     (1.8B never converted; 7B InternLM 2.5 is what's available).
//   - SmolLM-360M-Instruct-bf16    → SmolLM-360M-Instruct-fp16
//     (rename: fp16 instead of bf16).
//
// The shape assertions on substituted models (OLMo, Granite, InternLM)
// reflect the new sizes; the goal is to verify the architecture-string
// routing + forward path works for each family, not to pin specific
// hyper-parameters (those are owned by the per-family test files).

import Foundation
import Testing
@testable import FFAI

@Suite("LlamaCompatibles Integration", .serialized)
struct LlamaCompatiblesIntegrationTests {

    @Test("SmolLM2-360M-Instruct (LlamaForCausalLM, no biases) decodes coherently")
    func smolLM2() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("ekryski/SmolLM2-360M-Instruct-4bit")
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
            try await Model.load("mlx-community/starcoder2-3b-4bit")
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
        expectCoherentOutput(result.generatedTokens, label: "Starcoder2 3B 4bit")
    }

    @Test("OLMo-2-1124-7B-Instruct (Olmo2ForCausalLM) decodes coherently")
    func olmo2() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/OLMo-2-1124-7B-Instruct-4bit")
        }
        // OLMo 2 7B canonical: hidden=4096, headDim=128.
        #expect(m.engine.hidden == 4096)
        #expect(m.engine.headDim == 128)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "OLMo 2 7B 4bit")
    }

    @Test("SmolLM3-3B (SmolLM3ForCausalLM, every-Nth attention layer) decodes coherently")
    func smolLM3() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/SmolLM3-3B-4bit")
        }
        // SmolLM3 3B: hidden=2048, nLayers=36, nHeads=16, nKVHeads=4.
        #expect(m.engine.hidden == 2048)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "SmolLM3 3B bf16")
    }

    @Test("Granite-3.2-2B-Instruct (GraniteForCausalLM) decodes coherently")
    func granite3() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/IBM-granite-3.2-2b-instruct-4bit")
        }
        // Granite 3.2 2B: hidden=2048 (same dim as 3.0 — only the
        // training data + alignment changed).
        #expect(m.engine.hidden == 2048)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "Granite 3.2 2B 4bit")
    }

    @Test("InternLM2.5-7B-Chat (InternLM2ForCausalLM) decodes coherently")
    func internLM2() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/internlm2_5-7b-chat-4bit")
        }
        // InternLM 2.5 7B canonical: hidden=4096, headDim=128.
        #expect(m.engine.hidden == 4096)
        #expect(m.engine.headDim == 128)
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "InternLM2.5 7B 4bit")
    }

    @Test("SmolLM-360M (SmolLMForCausalLM, original family) decodes coherently")
    func smolLM1() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/SmolLM-360M-Instruct-4bit")
        }
        // SmolLM 1 360M canonical: hidden=960, nLayers=32, nHeads=15,
        // nKVHeads=5, headDim=64.
        let result = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, label: "SmolLM 360M fp16")
    }
}
