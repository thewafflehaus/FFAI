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
// SmolLM family integration coverage — HuggingFace's small Llama-3
// shaped dense models. Ships across three generations; we exercise
// one representative from each so loader refactors can't silently
// break the architecture-string routing or the forward path.
//
// Load failures propagate to the runner — a missing checkpoint is a
// real failure, not a silent pass.
//
// 2026-05-25 — mlx-community renamed / dropped several small bf16
// conversions; the IDs below have been verified via the HF API.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("SmolLM Integration", .serialized)
struct SmolLMIntegrationTests {

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

    @Test("SmolLM2-360M-Instruct (no biases) decodes coherently")
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
        expectCoherentOutput(result.generatedTokens, label: "SmolLM2 360M 4bit")
    }

    @Test("SmolLM3-3B (every-Nth attention layer) decodes coherently")
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
        expectCoherentOutput(result.generatedTokens, label: "SmolLM3 3B 4bit")
    }
}
