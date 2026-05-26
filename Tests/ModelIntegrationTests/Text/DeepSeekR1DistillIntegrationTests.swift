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
// Slow integration test for DeepSeek R1 distilled models. These are
// fine-tunes of Qwen 2 / 3-series-Llama architectures, not novel
// model classes — they load through the existing Qwen2 / Llama
// dispatch path. The test pins this so future loader refactors can't
// silently break the R1-distill checkpoints.
//
// Skipped automatically if the checkpoint isn't reachable.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("DeepSeek R1 Distill Integration", .serialized)
struct DeepSeekR1DistillIntegrationTests {

    @Test("R1-Distill-Llama-8B (Llama architecture) generates coherent output")
    func r1DistillLlama() async throws {
        // 8B parameters in 4-bit ≈ 4.5 GB on disk. Uses
        // model_type='llama' / arch='LlamaForCausalLM' so it flows
        // through our base Llama loader without any DeepSeek-specific
        // plumbing.
        let modelId = "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // 8B shape sanity:
        //   hidden = 4096, nLayers = 32, nHeads = 32, nKVHeads = 8,
        //   headDim = 128, intermediate = 14336.
        #expect(m.engine.hidden == 4096)
        #expect(m.engine.nLayers == 32)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 128)

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "R1-Distill-Llama-8B")
    }

    @Test("R1-Distill-Qwen-1.5B (Qwen 2 architecture) generates coherent output")
    func r1DistillQwen() async throws {
        // Smallest distill — 1.5B parameters in 4-bit ≈ 800 MB on disk.
        // Uses model_type='qwen2' / arch='Qwen2ForCausalLM', so it
        // flows through the bias-aware Linear path added with the
        // Qwen 2.x family wiring.
        let modelId = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // 1.5B shape sanity (per the HF config):
        //   hidden = 1536, nLayers = 28, nHeads = 12, nKVHeads = 2,
        //   headDim = 128, intermediate = 8960.
        #expect(m.engine.hidden == 1536)
        #expect(m.engine.nLayers == 28)
        #expect(m.engine.nHeads == 12)
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.headDim == 128)
        #expect(
            m.llama != nil,
            "R1-Distill-Qwen should load through the 3-series engine after bias-aware Linear")

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "R1-Distill-Qwen-1.5B")
    }
}
