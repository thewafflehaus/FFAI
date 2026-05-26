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
// Slow integration test for Phi-3 mini. Phi-3 differs from Llama in
// two layout details:
//   - fused `qkv_proj` (we slice into q/k/v Tensor views)
//   - fused `gate_up_proj` (we slice into gate/up Tensor views)
//
// The integration test confirms the fused-weight slicing path produces
// coherent generated text end-to-end, since per-kernel coverage doesn't
// guard against a shape-misalignment in the slice math.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("Phi Integration", .serialized)
struct Phi3IntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        // Phi-3-mini-4k-instruct: 4k context, no longrope. The 128k
        // variant ships with `rope_scaling.type = "longrope"` and
        // throws PhiError.unsupportedRopeScaling — see Phi.swift for
        // the SuScaledRoPE Phase 6.x follow-up.
        let modelId = "mlx-community/Phi-3-mini-4k-instruct-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Phi-3 mini canonical shapes (3.8B parameters):
        //   hidden = 3072, nLayers = 32, nHeads = 32, nKVHeads = 32 (MHA),
        //   headDim = 96, intermediate = 8192.
        #expect(m.engine.hidden == 3072)
        #expect(m.engine.nLayers == 32)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.headDim == 96)
        #expect(
            m.llama != nil, "Phi-3 should load through the Llama engine after fused-weight slicing")

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "Phi-3 mini 4-bit")
    }
}
