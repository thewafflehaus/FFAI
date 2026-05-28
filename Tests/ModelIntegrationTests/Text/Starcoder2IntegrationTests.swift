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
// Starcoder 2 family integration coverage — BigCode's code dense
// decoder. The header below was originally drafted thinking
// Starcoder2 was a Llama-shaped clone with attention biases, but on
// closer inspection it is structurally different:
//   - Uses LayerNorm (with `.bias`) instead of RMSNorm
//   - Single-projection GELU-tanh MLP with `c_fc` + `c_proj` names
//     (not the SwiGLU `gate_proj` + `up_proj` + `down_proj`)
//   - Config field is `norm_epsilon` (not `rms_norm_eps`)
//   - Attention biases (also present, but those alone aren't enough
//     for the Llama loader to handle Starcoder2 correctly)
//
// Today Starcoder2 is misrouted through the Llama-compatible
// dispatch list in Loader/Model.swift, which throws
// `Llama: required config field missing` because the config has
// `norm_epsilon` instead of `rms_norm_eps`. The test is gated on
// `enableStarcoder2Suite` (defaults to `false`) until a dedicated
// Starcoder2 loader lands — the gate flips back automatically
// once that loader exists.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Starcoder2 Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites
            && IntegrationGroupGating.enableStarcoder2Suite,
        IntegrationGroupGating.starcoder2SkipReason)
)
struct Starcoder2IntegrationTests {

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
}
