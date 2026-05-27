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
// InternLM 2 family integration coverage — Shanghai AI Lab's
// InternLM v2 Llama-shaped dense decoder. Some checkpoints use a
// fused `wqkv` projection that `loadLinear` handles transparently.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "InternLM2 Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct InternLM2IntegrationTests {

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
}
