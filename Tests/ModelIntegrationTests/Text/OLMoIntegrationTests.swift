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
// OLMo family integration coverage — Allen AI's Olmo / Olmo 2 Llama-
// shaped dense decoder. We exercise OLMo 2 (the largest / most-cited
// release) so loader refactors can't silently break the
// architecture-string routing or forward path.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "OLMo Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct OLMoIntegrationTests {

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
}
