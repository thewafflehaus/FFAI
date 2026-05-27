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
// Granite 3 family integration coverage — IBM's Granite v3 dense text
// models (granite-3.0, granite-3.1, granite-3.2). Llama-3-shaped
// weights routed through `LlamaDense`.
//
// Granite 4 (granite-4.0-h, GraniteMoeHybrid) is a different
// architecture (Mamba 2 / attention / MoE hybrid) and has its own
// integration test under `Granite4IntegrationTests.swift`.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Granite3 Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct Granite3IntegrationTests {

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
}
