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
// Qwen3TextTests — unit coverage for `Sources/FFAI/Models/Text/Qwen3Text.swift`.
//
// Offline. Covers the `Qwen3Dense` variant surface: capabilities + the
// audited mlx-swift-lm-tracking generation defaults (temp 0.6, top-p
// 0.95, top-k 20, min-p 0.0, 1024-token prefill chunk). The dense
// decoder + per-head q_norm / k_norm + AURA-cache plumbing are
// exercised by Tests/ModelTests/Qwen3IntegrationTests.swift.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3Dense Variant Surface")
struct Qwen3TextTests {

    @Test("Qwen3Dense advertises text in/out capabilities")
    func capabilities() {
        #expect(Qwen3Dense.availableCapabilities.contains(.textIn))
        #expect(Qwen3Dense.availableCapabilities.contains(.textOut))
        #expect(!Qwen3Dense.availableCapabilities.contains(.visionIn))
    }

    @Test("Qwen3Dense default generation parameters track Qwen 3 family")
    func defaultGenerationParameters() {
        // Per the file header: temp 0.6, top-p 0.95, top-k 20,
        // min-p 0.0, rep-penalty 1.0, prefill chunk 1024.
        let p = Qwen3Dense.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature == 0.6)
        #expect(p.topP == 0.95)
        #expect(p.topK == 20)
        #expect(p.prefillStepSize == 1024)
    }
}
