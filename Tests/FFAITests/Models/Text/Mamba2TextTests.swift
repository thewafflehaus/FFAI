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
// Mamba2TextTests — unit coverage for `Sources/FFAI/Models/Text/Mamba2Text.swift`.
//
// Offline. Covers the `Mamba2Dense` variant surface (capabilities +
// greedy generation defaults). The selective-SSM mixer + per-layer
// `Mamba2LayerCache` round-trip are exercised by
// Tests/ModelTests/Mamba2IntegrationTests.swift against a real
// checkpoint — this file is the lightweight surface guard.

import Foundation
import Testing

@testable import FFAI

@Suite("Mamba2Dense Variant Surface")
struct Mamba2TextTests {

    @Test("Mamba2Dense advertises text in/out capabilities")
    func capabilities() {
        #expect(Mamba2Dense.availableCapabilities.contains(.textIn))
        #expect(Mamba2Dense.availableCapabilities.contains(.textOut))
        #expect(!Mamba2Dense.availableCapabilities.contains(.imageIn))
    }

    /// Mamba 2 checkpoints are base-LM only — no chat tuning. Greedy
    /// (temp 0) is the documented default.
    @Test("Mamba2Dense default generation parameters are greedy")
    func defaultGenerationParameters() {
        let p = Mamba2Dense.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature == 0.0)
        #expect(p.topP == 1.0)
        #expect(p.topK == 0)
    }
}
