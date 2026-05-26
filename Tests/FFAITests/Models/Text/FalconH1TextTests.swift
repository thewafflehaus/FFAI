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
// FalconH1TextTests — unit coverage for `Sources/FFAI/Models/Text/FalconH1Text.swift`.
//
// Offline. Covers the `FalconH1Hybrid` variant surface (capabilities +
// greedy-by-default generation parameters). The full mixer + scalar-
// multiplier-fold loader is exercised in
// Tests/ModelTests/FalconH1IntegrationTests.swift; this file is the
// lightweight surface guard.

import Foundation
import Testing
@testable import FFAI

@Suite("FalconH1Hybrid Variant Surface")
struct FalconH1TextTests {

    @Test("FalconH1Hybrid advertises text in/out capabilities")
    func capabilities() {
        #expect(FalconH1Hybrid.availableCapabilities.contains(.textIn))
        #expect(FalconH1Hybrid.availableCapabilities.contains(.textOut))
        #expect(!FalconH1Hybrid.availableCapabilities.contains(.imageIn))
    }

    /// FalconH1 ships `-Instruct` chat-tuned checkpoints. Greedy by
    /// default keeps integration tests deterministic.
    @Test("FalconH1Hybrid default generation parameters are greedy")
    func defaultGenerationParameters() {
        let p = FalconH1Hybrid.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature == 0.0)
        #expect(p.topP == 1.0)
        #expect(p.topK == 0)
    }
}
