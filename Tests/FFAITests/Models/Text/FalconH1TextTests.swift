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
        #expect(!FalconH1Hybrid.availableCapabilities.contains(.visionIn))
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
