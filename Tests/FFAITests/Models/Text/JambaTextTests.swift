// JambaTextTests — unit coverage for `Sources/FFAI/Models/Text/JambaText.swift`.
//
// Offline. Covers:
//   • `JambaHybrid` variant surface (capabilities + greedy defaults),
//   • `JambaLayerKind.init(from:)` — the `layers_block_type` entry
//     parser (`"mamba"` / `"attention"` + the unknown-name rejection
//     path). The full Mamba 1 CPU scan + per-token loader is exercised
//     in Tests/ModelTests/JambaIntegrationTests.swift.

import Foundation
import Testing
@testable import FFAI

@Suite("JambaHybrid Variant Surface")
struct JambaTextVariantTests {

    @Test("JambaHybrid advertises text in/out capabilities")
    func capabilities() {
        #expect(JambaHybrid.availableCapabilities.contains(.textIn))
        #expect(JambaHybrid.availableCapabilities.contains(.textOut))
        #expect(!JambaHybrid.availableCapabilities.contains(.visionIn))
    }

    /// Jamba ships base + instruction-tuned checkpoints. Greedy by
    /// default keeps the integration suite deterministic.
    @Test("JambaHybrid default generation parameters are greedy")
    func defaultGenerationParameters() {
        let p = JambaHybrid.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature == 0.0)
        #expect(p.topP == 1.0)
        #expect(p.topK == 0)
    }
}

@Suite("JambaLayerKind layers_block_type Parser")
struct JambaLayerKindTests {

    @Test("layers_block_type entries map to mamba / attention")
    func validNames() throws {
        #expect(try JambaLayerKind(from: "mamba") == .mamba)
        #expect(try JambaLayerKind(from: "attention") == .attention)
    }

    @Test("unknown layers_block_type entry throws unsupportedConfig")
    func unknownRejected() {
        #expect(throws: JambaError.self) {
            _ = try JambaLayerKind(from: "moe")
        }
        #expect(throws: JambaError.self) {
            _ = try JambaLayerKind(from: "")
        }
    }
}
