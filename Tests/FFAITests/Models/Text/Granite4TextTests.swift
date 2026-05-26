// Granite4TextTests — unit coverage for `Sources/FFAI/Models/Text/Granite4Text.swift`.
//
// Offline. Covers:
//   • `Granite4Hybrid` variant surface (capabilities + greedy defaults),
//   • `Granite4LayerKind.init(from:)` — the `layer_types` entry parser
//     (`"mamba"` / `"attention"` + the unknown-name rejection path).
//
// The MoE-commit-path forward test lives in the existing
// `GraniteMoeHybridForwardTests.swift` (kept as a focused
// supplementary suite — see the wave-2 task notes).

import Foundation
import Testing
@testable import FFAI

@Suite("Granite4Hybrid Variant Surface")
struct Granite4TextVariantTests {

    @Test("Granite4Hybrid advertises text in/out capabilities")
    func capabilities() {
        #expect(Granite4Hybrid.availableCapabilities.contains(.textIn))
        #expect(Granite4Hybrid.availableCapabilities.contains(.textOut))
        #expect(!Granite4Hybrid.availableCapabilities.contains(.visionIn))
    }

    /// Granite-4 ships base + instruction-tuned checkpoints. Greedy by
    /// default keeps the integration suite deterministic.
    @Test("Granite4Hybrid default generation parameters are greedy")
    func defaultGenerationParameters() {
        let p = Granite4Hybrid.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature == 0.0)
        #expect(p.topP == 1.0)
        #expect(p.topK == 0)
    }
}

@Suite("Granite4LayerKind layer_types Parser")
struct Granite4LayerKindTests {

    @Test("layer_types entries map to mamba / attention")
    func validNames() throws {
        #expect(try Granite4LayerKind(from: "mamba") == .mamba)
        #expect(try Granite4LayerKind(from: "attention") == .attention)
    }

    @Test("unknown layer_types entry throws unsupportedConfig")
    func unknownRejected() {
        #expect(throws: Granite4Error.self) {
            _ = try Granite4LayerKind(from: "linear_attention")
        }
        #expect(throws: Granite4Error.self) {
            _ = try Granite4LayerKind(from: "moe")
        }
    }
}
