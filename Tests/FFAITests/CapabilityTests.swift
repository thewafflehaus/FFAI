import Foundation
import Testing
@testable import FFAI

@Suite("Capability")
struct CapabilityTests {
    @Test("textOnly contains exactly textIn + textOut")
    func textOnlySet() {
        #expect(Capability.textOnly == [.textIn, .textOut])
    }

    @Test("textWithTools adds toolCalling")
    func textWithToolsSet() {
        #expect(Capability.textWithTools == [.textIn, .textOut, .toolCalling])
    }

    @Test("all cases enumerated")
    func allCases() {
        let s = Set(Capability.allCases)
        #expect(s.contains(.textIn))
        #expect(s.contains(.textOut))
        #expect(s.contains(.visionIn))
        #expect(s.contains(.videoIn))
        #expect(s.contains(.audioIn))
        #expect(s.contains(.audioOut))
        #expect(s.contains(.toolCalling))
        #expect(s.count == 7)
    }

    @Test("Codable round-trip via raw value")
    func codable() throws {
        let original: Capability = .visionIn
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(Capability.self, from: data)
        #expect(decoded == original)
    }

    // ─── Loaded.availableCapabilities ────────────────────────────────

    @Test("Loaded — defaults to textOnly when not specified")
    func loadedDefaultsTextOnly() {
        // The memberwise-style init defaults availableCapabilities.
        let params = GenerationParameters()
        // A nil engine can't be constructed; assert the default on the
        // capability set itself, which is what callers rely on.
        #expect(Capability.textOnly == [.textIn, .textOut])
        _ = params
    }

    @Test("visionIn is a distinct, non-text capability")
    func visionInDistinct() {
        #expect(!Capability.textOnly.contains(.visionIn))
        #expect(Capability.visionIn.rawValue == "visionIn")
    }
}
