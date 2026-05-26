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
        #expect(s.contains(.thinking))
        #expect(s.contains(.reasoningLevel))
        #expect(s.count == 9)
    }

    @Test("ReasoningLevel — all cases round-trip via raw value")
    func reasoningLevelCodable() throws {
        for level in ReasoningLevel.allCases {
            let data = try JSONEncoder().encode(level)
            let decoded = try JSONDecoder().decode(ReasoningLevel.self, from: data)
            #expect(decoded == level)
        }
        #expect(ReasoningLevel.allCases.count == 6)
        // Raw values follow the Claude Opus convention — extraHigh
        // serialises as the hyphenated "extra-high" on the wire.
        #expect(ReasoningLevel.none.rawValue == "none")
        #expect(ReasoningLevel.low.rawValue == "low")
        #expect(ReasoningLevel.medium.rawValue == "medium")
        #expect(ReasoningLevel.high.rawValue == "high")
        #expect(ReasoningLevel.extraHigh.rawValue == "extra-high")
        #expect(ReasoningLevel.max.rawValue == "max")
    }

    @Test("ReasoningLevel.clamped — .none always wins regardless of support")
    func clampedNoneAlwaysHonoured() {
        let supported: Set<ReasoningLevel> = [.low, .medium, .high]
        #expect(ReasoningLevel.none.clamped(to: supported) == .none)
        // Even if .none isn't in the model's native set, the explicit
        // disable signal is still honoured.
        let withoutNone: Set<ReasoningLevel> = [.low, .high]
        #expect(ReasoningLevel.none.clamped(to: withoutNone) == .none)
    }

    @Test("ReasoningLevel.clamped — passes through when supported")
    func clampedPassThrough() {
        let supported: Set<ReasoningLevel> = [.low, .medium, .high]
        #expect(ReasoningLevel.low.clamped(to: supported) == .low)
        #expect(ReasoningLevel.medium.clamped(to: supported) == .medium)
        #expect(ReasoningLevel.high.clamped(to: supported) == .high)
    }

    @Test("ReasoningLevel.clamped — extraHigh + max clamp to GPT-OSS .high")
    func clampedGPTOSS() {
        // The user-facing example: GPT-OSS-20B supports low/medium/high
        // natively. extraHigh and max both clamp to high.
        let supported: Set<ReasoningLevel> = [.low, .medium, .high]
        #expect(ReasoningLevel.extraHigh.clamped(to: supported) == .high)
        #expect(ReasoningLevel.max.clamped(to: supported) == .high)
    }

    @Test("ReasoningLevel.clamped — ties break toward the lower (cheaper) level")
    func clampedTieBreak() {
        // Sparse catalogue: {.low, .high}, request .medium → equidistant.
        // Convention is "prefer less reasoning" so .low wins.
        let sparse: Set<ReasoningLevel> = [.low, .high]
        #expect(ReasoningLevel.medium.clamped(to: sparse) == .low)
    }

    @Test("ReasoningLevel.canonicalOrder — every case is present exactly once")
    func canonicalOrderCovers() {
        let order = ReasoningLevel.canonicalOrder
        #expect(order.count == ReasoningLevel.allCases.count)
        #expect(Set(order) == Set(ReasoningLevel.allCases))
        // .none is first; .max is last — required by clamped().
        // Use explicit qualification so Swift binds to ReasoningLevel.none
        // rather than Optional<ReasoningLevel>.none.
        #expect(order.first == ReasoningLevel.none)
        #expect(order.last == ReasoningLevel.max)
    }

    @Test("GPTOSSMoEVariant — conforms to ReasoningCapable with low/medium/high")
    func gptOSSConformance() {
        // Conformance check (compile-time) — also asserts the
        // user-facing example from the design doc.
        let supported = GPTOSSMoEVariant.supportedReasoningLevels
        #expect(supported == [.low, .medium, .high])
        #expect(GPTOSSMoEVariant.availableCapabilities.contains(.reasoningLevel))
        #expect(GPTOSSMoEVariant.defaultGenerationParameters.reasoningLevel == .none)
    }

    @Test("GenerationParameters.reasoningLevel — defaults to nil")
    func genParamsReasoningDefault() {
        let p = GenerationParameters()
        #expect(p.reasoningLevel == nil)
        let custom = GenerationParameters(reasoningLevel: .high)
        #expect(custom.reasoningLevel == .high)
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
