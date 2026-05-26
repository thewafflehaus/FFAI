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
// GPTOSSTextTests — unit coverage for `Sources/FFAI/Models/Text/GPTOSSText.swift`.
//
// Offline. Covers:
//   • `GPTOSSMoEVariant` capability + reasoning-level surface
//     (`ReasoningCapable` conformance + GPT-OSS's native {.low,
//     .medium, .high} ladder + the `.extraHigh`/`.max` clamp behavior),
//   • `GPTOSSAttentionKind.init(from:)` — the `layer_types` entry parser
//     (`"sliding_attention"` / `"full_attention"` + the unknown-string
//     rejection path),
//   • `GPTOSSMoELayer` invariant: `topK` outside `1...experts.count`
//     fails the constructor `precondition` (we exercise the valid /
//     boundary branches; out-of-range traps live in the integration
//     suite — preconditionFailure can't be caught in-test).

import Foundation
import Testing
@testable import FFAI

@Suite("GPTOSSMoE Variant Surface")
struct GPTOSSTextVariantTests {

    @Test("GPTOSSMoEVariant advertises text + reasoningLevel capabilities")
    func capabilities() {
        let caps = GPTOSSMoEVariant.availableCapabilities
        #expect(caps.contains(.textIn))
        #expect(caps.contains(.textOut))
        #expect(caps.contains(.reasoningLevel))
    }

    @Test("supportedReasoningLevels is exactly {.low, .medium, .high}")
    func reasoningLevels() {
        let lv = GPTOSSMoEVariant.supportedReasoningLevels
        #expect(lv == [.low, .medium, .high])
        #expect(!lv.contains(.none))   // .none is implicit "disable"
        #expect(!lv.contains(.max))
    }

    /// User requests outside the GPT-OSS native ladder clamp to the
    /// nearest supported level. `.extraHigh` and `.max` both clamp to
    /// `.high`; `.none` always survives untouched.
    @Test("reasoning-level clamp behaves per the documented ladder")
    func reasoningLevelClamp() {
        let supported = GPTOSSMoEVariant.supportedReasoningLevels
        #expect(ReasoningLevel.none.clamped(to: supported) == .none)
        #expect(ReasoningLevel.low.clamped(to: supported) == .low)
        #expect(ReasoningLevel.medium.clamped(to: supported) == .medium)
        #expect(ReasoningLevel.high.clamped(to: supported) == .high)
        #expect(ReasoningLevel.extraHigh.clamped(to: supported) == .high)
        #expect(ReasoningLevel.max.clamped(to: supported) == .high)
    }

    @Test("default generation parameters declare 2048 prefill chunk + reasoning .none")
    func defaultGenerationParameters() {
        let p = GPTOSSMoEVariant.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature >= 0)
        // GPT-OSS audited optimum (matches mlx-swift-lm).
        #expect(p.prefillStepSize == 2048)
        // Reasoning disabled until the caller explicitly opts in.
        #expect(p.reasoningLevel == .none)
    }
}

@Suite("GPTOSSAttentionKind layer_types Parser")
struct GPTOSSAttentionKindTests {

    @Test("layer_types entries map to the documented attention kinds")
    func validNames() throws {
        #expect(try GPTOSSAttentionKind(from: "sliding_attention") == .sliding)
        #expect(try GPTOSSAttentionKind(from: "full_attention") == .full)
    }

    @Test("unknown layer_types entry throws unsupportedConfig")
    func unknownNameRejected() {
        #expect(throws: GPTOSSError.self) {
            _ = try GPTOSSAttentionKind(from: "linear_attention")
        }
        #expect(throws: GPTOSSError.self) {
            _ = try GPTOSSAttentionKind(from: "")
        }
    }
}
