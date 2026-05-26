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
// Qwen3xTextTests — unit coverage for `Sources/FFAI/Models/Text/Qwen3xText.swift`.
//
// Offline. The "Qwen3x" file covers BOTH Qwen 3.5 and Qwen 3.6 (same
// stack-interleaved GDN + attention architecture, same `qwen3_5*`
// model_type strings; the 3.6 family root is a doc-only anchor). This
// file covers:
//   • `Qwen35Hybrid` variant surface (capabilities + greedy defaults),
//   • `Qwen35LayerKind.init(from:)` — the `layer_types` entry parser
//     (`"linear_attention"` = GDN / `"full_attention"` = attention,
//     plus the unknown-name rejection path).

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen35Hybrid Variant Surface")
struct Qwen35HybridTests {

    @Test("Qwen35Hybrid advertises text in/out capabilities")
    func capabilities() {
        #expect(Qwen35Hybrid.availableCapabilities.contains(.textIn))
        #expect(Qwen35Hybrid.availableCapabilities.contains(.textOut))
        #expect(!Qwen35Hybrid.availableCapabilities.contains(.imageIn))
    }

    /// Qwen 3.5 ships base + instruction-tuned checkpoints. Greedy by
    /// default keeps the integration suite deterministic.
    @Test("Qwen35Hybrid default generation parameters are greedy")
    func defaultGenerationParameters() {
        let p = Qwen35Hybrid.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature == 0.0)
        #expect(p.topP == 1.0)
        #expect(p.topK == 0)
        #expect((p.prefillStepSize ?? 0) >= 256)
    }
}

@Suite("Qwen35LayerKind layer_types Parser")
struct Qwen35LayerKindTests {

    @Test("layer_types entries map to GDN (linear_attention) and attention (full_attention)")
    func validNames() throws {
        #expect(try Qwen35LayerKind(from: "linear_attention") == .gdn)
        #expect(try Qwen35LayerKind(from: "full_attention") == .attention)
    }

    @Test("unknown layer_types entry throws unsupportedConfig")
    func unknownRejected() {
        #expect(throws: Qwen35Error.self) {
            _ = try Qwen35LayerKind(from: "mamba")
        }
        #expect(throws: Qwen35Error.self) {
            _ = try Qwen35LayerKind(from: "sliding_attention")
        }
    }
}
