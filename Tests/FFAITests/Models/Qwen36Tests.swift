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
// Qwen36Tests — root-file unit tests for `Sources/FFAI/Models/Qwen36.swift`.
//
// Offline. Qwen 3.6 ships under the SAME `qwen3_5*` model_type
// strings as Qwen 3.5 — the architecture is unchanged, only larger
// checkpoints and more MoE experts. `Models/Qwen36.swift` is a
// doc-only anchor that points at `Qwen35` types; this file's job is
// to assert that contract stays true so a reader scanning the
// Models/ directory finds the entry point they expect.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen36 Family Root (shared Qwen35 hybrid types)")
struct Qwen36RootTests {

    @Test("Qwen36 dispatches through the shared Qwen35 family metadata")
    func sharedFamilyMetadata() {
        // The Qwen 3.6 anchor file documents that Qwen 3.6 reuses every
        // Qwen 3.5 model_type / architecture label. Reading the union
        // through `Qwen35` is the canonical path.
        #expect(Qwen35.modelTypes.contains("qwen3_5"))
        #expect(Qwen35.modelTypes.contains("qwen3_5_moe"))
        #expect(Qwen35.architectures.contains("Qwen3_5ForCausalLM"))
        #expect(Qwen35.architectures.contains("Qwen3_5MoeForCausalLM"))
    }

    @Test("Qwen36 routes through Qwen35Hybrid (the shared backbone type)")
    func sharedVariantType() throws {
        let cfg = ModelConfig(architecture: "Qwen3_5MoeForCausalLM",
                              modelType: "qwen3_5_moe", raw: [:])
        let v = try Qwen35.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Qwen35Hybrid.self))
    }
}
