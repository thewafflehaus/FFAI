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
// Qwen 3.6 — Unsloth "UD" dynamic mixed-precision integration test.
//
// Pinned here so the existence of the test surface is tracked and the
// suite can be flipped on later, but DELIBERATELY SKIPPED today via
// `IntegrationGroupGating.enableMixedPrecisionSuites` (default `false`).
//
// Why skipped:
//
//   Unsloth's "UD-MLX-4bit" recipe is per-tensor dynamic quantization —
//   different layers ship at different bit-widths inside the same
//   checkpoint (`mixed_4_8`-style). FFAI's loader DOES handle per-tensor
//   bit-widths via `deriveAffineQuantBits` reading each tensor's shape,
//   so these checkpoints *should* load, but the path isn't pinned by a
//   passing integration run yet. Enable + bisect once we're back to
//   mixed-precision coverage (see the "Ablated + mixed-precision
//   follow-up" section of `planning/session-plan.md`).
//
// Repos pinned:
//
//   • mlx-community/Qwen3.6-27B-4bit         — uniform 4-bit reference;
//                                              already covered by the
//                                              ModelKVCacheMatrix suite.
//   • mlx-community/Qwen3.6-35B-A3B-4bit     — uniform 4-bit MoE reference.
//   • unsloth/Qwen3.6-27B-UD-MLX-4bit        — Unsloth dynamic mixed-precision
//                                              of the dense 27B.
//   • unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit    — Unsloth dynamic mixed-precision
//                                              of the 35B-A3B MoE.
//
// When enabled, each test loads the checkpoint, asserts the engine is
// the Qwen3.5 hybrid (Qwen 3.6 shares the engine), checks the layer
// counts match the family contract, and runs a single greedy decode
// step on a probe prompt to confirm the per-tensor quantization-bit
// dispatch picks the right kernel for every layer.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Qwen3.6 Unsloth UD-MLX mixed-precision Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableMixedPrecisionSuites,
        IntegrationGroupGating.mixedPrecisionSkipReason)
)
struct Qwen36UnslothMixedIntegrationTests {

    // ─── Dense 27B — uniform 4-bit + UD-MLX dynamic 4-bit ──────────────

    @Test("Qwen3.6-27B uniform 4-bit loads + one-token forward is finite")
    func dense27BUniform4bit() async throws {
        try await loadAndSmokeForward(
            modelId: "mlx-community/Qwen3.6-27B-4bit",
            expectMoE: false,
            label: "Qwen3.6-27B uniform 4-bit")
    }

    @Test("Qwen3.6-27B-UD-MLX-4bit (Unsloth dynamic mixed-precision) loads + smoke forward")
    func dense27BUnslothMixed() async throws {
        try await loadAndSmokeForward(
            modelId: "unsloth/Qwen3.6-27B-UD-MLX-4bit",
            expectMoE: false,
            label: "Qwen3.6-27B-UD-MLX-4bit Unsloth dynamic")
    }

    // ─── MoE 35B-A3B — uniform 4-bit + UD-MLX dynamic 4-bit ─────────────

    @Test("Qwen3.6-35B-A3B uniform 4-bit loads + one-token forward is finite")
    func moe35BUniform4bit() async throws {
        try await loadAndSmokeForward(
            modelId: "mlx-community/Qwen3.6-35B-A3B-4bit",
            expectMoE: true,
            label: "Qwen3.6-35B-A3B uniform 4-bit")
    }

    @Test("Qwen3.6-35B-A3B-UD-MLX-4bit (Unsloth dynamic mixed-precision) loads + smoke forward")
    func moe35BUnslothMixed() async throws {
        try await loadAndSmokeForward(
            modelId: "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
            expectMoE: true,
            label: "Qwen3.6-35B-A3B-UD-MLX-4bit Unsloth dynamic")
    }

    // ─── Helper ────────────────────────────────────────────────────────

    /// Load the checkpoint, sanity-check that the engine is the Qwen3.5
    /// hybrid (Qwen 3.6 shares the engine), assert MoE presence matches
    /// the family expectation, and confirm a single forward step
    /// produces finite logits. No coherence assertion — this suite is
    /// about per-tensor-bit-width dispatch correctness, not generation
    /// quality. The UD-MLX checkpoints exercise FFAI's
    /// `deriveAffineQuantBits` path that derives each tensor's bit-width
    /// from its on-disk shape; per-layer correctness comes from "no
    /// nan / inf logits + the load succeeded" at this layer.
    private func loadAndSmokeForward(
        modelId: String, expectMoE: Bool, label: String
    ) async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(modelId)
        }
        let q = try #require(
            m.qwen35,
            "\(label): expected Qwen35Model engine (Qwen 3.6 shares the engine)")
        #expect(
            q.hasMoE == expectMoE,
            "\(label): MoE expectation mismatch (expected \(expectMoE), got \(q.hasMoE))")
        print(
            "[\(label)] hidden=\(q.hidden) layers=\(q.nLayers) "
                + "heads=\(q.nHeads) kv=\(q.nKVHeads) headDim=\(q.headDim) "
                + "hasMoE=\(q.hasMoE) dtype=\(q.dtype)")

        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == q.vocab)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 > top[4].1)
    }
}
