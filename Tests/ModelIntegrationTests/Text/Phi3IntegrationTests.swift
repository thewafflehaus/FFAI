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
// Slow integration test for Phi-3 mini. Phi-3 differs from Llama in
// two layout details:
//   - fused `qkv_proj` (we slice into q/k/v Tensor views)
//   - fused `gate_up_proj` (we slice into gate/up Tensor views)
//
// The integration test confirms the fused-weight slicing path produces
// coherent generated text end-to-end, since per-kernel coverage doesn't
// guard against a shape-misalignment in the slice math.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Phi Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct Phi3IntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        // Phi-3-mini-4k-instruct: 4k context, no longrope. The 128k
        // variant ships with `rope_scaling.type = "longrope"` and
        // throws PhiError.unsupportedRopeScaling — see Phi.swift for
        // the SuScaledRoPE Phase 6.x follow-up.
        let modelId = "mlx-community/Phi-3-mini-4k-instruct-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        // ── 1. Load ──────────────────────────────────────────────────────
        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // ── 2. Interfaces we expect ──────────────────────────────────────
        #expect(
            m.llama != nil, "Phi-3 should load through the Llama engine after fused-weight slicing")
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)
        // Phi-3 mini canonical shapes (3.8B parameters): hidden = 3072,
        // nLayers = 32, nHeads = 32, nKVHeads = 32 (MHA), headDim = 96,
        // intermediate = 8192, vocab = 32_064.
        #expect(m.engine.hidden == 3072)
        #expect(m.engine.nLayers == 32)
        #expect(m.engine.nHeads == 32)
        #expect(m.engine.nKVHeads == 32)
        #expect(m.engine.headDim == 96)
        #expect(m.engine.vocab == 32_064)

        // ── 3. Errors we expect ─────────────────────────────────────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 32)
        for (i, c) in caches.enumerated() {
            if !(c is KVCache) {
                Issue.record("Phi-3: layer \(i) cache is \(type(of: c)), expected KVCache")
            }
        }

        // ── 4. Forward pass produces expected output ────────────────────
        // KNOWN FAILURE (2026-05-27): Phi-3 head_dim=96 isn't yet
        // emitted by Ops.sdpaDecode (supports {64, 128, 256, 512}). The
        // load + interface + cache-shape checks above pass; this forward
        // call (and the generate below) will crash with `head_dim must
        // be one of {64,128,256,512}` until the head_dim=96 kernel
        // specialization lands in metaltile. Suite stays ENABLED so the
        // gap stays visible. Tracked as a follow-up.
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 32_064)
        let top = Sampling.topN(logits, n: 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 > top[4].1)

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "Phi-3 mini 4-bit")
    }
}
