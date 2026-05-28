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
// Qwen3.6 integration coverage — load + interface + forward correctness
// on the same dense GDN-hybrid engine as Qwen3.5 (`m.qwen35`). Qwen3.6
// is a re-pretrained / re-instruction-tuned series that reuses the
// Qwen3.5 `Qwen3_5ForConditionalGeneration` architecture wholesale —
// shape assertions land on the published Qwen3.6-27B-4bit text_config:
//
//   hidden = 5120, nLayers = 64, nHeads = 24, nKVHeads = 4,
//   head_dim = 256, vocab = 248_320, full_attention_interval = 4
//   → 48 GDN linear-attention layers + 16 full-attention layers
//
// The 27B is too big to fit on a typical dev box (≈ 14 GB on disk for
// 4-bit; the bf16 source is ≈ 54 GB). The suite is gated behind both
// the text-group flag AND a local-cache predicate so a clean CI box
// doesn't try to download 14 GB during `make test-integration`. The
// per-model bench coverage (prefill/decode steady-state, batched vs
// per-token forwardMany, first-token argmax probe) lives in
// `Tests/BenchmarkTests/Qwen36TextBenchTest.swift` against the bigger
// 35B-A3B MoE checkpoint — disjoint suite, disjoint gate.
//
// KNOWN FAILURE (2026-05-27): the test currently fails the coherence
// assertion on Qwen3.6-27B-4bit — the loader + Qwen3.5-engine
// dispatch succeeds (load + interface + forward shapes all pass) but
// greedy decode emits token 0 ("!") every step. Same engine
// reproduces coherent output on Qwen3.5-0.8B-4bit, so the regression
// is Qwen3.6-large-scale-specific (likely in the loader's handling
// of one of the 27B-specific config fields — wider hidden=5120,
// wider intermediate=17408, or the 6:1 GQA — that the smaller
// Qwen3.5-0.8B path doesn't exercise). Tracked as a follow-up; this
// suite stays ENABLED so the failure is visible in every bisect run
// rather than disappearing behind a skip-reason.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

/// Local-cache predicate — Qwen3.6-27B is too big to download inside
/// the integration run on a typical dev box. Pre-download via
/// `ffai download mlx-community/Qwen3.6-27B-4bit` to enable.
private let qwen36CacheAvailable: Bool = {
    let cache =
        ("~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit"
            as NSString)
        .expandingTildeInPath
    return FileManager.default.fileExists(atPath: cache)
}()

@Suite(
    "Qwen3.6 Text Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites && qwen36CacheAvailable,
        "Qwen3.6 integration requires a cached `mlx-community/Qwen3.6-27B-4bit` (≈ 14 GB; fetch with `ffai download mlx-community/Qwen3.6-27B-4bit`) AND the text-suite group flag enabled."
    )
)
struct Qwen36TextIntegrationTests {

    @Test("Qwen3.6-27B dense GDN hybrid: load + interfaces + forward")
    func loadAndForward() async throws {
        let modelId = "mlx-community/Qwen3.6-27B-4bit"
        let prompt = "The history of the printing press began when"

        // ── 1. Load ──────────────────────────────────────────────────────
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(modelId)
        }

        // ── 2. Interfaces we expect ──────────────────────────────────────
        // Engine selection — Qwen3.6 routes through the Qwen3.5 dense GDN
        // hybrid engine (same `Qwen3_5ForConditionalGeneration` arch);
        // every other family slot stays nil.
        #expect(m.qwen35 != nil)
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)
        #expect(m.falconH1 == nil)
        #expect(m.nemotronH == nil)
        #expect(m.graniteMoeHybrid == nil)
        #expect(m.mamba2 == nil)

        // Shapes from the published Qwen3.6-27B-4bit `text_config`.
        #expect(m.engine.hidden == 5120)
        #expect(m.engine.nLayers == 64)
        #expect(m.engine.nHeads == 24)
        #expect(m.engine.nKVHeads == 4)
        #expect(m.engine.vocab == 248_320)
        if let q = m.qwen35 {
            // GDN mixer geometry — Qwen3.6 inherits the same head_dim 256
            // and full_attention_interval = 4 as Qwen3.5.
            #expect(q.headDim == 256)
            #expect(q.convKernel == 4)
            #expect(q.layers.count == 64)
            let gdnCount = q.layers.filter { $0 is Qwen35GDNLayer }.count
            let attnCount = q.layers.filter { $0 is Qwen35AttentionLayer }.count
            #expect(gdnCount == 48)
            #expect(attnCount == 16)
            // 27B-4bit is dense (no MoE; num_local_experts absent).
            #expect(q.hasMoE == false)
        }

        // ── 3. Errors we expect ─────────────────────────────────────────
        // Cache-kind alignment is part of the engine's interface contract:
        // a GDN layer MUST be paired with a Qwen35GDNLayerCache and an
        // attention layer with a KVCache. A mis-pair is a forward-time
        // crash, so we assert the alignment up-front rather than waiting
        // for forward(...) to panic.
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 64)
        if let q = m.qwen35 {
            for (i, layer) in q.layers.enumerated() {
                switch layer {
                case is Qwen35GDNLayer:
                    #expect(caches[i] is Qwen35GDNLayerCache)
                case is Qwen35AttentionLayer:
                    #expect(caches[i] is KVCache)
                default:
                    Issue.record("Qwen3.6: unexpected layer kind at index \(i)")
                }
            }
        }

        // ── 4. Forward pass produces expected output ────────────────────
        // Single-token forward smoke test — logits should be finite +
        // non-degenerate. A 27B model with 64 layers is slow enough that
        // a full multi-hundred-token greedy decode would dominate the
        // integration run; we keep the coherence assertion to a short
        // 32-token greedy decode (the bench file does the long sweeps).
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 248_320)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        // Strict ordering — all-equal logits indicate a forward-pass
        // numerical bug (e.g. an unscaled multiplier, mis-wired norm).
        #expect(top[0].1 > top[4].1)

        // Short greedy decode for the coherence-assertion limb of the
        // four-core integration check. Greedy temperature=0 + a 27B
        // base-tier model produces stable English prose for the first
        // few tens of tokens; this catches catastrophic regressions
        // (NaN logits, stuck-argmax loops) without the multi-minute
        // cost of a 200-token decode on a 27B 4-bit model.
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 32, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)

        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("Qwen3.6-27B-4bit decoded output: \(decoded)")

        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 16,
            minUniqueRatio: 0.15,
            label: "Qwen3.6-27B 4bit dense GDN hybrid"
        )
    }
}
