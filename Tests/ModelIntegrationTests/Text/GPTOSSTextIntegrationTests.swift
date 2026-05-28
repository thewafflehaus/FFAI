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
// Slow integration test: loads the smallest FFAI-runnable GPT-OSS-20B
// checkpoint (a 4-bit / MXFP4-quantized tag) and runs end-to-end greedy
// generation. Skipped automatically if no checkpoint is cached.
//
// GPT-OSS-20B is OpenAI's mixture-of-experts transformer — a 24-layer
// MoE stack (~20B total / ~3.6B active params) with three structural
// features this test exercises end-to-end:
//
//   1. An alternating attention schedule — `layer_types` assigns each
//      layer "sliding_attention" or "full_attention"; sliding layers
//      get a `.window` eviction KV cache, full layers stay unbounded.
//   2. Learned per-head attention sinks — a `self_attn.sinks` logit
//      vector folded into the softmax denominator as a per-head
//      post-hoc rescale of the SDPA output (the d64 SDPA kernel has no
//      native learned-sink support).
//   3. Bias-corrected q/k/v/o projections (`attention_bias`).
//
// The MoE experts ship MXFP4-quantized; the loader transcodes them to
// FFAI's affine-int4 format at load time (see GPTOSSMoE.swift).
//
// `loan-star/gpt-oss-20b-mlx-4Bit` (~11 GB) is the target — it fits
// in the dev box's memory budget. The transcode is lossy, so the test
// asserts pipeline-level *coherence* (the FFAI integration contract),
// not cross-implementation token parity. (We may switch to a
// `ekryski/` MXFP4 → affine-int4 conversion in future once we have
// our own published checkpoint; one canonical target keeps the
// suite consistent with the rest of the integration coverage.)
//
// Env-gated: GPT-OSS-20B is FFAI's heaviest integration target. A
// debug-build greedy decode of a ~20B MoE runs for minutes, so this
// suite runs only when `FFAI_BUILD_MACHINE` is set — keeping it out of
// the routine `make test-integration` gate and matching the GPT-OSS
// row of `ModelKVCacheMatrixIntegrationTests`.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "GPTOSS Integration", .serialized,
    .enabled(
        if: ProcessInfo.processInfo.environment["FFAI_BUILD_MACHINE"] != nil
            && IntegrationGroupGating.enableTextSuites,
        "GPT-OSS-20B is build-machine-only; set FFAI_BUILD_MACHINE AND flip IntegrationGroupGating.enableTextSuites = true")
)
struct GPTOSSTextIntegrationTests {

    @Test("load + greedy generate produces coherent MoE output")
    func loadAndGenerate() async throws {
        let modelId = "loan-star/gpt-oss-20b-mlx-4Bit"
        let prompt = "The history of the printing press began when"

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Engine should be GPT-OSS (not a dense / hybrid family).
        #expect(m.gptOSS != nil)
        #expect(m.jamba == nil)
        #expect(m.qwen3 == nil)

        // Shapes from the published GPT-OSS-20B config.
        #expect(m.engine.hidden == 2880)
        #expect(m.engine.nLayers == 24)
        #expect(m.engine.nHeads == 64)
        #expect(m.engine.nKVHeads == 8)
        #expect(m.engine.headDim == 64)
        #expect(m.engine.vocab == 201_088)

        if let g = m.gptOSS {
            // The alternating sliding/full attention schedule.
            #expect(g.attnKinds.count == 24)
            let slidingCount = g.attnKinds.filter { $0 == .sliding }.count
            let fullCount = g.attnKinds.filter { $0 == .full }.count
            #expect(slidingCount == 12)
            #expect(fullCount == 12)
            #expect(g.slidingWindow == 128)
            // Every layer carries a 32-expert MoE FFN, top-4 routing.
            #expect(g.layers.count == 24)
            for layer in g.layers {
                #expect(layer.moe.experts.count == 32)
                #expect(layer.moe.topK == 4)
            }
        }

        // ── Per-layer cache kinds match the attention schedule ────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 24)
        if let g = m.gptOSS {
            for (i, kind) in g.attnKinds.enumerated() {
                guard let kv = caches[i] as? KVCache else {
                    Issue.record("expected KVCache at layer \(i)")
                    continue
                }
                switch kind {
                case .sliding:
                    #expect(
                        kv.effectiveMaxSize == 128,
                        "sliding layer \(i) should cap at the 128-token window")
                case .full:
                    #expect(
                        kv.effectiveMaxSize > 128,
                        "full-attention layer \(i) should be unbounded")
                }
            }
        }

        // ── Forward-shape smoke test ──────────────────────────────────
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 201_088)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        // Degenerate (all-equal) logits → a forward-pass numerical bug.
        #expect(top[0].1 > top[4].1)

        // ── End-to-end greedy generation ──────────────────────────────
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)

        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("GPT-OSS-20B decoded output: \(decoded)")

        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            label: "GPT-OSS-20B 4bit"
        )
    }
}
