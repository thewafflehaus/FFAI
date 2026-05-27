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
// Slow integration test for Gemma 3 1B. First-light verification that
// the family's architectural quirks (4 norms per block, Gemma RMSNorm
// +1 fold, alternating RoPE base, per-head q/k norms, sqrt(hidden)
// embed scale, GELU MLP, queryPreAttnScalar, per-layer sliding-window
// KV cache) compose into coherent generated text on a real
// checkpoint.
//
// Uses the mlx-community 1B-it bf16 conversion. Skipped if not
// available locally.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Gemma3 Text Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct Gemma3TextIntegrationTests {

    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        let modelId = "mlx-community/gemma-3-1b-it-4bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 200

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // 1B canonical shapes (from HF config.json):
        //   hidden = 1152, nLayers = 26, nHeads = 4, nKVHeads = 1, headDim = 256.
        #expect(m.engine.hidden == 1152)
        #expect(m.engine.nLayers == 26)
        #expect(m.engine.nHeads == 4)
        #expect(m.engine.nKVHeads == 1)
        #expect(m.engine.headDim == 256)

        // Verify per-layer KV cache eviction policy: sliding layers
        // get .window(slidingWindow), global layers stay unbounded.
        // Pattern is `(i + 1) % slidingWindowPattern == 0` ⇒ global.
        // 1B default sliding_window_pattern = 6.
        let caches = m.engine.makeLayerCaches()
        var slidingCount = 0
        var globalCount = 0
        for (i, c) in caches.enumerated() {
            guard let kv = c as? any KVCacheProtocol else { continue }
            let isGlobal = (i + 1) % 6 == 0
            switch kv.eviction {
            case .window:
                #expect(!isGlobal, "layer \(i): expected global (.unbounded) but got .window")
                slidingCount += 1
            case .unbounded:
                #expect(isGlobal, "layer \(i): expected sliding (.window) but got .unbounded")
                globalCount += 1
            }
        }
        // 26 layers, pattern 6: layers 5, 11, 17, 23 are global → 4 global,
        // 22 sliding.
        #expect(globalCount == 4)
        #expect(slidingCount == 22)

        // Greedy decode + hard coherence assert. The first-light
        // all-NaN bug was a bf16 tanh overflow inside the GELU
        // template; fixed in metaltile preamble.rs by computing GELU
        // in fp32 with the tanh argument clamped to ±15. See
        // papers/gemma3-coherence-investigation-2026-05-19.md.
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, label: "Gemma 3 1B-it bf16")
    }
}
