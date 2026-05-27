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
// End-to-end sliding-window KV cache integration. Verifies that:
//
//   1. A small model loads with .window eviction wired through
//      LoadOptions.kvEviction.
//   2. Greedy decode through a prompt that EXCEEDS the window size
//      still produces coherent output (the cache rotates older
//      positions out as expected, the kernels see the rotated layout
//      transparently).
//   3. effectiveMaxSize on the layer caches reports the window's
//      maxSize, not the model's max_position_embeddings — sanity
//      check that the policy is actually applied.
//
// Llama 3.2 1B is small + license-free + already locally cached on
// the dev machines that run this suite. If the checkpoint isn't
// available the test skips gracefully.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Sliding Window KV Cache Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableQuantizedSuites,
        IntegrationGroupGating.quantizedSkipReason)
)
struct SlidingWindowIntegrationTests {

    @Test("Llama 3.2 1B with .window(maxSize: 64) produces coherent output beyond window")
    func windowedDecode() async throws {
        let modelId = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        // Prompt is short; we just want the *generation* to push the
        // cache past maxSize so we exercise the rotation path. The
        // window is intentionally small (64) so 64 generated tokens
        // wraps the buffer roughly once. The first 4 slots are pinned
        // as attention sinks (Xiao et al. 2023 default).
        let prompt = "Once upon a time"
        let maxTokens = 200
        let windowSize = 64
        let keep = 4

        let opts = LoadOptions(
            kvCache: .raw,
            kvEviction: .window(maxSize: windowSize, keep: keep)
        )
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(modelId, options: opts)
        }

        // Sanity: every layer cache reports the window we asked for,
        // not the model's max_position_embeddings.
        let caches = m.engine.makeLayerCaches()
        for c in caches {
            guard let kv = c as? any KVCacheProtocol else { continue }
            #expect(
                kv.effectiveMaxSize == windowSize,
                "cache effectiveMaxSize \(kv.effectiveMaxSize) should match window size \(windowSize)"
            )
            switch kv.eviction {
            case .window(let m, let k):
                #expect(m == windowSize)
                #expect(k == keep)
            case .unbounded:
                Issue.record("expected .window eviction; got .unbounded")
            }
        }

        // Drive greedy decode through Model.generate. The cache rolls
        // past `windowSize` once we generate enough tokens; output
        // should remain coherent because RoPE was baked in at
        // insertion time and the kernels see the rotated layout.
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(
            result.generatedTokens, label: "Llama 3.2 1B sliding-window(64, keep=4)")
    }
}
