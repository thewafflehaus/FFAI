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
// End-to-end validation of the incrementally-grown KV cache. The
// per-cache unit tests in `Tests/FFAITests/KVCache/KVCacheTests.swift`
// prove the realloc + strided-copy preserves data in isolation; this
// test proves a REAL model stays coherent when the cache grows
// repeatedly DURING decode — i.e. that SDPA reads the freshly-grown
// buffer at the new stride correctly, on the same command buffer the
// growth happened on.
//
// Mechanism: temporarily lower `KVCache.defaultInitialCapacity` to a
// tiny value so a normal ~64-token greedy decode crosses several
// growth boundaries (16 → 32 → 64 → 128). If the strided re-layout or
// the SDPA-stride bookkeeping were wrong, attention would read garbage
// past the first growth and the output would degenerate — the
// coherence assertion catches that. The global knob is restored in a
// `defer` so no other suite is affected (integration tests are
// serialized).

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "KV Cache Growth Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableTextSuites,
        IntegrationGroupGating.textSkipReason)
)
struct KVCacheGrowthIntegrationTests {

    @Test("decode stays coherent across repeated KV-cache growth boundaries")
    func growsDuringDecodeWithoutCorruption() async throws {
        let modelId = "mlx-community/Qwen3.5-0.8B-4bit"
        let prompt = "The history of the printing press began when"

        // Force growth during decode: start the cache at 16 slots so a
        // 64-token generation crosses 16 → 32 → 64 → 128. Restore the
        // production default afterwards.
        let savedDefault = KVCache.defaultInitialCapacity
        KVCache.defaultInitialCapacity = 16
        defer { KVCache.defaultInitialCapacity = savedDefault }

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        #expect(m.qwen35 != nil)

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 64, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)

        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("KV-growth decode output: \(decoded)")

        // Same coherence bar as the dense Qwen3.5 suite (0.8B greedy
        // loops on short patterns — the floor catches degeneration, not
        // the loop). If growth corrupted the cache, attention past the
        // first boundary would produce stuck-argmax / gibberish that
        // trips this.
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            minUniqueRatio: 0.05,
            label: "Qwen3.5-0.8B 4bit (KV growth forced @ initialCapacity=16)"
        )
    }
}
