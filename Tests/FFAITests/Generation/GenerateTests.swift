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
// Generate — value-type accessors on GenerationResult and
// GenerationChunk. The end-to-end driveGeneration loop is exercised by
// every ModelTests integration suite; this file covers the small
// public types that ride on top of it.

import Foundation
import Testing
@testable import FFAI

@Suite("Generate")
struct GenerateTests {

    @Test("GenerationResult convenience accessors read through stats")
    func generationResultAccessors() {
        let nonZeroStats = GenerationStats(
            promptTokens: 1, generatedTokens: 4, contextSize: 4096,
            prefillTimeS: 0.01, decodeTimeS: 1.0, timeToFirstTokenMs: 10.0,
            steadyTokensPerSecond: 4.5,
            baselineGPUBytes: 0, postPrefillGPUBytes: 0, postDecodeGPUBytes: 0,
            prefillPeakGPUBytes: 0, decodePeakGPUBytes: 0,
            wiredTicketBytes: 0, weightsBytes: 0,
            kvCacheAllocatedBytes: 0, kvCacheUsedBytes: 0,
            thinkPerplexity: nil, genPerplexity: nil,
            thinkKLDivergence: nil, genKLDivergence: nil,
            thinkTokenCount: nil, genTokenCount: nil
        )
        let r = GenerationResult(
            promptTokens: [1], generatedTokens: [4, 5, 6, 7],
            text: "abcd", stats: nonZeroStats
        )
        #expect(r.tokensPerSecond == 4)
        #expect(r.prefillTokensPerSecond == 100)
        #expect(r.prefillTimeS == 0.01)
        #expect(r.decodeTimeS == 1.0)
        #expect(r.timeToFirstTokenS == 0.01)
    }

    @Test("GenerationChunk.isFinal — true iff stats attached")
    func generationChunkIsFinal() {
        let mid = GenerationChunk(text: "hi", tokens: [42], position: 5)
        #expect(mid.isFinal == false)
        #expect(mid.stats == nil)

        let dummyStats = GenerationStats(
            promptTokens: 1, generatedTokens: 1, contextSize: 0,
            prefillTimeS: 0, decodeTimeS: 0, timeToFirstTokenMs: 0,
            steadyTokensPerSecond: nil,
            baselineGPUBytes: 0, postPrefillGPUBytes: 0, postDecodeGPUBytes: 0,
            prefillPeakGPUBytes: 0, decodePeakGPUBytes: 0,
            wiredTicketBytes: 0, weightsBytes: 0,
            kvCacheAllocatedBytes: 0, kvCacheUsedBytes: 0,
            thinkPerplexity: nil, genPerplexity: nil,
            thinkKLDivergence: nil, genKLDivergence: nil,
            thinkTokenCount: nil, genTokenCount: nil
        )
        let final = GenerationChunk(text: "", tokens: [], position: 5,
                                    stats: dummyStats)
        #expect(final.isFinal == true)
    }
}
