// GenerationStatsTests — derivation properties and `formatted()`
// rendering. Real wallclock + memory numbers are exercised by the
// integration tests; here we hand-build values + assert the math.

import Foundation
import Testing
@testable import FFAI

@Suite("GenerationStats")
struct GenerationStatsTests {

    private func makeStats(
        promptTokens: Int = 5, generatedTokens: Int = 16,
        prefillS: Double = 0.1, decodeS: Double = 1.0,
        baseline: Int = 1_000_000_000, postPrefill: Int = 1_100_000_000,
        postDecode: Int = 1_120_000_000,
        prefillPeak: Int = 1_150_000_000, decodePeak: Int = 1_125_000_000
    ) -> GenerationStats {
        GenerationStats(
            promptTokens: promptTokens, generatedTokens: generatedTokens,
            contextSize: 4096,
            prefillTimeS: prefillS, decodeTimeS: decodeS,
            timeToFirstTokenMs: prefillS * 1000,
            steadyTokensPerSecond: 18.0,
            baselineGPUBytes: baseline,
            postPrefillGPUBytes: postPrefill,
            postDecodeGPUBytes: postDecode,
            prefillPeakGPUBytes: prefillPeak,
            decodePeakGPUBytes: decodePeak,
            wiredTicketBytes: 16 * 1024 * 1024 * 1024,
            weightsBytes: 800_000_000,
            kvCacheAllocatedBytes: 64 * 1024 * 1024,
            kvCacheUsedBytes: 12 * 1024 * 1024,
            thinkPerplexity: nil, genPerplexity: nil,
            thinkKLDivergence: nil, genKLDivergence: nil,
            thinkTokenCount: nil, genTokenCount: nil
        )
    }

    @Test("Throughput derivations from time + token counts")
    func throughput() {
        let s = makeStats(promptTokens: 100, generatedTokens: 50,
                          prefillS: 1.0, decodeS: 2.0)
        #expect(s.prefillTokensPerSecond == 100.0)
        #expect(s.decodeTokensPerSecond == 25.0)
    }

    @Test("Throughput is 0 when phase time is 0")
    func zeroTime() {
        let s = makeStats(prefillS: 0, decodeS: 0)
        #expect(s.prefillTokensPerSecond == 0)
        #expect(s.decodeTokensPerSecond == 0)
    }

    @Test("Memory derivations: peak / prefill / decode growth")
    func memoryDerivations() {
        let s = makeStats()
        #expect(s.peakGPUBytes == 1_150_000_000)
        #expect(s.prefillGrowthBytes == 100_000_000)
        #expect(s.decodeGrowthBytes == 20_000_000)
    }

    @Test("formatted() emits header + every section")
    func formattedOutput() {
        let s = makeStats()
        let out = s.formatted()
        #expect(out.hasPrefix("[STATS]"))
        for needle in ["prompt:", "generated:", "ttft:", "prefill:",
                       "decode:", "baseline GPU:", "weights:",
                       "KV cache (alloc):", "KV cache (used):",
                       "wired ticket:"] {
            #expect(out.contains(needle), "missing \(needle) in formatted output")
        }
    }

    @Test("Optional perplexity / KLD lines only render when set")
    func optionalLinesOmitted() {
        var s = makeStats()
        #expect(!s.formatted().contains("perplexity"))
        s = GenerationStats(
            promptTokens: s.promptTokens, generatedTokens: s.generatedTokens,
            contextSize: s.contextSize,
            prefillTimeS: s.prefillTimeS, decodeTimeS: s.decodeTimeS,
            timeToFirstTokenMs: s.timeToFirstTokenMs,
            steadyTokensPerSecond: s.steadyTokensPerSecond,
            baselineGPUBytes: s.baselineGPUBytes,
            postPrefillGPUBytes: s.postPrefillGPUBytes,
            postDecodeGPUBytes: s.postDecodeGPUBytes,
            prefillPeakGPUBytes: s.prefillPeakGPUBytes,
            decodePeakGPUBytes: s.decodePeakGPUBytes,
            wiredTicketBytes: s.wiredTicketBytes,
            weightsBytes: s.weightsBytes,
            kvCacheAllocatedBytes: s.kvCacheAllocatedBytes,
            kvCacheUsedBytes: s.kvCacheUsedBytes,
            thinkPerplexity: 5.5, genPerplexity: 6.1,
            thinkKLDivergence: 0.04, genKLDivergence: 0.03,
            thinkTokenCount: 100, genTokenCount: 50
        )
        let out = s.formatted()
        #expect(out.contains("gen perplexity"))
        #expect(out.contains("think perplexity"))
        #expect(out.contains("gen KLD"))
        #expect(out.contains("think KLD"))
        #expect(out.contains("think / gen split"))
    }

    @Test("Steady-state nil renders no row")
    func steadyOmitted() {
        let s = GenerationStats(
            promptTokens: 1, generatedTokens: 5, contextSize: 0,
            prefillTimeS: 0.1, decodeTimeS: 0.5, timeToFirstTokenMs: 100,
            steadyTokensPerSecond: nil,
            baselineGPUBytes: 0, postPrefillGPUBytes: 0, postDecodeGPUBytes: 0,
            prefillPeakGPUBytes: 0, decodePeakGPUBytes: 0,
            wiredTicketBytes: 0, weightsBytes: 0,
            kvCacheAllocatedBytes: 0, kvCacheUsedBytes: 0,
            thinkPerplexity: nil, genPerplexity: nil,
            thinkKLDivergence: nil, genKLDivergence: nil,
            thinkTokenCount: nil, genTokenCount: nil
        )
        #expect(!s.formatted().contains("decode (steady)"))
    }
}
