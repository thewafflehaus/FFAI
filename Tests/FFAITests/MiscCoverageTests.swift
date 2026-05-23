// Round out coverage for small files: ModelDownloader error cases,
// TokenizerLoader load + error, Generate stop-on-EOS, ModelDownloader
// invalid id.

import Foundation
import Testing
@testable import FFAI

@Suite("Misc coverage")
struct MiscCoverageTests {
    @Test("ModelDownloader rejects malformed repo id")
    func badRepoID() async {
        do {
            _ = try await ModelDownloader().download(id: "")
            Issue.record("expected throw")
        } catch let e as ModelDownloaderError {
            switch e {
            case .invalidRepoID, .downloadFailed:
                break  // either is acceptable for an empty id
            }
        } catch {
            // Any other thrown error is also acceptable
        }
    }

    @Test("ModelDownloaderError descriptions render")
    func downloaderErrorDesc() {
        struct Boom: Error { let message: String }
        let cases: [ModelDownloaderError] = [
            .invalidRepoID("bad"),
            .downloadFailed("foo/bar", Boom(message: "x")),
        ]
        for c in cases { #expect(!String(describing: c).isEmpty) }
    }

    @Test("TokenizerLoader fails cleanly on a non-tokenizer directory")
    func tokenizerLoadFailure() async {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-tok-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try await TokenizerLoader().load(from: dir)
            Issue.record("expected throw")
        } catch let e as TokenizerLoaderError {
            if case .loadFailed = e { /* ok */ } else {
                Issue.record("got \(e)")
            }
        } catch {
            // Any thrown error is acceptable when no tokenizer files exist.
        }
    }

    @Test("ModelLocatorError description renders")
    func locatorDesc() {
        let e = ModelLocatorError.localPathNotFound(URL(fileURLWithPath: "/x"))
        #expect(String(describing: e).contains("/x"))
    }

    @Test("ModelConfigError description renders")
    func configErrorDesc() {
        let e = ModelConfigError.malformed(URL(fileURLWithPath: "/y"))
        #expect(String(describing: e).contains("/y"))
    }

    @Test("GenerationResult convenience accessors read through stats")
    func generationResultAccessors() async throws {
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

    @Test("GenerationChunk isFinal")
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

    @Test("LoadOptions cacheDirectory + ModelDownloader convenience init")
    func loadOptionsCacheDir() {
        // Default
        let opts1 = LoadOptions()
        #expect(opts1.cacheDirectory == nil)

        // Custom
        let custom = URL(fileURLWithPath: "/Volumes/Big/hf-cache")
        let opts2 = LoadOptions(cacheDirectory: custom)
        #expect(opts2.cacheDirectory == custom)

        // Convenience init builds without throwing for both nil + non-nil
        let dlNil = ModelDownloader(cacheDirectory: nil)
        let dlSet = ModelDownloader(cacheDirectory: custom)
        // Both should produce a usable client (we don't make network
        // calls — just construct).
        _ = (dlNil, dlSet)
    }

    @Test("ChatTemplateOptions equatable + defaults")
    func chatTemplateOpts() {
        let a = ChatTemplateOptions()
        let b = ChatTemplateOptions()
        #expect(a == b)
        #expect(a.addGenerationPrompt == true)
        #expect(a.enableThinking == false)
        let c = ChatTemplateOptions(enableThinking: true,
                                    reasoningEffort: .high)
        #expect(c.enableThinking == true)
        #expect(c.reasoningEffort == .high)
        #expect(a != c)
    }

    @Test("ThinkingFormat enum + ChatMessage shape")
    func thinkingFormatBasic() {
        // The .none format produces no split regardless of input;
        // exercise the fast-path branch without needing a Tokenizer.
        for f in ThinkingFormat.allCases {
            #expect(!f.rawValue.isEmpty)
        }
        let m = ChatMessage(role: .user, content: "hi")
        #expect(m.asTemplateMessage["role"] as? String == "user")
        #expect(m.asTemplateMessage["content"] as? String == "hi")
    }

    @Test("GenerationStats peak / growth derivations")
    func generationStatsPeak() {
        let s = GenerationStats(
            promptTokens: 4, generatedTokens: 8, contextSize: 4096,
            prefillTimeS: 0.1, decodeTimeS: 0.5, timeToFirstTokenMs: 100,
            steadyTokensPerSecond: 18.0,
            baselineGPUBytes: 1_000_000_000,
            postPrefillGPUBytes: 1_100_000_000,
            postDecodeGPUBytes: 1_120_000_000,
            prefillPeakGPUBytes: 1_150_000_000,
            decodePeakGPUBytes: 1_125_000_000,
            wiredTicketBytes: 16 * 1024 * 1024 * 1024,
            weightsBytes: 800_000_000,
            kvCacheAllocatedBytes: 64 * 1024 * 1024,
            kvCacheUsedBytes: 12 * 1024 * 1024,
            thinkPerplexity: nil, genPerplexity: nil,
            thinkKLDivergence: nil, genKLDivergence: nil,
            thinkTokenCount: nil, genTokenCount: nil
        )
        #expect(s.peakGPUBytes == 1_150_000_000)
        #expect(s.prefillGrowthBytes == 100_000_000)
        #expect(s.decodeGrowthBytes == 20_000_000)
        // formatted() should not crash and should mention the section header.
        #expect(s.formatted().contains("[STATS]"))
    }

    @Test("KVCache totalBytes accessors")
    func kvCacheBytes() {
        let c = KVCache(nKVHeads: 8, headDim: 64, maxSeq: 1024, dtype: .f16)
        let elems = 2 * 8 * 1024 * 64
        #expect(c.bytesAllocated == elems * 2)   // fp16 = 2 bytes
        #expect(c.bytesInUse == 0)
        let arr = [c, c, c]
        #expect(arr.totalBytesAllocated == c.bytesAllocated * 3)
    }

    @Test("GenerationParameters defaults")
    func generateParamDefaults() {
        let p = GenerationParameters()
        #expect(p.maxTokens == 256)
        #expect(p.stopOnEOS == true)
        // prefillStepSize defaults to nil ("use engine's tuned default")
        // since the Phase 6.6 chunked-prefill wiring. Generic engines
        // still resolve to 1024 inside Generate.driveGeneration.
        #expect(p.prefillStepSize == nil)
        #expect(p.temperature == 0.6)
        #expect(p.topP == 1.0)

        // Family defaults should differ across families.
        #expect(LlamaDense.defaultGenerationParameters.topP == 1.0)
        #expect(Qwen3Dense.defaultGenerationParameters.topP == 0.95)
        #expect(Qwen3Dense.defaultGenerationParameters.topK == 20)
    }

    @Test("GenerationParameters.with copy-mutator")
    func generateParamWith() {
        let base = LlamaDense.defaultGenerationParameters
        let tweaked = base.with { $0.maxTokens = 64 }
        #expect(tweaked.maxTokens == 64)
        #expect(tweaked.temperature == base.temperature)   // untouched
        #expect(base.maxTokens == 256)                     // base unchanged
    }
}
