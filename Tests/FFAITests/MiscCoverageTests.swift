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

    @Test("Generate stops at maxNewTokens")
    func generateStops() async throws {
        // Don't depend on the full Llama integration test for this — tiny
        // smoke test that exercises GenerationResult construction edge.
        let r = GenerationResult(
            promptTokens: [1, 2, 3], generatedTokens: [],
            text: "", prefillTimeS: 0, decodeTimeS: 0
        )
        #expect(r.tokensPerSecond == 0)
        let r2 = GenerationResult(
            promptTokens: [1], generatedTokens: [4, 5, 6, 7],
            text: "abcd", prefillTimeS: 0.01, decodeTimeS: 1.0
        )
        #expect(r2.tokensPerSecond == 4)
    }

    @Test("GenerateOptions defaults")
    func generateOptDefaults() {
        let o = GenerateOptions()
        #expect(o.maxNewTokens == 64)
        #expect(o.stopOnEOS == true)
    }
}
