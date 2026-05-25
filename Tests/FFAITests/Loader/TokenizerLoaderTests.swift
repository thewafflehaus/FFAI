// TokenizerLoader — failure paths. The success path is exercised by
// every integration test that decodes tokens; this file covers the
// "load from a directory with no tokenizer files" failure case.

import Foundation
import Testing
@testable import FFAI

@Suite("TokenizerLoader")
struct TokenizerLoaderTests {

    @Test("fails cleanly on a directory with no tokenizer files")
    func loadFailureOnEmptyDirectory() async {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-tok-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            _ = try await TokenizerLoader().load(from: dir)
            Issue.record("expected throw")
        } catch let e as TokenizerLoaderError {
            if case .loadFailed = e {
                // ok
            } else {
                Issue.record("expected .loadFailed, got \(e)")
            }
        } catch {
            // Any other thrown error is acceptable when no tokenizer
            // files exist — the contract is just "don't succeed".
        }
    }
}
