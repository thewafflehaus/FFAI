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
