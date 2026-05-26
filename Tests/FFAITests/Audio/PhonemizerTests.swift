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
// Phonemizer protocol + registry — exercise the BYO surface without a
// bundled implementation. The actual phonemizer-content tests land
// when (if) FFAI ships an in-tree CMUdict / Misaki / espeak port.

import Foundation
import Testing

@testable import FFAI

@Suite("Phonemizer")
struct PhonemizerTests {

    /// A tiny test-only phonemizer used to exercise the registry.
    /// Maps each letter to itself, lowercase-only.
    private struct EchoPhonemizer: Phonemizer {
        let languages: [String]
        let phonemeVocabulary: [String: Int] = [
            "h": 1, "e": 2, "l": 3, "o": 4, " ": 0,
        ]
        func phonemize(_ text: String, language: String?) throws -> String {
            return text.lowercased()
        }
    }

    @Test("Phonemizer — default tokenize uses single-char vocab lookup")
    func defaultTokenize() throws {
        let p = EchoPhonemizer(languages: ["en"])
        let phones = try p.phonemize("Hello", language: "en")
        #expect(phones == "hello")
        let ids = try p.tokenize(phonemes: phones)
        #expect(ids == [1, 2, 3, 3, 4])
    }

    @Test("Phonemizer — default tokenize maps unknown chars to 0")
    func tokenizeUnknownIsZero() throws {
        let p = EchoPhonemizer(languages: ["en"])
        let ids = try p.tokenize(phonemes: "h?o")
        #expect(ids == [1, 0, 4])
    }

    @Test("PhonemizerError — descriptions render")
    func errorDescriptions() {
        let cases: [PhonemizerError] = [
            .languageNotSupported(requested: "kl", supported: ["en", "fr"]),
            .phonemizationFailed(text: "hi", reason: "oops"),
            .noProviderRegistered(forLanguage: "kl"),
            .missingResource(name: "cmudict.tsv"),
        ]
        for c in cases { #expect(!String(describing: c).isEmpty) }
    }

    @Test("PhonemizerRegistry — per-language registration resolves")
    func registryPerLanguage() async throws {
        // Use a dedicated registry instance to avoid touching the global one.
        let reg = PhonemizerRegistry()
        let en = EchoPhonemizer(languages: ["en-us"])
        await reg.register(en, for: ["en-us"])

        let resolved = try await reg.provider(forLanguage: "en-us")
        let phones = try resolved.phonemize("World", language: "en-us")
        #expect(phones == "world")
    }

    @Test("PhonemizerRegistry — fallback used when per-language missing")
    func registryFallback() async throws {
        let reg = PhonemizerRegistry()
        let fallback = EchoPhonemizer(languages: ["*"])
        await reg.registerFallback(fallback)

        // No "kl" provider registered → fallback is returned.
        let resolved = try await reg.provider(forLanguage: "kl")
        let phones = try resolved.phonemize("Klingon", language: "kl")
        #expect(phones == "klingon")
    }

    @Test("PhonemizerRegistry — throws noProviderRegistered when empty")
    func registryEmptyThrows() async {
        let reg = PhonemizerRegistry()
        await #expect(throws: PhonemizerError.self) {
            _ = try await reg.provider(forLanguage: "en")
        }
    }
}
