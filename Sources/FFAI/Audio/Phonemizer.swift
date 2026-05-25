// Phonemizer — text-to-phoneme front-end protocol for TTS families.
//
// Some TTS families (Kokoro / StyleTTS2 in particular) take *phonemes*
// rather than raw text as input — the acoustic model is trained on a
// phoneme alphabet (typically IPA or a CMUdict-derived set). The
// text-to-phoneme step is a separate concern:
//
//   • CMUdict-based (English) — lookup table + letter-to-sound rules,
//     small and pure Swift, redistribution-friendly (BSD-ish CMU
//     licence).
//   • espeak-ng — large multi-lingual rule engine, GPL-licensed →
//     can't bundle by default; needs a separate dependency users opt in to.
//   • Misaki (g2p) — Apache 2.0 G2P with CMUdict tables and rules;
//     the same front-end Kokoro upstream uses. Practical default for
//     English Kokoro voices once we ship the bundled tables.
//   • Neural G2P models (byT5 / piper-phonemize) — apache / mit, but
//     a separate model weight that has to be downloaded.
//
// FFAI's stance: define a `Phonemizer` protocol, ship one or more
// permissive-licence implementations in-tree for the common case
// (English CMUdict), and let users plug their own in for everything
// else. Mirrors the approach mlx-audio-swift takes with its
// `TextProcessor` protocol — see `MLXAudioTTS/.../KokoroMultilingual-
// Processor.swift` for the multi-lingual lexicon-lookup variant.

import Foundation

/// Convert a string of natural-language text into the phoneme alphabet
/// a downstream TTS acoustic model consumes. Each implementation is
/// licensed independently; FFAI's bundled defaults are permissive
/// (CMUdict-based English), and callers can BYO for other languages.
public protocol Phonemizer: Sendable {
    /// Languages this phonemizer supports, e.g. `["en-us", "en-gb"]`.
    /// Returned as BCP-47-style tags. An empty array means "language
    /// detection is internal / not exposed."
    var languages: [String] { get }

    /// Convert `text` into a phoneme string for the requested
    /// `language`. `nil` language asks the phonemizer to use its
    /// default (or to auto-detect, if supported). Throws
    /// `PhonemizerError.languageNotSupported` if the requested
    /// language isn't in `languages`.
    ///
    /// The phoneme alphabet is implementation-specific (IPA, ARPAbet,
    /// CMUdict, X-SAMPA …). The caller is expected to use a
    /// phonemizer that emits the alphabet its downstream acoustic
    /// model was trained on.
    func phonemize(_ text: String, language: String?) throws -> String

    /// Tokenise a phoneme string into the integer ids the acoustic
    /// model's text encoder consumes. The default implementation maps
    /// individual phoneme characters through `phonemeVocabulary`; rich
    /// phonemizers may override (e.g. to merge digraphs).
    func tokenize(phonemes: String) throws -> [Int]

    /// Phoneme → integer id mapping. The downstream acoustic model
    /// publishes this alongside its weights; the phonemizer must
    /// produce ids in this vocabulary or the model will mis-decode.
    var phonemeVocabulary: [String: Int] { get }
}

public extension Phonemizer {
    /// Default `tokenize` implementation — single-character lookup.
    /// Unknown characters map to 0 (treated as silence by most models).
    func tokenize(phonemes: String) throws -> [Int] {
        var ids: [Int] = []
        ids.reserveCapacity(phonemes.count)
        let vocab = phonemeVocabulary
        for ch in phonemes {
            ids.append(vocab[String(ch)] ?? 0)
        }
        return ids
    }
}

/// Errors raised by `Phonemizer` implementations and the dispatch
/// helpers around them.
public enum PhonemizerError: Error, CustomStringConvertible {
    case languageNotSupported(requested: String, supported: [String])
    case phonemizationFailed(text: String, reason: String)
    case noProviderRegistered(forLanguage: String)
    case missingResource(name: String)

    public var description: String {
        switch self {
        case .languageNotSupported(let r, let s):
            return "Phonemizer: language '\(r)' not supported — "
                + "supported: \(s.joined(separator: ", "))"
        case .phonemizationFailed(let t, let r):
            return "Phonemizer: failed to phonemize \"\(t)\" — \(r)"
        case .noProviderRegistered(let lang):
            return "Phonemizer: no provider registered for language '\(lang)'"
        case .missingResource(let n):
            return "Phonemizer: missing required resource '\(n)'"
        }
    }
}

// ─── BYO Phonemizer registry ─────────────────────────────────────────────

/// Lightweight registry callers use to supply their own `Phonemizer`
/// without having to thread it through every TTS constructor. The
/// registry is global, thread-safe, and overrides FFAI's built-in
/// defaults (when those land) on a per-language basis.
///
/// Mirrors mlx-audio-swift's BYO pattern — Kokoro upstream lets users
/// supply a `TextProcessor` instead of relying on the bundled one,
/// which is essential for the multi-lingual lexicon path.
public actor PhonemizerRegistry {
    public static let shared = PhonemizerRegistry()

    private var providers: [String: any Phonemizer] = [:]
    private var fallback: (any Phonemizer)?

    /// Register `phonemizer` as the provider for one or more languages.
    /// Subsequent `provider(forLanguage:)` calls for those languages
    /// return this instance.
    public func register(_ phonemizer: any Phonemizer, for languages: [String]) {
        for lang in languages {
            providers[lang] = phonemizer
        }
    }

    /// Register `phonemizer` as the fallback used when no per-language
    /// provider is registered. Pass `nil` to clear the fallback.
    public func registerFallback(_ phonemizer: (any Phonemizer)?) {
        fallback = phonemizer
    }

    /// Resolve the phonemizer for `language`. Returns the
    /// per-language provider if one is registered, then the fallback,
    /// otherwise throws `noProviderRegistered`.
    public func provider(forLanguage language: String) throws -> any Phonemizer {
        if let p = providers[language] { return p }
        if let f = fallback { return f }
        throw PhonemizerError.noProviderRegistered(forLanguage: language)
    }
}
