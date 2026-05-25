// Audio-side test helpers — checkpoint resolution, waveform fixtures,
// and STT phrase assertion shared by every audio integration suite.
//
// Three responsibilities:
//
//  1. `resolveCheckpoint(...)` — resolve a checkpoint directory. Some
//     audio checkpoints are only present in the HF-cache *mlx-audio*
//     sibling layout (`~/.cache/huggingface/hub/mlx-audio/<repo>/`),
//     a flat snapshot directory rather than the
//     `models--org--repo/snapshots/` blob layout `ModelLocator`
//     downloads into. When a family's HF-hub cache entry is incomplete
//     (index.json present, shards missing) the suites point
//     `ModelLocator` at the complete mlx-audio directory as a local
//     path. The helper returns the first candidate that exists and
//     looks complete (a config + at least one weight file).
//
//  2. Waveform fixtures — `clean001Waveform()` ("Sure, I can help you
//     with that.") and `conversationalAWaveform()` (~13 s
//     multi-sentence dialogue) loaded from `Tests/Resources/`.
//
//  3. `expectMentionsSureICanHelp(...)` — STT phrase assertion for the
//     clean_001 fixture, with flexible punctuation + capitalisation
//     matching ("Sure I can help you with that", "sure, i can help
//     you with that.", "Sure! I can help you with that..." all pass).
//
// Unlike the old catch-and-skip helpers, nothing here swallows a load
// failure: a missing checkpoint surfaces as a thrown error so the
// test FAILS rather than silently passing.

import Foundation
import Testing
import FFAI

public enum AudioTestHelpers {

    // MARK: - Checkpoint resolution

    /// Root of the HF cache (`~/.cache/huggingface/hub`).
    public static var hfCacheRoot: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/huggingface/hub")
    }

    /// A flat snapshot directory under the `mlx-audio` sibling cache,
    /// e.g. `mlx-audio/mlx-community_Kokoro-82M-bf16`. `repoSlug` is the
    /// repo id with `/` replaced by `_`.
    public static func mlxAudioDir(_ repoSlug: String) -> URL {
        hfCacheRoot.appendingPathComponent("mlx-audio/\(repoSlug)")
    }

    /// True when `dir` looks like a usable checkpoint snapshot: a
    /// `config.json` plus at least one weight file (`*.safetensors`).
    public static func isCompleteSnapshot(_ dir: URL) -> Bool {
        let fm = FileManager.default
        let config = dir.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: config.path) else { return false }
        guard let entries = try? fm.contentsOfDirectory(atPath: dir.path)
        else { return false }
        return entries.contains { $0.hasSuffix(".safetensors") }
    }

    /// Resolve a checkpoint directory, preferring `mlx-audio` slugs
    /// that are already complete on disk, then falling back to HF repo
    /// ids (which `ModelLocator` downloads or cache-hits). Every
    /// candidate is tried in order; the first that resolves to a
    /// complete snapshot wins. Throws if none resolve — the caller
    /// lets that fail the test.
    ///
    /// `mlxAudioSlugs` are checked first as local paths (no network);
    /// `repoIds` are HF ids tried after.
    public static func resolveCheckpoint(
        mlxAudioSlugs: [String] = [],
        repoIds: [String] = []
    ) async throws -> URL {
        var lastError: Error?
        // 1. Complete mlx-audio snapshots on disk — pure local, no network.
        for slug in mlxAudioSlugs {
            let dir = mlxAudioDir(slug)
            if isCompleteSnapshot(dir) { return dir }
        }
        // 2. HF repo ids — ModelLocator downloads or cache-hits.
        let locator = ModelLocator()
        for repoId in repoIds {
            do {
                return try await ModelLoadLock.shared.loadSerially {
                    try await locator.resolve(idOrPath: repoId)
                }
            } catch {
                lastError = error
            }
        }
        throw lastError ?? AudioTestHelpersError.noCheckpointResolved(
            mlxAudioSlugs + repoIds)
    }

    // MARK: - Waveform fixtures

    /// Load the bundled 16 kHz speech fixture as a mono float waveform.
    /// "Sure, I can help you with that." — clean synthetic speech, 1.85 s.
    /// Kept for VAD / STS / STT suites whose assertions are tuned to
    /// this clip (see `expectMentionsSureICanHelp(...)`).
    public static func clean001Waveform() throws -> [Float] {
        let url = resourceURL("clean_001.wav")
        return try AudioPreprocessing.loadWaveform(url: url, targetRate: 16_000)
    }

    /// Load the bundled conversational speech fixture as a mono 16 kHz
    /// float waveform. ~13 s of multi-sentence dialogue (24 kHz source,
    /// resampled at load), giving STT suites more text to assert
    /// transcription quality against than the 1.85 s "Sure, I…" clip.
    /// Sourced from ekryski/mlx-audio-swift @ ek/audio-benchmarks.
    public static func conversationalAWaveform() throws -> [Float] {
        let url = resourceURL("conversational_a.wav")
        return try AudioPreprocessing.loadWaveform(url: url, targetRate: 16_000)
    }

    // MARK: - STT phrase assertion

    /// Assert that an STT transcription of the `clean_001.wav` fixture
    /// contains "Sure I can help you with that" — case- and
    /// punctuation-insensitive. The clip's ground truth is exactly
    /// "Sure, I can help you with that." but model outputs vary in
    /// capitalisation, trailing punctuation, and quoting:
    ///
    ///   "Sure, I can help you with that."  ✓
    ///   "sure i can help you with that"    ✓
    ///   "Sure! I can help you with that..." ✓
    ///   "I can help you" (missing "sure")  ✗
    ///
    /// Uses `normalizeForMatch` from TextTestHelpers — lowercases,
    /// strips punctuation, collapses whitespace.
    public static func expectMentionsSureICanHelp(
        _ text: String, label: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let normalized = normalizeForMatch(text)
        let expected = "sure i can help you with that"
        let comment = Comment(
            rawValue: "\(label): STT output should contain \"Sure, I can help you with that.\" "
                + "(case + punctuation insensitive). Got: \(text)"
        )
        #expect(
            normalized.contains(expected),
            comment,
            sourceLocation: sourceLocation
        )
    }
}

public enum AudioTestHelpersError: Error, CustomStringConvertible {
    case noCheckpointResolved([String])

    public var description: String {
        switch self {
        case .noCheckpointResolved(let candidates):
            return "No audio checkpoint resolved from candidates: "
                + candidates.joined(separator: ", ")
        }
    }
}
