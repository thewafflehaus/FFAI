// Shared helpers for the audio integration suites (STT / TTS / Omni).
//
// Two responsibilities:
//
//  1. `cachedSnapshot(...)` — resolve a checkpoint directory. Some audio
//     checkpoints are only present in the HF-cache *mlx-audio* sibling
//     layout (`~/.cache/huggingface/hub/mlx-audio/<repo>/`), a flat
//     snapshot directory rather than the `models--org--repo/snapshots/`
//     blob layout `ModelLocator` downloads into. When a family's HF-hub
//     cache entry is incomplete (index.json present, shards missing) the
//     suites point `ModelLocator` at the complete mlx-audio directory as
//     a local path. The helper returns the first candidate that exists
//     and looks complete (a config + at least one weight file).
//
//  2. `clean001Waveform()` — load the bundled 16 kHz speech fixture
//     (`Resources/clean_001.wav`, "Sure, I can help you with that.")
//     for the STT suites. Referenced via a `#filePath`-relative path so
//     it works without a SwiftPM resource bundle.
//
// Unlike the old catch-and-skip helpers, nothing here swallows a load
// failure: a missing checkpoint surfaces as a thrown error so the test
// FAILS rather than silently passing.

import Foundation
@testable import FFAI

enum AudioFixtures {

    /// Root of the HF cache (`~/.cache/huggingface/hub`).
    static var hfCacheRoot: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/huggingface/hub")
    }

    /// A flat snapshot directory under the `mlx-audio` sibling cache,
    /// e.g. `mlx-audio/mlx-community_Kokoro-82M-bf16`. `repoSlug` is the
    /// repo id with `/` replaced by `_`.
    static func mlxAudioDir(_ repoSlug: String) -> URL {
        hfCacheRoot.appendingPathComponent("mlx-audio/\(repoSlug)")
    }

    /// True when `dir` looks like a usable checkpoint snapshot: a
    /// `config.json` plus at least one weight file (`*.safetensors`).
    static func isCompleteSnapshot(_ dir: URL) -> Bool {
        let fm = FileManager.default
        let config = dir.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: config.path) else { return false }
        guard let entries = try? fm.contentsOfDirectory(atPath: dir.path)
        else { return false }
        return entries.contains { $0.hasSuffix(".safetensors") }
    }

    /// Resolve a checkpoint directory, preferring `mlx-audio` slugs that
    /// are already complete on disk, then falling back to HF repo ids
    /// (which `ModelLocator` downloads / cache-hits). Every candidate is
    /// tried in order; the first that resolves to a complete snapshot
    /// wins. Throws if none resolve — the caller lets that fail the test.
    ///
    /// `mlxAudioSlugs` are checked first as local paths (no network);
    /// `repoIds` are HF ids tried after.
    static func resolveCheckpoint(
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
        throw lastError ?? AudioFixtureError.noCheckpointResolved(
            mlxAudioSlugs + repoIds)
    }

    /// Load the bundled 16 kHz speech fixture as a mono float waveform.
    /// "Sure, I can help you with that." — clean synthetic speech, 1.85 s.
    static func clean001Waveform() throws -> [Float] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Resources/clean_001.wav")
        return try AudioPreprocessing.loadWaveform(url: url, targetRate: 16_000)
    }
}

enum AudioFixtureError: Error, CustomStringConvertible {
    case noCheckpointResolved([String])

    var description: String {
        switch self {
        case .noCheckpointResolved(let candidates):
            return "No audio checkpoint resolved from candidates: "
                + candidates.joined(separator: ", ")
        }
    }
}
