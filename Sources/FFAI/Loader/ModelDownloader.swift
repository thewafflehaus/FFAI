// ModelDownloader
//
// Thin wrapper over swift-huggingface's HubClient for downloading model
// snapshots from Hugging Face Hub. Caches into the standard
// `~/.cache/huggingface/hub/` location (same path Python's huggingface_hub
// uses, so caches are shared).

import Foundation
import HuggingFace

public enum ModelDownloaderError: Error, CustomStringConvertible {
    case invalidRepoID(String)
    case downloadFailed(String, Error)

    public var description: String {
        switch self {
        case .invalidRepoID(let id):
            return "Invalid HF repo id: \(id) (expected 'org/name')"
        case .downloadFailed(let id, let underlying):
            return "Failed to download \(id): \(underlying)"
        }
    }
}

public struct ModelDownloader: Sendable {
    public let client: HubClient

    public init(client: HubClient = .default) {
        self.client = client
    }

    /// Convenience init that builds a `HubClient` with the standard
    /// auto-detected endpoint + token but overrides the cache root.
    /// `nil` keeps the standard discovery order (HF_HOME →
    /// `~/.cache/huggingface/hub`); a non-nil URL points the cache at
    /// a specific directory (e.g. an external SSD).
    public init(cacheDirectory: URL?) {
        if let dir = cacheDirectory {
            self.client = HubClient(cache: HubCache(cacheDirectory: dir))
        } else {
            self.client = .default
        }
    }

    /// Download (or hit cache) a model snapshot. Returns the local snapshot
    /// directory containing config.json, tokenizer.json, *.safetensors, etc.
    ///
    /// - Parameters:
    ///   - id: Repo id like `"meta-llama/Llama-3.2-1B"`.
    ///   - revision: Branch, tag, or commit. Defaults to `"main"`.
    ///   - patterns: Glob patterns to filter files. Empty = all files.
    ///   - localFilesOnly: If true, fail if not in cache (no network).
    ///   - progressHandler: Called on main actor with bytes downloaded / total.
    public func download(
        id: String,
        revision: String = "main",
        matching patterns: [String] = [],
        localFilesOnly: Bool = false,
        progressHandler: (@MainActor @Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        guard let repoID = Repo.ID(rawValue: id) else {
            throw ModelDownloaderError.invalidRepoID(id)
        }
        do {
            return try await client.downloadSnapshot(
                of: repoID,
                revision: revision,
                matching: patterns,
                localFilesOnly: localFilesOnly,
                progressHandler: progressHandler
            )
        } catch {
            throw ModelDownloaderError.downloadFailed(id, error)
        }
    }
}
