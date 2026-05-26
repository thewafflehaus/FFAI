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
// ModelLocator
//
// Resolves a user-facing model identifier (HF repo id or local path) to
// a local directory containing the model files. Local paths are returned
// as-is; HF repo ids are downloaded into the HF cache.

import Foundation

public enum ModelLocatorError: Error, CustomStringConvertible {
    case localPathNotFound(URL)

    public var description: String {
        switch self {
        case .localPathNotFound(let url):
            return "Local model path does not exist: \(url.path)"
        }
    }
}

public struct ModelLocator: Sendable {
    public let downloader: ModelDownloader

    public init(downloader: ModelDownloader = ModelDownloader()) {
        self.downloader = downloader
    }

    /// Heuristic: looks like a path if it starts with `/`, `./`, `../`, or `~`.
    public static func isLocalPath(_ s: String) -> Bool {
        s.hasPrefix("/") || s.hasPrefix("./") || s.hasPrefix("../") || s.hasPrefix("~")
    }

    /// Resolve to a local directory. If `idOrPath` looks like a path, validate
    /// it exists. Otherwise download from HF.
    public func resolve(
        idOrPath: String,
        revision: String = "main",
        matching patterns: [String] = ModelLocator.defaultDownloadPatterns,
        progressHandler: (@MainActor @Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        if Self.isLocalPath(idOrPath) {
            let expanded = (idOrPath as NSString).expandingTildeInPath
            let url = URL(fileURLWithPath: expanded, isDirectory: true)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ModelLocatorError.localPathNotFound(url)
            }
            return url
        }
        return try await downloader.download(
            id: idOrPath,
            revision: revision,
            matching: patterns,
            progressHandler: progressHandler
        )
    }

    /// Default patterns: weights + tokenizer + config files. Skips
    /// preview videos, .pth artifacts, original Llama folders, etc.
    public static let defaultDownloadPatterns: [String] = [
        "*.safetensors",
        "*.json",
        "*.jinja",
        "tokenizer.model",
        "*.txt",
    ]
}
