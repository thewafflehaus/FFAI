// TokenizerLoader
//
// Thin wrapper over swift-transformers' AutoTokenizer.from(modelFolder:).

import Foundation
import Hub
import Tokenizers

public enum TokenizerLoaderError: Error, CustomStringConvertible {
    case loadFailed(URL, Error)

    public var description: String {
        switch self {
        case .loadFailed(let url, let underlying):
            return "Failed to load tokenizer at \(url.path): \(underlying)"
        }
    }
}

public struct TokenizerLoader: Sendable {
    public init() {}

    /// Load a tokenizer from a model directory containing tokenizer.json
    /// (and tokenizer_config.json + special_tokens_map.json + chat template).
    public func load(from directory: URL) async throws -> any Tokenizer {
        do {
            return try await AutoTokenizer.from(modelFolder: directory)
        } catch {
            throw TokenizerLoaderError.loadFailed(directory, error)
        }
    }
}
