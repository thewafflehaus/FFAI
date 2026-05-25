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
    ///
    /// `strict: false` lets swift-transformers fall back to the generic
    /// BPE implementation for tokenizer classes it hasn't registered
    /// explicitly (e.g. GPT-NeoX, which Mamba 2 ships with). The
    /// fallback warns once on stderr and uses byte-level BPE, which is
    /// the same encoding GPT-NeoX uses — no functional difference for
    /// inference.
    public func load(from directory: URL) async throws -> any Tokenizer {
        do {
            return try await AutoTokenizer.from(modelFolder: directory, strict: false)
        } catch {
            throw TokenizerLoaderError.loadFailed(directory, error)
        }
    }
}
