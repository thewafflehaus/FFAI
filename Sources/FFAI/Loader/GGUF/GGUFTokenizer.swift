// Copyright 2026 Tom Turney (@TheTom)
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
// GGUF tokenizer adapter — reconstruct a swift-transformers
// `PreTrainedTokenizer` from the `tokenizer.ggml.*` metadata block.
//
// GGUF embeds the tokenizer alongside the model weights via a
// well-defined metadata schema (canonical fields documented at
// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). The
// shape mirrors the HF `tokenizer.json` / `tokenizer_config.json`
// pair, just under a different key namespace. The adapter
// translates between the two so the rest of FFAI (which already
// consumes `PreTrainedTokenizer`) can use a GGUF checkpoint
// transparently.
//
// Supported tokenizer kinds (`tokenizer.ggml.model`):
//   - `gpt2`, `llama`, `deepseek-llm`, `deepseek-coder` — all
//     BPE; built as `PreTrainedTokenizer` with tokenizer_class
//     `"PreTrainedTokenizerFast"` and the matching pretokenizer
//     regex.
//
// Other kinds (`bert`, `unigram`) will throw `unsupportedKind`
// until they're needed.

import Foundation
import Hub
import Tokenizers

enum GGUFTokenizerAdapter {
    enum Error: Swift.Error, CustomStringConvertible {
        case missingField(String)
        case unsupportedKind(String)
        case buildFailed(underlying: Swift.Error)

        var description: String {
            switch self {
            case .missingField(let f):
                return "GGUFTokenizerAdapter: required metadata field missing: \(f)"
            case .unsupportedKind(let k):
                return
                    "GGUFTokenizerAdapter: tokenizer.ggml.model='\(k)' not supported yet (only BPE-family kinds — gpt2 / llama / deepseek-llm / deepseek-coder)"
            case .buildFailed(let err):
                return "GGUFTokenizerAdapter: swift-transformers init failed: \(err)"
            }
        }
    }

    /// Build a `PreTrainedTokenizer` from a GGUF reader. The reader
    /// must have a populated `tokenizer.ggml.*` metadata block (every
    /// official GGUF checkpoint does).
    static func build(reader: GGUFReader) throws -> any Tokenizer {
        let kind = reader.metadataString("tokenizer.ggml.model") ?? ""
        guard isBPEKind(kind) else {
            throw Error.unsupportedKind(kind)
        }

        guard let tokens = reader.metadataStringArray("tokenizer.ggml.tokens") else {
            throw Error.missingField("tokenizer.ggml.tokens")
        }
        guard let merges = reader.metadataStringArray("tokenizer.ggml.merges") else {
            throw Error.missingField("tokenizer.ggml.merges")
        }

        // Build the vocab dict (token → id). GGUF stores tokens as a
        // positionally-indexed array; ID = array index.
        var vocab: [NSString: Any] = [:]
        vocab.reserveCapacity(tokens.count)
        for (i, t) in tokens.enumerated() {
            vocab[t as NSString] = i
        }

        // BOS / EOS / UNK / PAD lookups. The IDs are stored as u32 in
        // GGUF; the token strings come from `tokens[id]`.
        let bosTokenStr = lookupToken(reader: reader, key: "tokenizer.ggml.bos_token_id", tokens: tokens)
        let eosTokenStr = lookupToken(reader: reader, key: "tokenizer.ggml.eos_token_id", tokens: tokens)
        let unkTokenStr = lookupToken(reader: reader, key: "tokenizer.ggml.unknown_token_id", tokens: tokens)
        let padTokenStr = lookupToken(reader: reader, key: "tokenizer.ggml.padding_token_id", tokens: tokens)

        // Pre-tokenizer hint (added in late 2024 — `tokenizer.ggml.pre`):
        // GGUF carries llama.cpp-side regex-group names (`"joyai-llm"`,
        // `"deepseek-llm"`, `"qwen2"`, …) which don't map 1:1 to
        // swift-transformers' enum — so we normalise to the closest
        // swift-transformers-recognised type. `gpt2`-family models
        // collapse to ByteLevel, `llama`-family to Metaspace; unknown
        // model kinds fall back to ByteLevel (the GPT-2 BPE default).
        let preHint = normalisedPreType(
            forKind: kind, hint: reader.metadataString("tokenizer.ggml.pre"))
        let chatTemplate = reader.metadataString("tokenizer.chat_template")

        // ── tokenizerData: mirrors HF tokenizer.json structure ──
        let modelDict: [NSString: Any] = [
            "type": "BPE",
            "vocab": vocab,
            // GGUF stores merges as space-separated strings ("a b").
            // swift-transformers' `mergesFromConfig` accepts that
            // legacy shape directly — no conversion needed.
            "merges": merges,
            // `byte_fallback` is the default for SentencePiece-style
            // models (llama family); BPE-pure tokenizers (gpt2,
            // deepseek-coder) set this false. Conservative default
            // off; explicit GGUF metadata wins when present.
            "byte_fallback": reader.metadataBool("tokenizer.ggml.add_bos_token") ?? false,
        ]
        var tokenizerDataDict: [NSString: Any] = [
            "model": modelDict,
            "pre_tokenizer": ["type": preHint] as [NSString: Any],
        ]
        // `added_tokens` is the override layer for special tokens
        // that already appear in the vocab. We don't need to inject
        // anything here — BOS/EOS already live at their ID positions
        // in `tokens` — but the field's shape needs to exist so
        // PreTrainedTokenizer doesn't choke on a missing key.
        tokenizerDataDict["added_tokens"] = [Any]()

        // ── tokenizerConfig ──
        var tokenizerConfigDict: [NSString: Any] = [
            "tokenizer_class": "PreTrainedTokenizerFast"
        ]
        if let bos = bosTokenStr { tokenizerConfigDict["bos_token"] = bos }
        if let eos = eosTokenStr { tokenizerConfigDict["eos_token"] = eos }
        if let unk = unkTokenStr { tokenizerConfigDict["unk_token"] = unk }
        if let pad = padTokenStr { tokenizerConfigDict["pad_token"] = pad }
        if let tpl = chatTemplate { tokenizerConfigDict["chat_template"] = tpl }
        if let addBos = reader.metadataBool("tokenizer.ggml.add_bos_token") {
            tokenizerConfigDict["add_bos_token"] = addBos
        }
        if let addEos = reader.metadataBool("tokenizer.ggml.add_eos_token") {
            tokenizerConfigDict["add_eos_token"] = addEos
        }

        let tokenizerConfig = Config(tokenizerConfigDict)
        let tokenizerData = Config(tokenizerDataDict)

        do {
            return try PreTrainedTokenizer(
                tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: false)
        } catch {
            throw Error.buildFailed(underlying: error)
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────

    /// Map a `tokenizer.ggml.*_token_id` u32 to its string from the
    /// vocab. Returns `nil` when the id field is absent or out of
    /// range.
    private static func lookupToken(reader: GGUFReader, key: String, tokens: [String]) -> String? {
        guard let id = reader.metadataUInt32(key), Int(id) < tokens.count else { return nil }
        return tokens[Int(id)]
    }

    /// BPE-family tokenizer kinds. The GGUF `tokenizer.ggml.model` enum
    /// covers a wider set (SentencePiece-Unigram, BERT-WordPiece, …);
    /// this is the subset we know swift-transformers' `BPETokenizer`
    /// handles correctly. New kinds get added here once their
    /// pretokenizer regex is wired in.
    private static func isBPEKind(_ kind: String) -> Bool {
        switch kind {
        case "gpt2", "llama", "deepseek-llm", "deepseek-coder",
            "qwen2", "chatglm-bpe", "mpt", "starcoder", "falcon", "refact",
            "command-r", "olmo", "phi-3", "smaug-bpe":
            return true
        default:
            return false
        }
    }

    /// PreTokenizer type whitelist that swift-transformers' factory
    /// accepts without fatalError'ing. Anything else gets remapped
    /// to a safe default keyed by `model` (gpt2 → ByteLevel, llama
    /// → Metaspace).
    private static let knownPreTypes: Set<String> = [
        "Sequence", "ByteLevel", "Punctuation", "Digits", "Split",
        "Whitespace", "WhitespaceSplit", "Metaspace", "BertPreTokenizer",
    ]

    /// Normalise the `tokenizer.ggml.pre` hint to one of the
    /// swift-transformers-recognised PreTokenizer kinds. The hint
    /// string in GGUF carries llama.cpp's internal regex-family name
    /// (`"joyai-llm"`, `"deepseek-llm"`, `"qwen2"`, …); for the BPE
    /// kinds we support, all of those collapse into one of two
    /// behaviours at the tokenization layer.
    private static func normalisedPreType(forKind kind: String, hint: String?) -> String {
        if let hint, knownPreTypes.contains(hint) {
            return hint
        }
        switch kind {
        case "gpt2", "qwen2", "deepseek-coder", "starcoder", "falcon", "refact",
            "command-r", "olmo", "phi-3", "smaug-bpe", "mpt", "chatglm-bpe":
            return "ByteLevel"
        case "llama", "deepseek-llm":
            return "Metaspace"
        default:
            return "ByteLevel"
        }
    }
}
