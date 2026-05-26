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
// ThinkingSplit — partition a generated token stream into "thinking"
// vs "user-visible answer" segments so per-phase stats can be reported.
//
// Format detection is per-family. Different reasoning models emit
// different boundary marker conventions:
//
//   • ChatML        — `<think>` … `</think>` (Qwen 3 / DeepSeek-R1).
//   • Harmony       — GPT-OSS channel markers
//                     (`<|start|>assistant<|channel|>analysis<|message|>` …
//                      `<|end|><|start|>assistant<|channel|>final<|message|>` …)
//   • Gemma channel — Gemma 3/4 reasoning mode
//                     (`<channel|reasoning|>` … `<channel|final|>`).
//   • None          — model doesn't emit a thinking segment; whole
//                     stream counts as gen.
//
// Auto-detected from `model.config.modelType` + tokenizer vocab
// presence. Per-format implementations share the `Split` shape but
// scan with their own marker rules.
//
// Today only the ChatML scanner is implemented end-to-end. Harmony
// and Gemma scanners ship alongside the GPT-OSS / Gemma family files
// (planned / planned). The data model is in place so the bench
// writer + GenerationStats schema don't churn when those land.

import Foundation
import Tokenizers

public enum ThinkingFormat: String, Sendable, Equatable, CaseIterable {
    case none
    case chatML
    case harmony
    case gemmaChannel
}

public enum ThinkingSplit {
    public struct Split: Sendable, Equatable {
        /// Tokens between the open + close markers, exclusive of the
        /// markers themselves.
        public let thinkTokens: ArraySlice<Int>
        /// Tokens after the close marker. Empty if the model never
        /// closed the thinking block.
        public let genTokens: ArraySlice<Int>
        /// Format that produced this split.
        public let format: ThinkingFormat
    }

    // MARK: - Format detection

    /// Pick a format for the given model. Inspects `config.modelType`
    /// first (so e.g. GPT-OSS goes straight to Harmony) and falls back
    /// to checking the tokenizer's vocab for ChatML markers.
    public static func detectFormat(model: Model) -> ThinkingFormat {
        let mt = (model.config.modelType ?? "").lowercased()
        if mt.contains("gpt-oss") || mt.contains("gpt_oss") { return .harmony }
        if mt.contains("gemma3") || mt.contains("gemma4")  { return .gemmaChannel }
        if chatMLMarkers(tokenizer: model.tokenizer) != nil { return .chatML }
        return .none
    }

    // MARK: - Public split entry points

    /// Split a generated token sequence using the model's auto-detected
    /// format. Returns `nil` when the format is `.none` or no split
    /// boundary was found.
    public static func split(tokens: [Int], model: Model) -> Split? {
        split(tokens: tokens, format: detectFormat(model: model),
              tokenizer: model.tokenizer)
    }

    /// Split using an explicit format — bypasses auto-detection.
    public static func split(tokens: [Int],
                             format: ThinkingFormat,
                             tokenizer: any Tokenizer) -> Split? {
        switch format {
        case .none:         return nil
        case .chatML:       return splitChatML(tokens: tokens, tokenizer: tokenizer)
        case .harmony:      return splitHarmony(tokens: tokens, tokenizer: tokenizer)
        case .gemmaChannel: return splitGemmaChannel(tokens: tokens, tokenizer: tokenizer)
        }
    }

    // MARK: - ChatML scanner (Qwen 3 / DeepSeek-R1)

    /// Look up the `<think>` / `</think>` token ids in the tokenizer's
    /// vocab. Returns `nil` when the tokenizer doesn't carry them.
    public static func chatMLMarkers(tokenizer: any Tokenizer) -> (open: Int, close: Int)? {
        let openCandidates  = ["<think>", "<|think|>", "<thinking>"]
        let closeCandidates = ["</think>", "<|/think|>", "</thinking>"]
        var open: Int?
        var close: Int?
        for s in openCandidates {
            let ids = tokenizer.encode(text: s)
            if ids.count == 1 { open = ids[0]; break }
        }
        for s in closeCandidates {
            let ids = tokenizer.encode(text: s)
            if ids.count == 1 { close = ids[0]; break }
        }
        guard let o = open, let c = close else { return nil }
        return (o, c)
    }

    private static func splitChatML(tokens: [Int],
                                    tokenizer: any Tokenizer) -> Split? {
        guard let (open, close) = chatMLMarkers(tokenizer: tokenizer) else { return nil }
        return splitChatML(tokens: tokens, openMarker: open, closeMarker: close)
    }

    /// Tokenizer-free ChatML scanner — partition `tokens` on the
    /// supplied open/close marker ids. Useful for tests that don't
    /// want to mock a full `Tokenizer` conformance.
    public static func splitChatML(tokens: [Int],
                                   openMarker: Int,
                                   closeMarker: Int) -> Split? {
        guard let openIdx = tokens.firstIndex(of: openMarker) else { return nil }
        guard let closeIdx = tokens[openIdx...].firstIndex(of: closeMarker) else { return nil }
        return Split(thinkTokens: tokens[(openIdx + 1)..<closeIdx],
                     genTokens: tokens[(closeIdx + 1)...],
                     format: .chatML)
    }

    // MARK: - Harmony scanner (GPT-OSS) — TODO planned

    private static func splitHarmony(tokens: [Int],
                                     tokenizer: any Tokenizer) -> Split? {
        // Harmony emits multi-token channel sequences:
        //   `<|start|>assistant<|channel|>analysis<|message|>` … (think)
        //   `<|end|>` … `<|start|>assistant<|channel|>final<|message|>` … (gen)
        // The full marker is several tokens; need to scan for the
        // [`<|channel|>`, `analysis`|`final`, `<|message|>`] subsequence
        // and partition on it.
        //
        // Stub for now — lands with the GPT-OSS family file. See
        // mlx-swift-lm's HarmonyChannelDecoder for the full shape.
        _ = (tokens, tokenizer)
        return nil
    }

    // MARK: - Gemma channel scanner — TODO planned

    private static func splitGemmaChannel(tokens: [Int],
                                          tokenizer: any Tokenizer) -> Split? {
        // Gemma 3/4 reasoning mode uses `<channel|reasoning|>` …
        // `<channel|final|>` markers. Scanner lands with the Gemma
        // family file; right now FFAI doesn't ship a Gemma family.
        _ = (tokens, tokenizer)
        return nil
    }
}
