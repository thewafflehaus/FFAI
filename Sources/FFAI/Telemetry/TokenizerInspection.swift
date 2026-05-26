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
// TokenizerInspection — parse a model directory's `tokenizer_config.json`
// + `config.json` and categorize every special token by role:
// BOS, EOS / end-of-turn, chat-turn markers, reasoning / thinking,
// tool calling, multimodal placeholders, utility (pad / unk / mask).
//
// Designed for one purpose: feed `ffai inspect` so chat-template /
// tool-calling / multi-turn debugging starts with a single command
// that surfaces every model-specific control token. The role
// taxonomy mirrors the categories that production chat templates
// actually depend on (Llama 3 `<|eot_id|>`, Qwen `<|im_end|>`,
// Gemma `<end_of_turn>`, Qwen `<think>`/`<tool_call>`, etc.).
//
// Heuristic-driven on purpose — tokenizer formats vary across model
// families and HF doesn't standardize the metadata. We err on the
// side of categorising too eagerly (anything that smells like a
// chat-turn marker gets bucketed there) rather than miss control
// tokens.

import Foundation
import Tokenizers

public enum SpecialTokenCategory: String, Sendable, CaseIterable {
    case bos = "BOS"
    case eos = "EOS / end-of-turn"
    case chatTurn = "Chat turn"
    case reasoning = "Reasoning"
    case toolCall = "Tool calling"
    case multimodal = "Multimodal"
    case utility = "Utility"
    case other = "Other"
}

public struct SpecialTokenInfo: Sendable, Equatable {
    public let id: Int
    public let content: String
    public let category: SpecialTokenCategory
}

public struct TokenizerInspection: Sendable {
    /// BOS as declared by `bos_token_id` in `config.json` (falls
    /// back to `tokenizer_config.json` if absent). `nil` when the
    /// model doesn't declare one (e.g. some BERT-derived bases).
    public let bosTokenId: Int?
    /// The `eos_token_id` field — may be a single id or an array
    /// (Llama 3 declares `[<|eot_id|>, <|end_of_text|>]`). Stored
    /// in declared order.
    public let eosTokenIds: [Int]
    public let padTokenId: Int?
    /// Every special token from `added_tokens_decoder`, bucketed
    /// into the categories above. Sorted by id within each bucket.
    public let specialTokens: [SpecialTokenInfo]
    /// True when `tokenizer_config.json` has a `chat_template`
    /// field. False when the user must compose chat input manually.
    public let hasChatTemplate: Bool
    /// Short summary of which template markers the chat_template
    /// string mentions (helps debug template / token-id mismatches:
    /// if the template references `<|im_end|>` but the tokenizer
    /// doesn't have it as a special token, generation will produce
    /// raw `<|im_end|>` text instead of an end-of-turn signal).
    public let chatTemplateMarkers: [String]

    public init(
        bosTokenId: Int?,
        eosTokenIds: [Int],
        padTokenId: Int?,
        specialTokens: [SpecialTokenInfo],
        hasChatTemplate: Bool,
        chatTemplateMarkers: [String]
    ) {
        self.bosTokenId = bosTokenId
        self.eosTokenIds = eosTokenIds
        self.padTokenId = padTokenId
        self.specialTokens = specialTokens
        self.hasChatTemplate = hasChatTemplate
        self.chatTemplateMarkers = chatTemplateMarkers
    }

    /// Tokens in a single category, sorted by id.
    public func tokens(in category: SpecialTokenCategory) -> [SpecialTokenInfo] {
        specialTokens.filter { $0.category == category }
    }
}

/// Parse `tokenizer_config.json` + `config.json` from a loaded
/// model's directory. Returns a best-effort categorized view —
/// always succeeds (even when files are missing, returns empty
/// `specialTokens`).
public func inspectTokenizer(
    modelDirectory: URL,
    config: ModelConfig
) -> TokenizerInspection {
    let tokenizerCfg = loadJSON(modelDirectory.appendingPathComponent("tokenizer_config.json"))

    let bosId = config.int("bos_token_id")
    let padId = config.int("pad_token_id")
    let eosIds: [Int]
    if let arr = config.intArray("eos_token_id") {
        eosIds = arr
    } else if let one = config.int("eos_token_id") {
        eosIds = [one]
    } else {
        eosIds = []
    }

    // Decode `added_tokens_decoder` → `{id: {content, special, …}}`.
    // Some HF tokenizer dumps key on string ids ("128000"), others
    // on ints. We accept both via JSON traversal.
    var tokens: [SpecialTokenInfo] = []
    if let atd = tokenizerCfg?["added_tokens_decoder"] as? [String: Any] {
        for (key, value) in atd {
            guard let id = Int(key),
                  let entry = value as? [String: Any],
                  let content = entry["content"] as? String
            else { continue }
            let isSpecial = (entry["special"] as? Bool) ?? false
            // Even non-special tokens get included if they smell
            // like a control marker (some checkpoints use
            // special=false for chat-turn tokens — Gemma 3's
            // `<start_of_image>` for example).
            let category = categorize(content: content, id: id, eosIds: eosIds, bosId: bosId, padId: padId)
            if isSpecial || category != .other {
                tokens.append(SpecialTokenInfo(id: id, content: content, category: category))
            }
        }
    }
    tokens.sort { $0.id < $1.id }

    let chatTemplate = tokenizerCfg?["chat_template"] as? String
    let hasChatTemplate = (chatTemplate != nil)
    let markers = chatTemplate.map(extractTemplateMarkers) ?? []

    return TokenizerInspection(
        bosTokenId: bosId,
        eosTokenIds: eosIds,
        padTokenId: padId,
        specialTokens: tokens,
        hasChatTemplate: hasChatTemplate,
        chatTemplateMarkers: markers
    )
}

// MARK: - Internal helpers

private func loadJSON(_ url: URL) -> [String: Any]? {
    guard let data = try? Data(contentsOf: url),
          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    return obj
}

/// Bucket a special-token content string into its category. The
/// matching prefers the most-specific bucket over the most generic
/// (a token containing both "tool" and "end" goes to `toolCall`,
/// not `eos`).
private func categorize(
    content: String, id: Int,
    eosIds: [Int], bosId: Int?, padId: Int?
) -> SpecialTokenCategory {
    let lc = content.lowercased()

    // Tool calling — most specific, evaluated first so a token
    // like `[/TOOL_CALLS]` doesn't get mis-bucketed as EOS.
    if lc.contains("tool") || lc.contains("function_call")
       || lc.contains("python_tag") || lc.contains("/call")
       || lc == "<function>" || lc == "</function>" {
        return .toolCall
    }

    // Reasoning / thinking — Qwen 3 `<think>`, DeepSeek
    // `<scratchpad>`, Claude-style `<thinking>`.
    if lc == "<think>" || lc == "</think>"
       || lc.contains("scratchpad") || lc.contains("reasoning")
       || lc == "<thinking>" || lc == "</thinking>" {
        return .reasoning
    }

    // Multimodal placeholders.
    if lc.contains("image") || lc.contains("vision") || lc.contains("video")
       || lc.contains("audio") || lc.contains("box") || lc.contains("quad")
       || lc.contains("object_ref") {
        return .multimodal
    }

    // BOS — explicit id match wins over name match.
    if let b = bosId, id == b { return .bos }
    if lc.contains("begin_of_text") || lc == "<bos>" || lc == "<s>"
       || lc.contains("start_of_text") {
        return .bos
    }

    // EOS / end-of-turn — id matches first.
    if eosIds.contains(id) { return .eos }
    if lc.contains("end_of_text") || lc.contains("eot")
       || lc.contains("eom") || lc.contains("im_end")
       || lc.contains("end_of_turn") || lc == "</s>"
       || lc.contains("endoftext") || lc == "<eos>" {
        return .eos
    }

    // Chat-turn markers (user / assistant / system role + start/header).
    if lc.contains("im_start") || lc.contains("start_header")
       || lc.contains("end_header") || lc.contains("start_of_turn")
       || lc == "<|user|>" || lc == "<|assistant|>" || lc == "<|system|>"
       || lc.contains("user_role") || lc.contains("assistant_role") {
        return .chatTurn
    }

    // Utility tokens.
    if let p = padId, id == p { return .utility }
    if lc.contains("pad") || lc == "<unk>" || lc == "<mask>"
       || lc.contains("right_pad") {
        return .utility
    }

    return .other
}

/// Extract a small set of well-known markers from a Jinja-style
/// chat template. Surface them in inspect so chat-template-vs-
/// tokenizer mismatches are visible without reading the raw
/// template (which is often hundreds of lines).
private func extractTemplateMarkers(_ template: String) -> [String] {
    let patterns: [String] = [
        "<|im_start|>", "<|im_end|>",
        "<|begin_of_text|>", "<|end_of_text|>",
        "<|eot_id|>", "<|eom_id|>",
        "<|start_header_id|>", "<|end_header_id|>",
        "<|python_tag|>",
        "<start_of_turn>", "<end_of_turn>",
        "<think>", "</think>",
        "<thinking>", "</thinking>",
        "<tool_call>", "</tool_call>",
        "[INST]", "[/INST]",
        "[TOOL_CALLS]", "[/TOOL_CALLS]",
        "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]",
        "<|user|>", "<|assistant|>", "<|system|>", "<|end|>",
        "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
    ]
    return patterns.filter { template.contains($0) }
}
