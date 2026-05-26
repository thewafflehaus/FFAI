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
// ModelConfig
//
// Decodes a model's `config.json` into a typed struct that family files
// can pattern-match against. Only the fields we actually use are typed;
// unknown fields land in `extra` for variant-specific decoding.

import Foundation

public struct ModelConfig: @unchecked Sendable {
    /// `architectures[0]` from config.json (e.g. "LlamaForCausalLM").
    public let architecture: String?
    /// `model_type` from config.json (e.g. "llama").
    public let modelType: String?
    /// Raw JSON object for variant-specific decoding.
    public let raw: [String: Any]

    /// Decode `config.json` from a model directory.
    ///
    /// Some HF configs (notably Mamba 2's `time_step_limit: [0.0,
    /// Infinity]`) ship non-standard JSON literals that Foundation's
    /// `JSONSerialization` rejects unless `.json5Allowed` is set. We
    /// enable JSON5 across the board — it's a superset of JSON, so
    /// strict configs still parse.
    public static func load(from directory: URL) throws -> ModelConfig {
        let url = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        let parsed: Any
        if #available(macOS 12.0, iOS 15.0, *) {
            parsed = try JSONSerialization.jsonObject(with: data, options: [.json5Allowed])
        } else {
            parsed = try JSONSerialization.jsonObject(with: data)
        }
        guard let obj = parsed as? [String: Any] else {
            throw ModelConfigError.malformed(url)
        }
        let architecture = (obj["architectures"] as? [String])?.first
        let modelType = obj["model_type"] as? String
        return ModelConfig(architecture: architecture, modelType: modelType, raw: obj)
    }

    // ─── Typed accessors ──────────────────────────────────────────────

    public func int(_ key: String) -> Int? {
        if let v = raw[key] as? Int { return v }
        if let v = raw[key] as? Double { return Int(v) }
        return nil
    }

    public func float(_ key: String) -> Double? {
        if let v = raw[key] as? Double { return v }
        if let v = raw[key] as? Int { return Double(v) }
        return nil
    }

    public func string(_ key: String) -> String? { raw[key] as? String }

    public func bool(_ key: String) -> Bool? { raw[key] as? Bool }

    public func has(_ key: String) -> Bool { raw[key] != nil }

    public func intArray(_ key: String) -> [Int]? { raw[key] as? [Int] }

    public func nested(_ key: String) -> [String: Any]? {
        raw[key] as? [String: Any]
    }

    /// A `ModelConfig` view onto a nested sub-dictionary — e.g. a VLM's
    /// `text_config` or `vision_config`. The sub-view keeps the parent's
    /// `architecture` (the sub-dict has its own `model_type`), so the
    /// text backbone can load from a VL checkpoint's `text_config`
    /// exactly as it loads a stand-alone text config.
    public func subConfig(_ key: String) -> ModelConfig? {
        guard let sub = nested(key) else { return nil }
        return ModelConfig(
            architecture: architecture,
            modelType: sub["model_type"] as? String ?? modelType,
            raw: sub)
    }

    /// `vocab_size`
    public var vocabSize: Int? { int("vocab_size") }
    /// `hidden_size`
    public var hiddenSize: Int? { int("hidden_size") }
    /// `intermediate_size`
    public var intermediateSize: Int? { int("intermediate_size") }
    /// `num_hidden_layers`
    public var numLayers: Int? { int("num_hidden_layers") }
    /// `num_attention_heads`
    public var numAttentionHeads: Int? { int("num_attention_heads") }
    /// `num_key_value_heads` — defaults to `num_attention_heads` if absent.
    public var numKeyValueHeads: Int? {
        int("num_key_value_heads") ?? numAttentionHeads
    }
    /// `head_dim` — explicit, or derived from hidden_size / num_attention_heads.
    public var headDim: Int? {
        if let h = int("head_dim") { return h }
        if let hs = hiddenSize, let nh = numAttentionHeads, nh > 0 {
            return hs / nh
        }
        return nil
    }
    /// `rms_norm_eps`
    public var rmsNormEps: Double? { float("rms_norm_eps") }
    /// `rope_theta` — base frequency for RoPE
    public var ropeTheta: Double? { float("rope_theta") }
    /// `tie_word_embeddings` — if true, lm_head shares weights with embed_tokens.
    public var tieWordEmbeddings: Bool { bool("tie_word_embeddings") ?? false }
    /// `eos_token_id` — single id or the first in a list.
    public var eosTokenId: Int? {
        if let v = int("eos_token_id") { return v }
        if let v = (raw["eos_token_id"] as? [Int])?.first { return v }
        return nil
    }
    /// All `eos_token_id` entries — Gemma 3+ and several Qwen variants
    /// publish a list of EOS-equivalent ids (model-EOS plus end-of-turn,
    /// `<|im_end|>`, etc.). Returns every id when the field is a list;
    /// returns the single id wrapped when it's a scalar; empty array
    /// when absent. Generation should stop on any of these.
    public var eosTokenIds: [Int] {
        if let arr = raw["eos_token_id"] as? [Int] { return arr }
        if let v = int("eos_token_id") { return [v] }
        return []
    }
    /// `bos_token_id`
    public var bosTokenId: Int? { int("bos_token_id") }

    /// MLX quantization block: `{ "group_size": Int, "bits": Int }`.
    /// Returns nil for unquantized checkpoints.
    public struct QuantizationConfig: Sendable {
        public let bits: Int
        public let groupSize: Int
    }

    public var quantization: QuantizationConfig? {
        guard let q = nested("quantization"),
            let bits = q["bits"] as? Int,
            let group = q["group_size"] as? Int
        else { return nil }
        return QuantizationConfig(bits: bits, groupSize: group)
    }
}

public enum ModelConfigError: Error, CustomStringConvertible {
    case malformed(URL)

    public var description: String {
        switch self {
        case .malformed(let url):
            return "Malformed config.json at \(url.path)"
        }
    }
}
