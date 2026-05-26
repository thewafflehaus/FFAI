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
// FishSpeechConfig — config structs decoded from config.json.
//
// FishSpeechSubConfig: per-sub-model (text backbone / audio decoder) config.
// FishSpeechConfig: top-level model config parsed by `FishSpeechModel.load`.

import Foundation

// ─── Sub-model config ─────────────────────────────────────────────────────

/// Per-sub-model (text backbone / audio decoder) transformer configuration,
/// decoded from `text_config` / `audio_decoder_config` sub-objects.
public struct FishSpeechSubConfig: Sendable {
    let modelType: String
    let vocabSize: Int
    let nLayer: Int
    let nHead: Int
    let dim: Int
    let intermediateSize: Int
    let nLocalHeads: Int
    let headDim: Int
    let ropeBase: Float
    let normEps: Float
    let maxSeqLen: Int
    let tieWordEmbeddings: Bool
    let attentionQKVBias: Bool
    let attentionOBias: Bool
    let attentionQKNorm: Bool

    /// GQA: number of KV heads (clamped to nHead when nLocalHeads ≤ 0).
    var nKVHeads: Int { nLocalHeads > 0 ? nLocalHeads : nHead }

    init(raw: [String: Any], defaults: FishSpeechSubConfig.Defaults) {
        self.modelType = raw["model_type"] as? String ?? defaults.modelType
        self.vocabSize = raw["vocab_size"] as? Int ?? defaults.vocabSize
        self.nLayer = raw["n_layer"] as? Int ?? defaults.nLayer
        self.nHead = raw["n_head"] as? Int ?? defaults.nHead
        self.dim = raw["dim"] as? Int ?? defaults.dim
        self.intermediateSize = raw["intermediate_size"] as? Int ?? defaults.intermediateSize
        let nLH = raw["n_local_heads"] as? Int ?? defaults.nLocalHeads
        self.nLocalHeads = nLH > 0 ? nLH : defaults.nLocalHeads
        self.headDim = raw["head_dim"] as? Int ?? defaults.headDim
        let ropeBaseRaw = raw["rope_base"]
        if let d = ropeBaseRaw as? Double { self.ropeBase = Float(d) }
        else if let i = ropeBaseRaw as? Int { self.ropeBase = Float(i) }
        else { self.ropeBase = defaults.ropeBase }
        let eps = raw["norm_eps"]
        if let d = eps as? Double { self.normEps = Float(d) }
        else if let i = eps as? Int { self.normEps = Float(i) }
        else { self.normEps = defaults.normEps }
        self.maxSeqLen = raw["max_seq_len"] as? Int ?? defaults.maxSeqLen
        self.tieWordEmbeddings = raw["tie_word_embeddings"] as? Bool ?? defaults.tieWordEmbeddings
        self.attentionQKVBias = raw["attention_qkv_bias"] as? Bool ?? defaults.attentionQKVBias
        self.attentionOBias = raw["attention_o_bias"] as? Bool ?? defaults.attentionOBias
        self.attentionQKNorm = raw["attention_qk_norm"] as? Bool ?? defaults.attentionQKNorm
    }

    /// Default values per sub-model. Matches the fish-audio-s2-pro-8bit checkpoint.
    struct Defaults {
        var modelType: String
        var vocabSize: Int
        var nLayer: Int
        var nHead: Int
        var dim: Int
        var intermediateSize: Int
        var nLocalHeads: Int
        var headDim: Int
        var ropeBase: Float
        var normEps: Float
        var maxSeqLen: Int
        var tieWordEmbeddings: Bool
        var attentionQKVBias: Bool
        var attentionOBias: Bool
        var attentionQKNorm: Bool

        static let textBackbone = Defaults(
            modelType: "fish_qwen3",
            vocabSize: 155_776,
            nLayer: 36, nHead: 32, dim: 2560, intermediateSize: 9728,
            nLocalHeads: 8, headDim: 128, ropeBase: 1_000_000, normEps: 1e-6,
            maxSeqLen: 32_768, tieWordEmbeddings: true,
            attentionQKVBias: false, attentionOBias: false, attentionQKNorm: true
        )
        static let audioDecoder = Defaults(
            modelType: "fish_qwen3_audio_decoder",
            vocabSize: 4_096,
            nLayer: 4, nHead: 32, dim: 2560, intermediateSize: 9728,
            nLocalHeads: 8, headDim: 128, ropeBase: 1_000_000, normEps: 1e-6,
            maxSeqLen: 11, tieWordEmbeddings: false,
            attentionQKVBias: false, attentionOBias: false, attentionQKNorm: false
        )
    }
}

// ─── Top-level config ──────────────────────────────────────────────────────

/// Top-level config decoded from the checkpoint's `config.json`.
public struct FishSpeechConfig: Sendable {
    let modelType: String
    let padTokenID: Int
    let eosTokenID: Int
    let audioPadTokenID: Int
    let semanticStartTokenID: Int
    let semanticEndTokenID: Int
    let sampleRate: Int
    let textConfig: FishSpeechSubConfig
    let audioDecoderConfig: FishSpeechSubConfig
    let quantization: ModelConfig.QuantizationConfig?
    let numCodebooks: Int

    static func load(from config: ModelConfig) throws -> FishSpeechConfig {
        let raw = config.raw
        let textRaw = (raw["text_config"] as? [String: Any]) ?? [:]
        let audioRaw = (raw["audio_decoder_config"] as? [String: Any]) ?? [:]
        // num_codebooks is nested inside audio_decoder_config.
        let numCB = audioRaw["num_codebooks"] as? Int ?? 10
        return FishSpeechConfig(
            modelType: config.modelType ?? "fish_qwen3_omni",
            padTokenID: config.int("pad_token_id") ?? 151_669,
            eosTokenID: config.int("eos_token_id") ?? 151_645,
            audioPadTokenID: config.int("audio_pad_token_id") ?? 151_677,
            semanticStartTokenID: config.int("semantic_start_token_id") ?? 151_678,
            semanticEndTokenID: config.int("semantic_end_token_id") ?? 155_773,
            sampleRate: config.int("sample_rate") ?? 44_100,
            textConfig: FishSpeechSubConfig(
                raw: textRaw, defaults: .textBackbone
            ),
            audioDecoderConfig: FishSpeechSubConfig(
                raw: audioRaw, defaults: .audioDecoder
            ),
            quantization: config.quantization,
            numCodebooks: numCB
        )
    }
}
