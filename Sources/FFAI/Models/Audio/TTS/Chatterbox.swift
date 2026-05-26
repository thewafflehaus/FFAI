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
// Chatterbox — ResembleAI's two-stage text-to-speech family.
//
// Chatterbox is a two-stage TTS pipeline:
//
//   Stage 1 — T3 (Token-To-Token):
//     text tokens ──backbone (LLaMA 520M or GPT-2 Medium)──▶ speech tokens
//     conditioned on speaker embedding + optional prompt + emotion scalar
//
//   Stage 2 — S3Gen (Conformer + flow matching + HiFi-GAN):
//     speech tokens ──flow matching decoder──▶ mel spectrogram
//     mel spectrogram ──HiFi-GAN vocoder──▶ waveform at 24 kHz
//
// Two variants:
//   • Regular (`Chatterbox-TTS-fp16`): LLaMA 520M backbone, 23 languages,
//     emotion control (`model_type = "chatterbox"`).
//   • Turbo (`chatterbox-turbo-fp16`): GPT-2 Medium backbone, English only,
//     faster distilled flow matching (`model_type = "chatterbox_turbo"`).
//
// ## Scope note — STAGED PORT
//
// Chatterbox lands in stages. This file is **stage 1**: config decoding,
// `AudioModelRegistry` detection, checkpoint weight-bundle retention, and
// the top-level `ChatterboxModel` scaffold. Until the T3 backbone, S3Gen
// flow decoder, VoiceEncoder, and HiFi-GAN vocoder stages land,
// `synthesize` throws `ChatterboxError.synthesisNotWired` — see the error
// description for what is missing.
//
// Reference implementation:
//   ~/Development/personal/ai/mlx-audio-swift/Sources/MLXAudioTTS/Models/Chatterbox/
// Ported from Python: https://github.com/resemble-ai/chatterbox

import Foundation

// ─── Errors ──────────────────────────────────────────────────────────

public enum ChatterboxError: Error, CustomStringConvertible {
    /// Full synthesis pipeline not wired yet. Config + detection land
    /// first; T3 backbone + S3Gen flow decoder + HiFi-GAN are follow-on
    /// stages. Use the staged interface (`ChatterboxModel.weights`) to
    /// inspect loaded weights in the interim.
    case synthesisNotWired
    /// Required config field missing.
    case missingConfig(String)

    public var description: String {
        switch self {
        case .synthesisNotWired:
            return "Chatterbox: the T3 backbone, S3Gen flow decoder, "
                + "VoiceEncoder, and HiFi-GAN vocoder are not yet wired "
                + "in this build. Stage 1 ships config decoding + detection. "
                + "Follow-on stages will wire the two-stage synthesis pipeline."
        case .missingConfig(let field):
            return "Chatterbox: required config field missing: \(field)"
        }
    }
}

// ─── Configuration ───────────────────────────────────────────────────

// MARK: - GPT-2 Backbone (Turbo)

/// GPT-2 Medium configuration used by the Turbo T3 backbone.
/// Maps to Python's `GPT2_MEDIUM_CONFIG` dict.
public struct ChatterboxGPT2Config: Sendable {
    /// Feed-forward activation (`"gelu_new"` for GPT-2).
    public let activationFunction: String
    /// Maximum sequence context length.
    public let nCtx: Int
    /// Hidden (embedding) dimension.
    public let hiddenSize: Int
    /// Attention head count.
    public let nHead: Int
    /// Transformer block count.
    public let nLayer: Int
    /// Vocabulary size.
    public let vocabSize: Int
    /// Layer-norm epsilon.
    public let layerNormEpsilon: Float

    /// Per-head dimension (`hiddenSize / nHead`).
    public var headDim: Int { hiddenSize / nHead }
    /// Feed-forward intermediate size (4x hidden for GPT-2).
    public var intermediateSize: Int { hiddenSize * 4 }

    /// Default GPT-2 Medium constants matching Python's `GPT2_MEDIUM_CONFIG`.
    public static let medium = ChatterboxGPT2Config(
        activationFunction: "gelu_new",
        nCtx: 8196,
        hiddenSize: 1024,
        nHead: 16,
        nLayer: 24,
        vocabSize: 50276,
        layerNormEpsilon: 1e-05
    )

    /// Decode the `gpt2` sub-block of a Chatterbox Turbo `config.json`.
    /// Falls back to `.medium` defaults for any missing field.
    public static func from(_ raw: [String: Any]) -> ChatterboxGPT2Config {
        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }
        func f(_ k: String, _ d: Float) -> Float {
            if let v = raw[k] as? Double { return Float(v) }
            if let v = raw[k] as? Int { return Float(v) }
            return d
        }
        return ChatterboxGPT2Config(
            activationFunction: raw["activation_function"] as? String ?? "gelu_new",
            nCtx: i("n_ctx", 8196),
            // Turbo config uses both `n_embd` and `hidden_size`; prefer hidden_size.
            hiddenSize: i("hidden_size", i("n_embd", 1024)),
            nHead: i("n_head", 16),
            nLayer: i("n_layer", 24),
            vocabSize: i("vocab_size", 50276),
            layerNormEpsilon: f("layer_norm_epsilon", 1e-05)
        )
    }

    public init(activationFunction: String, nCtx: Int, hiddenSize: Int,
                nHead: Int, nLayer: Int, vocabSize: Int, layerNormEpsilon: Float) {
        self.activationFunction = activationFunction
        self.nCtx = nCtx
        self.hiddenSize = hiddenSize
        self.nHead = nHead
        self.nLayer = nLayer
        self.vocabSize = vocabSize
        self.layerNormEpsilon = layerNormEpsilon
    }
}

// MARK: - T3 Configuration

/// Configuration for the T3 (Token-To-Token) model.
/// Shared by both the LLaMA (Regular) and GPT-2 (Turbo) variants.
/// Maps to Python's `T3Config` dataclass.
public struct ChatterboxT3Config: Sendable {
    /// Text vocabulary size.
    public let textTokensDictSize: Int
    /// Start-of-text token id.
    public let startTextToken: Int
    /// End-of-text token id (stop signal).
    public let stopTextToken: Int
    /// Maximum text token sequence length.
    public let maxTextTokens: Int
    /// Speech vocabulary size (including special tokens).
    public let speechTokensDictSize: Int
    /// Start-of-speech token id.
    public let startSpeechToken: Int
    /// End-of-speech token id (stop signal).
    public let stopSpeechToken: Int
    /// Maximum speech token sequence length.
    public let maxSpeechTokens: Int
    /// Backbone config name — `"GPT2_medium"` for Turbo, `"Llama_520M"` for Regular.
    public let llamaConfigName: String
    /// Conditioning speech-prompt length (number of speech tokens).
    public let speechCondPromptLen: Int
    /// Speaker embedding dimension (always 256).
    public let speakerEmbedSize: Int
    /// Whether to use a Perceiver resampler for the conditioning (Regular only).
    public let usePerceiverResampler: Bool
    /// Whether to use emotion adversarial conditioning (Regular only).
    public let emotionAdv: Bool

    /// True when this config uses the GPT-2 (Turbo) backbone.
    public var isGPT: Bool { llamaConfigName.contains("GPT2") }

    /// Number of transformer layers in the backbone.
    public var numLayers: Int {
        isGPT ? ChatterboxGPT2Config.medium.nLayer : 30  // LLaMA 520M has 30 layers
    }

    /// Hidden size of the backbone.
    public var hiddenSize: Int {
        1024  // Both LLaMA 520M and GPT-2 Medium use 1024
    }

    /// Default English-only config (LLaMA backbone, Regular model).
    public static let englishOnly = ChatterboxT3Config(
        textTokensDictSize: 704,
        startTextToken: 255,
        stopTextToken: 0,
        maxTextTokens: 2048,
        speechTokensDictSize: 8194,
        startSpeechToken: 6561,
        stopSpeechToken: 6562,
        maxSpeechTokens: 4096,
        llamaConfigName: "Llama_520M",
        speechCondPromptLen: 150,
        speakerEmbedSize: 256,
        usePerceiverResampler: true,
        emotionAdv: true
    )

    /// Default Turbo config (GPT-2 backbone).
    public static let turbo = ChatterboxT3Config(
        textTokensDictSize: 50276,
        startTextToken: 255,
        stopTextToken: 0,
        maxTextTokens: 2048,
        speechTokensDictSize: 6563,
        startSpeechToken: 6561,
        stopSpeechToken: 6562,
        maxSpeechTokens: 4096,
        llamaConfigName: "GPT2_medium",
        speechCondPromptLen: 375,
        speakerEmbedSize: 256,
        usePerceiverResampler: false,
        emotionAdv: false
    )

    /// Decode the `t3` (or `t3_config`) sub-block of a `config.json`.
    /// Returns `nil` when mandatory fields are absent.
    public static func from(_ raw: [String: Any]) -> ChatterboxT3Config? {
        func i(_ k: String) -> Int? {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return nil
        }
        guard let speechTokensDictSize = i("speech_tokens_dict_size"),
              let startSpeechToken = i("start_speech_token"),
              let stopSpeechToken = i("stop_speech_token")
        else { return nil }

        let llamaConfigName = raw["llama_config_name"] as? String ?? "Llama_520M"
        return ChatterboxT3Config(
            textTokensDictSize: i("text_tokens_dict_size") ?? 704,
            startTextToken: i("start_text_token") ?? 255,
            stopTextToken: i("stop_text_token") ?? 0,
            maxTextTokens: i("max_text_tokens") ?? 2048,
            speechTokensDictSize: speechTokensDictSize,
            startSpeechToken: startSpeechToken,
            stopSpeechToken: stopSpeechToken,
            maxSpeechTokens: i("max_speech_tokens") ?? 4096,
            llamaConfigName: llamaConfigName,
            speechCondPromptLen: i("speech_cond_prompt_len") ?? 150,
            speakerEmbedSize: i("speaker_embed_size") ?? 256,
            usePerceiverResampler: raw["use_perceiver_resampler"] as? Bool ?? true,
            emotionAdv: raw["emotion_adv"] as? Bool ?? true
        )
    }

    public init(textTokensDictSize: Int, startTextToken: Int, stopTextToken: Int,
                maxTextTokens: Int, speechTokensDictSize: Int, startSpeechToken: Int,
                stopSpeechToken: Int, maxSpeechTokens: Int, llamaConfigName: String,
                speechCondPromptLen: Int, speakerEmbedSize: Int,
                usePerceiverResampler: Bool, emotionAdv: Bool) {
        self.textTokensDictSize = textTokensDictSize
        self.startTextToken = startTextToken
        self.stopTextToken = stopTextToken
        self.maxTextTokens = maxTextTokens
        self.speechTokensDictSize = speechTokensDictSize
        self.startSpeechToken = startSpeechToken
        self.stopSpeechToken = stopSpeechToken
        self.maxSpeechTokens = maxSpeechTokens
        self.llamaConfigName = llamaConfigName
        self.speechCondPromptLen = speechCondPromptLen
        self.speakerEmbedSize = speakerEmbedSize
        self.usePerceiverResampler = usePerceiverResampler
        self.emotionAdv = emotionAdv
    }
}

// MARK: - Top-Level Config

/// Top-level Chatterbox model configuration, decoded from `config.json`.
///
/// Supports two config formats:
///   - Regular: minimal `{"model_type": "chatterbox"}` with per-field defaults
///   - Turbo: full config with `t3`, `gpt2`, `voice_encoder`, `s3gen` sections
public struct ChatterboxConfig: Sendable {
    /// `model_type` from `config.json` — `"chatterbox"` or `"chatterbox_turbo"`.
    public let modelType: String
    /// T3 backbone configuration (speaker tokens, backbone name, etc.).
    public let t3: ChatterboxT3Config
    /// GPT-2 backbone config (Turbo only, `nil` for Regular).
    public let gpt2: ChatterboxGPT2Config?
    /// S3 tokenizer sample rate — 16 kHz for both variants.
    public let s3SampleRate: Int
    /// S3Gen output sample rate — 24 kHz for both variants.
    public let s3genSampleRate: Int
    /// Output audio sample rate (same as `s3genSampleRate`).
    public let sampleRate: Int
    /// Reference audio conditioning window in samples (encoder side, 16 kHz).
    public let encCondLen: Int
    /// Reference audio conditioning window in samples (decoder side, 24 kHz).
    public let decCondLen: Int
    /// Whether the flow decoder uses distilled meanflow (Turbo) or ODE (Regular).
    public let meanflow: Bool
    /// Flow decoder input channel count (conv-style).
    public let decoderInChannels: Int
    /// Flow decoder mel output channel count (80 mel bins).
    public let decoderOutChannels: Int
    /// Flow decoder conv channel widths.
    public let decoderChannels: [Int]
    /// Flow decoder residual blocks per stage.
    public let decoderNBlocks: Int
    /// Flow decoder mid-block count.
    public let decoderNumMidBlocks: Int
    /// Flow decoder attention head count.
    public let decoderNumHeads: Int
    /// Flow decoder attention head dimension.
    public let decoderAttentionHeadDim: Int

    /// True when the T3 backbone is GPT-2 (Turbo), false for LLaMA (Regular).
    public var isTurbo: Bool {
        modelType == "chatterbox_turbo" || t3.isGPT
    }

    /// Default Regular model configuration.
    public static let `default` = ChatterboxConfig(
        modelType: "chatterbox",
        t3: .englishOnly,
        gpt2: nil,
        s3SampleRate: 16_000,
        s3genSampleRate: 24_000,
        sampleRate: 24_000,
        encCondLen: 6 * 16_000,
        decCondLen: 10 * 24_000,
        meanflow: false,
        decoderInChannels: 320,
        decoderOutChannels: 80,
        decoderChannels: [256],
        decoderNBlocks: 4,
        decoderNumMidBlocks: 12,
        decoderNumHeads: 8,
        decoderAttentionHeadDim: 64
    )

    /// Default Turbo model configuration.
    public static let turbo = ChatterboxConfig(
        modelType: "chatterbox_turbo",
        t3: .turbo,
        gpt2: .medium,
        s3SampleRate: 16_000,
        s3genSampleRate: 24_000,
        sampleRate: 24_000,
        encCondLen: 15 * 16_000,
        decCondLen: 10 * 24_000,
        meanflow: true,
        decoderInChannels: 320,
        decoderOutChannels: 80,
        decoderChannels: [256],
        decoderNBlocks: 4,
        decoderNumMidBlocks: 12,
        decoderNumHeads: 8,
        decoderAttentionHeadDim: 64
    )

    /// Decode a `ModelConfig`-wrapped `config.json`.
    ///
    /// Handles both the minimal `{"model_type": "chatterbox"}` format and the
    /// full Turbo format with nested `t3`, `gpt2`, `voice_encoder`, `s3gen` keys.
    public static func from(_ config: ModelConfig) -> ChatterboxConfig? {
        guard let mt = config.modelType,
              ["chatterbox", "chatterbox_turbo"].contains(mt) else { return nil }

        let raw = config.raw
        let isTurboType = (mt == "chatterbox_turbo")

        // --- T3 config: try "t3_config" key, then "t3", then defaults ---
        let t3Config: ChatterboxT3Config
        if let t3Raw = raw["t3_config"] as? [String: Any],
           let parsed = ChatterboxT3Config.from(t3Raw) {
            t3Config = parsed
        } else if let t3Raw = raw["t3"] as? [String: Any],
                  let parsed = ChatterboxT3Config.from(t3Raw) {
            t3Config = parsed
        } else {
            t3Config = isTurboType ? .turbo : .englishOnly
        }

        // --- GPT-2 config (Turbo only) ---
        let gpt2Config: ChatterboxGPT2Config?
        if let gpt2Raw = raw["gpt2"] as? [String: Any] {
            gpt2Config = ChatterboxGPT2Config.from(gpt2Raw)
        } else if isTurboType || t3Config.isGPT {
            gpt2Config = .medium
        } else {
            gpt2Config = nil
        }

        // --- Sample rates ---
        let s3Sr = config.int("s3_sr") ?? 16_000
        let s3genSr = config.int("s3gen_sr") ?? config.int("sample_rate") ?? 24_000

        // --- Conditioning window lengths (absolute or seconds-based) ---
        let encCondLen: Int
        if let enc = config.int("enc_cond_len") {
            encCondLen = enc
        } else if let encSec = config.int("enc_cond_len_seconds") {
            encCondLen = encSec * s3Sr
        } else {
            encCondLen = (isTurboType ? 15 : 6) * s3Sr
        }

        let decCondLen: Int
        if let dec = config.int("dec_cond_len") {
            decCondLen = dec
        } else if let decSec = config.int("dec_cond_len_seconds") {
            decCondLen = decSec * s3genSr
        } else {
            decCondLen = 10 * s3genSr
        }

        // --- S3Gen decoder config (from nested "s3gen" block or top-level keys) ---
        let s3genRaw = raw["s3gen"] as? [String: Any]
        func s3i(_ key: String, _ topKey: String, _ def: Int) -> Int {
            if let v = s3genRaw?[key] as? Int { return v }
            if let v = raw[topKey] as? Int { return v }
            return def
        }

        // meanflow=true for Turbo (distilled), false for Regular (ODE + CFG)
        let meanflow: Bool
        if let mf = s3genRaw?["meanflow"] as? Bool {
            meanflow = mf
        } else if let mf = raw["meanflow"] as? Bool {
            meanflow = mf
        } else {
            meanflow = isTurboType || t3Config.isGPT
        }

        let decoderChannels: [Int]
        if let dc = s3genRaw?["decoder_channels"] as? [Int] {
            decoderChannels = dc
        } else if let dc = raw["decoder_channels"] as? [Int] {
            decoderChannels = dc
        } else {
            decoderChannels = [256]
        }

        return ChatterboxConfig(
            modelType: mt,
            t3: t3Config,
            gpt2: gpt2Config,
            s3SampleRate: s3Sr,
            s3genSampleRate: s3genSr,
            sampleRate: s3genSr,
            encCondLen: encCondLen,
            decCondLen: decCondLen,
            meanflow: meanflow,
            decoderInChannels: s3i("decoder_in_channels", "decoder_in_channels", 320),
            decoderOutChannels: s3i("decoder_out_channels", "decoder_out_channels", 80),
            decoderChannels: decoderChannels,
            decoderNBlocks: s3i("decoder_n_blocks", "decoder_n_blocks", 4),
            decoderNumMidBlocks: s3i("decoder_num_mid_blocks", "decoder_num_mid_blocks", 12),
            decoderNumHeads: s3i("decoder_num_heads", "decoder_num_heads", 8),
            decoderAttentionHeadDim: s3i("decoder_attention_head_dim",
                                          "decoder_attention_head_dim", 64)
        )
    }

    public init(modelType: String, t3: ChatterboxT3Config,
                gpt2: ChatterboxGPT2Config?,
                s3SampleRate: Int, s3genSampleRate: Int, sampleRate: Int,
                encCondLen: Int, decCondLen: Int, meanflow: Bool,
                decoderInChannels: Int, decoderOutChannels: Int,
                decoderChannels: [Int], decoderNBlocks: Int,
                decoderNumMidBlocks: Int, decoderNumHeads: Int,
                decoderAttentionHeadDim: Int) {
        self.modelType = modelType
        self.t3 = t3
        self.gpt2 = gpt2
        self.s3SampleRate = s3SampleRate
        self.s3genSampleRate = s3genSampleRate
        self.sampleRate = sampleRate
        self.encCondLen = encCondLen
        self.decCondLen = decCondLen
        self.meanflow = meanflow
        self.decoderInChannels = decoderInChannels
        self.decoderOutChannels = decoderOutChannels
        self.decoderChannels = decoderChannels
        self.decoderNBlocks = decoderNBlocks
        self.decoderNumMidBlocks = decoderNumMidBlocks
        self.decoderNumHeads = decoderNumHeads
        self.decoderAttentionHeadDim = decoderAttentionHeadDim
    }
}

// ─── Constants ────────────────────────────────────────────────────────

/// Global constants for Chatterbox.
public enum ChatterboxConstants {
    /// S3 tokenizer sample rate (16 kHz).
    public static let s3SampleRate = 16_000
    /// S3Gen output sample rate (24 kHz).
    public static let s3genSampleRate = 24_000
    /// Speech vocabulary size (before special tokens).
    public static let speechVocabSize = 6561
    /// S3Gen silence token id.
    public static let silenceToken = 4299
    /// Default encoder conditioning window: 6 s at 16 kHz (Regular).
    public static let encCondLenRegular = 6 * s3SampleRate
    /// Default encoder conditioning window: 15 s at 16 kHz (Turbo).
    public static let encCondLenTurbo = 15 * s3SampleRate
    /// Default decoder conditioning window: 10 s at 24 kHz.
    public static let decCondLen = 10 * s3genSampleRate
}

// ─── Model ────────────────────────────────────────────────────────────

/// A loaded Chatterbox TTS model.
///
/// Stage 1 (this file): config decoding + registry detection + retained
/// safetensors weight bundle. The T3 backbone, S3Gen decoder, VoiceEncoder,
/// and HiFi-GAN vocoder are follow-on stages; `synthesize` throws a
/// descriptive `ChatterboxError.synthesisNotWired` until they land.
///
/// - SeeAlso: `ChatterboxError.synthesisNotWired` for staging details.
public final class ChatterboxModel: @unchecked Sendable {
    /// Decoded configuration.
    public let config: ChatterboxConfig
    /// Retained safetensors bundle — available for future stage inspection.
    public let weights: SafeTensorsBundle

    public var sampleRate: Int { config.sampleRate }

    public init(config: ChatterboxConfig, weights: SafeTensorsBundle) {
        self.config = config
        self.weights = weights
    }

    /// Synthesize speech from `text`. Throws `ChatterboxError.synthesisNotWired`
    /// until the T3 + S3Gen + VoiceEncoder + HiFi-GAN stages land.
    ///
    /// Stage 2 (T3 backbone), stage 3 (S3Gen flow decoder), stage 4 (VoiceEncoder
    /// + HiFi-GAN vocoder) are the three remaining sub-ports. See
    /// `ChatterboxError.synthesisNotWired` for the full description.
    public func synthesize(
        text: String,
        device: Device = .shared
    ) throws -> [Float] {
        _ = text
        _ = device
        throw ChatterboxError.synthesisNotWired
    }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension ChatterboxModel {

    /// `model_type` values this family handles.
    public static let modelTypes: Set<String> = ["chatterbox", "chatterbox_turbo"]

    /// Whether a decoded `config.json` describes a Chatterbox checkpoint.
    ///
    /// Detection strategy:
    ///   1. `model_type` ∈ `modelTypes` — canonical marker.
    ///   2. `architecture` = `"chatterbox"` or `"chatterbox_turbo"` — some HF
    ///      checkpoints (e.g. `chatterbox-turbo-fp16`) set this field.
    ///   3. Structural: a `t3` or `t3_config` block whose
    ///      `speech_tokens_dict_size` falls in the expected Chatterbox range.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, modelTypes.contains(arch) { return true }
        // Structural detection: the T3 config is the canonical marker.
        for key in ["t3", "t3_config"] {
            if let t3Raw = config.raw[key] as? [String: Any],
               let speechVocab = t3Raw["speech_tokens_dict_size"] as? Int,
               (6000 ... 9000).contains(speechVocab) {
                return true
            }
        }
        return false
    }

    /// Load a Chatterbox checkpoint from a resolved snapshot directory.
    ///
    /// Loads and decodes `config.json`, retains the safetensors weight bundle
    /// for future stage inspection, and returns a `ChatterboxModel`. A missing
    /// `config.json` falls back to per-variant defaults (Regular or Turbo) based
    /// on the safetensors key prefix patterns.
    public static func load(
        directory: URL,
        device: Device = .shared
    ) throws -> ChatterboxModel {
        // Load config — fall back to defaults when config.json is absent or
        // minimal (e.g. the Regular model ships `{"model_type":"chatterbox","version":"1.0"}`).
        let modelConfig = try ModelConfig.load(from: directory)
        let chatterboxConfig = ChatterboxConfig.from(modelConfig) ?? .default

        // Load safetensors bundle (excludes conds.safetensors by design;
        // that file is conditioning-only and loaded separately at inference time).
        let bundle = try SafeTensorsBundle(directory: directory, device: device)

        return ChatterboxModel(config: chatterboxConfig, weights: bundle)
    }
}
