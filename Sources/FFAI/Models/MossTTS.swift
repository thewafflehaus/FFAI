// MossTTS — OpenMOSS MOSS-TTS family (Qwen3 LLM backbone, delay-pattern codec).
//
// MOSS-TTS is a two-stage LLM-based TTS pipeline:
//
//   Stage 1 — Language Model (Qwen3 backbone):
//     text + audio tokens ──Qwen3 decoder──▶ audio token logits for nVQ channels
//     conditioned via a multi-channel delay pattern: each codebook lags by one
//     step relative to the previous, enabling parallel generation in a single
//     autoregressive pass.
//
//   Stage 2 — Audio Tokenizer (MOSS audio codec):
//     audio token IDs ──codec decoder──▶ 24 kHz waveform
//
// The MOSS-TTS-8B checkpoint uses:
//   • Qwen3-8B backbone (`language_config` with `model_type: "qwen3"`)
//   • 32 VQ codebooks (`n_vq = 32`)
//   • 24 kHz sample rate (`sampling_rate = 24000`)
//   • `MossTTSDelayModel` architecture with delay-pattern token generation
//
// ## Scope note — STAGED PORT
//
// This file is stage 1: config decoding, `AudioModelRegistry` detection,
// checkpoint weight-bundle retention, and the top-level `MossTTSModel`
// scaffold. The full delay-pattern autoregressive generation loop and the
// MOSS audio codec decoder (MOSS-Audio-Tokenizer) are follow-on stages.
// Until they land, `synthesize` throws `MossTTSError.synthesisNotWired` —
// see the error description for what is missing.
//
// Reference implementation:
//   ~/Development/personal/ai/mlx-audio-swift/Sources/MLXAudioTTS/Models/MossTTS/
// Checkpoint: `mlx-community/MOSS-TTS-8B-8bit`

import Foundation

// ─── Errors ──────────────────────────────────────────────────────────

public enum MossTTSError: Error, CustomStringConvertible {
    /// The full synthesis pipeline is not wired yet. Config + detection
    /// land first; the Qwen3 generation loop and MOSS audio tokenizer
    /// decoder are follow-on stages. Use `MossTTSModel.weights` to
    /// inspect loaded weights in the interim.
    case synthesisNotWired
    /// A required config field is missing from `config.json`.
    case missingConfig(String)

    public var description: String {
        switch self {
        case .synthesisNotWired:
            return "MossTTS: the delay-pattern generation loop and MOSS audio "
                + "tokenizer decoder are not yet wired in this build. Stage 1 "
                + "ships config decoding + detection. Follow-on stages will wire "
                + "the full synthesis pipeline (Qwen3 backbone forward + codec decode)."
        case .missingConfig(let field):
            return "MossTTS: required config field missing: \(field)"
        }
    }
}

// ─── Language model sub-config ────────────────────────────────────────

/// Qwen3 language-model configuration nested under `language_config`
/// in the MOSS-TTS `config.json`. Matches the Python `MossQwen3Config`.
public struct MossTTSLanguageConfig: Sendable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float

    /// Decode from the nested `language_config` dictionary.
    public static func from(_ raw: [String: Any]) -> MossTTSLanguageConfig {
        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }
        func f(_ k: String, _ d: Float) -> Float {
            if let v = raw[k] as? Double { return Float(v) }
            if let v = raw[k] as? Int { return Float(v) }
            // rope_theta may be nested under rope_parameters
            if let rp = raw["rope_parameters"] as? [String: Any],
               let inner = rp[k] as? Double { return Float(inner) }
            return d
        }
        let hidden = i("hidden_size", 4096)
        let nHeads = i("num_attention_heads", 32)
        return MossTTSLanguageConfig(
            modelType: raw["model_type"] as? String ?? "qwen3",
            vocabSize: i("vocab_size", 155_648),
            hiddenSize: hidden,
            numHiddenLayers: i("num_hidden_layers", 36),
            intermediateSize: i("intermediate_size", 12_288),
            numAttentionHeads: nHeads,
            numKeyValueHeads: i("num_key_value_heads", 8),
            headDim: i("head_dim", hidden / nHeads),
            rmsNormEps: f("rms_norm_eps", 1e-6),
            maxPositionEmbeddings: i("max_position_embeddings", 40_960),
            ropeTheta: f("rope_theta", 1_000_000)
        )
    }
}

// ─── Top-level configuration ──────────────────────────────────────────

/// MOSS-TTS hyper-parameters decoded from `config.json`.
/// Matches the Python `MossTTSDelayConfig`.
public struct MossTTSConfig: Sendable {
    /// Always `"moss_tts"` for the 8B checkpoint.
    public let modelType: String
    /// Architecture string from the `architectures` array (e.g. `"MossTTSDelayModel"`).
    public let architecture: String?
    /// Qwen3 language model sub-config.
    public let languageConfig: MossTTSLanguageConfig
    /// Number of VQ codebooks (32 for MOSS-TTS-8B).
    public let nVQ: Int
    /// Vocabulary size for audio tokens (1024).
    public let audioVocabSize: Int
    /// Token id for audio user slot (151654).
    public let audioUserSlotTokenID: Int
    /// Token id for audio assistant generation slot (151656).
    public let audioAssistantGenSlotTokenID: Int
    /// Token id for audio assistant delay slot (151662).
    public let audioAssistantDelaySlotTokenID: Int
    /// Token id for `<audio_start>` (151652).
    public let audioStartTokenID: Int
    /// Token id for `<audio_end>` (151653).
    public let audioEndTokenID: Int
    /// Padding code for audio channels (1024).
    public let audioPadCode: Int
    /// Padding token id (151643).
    public let padTokenID: Int
    /// `<|im_start|>` token id (151644).
    public let imStartTokenID: Int
    /// `<|im_end|>` token id (151645).
    public let imEndTokenID: Int
    /// Output waveform sample rate (24000).
    public let samplingRate: Int
    /// Pretrained name or path for the MOSS audio tokenizer, if specified.
    public let audioTokenizerPretrainedNameOrPath: String?

    public var sampleRate: Int { samplingRate }

    /// Decode from a top-level `ModelConfig`.
    public static func from(_ config: ModelConfig) -> MossTTSConfig? {
        let raw = config.raw

        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }

        let langRaw = raw["language_config"] as? [String: Any] ?? [:]
        let langConfig = MossTTSLanguageConfig.from(langRaw)

        return MossTTSConfig(
            modelType: config.modelType ?? "moss_tts",
            architecture: config.architecture,
            languageConfig: langConfig,
            nVQ: i("n_vq", 32),
            audioVocabSize: i("audio_vocab_size", 1024),
            audioUserSlotTokenID: i("audio_user_slot_token_id", 151_654),
            audioAssistantGenSlotTokenID: i("audio_assistant_gen_slot_token_id", 151_656),
            audioAssistantDelaySlotTokenID: i("audio_assistant_delay_slot_token_id", 151_662),
            audioStartTokenID: i("audio_start_token_id", 151_652),
            audioEndTokenID: i("audio_end_token_id", 151_653),
            audioPadCode: i("audio_pad_code", 1024),
            padTokenID: i("pad_token_id", 151_643),
            imStartTokenID: i("im_start_token_id", 151_644),
            imEndTokenID: i("im_end_token_id", 151_645),
            samplingRate: i("sampling_rate", i("sample_rate", 24_000)),
            audioTokenizerPretrainedNameOrPath: raw["audio_tokenizer_pretrained_name_or_path"] as? String
        )
    }
}

// ─── Model ───────────────────────────────────────────────────────────

/// A loaded MOSS-TTS model. Owns the decoded config and the safetensors
/// weight bundle for future stage inspection.
///
/// The full synthesis pipeline — delay-pattern autoregressive generation
/// over the Qwen3 backbone and MOSS audio tokenizer decode — is a follow-on
/// stage. Set `synthesize` to throw `MossTTSError.synthesisNotWired` in
/// stage 1.
public final class MossTTSModel: @unchecked Sendable {
    /// Decoded configuration.
    public let config: MossTTSConfig
    /// Retained safetensors bundle — available for future stage inspection.
    public let weights: SafeTensorsBundle

    public var sampleRate: Int { config.samplingRate }

    public init(config: MossTTSConfig, weights: SafeTensorsBundle) {
        self.config = config
        self.weights = weights
    }

    /// Synthesize speech from `text`. Throws `MossTTSError.synthesisNotWired`
    /// until the Qwen3 backbone generation loop and MOSS audio tokenizer
    /// decoder stages land.
    ///
    /// Stage 2 will wire the delay-pattern autoregressive generation (Qwen3
    /// forward + multi-codebook sampling). Stage 3 will wire the MOSS audio
    /// tokenizer decoder (codec token IDs → 24 kHz waveform).
    public func synthesize(
        text: String,
        device: Device = .shared
    ) throws -> [Float] {
        _ = text
        _ = device
        throw MossTTSError.synthesisNotWired
    }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension MossTTSModel {

    /// `model_type` values this family handles.
    public static let modelTypes: Set<String> = ["moss_tts", "moss_tts_delay"]
    /// Architecture strings this family handles.
    public static let architectures: Set<String> = ["MossTTSDelayModel"]

    /// Whether a decoded `config.json` describes a MOSS-TTS checkpoint.
    ///
    /// Detection strategy:
    ///   1. `model_type` ∈ `modelTypes` — canonical marker.
    ///   2. `architecture` ∈ `architectures` — some checkpoints set this.
    ///   3. Structural: a `language_config` block with `model_type == "qwen3"`
    ///      plus `n_vq` present — the structural marker for MOSS-TTS (8B).
    ///
    /// MossTTSNano is checked before MossTTS in the registry so this method
    /// never needs to exclude nano configs explicitly.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) { return true }
        // Structural fallback: Qwen3 language_config sub-block + n_vq.
        if let langRaw = config.raw["language_config"] as? [String: Any],
           langRaw["model_type"] as? String == "qwen3",
           config.raw["n_vq"] != nil {
            return true
        }
        return false
    }

    /// Load a MOSS-TTS checkpoint from a resolved snapshot directory.
    ///
    /// Loads and decodes `config.json`, retains the safetensors weight bundle,
    /// and returns a `MossTTSModel`. The synthesis pipeline is not wired yet
    /// — `synthesize` will throw `MossTTSError.synthesisNotWired`.
    public static func load(
        directory: URL,
        device: Device = .shared
    ) throws -> MossTTSModel {
        let modelConfig = try ModelConfig.load(from: directory)
        guard let config = MossTTSConfig.from(modelConfig) else {
            throw MossTTSError.missingConfig("language_config")
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return MossTTSModel(config: config, weights: bundle)
    }
}
