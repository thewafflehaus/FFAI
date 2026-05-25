// MossTTSNano — OpenMOSS MOSS-TTS-Nano family (GPT-2 backbone, local transformer codec).
//
// MOSS-TTS-Nano is a compact (~100M parameter) TTS model with a two-model
// pipeline:
//
//   Global transformer (GPT-2 based):
//     text + audio embeddings ──GPT-2──▶ global hidden state
//
//   Local transformer (smaller GPT-2 based):
//     global hidden ──local GPT-2──▶ per-codebook audio token logits
//     generates all nVQ codebook tokens sequentially within one frame
//
// The MOSS-TTS-Nano-100M checkpoint uses:
//   • GPT-2 Small backbone (768 hidden, 12 layers, 12 heads)
//   • 16 VQ codebooks (`n_vq = 16`)
//   • 48 kHz sample rate (`audio_tokenizer_sample_rate = 48000`)
//   • `MossTTSNanoForCausalLM` architecture
//   • MOSS-Audio-Tokenizer-Nano codec (`mlx-community/MOSS-Audio-Tokenizer-Nano`)
//
// ## Scope note — STAGED PORT
//
// This file is stage 1: config decoding, `AudioModelRegistry` detection,
// checkpoint weight-bundle retention, and the top-level `MossTTSNanoModel`
// scaffold. The full local-transformer generation loop and the MOSS Nano
// audio codec decoder are follow-on stages. Until they land, `synthesize`
// throws `MossTTSNanoError.synthesisNotWired` — see the error description
// for what is missing.
//
// Reference implementation:
//   ~/Development/personal/ai/mlx-audio-swift/Sources/MLXAudioTTS/Models/MossTTSNano/
// Checkpoint: `mlx-community/MOSS-TTS-Nano-100M`
// Codec dep:  `mlx-community/MOSS-Audio-Tokenizer-Nano`

import Foundation

// ─── Errors ──────────────────────────────────────────────────────────

public enum MossTTSNanoError: Error, CustomStringConvertible {
    /// The full synthesis pipeline is not wired yet. Config + detection
    /// land first; the local-transformer generation loop and MOSS Nano
    /// audio tokenizer decoder are follow-on stages. Use
    /// `MossTTSNanoModel.weights` to inspect loaded weights in the interim.
    case synthesisNotWired
    /// A required config field is missing from `config.json`.
    case missingConfig(String)

    public var description: String {
        switch self {
        case .synthesisNotWired:
            return "MossTTSNano: the local-transformer generation loop and MOSS Nano "
                + "audio tokenizer decoder are not yet wired in this build. Stage 1 "
                + "ships config decoding + detection. Follow-on stages will wire "
                + "the full synthesis pipeline (global GPT-2 + local GPT-2 + codec decode)."
        case .missingConfig(let field):
            return "MossTTSNano: required config field missing: \(field)"
        }
    }
}

// ─── GPT-2 sub-config ─────────────────────────────────────────────────

/// GPT-2 configuration nested under `gpt2_config` in the MOSS-TTS-Nano
/// `config.json`. Matches the Python `MossGPT2Config`.
public struct MossTTSNanoGPT2Config: Sendable {
    public let modelType: String
    /// Vocabulary size (default 16384).
    public let vocabSize: Int
    /// Maximum sequence length (default 32768).
    public let nPositions: Int
    /// Context length (defaults to `nPositions`).
    public let nCtx: Int
    /// Hidden (embedding) dimension (default 768).
    public let nEmbd: Int
    /// Number of transformer layers (default 12).
    public let nLayer: Int
    /// Number of attention heads (default 12).
    public let nHead: Int
    /// Feed-forward inner size (default 3072 = 4 × nEmbd).
    public let nInner: Int
    /// Layer-norm epsilon (default 1e-5).
    public let layerNormEpsilon: Float
    /// Positional embedding type (`"rope"` for MOSS Nano).
    public let positionEmbeddingType: String
    /// RoPE base frequency (default 10000).
    public let ropeBase: Float
    /// Padding token id (default 3).
    public let padTokenID: Int
    /// BOS token id (default 1).
    public let bosTokenID: Int
    /// EOS token id (default 2).
    public let eosTokenID: Int

    public var headDim: Int { nEmbd / nHead }
    public var intermediateSize: Int { nInner }
    public var hiddenSize: Int { nEmbd }

    /// Decode from the nested `gpt2_config` dictionary.
    public static func from(_ raw: [String: Any]) -> MossTTSNanoGPT2Config {
        func i(_ k: String, _ d: Int) -> Int {
            // Support both snake_case ("n_embd") and alternate keys ("hidden_size")
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }
        func f(_ k: String, _ d: Float) -> Float {
            if let v = raw[k] as? Double { return Float(v) }
            if let v = raw[k] as? Int { return Float(v) }
            return d
        }
        let nEmbd = i("n_embd", i("hidden_size", 768))
        let nPositions = i("n_positions", 32_768)
        let nInner = i("n_inner", i("intermediate_size", 4 * nEmbd))
        return MossTTSNanoGPT2Config(
            modelType: raw["model_type"] as? String ?? "gpt2",
            vocabSize: i("vocab_size", 16_384),
            nPositions: nPositions,
            nCtx: i("n_ctx", nPositions),
            nEmbd: nEmbd,
            nLayer: i("n_layer", i("num_hidden_layers", 12)),
            nHead: i("n_head", i("num_attention_heads", 12)),
            nInner: nInner,
            layerNormEpsilon: f("layer_norm_epsilon", 1e-5),
            positionEmbeddingType: raw["position_embedding_type"] as? String ?? "rope",
            ropeBase: f("rope_base", 10_000),
            padTokenID: i("pad_token_id", 3),
            bosTokenID: i("bos_token_id", 1),
            eosTokenID: i("eos_token_id", 2)
        )
    }
}

// ─── Top-level configuration ──────────────────────────────────────────

/// MOSS-TTS-Nano hyper-parameters decoded from `config.json`.
/// Matches the Python `MossTTSNanoConfig`.
public struct MossTTSNanoConfig: Sendable {
    /// Always `"moss_tts_nano"` for the Nano checkpoint.
    public let modelType: String
    /// Architecture string (e.g. `"MossTTSNanoForCausalLM"`).
    public let architecture: String?
    /// GPT-2 backbone configuration.
    public let gpt2Config: MossTTSNanoGPT2Config
    /// Number of VQ codebooks (16 for MOSS-TTS-Nano).
    public let nVQ: Int
    /// Vocabulary size for audio tokens per codebook (1024).
    public let audioVocabSize: Int
    /// Sizes per codebook (all 1024 for the current Nano checkpoint).
    public let audioCodebookSizes: [Int]
    /// Padding token id for audio channels (1024).
    public let audioPadTokenID: Int
    /// Text padding token id (3).
    public let padTokenID: Int
    /// `<|im_start|>` token id (4).
    public let imStartTokenID: Int
    /// `<|im_end|>` token id (5).
    public let imEndTokenID: Int
    /// `<audio_start>` token id (6).
    public let audioStartTokenID: Int
    /// `<audio_end>` token id (7).
    public let audioEndTokenID: Int
    /// Audio user slot token id (8).
    public let audioUserSlotTokenID: Int
    /// Audio assistant slot token id (9).
    public let audioAssistantSlotTokenID: Int
    /// Audio tokenizer type identifier (`"moss-audio-tokenizer-nano"`).
    public let audioTokenizerType: String
    /// Pretrained name or path for the MOSS Nano audio tokenizer.
    public let audioTokenizerPretrainedNameOrPath: String?
    /// Sample rate of the audio tokenizer (48000 Hz for Nano).
    public let audioTokenizerSampleRate: Int
    /// Number of layers in the local transformer (1 for Nano).
    public let localTransformerLayers: Int
    /// Hidden size of the GPT-2 backbone (matches `gpt2Config.nEmbd`).
    public let hiddenSize: Int
    /// Vocabulary size (matches `gpt2Config.vocabSize`).
    public let vocabSize: Int

    public var sampleRate: Int { audioTokenizerSampleRate }

    /// Build a GPT-2 config for the local transformer: same backbone dims
    /// but with `nPositions = nVQ + 1` and `nLayer = localTransformerLayers`.
    public func localGPT2Config() -> MossTTSNanoGPT2Config {
        MossTTSNanoGPT2Config(
            modelType: gpt2Config.modelType,
            vocabSize: gpt2Config.vocabSize,
            nPositions: nVQ + 1,
            nCtx: nVQ + 1,
            nEmbd: gpt2Config.nEmbd,
            nLayer: localTransformerLayers,
            nHead: gpt2Config.nHead,
            nInner: gpt2Config.nInner,
            layerNormEpsilon: gpt2Config.layerNormEpsilon,
            positionEmbeddingType: gpt2Config.positionEmbeddingType,
            ropeBase: gpt2Config.ropeBase,
            padTokenID: gpt2Config.padTokenID,
            bosTokenID: gpt2Config.bosTokenID,
            eosTokenID: gpt2Config.eosTokenID
        )
    }

    /// Decode from a top-level `ModelConfig`.
    public static func from(_ config: ModelConfig) -> MossTTSNanoConfig? {
        let raw = config.raw

        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }

        let gptRaw = raw["gpt2_config"] as? [String: Any] ?? [:]
        let gpt = MossTTSNanoGPT2Config.from(gptRaw)
        let nVQ = i("n_vq", 16)
        let audioVocabSize = i("audio_vocab_size", 1024)
        let codebookSizes = raw["audio_codebook_sizes"] as? [Int]
            ?? Array(repeating: audioVocabSize, count: nVQ)

        return MossTTSNanoConfig(
            modelType: config.modelType ?? "moss_tts_nano",
            architecture: config.architecture,
            gpt2Config: gpt,
            nVQ: nVQ,
            audioVocabSize: audioVocabSize,
            audioCodebookSizes: codebookSizes,
            audioPadTokenID: i("audio_pad_token_id", 1024),
            padTokenID: i("pad_token_id", 3),
            imStartTokenID: i("im_start_token_id", 4),
            imEndTokenID: i("im_end_token_id", 5),
            audioStartTokenID: i("audio_start_token_id", 6),
            audioEndTokenID: i("audio_end_token_id", 7),
            audioUserSlotTokenID: i("audio_user_slot_token_id", 8),
            audioAssistantSlotTokenID: i("audio_assistant_slot_token_id", 9),
            audioTokenizerType: raw["audio_tokenizer_type"] as? String ?? "moss-audio-tokenizer-nano",
            audioTokenizerPretrainedNameOrPath: raw["audio_tokenizer_pretrained_name_or_path"] as? String,
            audioTokenizerSampleRate: i("audio_tokenizer_sample_rate", 48_000),
            localTransformerLayers: i("local_transformer_layers", 1),
            hiddenSize: i("hidden_size", gpt.nEmbd),
            vocabSize: i("vocab_size", gpt.vocabSize)
        )
    }
}

// ─── Model ───────────────────────────────────────────────────────────

/// A loaded MOSS-TTS-Nano model. Owns the decoded config and the safetensors
/// weight bundle for future stage inspection.
///
/// The full synthesis pipeline — global GPT-2 forward, local GPT-2 per-frame
/// codebook generation, and MOSS Nano audio tokenizer decode — is a follow-on
/// stage. Set `synthesize` to throw `MossTTSNanoError.synthesisNotWired` in
/// stage 1.
public final class MossTTSNanoModel: @unchecked Sendable {
    /// Decoded configuration.
    public let config: MossTTSNanoConfig
    /// Retained safetensors bundle — available for future stage inspection.
    public let weights: SafeTensorsBundle

    public var sampleRate: Int { config.audioTokenizerSampleRate }

    public init(config: MossTTSNanoConfig, weights: SafeTensorsBundle) {
        self.config = config
        self.weights = weights
    }

    /// Synthesize speech from `text`. Throws `MossTTSNanoError.synthesisNotWired`
    /// until the global GPT-2 + local GPT-2 generation loop and MOSS Nano
    /// audio tokenizer decoder stages land.
    ///
    /// Stage 2 will wire the global transformer forward and local-transformer
    /// per-codebook sampling. Stage 3 will wire the MOSS Nano audio tokenizer
    /// decoder (codec token IDs → 48 kHz waveform).
    public func synthesize(
        text: String,
        device: Device = .shared
    ) throws -> [Float] {
        _ = text
        _ = device
        throw MossTTSNanoError.synthesisNotWired
    }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension MossTTSNanoModel {

    /// `model_type` values this family handles.
    public static let modelTypes: Set<String> = ["moss_tts_nano"]
    /// Architecture strings this family handles.
    public static let architectures: Set<String> = ["MossTTSNanoForCausalLM"]

    /// Whether a decoded `config.json` describes a MOSS-TTS-Nano checkpoint.
    ///
    /// Detection strategy:
    ///   1. `model_type` ∈ `modelTypes` — canonical marker.
    ///   2. `architecture` ∈ `architectures` — some checkpoints set this.
    ///   3. Structural: a `gpt2_config` block plus `n_vq` present — the
    ///      structural marker for MOSS-TTS-Nano.
    ///
    /// Must be checked BEFORE `MossTTSModel.handles` in the registry because
    /// this is the more specific family.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) { return true }
        // Structural fallback: gpt2_config sub-block + n_vq.
        if config.raw["gpt2_config"] is [String: Any],
           config.raw["n_vq"] != nil {
            return true
        }
        return false
    }

    /// Load a MOSS-TTS-Nano checkpoint from a resolved snapshot directory.
    ///
    /// Loads and decodes `config.json`, retains the safetensors weight bundle,
    /// and returns a `MossTTSNanoModel`. The synthesis pipeline is not wired
    /// yet — `synthesize` will throw `MossTTSNanoError.synthesisNotWired`.
    public static func load(
        directory: URL,
        device: Device = .shared
    ) throws -> MossTTSNanoModel {
        let modelConfig = try ModelConfig.load(from: directory)
        guard let config = MossTTSNanoConfig.from(modelConfig) else {
            throw MossTTSNanoError.missingConfig("gpt2_config")
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return MossTTSNanoModel(config: config, weights: bundle)
    }
}
