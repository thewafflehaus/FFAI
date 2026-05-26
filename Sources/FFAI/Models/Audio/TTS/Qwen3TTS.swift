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
// Qwen3TTS — Qwen's text-to-speech family (Qwen3-TTS-Flash and kin).
//
// Qwen3TTS is the largest of the FFAI TTS ports: it is a four-part
// model — a **talker** (a Qwen3-style autoregressive acoustic
// transformer with 3D mRoPE), a **code predictor** (a small transformer
// that expands each talker step into a group of codec codes), a
// **speaker encoder** (an ECAPA-TDNN that turns a reference waveform
// into a speaker embedding), and an intrinsic **speech-tokenizer
// codec** (a conv + transformer encoder/decoder pair) that turns codec
// codes into a 24 kHz waveform:
//
//   text + ref-audio ──speaker encoder──▶ speaker embed ──┐
//        text ──text embed + projection──────────────────┼─▶ talker
//        ──autoregressive──▶ talker codes ──code predictor──▶ codec
//        codes ──speech-tokenizer decoder──▶ waveform
//
// ## Scope note — STAGED PORT
//
// Qwen3TTS lands in stages. This file is **stage 1**: the config
// decoding + family detection + registry plumbing. It decodes the
// nested `talker_config` / `tokenizer_config` / `speaker_encoder_config`
// blocks so `AudioModelRegistry` recognizes a Qwen3TTS checkpoint and
// reports `textToSpeech`, and it pins down the typed surface
// (`Qwen3TTSModel`, the talker hyper-parameters) the later stages build
// on.
//
// Stage 2 (the talker transformer — a Qwen3 block stack with 3D mRoPE),
// stage 3 (the code predictor), and stage 4 (the ECAPA speaker encoder
// + the speech-tokenizer codec) are follow-on ports. Until they land,
// `synthesize` throws `Qwen3TTSError.synthesisNotWired` — see the error
// description for what is missing. The talker's 3D mRoPE and the
// intrinsic codec are the two non-trivial pieces (an `Ops.rope2D`
// exists but mRoPE's interleaved 3-section layout differs); they are
// called out here so the staging is explicit rather than silent.

import Foundation

// ─── Configuration ───────────────────────────────────────────────────

/// Qwen3TTS talker hyper-parameters — the autoregressive acoustic
/// transformer at the heart of the model. A Qwen3-style block stack
/// (q_norm / k_norm, SwiGLU) driven by 3D multimodal RoPE.
public struct Qwen3TTSTalkerConfig: Sendable {
    /// Codec-token vocabulary size (the talker's output vocabulary).
    public let vocabSize: Int
    /// Talker hidden dimension.
    public let hidden: Int
    /// Feed-forward intermediate dimension.
    public let intermediate: Int
    /// Transformer block count.
    public let nLayers: Int
    /// Attention head count.
    public let nHeads: Int
    /// Key/value head count (GQA).
    public let nKVHeads: Int
    /// Per-head dimension.
    public let headDim: Int
    /// RMSNorm epsilon.
    public let rmsNormEps: Float
    /// RoPE theta base.
    public let ropeTheta: Float
    /// 3D mRoPE section split `[t, h, w]` — absent for a plain 1D RoPE.
    public let mropeSection: [Int]?
    /// Text-side hidden dim (the talker projects text embeddings from
    /// this into `hidden`).
    public let textHidden: Int
    /// Text-side vocabulary size.
    public let textVocabSize: Int
    /// Codec codes per group the code predictor expands each step into.
    public let numCodeGroups: Int

    public init(
        vocabSize: Int, hidden: Int, intermediate: Int,
        nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        rmsNormEps: Float, ropeTheta: Float, mropeSection: [Int]?,
        textHidden: Int, textVocabSize: Int, numCodeGroups: Int
    ) {
        self.vocabSize = vocabSize
        self.hidden = hidden
        self.intermediate = intermediate
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.mropeSection = mropeSection
        self.textHidden = textHidden
        self.textVocabSize = textVocabSize
        self.numCodeGroups = numCodeGroups
    }

    /// Decode the `talker_config` block of a Qwen3TTS `config.json`.
    public static func from(_ talker: [String: Any]) -> Qwen3TTSTalkerConfig? {
        func i(_ k: String, _ d: Int) -> Int {
            if let v = talker[k] as? Int { return v }
            if let v = talker[k] as? Double { return Int(v) }
            return d
        }
        func f(_ k: String, _ d: Float) -> Float {
            if let v = talker[k] as? Double { return Float(v) }
            if let v = talker[k] as? Int { return Float(v) }
            return d
        }
        // The mRoPE section lives under `rope_scaling.mrope_section`.
        var mrope: [Int]? = nil
        if let rs = talker["rope_scaling"] as? [String: Any],
            let sec = rs["mrope_section"] as? [Int]
        {
            mrope = sec
        }
        return Qwen3TTSTalkerConfig(
            vocabSize: i("vocab_size", 3072),
            hidden: i("hidden_size", 1024),
            intermediate: i("intermediate_size", 3072),
            nLayers: i("num_hidden_layers", 28),
            nHeads: i("num_attention_heads", 16),
            nKVHeads: i("num_key_value_heads", 8),
            headDim: i("head_dim", 128),
            rmsNormEps: f("rms_norm_eps", 1e-6),
            ropeTheta: f("rope_theta", 1_000_000),
            mropeSection: mrope,
            textHidden: i("text_hidden_size", 2048),
            textVocabSize: i("text_vocab_size", 151_936),
            numCodeGroups: i("num_code_groups", 16))
    }
}

/// Qwen3TTS model hyper-parameters, decoded from `config.json`.
public struct Qwen3TTSConfig: Sendable {
    /// The talker transformer config.
    public let talker: Qwen3TTSTalkerConfig
    /// Output waveform sample rate (24 kHz).
    public let sampleRate: Int
    /// Codec EOS token id — the talker emits it to end an utterance.
    public let codecEosTokenId: Int
    /// TTS BOS / EOS / PAD token ids in the text vocabulary.
    public let ttsBosTokenId: Int
    public let ttsEosTokenId: Int
    public let ttsPadTokenId: Int

    public init(
        talker: Qwen3TTSTalkerConfig, sampleRate: Int,
        codecEosTokenId: Int, ttsBosTokenId: Int,
        ttsEosTokenId: Int, ttsPadTokenId: Int
    ) {
        self.talker = talker
        self.sampleRate = sampleRate
        self.codecEosTokenId = codecEosTokenId
        self.ttsBosTokenId = ttsBosTokenId
        self.ttsEosTokenId = ttsEosTokenId
        self.ttsPadTokenId = ttsPadTokenId
    }

    /// Decode a Qwen3TTS `config.json`. The talker is nested under
    /// `talker_config`; the codec under `tokenizer_config`.
    public static func from(_ config: ModelConfig) -> Qwen3TTSConfig? {
        guard let talkerBlock = config.nested("talker_config"),
            let talker = Qwen3TTSTalkerConfig.from(talkerBlock)
        else { return nil }
        // Codec EOS lives on the talker block; tts BOS/EOS/PAD top-level.
        let codecEos = (talkerBlock["codec_eos_token_id"] as? Int) ?? 2150
        return Qwen3TTSConfig(
            talker: talker,
            sampleRate: config.int("sample_rate") ?? 24_000,
            codecEosTokenId: codecEos,
            ttsBosTokenId: config.int("tts_bos_token_id") ?? 151_672,
            ttsEosTokenId: config.int("tts_eos_token_id") ?? 151_673,
            ttsPadTokenId: config.int("tts_pad_token_id") ?? 151_671)
    }
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum Qwen3TTSError: Error, CustomStringConvertible {
    /// The synthesis path (talker + code predictor + speaker encoder +
    /// codec) is not wired in this build — Qwen3TTS is a staged port and
    /// this file is stage 1 (config + detection). See the scope note at
    /// the top of `Qwen3TTS.swift`.
    case synthesisNotWired
    case missingConfig

    public var description: String {
        switch self {
        case .synthesisNotWired:
            return "Qwen3TTS: synthesis is not wired in this build. "
                + "Qwen3TTS is a staged port — this build ships stage 1 "
                + "(config decoding + family detection). The talker "
                + "transformer (Qwen3 stack + 3D mRoPE), the code "
                + "predictor, the ECAPA speaker encoder and the "
                + "speech-tokenizer codec are follow-on stages."
        case .missingConfig:
            return "Qwen3TTS: config.json has no talker_config — not a "
                + "Qwen3TTS checkpoint"
        }
    }
}

// ─── Model ───────────────────────────────────────────────────────────

/// A loaded Qwen3TTS model. Stage 1 carries the decoded config and the
/// raw weight bundle; the talker / code-predictor / speaker-encoder /
/// codec sub-models are wired in follow-on stages (see the scope note).
public final class Qwen3TTSModel: @unchecked Sendable {
    public let config: Qwen3TTSConfig
    /// The checkpoint weight bundle — retained so the follow-on stages
    /// can build their sub-models without re-reading the snapshot.
    public let weights: SafeTensorsBundle

    public init(config: Qwen3TTSConfig, weights: SafeTensorsBundle) {
        self.config = config
        self.weights = weights
    }

    public var sampleRate: Int { config.sampleRate }

    /// Full text→waveform synthesis. Throws `Qwen3TTSError.synthesisNotWired`
    /// until the talker / code-predictor / codec stages land — see the
    /// scope note at the top of this file.
    public func synthesize(
        text: String, voice: String? = nil,
        device: Device = .shared
    ) throws -> Tensor {
        _ = (text, voice, device)
        throw Qwen3TTSError.synthesisNotWired
    }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension Qwen3TTSModel {
    public static let modelTypes: Set<String> = ["qwen3_tts", "qwen3tts"]
    public static let architectures: Set<String> = [
        "Qwen3TTSForConditionalGeneration"
    ]

    /// Whether a decoded `config.json` describes a Qwen3TTS checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        // Structural fallback — a nested talker_config block is the
        // distinguishing Qwen3TTS marker.
        return config.has("talker_config")
            && config.has("speaker_encoder_config")
    }

    /// Load a Qwen3TTS checkpoint from a resolved snapshot directory.
    /// Stage 1 decodes the config and retains the weight bundle; the
    /// sub-models are built in follow-on stages.
    public static func load(directory: URL, device: Device = .shared)
        throws -> Qwen3TTSModel
    {
        let config = try ModelConfig.load(from: directory)
        guard let qc = Qwen3TTSConfig.from(config) else {
            throw Qwen3TTSError.missingConfig
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return Qwen3TTSModel(config: qc, weights: bundle)
    }
}
