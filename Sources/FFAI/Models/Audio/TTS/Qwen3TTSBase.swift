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
// Qwen3TTSBase — Qwen3 LLM backbone used as a plain TTS engine.
//
// This is the "VyvoTTS" / base Qwen3 TTS pattern: a standard Qwen3
// causal-language model with an extended vocabulary that maps audio
// codec tokens onto the same token embedding table. The backbone is
// identical to FFAI's `Qwen3Model` (same Qwen3 dense transformer: SwiGLU
// MLP, q_norm/k_norm per head, 1D RoPE). The acoustic extension adds:
//
//   • Seven special-token ids appended after the Qwen3 base vocab
//     (`startOfSpeech`, `endOfSpeech`, `startOfHuman`, `endOfHuman`, …)
//   • Audio-codec tokens at `audioTokensStart` (151 679+) for the SNAC
//     24 kHz neural codec
//   • A chat-style prompt frame (`[SOH] text [EOT][EOH]`) consumed by
//     the autoregressive SNAC-code decode loop
//   • A SNAC decoder tail that turns de-interleaved code planes into a
//     24 kHz waveform (separate port — `SNACDecoding` defined in this file)
//
// ## Distinction from Qwen3TTS (the 12Hz variant)
//
// `Qwen3TTS.swift` is the Flash / 12Hz family: a four-part architecture
// with a dedicated `talker_config` nested block, a code predictor, an
// ECAPA-TDNN speaker encoder, and a speech-tokenizer codec. Its
// `config.json` is detected by the presence of `talker_config` +
// `speaker_encoder_config`.
//
// `Qwen3TTSBase` is detected differently: it is a plain `Qwen3ForCausalLM`
// with an extended vocabulary that exceeds the base Qwen3 vocab size
// (151 936) and declares a `sample_rate` — the same structural marker
// LlamaTTS uses for Orpheus checkpoints, adapted for the Qwen3 vocab
// layout. If a config has `talker_config` it is Qwen3TTS (Flash), not
// Qwen3TTSBase.
//
// ## Checkpoint
//
// Targets: `mlx-community/VyvoTTS-EN-Beta-4bit` (primary, quantized) and
// any other Qwen3-backbone TTS checkpoint whose config lacks `talker_config`
// but has an extended vocab and `sample_rate`.
//
// ## Scope note — STAGED PORT
//
// This file is stage 1: config decode + family detection + registry
// plumbing + the generate-codes loop. The SNAC neural-codec decoder
// (waveform synthesis) is a separate port; until it lands, `synthesize`
// throws `Qwen3TTSBaseError.codecUnavailable`. `generateCodes` is the
// supported entry point and returns the de-interleaved SNAC code planes.
// The Orpheus (Llama-backbone) pattern is the peer model LlamaTTS.swift
// (not present in this build stage).

import Foundation
import Metal
import Tokenizers

// ─── SNAC codec decoder boundary ────────────────────────────────────

/// The waveform-synthesis tail for LLM-backbone TTS families. A SNAC
/// neural-codec decoder consumes the three de-interleaved code planes and
/// returns a 24 kHz waveform. The concrete codec is a separate port; TTS
/// models accept any conforming decoder so the acoustic LLM and the codec
/// can land independently.
///
/// The `SNACDecoding` protocol itself lives in `LlamaTTS.swift` (the
/// canonical Orpheus TTS family that shipped first). Qwen3TTSBase reuses
/// it as-is — same three-layer up-sampling, same decode signature.

// ─── VyvoTTS / Qwen3-base special tokens ────────────────────────────

/// Special-token ids for the VyvoTTS / Qwen3 base TTS protocol.
/// These are appended immediately after the Qwen3 base vocabulary
/// (151 669 tokens) and frame the prompt / audio stream in the same
/// spirit as Orpheus's special tokens on the Llama backbone.
public enum Qwen3TTSBaseTokens {
    /// Length of the base Qwen3 tokenizer vocabulary.
    public static let baseVocabSize = 151_669

    // Special tokens beyond the base vocab:
    public static let startOfText = 151_643  // <|im_start|> (reused from base)
    public static let endOfText = 151_645  // <|im_end|>  (reused from base)
    public static let startOfSpeech = baseVocabSize + 1  // 151 670
    public static let endOfSpeech = baseVocabSize + 2  // 151 671
    public static let startOfHuman = baseVocabSize + 3  // 151 672
    public static let endOfHuman = baseVocabSize + 4  // 151 673
    public static let startOfAI = baseVocabSize + 5  // 151 674
    public static let endOfAI = baseVocabSize + 6  // 151 675
    public static let padToken = baseVocabSize + 7  // 151 676

    /// Audio-codec tokens start here. Each token minus this offset gives
    /// the raw SNAC code value (same stride/offset layout as Orpheus).
    public static let audioTokensStart = baseVocabSize + 10  // 151 679

    /// SNAC encodes 7 codes per frame with a per-position offset that is
    /// a multiple of this stride (mirrors the Orpheus/LlamaTTS layout).
    public static let snacCodebookStride = 4_096
}

// ─── Configuration ───────────────────────────────────────────────────

/// Qwen3TTSBase hyper-parameters decoded from `config.json`.
/// The transformer fields are identical to the standard Qwen3 config;
/// `sampleRate` is the TTS-specific addition.
public struct Qwen3TTSBaseConfig: Sendable {
    public let vocabSize: Int
    public let hidden: Int
    public let intermediate: Int
    public let nLayers: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let tieWordEmbeddings: Bool
    /// Output waveform sample rate (24 kHz for the SNAC codec).
    public let sampleRate: Int

    public init(
        vocabSize: Int, hidden: Int, intermediate: Int,
        nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        rmsNormEps: Float, ropeTheta: Float,
        maxPositionEmbeddings: Int, tieWordEmbeddings: Bool,
        sampleRate: Int
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
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.tieWordEmbeddings = tieWordEmbeddings
        self.sampleRate = sampleRate
    }

    /// Decode from a `config.json` ModelConfig.
    public static func from(_ config: ModelConfig) -> Qwen3TTSBaseConfig? {
        guard let vocab = config.vocabSize,
            let hidden = config.hiddenSize,
            let inter = config.intermediateSize,
            let nLayers = config.numLayers,
            let nHeads = config.numAttentionHeads,
            let headDim = config.headDim,
            let eps = config.rmsNormEps
        else { return nil }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let theta = Float(config.ropeTheta ?? 1_000_000)
        let maxPos = config.int("max_position_embeddings") ?? 32_768
        let tieEmbed = config.tieWordEmbeddings
        let rate = config.int("sample_rate") ?? 24_000
        return Qwen3TTSBaseConfig(
            vocabSize: vocab,
            hidden: hidden, intermediate: inter,
            nLayers: nLayers, nHeads: nHeads, nKVHeads: nKVHeads,
            headDim: headDim,
            rmsNormEps: Float(eps),
            ropeTheta: theta,
            maxPositionEmbeddings: maxPos,
            tieWordEmbeddings: tieEmbed,
            sampleRate: rate
        )
    }
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum Qwen3TTSBaseError: Error, CustomStringConvertible {
    /// The SNAC codec decoder has not been wired — `synthesize` needs it.
    /// Use `generateCodes` to obtain the three SNAC code planes and feed
    /// them to an external SNAC decoder.
    case codecUnavailable
    /// Generation produced no audio-code tokens (prompt too short,
    /// budget too tight, or the model did not reach `endOfSpeech`).
    case noAudioCodes
    /// The config.json is missing required Qwen3 fields.
    case missingConfig

    public var description: String {
        switch self {
        case .codecUnavailable:
            return "Qwen3TTSBase: no SNAC decoder is wired; "
                + "use generateCodes to obtain SNAC code planes "
                + "and decode them with an external SNAC codec"
        case .noAudioCodes:
            return "Qwen3TTSBase: generation produced no audio-code tokens"
        case .missingConfig:
            return "Qwen3TTSBase: config.json is missing required Qwen3 fields"
        }
    }
}

// ─── Model ───────────────────────────────────────────────────────────

/// A loaded Qwen3TTSBase model. Owns the Qwen3 acoustic backbone
/// (FFAI's `Qwen3Model` engine) plus the VyvoTTS token protocol.
///
/// The SNAC neural-codec decoder (the waveform-synthesis tail) is a
/// separate port; set `snacDecoder` to enable end-to-end `synthesize`.
/// Until then, `generateCodes` is the supported entry point.
public final class Qwen3TTSBaseModel: @unchecked Sendable {
    public let config: Qwen3TTSBaseConfig
    /// The Qwen3 acoustic backbone — a standard FFAI dense transformer.
    public let backbone: Qwen3Model
    /// The text tokenizer (Qwen3 extended vocabulary).
    public let tokenizer: any Tokenizer
    /// The SNAC codec decoder — `nil` until the codec port lands.
    /// Wire a conforming decoder to enable `synthesize`.
    public var snacDecoder: (any SNACDecoding)?

    public init(
        config: Qwen3TTSBaseConfig,
        backbone: Qwen3Model,
        tokenizer: any Tokenizer,
        snacDecoder: (any SNACDecoding)? = nil
    ) {
        self.config = config
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.snacDecoder = snacDecoder
    }

    public var sampleRate: Int { config.sampleRate }

    // ─── Prompt framing ──────────────────────────────────────────────

    /// Build the VyvoTTS input-token sequence for a prompt.
    ///
    /// Frame: `[SOH] (voice: ) prompt [EOT][EOH]`
    ///
    /// The optional `voice` prefix selects a speaker when the checkpoint
    /// was trained with named voices (e.g. `"en-us-1"`).
    public func promptTokens(text: String, voice: String? = nil) -> [Int] {
        let body = voice.map { "\($0): \(text)" } ?? text
        var ids = [Qwen3TTSBaseTokens.startOfHuman]
        ids.append(contentsOf: tokenizer.encode(text: body))
        ids.append(Qwen3TTSBaseTokens.endOfText)
        ids.append(Qwen3TTSBaseTokens.endOfHuman)
        return ids
    }

    // ─── Code extraction ─────────────────────────────────────────────

    /// Turn a flat run of audio-code tokens into the three de-interleaved
    /// SNAC code planes. The VyvoTTS / Qwen3-base layout is identical to
    /// the Orpheus (LlamaTTS) layout: 7 codes per frame, each position
    /// offset by a multiple of `snacCodebookStride`.
    ///
    /// Mirrors `LlamaTTSModel.deinterleaveSNACCodes` for the Llama
    /// backbone. A partial trailing frame (non-multiple-of-7) is dropped.
    public static func deinterleaveSNACCodes(_ tokens: [Int]) -> [[Int]] {
        let stride = Qwen3TTSBaseTokens.snacCodebookStride
        let frameSize = 7
        let usable = (tokens.count / frameSize) * frameSize
        var layer1: [Int] = []
        var layer2: [Int] = []
        var layer3: [Int] = []
        let groups = usable / frameSize
        for g in 0 ..< groups {
            let base = frameSize * g
            layer1.append(tokens[base])
            layer2.append(tokens[base + 1] - stride)
            layer3.append(tokens[base + 2] - 2 * stride)
            layer3.append(tokens[base + 3] - 3 * stride)
            layer2.append(tokens[base + 4] - 4 * stride)
            layer3.append(tokens[base + 5] - 5 * stride)
            layer3.append(tokens[base + 6] - 6 * stride)
        }
        return [layer1, layer2, layer3]
    }

    // ─── Generation ──────────────────────────────────────────────────

    /// Autoregressively decode SNAC codes for a prompt. Runs the Qwen3
    /// backbone token-by-token, collecting audio-code tokens between
    /// the `startOfSpeech` and `endOfSpeech` markers, then de-interleaves
    /// them into the three SNAC planes.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize.
    ///   - voice: Optional speaker prefix (e.g. `"en-us-1"`).
    ///   - maxFrames: Budget in SNAC frames (each frame = 7 code tokens).
    ///   - temperature: Sampling temperature; 0 = greedy argmax.
    ///   - seed: RNG seed for reproducible sampling.
    ///   - device: Metal device to run on.
    /// - Returns: Three de-interleaved SNAC code planes `[layer1, layer2, layer3]`.
    ///   layer2 is 2× and layer3 is 4× the length of layer1.
    /// - Throws: `Qwen3TTSBaseError.noAudioCodes` when no codes were emitted.
    public func generateCodes(
        text: String,
        voice: String? = nil,
        maxFrames: Int = 1200,
        temperature: Float = 0.6,
        seed: UInt64 = 0,
        device: Device = .shared
    ) throws -> [[Int]] {
        let prompt = promptTokens(text: text, voice: voice)
        let caches = backbone.makeLayerCaches(device: device)

        // Prefill — feed every prompt token except the last; keep logits
        // for the last token in the decode loop.
        var position = 0
        for tok in prompt.dropLast() {
            _ = backbone.forward(
                tokenId: tok, position: position,
                caches: caches, device: device)
            position += 1
        }
        var nextInput = prompt.last ?? Qwen3TTSBaseTokens.endOfHuman

        // Decode loop — collect audio-code tokens after startOfSpeech.
        var rng = SeededRandomNumberGenerator(seed: seed)
        var codeTokens: [Int] = []
        let maxTokens = maxFrames * 7
        var inSpeech = false

        for _ in 0 ..< (maxTokens + prompt.count) {
            let token: Int
            if temperature > 0 {
                let draw = Float.random(in: 0 ..< 1, using: &rng)
                token = backbone.forwardSampleCategorical(
                    tokenId: nextInput, position: position, caches: caches,
                    temperature: temperature, uniformDraw: draw,
                    device: device)
            } else {
                token = backbone.forwardSample(
                    tokenId: nextInput, position: position,
                    caches: caches, device: device)
            }
            position += 1
            nextInput = token

            if token == Qwen3TTSBaseTokens.startOfSpeech {
                inSpeech = true
                continue
            }
            if token == Qwen3TTSBaseTokens.endOfSpeech { break }
            if inSpeech, token >= Qwen3TTSBaseTokens.audioTokensStart {
                codeTokens.append(token - Qwen3TTSBaseTokens.audioTokensStart)
            }
            if codeTokens.count >= maxTokens { break }
        }

        guard !codeTokens.isEmpty else { throw Qwen3TTSBaseError.noAudioCodes }
        return Qwen3TTSBaseModel.deinterleaveSNACCodes(codeTokens)
    }

    /// Full text→waveform synthesis. Requires a SNAC decoder; throws
    /// `Qwen3TTSBaseError.codecUnavailable` when none is wired.
    ///
    /// The SNAC decoder port is separate from this file. Wire one by
    /// setting `snacDecoder` on the loaded model.
    public func synthesize(
        text: String,
        voice: String? = nil,
        maxFrames: Int = 1200,
        temperature: Float = 0.6,
        seed: UInt64 = 0,
        device: Device = .shared
    ) throws -> Tensor {
        guard let decoder = snacDecoder else {
            throw Qwen3TTSBaseError.codecUnavailable
        }
        let codes = try generateCodes(
            text: text, voice: voice,
            maxFrames: maxFrames, temperature: temperature,
            seed: seed, device: device)
        return decoder.decode(codes: codes, device: device)
    }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension Qwen3TTSBaseModel {
    /// Model type strings this family handles. A plain `"qwen3"` with
    /// TTS-extended vocabulary — no Flash / talker_config needed.
    public static let modelTypes: Set<String> = ["qwen3_tts_base", "qwen3base_tts"]
    /// Architecture strings this family handles.
    public static let architectures: Set<String> = [
        "Qwen3TTSBaseForConditionalGeneration"
    ]

    /// Whether a decoded `config.json` describes a Qwen3TTSBase checkpoint.
    ///
    /// Detection precedence (called after `Qwen3TTSModel.handles` has been
    /// ruled out, so we don't need to repeat its guard):
    ///
    /// 1. Explicit `model_type` / `architecture` match — highest priority.
    /// 2. Structural fallback: a Qwen3 architecture (`Qwen3ForCausalLM`
    ///    or `model_type == "qwen3"`) whose vocabulary exceeds the base
    ///    Qwen3 size (151 936) — the codec-token extension is the distinctive
    ///    structural marker. (VyvoTTS-EN-Beta-4bit ships without a
    ///    `sample_rate` field, so we no longer require it; the loader
    ///    defaults to 24 kHz when missing.)
    ///
    /// Checkpoints that carry `talker_config` + `speaker_encoder_config`
    /// are NOT matched here — those belong to `Qwen3TTSModel` (the 12Hz
    /// Flash family).
    public static func handles(_ config: ModelConfig) -> Bool {
        // Explicit model_type / architecture (canonical TTS-Base name):
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        // Structural guard: must NOT be the Flash variant.
        if config.has("talker_config") && config.has("speaker_encoder_config") {
            return false
        }
        // Structural fallback: a Qwen3 backbone (model_type "qwen3" OR
        // architecture "Qwen3ForCausalLM") whose vocabulary exceeds the
        // base Qwen3 size (151 936). VyvoTTS-EN-Beta-4bit ships
        // architectures=["Qwen3ForCausalLM"] + model_type="qwen3" +
        // vocab_size=180 352 — the codec-token extension is the
        // distinguishing structural marker. The vocab guard keeps plain
        // Qwen3 text checkpoints out of the audio routing path.
        let isQwen3 =
            (config.modelType == "qwen3")
            || (config.architecture == "Qwen3ForCausalLM")
        let hasAudioVocab = (config.vocabSize ?? 0) > 151_936
        return isQwen3 && hasAudioVocab
    }

    /// Load a Qwen3TTSBase checkpoint from a resolved snapshot directory.
    ///
    /// The Qwen3 backbone reuses FFAI's `Qwen3Dense` loader (same weight
    /// layout as the text model). The SNAC decoder is left unset until
    /// the codec port lands.
    public static func load(
        directory: URL,
        options: LoadOptions = LoadOptions(),
        device: Device = .shared
    ) async throws -> Qwen3TTSBaseModel {
        let config = try ModelConfig.load(from: directory)
        guard let ttsConfig = Qwen3TTSBaseConfig.from(config) else {
            throw Qwen3TTSBaseError.missingConfig
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        // Reuse the Qwen3Dense backbone loader — the weight layout is
        // identical for the text model and the TTS extended-vocab model.
        let backbone = try Qwen3Dense.loadModel(
            config: config, weights: bundle,
            options: options, device: device)
        let tokenizer = try await TokenizerLoader().load(from: directory)
        return Qwen3TTSBaseModel(
            config: ttsConfig, backbone: backbone,
            tokenizer: tokenizer)
    }
}
