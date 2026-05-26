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
// LlamaTTS — Orpheus-style text-to-speech on a Llama 3.x backbone.
//
// Orpheus-TTS is a plain Llama causal decoder whose vocabulary is
// extended with audio-codec tokens. Synthesis is autoregressive text
// generation that emits SNAC neural-codec codes; a SNAC decoder turns
// those codes back into a 24 kHz waveform:
//
//   text ──tokenizer──▶ [SOH] prompt [EOT][EOH] ──Llama backbone──▶
//        ──autoregressive decode──▶ audio-codec tokens ──┐
//        ──parse + de-offset──▶ SNAC codes ──SNAC decoder──▶ waveform
//
// ## Scope note
//
// FFAI's contribution here is the **acoustic LLM**: the Llama backbone
// reuses FFAI's existing `LlamaModel` engine (the same dense-transformer
// code path Llama 3.x text models run on), plus the Orpheus token
// protocol — prompt framing, the autoregressive code-token decode loop,
// and SNAC code extraction.
//
// The **SNAC neural codec** (the waveform-synthesis tail) is a separate
// codec port. When a SNAC decoder is wired, `synthesize` produces a
// waveform end-to-end; until then `generateCodes` is the supported
// entry point — it returns the de-interleaved SNAC codes a caller feeds
// to an external SNAC decoder. `synthesize` reports the codec as
// unavailable when no decoder is set. See `LlamaTTSError`.

import Foundation
import Metal
import Tokenizers

// ─── Orpheus special tokens ──────────────────────────────────────────

/// Orpheus tokenizer special-token ids. These sit at the top of the
/// extended Llama vocabulary and frame the prompt / audio stream.
public enum OrpheusTokens {
    public static let startOfHuman = 128_259
    public static let endOfHuman = 128_260
    public static let endOfText = 128_009
    public static let startOfSpeech = 128_257
    public static let endOfSpeech = 128_258
    public static let padToken = 128_263
    public static let audioStart = 128_261
    public static let audioEnd = 128_262
    /// Audio-codec tokens start at this id; subtracting it yields the
    /// raw SNAC code value.
    public static let audioTokenOffset = 128_266
    /// SNAC per-codebook stride — the codec packs 7 codes per group and
    /// offsets each by a multiple of this value.
    public static let snacCodebookStride = 4096
}

// ─── Configuration ───────────────────────────────────────────────────

/// LlamaTTS hyper-parameters. The transformer fields are the standard
/// Llama config; `sampleRate` is the TTS-specific addition.
public struct LlamaTTSConfig: Sendable {
    /// Output waveform sample rate (24 kHz for Orpheus).
    public let sampleRate: Int

    public init(sampleRate: Int = 24_000) {
        self.sampleRate = sampleRate
    }

    /// Build from a decoded `config.json`.
    public static func from(_ config: ModelConfig) -> LlamaTTSConfig {
        LlamaTTSConfig(sampleRate: config.int("sample_rate") ?? 24_000)
    }
}

// ─── SNAC decoder boundary ───────────────────────────────────────────

/// The waveform-synthesis tail. A SNAC neural-codec decoder consumes
/// the three de-interleaved code planes Orpheus emits and returns a
/// 24 kHz waveform. The concrete codec is a separate port; LlamaTTS
/// accepts any conforming decoder so the acoustic LLM and the codec can
/// land independently.
public protocol SNACDecoding: Sendable {
    /// Decode three SNAC code planes into a `[outLen]` waveform.
    /// `codes` is `[layer1, layer2, layer3]` where layer2 is 2× and
    /// layer3 is 4× the length of layer1 (the SNAC up-sampling factors).
    func decode(codes: [[Int]], device: Device) -> Tensor
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum LlamaTTSError: Error, CustomStringConvertible {
    /// No SNAC decoder wired — `synthesize` needs the codec tail.
    /// `generateCodes` works regardless.
    case codecUnavailable
    /// Decode produced no usable audio-code tokens.
    case noAudioCodes
    case missingConfig

    public var description: String {
        switch self {
        case .codecUnavailable:
            return "LlamaTTS: no SNAC decoder is wired in this build; "
                + "use generateCodes to obtain SNAC codes and decode "
                + "them with an external SNAC codec"
        case .noAudioCodes:
            return "LlamaTTS: generation produced no audio-code tokens"
        case .missingConfig:
            return "LlamaTTS: required config field missing"
        }
    }
}

// ─── Model ───────────────────────────────────────────────────────────

/// A loaded Orpheus-style TTS model. Owns the Llama acoustic backbone
/// (FFAI's `LlamaModel` engine) and, when wired, a SNAC codec decoder.
public final class LlamaTTSModel: @unchecked Sendable {
    public let config: LlamaTTSConfig
    /// The Llama acoustic backbone — a standard FFAI dense transformer.
    public let backbone: LlamaModel
    /// The text tokenizer (extended Llama vocabulary).
    public let tokenizer: any Tokenizer
    /// The SNAC codec decoder — `nil` until the codec port lands.
    public var snacDecoder: (any SNACDecoding)?

    public init(
        config: LlamaTTSConfig, backbone: LlamaModel,
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

    /// Build the Orpheus input-token sequence for a prompt:
    /// `[SOH] (voice:) prompt [EOT][EOH]`. The `voice` prefix selects a
    /// speaker when the checkpoint was trained with named voices.
    public func promptTokens(text: String, voice: String? = nil) -> [Int] {
        let body = voice.map { "\($0): \(text)" } ?? text
        var ids = [OrpheusTokens.startOfHuman]
        ids.append(contentsOf: tokenizer.encode(text: body))
        ids.append(OrpheusTokens.endOfText)
        ids.append(OrpheusTokens.endOfHuman)
        return ids
    }

    // ─── Code extraction ─────────────────────────────────────────────

    /// Turn a flat run of generated audio-code tokens into the three
    /// de-interleaved SNAC code planes. Orpheus emits 7 codes per frame
    /// with a fixed interleave + per-position offset; this reverses it.
    /// Mirrors the reference `redistribute_codes` in Orpheus-TTS.
    public static func deinterleaveSNACCodes(_ tokens: [Int]) -> [[Int]] {
        let stride = OrpheusTokens.snacCodebookStride
        // Drop tokens that aren't a multiple of 7 — a partial frame.
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

    /// Autoregressively decode SNAC codes for a prompt. Runs the Llama
    /// backbone token-by-token, collecting audio-code tokens between the
    /// start- and end-of-speech markers, then de-interleaves them into
    /// the three SNAC planes.
    ///
    /// `maxFrames` caps generation; each SNAC frame is 7 code tokens.
    public func generateCodes(
        text: String, voice: String? = nil,
        maxFrames: Int = 1200,
        temperature: Float = 0.6,
        seed: UInt64 = 0,
        device: Device = .shared
    ) throws -> [[Int]] {
        let prompt = promptTokens(text: text, voice: voice)
        let caches = backbone.makeLayerCaches(device: device)

        // Prefill — feed every prompt token; keep the last logits.
        var position = 0
        for tok in prompt.dropLast() {
            _ = backbone.forward(
                tokenId: tok, position: position,
                caches: caches, device: device)
            position += 1
        }
        var nextInput = prompt.last ?? OrpheusTokens.endOfHuman

        // Decode loop — collect audio-code tokens.
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

            if token == OrpheusTokens.startOfSpeech {
                inSpeech = true
                continue
            }
            if token == OrpheusTokens.endOfSpeech { break }
            if inSpeech, token >= OrpheusTokens.audioTokenOffset {
                codeTokens.append(token - OrpheusTokens.audioTokenOffset)
            }
            if codeTokens.count >= maxTokens { break }
        }

        guard !codeTokens.isEmpty else { throw LlamaTTSError.noAudioCodes }
        return LlamaTTSModel.deinterleaveSNACCodes(codeTokens)
    }

    /// Full text→waveform synthesis. Requires a SNAC decoder; throws
    /// `LlamaTTSError.codecUnavailable` when one is not wired (see the
    /// scope note at the top of this file).
    public func synthesize(
        text: String, voice: String? = nil,
        maxFrames: Int = 1200,
        temperature: Float = 0.6,
        seed: UInt64 = 0,
        device: Device = .shared
    ) throws -> Tensor {
        guard let decoder = snacDecoder else {
            throw LlamaTTSError.codecUnavailable
        }
        let codes = try generateCodes(
            text: text, voice: voice,
            maxFrames: maxFrames,
            temperature: temperature,
            seed: seed, device: device)
        return decoder.decode(codes: codes, device: device)
    }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension LlamaTTSModel {
    public static let modelTypes: Set<String> = ["llama_tts", "orpheus"]
    public static let architectures: Set<String> = [
        "OrpheusForConditionalGeneration"
    ]

    /// Whether a decoded `config.json` describes a LlamaTTS / Orpheus
    /// checkpoint. Orpheus checkpoints commonly carry a plain
    /// `LlamaForCausalLM` architecture, so structural detection looks
    /// for the TTS-extended vocabulary plus a `sample_rate` hint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        // Structural fallback: a Llama causal LM whose vocabulary is
        // large enough to hold the Orpheus audio-codec tokens and which
        // declares a TTS sample rate.
        let isLlama =
            (config.modelType == "llama")
            || (config.architecture == "LlamaForCausalLM")
        let hasAudioVocab =
            (config.vocabSize ?? 0)
            > OrpheusTokens.audioTokenOffset
        return isLlama && hasAudioVocab && config.has("sample_rate")
    }

    /// Load a LlamaTTS checkpoint from a resolved snapshot directory.
    /// The Llama backbone reuses FFAI's `LlamaDense` loader; the SNAC
    /// codec decoder is left unset (separate port — see scope note).
    public static func load(
        directory: URL,
        options: LoadOptions = LoadOptions(),
        device: Device = .shared
    ) async throws
        -> LlamaTTSModel
    {
        let config = try ModelConfig.load(from: directory)
        let ttsConfig = LlamaTTSConfig.from(config)
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        let backbone = try LlamaDense.loadModel(
            config: config, weights: bundle, options: options, device: device)
        let tokenizer = try await TokenizerLoader().load(from: directory)
        return LlamaTTSModel(
            config: ttsConfig, backbone: backbone,
            tokenizer: tokenizer)
    }
}
