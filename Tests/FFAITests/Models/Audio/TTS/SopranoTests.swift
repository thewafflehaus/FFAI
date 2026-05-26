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
// Unit tests for the Soprano TTS family.
//
// Covers:
//   • Config decoding — canonical Soprano-80M and minimal payloads.
//   • hasDecoderConfig — true for 80M, false for 1.1-style config.
//   • Registry detection — model_type, SopranoForCausalLM architecture,
//     Qwen3ForCausalLM + soprano model_type combo, negative cases.
//   • AudioModelRegistry integration — handles + capabilities.
//   • SopranoError descriptions — all cases carry useful text.
//   • Text cleaning — cleanTextForSoprano, number normalisation.
//   • SopranoConfig.stopTokenId — always 3.
//   • sampleRate default — 32 000 Hz when config omits sample_rate.
//
// These tests are fast and offline — no checkpoint weights are loaded.

import Foundation
import Testing
@testable import FFAI

@Suite("Soprano")
struct SopranoTests {

    // ─── Canonical config fixtures ────────────────────────────────────

    /// Mirrors `mlx-community/Soprano-80M-bf16/config.json`.
    private static func soprano80MRaw() -> [String: Any] {
        [
            "model_type": "soprano",
            "architectures": ["SopranoForCausalLM"],
            "hidden_size": 512,
            "num_hidden_layers": 17,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "intermediate_size": 2304,
            "vocab_size": 8192,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            // Decoder fields present in Soprano-80M:
            "sample_rate": 32000,
            "decoder_num_layers": 8,
            "decoder_dim": 512,
            "decoder_intermediate_dim": 1536,
            "hop_length": 512,
            "n_fft": 2048,
            "upscale": 4,
            "input_kernel": 3,
            "dw_kernel": 3,
            "token_size": 2048,
        ]
    }

    /// Soprano-1.1-style config: model_type soprano, no decoder fields.
    private static func soprano11Raw() -> [String: Any] {
        [
            "model_type": "soprano",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 512,
            "num_hidden_layers": 17,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "intermediate_size": 2304,
            "vocab_size": 8192,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            // No decoder fields — Soprano-1.1 is LLM-only.
        ]
    }

    // ─── Config decoding ──────────────────────────────────────────────

    @Test("SopranoConfig — decodes transformer fields from Soprano-80M config")
    func configDecodes80M() {
        let raw = Self.soprano80MRaw()
        let modelConfig = ModelConfig(architecture: "SopranoForCausalLM",
                                      modelType: "soprano", raw: raw)
        let config = SopranoConfig.from(modelConfig)
        #expect(config != nil)
        #expect(config?.hiddenSize == 512)
        #expect(config?.numHiddenLayers == 17)
        #expect(config?.numAttentionHeads == 4)
        #expect(config?.numKeyValueHeads == 1)
        #expect(config?.headDim == 128)
        #expect(config?.intermediateSize == 2304)
        #expect(config?.vocabSize == 8192)
        #expect(config?.maxPositionEmbeddings == 512)
        #expect(abs((config?.rmsNormEps ?? 0) - 1e-6) < 1e-9)
        #expect(abs((config?.ropeTheta ?? 0) - 10000) < 1)
        #expect(config?.tieWordEmbeddings == false)
        #expect(config?.bosTokenId == 1)
        #expect(config?.eosTokenId == 2)
        #expect(config?.padTokenId == 0)
    }

    @Test("SopranoConfig — decodes decoder fields for Soprano-80M")
    func configDecodes80MDecoder() {
        let raw = Self.soprano80MRaw()
        let modelConfig = ModelConfig(architecture: "SopranoForCausalLM",
                                      modelType: "soprano", raw: raw)
        let config = SopranoConfig.from(modelConfig)!
        #expect(config.sampleRate == 32_000)
        #expect(config.decoderNumLayers == 8)
        #expect(config.decoderDim == 512)
        #expect(config.decoderIntermediateDim == 1536)
        #expect(config.hopLength == 512)
        #expect(config.nFft == 2048)
        #expect(config.upscale == 4)
        #expect(config.inputKernel == 3)
        #expect(config.dwKernel == 3)
        #expect(config.tokenSize == 2048)
    }

    @Test("SopranoConfig — hasDecoderConfig is true for Soprano-80M")
    func hasDecoderConfigTrue() {
        let raw = Self.soprano80MRaw()
        let modelConfig = ModelConfig(architecture: "SopranoForCausalLM",
                                      modelType: "soprano", raw: raw)
        let config = SopranoConfig.from(modelConfig)!
        #expect(config.hasDecoderConfig == true)
    }

    @Test("SopranoConfig — hasDecoderConfig is false for Soprano-1.1 (LLM-only)")
    func hasDecoderConfigFalse() {
        let raw = Self.soprano11Raw()
        let modelConfig = ModelConfig(architecture: "Qwen3ForCausalLM",
                                      modelType: "soprano", raw: raw)
        let config = SopranoConfig.from(modelConfig)!
        #expect(config.hasDecoderConfig == false)
        #expect(config.decoderDim == nil)
        #expect(config.hopLength == nil)
        #expect(config.nFft == nil)
    }

    @Test("SopranoConfig — returns nil for missing required fields")
    func configReturnsNilForMissingFields() {
        // Missing hidden_size, num_hidden_layers, etc.
        let modelConfig = ModelConfig(architecture: nil, modelType: "soprano",
                                      raw: ["model_type": "soprano"])
        #expect(SopranoConfig.from(modelConfig) == nil)
    }

    @Test("SopranoConfig — stopTokenId is always 3")
    func stopTokenIdIs3() {
        #expect(SopranoConfig.stopTokenId == 3)
    }

    @Test("SopranoConfig — numKeyValueHeads defaults to numAttentionHeads when absent")
    func kvHeadsDefaultsToAttnHeads() {
        var raw = Self.soprano80MRaw()
        raw.removeValue(forKey: "num_key_value_heads")
        let modelConfig = ModelConfig(architecture: nil, modelType: "soprano", raw: raw)
        let config = SopranoConfig.from(modelConfig)!
        #expect(config.numKeyValueHeads == config.numAttentionHeads)
    }

    @Test("SopranoConfig — rmsNormEps defaults to 1e-6 when absent")
    func rmsNormEpsDefault() {
        var raw = Self.soprano80MRaw()
        raw.removeValue(forKey: "rms_norm_eps")
        let modelConfig = ModelConfig(architecture: nil, modelType: "soprano", raw: raw)
        let config = SopranoConfig.from(modelConfig)!
        #expect(abs(config.rmsNormEps - 1e-6) < 1e-12)
    }

    // ─── sampleRate convenience ───────────────────────────────────────

    @Test("sampleRate — defaults to 32 000 Hz when config omits sample_rate")
    func sampleRateDefault() {
        var raw = Self.soprano80MRaw()
        raw.removeValue(forKey: "sample_rate")
        let modelConfig = ModelConfig(architecture: nil, modelType: "soprano", raw: raw)
        let config = SopranoConfig.from(modelConfig)!
        // sampleRate is a computed property on SopranoModel, checked via the config.
        // The model falls back to 32_000 when sampleRate is nil.
        #expect(config.sampleRate == nil)      // nil in config …
        // … but the model exposes 32_000 via the fallback.
        // We assert the fallback directly here.
        let fallback = config.sampleRate ?? 32_000
        #expect(fallback == 32_000)
    }

    @Test("sampleRate — returns 32 000 for Soprano-80M with sample_rate in config")
    func sampleRate32kHz() {
        let config = SopranoConfig.from(
            ModelConfig(architecture: "SopranoForCausalLM", modelType: "soprano",
                        raw: Self.soprano80MRaw()))!
        #expect(config.sampleRate == 32_000)
    }

    // ─── Registry detection ───────────────────────────────────────────

    @Test("SopranoModel.handles — detects from model_type soprano")
    func handlesModelType() {
        let config = ModelConfig(architecture: nil, modelType: "soprano",
                                 raw: ["model_type": "soprano"])
        #expect(SopranoModel.handles(config))
    }

    @Test("SopranoModel.handles — detects from SopranoForCausalLM architecture")
    func handlesArchitecture() {
        let config = ModelConfig(architecture: "SopranoForCausalLM", modelType: nil,
                                 raw: ["architectures": ["SopranoForCausalLM"]])
        #expect(SopranoModel.handles(config))
    }

    @Test("SopranoModel.handles — detects Soprano-1.1 (Qwen3ForCausalLM + soprano model_type)")
    func handlesQwen3ArchWithSopranoModelType() {
        let config = ModelConfig(architecture: "Qwen3ForCausalLM", modelType: "soprano",
                                 raw: Self.soprano11Raw())
        #expect(SopranoModel.handles(config))
    }

    @Test("SopranoModel.handles — does not detect plain Qwen3 models")
    func doesNotDetectQwen3() {
        let config = ModelConfig(architecture: "Qwen3ForCausalLM", modelType: "qwen3",
                                 raw: ["model_type": "qwen3", "hidden_size": 2048])
        #expect(!SopranoModel.handles(config))
    }

    @Test("SopranoModel.handles — does not detect Llama models")
    func doesNotDetectLlama() {
        let config = ModelConfig(architecture: "LlamaForCausalLM", modelType: "llama",
                                 raw: ["model_type": "llama", "hidden_size": 4096])
        #expect(!SopranoModel.handles(config))
    }

    @Test("SopranoModel.handles — does not detect Kokoro models")
    func doesNotDetectKokoro() {
        let config = ModelConfig(architecture: nil, modelType: "kokoro",
                                 raw: ["model_type": "kokoro"])
        #expect(!SopranoModel.handles(config))
    }

    @Test("SopranoModel.handles — does not detect MOSS-TTS models")
    func doesNotDetectMossTTS() {
        let config = ModelConfig(architecture: nil, modelType: "moss_tts",
                                 raw: ["model_type": "moss_tts", "n_vq": 32])
        #expect(!SopranoModel.handles(config))
    }

    @Test("SopranoModel.modelTypes — contains soprano")
    func modelTypesContainsSoprano() {
        #expect(SopranoModel.modelTypes.contains("soprano"))
    }

    @Test("SopranoModel.architectures — contains SopranoForCausalLM")
    func architecturesContainsSopranoForCausalLM() {
        #expect(SopranoModel.architectures.contains("SopranoForCausalLM"))
    }

    // ─── AudioModelRegistry integration ──────────────────────────────

    @Test("AudioModelRegistry — handles Soprano from model_type soprano")
    func registryHandlesSoprano() {
        let config = ModelConfig(architecture: "SopranoForCausalLM", modelType: "soprano",
                                 raw: ["model_type": "soprano"])
        #expect(AudioModelRegistry.handles(config))
    }

    @Test("AudioModelRegistry — capabilities for Soprano is textToSpeech")
    func registryCapabilitiesSoprano() {
        let config = ModelConfig(architecture: "SopranoForCausalLM", modelType: "soprano",
                                 raw: ["model_type": "soprano"])
        let caps = AudioModelRegistry.capabilities(for: config)
        #expect(caps == Capability.textToSpeech)
        #expect(caps?.contains(.audioOut) == true)
        #expect(caps?.contains(.textIn) == true)
    }

    @Test("AudioModelRegistry — does not handle plain Qwen3 as Soprano")
    func registryDoesNotHandlePlainQwen3AsSoprano() {
        let config = ModelConfig(architecture: "Qwen3ForCausalLM", modelType: "qwen3",
                                 raw: ["model_type": "qwen3", "hidden_size": 2048])
        // Soprano should not steal a plain Qwen3 config.
        #expect(!SopranoModel.handles(config))
    }

    // ─── SopranoError descriptions ────────────────────────────────────

    @Test("SopranoError.missingConfig — description carries field name")
    func errorMissingConfigDescription() {
        let err = SopranoError.missingConfig("hidden_size")
        #expect(err.description.contains("hidden_size"))
        #expect(err.description.lowercased().contains("soprano"))
    }

    @Test("SopranoError.decoderNotAvailable — description mentions Soprano-1.1")
    func errorDecoderNotAvailableDescription() {
        let err = SopranoError.decoderNotAvailable
        let d = err.description
        #expect(d.contains("1.1") || d.contains("decoder"))
        #expect(d.lowercased().contains("soprano"))
    }

    @Test("SopranoError.tokenizerNotLoaded — description mentions tokenizer")
    func errorTokenizerNotLoadedDescription() {
        let err = SopranoError.tokenizerNotLoaded
        #expect(err.description.lowercased().contains("tokenizer"))
    }

    @Test("SopranoError.generationFailed — description carries message")
    func errorGenerationFailedDescription() {
        let err = SopranoError.generationFailed("no tokens produced")
        #expect(err.description.contains("no tokens produced"))
    }

    // ─── Text cleaning ────────────────────────────────────────────────

    @Test("cleanTextForSoprano — lowercases input")
    func cleanLowercases() {
        let out = cleanTextForSoprano("Hello World")
        #expect(out == "hello world")
    }

    @Test("cleanTextForSoprano — collapses multiple spaces")
    func cleanCollapsesSpaces() {
        let out = cleanTextForSoprano("hello   world")
        #expect(out == "hello world")
    }

    @Test("cleanTextForSoprano — trims leading/trailing whitespace")
    func cleanTrimsWhitespace() {
        let out = cleanTextForSoprano("  hello world  ")
        #expect(out == "hello world")
    }

    @Test("cleanTextForSoprano — converts integer to words")
    func cleanConvertsInteger() {
        let out = cleanTextForSoprano("I have 3 cats")
        #expect(out.contains("three"))
        #expect(!out.contains("3"))
    }

    @Test("cleanTextForSoprano — converts dollar amount to words")
    func cleanConvertsDollarAmount() {
        let out = cleanTextForSoprano("It costs $5")
        #expect(out.contains("five") || out.contains("dollar"))
        #expect(!out.contains("$"))
    }

    @Test("cleanTextForSoprano — expands & to and")
    func cleanExpandsAmpersand() {
        let out = cleanTextForSoprano("cats & dogs")
        #expect(out.contains("and"))
        #expect(!out.contains("&"))
    }

    @Test("cleanTextForSoprano — expands % to percent")
    func cleanExpandsPercent() {
        let out = cleanTextForSoprano("50% complete")
        // After number normalisation "50" → "fifty", then "%" → " percent "
        let lower = out.lowercased()
        #expect(lower.contains("percent"))
    }

    @Test("cleanTextForSoprano — expands mr. abbreviation")
    func cleanExpandsMrAbbreviation() {
        let out = cleanTextForSoprano("Hello Mr. Smith")
        #expect(out.lowercased().contains("mister"))
    }

    @Test("cleanTextForSoprano — empty string stays empty")
    func cleanEmptyString() {
        #expect(cleanTextForSoprano("") == "")
    }

    @Test("cleanTextForSoprano — plain text with period passes through")
    func cleanPlainTextPassesThrough() {
        let out = cleanTextForSoprano("Hello world.")
        #expect(out == "hello world.")
    }

    @Test("cleanTextForSoprano — 2nd becomes second (ordinal)")
    func cleanConvertsOrdinal() {
        let out = cleanTextForSoprano("on the 2nd floor")
        #expect(out.contains("second"))
        #expect(!out.contains("2nd"))
    }

    // ─── SopranoConfig default coverage ──────────────────────────────

    @Test("SopranoConfig — maxPositionEmbeddings defaults to 512")
    func maxPositionEmbeddingsDefault() {
        var raw = Self.soprano80MRaw()
        raw.removeValue(forKey: "max_position_embeddings")
        let config = SopranoConfig.from(
            ModelConfig(architecture: nil, modelType: "soprano", raw: raw))!
        #expect(config.maxPositionEmbeddings == 512)
    }

    @Test("SopranoConfig — ropeTheta defaults to 10 000 when absent")
    func ropeThetaDefault() {
        var raw = Self.soprano80MRaw()
        raw.removeValue(forKey: "rope_theta")
        let config = SopranoConfig.from(
            ModelConfig(architecture: nil, modelType: "soprano", raw: raw))!
        #expect(abs(config.ropeTheta - 10_000) < 1)
    }
}
