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
// MarvisTests — unit tests for the Marvis / CSM TTS family: CSM
// transformer config decoding, top-level MarvisConfig + nested
// depth_decoder_config, family detection, registry routing, and
// MarvisError surface. All tests run offline, no checkpoint required.
//
// Validates:
//   * MarvisConfig.from(_:) parses backbone + depth-decoder transformers
//     and required text/audio vocab counts.
//   * MarvisModel.handles(_:) accepts canonical model_types,
//     architectures, and structural fallback (audio_num_codebooks +
//     depth_decoder_config / backbone_flavor).
//   * AudioModelRegistry routes Marvis configs to Capability.textToSpeech.
//   * MarvisError.description renders the family prefix.

import Foundation
import Testing
@testable import FFAI

@Suite("Marvis")
struct MarvisTests {

    // ─── Helper ──────────────────────────────────────────────────────────

    private static func canonicalRaw() -> [String: Any] {
        [
            "model_type": "csm",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "intermediate_size": 8192,
            "head_dim": 128,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500_000,
            "vocab_size": 128_256,
            "text_vocab_size": 128_256,
            "audio_vocab_size": 2051,
            "audio_num_codebooks": 32,
            "sample_rate": 24_000,
            "depth_decoder_config": [
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "intermediate_size": 4096,
                "head_dim": 128,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
            ] as [String: Any],
        ]
    }

    // ─── Config decoding ─────────────────────────────────────────────────

    @Test("MarvisConfig.from — decodes backbone and depth-decoder configs")
    func configDecodesCanonical() {
        let raw = Self.canonicalRaw()
        let config = ModelConfig(architecture: nil, modelType: "csm", raw: raw)
        let mc = MarvisConfig.from(config)
        #expect(mc != nil)
        #expect(mc?.backbone.hidden == 2048)
        #expect(mc?.backbone.nLayers == 16)
        #expect(mc?.backbone.nHeads == 16)
        #expect(mc?.backbone.intermediate == 8192)
        #expect(mc?.backbone.headDim == 128)
        #expect(mc?.backbone.nKVHeads == 8)
        #expect(mc?.decoder.hidden == 1024)
        #expect(mc?.decoder.nLayers == 4)
        #expect(mc?.decoder.nHeads == 8)
        #expect(mc?.decoder.intermediate == 4096)
        #expect(mc?.textVocabSize == 128_256)
        #expect(mc?.audioVocabSize == 2051)
        #expect(mc?.audioNumCodebooks == 32)
        #expect(mc?.sampleRate == 24_000)
    }

    @Test("MarvisConfig.from — depth-decoder absent falls back to backbone shape")
    func configFallsBackDecoderToBackbone() {
        var raw = Self.canonicalRaw()
        raw.removeValue(forKey: "depth_decoder_config")
        let config = ModelConfig(architecture: nil, modelType: "csm", raw: raw)
        let mc = MarvisConfig.from(config)
        #expect(mc != nil)
        // Decoder mirrors the backbone in the single-transformer fallback.
        #expect(mc?.decoder.hidden == mc?.backbone.hidden)
        #expect(mc?.decoder.nLayers == mc?.backbone.nLayers)
    }

    @Test("MarvisConfig.from — returns nil when required text/audio vocab fields are missing")
    func configReturnsNilForMissingVocab() {
        var raw = Self.canonicalRaw()
        raw.removeValue(forKey: "audio_num_codebooks")
        raw.removeValue(forKey: "audio_vocab_size")
        let config = ModelConfig(architecture: nil, modelType: "csm", raw: raw)
        #expect(MarvisConfig.from(config) == nil)
    }

    @Test("MarvisConfig.from — sampleRate defaults to 24 kHz when absent")
    func configSampleRateDefault() {
        var raw = Self.canonicalRaw()
        raw.removeValue(forKey: "sample_rate")
        let config = ModelConfig(architecture: nil, modelType: "csm", raw: raw)
        #expect(MarvisConfig.from(config)?.sampleRate == 24_000)
    }

    // ─── Registry detection ──────────────────────────────────────────────

    @Test("MarvisModel.modelTypes — contains csm and marvis")
    func modelTypesContents() {
        let types = MarvisModel.modelTypes
        #expect(types.contains("csm"))
        #expect(types.contains("marvis"))
    }

    @Test("MarvisModel.architectures — contains CSMForConditionalGeneration")
    func architecturesContents() {
        #expect(MarvisModel.architectures.contains("CSMForConditionalGeneration"))
    }

    @Test("MarvisModel.handles — true for csm model_type")
    func handlesByModelType() {
        let config = ModelConfig(architecture: nil, modelType: "csm",
                                 raw: ["model_type": "csm"])
        #expect(MarvisModel.handles(config))
    }

    @Test("MarvisModel.handles — structural fallback via audio_num_codebooks + depth_decoder_config")
    func handlesStructural() {
        let raw: [String: Any] = [
            "audio_num_codebooks": 32,
            "depth_decoder_config": [:] as [String: Any],
        ]
        let config = ModelConfig(architecture: nil, modelType: nil, raw: raw)
        #expect(MarvisModel.handles(config))
    }

    @Test("MarvisModel.handles — false for unrelated text model")
    func handlesFalseForTextModel() {
        let config = ModelConfig(architecture: "LlamaForCausalLM",
                                 modelType: "llama",
                                 raw: ["model_type": "llama"])
        #expect(!MarvisModel.handles(config))
    }

    // ─── AudioModelRegistry routing ──────────────────────────────────────

    @Test("AudioModelRegistry.capabilities — Marvis maps to textToSpeech")
    func registryCapability() {
        let config = ModelConfig(architecture: nil, modelType: "csm",
                                 raw: ["model_type": "csm"])
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.textToSpeech)
    }

    // ─── Error stringification ───────────────────────────────────────────

    @Test("MarvisError.description — codecUnavailable mentions Marvis")
    func errorDescriptionCodec() {
        let err = MarvisError.codecUnavailable
        #expect(err.description.contains("Marvis") || err.description.contains("Mimi"))
    }

    @Test("MarvisError.description — missingConfig mentions config")
    func errorDescriptionMissing() {
        let err = MarvisError.missingConfig
        let desc = err.description
        #expect(desc.contains("Marvis") || desc.contains("config"))
    }

    @Test("MarvisError.description — noFrames mentions frames or audio")
    func errorDescriptionNoFrames() {
        let err = MarvisError.noFrames
        let desc = err.description
        #expect(desc.contains("Marvis") || desc.contains("frame")
                || desc.contains("audio"))
    }
}
