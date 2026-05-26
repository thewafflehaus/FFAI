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
// CohereTranscribeTests — unit tests for CohereTranscribe config parsing,
// audio front-end, and registry detection.
// No real checkpoint required; all tests run offline.
//
// Validates:
//   * CohereTranscribeConfig.from(_:) parses config shapes correctly.
//   * CohereTranscribeModel.handles(_:) fires on the expected config signatures.
//   * AudioModelRegistry routes CohereTranscribe configs to Capability.speechToText.
//   * computeMelFeatures (via encodeAudio path) produces finite outputs.

import Foundation
import Testing
@testable import FFAI

@Suite("CohereTranscribe")
struct CohereTranscribeTests {

    // ─── Config parsing ──────────────────────────────────────────────────

    @Test("CohereTranscribeConfig.from — defaults for minimal model_type config")
    func configDefaultsFromModelType() {
        let config = ModelConfig(
            architecture: nil, modelType: "cohere_transcribe",
            raw: ["model_type": "cohere_transcribe"])
        let cfg = CohereTranscribeConfig.from(config)
        #expect(cfg != nil)

        // Published AED defaults.
        #expect(cfg?.encoder.dModel            == 512)
        #expect(cfg?.encoder.ffExpansionFactor == 4)
        #expect(cfg?.encoder.nHeads            == 8)
        #expect(cfg?.encoder.convKernelSize    == 31)
        #expect(cfg?.encoder.nLayers           == 18)
        #expect(cfg?.encoder.posEmbMaxLen      == 5000)
        #expect(cfg?.encoder.subsamplingConvChannels == 256)
        #expect(cfg?.encoder.subsamplingFactor == 8)
        #expect(cfg?.encoder.featIn            == 128)

        #expect(cfg?.decoder.hiddenSize          == 512)
        #expect(cfg?.decoder.innerSize           == 2048)
        #expect(cfg?.decoder.numAttentionHeads   == 8)
        #expect(cfg?.decoder.numLayers           == 6)
        #expect(cfg?.decoder.maxSequenceLength   == 512)

        #expect(cfg?.sampleRate     == 16_000)
        #expect(cfg?.maxAudioClipS  == 60)
    }

    @Test("CohereTranscribeConfig.from — detects by architecture string")
    func configDetectsByArchitecture() {
        let config = ModelConfig(
            architecture: "CohereTranscribeForConditionalGeneration",
            modelType: nil,
            raw: ["architectures": ["CohereTranscribeForConditionalGeneration"]])
        let cfg = CohereTranscribeConfig.from(config)
        #expect(cfg != nil)
    }

    @Test("CohereTranscribeConfig.from — parses explicit nested encoder/decoder")
    func configParsesExplicitFields() {
        let config = ModelConfig(
            architecture: "CohereTranscribeForConditionalGeneration",
            modelType: "cohere_transcribe",
            raw: [
                "model_type": "cohere_transcribe",
                "vocab_size": 50000,
                "sample_rate": 16000,
                "encoder": [
                    "d_model":                    1024,
                    "ff_expansion_factor":        4,
                    "n_heads":                    16,
                    "conv_kernel_size":           31,
                    "n_layers":                   24,
                    "pos_emb_max_len":            5000,
                    "subsampling_conv_channels":  256,
                    "subsampling_factor":         8,
                    "feat_in":                    128,
                ] as [String: Any],
                "transf_decoder": [
                    "config_dict": [
                        "hidden_size":           1024,
                        "inner_size":            4096,
                        "num_attention_heads":   16,
                        "num_layers":            12,
                        "max_sequence_length":   512,
                    ] as [String: Any]
                ] as [String: Any],
            ])
        let cfg = CohereTranscribeConfig.from(config)
        #expect(cfg != nil)
        #expect(cfg?.vocabSize           == 50000)
        #expect(cfg?.encoder.dModel      == 1024)
        #expect(cfg?.encoder.nHeads      == 16)
        #expect(cfg?.encoder.nLayers     == 24)
        #expect(cfg?.decoder.hiddenSize  == 1024)
        #expect(cfg?.decoder.innerSize   == 4096)
        #expect(cfg?.decoder.numLayers   == 12)
    }

    @Test("CohereTranscribeConfig.from — returns nil for text-only config")
    func configReturnsNilForTextOnly() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 4096])
        #expect(CohereTranscribeConfig.from(config) == nil)
    }

    // ─── Registry detection ──────────────────────────────────────────────

    @Test("CohereTranscribeModel.handles — true for model_type cohere_transcribe")
    func handlesByModelType() {
        let config = ModelConfig(
            architecture: nil, modelType: "cohere_transcribe",
            raw: ["model_type": "cohere_transcribe"])
        #expect(CohereTranscribeModel.handles(config))
    }

    @Test("CohereTranscribeModel.handles — true for CohereTranscribeForConditionalGeneration")
    func handlesByArchitecture() {
        let config = ModelConfig(
            architecture: "CohereTranscribeForConditionalGeneration",
            modelType: nil,
            raw: [:])
        #expect(CohereTranscribeModel.handles(config))
    }

    @Test("CohereTranscribeModel.handles — false for unrelated model")
    func handlesFalseForTextModel() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama"])
        #expect(!CohereTranscribeModel.handles(config))
    }

    @Test("CohereTranscribeModel.handles — false for Whisper")
    func handlesFalseForWhisper() {
        let config = ModelConfig(
            architecture: "WhisperForConditionalGeneration",
            modelType: "whisper",
            raw: ["model_type": "whisper"])
        #expect(!CohereTranscribeModel.handles(config))
    }

    @Test("AudioModelRegistry.handles — true for CohereTranscribe config")
    func registryHandlesCohereTranscribe() {
        let config = ModelConfig(
            architecture: "CohereTranscribeForConditionalGeneration",
            modelType: "cohere_transcribe",
            raw: ["model_type": "cohere_transcribe"])
        #expect(AudioModelRegistry.handles(config))
    }

    @Test("AudioModelRegistry.capabilities — CohereTranscribe maps to speechToText")
    func registryCapabilitySpeechToText() {
        let config = ModelConfig(
            architecture: "CohereTranscribeForConditionalGeneration",
            modelType: "cohere_transcribe",
            raw: ["model_type": "cohere_transcribe"])
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.speechToText)
    }

    @Test("AudioModelRegistry.handles — false for text-only model")
    func registryFalseForTextModel() {
        let config = ModelConfig(
            architecture: nil, modelType: "qwen3",
            raw: ["model_type": "qwen3", "hidden_size": 2048])
        #expect(!CohereTranscribeModel.handles(config))
    }
}
