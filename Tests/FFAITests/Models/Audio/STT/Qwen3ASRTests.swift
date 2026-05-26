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
import Foundation
import Testing

@testable import FFAI

// Unit tests for the Qwen3ASR speech-to-text family.
// Exercises config decoding and registry detection using synthetic
// `ModelConfig` objects (no real checkpoint required).
@Suite("Qwen3ASR")
struct Qwen3ASRTests {

    // ─── Registry detection ──────────────────────────────────────────

    @Test("AudioModelRegistry — detects Qwen3ASR from model_type")
    func registryDetectsQwen3ASRByModelType() {
        let config = ModelConfig(
            architecture: "Qwen3ASRForConditionalGeneration",
            modelType: "qwen3_asr",
            raw: [
                "model_type": "qwen3_asr",
                "thinker_config": [
                    "audio_config": [
                        "d_model": 896, "encoder_layers": 18,
                        "num_mel_bins": 128,
                    ],
                    "text_config": [
                        "hidden_size": 1024,
                        "num_hidden_layers": 28,
                    ],
                ],
            ])
        #expect(Qwen3ASRModel.handles(config))
        #expect(AudioModelRegistry.handles(config))
        #expect(
            AudioModelRegistry.capabilities(for: config)
                == Capability.speechToText)
    }

    @Test("AudioModelRegistry — detects Qwen3ASR from architecture")
    func registryDetectsQwen3ASRByArchitecture() {
        // Some mlx-community conversions omit model_type but keep the
        // architecture string.
        let config = ModelConfig(
            architecture: "Qwen3ASRForConditionalGeneration",
            modelType: nil,
            raw: [
                "architectures": ["Qwen3ASRForConditionalGeneration"],
                "thinker_config": [
                    "audio_config": ["d_model": 896, "encoder_layers": 18],
                    "text_config": ["hidden_size": 1024],
                ],
            ])
        #expect(Qwen3ASRModel.handles(config))
    }

    @Test("AudioModelRegistry — Qwen3ASR is speechToText capability")
    func qwen3ASRCapability() {
        let config = ModelConfig(
            architecture: nil, modelType: "qwen3_asr",
            raw: [
                "model_type": "qwen3_asr",
                "thinker_config": [
                    "audio_config": ["d_model": 896],
                    "text_config": ["hidden_size": 1024],
                ],
            ])
        #expect(
            AudioModelRegistry.capabilities(for: config)
                == Capability.speechToText)
    }

    @Test("AudioModelRegistry — Qwen3ASR takes priority over QwenOmni")
    func qwen3ASRPriorityOverQwenOmni() {
        // A Qwen3ASR config also has nested audio/text configs, but must be
        // routed to Qwen3ASR — not QwenOmni — based on model_type.
        let config = ModelConfig(
            architecture: nil, modelType: "qwen3_asr",
            raw: [
                "model_type": "qwen3_asr",
                "thinker_config": [
                    "audio_config": ["d_model": 896],
                    "text_config": ["hidden_size": 1024],
                ],
            ])
        #expect(Qwen3ASRModel.handles(config))
        // Qwen3ASR capability is speechToText — not omniAudio.
        #expect(
            AudioModelRegistry.capabilities(for: config)
                != Capability.omniAudio)
    }

    @Test("AudioModelRegistry — text-only config is not Qwen3ASR")
    func textOnlyNotQwen3ASR() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 1024])
        #expect(!Qwen3ASRModel.handles(config))
    }

    // ─── Config parsing ──────────────────────────────────────────────

    @Test("Qwen3ASRConfig — decodes from thinker_config layout")
    func configDecodesFromThinkerConfig() {
        // Representative 0.6B checkpoint config shape.
        let config = ModelConfig(
            architecture: "Qwen3ASRForConditionalGeneration",
            modelType: "qwen3_asr",
            raw: [
                "model_type": "qwen3_asr",
                "thinker_config": [
                    "audio_config": [
                        "d_model": 896,
                        "encoder_layers": 18,
                        "encoder_attention_heads": 14,
                        "encoder_ffn_dim": 3584,
                        "num_mel_bins": 128,
                        "downsample_hidden_size": 480,
                        "output_dim": 1024,
                        "n_window": 50,
                        "max_source_positions": 1500,
                    ],
                    "text_config": [
                        "hidden_size": 1024,
                        "num_hidden_layers": 28,
                        "num_attention_heads": 16,
                        "num_key_value_heads": 8,
                        "head_dim": 128,
                        "intermediate_size": 3072,
                        "rms_norm_eps": 1e-6,
                        "rope_theta": 1_000_000,
                        "vocab_size": 151936,
                        "tie_word_embeddings": true,
                    ],
                    "audio_token_id": 151676,
                    "eos_token_id": [151643, 151645],
                    "pad_token_id": 151643,
                ],
            ])
        let qc = Qwen3ASRConfig.from(config)
        #expect(qc != nil)

        // Audio encoder hyper-parameters.
        #expect(qc?.audioConfig.dModel == 896)
        #expect(qc?.audioConfig.encoderLayers == 18)
        #expect(qc?.audioConfig.encoderAttentionHeads == 14)
        #expect(qc?.audioConfig.encoderFfnDim == 3584)
        #expect(qc?.audioConfig.numMelBins == 128)
        #expect(qc?.audioConfig.downsampleHiddenSize == 480)
        #expect(qc?.audioConfig.outputDim == 1024)
        #expect(qc?.audioConfig.nWindow == 50)
        #expect(qc?.audioConfig.maxSourcePositions == 1500)

        // Text decoder hyper-parameters.
        #expect(qc?.textHidden == 1024)
        #expect(qc?.textLayers == 28)
        #expect(qc?.textHeads == 16)
        #expect(qc?.textKVHeads == 8)
        #expect(qc?.headDim == 128)
        #expect(qc?.textIntermediate == 3072)
        #expect(qc?.vocabSize == 151936)
        #expect(qc?.tieWordEmbeddings == true)

        // Token ids.
        #expect(qc?.audioTokenId == 151676)
        #expect(qc?.eosTokenIds == [151643, 151645])
        #expect(qc?.padTokenId == 151643)
    }

    @Test("Qwen3ASRConfig — falls back to defaults for omitted fields")
    func configUsesDefaults() {
        // Minimal config — only the presence of thinker_config matters for
        // detection; all values fall back to the 0.6B defaults.
        let config = ModelConfig(
            architecture: nil, modelType: "qwen3_asr",
            raw: [
                "model_type": "qwen3_asr",
                "thinker_config": [
                    "audio_config": [:] as [String: Any],
                    "text_config": [:] as [String: Any],
                ],
            ])
        let qc = Qwen3ASRConfig.from(config)
        #expect(qc != nil)
        // Published 0.6B defaults.
        #expect(qc?.audioConfig.dModel == 896)
        #expect(qc?.audioConfig.encoderLayers == 18)
        #expect(qc?.textHidden == 1024)
        #expect(qc?.textLayers == 28)
        #expect(qc?.headDim == 128)
        #expect(qc?.vocabSize == 151936)
    }

    @Test("Qwen3ASRConfig — returns nil when thinker_config absent and no top-level audio/text")
    func configReturnsNilForTextOnly() {
        let config = ModelConfig(
            architecture: nil, modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 1024])
        #expect(Qwen3ASRConfig.from(config) == nil)
    }

    @Test("Qwen3ASRConfig — EOS as single int is promoted to an array")
    func configSingleEos() {
        let config = ModelConfig(
            architecture: nil, modelType: "qwen3_asr",
            raw: [
                "model_type": "qwen3_asr",
                "thinker_config": [
                    "audio_config": ["d_model": 896],
                    "text_config": ["hidden_size": 1024],
                    "eos_token_id": 151643,
                ],
            ])
        let qc = Qwen3ASRConfig.from(config)
        #expect(qc?.eosTokenIds == [151643])
    }
}
