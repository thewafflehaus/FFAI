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
// FishSpeechTests — unit tests for config parsing and registry dispatch.
//
// These tests run without loading any checkpoint (no GPU required) and
// complete in milliseconds.

import Foundation
import Testing

@testable import FFAI

@Suite("FishSpeech")
struct FishSpeechTests {

    // ─── Config parsing ──────────────────────────────────────────────────

    @Test("FishSpeechConfig parses s2-pro-8bit defaults")
    func configDefaultFields() throws {
        // Build a raw ModelConfig that matches the fish-audio-s2-pro-8bit
        // config.json without touching the filesystem.
        let rawJSON: [String: Any] = [
            "model_type": "fish_qwen3_omni",
            "pad_token_id": 151_669,
            "eos_token_id": 151_645,
            "audio_pad_token_id": 151_677,
            "semantic_start_token_id": 151_678,
            "semantic_end_token_id": 155_773,
            "sample_rate": 44_100,
            "text_config": [
                "model_type": "fish_qwen3",
                "vocab_size": 155_776,
                "n_layer": 36,
                "n_head": 32,
                "dim": 2560,
                "intermediate_size": 9728,
                "n_local_heads": 8,
                "head_dim": 128,
                "rope_base": 1_000_000,
                "norm_eps": 1e-6,
                "max_seq_len": 32_768,
                "tie_word_embeddings": true,
                "attention_qkv_bias": false,
                "attention_o_bias": false,
                "attention_qk_norm": true,
            ] as [String: Any],
            "audio_decoder_config": [
                "model_type": "fish_qwen3_audio_decoder",
                "vocab_size": 4096,
                "n_layer": 4,
                "n_head": 32,
                "dim": 2560,
                "intermediate_size": 9728,
                "n_local_heads": 8,
                "head_dim": 128,
                "rope_base": 1_000_000,
                "norm_eps": 1e-6,
                "max_seq_len": 11,
                "num_codebooks": 10,
                "tie_word_embeddings": false,
                "attention_qkv_bias": false,
                "attention_o_bias": false,
                "attention_qk_norm": false,
            ] as [String: Any],
            "quantization": [
                "bits": 8,
                "group_size": 64,
            ] as [String: Any],
        ]

        let mc = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: rawJSON)
        let cfg = try FishSpeechConfig.load(from: mc)

        // Top-level fields
        #expect(cfg.modelType == "fish_qwen3_omni")
        #expect(cfg.padTokenID == 151_669)
        #expect(cfg.eosTokenID == 151_645)
        #expect(cfg.audioPadTokenID == 151_677)
        #expect(cfg.semanticStartTokenID == 151_678)
        #expect(cfg.semanticEndTokenID == 155_773)
        #expect(cfg.sampleRate == 44_100)
        #expect(cfg.numCodebooks == 10)

        // Text backbone
        #expect(cfg.textConfig.vocabSize == 155_776)
        #expect(cfg.textConfig.nLayer == 36)
        #expect(cfg.textConfig.nHead == 32)
        #expect(cfg.textConfig.dim == 2560)
        #expect(cfg.textConfig.nKVHeads == 8)
        #expect(cfg.textConfig.headDim == 128)
        #expect(cfg.textConfig.attentionQKNorm == true)
        #expect(cfg.textConfig.tieWordEmbeddings == true)

        // Audio decoder
        #expect(cfg.audioDecoderConfig.vocabSize == 4096)
        #expect(cfg.audioDecoderConfig.nLayer == 4)
        #expect(cfg.audioDecoderConfig.nKVHeads == 8)
        #expect(cfg.audioDecoderConfig.attentionQKNorm == false)
        #expect(cfg.audioDecoderConfig.tieWordEmbeddings == false)

        // Quantization
        #expect(cfg.quantization != nil)
        #expect(cfg.quantization?.bits == 8)
        #expect(cfg.quantization?.groupSize == 64)
    }

    @Test("FishSpeechConfig uses sub-config defaults when keys are absent")
    func configFallsBackToDefaults() throws {
        // Minimal config — only model_type, no nested sub-configs.
        let raw: [String: Any] = ["model_type": "fish_speech"]
        let mc = ModelConfig(architecture: nil, modelType: "fish_speech", raw: raw)
        let cfg = try FishSpeechConfig.load(from: mc)

        // Should hit textBackbone defaults.
        #expect(cfg.textConfig.vocabSize == 155_776)
        #expect(cfg.textConfig.nLayer == 36)
        #expect(cfg.textConfig.dim == 2560)
        #expect(cfg.textConfig.attentionQKNorm == true)

        // Should hit audioDecoder defaults.
        #expect(cfg.audioDecoderConfig.vocabSize == 4096)
        #expect(cfg.audioDecoderConfig.nLayer == 4)
        #expect(cfg.audioDecoderConfig.attentionQKNorm == false)

        // Default numCodebooks.
        #expect(cfg.numCodebooks == 10)

        // No quantization in raw → nil.
        #expect(cfg.quantization == nil)
    }

    // ─── Family / registry dispatch ──────────────────────────────────────

    @Test("FishSpeech.modelTypes contains expected identifiers")
    func fishSpeechModelTypesNotEmpty() {
        #expect(FishSpeech.modelTypes.contains("fish_speech"))
        #expect(FishSpeech.modelTypes.contains("fish_qwen3_omni"))
    }

    @Test("AudioModelRegistry dispatches fish_speech model_type")
    func registryDispatchesFishSpeech() {
        // AudioModelRegistry.load requires a real SafeTensorsBundle and directory.
        // We only check the dispatch logic path here — the model_type in the
        // config matches FishSpeech.modelTypes. Full load is in the integration test.
        let raw: [String: Any] = ["model_type": "fish_speech"]
        let mc = ModelConfig(architecture: nil, modelType: "fish_speech", raw: raw)
        #expect(FishSpeech.modelTypes.contains(mc.modelType ?? ""))
    }

    @Test("AudioGenerationParameters has expected defaults")
    func audioGenerationParameterDefaults() {
        let p = AudioGenerationParameters()
        #expect(p.maxTokens == 1024)
        #expect(abs(p.temperature - 0.7) < 1e-5)
        #expect(abs(p.topP - 0.7) < 1e-5)
        #expect(p.topK == 30)
        #expect(abs(p.speed - 1.0) < 1e-5)
    }

    @Test("FishSpeech.defaultParameters are within expected ranges")
    func fishSpeechDefaultParameters() {
        let p = FishSpeech.defaultParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature > 0 && p.temperature <= 2.0)
        #expect(p.topP > 0 && p.topP <= 1.0)
        #expect(p.topK > 0)
    }

    @Test("FishSpeechConfig.sampleRate is 44100")
    func fishSpeechSampleRate() throws {
        let raw: [String: Any] = ["model_type": "fish_qwen3_omni"]
        let mc = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: raw)
        let cfg = try FishSpeechConfig.load(from: mc)
        #expect(cfg.sampleRate == 44_100)
    }
}
