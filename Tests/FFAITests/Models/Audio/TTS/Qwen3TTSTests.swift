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
// Qwen3TTSTests — unit tests for the Qwen3TTS (12 Hz Flash) family
// orchestrator: talker_config / speaker_encoder_config decoding,
// codec / TTS BOS/EOS/PAD token ids, family detection, registry
// routing, and the staged-synthesis error surface.
//
// The lower-level Qwen3TTSBase family is covered by Qwen3TTSBaseTests;
// this suite only exercises what differs in the Flash orchestrator.

import Foundation
import Testing

@testable import FFAI

@Suite("Qwen3TTS")
struct Qwen3TTSTests {

    // ─── Helper: canonical raw config ────────────────────────────────────

    private static func canonicalRaw() -> [String: Any] {
        [
            "model_type": "qwen3_tts",
            "architectures": ["Qwen3TTSForConditionalGeneration"],
            "sample_rate": 24_000,
            "tts_bos_token_id": 151_672,
            "tts_eos_token_id": 151_673,
            "tts_pad_token_id": 151_671,
            "talker_config": [
                "vocab_size": 3072,
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000,
                "text_hidden_size": 2048,
                "text_vocab_size": 151_936,
                "num_code_groups": 16,
                "codec_eos_token_id": 2150,
                "rope_scaling": [
                    "mrope_section": [24, 20, 20] as [Int]
                ] as [String: Any],
            ] as [String: Any],
            "speaker_encoder_config": [
                "channels": 512
            ] as [String: Any],
        ]
    }

    // ─── Talker config decoding ──────────────────────────────────────────

    @Test("Qwen3TTSTalkerConfig.from — decodes canonical hyperparameters")
    func talkerConfigDecodes() {
        let talker = (Self.canonicalRaw()["talker_config"] as! [String: Any])
        let cfg = Qwen3TTSTalkerConfig.from(talker)
        #expect(cfg != nil)
        #expect(cfg?.vocabSize == 3072)
        #expect(cfg?.hidden == 1024)
        #expect(cfg?.intermediate == 3072)
        #expect(cfg?.nLayers == 28)
        #expect(cfg?.nHeads == 16)
        #expect(cfg?.nKVHeads == 8)
        #expect(cfg?.headDim == 128)
        #expect(cfg?.textHidden == 2048)
        #expect(cfg?.textVocabSize == 151_936)
        #expect(cfg?.numCodeGroups == 16)
        #expect(cfg?.mropeSection == [24, 20, 20])
    }

    @Test("Qwen3TTSTalkerConfig.from — falls back to published defaults on empty block")
    func talkerConfigDefaults() {
        let cfg = Qwen3TTSTalkerConfig.from([:])
        #expect(cfg?.hidden == 1024)
        #expect(cfg?.nHeads == 16)
        #expect(cfg?.numCodeGroups == 16)
        #expect(cfg?.mropeSection == nil)
    }

    // ─── Top-level config decoding ───────────────────────────────────────

    @Test("Qwen3TTSConfig.from — decodes talker, sample rate, and special tokens")
    func configDecodesCanonical() {
        let config = ModelConfig(
            architecture: nil, modelType: "qwen3_tts",
            raw: Self.canonicalRaw())
        let qc = Qwen3TTSConfig.from(config)
        #expect(qc != nil)
        #expect(qc?.sampleRate == 24_000)
        #expect(qc?.ttsBosTokenId == 151_672)
        #expect(qc?.ttsEosTokenId == 151_673)
        #expect(qc?.ttsPadTokenId == 151_671)
        #expect(qc?.codecEosTokenId == 2150)
        #expect(qc?.talker.hidden == 1024)
    }

    @Test("Qwen3TTSConfig.from — returns nil when talker_config absent")
    func configReturnsNilForMissingTalker() {
        let raw: [String: Any] = ["model_type": "qwen3_tts"]
        let config = ModelConfig(architecture: nil, modelType: "qwen3_tts", raw: raw)
        #expect(Qwen3TTSConfig.from(config) == nil)
    }

    @Test("Qwen3TTSConfig.from — codecEosTokenId defaults to 2150 when absent")
    func configCodecEosDefault() {
        var raw = Self.canonicalRaw()
        var talker = raw["talker_config"] as! [String: Any]
        talker.removeValue(forKey: "codec_eos_token_id")
        raw["talker_config"] = talker
        let config = ModelConfig(architecture: nil, modelType: "qwen3_tts", raw: raw)
        let qc = Qwen3TTSConfig.from(config)
        #expect(qc?.codecEosTokenId == 2150)
    }

    // ─── Registry detection ──────────────────────────────────────────────

    @Test("Qwen3TTSModel.modelTypes — contains qwen3_tts and qwen3tts")
    func modelTypesContents() {
        let types = Qwen3TTSModel.modelTypes
        #expect(types.contains("qwen3_tts"))
        #expect(types.contains("qwen3tts"))
    }

    @Test("Qwen3TTSModel.architectures — contains canonical entry")
    func architecturesContents() {
        #expect(
            Qwen3TTSModel.architectures.contains(
                "Qwen3TTSForConditionalGeneration"))
    }

    @Test("Qwen3TTSModel.handles — true for qwen3_tts model_type")
    func handlesByModelType() {
        let config = ModelConfig(
            architecture: nil, modelType: "qwen3_tts",
            raw: ["model_type": "qwen3_tts"])
        #expect(Qwen3TTSModel.handles(config))
    }

    @Test("Qwen3TTSModel.handles — true for canonical Qwen3TTS architecture")
    func handlesByArchitecture() {
        let config = ModelConfig(
            architecture: "Qwen3TTSForConditionalGeneration",
            modelType: nil, raw: [:])
        #expect(Qwen3TTSModel.handles(config))
    }

    @Test("Qwen3TTSModel.handles — structural fallback via talker_config + speaker_encoder_config")
    func handlesStructural() {
        let raw: [String: Any] = [
            "talker_config": [:] as [String: Any],
            "speaker_encoder_config": [:] as [String: Any],
        ]
        let config = ModelConfig(architecture: nil, modelType: nil, raw: raw)
        #expect(Qwen3TTSModel.handles(config))
    }

    @Test("Qwen3TTSModel.handles — false for unrelated text model")
    func handlesFalseForTextModel() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM",
            modelType: "llama",
            raw: ["model_type": "llama"])
        #expect(!Qwen3TTSModel.handles(config))
    }

    // ─── AudioModelRegistry routing ──────────────────────────────────────

    @Test("AudioModelRegistry.capabilities — Qwen3TTS maps to textToSpeech")
    func registryCapability() {
        let config = ModelConfig(
            architecture: nil, modelType: "qwen3_tts",
            raw: Self.canonicalRaw())
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.textToSpeech)
    }

    // ─── Error stringification ───────────────────────────────────────────

    @Test("Qwen3TTSError.description — synthesisNotWired mentions Qwen3TTS")
    func errorDescriptionSynthesis() {
        let err = Qwen3TTSError.synthesisNotWired
        let desc = err.description
        #expect(desc.contains("Qwen3TTS"))
        #expect(desc.contains("stage") || desc.contains("synthesis"))
    }

    @Test("Qwen3TTSError.description — missingConfig mentions Qwen3TTS")
    func errorDescriptionMissingConfig() {
        let err = Qwen3TTSError.missingConfig
        #expect(err.description.contains("Qwen3TTS"))
    }
}
