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
// FishSpeechConfigTests — unit tests for the FishSpeechConfig + the
// FishSpeechSubConfig decoders. Focuses on the config-decoding surface
// only; the family-detection / registry plumbing lives in
// FishSpeechTests.
//
// Validates:
//   * FishSpeechSubConfig — defaults vs custom values for text-backbone
//     and audio-decoder shapes.
//   * FishSpeechConfig.load — top-level fields decoded from the canonical
//     fish-audio-s2-pro-8bit config layout.
//   * Numeric coercion paths (Int vs Double for rope_base / norm_eps).

import Foundation
import Testing

@testable import FFAI

@Suite("FishSpeechConfig")
struct FishSpeechConfigTests {

    // ─── Sub-config decoding ─────────────────────────────────────────────

    @Test("FishSpeechSubConfig — text-backbone defaults match upstream")
    func textBackboneDefaults() throws {
        // Empty raw block — every field should pick its textBackbone default.
        let raw: [String: Any] = ["model_type": "fish_qwen3_omni"]
        let mc = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: raw)
        let cfg = try FishSpeechConfig.load(from: mc)
        let text = cfg.textConfig
        #expect(text.modelType == "fish_qwen3")
        #expect(text.vocabSize == 155_776)
        #expect(text.nLayer == 36)
        #expect(text.nHead == 32)
        #expect(text.dim == 2560)
        #expect(text.intermediateSize == 9728)
        #expect(text.nLocalHeads == 8)
        #expect(text.headDim == 128)
        #expect(text.maxSeqLen == 32_768)
        #expect(text.tieWordEmbeddings == true)
        #expect(text.attentionQKNorm == true)
        #expect(text.attentionQKVBias == false)
        #expect(text.attentionOBias == false)
        // nKVHeads clamps to nLocalHeads when positive.
        #expect(text.nKVHeads == 8)
    }

    @Test("FishSpeechSubConfig — audio-decoder defaults match upstream")
    func audioDecoderDefaults() throws {
        let raw: [String: Any] = ["model_type": "fish_qwen3_omni"]
        let mc = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: raw)
        let cfg = try FishSpeechConfig.load(from: mc)
        let audio = cfg.audioDecoderConfig
        #expect(audio.modelType == "fish_qwen3_audio_decoder")
        #expect(audio.vocabSize == 4096)
        #expect(audio.nLayer == 4)
        #expect(audio.maxSeqLen == 11)
        #expect(audio.tieWordEmbeddings == false)
        #expect(audio.attentionQKNorm == false)
    }

    @Test("FishSpeechSubConfig — explicit n_local_heads <= 0 falls back to default")
    func subConfigClampsNonPositiveKVHeads() throws {
        let raw: [String: Any] = [
            "model_type": "fish_qwen3_omni",
            "text_config": [
                "n_local_heads": 0,  // not positive → falls back to default 8
                "dim": 2560,
                "n_head": 32,
            ] as [String: Any],
        ]
        let mc = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: raw)
        let cfg = try FishSpeechConfig.load(from: mc)
        // The clamp falls back to the published default of 8.
        #expect(cfg.textConfig.nLocalHeads == 8)
        #expect(cfg.textConfig.nKVHeads == 8)
    }

    @Test("FishSpeechSubConfig — rope_base accepts both Int and Double")
    func subConfigRopeBaseCoercion() throws {
        // Double-form rope_base.
        let raw1: [String: Any] = [
            "model_type": "fish_qwen3_omni",
            "text_config": ["rope_base": 500_000.0] as [String: Any],
        ]
        let mc1 = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: raw1)
        let cfg1 = try FishSpeechConfig.load(from: mc1)
        #expect(abs(cfg1.textConfig.ropeBase - 500_000) < 1)

        // Int-form rope_base.
        let raw2: [String: Any] = [
            "model_type": "fish_qwen3_omni",
            "text_config": ["rope_base": 250_000] as [String: Any],
        ]
        let mc2 = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: raw2)
        let cfg2 = try FishSpeechConfig.load(from: mc2)
        #expect(abs(cfg2.textConfig.ropeBase - 250_000) < 1)
    }

    @Test("FishSpeechSubConfig — norm_eps accepts both Double and Int (falls back when missing)")
    func subConfigNormEps() throws {
        let raw: [String: Any] = [
            "model_type": "fish_qwen3_omni",
            "text_config": ["norm_eps": 1e-5] as [String: Any],
        ]
        let mc = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: raw)
        let cfg = try FishSpeechConfig.load(from: mc)
        #expect(abs(cfg.textConfig.normEps - 1e-5) < 1e-9)
    }

    // ─── Top-level config ────────────────────────────────────────────────

    @Test("FishSpeechConfig.load — top-level token ids decode from canonical raw")
    func topLevelTokenIDs() throws {
        let raw: [String: Any] = [
            "model_type": "fish_qwen3_omni",
            "pad_token_id": 151_669,
            "eos_token_id": 151_645,
            "audio_pad_token_id": 151_677,
            "semantic_start_token_id": 151_678,
            "semantic_end_token_id": 155_773,
            "sample_rate": 44_100,
        ]
        let mc = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: raw)
        let cfg = try FishSpeechConfig.load(from: mc)
        #expect(cfg.padTokenID == 151_669)
        #expect(cfg.eosTokenID == 151_645)
        #expect(cfg.audioPadTokenID == 151_677)
        #expect(cfg.semanticStartTokenID == 151_678)
        #expect(cfg.semanticEndTokenID == 155_773)
        #expect(cfg.sampleRate == 44_100)
    }

    @Test("FishSpeechConfig.load — sampleRate defaults to 44100 when omitted")
    func sampleRateDefault() throws {
        let raw: [String: Any] = ["model_type": "fish_qwen3_omni"]
        let mc = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: raw)
        let cfg = try FishSpeechConfig.load(from: mc)
        #expect(cfg.sampleRate == 44_100)
    }

    @Test("FishSpeechConfig.load — numCodebooks reads nested audio_decoder_config")
    func numCodebooksDecodesFromNested() throws {
        let raw: [String: Any] = [
            "model_type": "fish_qwen3_omni",
            "audio_decoder_config": ["num_codebooks": 16] as [String: Any],
        ]
        let mc = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: raw)
        let cfg = try FishSpeechConfig.load(from: mc)
        #expect(cfg.numCodebooks == 16)
    }

    @Test("FishSpeechConfig.load — numCodebooks defaults to 10 when absent")
    func numCodebooksDefault() throws {
        let raw: [String: Any] = ["model_type": "fish_qwen3_omni"]
        let mc = ModelConfig(architecture: nil, modelType: "fish_qwen3_omni", raw: raw)
        let cfg = try FishSpeechConfig.load(from: mc)
        #expect(cfg.numCodebooks == 10)
    }
}
