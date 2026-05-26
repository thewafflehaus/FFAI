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
// QwenOmniTests — unit tests for the QwenOmni omni-modal family
// (Qwen2.5-Omni / Qwen3-Omni). All tests run offline, no checkpoint
// required.
//
// Validates:
//   * QwenOmniAudioConfig.from(_:) parses canonical top-level and
//     thinker-nested layouts.
//   * QwenOmniModel.handles(_:) accepts both Qwen-Omni model_types and
//     structural fallback (audio_config presence), and rejects unrelated
//     models.
//   * AudioModelRegistry routes QwenOmni configs to Capability.omniAudio.
//   * QwenOmniError.description includes the expected family prefix.

import Foundation
import Testing
@testable import FFAI

@Suite("QwenOmni")
struct QwenOmniTests {

    // ─── Config parsing ──────────────────────────────────────────────────

    @Test("QwenOmniAudioConfig.from — top-level audio_config layout decodes")
    func configDecodesTopLevel() {
        let raw: [String: Any] = [
            "model_type": "qwen3_omni",
            "hidden_size": 2048,
            "audio_config": [
                "d_model": 1280,
                "encoder_layers": 32,
                "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120,
                "num_mel_bins": 128,
                "max_source_positions": 1500,
                "output_dim": 2048,
            ] as [String: Any],
        ]
        let config = ModelConfig(architecture: nil, modelType: "qwen3_omni", raw: raw)
        let qc = QwenOmniAudioConfig.from(config)
        #expect(qc != nil)
        #expect(qc?.encoderHidden == 1280)
        #expect(qc?.encoderLayers == 32)
        #expect(qc?.encoderHeads == 20)
        #expect(qc?.encoderIntermediate == 5120)
        #expect(qc?.nMels == 128)
        #expect(qc?.maxAudioCtx == 1500)
        #expect(qc?.textHidden == 2048)
    }

    @Test("QwenOmniAudioConfig.from — thinker-nested audio_config layout decodes")
    func configDecodesThinkerNested() {
        let raw: [String: Any] = [
            "model_type": "qwen2_5_omni",
            "thinker_config": [
                "audio_config": [
                    "d_model": 1280,
                    "encoder_layers": 32,
                    "encoder_attention_heads": 20,
                ] as [String: Any],
                "text_config": [
                    "hidden_size": 3584,
                ] as [String: Any],
            ] as [String: Any],
        ]
        let config = ModelConfig(architecture: nil, modelType: "qwen2_5_omni", raw: raw)
        let qc = QwenOmniAudioConfig.from(config)
        #expect(qc != nil)
        #expect(qc?.encoderHidden == 1280)
        // textHidden falls back to text_config.hidden_size when output_dim absent.
        #expect(qc?.textHidden == 3584)
    }

    @Test("QwenOmniAudioConfig.from — falls back to num_hidden_layers when encoder_layers absent")
    func configFallsBackNumHiddenLayers() {
        let raw: [String: Any] = [
            "model_type": "qwen3_omni",
            "audio_config": [
                "d_model": 1280,
                "num_hidden_layers": 24,
                "encoder_attention_heads": 16,
            ] as [String: Any],
        ]
        let config = ModelConfig(architecture: nil, modelType: "qwen3_omni", raw: raw)
        let qc = QwenOmniAudioConfig.from(config)
        #expect(qc?.encoderLayers == 24)
    }

    @Test("QwenOmniAudioConfig.from — returns nil for unrelated configs")
    func configReturnsNilForUnrelated() {
        let raw: [String: Any] = ["model_type": "llama", "hidden_size": 4096]
        let config = ModelConfig(architecture: "LlamaForCausalLM",
                                 modelType: "llama", raw: raw)
        #expect(QwenOmniAudioConfig.from(config) == nil)
    }

    @Test("QwenOmniAudioConfig.frontEnd — uses 16 kHz Whisper-style front-end")
    func frontEndConfigDefaults() {
        let qc = QwenOmniAudioConfig(
            nMels: 128, encoderHidden: 1280, encoderIntermediate: 5120,
            encoderLayers: 32, encoderHeads: 20, maxAudioCtx: 1500,
            textHidden: 2048)
        let fe = qc.frontEnd
        #expect(fe.sampleRate == 16_000)
        #expect(fe.nFFT == 400)
        #expect(fe.hopLength == 160)
        #expect(fe.nMels == 128)
    }

    // ─── Registry detection ──────────────────────────────────────────────

    @Test("QwenOmniModel.modelTypes — contains canonical entries")
    func modelTypesContents() {
        let types = QwenOmniModel.modelTypes
        #expect(types.contains("qwen2_5_omni"))
        #expect(types.contains("qwen3_omni"))
        #expect(types.contains("qwen2_5_omni_thinker"))
    }

    @Test("QwenOmniModel.architectures — contains canonical entries")
    func architecturesContents() {
        let archs = QwenOmniModel.architectures
        #expect(archs.contains("Qwen2_5OmniForConditionalGeneration"))
        #expect(archs.contains("Qwen3OmniMoeForConditionalGeneration"))
    }

    @Test("QwenOmniModel.handles — true for qwen3_omni model_type")
    func handlesByModelType() {
        let config = ModelConfig(architecture: nil, modelType: "qwen3_omni",
                                 raw: ["model_type": "qwen3_omni"])
        #expect(QwenOmniModel.handles(config))
    }

    @Test("QwenOmniModel.handles — true for omni architecture string")
    func handlesByArchitecture() {
        let config = ModelConfig(
            architecture: "Qwen2_5OmniForConditionalGeneration",
            modelType: nil, raw: [:])
        #expect(QwenOmniModel.handles(config))
    }

    @Test("QwenOmniModel.handles — structural fallback via top-level audio_config")
    func handlesStructural() {
        let raw: [String: Any] = [
            "audio_config": ["d_model": 1280] as [String: Any],
        ]
        let config = ModelConfig(architecture: nil, modelType: nil, raw: raw)
        #expect(QwenOmniModel.handles(config))
    }

    @Test("QwenOmniModel.handles — false for unrelated text model")
    func handlesFalseForTextModel() {
        let config = ModelConfig(architecture: "LlamaForCausalLM",
                                 modelType: "llama",
                                 raw: ["model_type": "llama"])
        #expect(!QwenOmniModel.handles(config))
    }

    // ─── AudioModelRegistry routing ──────────────────────────────────────

    @Test("AudioModelRegistry.capabilities — QwenOmni maps to omniAudio")
    func registryCapabilityOmni() {
        let config = ModelConfig(architecture: nil, modelType: "qwen3_omni",
                                 raw: ["model_type": "qwen3_omni"])
        let caps = AudioModelRegistry.capabilities(for: config)
        #expect(caps == Capability.omniAudio)
    }

    // ─── Error stringification ───────────────────────────────────────────

    @Test("QwenOmniError.description — textBackboneUnavailable mentions QwenOmni")
    func errorDescription() {
        let err = QwenOmniError.textBackboneUnavailable
        #expect(err.description.contains("QwenOmni"))
        #expect(err.description.contains("audio"))
    }
}
