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
// WhisperTests — unit tests for the OpenAI Whisper STT family: config
// parsing, registry detection, AudioModelRegistry routing, and pure-CPU
// helpers (pad_or_trim). All tests run offline, no checkpoint required.
//
// Validates:
//   * WhisperConfig.from(_:) parses the canonical HF Whisper config layout.
//   * WhisperModel.handles(_:) accepts whisper model_type / arch and
//     rejects unrelated configs.
//   * AudioModelRegistry routes Whisper configs to Capability.speechToText.
//   * WhisperModel.padOrTrim pads short clips and trims long ones to the
//     exact 30 s analysis window the encoder is trained on.

import Foundation
import Testing

@testable import FFAI

@Suite("Whisper")
struct WhisperTests {

    // ─── Config parsing ──────────────────────────────────────────────────

    @Test("WhisperConfig.from — decodes canonical tiny config")
    func configDecodesTiny() {
        let raw: [String: Any] = [
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
            "d_model": 384,
            "encoder_layers": 4,
            "encoder_attention_heads": 6,
            "decoder_layers": 4,
            "decoder_attention_heads": 6,
            "vocab_size": 51865,
            "num_mel_bins": 80,
            "max_target_positions": 448,
            "max_source_positions": 1500,
        ]
        let config = ModelConfig(
            architecture: "WhisperForConditionalGeneration",
            modelType: "whisper", raw: raw)
        let wc = WhisperConfig.from(config)
        #expect(wc != nil)
        #expect(wc?.hidden == 384)
        #expect(wc?.encoderLayers == 4)
        #expect(wc?.encoderHeads == 6)
        #expect(wc?.decoderLayers == 4)
        #expect(wc?.decoderHeads == 6)
        #expect(wc?.vocab == 51865)
        #expect(wc?.nMels == 80)
        #expect(wc?.maxDecoderCtx == 448)
        #expect(wc?.maxAudioCtx == 1500)
        // Intermediate defaults to 4 × hidden when encoder_ffn_dim absent.
        #expect(wc?.intermediate == 1536)
    }

    @Test("WhisperConfig.from — large-v3 uses 128 mel bins")
    func configDecodesLargeV3() {
        let raw: [String: Any] = [
            "model_type": "whisper",
            "d_model": 1280,
            "encoder_layers": 32,
            "encoder_attention_heads": 20,
            "decoder_layers": 32,
            "decoder_attention_heads": 20,
            "vocab_size": 51866,
            "num_mel_bins": 128,
            "encoder_ffn_dim": 5120,
        ]
        let config = ModelConfig(architecture: nil, modelType: "whisper", raw: raw)
        let wc = WhisperConfig.from(config)
        #expect(wc?.nMels == 128)
        #expect(wc?.intermediate == 5120)
        // headDim derives from hidden / decoderHeads.
        #expect(wc?.decoderHeadDim == 64)
    }

    @Test("WhisperConfig.from — returns nil for unrelated configs")
    func configReturnsNilForUnrelated() {
        let raw: [String: Any] = ["model_type": "llama", "hidden_size": 4096]
        let config = ModelConfig(
            architecture: "LlamaForCausalLM",
            modelType: "llama", raw: raw)
        #expect(WhisperConfig.from(config) == nil)
    }

    @Test("WhisperConfig.frontEnd — 16 kHz / 400 nFFT / 160 hop")
    func frontEndConfig() {
        let wc = WhisperConfig(
            nMels: 80, hidden: 384,
            encoderLayers: 4, encoderHeads: 6,
            decoderLayers: 4, decoderHeads: 6,
            intermediate: 1536, vocab: 51865)
        let fe = wc.frontEnd
        #expect(fe.sampleRate == 16_000)
        #expect(fe.nFFT == 400)
        #expect(fe.hopLength == 160)
        #expect(fe.nMels == 80)
    }

    // ─── Registry detection ──────────────────────────────────────────────

    @Test("WhisperModel.modelTypes — contains whisper")
    func modelTypesContents() {
        #expect(WhisperModel.modelTypes.contains("whisper"))
    }

    @Test("WhisperModel.architectures — contains canonical entries")
    func architecturesContents() {
        let archs = WhisperModel.architectures
        #expect(archs.contains("WhisperForConditionalGeneration"))
        #expect(archs.contains("WhisperModel"))
    }

    @Test("WhisperModel.handles — true for whisper model_type")
    func handlesByModelType() {
        let config = ModelConfig(
            architecture: nil, modelType: "whisper",
            raw: ["model_type": "whisper"])
        #expect(WhisperModel.handles(config))
    }

    @Test("WhisperModel.handles — true for WhisperForConditionalGeneration arch")
    func handlesByArchitecture() {
        let config = ModelConfig(
            architecture: "WhisperForConditionalGeneration",
            modelType: nil, raw: [:])
        #expect(WhisperModel.handles(config))
    }

    @Test("WhisperModel.handles — false for unrelated text model")
    func handlesFalseForTextModel() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM",
            modelType: "llama",
            raw: ["model_type": "llama"])
        #expect(!WhisperModel.handles(config))
    }

    // ─── AudioModelRegistry routing ──────────────────────────────────────

    @Test("AudioModelRegistry.capabilities — Whisper maps to speechToText")
    func registryCapability() {
        let config = ModelConfig(
            architecture: nil, modelType: "whisper",
            raw: ["model_type": "whisper"])
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.speechToText)
    }

    // ─── padOrTrim ──────────────────────────────────────────────────────

    @Test("WhisperModel.padOrTrim — short clip is zero-padded to target length")
    func padOrTrimPads() {
        let clip = [Float](repeating: 0.5, count: 100)
        let padded = WhisperModel.padOrTrim(clip, to: 500)
        #expect(padded.count == 500)
        // First 100 samples preserved.
        #expect(padded[0] == 0.5)
        #expect(padded[99] == 0.5)
        // Trailing samples zero.
        #expect(padded[100] == 0)
        #expect(padded[499] == 0)
    }

    @Test("WhisperModel.padOrTrim — long clip is truncated to target length")
    func padOrTrimTrims() {
        let clip = (0 ..< 1000).map { Float($0) }
        let trimmed = WhisperModel.padOrTrim(clip, to: 500)
        #expect(trimmed.count == 500)
        #expect(trimmed[0] == 0)
        #expect(trimmed[499] == 499)
    }

    @Test("WhisperModel.padOrTrim — exact-length clip is returned unchanged")
    func padOrTrimExact() {
        let clip = (0 ..< 500).map { Float($0) }
        let out = WhisperModel.padOrTrim(clip, to: 500)
        #expect(out.count == 500)
        #expect(out == clip)
    }

    @Test("WhisperModel.whisperWindowSamples — 30 s × 16 kHz = 480,000")
    func whisperWindowSamplesConstant() {
        #expect(WhisperModel.whisperWindowSamples == 480_000)
    }
}
