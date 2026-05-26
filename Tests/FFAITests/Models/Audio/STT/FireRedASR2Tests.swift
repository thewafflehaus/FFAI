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
// FireRedASR2Tests — unit tests for FireRedASR2 config parsing, tokenizer,
// audio front-end, and registry detection.
// No real checkpoint required; all tests run offline.
//
// Validates:
//   * FireRedASR2Config.from(_:) parses config shapes correctly.
//   * FireRedASR2Model.handles(_:) fires on the expected config signatures.
//   * AudioModelRegistry routes FireRedASR2 configs to Capability.speechToText.
//   * FireRedASR2Tokenizer.decode applies SentencePiece and special-token rules.
//   * FireRedASR2Model.kaldiFbank produces finite, correctly-shaped features.

import Foundation
import Testing
@testable import FFAI

@Suite("FireRedASR2")
struct FireRedASR2Tests {

    // ─── Config parsing ──────────────────────────────────────────────────

    @Test("FireRedASR2Config.from — defaults for minimal model_type config")
    func configDefaultsFromModelType() {
        let config = ModelConfig(
            architecture: nil, modelType: "fireredasr2",
            raw: ["model_type": "fireredasr2"])
        let cfg = FireRedASR2Config.from(config)
        #expect(cfg != nil)

        // Published AED-large defaults.
        #expect(cfg?.idim    == 80)
        #expect(cfg?.odim    == 8667)
        #expect(cfg?.dModel  == 1280)
        #expect(cfg?.sosID   == 3)
        #expect(cfg?.eosID   == 4)
        #expect(cfg?.padID   == 2)
        #expect(cfg?.blankID == 0)

        // Encoder defaults.
        #expect(cfg?.encoder.nLayers    == 16)
        #expect(cfg?.encoder.nHead      == 20)
        #expect(cfg?.encoder.dModel     == 1280)
        #expect(cfg?.encoder.kernelSize == 33)
        #expect(cfg?.encoder.peMaxlen   == 5000)

        // Decoder defaults.
        #expect(cfg?.decoder.nLayers  == 16)
        #expect(cfg?.decoder.nHead    == 20)
        #expect(cfg?.decoder.dModel   == 1280)
        #expect(cfg?.decoder.peMaxlen == 5000)
    }

    @Test("FireRedASR2Config.from — parses explicit nested encoder/decoder")
    func configParsesExplicitFields() {
        let config = ModelConfig(
            architecture: "FireRedASR2ForConditionalGeneration",
            modelType: "fireredasr2",
            raw: [
                "model_type": "fireredasr2",
                "idim":   80,
                "odim":   8667,
                "sos_id": 3,
                "eos_id": 4,
                "pad_id": 2,
                "blank_id": 0,
                "encoder": [
                    "n_layers":    16,
                    "n_head":      20,
                    "d_model":     1280,
                    "kernel_size": 33,
                    "pe_maxlen":   5000
                ] as [String: Any],
                "decoder": [
                    "n_layers":  16,
                    "n_head":    20,
                    "d_model":   1280,
                    "pe_maxlen": 5000
                ] as [String: Any]
            ])
        let cfg = FireRedASR2Config.from(config)
        #expect(cfg != nil)
        #expect(cfg?.encoder.nLayers == 16)
        #expect(cfg?.encoder.nHead   == 20)
        #expect(cfg?.encoder.dModel  == 1280)
        #expect(cfg?.decoder.nLayers == 16)
        #expect(cfg?.odim            == 8667)
    }

    @Test("FireRedASR2Config.from — detects by architecture string")
    func configDetectsByArchitecture() {
        // Some mlx-community conversions omit model_type but keep architecture.
        let config = ModelConfig(
            architecture: "FireRedASR2ForConditionalGeneration",
            modelType: nil,
            raw: ["architectures": ["FireRedASR2ForConditionalGeneration"]])
        let cfg = FireRedASR2Config.from(config)
        #expect(cfg != nil)
    }

    @Test("FireRedASR2Config.from — returns nil for text-only config")
    func configReturnsNilForTextOnly() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 4096])
        #expect(FireRedASR2Config.from(config) == nil)
    }

    // ─── Registry detection ──────────────────────────────────────────────

    @Test("FireRedASR2Model.handles — true for model_type fireredasr2")
    func handlesFireRedASR2ByModelType() {
        let config = ModelConfig(
            architecture: nil, modelType: "fireredasr2",
            raw: ["model_type": "fireredasr2"])
        #expect(FireRedASR2Model.handles(config))
    }

    @Test("FireRedASR2Model.handles — true for FireRedASR2ForConditionalGeneration architecture")
    func handlesFireRedASR2ByArchitecture() {
        let config = ModelConfig(
            architecture: "FireRedASR2ForConditionalGeneration",
            modelType: nil,
            raw: [:])
        #expect(FireRedASR2Model.handles(config))
    }

    @Test("FireRedASR2Model.handles — false for unrelated model")
    func handlesFalseForTextModel() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama"])
        #expect(!FireRedASR2Model.handles(config))
    }

    @Test("AudioModelRegistry.handles — true for FireRedASR2 config")
    func registryHandlesFireRedASR2() {
        let config = ModelConfig(
            architecture: "FireRedASR2ForConditionalGeneration",
            modelType: "fireredasr2",
            raw: ["model_type": "fireredasr2"])
        #expect(AudioModelRegistry.handles(config))
    }

    @Test("AudioModelRegistry.capabilities — FireRedASR2 maps to speechToText")
    func registryCapabilitySpeechToText() {
        let config = ModelConfig(
            architecture: "FireRedASR2ForConditionalGeneration",
            modelType: "fireredasr2",
            raw: ["model_type": "fireredasr2"])
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.speechToText)
    }

    @Test("AudioModelRegistry.handles — false for text-only model")
    func registryFalseForTextModel() {
        let config = ModelConfig(
            architecture: nil, modelType: "qwen3",
            raw: ["model_type": "qwen3", "hidden_size": 2048])
        #expect(!AudioModelRegistry.handles(config))
    }

    // ─── Tokenizer ───────────────────────────────────────────────────────

    @Test("FireRedASR2Tokenizer.decode — joins pieces and lowercases")
    func tokenizerJoinsAndLowercases() {
        let vocab = ["Hello", " ", "world", "!"]
        let tok = FireRedASR2Tokenizer(vocabulary: vocab)
        let text = tok.decode(tokenIds: [0, 1, 2, 3])
        // decode lowercases the output
        #expect(text == "hello world!")
    }

    @Test("FireRedASR2Tokenizer.decode — replaces SentencePiece U+2581 with space")
    func tokenizerSentencePieceReplacement() {
        let vocab = ["\u{2581}Hello", "\u{2581}world"]
        let tok = FireRedASR2Tokenizer(vocabulary: vocab)
        let text = tok.decode(tokenIds: [0, 1])
        // Leading space is stripped; ▁ between words becomes space.
        #expect(text == "hello world")
    }

    @Test("FireRedASR2Tokenizer.decode — strips <blank> and <sil> tokens")
    func tokenizerStripsSpecialTokens() {
        let vocab = ["<blank>", "hello", "<sil>", " ", "world"]
        let tok = FireRedASR2Tokenizer(vocabulary: vocab)
        let text = tok.decode(tokenIds: [0, 1, 2, 3, 4])
        #expect(text == "hello world")
    }

    @Test("FireRedASR2Tokenizer.decode — skips out-of-range token ids")
    func tokenizerSkipsOutOfRange() {
        let vocab = ["a", "b"]
        let tok = FireRedASR2Tokenizer(vocabulary: vocab)
        let text = tok.decode(tokenIds: [0, 99, 1])
        #expect(text == "ab")
    }

    @Test("FireRedASR2Tokenizer.decode — empty token id list produces empty string")
    func tokenizerEmptyIds() {
        let vocab = ["a", "b", "c"]
        let tok = FireRedASR2Tokenizer(vocabulary: vocab)
        #expect(tok.decode(tokenIds: []).isEmpty)
    }

    // ─── Audio front-end ─────────────────────────────────────────────────

    @Test("FireRedASR2 Kaldi fbank — finite, correctly-shaped output")
    func kaldiFbankFiniteShape() {
        // 1 second of 440 Hz tone at 16 kHz.
        let sr = 16_000
        var wave = [Float](repeating: 0, count: sr)
        for i in 0..<sr {
            wave[i] = 0.3 * sin(2.0 * Float.pi * 440.0 * Float(i) / Float(sr))
        }

        let idim = 80
        let feats = FireRedASR2Model.kaldiFbank(waveform: wave, idim: idim)

        // Frame count: (n - 400) / 160 + 1 ≈ 98 frames for 16000 samples.
        let nFrames = feats.count / idim
        #expect(nFrames > 0, "kaldiFbank produced zero frames")
        #expect(feats.count == nFrames * idim)
        #expect(feats.allSatisfy { $0.isFinite },
                "Kaldi fbank front-end produced non-finite values")
    }

    @Test("FireRedASR2 Kaldi fbank — empty waveform returns empty")
    func kaldiFbankEmpty() {
        let feats = FireRedASR2Model.kaldiFbank(waveform: [], idim: 80)
        #expect(feats.isEmpty)
    }

    @Test("FireRedASR2 Kaldi fbank — pre-normalised PCM (peak ≤ 1) is rescaled")
    func kaldiFbankNormalisedPCM() {
        // Normalised PCM in [-1, 1] — front-end should rescale to int16 range.
        let sr = 16_000
        var wave = [Float](repeating: 0, count: sr)
        for i in 0..<sr {
            wave[i] = 0.1 * sin(2.0 * Float.pi * 300.0 * Float(i) / Float(sr))
        }
        let feats = FireRedASR2Model.kaldiFbank(waveform: wave, idim: 80)
        #expect(!feats.isEmpty)
        #expect(feats.allSatisfy { $0.isFinite })
    }
}
