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
// SenseVoiceTests — unit tests for SenseVoice config parsing, family
// detection, registry routing, and pure-CPU helpers in the Kaldi-style
// FBANK front-end. All tests run offline, no checkpoint required.
//
// Validates:
//   * SenseVoiceConfig.from(_:) parses nested encoder_conf / frontend_conf.
//   * SenseVoiceModel.handles(_:) accepts both model_types and structural
//     fallback (tp_blocks marker).
//   * AudioModelRegistry routes SenseVoice configs to Capability.speechToText.
//   * SenseVoiceFrontEnd CPU helpers (Hamming window, nextPowerOfTwo,
//     low-frame-rate stacking) return numerically-sane outputs.

import Foundation
import Testing
@testable import FFAI

@Suite("SenseVoice")
struct SenseVoiceTests {

    // ─── Config parsing ──────────────────────────────────────────────────

    @Test("SenseVoiceConfig.from — decodes published SenseVoiceSmall defaults")
    func configDecodesDefaults() {
        let raw: [String: Any] = [
            "model_type": "sensevoice",
            "vocab_size": 25_055,
            "encoder_conf": [
                "output_size": 512,
                "attention_heads": 4,
                "linear_units": 2048,
                "num_blocks": 50,
                "tp_blocks": 20,
                "kernel_size": 11,
            ] as [String: Any],
            "frontend_conf": [
                "fs": 16_000,
                "n_mels": 80,
                "lfr_m": 7,
                "lfr_n": 6,
            ] as [String: Any],
        ]
        let config = ModelConfig(architecture: nil, modelType: "sensevoice", raw: raw)
        let sc = SenseVoiceConfig.from(config)
        #expect(sc != nil)
        #expect(sc?.vocab == 25_055)
        #expect(sc?.hidden == 512)
        #expect(sc?.heads == 4)
        #expect(sc?.intermediate == 2048)
        #expect(sc?.numBlocks == 50)
        #expect(sc?.tpBlocks == 20)
        #expect(sc?.fsmnKernel == 11)
        #expect(sc?.frontEnd.sampleRate == 16_000)
        #expect(sc?.frontEnd.nMels == 80)
        #expect(sc?.frontEnd.lfrM == 7)
        #expect(sc?.frontEnd.lfrN == 6)
    }

    @Test("SenseVoiceConfig.from — fills in defaults for missing fields")
    func configFillsDefaults() {
        let raw: [String: Any] = ["vocab_size": 25_055]
        let config = ModelConfig(architecture: nil, modelType: "sensevoice", raw: raw)
        let sc = SenseVoiceConfig.from(config)
        #expect(sc != nil)
        #expect(sc?.hidden == 512)
        #expect(sc?.heads == 4)
        #expect(sc?.frontEnd.sampleRate == 16_000)
        #expect(sc?.frontEnd.lfrM == 7)
    }

    @Test("SenseVoiceConfig.from — returns nil when vocab_size absent")
    func configRequiresVocabSize() {
        let config = ModelConfig(architecture: nil, modelType: "sensevoice",
                                 raw: [:])
        #expect(SenseVoiceConfig.from(config) == nil)
    }

    @Test("SenseVoiceConfig.headDim — equals hidden / heads")
    func configHeadDim() {
        let frontEnd = SenseVoiceFrontEndConfig()
        let sc = SenseVoiceConfig(
            vocab: 25_055, inputSize: 560, hidden: 512, heads: 4,
            intermediate: 2048, numBlocks: 50, tpBlocks: 20,
            fsmnKernel: 11, fsmnShift: 0, frontEnd: frontEnd)
        #expect(sc.headDim == 128)
    }

    @Test("SenseVoiceFrontEndConfig — winLength + hopLength derive from ms × sampleRate")
    func frontEndSampleMath() {
        let fe = SenseVoiceFrontEndConfig()
        // Defaults: 25 ms × 16 kHz = 400 samples, 10 ms × 16 kHz = 160 samples.
        #expect(fe.winLength == 400)
        #expect(fe.hopLength == 160)
    }

    // ─── Registry detection ──────────────────────────────────────────────

    @Test("SenseVoiceModel.modelTypes — contains canonical entries")
    func modelTypesContents() {
        let types = SenseVoiceModel.modelTypes
        #expect(types.contains("sensevoice"))
        #expect(types.contains("sense_voice"))
    }

    @Test("SenseVoiceModel.architectures — contains canonical entries")
    func architecturesContents() {
        let archs = SenseVoiceModel.architectures
        #expect(archs.contains("SenseVoiceSmall"))
        #expect(archs.contains("SenseVoice"))
    }

    @Test("SenseVoiceModel.handles — true for sensevoice model_type")
    func handlesByModelType() {
        let config = ModelConfig(architecture: nil, modelType: "sensevoice",
                                 raw: ["model_type": "sensevoice"])
        #expect(SenseVoiceModel.handles(config))
    }

    @Test("SenseVoiceModel.handles — true for SenseVoiceSmall architecture")
    func handlesByArchitecture() {
        let config = ModelConfig(architecture: "SenseVoiceSmall",
                                 modelType: nil, raw: [:])
        #expect(SenseVoiceModel.handles(config))
    }

    @Test("SenseVoiceModel.handles — structural fallback via tp_blocks marker")
    func handlesStructural() {
        let raw: [String: Any] = [
            "encoder_conf": ["tp_blocks": 20] as [String: Any],
        ]
        let config = ModelConfig(architecture: nil, modelType: nil, raw: raw)
        #expect(SenseVoiceModel.handles(config))
    }

    @Test("SenseVoiceModel.handles — false for unrelated text model")
    func handlesFalseForTextModel() {
        let config = ModelConfig(architecture: "LlamaForCausalLM",
                                 modelType: "llama",
                                 raw: ["model_type": "llama"])
        #expect(!SenseVoiceModel.handles(config))
    }

    // ─── AudioModelRegistry routing ──────────────────────────────────────

    @Test("AudioModelRegistry.capabilities — SenseVoice maps to speechToText")
    func registryCapability() {
        let config = ModelConfig(architecture: nil, modelType: "sensevoice",
                                 raw: ["model_type": "sensevoice"])
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.speechToText)
    }

    // ─── SenseVoiceFrontEnd CPU helpers ─────────────────────────────────

    @Test("SenseVoiceFrontEnd.hammingWindow — length matches and is symmetric")
    func hammingWindow() {
        let w = SenseVoiceFrontEnd.hammingWindow(400)
        #expect(w.count == 400)
        // Hamming endpoints (n-1 divisor) are 0.54 - 0.46 = 0.08.
        #expect(abs(w[0] - 0.08) < 1e-4)
        #expect(abs(w[399] - 0.08) < 1e-4)
        // Centre is at 1.0.
        #expect(abs(w[200] - 1.0) < 1e-2)
    }

    @Test("SenseVoiceFrontEnd.nextPowerOfTwo — small cases")
    func nextPowerOfTwo() {
        #expect(SenseVoiceFrontEnd.nextPowerOfTwo(1) == 1)
        #expect(SenseVoiceFrontEnd.nextPowerOfTwo(2) == 2)
        #expect(SenseVoiceFrontEnd.nextPowerOfTwo(3) == 4)
        #expect(SenseVoiceFrontEnd.nextPowerOfTwo(400) == 512)
        #expect(SenseVoiceFrontEnd.nextPowerOfTwo(513) == 1024)
    }

    @Test("SenseVoiceFrontEnd.lowFrameRate — empty FBANK yields empty output")
    func lowFrameRateEmpty() {
        let out = SenseVoiceFrontEnd.lowFrameRate([], nMels: 80, lfrM: 7, lfrN: 6)
        #expect(out.isEmpty)
    }

    @Test("SenseVoiceFrontEnd.lowFrameRate — produces ceil(nFrames / lfrN) rows")
    func lowFrameRateShape() {
        // 12 frames, lfrM=7, lfrN=6, nMels=2 → ceil(12/6) = 2 output rows,
        // each containing 7 * 2 = 14 floats.
        let nMels = 2
        let frames = 12
        let fbank = [Float](repeating: 1.0, count: frames * nMels)
        let out = SenseVoiceFrontEnd.lowFrameRate(fbank, nMels: nMels,
                                                  lfrM: 7, lfrN: 6)
        #expect(out.count == 2 * 7 * nMels)
    }

    // ─── Constants ───────────────────────────────────────────────────────

    @Test("SenseVoiceModel — blankToken and queryPrefixLength match upstream")
    func constantsMatchUpstream() {
        #expect(SenseVoiceModel.blankToken == 0)
        #expect(SenseVoiceModel.queryPrefixLength == 4)
    }
}
