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
// Unit tests for the MossFormer2SE family (speech-enhancement, audio in → audio out).
//
// Covers:
//   • Config decoding — from canonical and minimal config.json payloads.
//   • Registry detection — model_type path and structural path.
//   • Negative detection — unrelated configs are not matched.
//   • Capability set verification.
//   • AudioModelRegistry routes correctly.
//
// These tests are fast and offline — no checkpoint weights are loaded.

import Foundation
import Testing
@testable import FFAI

@Suite("MossFormer2SE")
struct MossFormer2SETests {

    // ─── Config decoding ──────────────────────────────────────────────

    @Test("MossFormer2SEConfig — decodes from canonical model_type")
    func configDecodesFromModelType() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "mossformer2_se",
            raw: [
                "model_type": "mossformer2_se",
                "sample_rate": 48_000,
                "win_len": 1920,
                "win_inc": 384,
                "fft_len": 1920,
                "num_mels": 60,
                "win_type": "hamming",
                "preemphasis": 0.97,
                "in_channels": 180,
                "out_channels": 512,
                "out_channels_final": 961,
                "num_blocks": 24,
            ])
        let se = MossFormer2SEConfig.from(config)
        #expect(se.modelType == "mossformer2_se")
        #expect(se.sampleRate == 48_000)
        #expect(se.winLen == 1920)
        #expect(se.winInc == 384)
        #expect(se.fftLen == 1920)
        #expect(se.numMels == 60)
        #expect(se.winType == "hamming")
        #expect(abs(se.preemphasis - 0.97) < 1e-5)
        #expect(se.inChannels == 180)
        #expect(se.outChannels == 512)
        #expect(se.outChannelsFinal == 961)
        #expect(se.numBlocks == 24)
    }

    @Test("MossFormer2SEConfig — decodes from minimal config with defaults")
    func configDecodesFromMinimal() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "mossformer2_se",
            raw: ["model_type": "mossformer2_se"])
        let se = MossFormer2SEConfig.from(config)
        // All defaults should match published checkpoint values.
        #expect(se.sampleRate == 48_000)
        #expect(se.winLen == 1920)
        #expect(se.winInc == 384)
        #expect(se.fftLen == 1920)
        #expect(se.numMels == 60)
        #expect(se.inChannels == 180)
        #expect(se.outChannels == 512)
        #expect(se.outChannelsFinal == 961)
        #expect(se.numBlocks == 24)
    }

    @Test("MossFormer2SEConfig — decodes alternative mossformer2se model_type")
    func configDecodesAlternativeModelType() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "mossformer2se",
            raw: ["model_type": "mossformer2se", "sample_rate": 16_000])
        let se = MossFormer2SEConfig.from(config)
        #expect(se.sampleRate == 16_000)
    }

    @Test("MossFormer2SEConfig — default init matches published checkpoint")
    func defaultInitMatchesCheckpoint() {
        let se = MossFormer2SEConfig()
        #expect(se.sampleRate == 48_000)
        #expect(se.inChannels == 180)
        #expect(se.outChannels == 512)
        #expect(se.outChannelsFinal == 961)
        #expect(se.numBlocks == 24)
        #expect(3 * se.numMels == se.inChannels)
    }

    // ─── Registry detection ───────────────────────────────────────────

    @Test("MossFormer2SEModel.handles — detects from model_type mossformer2_se")
    func handlesModelType() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "mossformer2_se",
            raw: ["model_type": "mossformer2_se"])
        #expect(MossFormer2SEModel.handles(config))
    }

    @Test("MossFormer2SEModel.handles — detects from model_type mossformer2se (no underscore)")
    func handlesModelTypeNoUnderscore() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "mossformer2se",
            raw: ["model_type": "mossformer2se"])
        #expect(MossFormer2SEModel.handles(config))
    }

    @Test("MossFormer2SEModel.handles — detects structurally (in_channels + out_channels_final)")
    func handlesStructural() {
        let config = ModelConfig(
            architecture: nil,
            modelType: nil,
            raw: [
                "in_channels": 180,
                "out_channels_final": 961,
                "sample_rate": 48_000,
            ])
        #expect(MossFormer2SEModel.handles(config))
    }

    @Test("MossFormer2SEModel.handles — structural detection rejected when hidden_size present")
    func structuralRejectedForLLM() {
        // A model with hidden_size should not be misidentified as SE.
        let config = ModelConfig(
            architecture: nil,
            modelType: nil,
            raw: [
                "in_channels": 180,
                "out_channels_final": 961,
                "hidden_size": 4096,
            ])
        #expect(!MossFormer2SEModel.handles(config))
    }

    @Test("MossFormer2SEModel.handles — does not detect Llama")
    func doesNotDetectLlama() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM",
            modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 4_096])
        #expect(!MossFormer2SEModel.handles(config))
    }

    @Test("MossFormer2SEModel.handles — does not detect empty config")
    func doesNotDetectEmptyConfig() {
        let config = ModelConfig(architecture: nil, modelType: nil, raw: [:])
        #expect(!MossFormer2SEModel.handles(config))
    }

    // ─── AudioModelRegistry integration ──────────────────────────────

    @Test("AudioModelRegistry — handles MossFormer2SE from model_type")
    func registryHandlesMossFormer2SE() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "mossformer2_se",
            raw: ["model_type": "mossformer2_se"])
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.speechToSpeech)
    }

    @Test("AudioModelRegistry — does not handle unrelated models")
    func registryDoesNotHandleQwen3() {
        let config = ModelConfig(
            architecture: "Qwen3ForCausalLM",
            modelType: "qwen3",
            raw: ["model_type": "qwen3", "hidden_size": 4_096])
        #expect(!AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == nil)
    }

    // ─── Capability set ───────────────────────────────────────────────

    @Test("Capability.speechToSpeech contains audioIn + audioOut")
    func speechToSpeechCapability() {
        let caps = Capability.speechToSpeech
        #expect(caps.contains(.audioIn))
        #expect(caps.contains(.audioOut))
        #expect(caps.count == 2)
    }

    // ─── Error paths ──────────────────────────────────────────────────

    @Test("MossFormer2SEError.invalidInput description")
    func errorInvalidInput() {
        let err = MossFormer2SEError.invalidInput("waveform is empty")
        #expect(err.description.contains("waveform"))
    }

    @Test("MossFormer2SEError.missingWeight description")
    func errorMissingWeight() {
        let err = MossFormer2SEError.missingWeight("model.mossformer.norm.weight")
        #expect(err.description.contains("model.mossformer.norm.weight"))
    }

    @Test("MossFormer2SEError.noSafetensorsFound description")
    func errorNoSafetensors() {
        let url = URL(fileURLWithPath: "/tmp/fake-dir")
        let err = MossFormer2SEError.noSafetensorsFound(url)
        #expect(err.description.contains("fake-dir"))
    }

    @Test("MossFormer2SEError.missingConfig description")
    func errorMissingConfig() {
        let err = MossFormer2SEError.missingConfig("sample_rate")
        #expect(err.description.contains("sample_rate"))
    }

    // ─── Checkpoint cache detection ───────────────────────────────────

    @Test("MossFormer2SEModel — cached checkpoint config.json decodes correctly")
    func cachedConfigDecodes() {
        let hfRoot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        let slug = "models--starkdmi--MossFormer2-SE-fp16"
        let snapshots = hfRoot.appendingPathComponent(slug)
            .appendingPathComponent("snapshots")
        guard let subs = try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil),
              let snap = subs.first
        else {
            // No checkpoint on disk — skip.
            print("MossFormer2SETests: cached checkpoint not found, skipping")
            return
        }
        guard let modelConfig = try? ModelConfig.load(from: snap) else {
            print("MossFormer2SETests: config.json load failed, skipping")
            return
        }
        let se = MossFormer2SEConfig.from(modelConfig)
        #expect(se.sampleRate == 48_000)
        #expect(se.inChannels == 180)
        #expect(se.outChannels == 512)
        #expect(se.outChannelsFinal == 961)
        #expect(MossFormer2SEModel.handles(modelConfig))
        print("[MossFormer2SE] Cached config decoded: sr=\(se.sampleRate), "
              + "inC=\(se.inChannels), outC=\(se.outChannels), "
              + "outCF=\(se.outChannelsFinal), nBlocks=\(se.numBlocks)")
    }
}
