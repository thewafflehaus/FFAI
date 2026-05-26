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

// Unit tests for the LFMAudio family — config parsing, registry
// detection, and the CPU-side preprocessor + encoder geometry. All tests
// use synthetic data and run fully offline.
@Suite("LFMAudio")
struct LFMAudioTests {

    // ─── Config parsing ──────────────────────────────────────────────

    /// Canonical `config.json` shape for the LFM2.5-Audio-1.5B checkpoint.
    private func makeRawConfig() -> [String: Any] {
        [
            "model_type": "lfm_audio",
            "architectures": ["Lfm2AudioForConditionalGeneration"],
            "sample_rate": 24_000,
            "codebooks": 8,
            "audio_vocab_size": 2049,
            "adapter_hidden_dims": [2048],
            "adapter_use_layer_norm": true,
            "preprocessor": [
                "sample_rate": 16_000,
                "features": 128,
                "n_fft": 512,
                "window_size": 0.025,
                "window_stride": 0.01,
                "preemph": 0.97,
                "dither": 1e-5,
                "normalize": "per_feature",
            ] as [String: Any],
            "encoder": [
                "feat_in": 128,
                "n_layers": 17,
                "d_model": 512,
                "n_heads": 8,
                "ff_expansion_factor": 4,
                "subsampling_factor": 8,
                "subsampling_conv_channels": 256,
                "pos_emb_max_len": 5000,
                "conv_kernel_size": 9,
            ] as [String: Any],
            "lfm": [
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "num_attention_heads": 16,
                "vocab_size": 32_000,
            ] as [String: Any],
        ]
    }

    @Test("LFMAudioConfig — parses all fields from a canonical config.json")
    func configParsesAllFields() throws {
        let raw = makeRawConfig()
        let modelConfig = ModelConfig(
            architecture: "Lfm2AudioForConditionalGeneration",
            modelType: "lfm_audio", raw: raw)
        let cfg = try LFMAudioConfig.from(modelConfig)

        // Top-level
        #expect(cfg.modelType == "lfm_audio")
        #expect(cfg.sampleRate == 24_000)
        #expect(cfg.codebooks == 8)
        #expect(cfg.audioVocabSize == 2049)
        #expect(cfg.adapterHiddenDims == [2048])
        #expect(cfg.adapterUseLayerNorm == true)
        #expect(cfg.lfmHidden == 2048)

        // Preprocessor
        #expect(cfg.preprocessor.sampleRate == 16_000)
        #expect(cfg.preprocessor.nMels == 128)
        #expect(cfg.preprocessor.nFFT == 512)
        #expect(cfg.preprocessor.winLength == 400)  // 0.025 * 16000
        #expect(cfg.preprocessor.hopLength == 160)  // 0.01  * 16000
        #expect(cfg.preprocessor.preemph == 0.97)
        #expect(cfg.preprocessor.normalise == "per_feature")

        // ConformerEncoder
        #expect(cfg.encoder.featIn == 128)
        #expect(cfg.encoder.nLayers == 17)
        #expect(cfg.encoder.dModel == 512)
        #expect(cfg.encoder.nHeads == 8)
        #expect(cfg.encoder.ffExpansionFactor == 4)
        #expect(cfg.encoder.subsamplingFactor == 8)
        #expect(cfg.encoder.subsamplingConvChannels == 256)
        #expect(cfg.encoder.posEmbMaxLen == 5000)
        #expect(cfg.encoder.convKernelSize == 9)

        // Derived properties
        #expect(cfg.encoder.headDim == 64)  // 512 / 8
        #expect(cfg.encoder.ffHidden == 2048)  // 512 * 4
    }

    @Test("LFMAudioConfig — falls back to defaults for omitted fields")
    func configFallsBackToDefaults() throws {
        // Minimal valid config — only the nested blocks required for parsing.
        let raw: [String: Any] = [
            "model_type": "lfm_audio",
            "preprocessor": ["sample_rate": 16_000] as [String: Any],
            "encoder": [:] as [String: Any],
            "lfm": ["hidden_size": 2048] as [String: Any],
        ]
        let modelConfig = ModelConfig(
            architecture: nil, modelType: "lfm_audio", raw: raw)
        let cfg = try LFMAudioConfig.from(modelConfig)

        // All defaults must match the LFM2.5-Audio-1.5B published values.
        #expect(cfg.encoder.nLayers == 17)
        #expect(cfg.encoder.dModel == 512)
        #expect(cfg.encoder.nHeads == 8)
        #expect(cfg.preprocessor.nMels == 128)
        #expect(cfg.lfmHidden == 2048)
    }

    @Test("LFMAudioConfig — throws missingConfig when preprocessor block absent")
    func configThrowsMissingPreprocessor() {
        let raw: [String: Any] = [
            "model_type": "lfm_audio",
            "encoder": [:] as [String: Any],
            "lfm": ["hidden_size": 2048] as [String: Any],
        ]
        let modelConfig = ModelConfig(
            architecture: nil, modelType: "lfm_audio", raw: raw)
        #expect(throws: LFMAudioError.self) {
            _ = try LFMAudioConfig.from(modelConfig)
        }
    }

    @Test("LFMAudioConfig — throws missingConfig when lfm block absent")
    func configThrowsMissingLFM() {
        let raw: [String: Any] = [
            "model_type": "lfm_audio",
            "preprocessor": ["sample_rate": 16_000] as [String: Any],
            "encoder": [:] as [String: Any],
        ]
        let modelConfig = ModelConfig(
            architecture: nil, modelType: "lfm_audio", raw: raw)
        #expect(throws: LFMAudioError.self) {
            _ = try LFMAudioConfig.from(modelConfig)
        }
    }

    // ─── Registry detection ──────────────────────────────────────────

    @Test("LFMAudioModel.handles — detects by model_type")
    func handlesModelType() {
        let config = ModelConfig(
            architecture: nil, modelType: "lfm_audio",
            raw: makeRawConfig())
        #expect(LFMAudioModel.handles(config))
    }

    @Test("LFMAudioModel.handles — detects by architecture string")
    func handlesArchitecture() {
        var raw = makeRawConfig()
        raw["model_type"] = "unknown"  // force architecture path
        let config = ModelConfig(
            architecture: "Lfm2AudioForConditionalGeneration",
            modelType: "unknown", raw: raw)
        #expect(LFMAudioModel.handles(config))
    }

    @Test("LFMAudioModel.handles — rejects unrelated families")
    func handlesRejectsUnrelated() {
        let whisper = ModelConfig(
            architecture: "WhisperForConditionalGeneration",
            modelType: "whisper",
            raw: ["model_type": "whisper"])
        #expect(!LFMAudioModel.handles(whisper))

        let llama = ModelConfig(
            architecture: "LlamaForCausalLM",
            modelType: "llama",
            raw: ["model_type": "llama"])
        #expect(!LFMAudioModel.handles(llama))
    }

    @Test("AudioModelRegistry — detects LFMAudio and reports omniAudio")
    func registryDetectsLFMAudio() {
        let config = ModelConfig(
            architecture: "Lfm2AudioForConditionalGeneration",
            modelType: "lfm_audio",
            raw: makeRawConfig())
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.omniAudio)
    }

    @Test("AudioModelRegistry — LFMAudio does not shadow Whisper or QwenOmni")
    func registryNoShadowing() {
        let whisper = ModelConfig(
            architecture: "WhisperForConditionalGeneration",
            modelType: "whisper",
            raw: [
                "d_model": 512, "encoder_layers": 6,
                "encoder_attention_heads": 8, "decoder_layers": 6,
                "decoder_attention_heads": 8, "vocab_size": 51865,
            ])
        #expect(AudioModelRegistry.handles(whisper))
        #expect(
            AudioModelRegistry.capabilities(for: whisper)
                == Capability.speechToText)

        let qwenOmni = ModelConfig(
            architecture: "Qwen2_5OmniForConditionalGeneration",
            modelType: "qwen2_5_omni",
            raw: [
                "model_type": "qwen2_5_omni",
                "audio_config": [
                    "d_model": 1280, "encoder_layers": 32,
                    "encoder_attention_heads": 20,
                    "num_mel_bins": 128,
                ],
                "text_config": ["hidden_size": 3584],
            ])
        #expect(AudioModelRegistry.handles(qwenOmni))
        // QwenOmni is checked before LFMAudio in the registry.
        #expect(
            AudioModelRegistry.capabilities(for: qwenOmni)
                == Capability.omniAudio)
        #expect(!LFMAudioModel.handles(qwenOmni))
    }

    // ─── Preprocessor geometry ───────────────────────────────────────

    @Test("LFMAudioPreprocessorConfig — window lengths derive from rates")
    func preprocessorWindowLengths() throws {
        let raw: [String: Any] = [
            "model_type": "lfm_audio",
            "preprocessor": [
                "sample_rate": 16_000,
                "features": 80,
                "n_fft": 400,
                "window_size": 0.025,  // 400 samples
                "window_stride": 0.01,  // 160 samples
                "preemph": 0.97,
                "dither": 0.0,
                "normalize": "per_feature",
            ] as [String: Any],
            "encoder": [:] as [String: Any],
            "lfm": ["hidden_size": 2048] as [String: Any],
        ]
        let cfg = try LFMAudioConfig.from(
            ModelConfig(architecture: nil, modelType: "lfm_audio", raw: raw))
        #expect(cfg.preprocessor.winLength == 400)
        #expect(cfg.preprocessor.hopLength == 160)
        #expect(cfg.preprocessor.nMels == 80)
    }
}
