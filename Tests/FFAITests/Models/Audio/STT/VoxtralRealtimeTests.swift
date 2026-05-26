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

// Unit tests for the VoxtralRealtime speech-to-text family.
// Exercises config parsing and registry detection using synthetic
// `ModelConfig` objects — no real checkpoint required.
@Suite("VoxtralRealtime")
struct VoxtralRealtimeTests {

    // ─── Registry detection ──────────────────────────────────────────

    @Test("AudioModelRegistry — detects VoxtralRealtime from model_type")
    func registryDetectsByModelType() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "voxtral_realtime",
            raw: [
                "model_type": "voxtral_realtime",
                "encoder_args": [
                    "dim": 1280, "n_layers": 32, "n_heads": 32,
                    "head_dim": 64, "hidden_dim": 5120, "n_kv_heads": 32,
                    "sliding_window": 750, "downsample_factor": 4,
                    "audio_encoding_args": [
                        "sampling_rate": 16000, "num_mel_bins": 128,
                        "hop_length": 160, "window_size": 400,
                    ],
                ] as [String: Any],
                "decoder": [
                    "dim": 3072, "n_layers": 26, "n_heads": 32,
                    "n_kv_heads": 8, "head_dim": 128,
                    "hidden_dim": 9216, "vocab_size": 131072,
                    "tied_embeddings": true,
                    "ada_rms_norm_t_cond": true,
                    "ada_rms_norm_t_cond_dim": 32,
                ] as [String: Any],
            ])
        #expect(VoxtralRealtimeModel.handles(config))
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.speechToText)
    }

    @Test("AudioModelRegistry — non-voxtral config is not detected")
    func registryIgnoresOtherModelTypes() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM",
            modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 4096])
        #expect(!VoxtralRealtimeModel.handles(config))
    }

    @Test("AudioModelRegistry — VoxtralRealtime is speechToText capability")
    func voxtralCapabilityIsSpeechToText() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "voxtral_realtime",
            raw: ["model_type": "voxtral_realtime"])
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.speechToText)
    }

    // ─── Config parsing ──────────────────────────────────────────────

    @Test("VoxtralRealtimeConfig — decodes full config from JSON dict")
    func configDecodesFullConfig() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "voxtral_realtime",
            raw: [
                "model_type": "voxtral_realtime",
                "encoder_args": [
                    "dim": 1280,
                    "n_layers": 32,
                    "n_heads": 32,
                    "head_dim": 64,
                    "hidden_dim": 5120,
                    "n_kv_heads": 32,
                    "norm_eps": 1e-5,
                    "rope_theta": 1000000.0,
                    "sliding_window": 750,
                    "causal": true,
                    "use_biases": true,
                    "downsample_factor": 4,
                    "audio_encoding_args": [
                        "sampling_rate": 16000,
                        "frame_rate": 12.5,
                        "num_mel_bins": 128,
                        "hop_length": 160,
                        "window_size": 400,
                        "global_log_mel_max": 1.5,
                    ] as [String: Any],
                ] as [String: Any],
                "decoder": [
                    "dim": 3072,
                    "n_layers": 26,
                    "n_heads": 32,
                    "n_kv_heads": 8,
                    "head_dim": 128,
                    "hidden_dim": 9216,
                    "vocab_size": 131072,
                    "norm_eps": 1e-5,
                    "rope_theta": 1000000.0,
                    "sliding_window": 8192,
                    "tied_embeddings": true,
                    "ada_rms_norm_t_cond": true,
                    "ada_rms_norm_t_cond_dim": 32,
                ] as [String: Any],
                "transcription_delay_ms": 480,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "streaming_pad_token_id": 32,
                "n_left_pad_tokens": 32,
            ])

        let vc = VoxtralRealtimeConfig.from(config)
        #expect(vc != nil)

        // Encoder hyper-parameters.
        #expect(vc?.encoderConfig.dim == 1280)
        #expect(vc?.encoderConfig.nLayers == 32)
        #expect(vc?.encoderConfig.nHeads == 32)
        #expect(vc?.encoderConfig.headDim == 64)
        #expect(vc?.encoderConfig.hiddenDim == 5120)
        #expect(vc?.encoderConfig.nKVHeads == 32)
        #expect(vc?.encoderConfig.slidingWindow == 750)
        #expect(vc?.encoderConfig.downsampleFactor == 4)
        #expect(vc?.encoderConfig.causal == true)
        #expect(vc?.encoderConfig.useBiases == true)

        // Audio front-end.
        #expect(vc?.audioConfig.samplingRate == 16_000)
        #expect(vc?.audioConfig.numMelBins == 128)
        #expect(vc?.audioConfig.hopLength == 160)
        #expect(vc?.audioConfig.windowSize == 400)
        #expect(vc?.audioConfig.globalLogMelMax == 1.5)

        // Decoder hyper-parameters.
        #expect(vc?.decoderConfig.dim == 3072)
        #expect(vc?.decoderConfig.nLayers == 26)
        #expect(vc?.decoderConfig.nHeads == 32)
        #expect(vc?.decoderConfig.nKVHeads == 8)
        #expect(vc?.decoderConfig.headDim == 128)
        #expect(vc?.decoderConfig.hiddenDim == 9216)
        #expect(vc?.decoderConfig.vocabSize == 131072)
        #expect(vc?.decoderConfig.tiedEmbeddings == true)
        #expect(vc?.decoderConfig.adaRmsNormTCond == true)
        #expect(vc?.decoderConfig.adaRmsNormTCondDim == 32)

        // Top-level fields.
        #expect(vc?.transcriptionDelayMs == 480)
        #expect(vc?.bosTokenId == 1)
        #expect(vc?.eosTokenId == 2)
        #expect(vc?.streamingPadTokenId == 32)
        #expect(vc?.nLeftPadTokens == 32)
    }

    @Test("VoxtralRealtimeConfig — falls back to Mini-4B defaults")
    func configFallsBackToDefaults() {
        // Minimal config with only model_type.
        let config = ModelConfig(
            architecture: nil,
            modelType: "voxtral_realtime",
            raw: ["model_type": "voxtral_realtime"])

        let vc = VoxtralRealtimeConfig.from(config)
        #expect(vc != nil)
        // Published Mini-4B defaults.
        #expect(vc?.encoderConfig.dim == 1280)
        #expect(vc?.encoderConfig.nLayers == 32)
        #expect(vc?.encoderConfig.headDim == 64)
        #expect(vc?.encoderConfig.slidingWindow == 750)
        #expect(vc?.encoderConfig.downsampleFactor == 4)
        #expect(vc?.decoderConfig.dim == 3072)
        #expect(vc?.decoderConfig.nLayers == 26)
        #expect(vc?.decoderConfig.nKVHeads == 8)
        #expect(vc?.decoderConfig.headDim == 128)
        #expect(vc?.decoderConfig.vocabSize == 131072)
        #expect(vc?.audioConfig.numMelBins == 128)
        #expect(vc?.audioConfig.samplingRate == 16_000)
        #expect(vc?.transcriptionDelayMs == 480)
        #expect(vc?.nLeftPadTokens == 32)
    }

    @Test("VoxtralRealtimeConfig — returns nil for non-voxtral model_type")
    func configReturnsNilForNonVoxtral() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 4096])
        #expect(VoxtralRealtimeConfig.from(config) == nil)
    }

    @Test("VoxtralRealtimeConfig — handles nested audio_encoding_args inside encoder_args")
    func configHandlesNestedAudioArgs() {
        // Real checkpoint layout: audio_encoding_args is inside encoder_args.
        let config = ModelConfig(
            architecture: nil,
            modelType: "voxtral_realtime",
            raw: [
                "model_type": "voxtral_realtime",
                "encoder_args": [
                    "dim": 1280, "n_layers": 32,
                    "audio_encoding_args": [
                        "sampling_rate": 16000,
                        "global_log_mel_max": 1.5,
                    ] as [String: Any],
                ] as [String: Any],
                "decoder": ["dim": 3072] as [String: Any],
            ])

        let vc = VoxtralRealtimeConfig.from(config)
        #expect(vc != nil)
        #expect(vc?.audioConfig.samplingRate == 16_000)
        #expect(vc?.audioConfig.globalLogMelMax == 1.5)
        #expect(vc?.encoderConfig.dim == 1280)
    }
}
