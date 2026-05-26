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

// Unit tests for the GLMASR speech-to-text family.
// Exercises config decoding and registry detection using synthetic
// `ModelConfig` objects (no real checkpoint required).
@Suite("GLMASR")
struct GLMASRTests {

    // ─── Registry detection ──────────────────────────────────────────

    @Test("AudioModelRegistry — detects GLM-ASR from model_type")
    func registryDetectsGLMASRByModelType() {
        let config = ModelConfig(
            architecture: "GlmasrModel",
            modelType: "glmasr",
            raw: ["model_type": "glmasr",
                  "architectures": ["GlmasrModel"],
                  "merge_factor": 4,
                  "use_rope": true,
                  "max_whisper_length": 1500])
        #expect(GLMASRModel.handles(config))
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.speechToText)
    }

    @Test("AudioModelRegistry — detects GLM-ASR from architecture")
    func registryDetectsGLMASRByArchitecture() {
        // Some mlx-community conversions omit model_type but keep the
        // architecture string.
        let config = ModelConfig(
            architecture: "GlmasrModel",
            modelType: nil,
            raw: ["architectures": ["GlmasrModel"]])
        #expect(GLMASRModel.handles(config))
        #expect(AudioModelRegistry.handles(config))
    }

    @Test("AudioModelRegistry — GLM-ASR is speechToText capability")
    func glmASRCapability() {
        let config = ModelConfig(
            architecture: nil, modelType: "glmasr",
            raw: ["model_type": "glmasr"])
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.speechToText)
    }

    @Test("AudioModelRegistry — text-only config is not GLM-ASR")
    func textOnlyNotGLMASR() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 2048])
        #expect(!GLMASRModel.handles(config))
    }

    @Test("AudioModelRegistry — Whisper config is not GLM-ASR")
    func whisperNotGLMASR() {
        let config = ModelConfig(
            architecture: "WhisperForConditionalGeneration",
            modelType: "whisper",
            raw: ["model_type": "whisper"])
        #expect(!GLMASRModel.handles(config))
    }

    // ─── Config parsing ──────────────────────────────────────────────

    @Test("GLMASRConfig — decodes from sparse config.json (Nano defaults)")
    func configDecodesNanoDefaults() {
        // The Nano checkpoint ships a sparse config with only top-level knobs;
        // all architecture hyper-params fall back to Nano defaults.
        let config = ModelConfig(
            architecture: "GlmasrModel",
            modelType: "glmasr",
            raw: ["model_type": "glmasr",
                  "architectures": ["GlmasrModel"],
                  "adapter_type": "mlp",
                  "merge_factor": 4,
                  "max_whisper_length": 1500,
                  "use_rope": true,
                  "max_length": 65536,
                  "quantization": ["group_size": 64, "bits": 4, "mode": "affine"]])
        let gc = GLMASRConfig.from(config)
        #expect(gc != nil)

        // Whisper encoder defaults for the Nano checkpoint.
        #expect(gc?.numMelBins == 128)
        #expect(gc?.whisperDModel == 1280)
        #expect(gc?.whisperEncoderLayers == 32)
        #expect(gc?.whisperEncoderHeads == 20)
        #expect(gc?.whisperEncoderFfnDim == 5120)
        #expect(gc?.maxWhisperLength == 1500)
        #expect(gc?.useRope == true)
        #expect(gc?.mergeFactor == 4)

        // Text decoder defaults for the Nano checkpoint.
        #expect(gc?.lmHiddenSize == 2048)
        #expect(gc?.lmVocabSize == 59264)
        #expect(gc?.lmNumLayers == 28)
        #expect(gc?.lmNumHeads == 16)
        #expect(gc?.lmNumKVHeads == 4)
        #expect(gc?.lmHeadDim == 128)
        #expect(gc?.lmIntermediate == 6144)
        #expect(gc?.eosTokenIds == [59246, 59253, 59255])
    }

    @Test("GLMASRConfig — respects overridden merge_factor")
    func configMergeFactor() {
        let config = ModelConfig(
            architecture: nil, modelType: "glmasr",
            raw: ["model_type": "glmasr", "merge_factor": 8])
        let gc = GLMASRConfig.from(config)
        #expect(gc?.mergeFactor == 8)
    }

    @Test("GLMASRConfig — respects overridden use_rope = false")
    func configUseRopeFalse() {
        let config = ModelConfig(
            architecture: nil, modelType: "glmasr",
            raw: ["model_type": "glmasr", "use_rope": false])
        let gc = GLMASRConfig.from(config)
        #expect(gc?.useRope == false)
    }

    @Test("GLMASRConfig — EOS as single int is promoted to an array")
    func configSingleEos() {
        let config = ModelConfig(
            architecture: nil, modelType: "glmasr",
            raw: ["model_type": "glmasr", "eos_token_id": 59246])
        let gc = GLMASRConfig.from(config)
        #expect(gc?.eosTokenIds == [59246])
    }

    @Test("GLMASRConfig — EOS as int array is preserved")
    func configArrayEos() {
        let config = ModelConfig(
            architecture: nil, modelType: "glmasr",
            raw: ["model_type": "glmasr",
                  "eos_token_id": [59246, 59253, 59255]])
        let gc = GLMASRConfig.from(config)
        #expect(gc?.eosTokenIds == [59246, 59253, 59255])
    }

    @Test("GLMASRConfig — returns nil for non-GLM-ASR config")
    func configReturnsNilForOtherModel() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 2048])
        #expect(GLMASRConfig.from(config) == nil)
    }

    @Test("GLMASRConfig — defaults init produces Nano values")
    func defaultInitIsNano() {
        let gc = GLMASRConfig()
        // Verify a subset of important Nano defaults.
        #expect(gc.numMelBins == 128)
        #expect(gc.whisperDModel == 1280)
        #expect(gc.lmHiddenSize == 2048)
        #expect(gc.lmNumLayers == 28)
        #expect(gc.lmNumKVHeads == 4)
        #expect(gc.mergeFactor == 4)
        #expect(gc.useRope == true)
        #expect(gc.eosTokenIds == [59246, 59253, 59255])
    }
}
