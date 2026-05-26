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

// Config-parse unit tests for the FastVLM family (Apple's FastViTHD
// vision tower + mlp2x_gelu projector + Qwen2 text backbone,
// `LlavaQwen2ForCausalLM` / `llava_qwen2` checkpoint).
//
// Offline — covers VL routing and `FastVLMVisionConfig.decode`, which
// turns the nested `vision_config` into the FastViTHD geometry
// (stages, mixers, spatial resolution path).
@Suite("FastVLM Vision Config")
struct FastVLMTests {

    /// A representative FastVLM-0.5B `config.json` structure. The
    /// mlx-community conversion uses `model_type = "llava_qwen2"` at the
    /// top level and `LlavaQwen2ForCausalLM` as architecture.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "cls_ratio": 2.0,
            "down_patch_size": 7,
            "down_stride": 2,
            "downsamples": [true, true, true, true, true],
            "embed_dims": [96, 192, 384, 768, 1536],
            "hidden_size": 1024,
            "image_size": 1024,
            "intermediate_size": 3072,
            "layer_scale_init_value": 1e-5,
            "layers": [2, 12, 24, 4, 2],
            "mlp_ratios": [4, 4, 4, 4, 4],
            "num_classes": 1000,
            "patch_size": 64,
            "pos_embs_shapes": [NSNull(), NSNull(), NSNull(), [7, 7], [7, 7]],
            "projection_dim": 768,
            "repmixer_kernel_size": 3,
            "token_mixers": ["repmixer", "repmixer", "repmixer", "attention", "attention"],
        ]
        let raw: [String: Any] = [
            "architectures": ["LlavaQwen2ForCausalLM"],
            "model_type": "llava_qwen2",
            "eos_token_id": 151645,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "tie_word_embeddings": true,
            "mm_hidden_size": 3072,
            "mm_projector_type": "mlp2x_gelu",
            "vocab_size": 151936,
            "vision_config": visionConfig,
        ]
        return ModelConfig(architecture: "LlavaQwen2ForCausalLM",
                           modelType: "llava_qwen2", raw: raw)
    }

    @Test("routes as a vision-language checkpoint via architecture string")
    func routesAsVisionLanguageByArch() {
        let cfg = makeConfig()
        #expect(cfg.architecture == "LlavaQwen2ForCausalLM")
        #expect(FastVLM.architectures.contains("LlavaQwen2ForCausalLM"))
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        #expect(VisionLanguageArchitectures.architectures
            .contains("LlavaQwen2ForCausalLM"))
    }

    @Test("routes as a vision-language checkpoint via model_type")
    func routesAsVisionLanguageByModelType() {
        let cfg = makeConfig()
        #expect(cfg.modelType == "llava_qwen2")
        #expect(FastVLM.modelTypes.contains("llava_qwen2"))
        // The `vision_config` presence also triggers VL routing.
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
    }

    @Test("default image_token_id is -200")
    func defaultImageTokenId() {
        let cfg = makeConfig()
        // FastVLM doesn't store image_token_id in config.json; the
        // family default is -200 (matching the reference processor).
        #expect(FastVLM.defaultImageTokenId == -200)
        #expect(cfg.int("image_token_id") == nil)
    }

    @Test("vision_config decodes into FastViTHD geometry")
    func visionConfigDecode() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        let parsed = try FastVLMVisionConfig.decode(vc)
        #expect(parsed.embedDims == [96, 192, 384, 768, 1536])
        #expect(parsed.layers == [2, 12, 24, 4, 2])
        #expect(parsed.tokenMixers == ["repmixer", "repmixer", "repmixer",
                                       "attention", "attention"])
        #expect(parsed.downSamples == [true, true, true, true, true])
        #expect(parsed.imageSize == 1024)
        #expect(parsed.patchSize == 64)
        #expect(parsed.downPatchSize == 7)
        #expect(parsed.downStride == 2)
        #expect(parsed.clsRatio == 2.0)
        #expect(parsed.repMixerKernelSize == 3)
        #expect(parsed.nStages == 5)
        // mm_hidden_size = int(1536 * 2.0) = 3072.
        #expect(parsed.mmHiddenSize == 3072)
    }

    @Test("pos_embs_shapes parses null entries as nil")
    func posEmbShapesParsing() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        let parsed = try FastVLMVisionConfig.decode(vc)
        #expect(parsed.posEmbShapes[0] == nil)
        #expect(parsed.posEmbShapes[1] == nil)
        #expect(parsed.posEmbShapes[2] == nil)
        // Stages 3 and 4 have CPE positional embeddings.
        #expect(parsed.posEmbShapes[3] == [7, 7])
        #expect(parsed.posEmbShapes[4] == [7, 7])
    }

    @Test("spatial resolution path is correct for 1024px input")
    func spatialResolutionPath() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        let parsed = try FastVLMVisionConfig.decode(vc)
        let resolutions = parsed.spatialResolutions()
        // ConvStem: 1024/4 = 256.  After each stride-2 PatchEmbed: /2.
        // Stage 0: 256×256, Stage 1: 128×128, Stage 2: 64×64,
        // Stage 3: 32×32,   Stage 4: 16×16.
        #expect(resolutions[0] == (256, 256))
        #expect(resolutions[1] == (128, 128))
        #expect(resolutions[2] == (64, 64))
        #expect(resolutions[3] == (32, 32))
        #expect(resolutions[4] == (16, 16))
    }

    @Test("dispatch routing — llava_qwen2 model_type triggers VL branch")
    func dispatchRouting() {
        let cfg = makeConfig()
        let mt = cfg.modelType ?? ""
        let arch = cfg.architecture ?? ""
        // Both model_type and architecture routes hit FastVLM.
        #expect(FastVLM.modelTypes.contains(mt))
        #expect(FastVLM.architectures.contains(arch))
    }

    @Test("vision config defaults fall back gracefully")
    func visionConfigDefaults() throws {
        // Minimal config — only the absolutely required fields.
        let minimal = ModelConfig(architecture: nil, modelType: nil, raw: [
            "embed_dims": [64, 128],
            "layers": [2, 2],
            "token_mixers": ["repmixer", "attention"],
            "downsamples": [true, true],
            "mlp_ratios": [4, 4],
            "image_size": 256,
            "patch_size": 16,
            "down_patch_size": 7,
            "down_stride": 2,
        ])
        let parsed = try FastVLMVisionConfig.decode(minimal)
        // cls_ratio defaults to 2.0; mm_hidden_size = 128 * 2 = 256.
        #expect(parsed.clsRatio == 2.0)
        #expect(parsed.mmHiddenSize == 256)
        // repMixerKernelSize defaults to 3.
        #expect(parsed.repMixerKernelSize == 3)
        // posEmbShapes defaults to all nil.
        #expect(parsed.posEmbShapes.count == 2)
        #expect(parsed.posEmbShapes[0] == nil)
        #expect(parsed.posEmbShapes[1] == nil)
        // Spatial: 256/4=64 after stem, /2=32 after PE.
        let res = parsed.spatialResolutions()
        #expect(res[0] == (64, 64))
        #expect(res[1] == (32, 32))
    }
}
