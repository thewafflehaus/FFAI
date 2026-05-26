import Foundation
import Testing
@testable import FFAI

// Config-parse unit tests for the Mistral3 family (the Pixtral 2D-RoPE
// ViT + patch-merger projector + Mistral text backbone,
// `Mistral3ForConditionalGeneration` / `mistral3` checkpoint).
//
// Offline — covers VL routing and the decoded vision geometry (depth,
// head dim, patch count) as well as projector config fields
// (spatial_merge_size, merged token count).
@Suite("Mistral3 Vision Config")
struct Mistral3Tests {

    /// A representative Mistral-Small-3.1-24B `config.json` structure.
    /// The mlx-community conversion uses `model_type = "mistral3"` at the
    /// top level with `Mistral3ForConditionalGeneration` architecture.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "model_type": "pixtral",
            "num_hidden_layers": 24,
            "hidden_size": 1024,
            "head_dim": 64,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "image_size": 1540,
            "patch_size": 14,
            "num_channels": 3,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10_000.0,
        ]
        let textConfig: [String: Any] = [
            "model_type": "mistral",
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1_000_000_000.0,
            "vocab_size": 131_072,
        ]
        let raw: [String: Any] = [
            "architectures": ["Mistral3ForConditionalGeneration"],
            "model_type": "mistral3",
            "image_token_index": 10,
            "spatial_merge_size": 2,
            "multimodal_projector_bias": false,
            "vision_config": visionConfig,
            "text_config": textConfig,
        ]
        return ModelConfig(architecture: "Mistral3ForConditionalGeneration",
                           modelType: "mistral3", raw: raw)
    }

    @Test("routes as a vision-language checkpoint via model_type")
    func routesAsVisionLanguage() {
        let cfg = makeConfig()
        #expect(cfg.modelType == "mistral3")
        #expect(Mistral3.modelTypes.contains("mistral3"))
        // vision_config presence triggers VL routing.
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        // architecture string is in the VL set.
        #expect(VisionLanguageArchitectures.architectures
            .contains("Mistral3ForConditionalGeneration"))
        // defaults
        #expect(Mistral3.defaultImageTokenId == 10)
        #expect(Mistral3.defaultSpatialMergeSize == 2)
        #expect(cfg.int("image_token_index") == 10)
        #expect(cfg.int("spatial_merge_size") == 2)
    }

    @Test("architecture string is in Mistral3.architectures")
    func architectureInSet() {
        #expect(Mistral3.architectures.contains("Mistral3ForConditionalGeneration"))
    }

    @Test("vision_config decodes into Pixtral 2D-RoPE ViT geometry")
    func visionConfigDecode() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        let parsed = try PixtralVisionConfig.decode(vc)
        #expect(parsed.numLayers == 24)
        #expect(parsed.hiddenSize == 1024)
        #expect(parsed.headDim == 64)
        #expect(parsed.intermediateSize == 4096)
        #expect(parsed.numHeads == 16)
        #expect(parsed.imageSize == 1540)
        #expect(parsed.patchSize == 14)
        #expect(parsed.numChannels == 3)
        #expect(parsed.rmsNormEps == Float(1e-5))
        #expect(parsed.ropeTheta == Float(10_000.0))
        // Derived geometry.
        #expect(parsed.patchesPerSide == 110)  // 1540 / 14
        #expect(parsed.numPatches == 12100)     // 110 × 110
    }

    @Test("merged token count is patchesPerSide² / spatialMergeSize²")
    func mergedTokenCount() throws {
        let cfg = makeConfig()
        let vc = try #require(cfg.subConfig("vision_config"))
        let parsed = try PixtralVisionConfig.decode(vc)
        let s = cfg.int("spatial_merge_size") ?? Mistral3.defaultSpatialMergeSize
        let mergedPatches = parsed.numPatches / (s * s)
        // 12100 / (2*2) = 3025
        #expect(mergedPatches == 3025)
    }

    @Test("dispatch routing — mistral3 model_type triggers VL branch")
    func dispatchRouting() {
        let cfg = makeConfig()
        let mt = cfg.modelType ?? ""
        #expect(Mistral3.modelTypes.contains(mt))
        let arch = cfg.architecture ?? ""
        #expect(Mistral3.architectures.contains(arch))
    }

    @Test("text_config provides Mistral dense text backbone fields")
    func textConfigFields() throws {
        let tc = try #require(makeConfig().subConfig("text_config"))
        #expect(tc.int("hidden_size") == 5120)
        #expect(tc.int("num_hidden_layers") == 40)
        #expect(tc.int("num_attention_heads") == 32)
        #expect(tc.int("num_key_value_heads") == 8)
        #expect(tc.int("vocab_size") == 131_072)
    }
}
