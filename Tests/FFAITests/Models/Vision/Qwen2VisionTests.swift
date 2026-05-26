import Foundation
import Testing
@testable import FFAI

// Unit tests for the Qwen 2-VL family (the full-attention
// ViT + Qwen 2 text backbone `Qwen2VLForConditionalGeneration` checkpoint).
//
// Offline — covers VL routing and `Qwen2VLVisionConfig.decode`, which
// turns the nested `vision_config` into the ViT geometry (depth, head dim,
// spatial-merge unit) the vision tower is built from.
@Suite("Qwen2 Vision")
struct Qwen2VisionConfigTests {

    /// A representative `Qwen2VLForConditionalGeneration` config matching
    /// the mlx-community/Qwen2-VL-2B-Instruct-4bit checkpoint layout.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "depth": 32,
            "embed_dim": 1280,      // vision hidden (Qwen2-VL naming)
            "hidden_size": 1536,    // text backbone hidden (merger output)
            "mlp_ratio": 4.0,
            "num_heads": 16,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_chans": 3,
        ]
        let raw: [String: Any] = [
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "model_type": "qwen2_vl",
            "image_token_id": 151_655,
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "vision_config": visionConfig,
        ]
        return ModelConfig(architecture: "Qwen2VLForConditionalGeneration",
                           modelType: "qwen2_vl", raw: raw)
    }

    @Test("routes as a vision-language checkpoint")
    func routesAsVisionLanguage() {
        let cfg = makeConfig()
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        #expect(VisionLanguageArchitectures.architectures
            .contains("Qwen2VLForConditionalGeneration"))
        #expect(Qwen2VL.defaultImageTokenId == 151_655)
        #expect(cfg.int("image_token_id") == 151_655)
    }

    @Test("vision_config decodes into ViT geometry")
    func visionConfigDecode() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        let parsed = try Qwen2VLVisionConfig.decode(vc)
        #expect(parsed.depth == 32)
        #expect(parsed.hidden == 1280)       // embed_dim
        #expect(parsed.outHidden == 1536)    // hidden_size (text dim)
        #expect(parsed.numHeads == 16)
        #expect(parsed.patchSize == 14)
        #expect(parsed.spatialMergeSize == 2)
        #expect(parsed.temporalPatchSize == 2)
        #expect(parsed.inChannels == 3)
        // Derived geometry.
        #expect(parsed.headDim == 80)        // 1280 / 16
        #expect(parsed.mergeUnit == 4)       // 2 × 2 patches per merged token
        // intermediate = mlp_ratio × hidden = 4 × 1280 = 5120
        #expect(parsed.intermediate == 5120)
    }

    @Test("vision_config decode falls back to documented defaults")
    func visionConfigDefaults() throws {
        // Minimal config — only the required fields. `decode` fills
        // mlp_ratio / temporal_patch_size / in_channels.
        let minimal = ModelConfig(architecture: nil, modelType: nil, raw: [
            "depth": 16,
            "embed_dim": 1024,
            "num_heads": 8,
            "patch_size": 14,
            "spatial_merge_size": 2,
        ])
        let parsed = try Qwen2VLVisionConfig.decode(minimal)
        // mlp_ratio defaults to 4.0
        #expect(parsed.intermediate == 4096) // 4 × 1024
        // outHidden falls back to hidden when hidden_size absent
        #expect(parsed.outHidden == 1024)
        #expect(parsed.temporalPatchSize == 2)
        #expect(parsed.inChannels == 3)
        #expect(parsed.layerNormEps == 1e-6)
    }
}
