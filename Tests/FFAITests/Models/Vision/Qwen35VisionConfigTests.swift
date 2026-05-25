import Foundation
import Testing
@testable import FFAI

// Config-parse unit tests for the Qwen 3-VL-MoE family (the Qwen3-VL ViT
// tower + the Qwen 3.5 mixture-of-experts hybrid text backbone, the
// `Qwen3VLMoeForConditionalGeneration` checkpoint).
//
// Offline — covers VL routing and the shared `Qwen3VLVisionConfig.decode`
// (the MoE variant reuses the dense Qwen3-VL vision tower), plus the
// MoE-specific `text_config` keys (`num_experts`, `num_experts_per_tok`).
@Suite("Qwen35 Vision Config")
struct Qwen35VisionConfigTests {

    /// A representative `Qwen3VLMoeForConditionalGeneration` config.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "out_hidden_size": 2048,
            "num_heads": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 32 * 32,
            "in_channels": 3,
        ]
        // The MoE text backbone — block-sparse FFN.
        let textConfig: [String: Any] = [
            "model_type": "qwen3_vl_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_experts": 128,
            "num_experts_per_tok": 8,
        ]
        let raw: [String: Any] = [
            "architectures": ["Qwen3VLMoeForConditionalGeneration"],
            "model_type": "qwen3_vl_moe",
            "image_token_id": 151_655,
            "vision_config": visionConfig,
            "text_config": textConfig,
        ]
        return ModelConfig(architecture: "Qwen3VLMoeForConditionalGeneration",
                           modelType: "qwen3_vl_moe", raw: raw)
    }

    @Test("routes as a vision-language checkpoint")
    func routesAsVisionLanguage() {
        let cfg = makeConfig()
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        #expect(VisionLanguageArchitectures.architectures
            .contains("Qwen3VLMoeForConditionalGeneration"))
        #expect(Qwen3VLMoe.defaultImageTokenId == 151_655)
    }

    @Test("vision_config decodes into the shared Qwen3-VL ViT geometry")
    func visionConfigDecode() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        let parsed = try Qwen3VLVisionConfig.decode(vc)
        #expect(parsed.depth == 27)
        #expect(parsed.hidden == 1152)
        #expect(parsed.numHeads == 16)
        #expect(parsed.patchSize == 16)
        #expect(parsed.spatialMergeSize == 2)
        #expect(parsed.headDim == 72)           // 1152 / 16
        #expect(parsed.mergeUnit == 4)          // 2 × 2 patches per token
    }

    @Test("text_config carries the MoE expert geometry")
    func textConfigMoEGeometry() throws {
        let tc = try #require(makeConfig().subConfig("text_config"))
        #expect(tc.int("hidden_size") == 2048)
        #expect(tc.int("num_hidden_layers") == 48)
        // The MoE-defining keys — present only on the MoE variant.
        #expect(tc.int("num_experts") == 128)
        #expect(tc.int("num_experts_per_tok") == 8)
    }
}
