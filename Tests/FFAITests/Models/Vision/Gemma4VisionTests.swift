import Foundation
import Testing
@testable import FFAI

// Unit tests for the Gemma 4 VL family (the bespoke Gemma 4
// ViT tower + multi-modal embedder + Gemma 4 text backbone, the
// `Gemma4ForConditionalGeneration` checkpoint).
//
// Offline — covers VL routing (Gemma 4 is detected by `vision_config`
// presence, NOT an architecture string, since `Gemma4ForConditional-
// Generation` is shared with text-only Gemma 4) and
// `Gemma4VLVisionConfig.decode`, which turns the nested `vision_config`
// into the RoPE-attention ViT geometry.
@Suite("Gemma4 Vision")
struct Gemma4VisionConfigTests {

    /// A representative `Gemma4ForConditionalGeneration` VL config.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "num_hidden_layers": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 72,
            "patch_size": 14,
            "rms_norm_eps": 1e-6,
            "default_output_length": 280,
            "position_embedding_size": 10_240,
            "pooling_kernel_size": 3,
            "standardize": true,
            "rope_parameters": ["rope_theta": 100.0],
        ]
        let textConfig: [String: Any] = [
            "model_type": "gemma4_text",
            "hidden_size": 2048,
            "num_hidden_layers": 30,
        ]
        let raw: [String: Any] = [
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "image_token_id": 262_144,
            "vision_config": visionConfig,
            "text_config": textConfig,
        ]
        return ModelConfig(architecture: "Gemma4ForConditionalGeneration",
                           modelType: "gemma4", raw: raw)
    }

    @Test("routes as a vision-language checkpoint by vision_config presence")
    func routesAsVisionLanguage() {
        let cfg = makeConfig()
        // Gemma 4 VL is detected by the `vision_config` block, not the
        // architecture string (the arch is shared with text-only Gemma 4
        // and is deliberately absent from `architectures`).
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        #expect(!VisionLanguageArchitectures.architectures
            .contains("Gemma4ForConditionalGeneration"))
        #expect(Gemma4VL.defaultImageTokenId == 262_144)

        // A text-only Gemma 4 config (no vision_config) does NOT route VL.
        let textOnly = ModelConfig(
            architecture: "Gemma4ForConditionalGeneration",
            modelType: "gemma4",
            raw: ["architectures": ["Gemma4ForConditionalGeneration"],
                  "model_type": "gemma4", "hidden_size": 2048])
        #expect(!VisionLanguageArchitectures.isVisionLanguage(textOnly))
    }

    @Test("vision_config decodes into RoPE-attention ViT geometry")
    func visionConfigDecode() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        let parsed = try Gemma4VLVisionConfig.decode(vc)
        #expect(parsed.depth == 27)
        #expect(parsed.hidden == 1152)
        #expect(parsed.intermediate == 4304)
        #expect(parsed.numHeads == 16)
        #expect(parsed.numKVHeads == 16)
        #expect(parsed.headDim == 72)
        #expect(parsed.patchSize == 14)
        #expect(parsed.defaultOutputLength == 280)
        #expect(parsed.positionEmbeddingSize == 10_240)
        #expect(parsed.poolingKernelSize == 3)
        #expect(parsed.standardize == true)
        // The RoPE base frequency is nested under `rope_parameters`.
        #expect(parsed.ropeTheta == 100.0)
    }

    @Test("vision_config decode falls back to documented defaults")
    func visionConfigDefaults() throws {
        // Minimal config — only the required fields.
        let minimal = ModelConfig(architecture: nil, modelType: nil, raw: [
            "num_hidden_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "patch_size": 14,
        ])
        let parsed = try Gemma4VLVisionConfig.decode(minimal)
        #expect(parsed.numKVHeads == 12)        // num_heads fallback
        #expect(parsed.headDim == 64)           // hidden / num_heads
        #expect(parsed.intermediate == 768 * 4) // hidden * 4 fallback
        #expect(parsed.defaultOutputLength == 280)
        #expect(parsed.positionEmbeddingSize == 10_240)
        #expect(parsed.poolingKernelSize == 3)
        #expect(parsed.standardize == false)
        #expect(parsed.ropeTheta == 100.0)      // documented default
    }
}
