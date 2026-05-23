import Foundation
import Testing
@testable import FFAI

// Config-parse unit tests for the Pixtral family (the custom 2D-RoPE
// ViT + Mistral text backbone `LlavaForConditionalGeneration` /
// `pixtral` checkpoint).
//
// Offline — covers VL routing and `PixtralVisionConfig.decode`, which
// turns the nested `vision_config` into the 2D-RoPE ViT geometry (depth,
// head dim, patch count) the vision tower is built from.
@Suite("Pixtral config")
struct PixtralConfigTests {

    /// A representative Pixtral-12B `config.json` structure. The
    /// mlx-community conversion uses `model_type = "pixtral"` at the top
    /// level and `LlavaForConditionalGeneration` as architecture.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "model_type": "pixtral",
            "num_hidden_layers": 24,
            "hidden_size": 1024,
            "head_dim": 64,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "image_size": 336,
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
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000_000.0,
            "vocab_size": 131_072,
        ]
        let raw: [String: Any] = [
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "pixtral",
            "image_token_id": 10,
            "vision_config": visionConfig,
            "text_config": textConfig,
        ]
        return ModelConfig(architecture: "LlavaForConditionalGeneration",
                           modelType: "pixtral", raw: raw)
    }

    @Test("routes as a vision-language checkpoint via model_type")
    func routesAsVisionLanguage() {
        let cfg = makeConfig()
        // model_type check
        #expect(cfg.modelType == "pixtral")
        #expect(Pixtral.modelTypes.contains("pixtral"))
        // vision_config presence triggers VL routing
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        // architecture string is in the VL set
        #expect(VisionLanguageArchitectures.architectures
            .contains("LlavaForConditionalGeneration"))
        // default image token id
        #expect(Pixtral.defaultImageTokenId == 10)
        #expect(cfg.int("image_token_id") == 10)
    }

    @Test("vision_config decodes into 2D-RoPE ViT geometry")
    func visionConfigDecode() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        let parsed = try PixtralVisionConfig.decode(vc)
        #expect(parsed.numLayers == 24)
        #expect(parsed.hiddenSize == 1024)
        #expect(parsed.headDim == 64)
        #expect(parsed.intermediateSize == 4096)
        #expect(parsed.numHeads == 16)
        #expect(parsed.imageSize == 336)
        #expect(parsed.patchSize == 14)
        #expect(parsed.numChannels == 3)
        #expect(parsed.rmsNormEps == Float(1e-5))
        #expect(parsed.ropeTheta == Float(10_000.0))
        // Derived geometry.
        #expect(parsed.patchesPerSide == 24)  // 336 / 14
        #expect(parsed.numPatches == 576)      // 24 × 24
    }

    @Test("vision_config decode falls back to documented defaults")
    func visionConfigDefaults() throws {
        // Minimal config — only the fields required by `decode`.
        let minimal = ModelConfig(architecture: nil, modelType: nil, raw: [
            "num_hidden_layers": 12,
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "patch_size": 14,
        ])
        let parsed = try PixtralVisionConfig.decode(minimal)
        // Derived head_dim fallback: 512 / 8 = 64.
        #expect(parsed.headDim == 64)
        // Defaults.
        #expect(parsed.imageSize == 336)
        #expect(parsed.numChannels == 3)
        #expect(parsed.rmsNormEps == Float(1e-5))
        #expect(parsed.ropeTheta == Float(10_000.0))
        // Derived patch geometry.
        #expect(parsed.patchesPerSide == 24)   // 336 / 14
        #expect(parsed.numPatches == 576)
    }

    @Test("PixtralRoPE builds consistent tables for a small grid")
    func ropeTablesSmall() throws {
        // Build a minimal config with a 2×2 patch grid (image_size = 28,
        // patch_size = 14 → patchesPerSide = 2). headDim = 4, half = 2,
        // quarter = 1.
        let minimal = ModelConfig(architecture: nil, modelType: nil, raw: [
            "num_hidden_layers": 1,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_attention_heads": 1,
            "head_dim": 4,
            "patch_size": 14,
            "image_size": 28,
            "num_channels": 3,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10_000.0,
        ])
        let cfg = try PixtralVisionConfig.decode(minimal)
        let rope = PixtralRoPE(cfg: cfg)

        // Patch (0,0) — both row and col are 0, so all angles are 0:
        // cos = 1, sin = 0.
        let (cos00, sin00) = rope.cosSin(row: 0, col: 0)
        for v in cos00 { #expect(abs(v - 1.0) < 1e-5, "cos(0) should be 1") }
        for v in sin00 { #expect(abs(v) < 1e-5, "sin(0) should be 0") }

        // Table shape: maxPatchesPerSide = 2, headDim = 4, so 4 patches × 4 dims.
        #expect(rope.cosTable.count == 4 * 4)
        #expect(rope.sinTable.count == 4 * 4)

        // Patch (1,0) — non-trivial cos/sin from row position 1.
        let (cos10, sin10) = rope.cosSin(row: 1, col: 0)
        let arr10c = Array(cos10)
        let arr10s = Array(sin10)
        // All values should be in [-1, 1].
        for v in arr10c { #expect(v >= -1.0 && v <= 1.0) }
        for v in arr10s { #expect(v >= -1.0 && v <= 1.0) }
        // At least one cos should differ from 1 (non-zero angle).
        #expect(arr10c.contains { abs($0 - 1.0) > 1e-4 })
    }

    @Test("dispatch routing — Pixtral model_type triggers VL branch")
    func dispatchRouting() {
        let cfg = makeConfig()
        // model_type "pixtral" is in the Pixtral.modelTypes set — the
        // dispatch branch fires before the generic visionModelNotIntegrated
        // fallback. The ModelRegistry dispatches on model_type for Pixtral.
        let mt = cfg.modelType ?? ""
        #expect(Pixtral.modelTypes.contains(mt))
        // The architecture is also in the VL set.
        let arch = cfg.architecture ?? ""
        #expect(Pixtral.architectures.contains(arch))
    }
}
