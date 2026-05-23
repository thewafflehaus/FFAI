import Foundation
import Testing
@testable import FFAI

// Config-parse unit tests for the Gemma 3 VL family (the SigLIP +
// Gemma 3 `Gemma3ForConditionalGeneration` checkpoint).
//
// These run offline on a hand-built `ModelConfig` — no checkpoint — and
// cover the parts of the load path that translate `config.json` into
// typed geometry: VL routing, the nested `vision_config` / `text_config`
// split, the SigLIP `VisionEncoderConfig` geometry, and the sparse
// `text_config` default-merge.
@Suite("Gemma 3 VL config")
struct Gemma3VLConfigTests {

    /// A representative `Gemma3ForConditionalGeneration` config: a
    /// SigLIP-896 / patch-14 vision tower + a 4B Gemma 3 text backbone.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "model_type": "siglip_vision_model",
            "hidden_size": 1152,
            "image_size": 896,
            "patch_size": 14,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "layer_norm_eps": 1e-6,
        ]
        // Sparse text_config — HF omits class-default fields.
        let textConfig: [String: Any] = [
            "model_type": "gemma3_text",
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_hidden_layers": 34,
        ]
        let raw: [String: Any] = [
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
            "image_token_index": 262_144,
            "mm_tokens_per_image": 256,
            "vision_config": visionConfig,
            "text_config": textConfig,
        ]
        return ModelConfig(architecture: "Gemma3ForConditionalGeneration",
                           modelType: "gemma3", raw: raw)
    }

    @Test("routes as a vision-language checkpoint")
    func routesAsVisionLanguage() {
        let cfg = makeConfig()
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        #expect(VisionLanguageArchitectures.architectures
            .contains("Gemma3ForConditionalGeneration"))
        // The default image-placeholder token id.
        #expect(Gemma3VL.defaultImageTokenId == 262_144)
        #expect(cfg.int("image_token_index") == 262_144)
        #expect(cfg.int("mm_tokens_per_image") == 256)
    }

    @Test("vision_config parses into SigLIP VisionEncoder geometry")
    func visionConfigGeometry() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        #expect(vc.int("hidden_size") == 1152)
        #expect(vc.int("image_size") == 896)
        #expect(vc.int("patch_size") == 14)
        #expect(vc.int("num_hidden_layers") == 27)
        #expect(vc.int("num_attention_heads") == 16)

        // The VisionEncoderConfig derives the patch grid + head dim.
        let enc = VisionEncoderConfig(
            imageSize: vc.int("image_size")!,
            patchSize: vc.int("patch_size")!,
            hidden: vc.int("hidden_size")!,
            intermediate: vc.int("intermediate_size")!,
            nLayers: vc.int("num_hidden_layers")!,
            nHeads: vc.int("num_attention_heads")!,
            textHidden: vc.int("hidden_size")!)
        // 896 / 14 = 64 patches per side → 4096 raw patches.
        #expect(enc.patchesPerSide == 64)
        #expect(enc.numPatches == 4096)
        #expect(enc.headDim == 72)              // 1152 / 16
    }

    @Test("sparse text_config merges Gemma 3 text-model defaults")
    func textConfigDefaultMerge() {
        let sparse: [String: Any] = [
            "hidden_size": 2560,
            "num_hidden_layers": 34,
        ]
        let merged = gemma3TextConfigWithDefaults(sparse, vocabFallback: 262_208)
        // Checkpoint-declared fields survive.
        #expect((merged["hidden_size"] as? Int) == 2560)
        #expect((merged["num_hidden_layers"] as? Int) == 34)
        // Omitted fields fall back to the documented Gemma 3 defaults.
        #expect((merged["num_attention_heads"] as? Int) == 8)
        #expect((merged["num_key_value_heads"] as? Int) == 4)
        #expect((merged["head_dim"] as? Int) == 256)
        #expect((merged["vocab_size"] as? Int) == 262_208)
        #expect((merged["sliding_window"] as? Int) == 1024)
    }
}
