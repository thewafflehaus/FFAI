import Foundation
import Testing
@testable import FFAI

// Config-parse unit tests for the Nemotron-VLM family (NVIDIA's
// Nemotron Nano VL — a ViT tower + multi-modal projector + the
// NemotronH stack-interleaved hybrid text backbone).
//
// Offline — covers the Nemotron-specific VL routing
// (`isNemotronVisionLanguage`, which keys off the text backbone's
// `model_type == nemotron_h` since the VL conversion carries no single
// canonical top-level architecture string) and the SigLIP-style
// `vision_config` → `VisionEncoderConfig` geometry.
@Suite("NemotronH Vision Config")
struct NemotronHVisionConfigTests {

    /// A representative Nemotron Nano VL config: a ViT vision tower + a
    /// `nemotron_h`-typed `text_config`.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "model_type": "radio",
            "hidden_size": 1280,
            "image_size": 512,
            "patch_size": 16,
            "intermediate_size": 5120,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "layer_norm_eps": 1e-6,
        ]
        let textConfig: [String: Any] = [
            "model_type": "nemotron_h",
            "hidden_size": 4096,
            "num_hidden_layers": 52,
        ]
        let raw: [String: Any] = [
            // Nemotron Nano VL carries no single top-level VL arch
            // string — routing is by the text backbone's model_type.
            "model_type": "llama_nemotron_nano_vl",
            "image_token_id": 131_072,
            "vision_config": visionConfig,
            "text_config": textConfig,
        ]
        return ModelConfig(architecture: nil,
                           modelType: "llama_nemotron_nano_vl", raw: raw)
    }

    @Test("routes as a Nemotron vision-language checkpoint")
    func routesAsNemotronVisionLanguage() {
        let cfg = makeConfig()
        // The generic VL check passes on `vision_config` presence.
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        // And the Nemotron-specific check keys off text_config's
        // `model_type == nemotron_h`.
        #expect(isNemotronVisionLanguage(cfg))
        #expect(NemotronVL.defaultImageTokenId == 131_072)
        #expect(cfg.int("image_token_id") == 131_072)
    }

    @Test("a non-Nemotron VL config does not route to NemotronVL")
    func nonNemotronVLDoesNotRoute() {
        // Same shape, but the text backbone is a plain Llama — the
        // Nemotron-specific check must reject it.
        let raw: [String: Any] = [
            "model_type": "some_vl",
            "vision_config": ["hidden_size": 1024],
            "text_config": ["model_type": "llama"],
        ]
        let cfg = ModelConfig(architecture: nil, modelType: "some_vl", raw: raw)
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        #expect(!isNemotronVisionLanguage(cfg))
    }

    @Test("vision_config parses into ViT VisionEncoder geometry")
    func visionConfigGeometry() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        #expect(vc.int("hidden_size") == 1280)
        #expect(vc.int("image_size") == 512)
        #expect(vc.int("patch_size") == 16)
        #expect(vc.int("num_hidden_layers") == 32)
        #expect(vc.int("num_attention_heads") == 16)

        let enc = VisionEncoderConfig(
            imageSize: vc.int("image_size")!,
            patchSize: vc.int("patch_size")!,
            hidden: vc.int("hidden_size")!,
            intermediate: vc.int("intermediate_size")!,
            nLayers: vc.int("num_hidden_layers")!,
            nHeads: vc.int("num_attention_heads")!,
            textHidden: vc.int("hidden_size")!)
        // 512 / 16 = 32 patches per side → 1024 raw patches.
        #expect(enc.patchesPerSide == 32)
        #expect(enc.numPatches == 1024)
        #expect(enc.headDim == 80)              // 1280 / 16
    }

    @Test("text_config exposes the NemotronH backbone")
    func textConfigBackbone() throws {
        let tc = try #require(makeConfig().subConfig("text_config"))
        #expect(tc.string("model_type") == "nemotron_h")
        #expect(NemotronH.modelTypes.contains(tc.string("model_type")!))
        #expect(tc.int("hidden_size") == 4096)
        #expect(tc.int("num_hidden_layers") == 52)
    }
}
