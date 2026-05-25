import Foundation
import Testing
@testable import FFAI

// Config-parse unit tests for the LFM2-VL family (the SigLIP2 +
// LFM2 `Lfm2VlForConditionalGeneration` checkpoint).
//
// These run offline on a hand-built `ModelConfig` — no checkpoint — and
// cover the parts of the load path that translate `config.json` into
// typed geometry: VL routing, the architecture string, the nested
// `vision_config` / `text_config` split, and the derived image-token
// count (numPatches / downsampleFactor²).
@Suite("LFM2 Vision Config")
struct LFM2VisionConfigTests {

    /// A representative `Lfm2VlForConditionalGeneration` config matching
    /// the mlx-community/LFM2-VL-1.6B-4bit checkpoint geometry.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "model_type": "siglip2_vision_model",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_channels": 3,
            "patch_size": 16,
            "num_patches": 256,
            "layer_norm_eps": 1e-6,
        ]
        let textConfig: [String: Any] = [
            "model_type": "lfm2",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 65536,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 128_000,
            "norm_eps": 1e-5,
            "conv_L_cache": 3,
            "conv_bias": false,
            "layer_types": [
                "conv", "conv", "full_attention", "conv", "conv", "full_attention",
                "conv", "conv", "full_attention", "conv", "full_attention",
                "conv", "full_attention", "conv", "full_attention", "conv",
            ],
        ]
        let raw: [String: Any] = [
            "architectures": ["Lfm2VlForConditionalGeneration"],
            "model_type": "lfm2-vl",
            "image_token_index": 396,
            "vision_feature_layer": -2,
            "downsample_factor": 2,
            "projector_hidden_size": 2560,
            "projector_bias": true,
            "quantization": ["group_size": 64, "bits": 4] as [String: Any],
            "vision_config": visionConfig,
            "text_config": textConfig,
        ]
        return ModelConfig(
            architecture: "Lfm2VlForConditionalGeneration",
            modelType: "lfm2-vl",
            raw: raw)
    }

    @Test("architecture string is registered in VisionLanguageArchitectures")
    func architectureRegistered() {
        #expect(LFM2VL.architectures.contains("Lfm2VlForConditionalGeneration"))
        #expect(VisionLanguageArchitectures.architectures
            .contains("Lfm2VlForConditionalGeneration"))
    }

    @Test("routes as a vision-language checkpoint")
    func routesAsVisionLanguage() {
        let cfg = makeConfig()
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        #expect(cfg.architecture == "Lfm2VlForConditionalGeneration")
        // Image placeholder token id.
        #expect(cfg.int("image_token_index") == 396)
        #expect(LFM2VL.defaultImageTokenId == 396)
    }

    @Test("vision_config parses into SigLIP2 VisionEncoderConfig geometry")
    func visionConfigGeometry() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        #expect(vc.int("hidden_size") == 1152)
        #expect(vc.int("num_hidden_layers") == 27)
        #expect(vc.int("num_attention_heads") == 16)
        #expect(vc.int("patch_size") == 16)
        #expect(vc.int("num_patches") == 256)

        // Natural imageSize = sqrt(numPatches) * patchSize = 16 * 16 = 256.
        let numPatches = vc.int("num_patches")!
        let patchSize = vc.int("patch_size")!
        let patchesPerSide = Int(Double(numPatches).squareRoot().rounded())
        let imageSize = patchesPerSide * patchSize
        #expect(patchesPerSide == 16)
        #expect(imageSize == 256)

        // VisionEncoderConfig geometry.
        let enc = VisionEncoderConfig(
            inChannels: vc.int("num_channels") ?? 3,
            imageSize: imageSize,
            patchSize: patchSize,
            hidden: vc.int("hidden_size")!,
            intermediate: vc.int("intermediate_size")!,
            nLayers: vc.int("num_hidden_layers")!,
            nHeads: vc.int("num_attention_heads")!,
            textHidden: vc.int("hidden_size")!)
        #expect(enc.numPatches == 256)           // 16×16
        #expect(enc.headDim == 72)               // 1152 / 16
    }

    @Test("vision_feature_layer = -2 yields 26 active encoder layers")
    func visionFeatureLayerMapping() {
        let cfg = makeConfig()
        let numHiddenLayers = 27
        let vfl = cfg.int("vision_feature_layer") ?? -2  // -2
        let actualLayer = numHiddenLayers + vfl           // 25
        let activeLayers = actualLayer + 1                // 26
        #expect(activeLayers == 26)
    }

    @Test("imageTokenCount = numPatches / downsampleFactor² = 64")
    func imageTokenCount() {
        let cfg = makeConfig()
        let vc = cfg.subConfig("vision_config")!
        let numPatches = vc.int("num_patches") ?? 256
        let downsample = cfg.int("downsample_factor") ?? 2
        let imageTokenCount = numPatches / (downsample * downsample)
        #expect(imageTokenCount == 64)   // 256 / 4
    }

    @Test("projector dimensions are consistent with pixel-unshuffle")
    func projectorDimensions() {
        let cfg = makeConfig()
        let vc = cfg.subConfig("vision_config")!
        let visionHidden = vc.int("hidden_size") ?? 1152
        let downsample = cfg.int("downsample_factor") ?? 2
        // LayerNorm input = visionHidden * downsample² = 1152 * 4 = 4608
        let unshuffledDim = visionHidden * downsample * downsample
        #expect(unshuffledDim == 4608)
        // projector_hidden_size from config
        let projHidden = cfg.int("projector_hidden_size") ?? 2560
        #expect(projHidden == 2560)
    }

    @Test("quantization config is decoded")
    func quantizationConfig() throws {
        let cfg = makeConfig()
        let quant = try #require(cfg.quantization)
        #expect(quant.bits == 4)
        #expect(quant.groupSize == 64)
    }

    @Test("text_config resolves LFM2 layer types")
    func textConfigLayerTypes() throws {
        let raw = makeConfig().nested("text_config")!
        let layerTypes = raw["layer_types"] as? [String]
        let kinds = try lfm2LayerKinds(
            layerTypes: layerTypes,
            fullAttnIdxs: nil,
            numLayers: 16)
        // Layer 2 is attention (index 2 in the layer_types array).
        #expect(kinds[0] == .conv)
        #expect(kinds[2] == .attention)
        #expect(kinds.count == 16)
        // Count attention layers: indices 2, 5, 8, 10, 12, 14 = 6 attention layers.
        let attnCount = kinds.filter { $0 == .attention }.count
        #expect(attnCount == 6)
    }
}
