// LFM2 VL — LiquidAI's LFM2 vision-language model (the
// `Lfm2VlForConditionalGeneration` checkpoints).
//
// Composition:
//   • SigLIP2 vision tower — a standard ViT loaded into the shared
//     `VisionEncoder`. Weight keys are `vision_tower.embeddings.*` /
//     `vision_tower.encoder.layers.*` / `vision_tower.post_layernorm.*`,
//     matching `VisionEncoder.parameters()` exactly after prefixing.
//     The patch embed is a flattened linear projection `[hidden, 768]`
//     (= `[hidden, channels * patchSize * patchSize]`); reshaped to
//     `[hidden, channels, patchSize, patchSize]` (OIHW) it is exactly
//     the `Ops.conv2d` weight layout FFAI uses.
//   • Pixel-unshuffle — collapses a `downsampleFactor × downsampleFactor`
//     neighbourhood of adjacent ViT patches into one super-patch,
//     multiplying the feature dim by `downsampleFactor²` and reducing
//     the token count by the same factor. `downsample_factor = 2` is the
//     published default: 256 ViT tokens → 64 projected tokens.
//   • Multi-modal projector — `LayerNorm` over the pixel-unshuffled tokens
//     (dim = `hiddenSize * downsampleFactor²`), then `linear_1` (GELU) →
//     `linear_2` projecting into the text hidden dim.
//   • LFM2 text backbone — the existing `LFM2Model` stack-interleaved
//     hybrid, loaded from the `language_model.`-prefixed sub-tree.
//
// This file is the family orchestrator (load entry-point + the
// `image_token_index` token id). Vision tower internals live in
// `Models/Vision/LFM2VLVision.swift`.

import Foundation
import Metal

// ─── Errors ──────────────────────────────────────────────────────────

public enum LFM2VLError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)
    case unsupportedConfig(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "LFM2VL: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "LFM2VL: checkpoint is missing tensor '\(name)'"
        case .unsupportedConfig(let m):
            return "LFM2VL: unsupported config: \(m)"
        }
    }
}

// ─── Family entry point ──────────────────────────────────────────────

public enum LFM2VL {
    /// Architecture string the checkpoint declares.
    public static let architectures: Set<String> =
        ["Lfm2VlForConditionalGeneration"]

    /// `image_token_index` for LFM2-VL checkpoints.
    public static let defaultImageTokenId = 396

    /// Build a `VLModel` from a `Lfm2VlForConditionalGeneration`
    /// checkpoint: SigLIP2 vision tower + pixel-unshuffle projector +
    /// LFM2 text backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> VLModel {
        guard let visionConfigRaw = config.nested("vision_config"),
              let textConfigRaw  = config.nested("text_config")
        else {
            throw LFM2VLError.missingConfig
        }

        // ── Text backbone (LFM2 — quantized-aware) ──────────────────
        // Re-wrap the text_config sub-dict as a flat ModelConfig so the
        // standalone text-backbone loader resolves the right fields.
        // LFM2VL text checkpoints may be quantized (4-bit); use the
        // quantized-aware loader that calls loadLinear / loadEmbedding.
        let textConfig = ModelConfig(
            architecture: "Lfm2ForCausalLM",
            modelType: "lfm2",
            raw: textConfigRaw)
        let textWeights = weights.prefixed("language_model.")
        // Propagate the top-level quantization block into the text config
        // so loadLinear sees it (HF VLM configs put it at the top level).
        let quant = config.quantization
        let textEngine = try lfm2LoadModelQuantized(
            config: textConfig, weights: textWeights,
            quantization: quant, device: device)

        // ── SigLIP2 vision tower ─────────────────────────────────────
        let visionConfig = ModelConfig(
            architecture: nil,
            modelType: visionConfigRaw["model_type"] as? String,
            raw: visionConfigRaw)

        // The config's `vision_feature_layer` controls how many encoder
        // layers run. -2 (the published default) means use the second-to-last
        // layer's output. Map to a concrete layer count:
        //   actualLayer = numHiddenLayers + visionFeatureLayer  (when < 0)
        //   numLayers   = actualLayer + 1
        let numHiddenLayers = visionConfig.int("num_hidden_layers") ?? 27
        let visionFeatureLayerRaw = config.int("vision_feature_layer") ?? -2
        let activeLayers: Int
        if visionFeatureLayerRaw < 0 && visionFeatureLayerRaw > -numHiddenLayers {
            let actual = numHiddenLayers + visionFeatureLayerRaw
            activeLayers = actual + 1
        } else {
            activeLayers = numHiddenLayers
        }

        let visionWeights = weights.prefixed("vision_tower.")
        let visionEncoder = try lfm2vlLoadVisionEncoder(
            config: visionConfig, activeLayers: activeLayers,
            weights: visionWeights, device: device)

        // ── Multi-modal projector (pixel-unshuffle + MLP) ────────────
        let downsample = config.int("downsample_factor") ?? 2
        let projHidden = config.int("projector_hidden_size") ?? 2560
        let visionHidden = visionConfig.int("hidden_size") ?? 1152
        // Input dim after pixel-unshuffle: visionHidden * downsample²
        let unshuffledDim = visionHidden * downsample * downsample
        let projWeights = weights.prefixed("multi_modal_projector.")
        let projector = try LFM2VLProjector.load(
            unshuffledDim: unshuffledDim,
            projHidden: projHidden,
            textHidden: textEngine.hidden,
            downsampleFactor: downsample,
            weights: projWeights,
            quantization: quant,
            device: device)

        // imageTokenCount = numPatches / downsample² (after pixel-unshuffle)
        let numPatches = visionConfig.int("num_patches") ?? 256
        let imageTokenCount = numPatches / (downsample * downsample)

        // Compose the vision tower + projector behind a VisionEncoder
        // facade so VLModel's splice sees a single encode surface.
        let numPatches1D = Int(Double(numPatches).squareRoot().rounded())
        let patchSize = visionConfig.int("patch_size") ?? 16
        let imageSize = numPatches1D * patchSize   // natural resolution (256)
        let composed = LFM2VLComposedTower(
            encoder: visionEncoder,
            projector: projector,
            imageSize: imageSize,
            imageTokenCount: imageTokenCount,
            textHidden: textEngine.hidden,
            dtype: textEngine.dtype)

        let imageTokenId = config.int("image_token_index") ?? defaultImageTokenId
        return try VLModel(
            visionEncoder: composed.asVisionEncoder(),
            engine: textEngine,
            imageTokenId: imageTokenId,
            normalization: .siglip,
            imageTokenCount: imageTokenCount)
    }
}
