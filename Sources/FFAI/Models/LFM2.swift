// LFM2 family root — LiquidAI's Liquid Foundation Models 2 line
// (`lfm2` / `lfm2_moe` model_type) and the LFM2-VL orchestrator
// (`Lfm2VlForConditionalGeneration`).
//
// This file is the **main model interface** for the family:
//   • the text family enum `LFM2` (modelTypes, architectures, variant
//     dispatch — picks dense or MoE),
//   • the `LFM2Variant` protocol every concrete text variant conforms
//     to,
//   • the `LFM2Error` type the loader / decode site raises,
//   • the VL family orchestrator `LFM2VL` (and its `LFM2VLError`
//     type) — the SigLIP2 vision tower + pixel-unshuffle projector +
//     LFM2 text backbone splice.
//
// Concrete text variants + the hybrid decoder + per-layer impl live
// under `Models/Text/LFM2Text.swift`:
//   - `LFM2Dense` / `LFM2MoE` — stack-interleaved conv + attention
//     mixers; per-layer SwiGLU MLP (`lfm2`) or block-sparse MoE FFN
//     (`lfm2_moe`).
//   - `LFM2LayerKind`, `LFM2ConvCache`, `LFM2Model` — per-layer +
//     full-model impl.
//
// Vision tower internals live in `Models/Vision/LFM2Vision.swift`.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum LFM2 {
    public static let modelTypes: Set<String> = ["lfm2", "lfm2_moe"]
    public static let architectures: Set<String> =
        ["Lfm2ForCausalLM", "Lfm2MoeForCausalLM"]

    /// True when the config names the mixture-of-experts checkpoint.
    static func isMoE(_ config: ModelConfig) -> Bool {
        config.modelType == "lfm2_moe"
            || config.architecture == "Lfm2MoeForCausalLM"
    }

    public static func variant(for config: ModelConfig) throws -> any LFM2Variant.Type {
        return isMoE(config) ? LFM2MoE.self : LFM2Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol LFM2Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LFM2Model
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum LFM2Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "LFM2: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "LFM2: unsupported config: \(m)"
        }
    }
}

// ─── LFM2-VL orchestrator ────────────────────────────────────────────
//
// LFM2-VL (`Lfm2VlForConditionalGeneration`) composes:
//   • SigLIP2 vision tower (standard ViT, OIHW patch embed),
//   • pixel-unshuffle collapsing a `downsample_factor²` patch
//     neighbourhood into one super-patch,
//   • multi-modal projector (LayerNorm → linear_1 (GELU) → linear_2),
//   • LFM2 text backbone loaded from `language_model.`-prefixed
//     weights.

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

    /// Build a `VisionModel` from a `Lfm2VlForConditionalGeneration`
    /// checkpoint: SigLIP2 vision tower + pixel-unshuffle projector +
    /// LFM2 text backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> VisionModel {
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
        // facade so VisionModel's splice sees a single encode surface.
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
        return try VisionModel(
            visionEncoder: composed.asVisionEncoder(),
            engine: textEngine,
            imageTokenId: imageTokenId,
            normalization: .siglip,
            imageTokenCount: imageTokenCount)
    }
}
