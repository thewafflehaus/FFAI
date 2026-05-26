// Nemotron family — NVIDIA's Nemotron model line. Unified entry point
// covering four variants:
//
//   • NemotronH (text) — the stack-interleaved hybrid (Mamba 2 /
//     attention / dense-MLP / MoE) decoder. Covers the dense
//     Nemotron-H-4B-Base lineage AND the MoE Cascade-2-30B-A3B /
//     Nemotron-3 Nano/Super/Ultra lineage (all `nemotron_h` model_type).
//     Impl + family enum live in `Models/Text/NemotronHText.swift`.
//   • NemotronH (vision-language) — Nemotron Nano VL: the ViT vision
//     tower + multi-modal projector + the NemotronH hybrid text
//     backbone. Orchestrator (`enum NemotronVL`) lives below; tower
//     internals in `Models/Vision/NemotronHVision.swift`.
//   • NemotronDiffusion (text) — Nemotron-Labs-Diffusion, the
//     tri-mode (AR / block-diffusion / self-speculation) decoder.
//     Impl + family enum live in `Models/Text/NemotronDiffusionText.swift`.
//   • NemotronDiffusion (vision-language) — Nemotron-Labs-Diffusion-
//     VLM-8B: the Pixtral 2D-RoPE ViT + Mistral3-style patch-merger
//     projector + the NemotronDiffusion tri-mode text backbone.
//     Orchestrator (`enum NemotronDiffusionVL`) lives below; reuses
//     `Models/Vision/PixtralVision.swift` for the tower and
//     `Models/Vision/Mistral3Vision.swift` for the projector.
//
// The `enum Nemotron` below is the unified family root — it advertises
// the union of every variant's `modelTypes` + `architectures` so the
// ModelRegistry can ask "is this a Nemotron checkpoint?" with one
// lookup. Per-variant dispatch still routes to the right loader (Text/
// for text + diffusion, the NemotronVL block in this file for VL).

import Foundation
import Metal

/// Unified Nemotron family root. Covers the text NemotronH backbone
/// (dense + MoE), the NemotronH vision-language wrapper, the
/// NemotronDiffusion tri-mode decoder, and its vision-language
/// wrapper. Each variant retains its own per-variant enum
/// (`NemotronH`, `NemotronVL`, `NemotronDiffusion`,
/// `NemotronDiffusionVL`) — this root just unions the metadata so the
/// registry can dispatch with one membership check.
public enum Nemotron {
    /// Union of every Nemotron variant's HuggingFace `model_type`
    /// labels. Used by `ModelRegistry` to recognise "this is a
    /// Nemotron checkpoint" before dispatching to the per-variant
    /// loader.
    public static var modelTypes: Set<String> {
        NemotronH.modelTypes
            .union(NemotronVL.modelTypes)
            .union(NemotronDiffusion.modelTypes)
            .union(NemotronDiffusionVL.modelTypes)
    }

    /// Union of every Nemotron variant's HuggingFace `architectures[0]`
    /// labels.
    public static var architectures: Set<String> {
        NemotronH.architectures
            .union(NemotronDiffusion.architectures)
            .union(NemotronDiffusionVL.architectures)
        // NemotronVL doesn't ship a distinct architecture string — the
        // VL checkpoints carry `text_config.model_type = nemotron_h`
        // and the registry routes them via the vision-config sniff.
    }
}

// ─── NemotronVL — vision-language orchestrator ──────────────────────

public enum NemotronVLError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "NemotronVL: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "NemotronVL: checkpoint is missing tensor '\(name)'"
        }
    }
}

public enum NemotronVL {
    /// `model_type` labels this orchestrator recognises (the VL
    /// checkpoints actually carry `text_config.model_type = nemotron_h`
    /// — the union here is mostly for documentation / future
    /// dispatch flexibility).
    public static let modelTypes: Set<String> = []

    /// `image_token_id` fallback for Nemotron Nano VL checkpoints.
    public static let defaultImageTokenId = 131_072

    /// Build a `VisionModel` from a Nemotron Nano VL checkpoint: the ViT
    /// vision tower + multi-modal projector + the NemotronH hybrid text
    /// backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config"),
              let textConfigRaw = config.nested("text_config")
        else {
            throw NemotronVLError.missingConfig
        }

        // ── Text backbone — NemotronH hybrid engine ──
        // The standalone `NemotronHHybrid` loader reads top-level config
        // keys, so re-wrap the `text_config` sub-tree as a flat
        // `ModelConfig`.
        let textConfig = ModelConfig(
            architecture: "NemotronHForCausalLM",
            modelType: "nemotron_h",
            raw: textConfigRaw)
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try NemotronHHybrid.loadModel(
            config: textConfig, weights: textWeights,
            options: options, device: device)

        // ── ViT vision tower ──
        // The vision weights are namespaced under `vision_model.` (the
        // C-RADIO / SigLIP encoder); load straight into the shared
        // `VisionEncoder` core.
        let visionEncoder = try nemotronVLLoadVisionEncoder(
            config: visionConfig, weights: weights, device: device)

        // ── Multi-modal projector ──
        let projector = try NemotronVLProjector.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: weights, device: device)

        let composedTower = NemotronVLVisionTower(
            encoder: visionEncoder, projector: projector,
            textHidden: textEngine.hidden, dtype: textEngine.dtype)

        let imageTokenId = config.int("image_token_id")
            ?? config.int("image_token_index") ?? defaultImageTokenId
        return try VisionModel(
            visionEncoder: composedTower.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: .siglip,
            imageTokenCount: visionEncoder.config.numPatches)
    }
}

// ─── NemotronDiffusionVL — diffusion vision-language orchestrator ────

public enum NemotronDiffusionVLError: Error, CustomStringConvertible {
    case missingConfig

    public var description: String {
        switch self {
        case .missingConfig:
            return "NemotronDiffusionVL: checkpoint config is missing required fields"
        }
    }
}

/// Nemotron-Labs-Diffusion-VLM-8B — the diffusion VLM. Wraps the
/// NemotronDiffusion tri-mode text backbone with a Pixtral ViT vision
/// tower and a Mistral3-style multi-modal projector
/// (RMSNorm → 2×2 patch merger → linear_1 → GELU → linear_2). The
/// `vision_config.model_type` is explicitly `"pixtral"`, so the
/// shipped `PixtralVisionEncoder` is the correct tower and we reuse
/// `Mistral3Projector` for the merger.
///
/// Coherence-first port: vision attention runs on the CPU (already the
/// case in `PixtralVisionEncoder`); the text backbone runs through the
/// existing GPU-accelerated `NemotronDiffusionDense` engine. Image-only
/// inference for now; the `forwardBlock` diffusion / self-speculation
/// paths take the spliced embeddings unchanged.
public enum NemotronDiffusionVL {
    /// `model_type` labels this orchestrator recognises.
    public static let modelTypes: Set<String> = ["nemotron_labs_diffusion_vlm"]
    /// `architectures[0]` labels this orchestrator recognises.
    public static let architectures: Set<String> = ["NemotronLabsDiffusionVLMModel"]

    /// Patch-merger size default for the Nemotron-Labs-Diffusion VLM —
    /// the shipped 8B config sets `spatial_merge_size: 2`.
    public static let defaultSpatialMergeSize = 2
    /// `image_token_id` fallback. The 8B config doesn't specify one
    /// explicitly; the tokenizer's `[IMG]` (Pixtral lineage) resolves
    /// to id 10. Override via `config.image_token_id` /
    /// `image_token_index` when a checkpoint sets it.
    public static let defaultImageTokenId = 10

    /// Build a `VisionModel` from a Nemotron-Labs-Diffusion-VLM
    /// checkpoint: the Pixtral 2D-RoPE ViT + the Mistral3 patch-merger
    /// projector + the NemotronDiffusion tri-mode text backbone,
    /// joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config")
        else {
            throw NemotronDiffusionVLError.missingConfig
        }

        // ── Text backbone — NemotronDiffusion tri-mode engine ──
        // The diffusion VLM stores text hyper-parameters at the *root*
        // of the config (not under a nested `text_config` — confirmed
        // against the shipped 8B config.json). The text weights live
        // under `language_model.*` (standard VLM namespacing) so the
        // existing `encoder.*` / `diffusion_head.*` keys the standalone
        // loader expects become `language_model.encoder.*` /
        // `language_model.diffusion_head.*` once we prefix the bundle.
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try NemotronDiffusionDense.loadModel(
            config: config, weights: textWeights,
            options: options, device: device)

        // ── Vision tower — Pixtral ViT ──
        // `vision_config.model_type` is `"pixtral"`. Reuse the shipped
        // `PixtralVisionEncoder` verbatim. Weights live under
        // `vision_tower.*` (the mlx-community VLM convention); the
        // encoder reads them as `vision_model.*` so the prefix strip
        // matches what Mistral3 does for the same tower.
        let visionWeights = weights.prefixed("vision_tower.")
        let visionCfg = try PixtralVisionConfig.decode(visionConfig)
        let vision = try PixtralVisionEncoder.load(
            cfg: visionCfg, weights: visionWeights,
            dtype: textEngine.dtype, device: device)

        // ── Multi-modal projector — Mistral3 patch-merger ──
        // The 8B config sets `multimodal_projector_bias: false` and
        // `projector_hidden_act: "gelu"` — Mistral3Projector's defaults
        // match (GELU between linear_1 and linear_2, biases optional).
        // The merger reduces token count by spatialMergeSize².
        let spatialMergeSize = config.int("spatial_merge_size") ?? defaultSpatialMergeSize
        let projectorBias = config.bool("multimodal_projector_bias") ?? false
        let projQuant = weights.mistral3ProjectorQuantization
        let projector = try Mistral3Projector.load(
            visionHidden: visionCfg.hiddenSize,
            textHidden: textEngine.hidden,
            spatialMergeSize: spatialMergeSize,
            hasBias: projectorBias,
            quantization: projQuant,
            weights: weights, device: device)

        // Image token count for the default square-image path; the
        // composed encoder adjusts at encode time for dynamic images.
        let mergedPatches = visionCfg.numPatches
            / (spatialMergeSize * spatialMergeSize)

        let imageTokenId = config.int("image_token_id")
            ?? config.int("image_token_index")
            ?? defaultImageTokenId

        let composedTower = Mistral3ComposedTower(
            encoder: vision, projector: projector,
            visionCfg: visionCfg, spatialMergeSize: spatialMergeSize,
            textHidden: textEngine.hidden, dtype: textEngine.dtype)

        return try VisionModel(
            visionEncoder: composedTower.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: .clip,
            imageTokenCount: mergedPatches)
    }
}
