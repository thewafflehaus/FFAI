// Nemotron-VLM — NVIDIA's Nemotron Nano vision-language model (the
// Nemotron Nano VL `*ForConditionalGeneration` checkpoints).
//
// Composition:
//   • Vision tower — a ViT (the C-RADIO / SigLIP-shaped encoder NVIDIA
//     pairs with the Nemotron Nano backbone): a conv2d patch-embed, a
//     learned position embedding, a LayerNorm-pre-norm block stack with
//     bidirectional multi-head attention and a GELU MLP, and a
//     post-LayerNorm. This is exactly the shared `VisionEncoder` core,
//     loaded straight into it.
//   • Multi-modal projector — a two-layer MLP (`linear_1` → GELU →
//     `linear_2`) that maps the encoder hidden into the NemotronH text
//     hidden dim. Pixel-shuffle / patch-pooling variants reduce the
//     token count; this coherence-first port keeps one projected token
//     per encoder patch.
//   • NemotronH text backbone — the existing `NemotronHModel`
//     stack-interleaved hybrid (Mamba 2 / attention / dense-MLP layers
//     selected by a `hybrid_override_pattern`), loaded from the
//     `language_model.`-prefixed sub-tree with the checkpoint's
//     `text_config`.
//
// The three are joined by `VLModel`'s cross-modal token splice: each
// image-placeholder token (`image_token_id`) in the prompt takes one of
// the projected vision tokens.
//
// Vision tower internals (projector, composed encoder, helper types)
// live in `Models/Vision/NemotronVLVision.swift`. This file is the
// family orchestrator (load entrypoint + public constants).

import Foundation
import Metal

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
    /// `image_token_id` fallback for Nemotron Nano VL checkpoints.
    public static let defaultImageTokenId = 131_072

    /// Build a `VLModel` from a Nemotron Nano VL checkpoint: the ViT
    /// vision tower + multi-modal projector + the NemotronH hybrid text
    /// backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VLModel {
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
        return try VLModel(
            visionEncoder: composedTower.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: .siglip,
            imageTokenCount: visionEncoder.config.numPatches)
    }
}
