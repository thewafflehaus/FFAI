// Nemotron family — NVIDIA's Nemotron model line. Unified entry point
// covering three variants:
//
//   • NemotronH (text) — the stack-interleaved hybrid (Mamba 2 /
//     attention / dense-MLP) decoder. Impl + family enum live in
//     `Models/Text/NemotronHText.swift`.
//   • NemotronH (vision-language) — Nemotron Nano VL: the ViT vision
//     tower + multi-modal projector + the NemotronH hybrid text
//     backbone. Orchestrator (`enum NemotronVL`) lives below; tower
//     internals in `Models/Vision/NemotronHVision.swift`.
//   • NemotronDiffusion (text) — Nemotron-Labs-Diffusion, the
//     tri-mode (AR / block-diffusion / self-speculation) decoder.
//     Impl + family enum live in `Models/Text/NemotronDiffusionText.swift`.
//
// The `enum Nemotron` below is the unified family root — it advertises
// the union of every variant's `modelTypes` + `architectures` so the
// ModelRegistry can ask "is this a Nemotron checkpoint?" with one
// lookup. Per-variant dispatch still routes to the right loader (Text/
// for text + diffusion, the NemotronVL block in this file for VL).

import Foundation
import Metal

/// Unified Nemotron family root. Covers the text NemotronH backbone,
/// the NemotronH vision-language wrapper, and the NemotronDiffusion
/// tri-mode decoder. Each variant retains its own per-variant enum
/// (`NemotronH`, `NemotronVL`, `NemotronDiffusion`) — this root just
/// unions the metadata so the registry can dispatch with one
/// membership check.
public enum Nemotron {
    /// Union of every Nemotron variant's HuggingFace `model_type`
    /// labels. Used by `ModelRegistry` to recognise "this is a
    /// Nemotron checkpoint" before dispatching to the per-variant
    /// loader.
    public static var modelTypes: Set<String> {
        NemotronH.modelTypes
            .union(NemotronVL.modelTypes)
            .union(NemotronDiffusion.modelTypes)
    }

    /// Union of every Nemotron variant's HuggingFace `architectures[0]`
    /// labels.
    public static var architectures: Set<String> {
        NemotronH.architectures
            .union(NemotronDiffusion.architectures)
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
