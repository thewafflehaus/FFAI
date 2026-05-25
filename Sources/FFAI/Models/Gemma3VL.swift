// Gemma 3 VL — Google's Gemma 3 vision-language model (the 4B / 12B /
// 27B `Gemma3ForConditionalGeneration` checkpoints).
//
// Composition:
//   • SigLIP vision tower — a standard ViT, loaded straight into the
//     shared `VisionEncoder` (its `vision_tower.vision_model.*` weight
//     keys match `VisionEncoder.parameters()` exactly).
//   • Multi-modal projector — `4×4` average-pool of the `64×64` patch
//     grid down to `16×16 = 256` tokens (`mm_tokens_per_image`), a
//     GemmaRMSNorm, then a linear projection into the text hidden dim.
//   • Gemma 3 text backbone — the existing `Gemma3Model`, loaded from
//     the `language_model.`-prefixed sub-tree with the checkpoint's
//     `text_config`.
//
// The three are joined by `VLModel`'s cross-modal token splice: each
// `<image>` placeholder (`image_token_index`) in the prompt takes one
// of the 256 projected vision tokens.
//
// This file is the family orchestrator (load entry-point + the
// `<image>` token id). Vision tower internals live in
// `Models/Vision/Gemma3VLVision.swift`.

import Foundation
import Metal

public enum Gemma3VL {
    /// `image_token_index` default for Gemma 3 VL checkpoints.
    public static let defaultImageTokenId = 262_144

    /// Build a `VLModel` from a `Gemma3ForConditionalGeneration`
    /// checkpoint: SigLIP `VisionEncoder` + projector + Gemma 3 text
    /// backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VLModel {
        guard let visionConfig = config.subConfig("vision_config"),
              let textConfigRaw = config.nested("text_config")
        else {
            throw Gemma3Error.missingConfig
        }

        // ── Text backbone — load from the language_model. sub-tree ──
        // A VLM `text_config` is sparse: HF omits every field that
        // matches the Gemma 3 text-model class default. Merge those
        // defaults in so the standalone `Gemma3Dense` loader — which
        // needs explicit `num_attention_heads`, `rms_norm_eps`,
        // `vocab_size`, etc. — sees a complete config.
        let textConfig = ModelConfig(
            architecture: "Gemma3TextForCausalLM",
            modelType: "gemma3_text",
            raw: gemma3TextConfigWithDefaults(textConfigRaw,
                                              vocabFallback: config.int("vocab_size")))
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try Gemma3Dense.loadModel(
            config: textConfig, weights: textWeights,
            options: options, device: device)

        // ── SigLIP vision tower ──
        let visionWeights = weights.prefixed("vision_tower.vision_model.")
        let visionEncoder = try gemma3vlLoadVisionEncoder(
            config: visionConfig, textHidden: textEngine.hidden,
            weights: visionWeights, device: device)

        // ── Multi-modal projector ──
        let mmTokensPerImage = config.int("mm_tokens_per_image") ?? 256
        let projector = try Gemma3VLProjector.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            mmTokensPerImage: mmTokensPerImage, weights: weights,
            device: device)

        // The projector pools the encoder grid down to mmTokensPerImage
        // tokens, so the VLModel's image-token count is the pooled
        // count — wrap the encoder + projector behind a composed
        // `VisionEncoder`-shaped tower.
        let composedTower = Gemma3VLVisionTower(
            encoder: visionEncoder, projector: projector,
            tokensPerImage: mmTokensPerImage, textHidden: textEngine.hidden,
            dtype: textEngine.dtype)

        let imageTokenId = config.int("image_token_index") ?? defaultImageTokenId
        return try VLModel(
            visionEncoder: composedTower.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: .siglip,
            imageTokenCount: mmTokensPerImage)
    }
}
