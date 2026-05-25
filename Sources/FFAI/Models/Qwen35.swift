// Qwen 3-VL-MoE — Alibaba's mixture-of-experts Qwen3-VL vision-language
// model (the `Qwen3VLMoeForConditionalGeneration` checkpoints).
//
// Composition:
//   • Qwen 3-VL vision tower — bit-identical to the dense Qwen3-VL tower
//     (`Qwen3VLVisionModel` in `Qwen3VL.swift`): a dynamic-resolution
//     ViT with a flattened Conv3d patch-embed, LayerNorm pre-norms, a
//     learned position table, 2D M-RoPE, full bidirectional attention,
//     a GELU-MLP feed-forward, and a patch-merger into the text hidden
//     dim. The MoE variant changes only the *text* half.
//   • Qwen 3.5-MoE text backbone — the existing `Qwen35Model` hybrid
//     engine (Gated Delta Net ↔ full-attention alternation, block-sparse
//     MoE FFN with an always-on shared expert). Qwen3-VL-MoE stores its
//     text hyper-parameters under `text_config`, which `Qwen35.loadModel`
//     reads natively, so the whole config is forwarded unchanged and the
//     text weights are taken from the `language_model.`-prefixed
//     sub-tree.
//
// The two are joined by `VisionModel`'s cross-modal token splice. Coherence
// caveats are the same as the dense Qwen3-VL port (CPU vision attention,
// scalar text positions, deepstack omitted) — see `Qwen3VL.swift`.

import Foundation
import Metal

public enum Qwen3VLMoeError: Error, CustomStringConvertible {
    case missingConfig

    public var description: String {
        switch self {
        case .missingConfig:
            return "Qwen3VLMoe: checkpoint config is missing required fields"
        }
    }
}

public enum Qwen3VLMoe {
    /// `image_token_id` default for Qwen 3-VL-MoE checkpoints.
    public static let defaultImageTokenId = 151_655

    /// `video_token_id` default for Qwen 3-VL-MoE checkpoints
    /// (`<|video_pad|>`, same id as the dense Qwen 3-VL).
    public static let defaultVideoTokenId = 151_656

    /// Capabilities a Qwen 3-VL-MoE checkpoint exposes. Identical to the
    /// dense Qwen 3-VL — the MoE variant only swaps the text backbone;
    /// the vision tower (and its multi-frame `encode(frames:)` path) is
    /// shared with `Qwen3VL`.
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.visionIn, .videoIn])

    /// Build a `VisionModel` from a `Qwen3VLMoeForConditionalGeneration`
    /// checkpoint: the Qwen3-VL vision tower + the Qwen 3.5-MoE text
    /// backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Qwen3VLMoeError.missingConfig
        }

        // ── Text backbone — Qwen 3.5-MoE hybrid engine ──
        // `Qwen35.loadModel` reads its text hyper-parameters from the
        // `text_config` sub-tree itself (and decides dense-vs-MoE from
        // `num_experts`), so the whole `config` is forwarded unchanged;
        // only the weight sub-tree is narrowed to `language_model.*`.
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try Qwen35Hybrid.loadModel(
            config: config, weights: textWeights,
            options: options, device: device)

        // ── Vision tower — identical to the dense Qwen3-VL tower ──
        // Including the multi-frame `encode(frames:)` path: the MoE
        // variant ships the same vision weights at the same prefix and
        // produces tokens with the same temporal-patch unfold.
        let visionWeights = weights.prefixed("model.visual.")
        let vision = try Qwen3VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: visionWeights, dtype: textEngine.dtype, device: device)

        let imageTokenId = config.int("image_token_id")
            ?? config.int("image_token_index") ?? defaultImageTokenId
        let videoTokenId = config.int("video_token_id")
            ?? config.int("video_token_index") ?? defaultVideoTokenId
        return try VisionModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .clip,
            imageTokenCount: vision.mergedTokenCount)
    }
}
