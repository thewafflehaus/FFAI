// Qwen 2-VL — Alibaba's Qwen2-VL vision-language model (the
// `Qwen2VLForConditionalGeneration` checkpoints).
//
// Composition:
//   • Qwen 2-VL vision tower — a dynamic-resolution ViT that is the
//     predecessor to Qwen 2.5-VL and Qwen 3-VL, with:
//       – patch-embed via a flattened Conv3d (each input patch is a
//         `in_ch · temporalPatch · patch · patch` row, projected by a
//         single GEMM),
//       – LayerNorm (not RMSNorm) pre-norms,
//       – 2D rotary position embedding (M-RoPE) over the patch grid,
//       – full bidirectional attention on every block (no windowing),
//       – a GELU-MLP feed-forward (`mlp.fc1` / `mlp.fc2`), not the
//         SwiGLU used in Qwen 2.5-VL,
//       – a patch-merger that pools each `mergeSize × mergeSize`
//         neighbourhood and projects into the text hidden dim.
//   • Qwen 2 text backbone — the existing LlamaDense engine (Qwen 2.x
//     routes through LlamaDense), loaded from the top-level config with
//     the `language_model.`-prefixed weight sub-tree.
//
// The two are joined by `VisionModel`'s cross-modal token splice: each
// `<|image_pad|>` placeholder (`image_token_id`) in the prompt takes one
// of the merged vision tokens.
//
// Key differences from Qwen 2.5-VL:
//   – LayerNorm (with bias) not RMSNorm in the vision blocks.
//   – GELU MLP (`fc1`/`fc2`) not SwiGLU (`gate_proj`/`up_proj`/`down_proj`).
//   – No windowed attention schedule — all blocks use full attention.
//   – No learned position embedding table — pure M-RoPE.
//   – vision_config uses `embed_dim` (not `hidden_size`) and `mlp_ratio`
//     (not `intermediate_size`).
//   – Merger norm key is `merger.ln_q` (LayerNorm with bias, not RMSNorm).
//   – Merger MLP keys are `merger.mlp.0` / `merger.mlp.2`.
//
// Coherence-first port: vision attention + M-RoPE run on the CPU.
// The text M-RoPE is approximated by VisionModel's sequential scalar positions;
// the splice itself is exact.
//
// The vision tower internals live in `Models/Vision/Qwen2Vision.swift` —
// this file is the family orchestrator (load entrypoint + the
// `<|image_pad|>` / `<|video_pad|>` token ids the splice needs).

import Foundation
import Metal

public enum Qwen2VLError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "Qwen2VL: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "Qwen2VL: checkpoint is missing tensor '\(name)'"
        }
    }
}

public enum Qwen2VL {
    /// `image_token_id` default for Qwen 2-VL checkpoints.
    public static let defaultImageTokenId = 151_655
    /// `video_token_id` default for Qwen 2-VL checkpoints
    /// (`<|video_pad|>` — same id as Qwen 2.5-VL).
    public static let defaultVideoTokenId = 151_656

    /// Capabilities a Qwen 2-VL checkpoint declares to the loader.
    /// Text + image + video — the vision tower's Conv3d patch embed and
    /// temporal-patch unfold handle both single-image and multi-frame
    /// video paths.
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.visionIn, .videoIn])

    /// Build a `VisionModel` from a `Qwen2VLForConditionalGeneration`
    /// checkpoint: the dynamic-resolution vision tower + the Qwen 2
    /// text backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Qwen2VLError.missingConfig
        }

        // ── Text backbone ──
        // Qwen2-VL stores text hyper-parameters at the top level (same
        // layout as a standalone Qwen 2 checkpoint). Text weights are
        // under `language_model.*`; the LlamaDense engine handles the
        // Qwen 2.x architecture (tied embeddings, SiLU MLP, GQA).
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try LlamaDense.loadModel(
            config: config, weights: textWeights, options: options, device: device)

        // ── Vision tower ──
        // Vision weights are under `vision_tower.*`.
        let visionWeights = weights.prefixed("vision_tower.")
        let vision = try Qwen2VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: visionWeights, dtype: textEngine.dtype, device: device)

        let imageTokenId = config.int("image_token_id") ?? defaultImageTokenId
        let videoTokenId = config.int("video_token_id") ?? defaultVideoTokenId
        return try VisionModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .clip,
            imageTokenCount: vision.mergedTokenCount)
    }
}
