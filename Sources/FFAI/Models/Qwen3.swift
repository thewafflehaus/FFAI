// Qwen 3-VL — Alibaba's Qwen3-VL vision-language model (the
// `Qwen3VLForConditionalGeneration` checkpoints).
//
// Composition:
//   • Qwen 3-VL vision tower — a dynamic-resolution ViT that, unlike the
//     Qwen 2.5-VL tower, uses:
//       – patch-embed via a flattened Conv3d (each input patch is a
//         `in_ch · temporalPatch · patch · patch` row, projected by a
//         single GEMM),
//       – LayerNorm (not RMSNorm) pre-norms,
//       – a learned, bilinearly-interpolated position embedding looked
//         up from a `[numPositionEmbeddings, hidden]` table,
//       – 2D rotary position embedding (M-RoPE) over the patch grid,
//       – full bidirectional attention on every block (no windowing —
//         Qwen3-VL dropped the windowed-attention schedule),
//       – a GELU-MLP feed-forward (`linear_fc1` / `linear_fc2`), not the
//         Qwen 2.5-VL SwiGLU,
//       – a patch-merger that pools each `mergeSize × mergeSize`
//         neighbourhood and projects into the text hidden dim.
//   • Qwen 3 text backbone — the existing `Qwen3Model` dense engine,
//     loaded from the `language_model.`-prefixed sub-tree (Qwen3-VL
//     stores text weights under `language_model.model.*` /
//     `language_model.lm_head.*`).
//
// The two are joined by `VLModel`'s cross-modal token splice: each
// `<|image_pad|>` placeholder (`image_token_id`) in the prompt takes one
// of the merged vision tokens.
//
// Coherence-first port: the vision tower's attention + M-RoPE run on the
// CPU (vision token counts are at most a few thousand, so an O(n²·d)
// attention is cheap next to the GPU projection GEMMs and is
// unambiguously correct). The text M-RoPE — Qwen's 3D position scheme —
// is approximated by `VLModel`'s sequential scalar positions; the splice
// itself is exact. The Qwen3-VL `deepstack` feature (injecting
// intermediate vision features into the text stack) is omitted in this
// coherence-first port — only the final merged tokens are spliced. A
// head-dim-agnostic GPU vision SDPA, true text M-RoPE, and deepstack are
// later performance / fidelity passes.
//
// The vision tower internals live in `Models/Vision/Qwen3Vision.swift` —
// this file is the family orchestrator (load entrypoint + the
// `<|image_pad|>` / `<|video_pad|>` token ids the splice needs).

import Foundation
import Metal

public enum Qwen3VLError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "Qwen3VL: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "Qwen3VL: checkpoint is missing tensor '\(name)'"
        }
    }
}

public enum Qwen3VL {
    /// `image_token_id` default for Qwen 3-VL checkpoints.
    public static let defaultImageTokenId = 151_655
    /// `video_token_id` default for Qwen 3-VL checkpoints
    /// (`<|video_pad|>` — same id as Qwen 2.5-VL and Qwen 2-VL).
    public static let defaultVideoTokenId = 151_656

    /// Capabilities a Qwen 3-VL checkpoint declares to the loader.
    /// Text + image + video — the vision tower's Conv3d patch embed and
    /// temporal-patch unfold handle both single-image and multi-frame
    /// video paths.
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.visionIn, .videoIn])

    /// Build a `VLModel` from a `Qwen3VLForConditionalGeneration`
    /// checkpoint: the dynamic-resolution vision tower + the Qwen 3
    /// text backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VLModel {
        guard let visionConfig = config.subConfig("vision_config"),
              let textConfigRaw = config.nested("text_config")
        else {
            throw Qwen3VLError.missingConfig
        }

        // ── Text backbone — Qwen 3 dense engine ──
        // Qwen3-VL stores text hyper-parameters under `text_config`; the
        // standalone `Qwen3Dense` loader reads top-level config keys, so
        // re-wrap the `text_config` sub-tree as a flat `ModelConfig`.
        let textConfig = ModelConfig(
            architecture: "Qwen3ForCausalLM",
            modelType: "qwen3",
            raw: textConfigRaw)
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try Qwen3Dense.loadModel(
            config: textConfig, weights: textWeights,
            options: options, device: device)

        // ── Vision tower ──
        // Vision weights live under `vision_tower.*` on the current
        // mlx-community Qwen3-VL conversion (e.g. `vision_tower.patch_embed.
        // proj.weight`, `vision_tower.merger.*`). Older preview snapshots
        // used `model.visual.*`; fall back to that prefix when the new
        // one isn't present so both naming conventions load.
        let visionWeights: SafeTensorsBundle = {
            if weights.has("vision_tower.patch_embed.proj.weight") {
                return weights.prefixed("vision_tower.")
            }
            return weights.prefixed("model.visual.")
        }()
        let vision = try Qwen3VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: visionWeights, dtype: textEngine.dtype, device: device)

        let imageTokenId = config.int("image_token_id")
            ?? config.int("image_token_index") ?? defaultImageTokenId
        let videoTokenId = config.int("video_token_id") ?? defaultVideoTokenId
        return try VLModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .clip,
            imageTokenCount: vision.mergedTokenCount)
    }
}
