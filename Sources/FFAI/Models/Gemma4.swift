// Gemma 4 VL — Google's Gemma 4 vision-language model (the
// `Gemma4ForConditionalGeneration` checkpoints).
//
// Composition:
//   • Gemma 4 vision tower — a bespoke ViT that differs substantially
//     from the shared SigLIP `VisionEncoder`:
//       – patch-embed via a flattened linear projection (each
//         `3 · patch · patch` patch row is projected by one GEMM, after
//         a `2·(x − 0.5)` re-centring),
//       – a learned 2D position embedding: two `[positionEmbeddingSize,
//         hidden]` tables (one per spatial axis) summed by the patch's
//         `(x, y)` grid coordinate,
//       – RoPE attention with multi-dimensional positions (the head dim
//         is split into an x-rotary half and a y-rotary half) and
//         per-projection q / k / v RMSNorms,
//       – four GemmaRMSNorm "zero-shift" norms per block (input /
//         post-attention / pre-feedforward / post-feedforward) and a
//         GELU-gated SwiGLU MLP,
//       – an attention-pooling head that pools the patch grid down to
//         `default_output_length` soft tokens, then a `sqrt(hidden)`
//         scale and an optional standardization affine.
//   • Multi-modal embedder — a GemmaRMSNorm (no-scale) over the pooled
//     vision tokens followed by a linear projection into the text
//     hidden dim (`embed_vision.*`).
//   • Gemma 4 text backbone — the existing `Gemma4Model` engine, loaded
//     from the `language_model.`-prefixed sub-tree (the `Gemma4Loader`
//     probes the prefix and reads `text_config` itself).
//
// The three are joined by `VisionModel`'s cross-modal token splice: each
// `<image_soft_token>` placeholder (`image_token_id`) in the prompt
// takes one of the pooled, projected vision tokens.
//
// This file is the family orchestrator (load entry-point + the
// `<image_soft_token>` token id). Vision tower internals live in
// `Models/Vision/Gemma4Vision.swift`.

import Foundation
import Metal

public enum Gemma4VLError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "Gemma4VL: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "Gemma4VL: checkpoint is missing tensor '\(name)'"
        }
    }
}

public enum Gemma4VL {
    /// `image_token_id` default for Gemma 4 VL checkpoints.
    public static let defaultImageTokenId = 262_144

    /// Build a `VisionModel` from a `Gemma4ForConditionalGeneration`
    /// checkpoint: the Gemma 4 vision tower + multi-modal embedder +
    /// Gemma 4 text backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Gemma4VLError.missingConfig
        }

        // ── Text backbone — Gemma 4 engine ──
        // `Gemma4Loader` probes the weight prefix (`language_model.model.`
        // for a VLM conversion) and reads the text hyper-parameters from
        // the `text_config` sub-tree itself, so the whole `config` and
        // `weights` bundle is forwarded unchanged; the variant (dense /
        // E / MoE) is config-driven.
        let textVariant = try Gemma4.variant(for: config)
        let textEngine = try textVariant.loadModel(
            config: config, weights: weights, options: options, device: device)

        // ── Vision tower + multi-modal embedder ──
        let vision = try Gemma4VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: weights, dtype: textEngine.dtype,
            quantization: config.quantization, device: device)

        let imageTokenId = config.int("image_token_id")
            ?? config.int("image_token_index") ?? defaultImageTokenId
        return try VisionModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: .siglip,
            imageTokenCount: vision.tokensPerImage)
    }
}
