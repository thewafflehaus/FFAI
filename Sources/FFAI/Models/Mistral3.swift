// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Mistral3 — Mistral AI's Mistral Small 3.1 vision-language model
// (`mistral3` model_type / `Mistral3ForConditionalGeneration` architecture).
//
// Composition:
//   • Pixtral ViT vision tower — the same custom 2D-RoPE vision encoder
//     used by Pixtral-12B: patch embedding via Conv2d (no bias), RMSNorm
//     pre-norms, 2D Rotary Position Embeddings, SiLU-gated MLP, full
//     bidirectional attention. Weight keys live under `vision_tower.vision_model.*`.
//
//   • Mistral3 multi-modal projector — differs from Pixtral's two-layer MLP:
//       1. RMSNorm on the vision features.
//       2. Patch merger: a 2×2 spatial unfold that concatenates each 2×2
//          neighbourhood of patches into one token, then projects
//          [visionHidden * 4] → [visionHidden] via a linear (no bias by default).
//          Token count shrinks from H×W to (H/2)×(W/2).
//       3. linear_1: [visionHidden → textHidden] — with optional bias.
//       4. GELU activation.
//       5. linear_2: [textHidden → textHidden] — with optional bias.
//
//   • Mistral text backbone — the existing `LlamaDense` engine (Mistral
//     is architecturally identical to Llama). Weights live under
//     `language_model.*` (prefixed bundle).
//
// The three are joined by `VisionModel`'s cross-modal token splice.
//
// Image token count:
//   With patchSize=14 and spatialMergeSize=2, a P×P patch grid produces
//   (P/2)² merged tokens. The mlx-community Mistral-Small-3.1-24B-4bit
//   conversion uses image_size=1540 → 110×110 patches → 55×55 = 3025
//   tokens at the default input size. The count is dynamic and is inferred
//   from the actual padded image dimensions at encode time.
//
// Projector note (quantized weights):
//   The mlx-community 4-bit conversion quantizes the projector linears
//   (`patch_merger.merging_layer`, `linear_1`, `linear_2`). These are
//   loaded as `AnyLinear` via `loadLinear`. GPU batched GEMM (`Ops.gemm`)
//   only works on float-dtype weights; for quantized layers the dequant
//   kernel (`Ops.dequantGemv`) requires a single-row input. The helpers
//   `mistral3ApplyLinearRows` and `mistral3BroadcastAddBias` handle both
//   cases correctly — plain: one `Ops.gemm`; quantized: per-row gemv loop.
//
// Vision tower internals (config structs, projector, composed encoder,
// helper types) live in `Models/Vision/Mistral3Vision.swift`.

import Foundation
import Metal

// ─── Errors ──────────────────────────────────────────────────────────

public enum Mistral3Error: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "Mistral3: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "Mistral3: checkpoint is missing tensor '\(name)'"
        }
    }
}

// ─── Family registry ─────────────────────────────────────────────────

public enum Mistral3 {
    /// `model_type` values that identify a Mistral3 checkpoint.
    public static let modelTypes: Set<String> = ["mistral3"]

    /// Architecture strings used by Mistral3-family HF conversions.
    public static let architectures: Set<String> = [
        "Mistral3ForConditionalGeneration"
    ]

    /// Default `image_token_id` for Mistral3. The tokenizer's `[IMG]`
    /// special token resolves to id 10.
    public static let defaultImageTokenId = 10

    /// Default spatial merge size (2×2 pooling in the patch merger).
    public static let defaultSpatialMergeSize = 2

    /// Build a `VisionModel` from a Mistral3 checkpoint: the Pixtral 2D-RoPE ViT
    /// + the Mistral3 patch-merger projector + the Mistral text backbone,
    /// joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config"),
            let textConfig = config.subConfig("text_config")
        else {
            throw Mistral3Error.missingConfig
        }

        // ── Text backbone ──
        // Mistral3 VLM text weights live under `language_model.*`. The
        // text hyper-parameters live in `text_config`; fill in any fields
        // the sparse VLM config omits from Mistral's documented defaults.
        let mergedTextConfig = ModelConfig(
            architecture: "MistralForCausalLM",
            modelType: textConfig.modelType ?? "mistral",
            raw: pixtralTextConfigWithDefaults(
                textConfig.raw,
                vocabFallback: config.int("vocab_size")))
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try LlamaDense.loadModel(
            config: mergedTextConfig, weights: textWeights,
            options: options, device: device)

        // ── Vision tower ──
        // Mistral3 ships the same Pixtral ViT. Weights are stored under
        // `vision_tower.vision_model.*` in the mlx-community conversion —
        // passing `weights.prefixed("vision_tower.")` to PixtralVisionEncoder
        // exposes them as `vision_model.*`, which is what the encoder expects.
        let visionWeights = weights.prefixed("vision_tower.")
        let visionCfg = try PixtralVisionConfig.decode(visionConfig)
        let vision = try PixtralVisionEncoder.load(
            cfg: visionCfg, weights: visionWeights,
            dtype: textEngine.dtype, device: device)

        // ── Multi-modal projector ──
        // Mistral3 uses a patch-merger projector rather than Pixtral's
        // simple two-layer MLP. The projector is loaded from the top-level
        // `multi_modal_projector.*` weight prefix. The mlx-community 4-bit
        // conversion quantizes the projector linears; `quantization` is
        // derived from the bundle's key shapes so mixed-precision
        // checkpoints are handled correctly.
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

        // Image token count: each spatialMergeSize×spatialMergeSize
        // neighbourhood collapses to one token, so the total is
        // patchesPerSide²/spatialMergeSize². This is for the default
        // square-image path; dynamic images adjust at encode time via
        // the Mistral3ComposedEncoder override.
        let mergedPatches = visionCfg.numPatches / (spatialMergeSize * spatialMergeSize)

        let imageTokenId =
            config.int("image_token_id")
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
