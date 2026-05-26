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
// Pixtral — Mistral AI's Pixtral-12B vision-language model
// (`pixtral` model_type / `LlavaForConditionalGeneration` architecture).
//
// Composition:
//   • Pixtral ViT vision tower — a custom vision transformer with:
//       – patch embedding via Conv2d (no bias),
//       – RMSNorm pre-norms (not LayerNorm),
//       – 2D Rotary Position Embeddings (unique to Pixtral),
//       – SiLU-gated MLP (gate_proj / up_proj / down_proj, same as text),
//       – full bidirectional attention (no windowing, block-diagonal mask
//         prevents cross-image attention when batching),
//       – no post-encoder LayerNorm (the features go straight to the
//         projector).
//   • Multi-modal projector — two linear layers with GELU activation
//     (`linear_1` → GELU → `linear_2`), projecting from the vision
//     hidden dim into the text hidden dim.
//   • Mistral text backbone — the existing `LlamaDense` engine (Mistral
//     is architecturally identical to Llama; it routes through the same
//     dense loader). Text weights live under `language_model.*` in the
//     mlx-community conversion.
//
// The three are joined by `VisionModel`'s cross-modal token splice: each
// `[IMG]` placeholder (`image_token_id`) in the prompt is replaced by
// one of the projected vision tokens.
//
// 2D RoPE notes:
//   The Pixtral vision tower uses 2D rotary embeddings that encode both
//   height and width position of each patch independently. Given head-dim
//   D, the first D/2 components are driven by the patch's row position
//   and the second D/2 by its column position. This differs from standard
//   1D RoPE and from Qwen's M-RoPE. The rotation is applied using the
//   standard rotate-half scheme independently on each half of the head.
//
// Coherence-first port: vision attention runs on the CPU (patch counts
// are at most a few thousand). The text backbone runs through the existing
// GPU-accelerated LlamaDense engine.
//
// Vision tower internals (config structs, 2D RoPE, attention block,
// encoder, projector, composed tower) live in `Models/Vision/PixtralVision.swift`.

import Foundation
import Metal

// ─── Errors ──────────────────────────────────────────────────────────

public enum PixtralError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "Pixtral: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "Pixtral: checkpoint is missing tensor '\(name)'"
        }
    }
}

// ─── Family registry ─────────────────────────────────────────────────

public enum Pixtral {
    /// `model_type` values that identify a Pixtral checkpoint.
    public static let modelTypes: Set<String> = ["pixtral"]

    /// Architecture strings used by Pixtral-family HF conversions.
    /// Pixtral-12B ships as `LlavaForConditionalGeneration` (the HF
    /// auto-model mapping routes `pixtral` → Llava). The mlx-community
    /// conversion uses the same architecture string.
    public static let architectures: Set<String> = [
        "LlavaForConditionalGeneration",
    ]

    /// Default `image_token_id` for Pixtral-12B-4bit mlx-community.
    /// The tokenizer's `[IMG]` special token resolves to id 10.
    public static let defaultImageTokenId = 10

    /// Build a `VisionModel` from a Pixtral checkpoint: the custom 2D-RoPE
    /// ViT + the multi-modal projector + the Mistral text backbone,
    /// joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config"),
              let textConfig = config.subConfig("text_config")
        else {
            throw PixtralError.missingConfig
        }

        // ── Text backbone ──
        // The mlx-community Pixtral conversion namespaces text weights
        // under `language_model.*`. The text hyper-parameters live in the
        // nested `text_config`; supply any fields the sparse VLM
        // text_config omits from the documented Mistral defaults.
        let mergedTextConfig = ModelConfig(
            architecture: "MistralForCausalLM",
            modelType: textConfig.modelType ?? "mistral",
            raw: pixtralTextConfigWithDefaults(textConfig.raw,
                                               vocabFallback: config.int("vocab_size")))
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try LlamaDense.loadModel(
            config: mergedTextConfig, weights: textWeights,
            options: options, device: device)

        // ── Vision tower ──
        // Pixtral vision weights live at the top level under
        // `vision_tower.*` (the mlx-community conversion follows the
        // standard VLM key layout).
        let visionWeights = weights.prefixed("vision_tower.")
        let visionCfg = try PixtralVisionConfig.decode(visionConfig)
        let vision = try PixtralVisionEncoder.load(
            cfg: visionCfg, weights: visionWeights,
            dtype: textEngine.dtype, device: device)

        // ── Multi-modal projector ──
        // `multi_modal_projector.linear_1` and `.linear_2` (both with
        // bias) bridge the vision hidden dim to the text hidden dim.
        let projector = try PixtralProjector.load(
            visionHidden: visionCfg.hiddenSize,
            textHidden: textEngine.hidden,
            weights: weights, device: device)

        // Number of image tokens = number of patches produced by the
        // vision tower for the default test resolution. For Pixtral
        // this is dynamic (image-size dependent), but the token
        // replacement count must match the placeholder count in the
        // prompt. The prompt builder determines the count from the
        // image geometry; `VisionModel.imageTokenCount` is used only for
        // the facade config numPatches — pass the true patch count.
        let numPatches = visionCfg.patchesPerSide * visionCfg.patchesPerSide

        // Wrap encoder + projector as a composed VisionEncoder so the
        // VisionModel splice sees a single tower returning [numPatches,
        // textHidden] tokens.
        let composedTower = PixtralComposedTower(
            encoder: vision, projector: projector,
            visionCfg: visionCfg, textHidden: textEngine.hidden,
            dtype: textEngine.dtype)

        let imageTokenId = config.int("image_token_id")
            ?? config.int("image_token_index")
            ?? defaultImageTokenId

        return try VisionModel(
            visionEncoder: composedTower.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: .clip,
            imageTokenCount: numPatches)
    }
}
