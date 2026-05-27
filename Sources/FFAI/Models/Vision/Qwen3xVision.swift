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
// Qwen 3.5 / 3.6-VL vision-language orchestrators — the VL siblings of
// the dense Qwen 3.5 / 3.6 text releases. Parallels `Qwen3xText.swift`
// (which holds both the Qwen 3.5 and Qwen 3.6 text variants because the
// stack-interleaved GDN ↔ attention hybrid is the same architecture
// and the same `qwen3_5*` model_type strings cover both).
//
// The Qwen 3-VL vision tower module itself (`Qwen3VLVisionConfig`,
// `Qwen3VLVisionBlock`, `Qwen3VLVisionModel`) lives in
// `Qwen3Vision.swift` and is shared verbatim: the Qwen 3.5-VL config
// only changes the per-block sizing (depth=12, hidden=768 vs Qwen 3-VL's
// 24 / 1024), the tensor layout (`attn.qkv` / `attn.proj` /
// `mlp.linear_fc1`/`linear_fc2`, `merger.*`, `patch_embed.proj`,
// `pos_embed`) and forward path are identical. `deepstack_visual_indexes`
// is the only meaningful structural difference (Qwen 3-VL has
// `[5, 11, 17]`, Qwen 3.5-VL has `[]`); FFAI does not implement deepstack
// for either variant, so both load through the same `Qwen3VLVisionModel`.
//
// This file is named `Qwen3xVision.swift` rather than `Qwen35Vision.swift`
// for the same reason `Qwen3xText.swift` is — when the Qwen 3.6-VL
// release ships (using the same architecture), its orchestrator lands
// here next to `Qwen35VL` instead of getting a new file. The type names
// retain the `Qwen35*` prefix because they were named when 3.5 was the
// only family member.

import Foundation

// ─── Qwen 3.5-VL family orchestrator ─────────────────────────────────
//
// The vision-language variant of the dense Qwen 3.5 release. Shares its
// architecture string (`Qwen3_5ForConditionalGeneration`) with the pure-
// text Qwen 3.5 — the dispatcher disambiguates by probing for an actual
// vision tower in the safetensors (some text-only Qwen 3.5 checkpoints
// ship a vestigial `vision_config`).
//
// Composition mirrors `Qwen3VLMoe` (defined in `Qwen3Vision.swift`):
//   • vision tower — `Qwen3VLVisionModel`. Tensor layout matches
//     Qwen 3-VL exactly (the per-block sizing comes from `vision_config`).
//   • text backbone — `Qwen35Hybrid` (stack-interleaved GDN ↔ attention).
//     The dense 0.8B checkpoint uses GDN linear-attention layers
//     alternating with full-attention every 4 layers — `text_config`
//     drives that pattern, so the whole root config is forwarded.
//
// Weight prefix layouts in the wild:
//   * raw HF Qwen 3.5-VL release  — `model.language_model.*` + `model.visual.*`
//   * mlx-community conversions   — `language_model.model.*` + `vision_tower.*`
// Qwen35Hybrid is wired for the mlx-community shape (`language_model.model.X`).
// For the raw release the bundle is rewritten via
// `weights.prefixed("model.language_model.").withAddedPrefix("language_model.model.")`
// so the existing text loader runs unchanged — same trick as MiniCPM-V's
// strip+re-prepend recipe, just targeting a different namespace.
//
// Coherence caveats inherit from `Qwen3VL` (CPU vision attention,
// scalar text positions, deepstack omitted). `mtp.*` weights stay
// unloaded — the multi-token-prediction head is a speculative-decoding
// accelerator, not required for greedy text decode.

public enum Qwen35VL {
    /// `image_token_id` default for Qwen 3.5-VL checkpoints. The 0.8B
    /// release uses 248056 — published checkpoints carry the value
    /// explicitly under `config.image_token_id`, so this default only
    /// applies when the field is missing.
    public static let defaultImageTokenId = 248_056

    /// `video_token_id` default for Qwen 3.5-VL checkpoints — 248057
    /// on the 0.8B release. Same fallback semantics as the image id.
    public static let defaultVideoTokenId = 248_057

    /// Capabilities a Qwen 3.5-VL checkpoint exposes. Identical to the
    /// dense Qwen 3-VL — vision tower shares the same Conv3d patch
    /// embed + temporal-patch unfold that drives single-image and
    /// multi-frame video paths.
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.imageIn, .videoIn])

    /// Build a `VisionModel` from a `Qwen3_5ForConditionalGeneration`
    /// checkpoint that carries a `vision_config` sub-tree AND actual
    /// vision-tower tensors (the VLM variant): the Qwen 3-VL vision
    /// tower + the Qwen 3.5 hybrid text backbone, joined by the
    /// cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Qwen35Error.missingConfig("vision_config")
        }

        // ── Text backbone — Qwen 3.5 hybrid engine ──
        // Qwen35Hybrid expects the mlx-community layout
        // (`language_model.model.X`). The raw HF release ships
        // `model.language_model.X` — rewrite the bundle view so the
        // existing loader runs unchanged. mlx-community conversions
        // already use the expected layout, so the rewrite is a no-op
        // for them (no keys carry the `model.language_model.` prefix).
        let textWeights: SafeTensorsBundle
        if weights.has("model.language_model.embed_tokens.weight") {
            textWeights =
                weights
                .prefixed("model.language_model.")
                .withAddedPrefix("language_model.model.")
        } else {
            textWeights = weights
        }
        let textEngine = try Qwen35Hybrid.loadModel(
            config: config, weights: textWeights,
            options: options, device: device)

        // ── Vision tower — Qwen 3-VL tower ──
        // The raw release puts the tower under `model.visual.*`; older
        // mlx-community conversions use `vision_tower.*`. Pick whichever
        // is present so both layouts load.
        let visionWeights: SafeTensorsBundle = {
            if weights.has("model.visual.patch_embed.proj.weight") {
                return weights.prefixed("model.visual.")
            }
            return weights.prefixed("vision_tower.")
        }()
        let vision = try Qwen3VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: visionWeights, dtype: textEngine.dtype, device: device)

        let imageTokenId =
            config.int("image_token_id")
            ?? config.int("image_token_index") ?? defaultImageTokenId
        let videoTokenId =
            config.int("video_token_id")
            ?? config.int("video_token_index") ?? defaultVideoTokenId
        return try VisionModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .clip,
            imageTokenCount: vision.mergedTokenCount)
    }
}
