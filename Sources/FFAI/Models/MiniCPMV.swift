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
// MiniCPM-V 4.6 — OpenBMB's `MiniCPMV4_6ForConditionalGeneration`
// checkpoint (`model_type: minicpmv4_6`). An image/video VLM composed
// of a **SigLIP2-400M** vision encoder + the **Qwen3.5** text backbone
// (`qwen3_5_text`, which routes to the in-tree `Qwen35` family),
// joined by a two-stage projector:
//
//   * `vit_merger` — a window cross-attention merger injected MID-stack
//     after encoder layer `insert_layer_id` (default 6). Reduces a
//     `(grid_h × grid_w)` patch grid by a `(2, 2)` factor and projects
//     back to the encoder hidden dim.
//   * `merger`     — the final 2×2 reduction + projection into the text
//     hidden dim (`MergerBlock`: LayerNorm → Linear → GELU → Linear).
//
// The vision tokens are spliced into the text embedding stream at every
// `image_token_id` (248056) position by the shared `VisionModel`.
//
// ─── v1 scope ────────────────────────────────────────────────────────
//
// One 448×448 tile per image (single image, no LLaVA-UHD slicing). The
// SigLIP patch grid is 32×32 = 1024 patches; `vit_merger` (2,2) reduces
// it to 16×16 = 256; the final `merger` (2,2) reduces to 8×8 = 64
// tokens — matching the checkpoint's `query_num: 64`. The shipped
// `position_embedding` is `[4900, 1152]` (70×70 grid at the checkpoint
// `image_size: 980`); it is bilinearly interpolated to the runtime
// 32×32 grid once at load.
//
// The 4×-detail OCR mode (which skips `vit_merger`) and the multi-tile
// LLaVA-UHD path are deferred — they layer on top of the same building
// blocks.
//
// Reference: `mlx-vlm/mlx_vlm/models/minicpmv4_6/`
// (`minicpmv4_6.py`, `vision.py`, `processing_minicpmv4_6.py`).
//
// Vision tower internals (config structs, SigLIP encoder, vit_merger,
// merger, composed encoder, encode methods) live in
// `Models/Vision/MiniCPMVVision.swift`. This file is the family
// orchestrator (load entrypoint + public constants).

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum MiniCPMV4_6 {
    public static let modelTypes: Set<String> = ["minicpmv4_6"]
    public static let architectures: Set<String> =
        ["MiniCPMV4_6ForConditionalGeneration"]

    /// `image_token_id` default for MiniCPM-V-4.6 checkpoints — the
    /// placeholder the chat template emits for each image.
    /// Source: `openbmb/MiniCPM-V-4.6` config.json `image_token_id`.
    public static let defaultImageTokenId = 248056

    /// `video_token_id` default for MiniCPM-V-4.6 checkpoints — the
    /// placeholder the chat template emits for each video frame token.
    /// Source: `openbmb/MiniCPM-V-4.6` config.json `video_token_id`
    /// (value 248057; declared separately from image_token_id).
    public static let defaultVideoTokenId = 248057

    /// Capabilities a MiniCPM-V-4.6 checkpoint exposes. Text + image +
    /// video — MiniCPM-V-4.6 encodes each video frame as an independent
    /// image through the same SigLIP2 vision tower, producing
    /// `outputTokenCount` merged tokens per frame. The video splice
    /// concatenates the per-frame token runs and substitutes them at the
    /// `video_token_id` placeholder positions.
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.imageIn, .videoIn])

    /// Runtime tile resolution for the v1 path. 448 = 32×32 patches at
    /// `patch_size: 14`; `vit_merger` (2,2) → 16×16; `merger` (2,2) →
    /// 8×8 = 64 tokens — matches `query_num: 64`.
    public static let runtimeImageSize = 448

    /// Build a `VisionModel` from a `MiniCPMV4_6ForConditionalGeneration`
    /// checkpoint: SigLIP2-400M encoder + `vit_merger` + `merger` +
    /// Qwen3.5 text backbone, joined by `VisionModel`'s cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config"),
            config.nested("text_config") != nil
        else {
            throw MiniCPMVError.missingConfig(
                "vision_config / text_config")
        }

        // ── Text backbone: Qwen3.5 from `text_config` ─────────────────
        // mlx-community's MiniCPM-V 4.6-4bit conversion stores text
        // weights as `language_model.model.{embed_tokens, layers.*, norm}.weight`
        // (outer `language_model.` wrapping the inner `model.`). The
        // Qwen3.5 loader already probes the `language_model.model.`
        // prefix candidate, so we hand the full bundle through
        // unmodified — Qwen3.5 picks the right prefix from its own
        // `prefixCandidates` automatically. The earlier
        // `prefixed("model.language_model.").withAddedPrefix("model.")`
        // chain assumed an internal layout no mlx-community release
        // actually ships; on the real bundle no key carries the
        // `model.language_model.` prefix, so the strip returned an
        // empty view and the Qwen3.5 loader crashed at `embed_tokens.weight`.
        let textEngine = try Qwen35Hybrid.loadModel(
            config: config, weights: weights,
            options: options, device: device)

        // ── Vision tower: SigLIP2 encoder + vit_merger + merger ───────
        // All vision weights live at the top level — `vision_tower.*`
        // (SigLIP encoder), `vit_merger.*` (window-attention merger),
        // and `merger.*` (down-projection to text hidden) — with no
        // `model.` wrapper in the mlx-community release.
        let visionWeights = weights.prefixed("vision_tower.")
        let mergerWeights = weights.prefixed("merger.")
        let vitMergerWeights = weights.prefixed("vit_merger.")
        let composed = try MiniCPMVComposedEncoder.load(
            visionConfig: visionConfig,
            insertLayerId: config.int("insert_layer_id") ?? 6,
            mergerTimes: config.int("merger_times") ?? 1,
            runtimeImageSize: runtimeImageSize,
            textHidden: textEngine.hidden,
            visionWeights: visionWeights,
            mergerWeights: mergerWeights,
            vitMergerOverride: vitMergerWeights,
            device: device)

        let imageTokenId = config.int("image_token_id") ?? defaultImageTokenId
        let videoTokenId = config.int("video_token_id") ?? defaultVideoTokenId
        return try VisionModel(
            visionEncoder: composed, engine: textEngine,
            imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .siglip,
            imageTokenCount: composed.outputTokenCount)
    }
}

public enum MiniCPMVError: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "MiniCPM-V: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "MiniCPM-V: unsupported config: \(m)"
        }
    }
}
