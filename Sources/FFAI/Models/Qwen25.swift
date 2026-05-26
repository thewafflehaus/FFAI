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
// Qwen 2.5-VL — Alibaba's Qwen2.5-VL vision-language model (the
// `Qwen2_5_VLForConditionalGeneration` checkpoints).
//
// Composition:
//   • Qwen 2.5-VL vision tower — a dynamic-resolution ViT that differs
//     substantially from the shared SigLIP `VisionEncoder`:
//       – patch-embed via a flattened conv (each input patch is a
//         `in_ch · temporalPatch · patch · patch` row, projected by a
//         single GEMM),
//       – RMSNorm (not LayerNorm) pre-norms,
//       – 2D rotary position embedding (M-RoPE) over the patch grid,
//       – windowed attention on most blocks, full attention on a few
//         (`fullatt_block_indexes`),
//       – a patch-merger that pools each `mergeSize × mergeSize`
//         neighbourhood and projects into the text hidden dim.
//     The tower itself lives in `Models/Vision/Qwen25VL.swift` — this
//     file is the family orchestrator (load entrypoint + the
//     `<|image_pad|>` / `<|video_pad|>` token ids the splice needs).
//   • Qwen 2 / 2.5 text backbone — the existing Llama dense engine
//     (Qwen 2.x routes through `LlamaDense`), loaded from the
//     checkpoint's top-level (Qwen2.5-VL stores text weights under
//     `model.*` / `lm_head.*`, the standard text layout).
//
// The two are joined by `VisionModel`'s cross-modal token splice: each
// `<|image_pad|>` placeholder (`image_token_id`) in the prompt takes one
// of the merged vision tokens; for video, each `<|video_pad|>`
// placeholder (`video_token_id`) takes one of the per-temporal-patch
// merged tokens emitted by `encode(frames:)`.

import Foundation
import Metal

public enum Qwen25VLError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "Qwen25VL: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "Qwen25VL: checkpoint is missing tensor '\(name)'"
        }
    }
}

public enum Qwen25VL {
    /// `image_token_id` default for Qwen 2.5-VL checkpoints
    /// (`<|image_pad|>`).
    public static let defaultImageTokenId = 151_655
    /// `video_token_id` default for Qwen 2.5-VL checkpoints
    /// (`<|video_pad|>`).
    public static let defaultVideoTokenId = 151_656

    /// Capabilities a Qwen 2.5-VL checkpoint declares to the loader.
    /// Text + image + video — every published Qwen 2.5-VL conversion
    /// ships the same vision tower that handles both image (one frame
    /// repeated `temporal_patch_size` times) and video (N consecutive
    /// frames per temporal patch). The actual video-token splice is
    /// gated on the caller passing `videoFrames` to
    /// `VisionModel.generate(...)`.
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.visionIn, .videoIn])

    /// Build a `VisionModel` from a `Qwen2_5_VLForConditionalGeneration`
    /// checkpoint: the dynamic-resolution vision tower + the Qwen 2.x
    /// text backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Qwen25VLError.missingConfig
        }

        // ── Text backbone ──
        // The mlx-community Qwen 2.5-VL conversion namespaces its text
        // weights under `language_model.model.*` / `language_model.norm`;
        // the text hyper-parameters live at the config top level. Route
        // the `language_model.`-prefixed sub-tree through the Llama dense
        // engine (Qwen 2.x == LlamaDense).
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try LlamaDense.loadModel(
            config: config, weights: textWeights, options: options, device: device)

        // ── Vision tower ──
        // Vision weights are under `vision_tower.*` and (per the
        // checkpoint's `skip_vision` quantization flag) are not
        // quantized — plain f16 / bf16 tensors.
        let visionWeights = weights.prefixed("vision_tower.")
        let vision = try Qwen25VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: visionWeights, dtype: textEngine.dtype, device: device)

        let imageTokenId = config.int("image_token_id") ?? defaultImageTokenId
        let videoTokenId = config.int("video_token_id") ?? defaultVideoTokenId
        // The vision tower decides its own merged-token count from the
        // dynamic image geometry; the encoder facade reports it.
        return try VisionModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine,
            imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .clip,
            imageTokenCount: vision.mergedTokenCount)
    }
}
