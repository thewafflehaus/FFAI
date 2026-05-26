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
// FastVLM — Apple's FastVLM vision-language model
// (`llava_qwen2` model_type / `LlavaQwen2ForCausalLM` architecture).
//
// Composition:
//   • FastViTHD vision tower — Apple's fast hybrid ViT with structural
//     reparameterization (inference-time fused convolutions). Architecture:
//       – ConvolutionalStem (3 MobileOne blocks, total 4× downsampling),
//       – Stage alternation: RepMixerBlocks (depthwise conv token-mixing)
//         and AttentionBlocks (MHSA), interleaved with PatchEmbed
//         (LargeKernelBlock + pointwise, stride-2 downsampling) and
//         RepCPE (conditional positional encoding depthwise conv),
//       – conv_exp (depthwise expand + Squeeze-Excite) applied to final
//         stage feature map,
//       – output: the conv_exp feature map [B, H, W, mm_hidden_size].
//   • mlp2x_gelu projector — two linear layers with GELU activation,
//     projecting from mm_hidden_size (3072) into the text hidden dim.
//   • Qwen2 text backbone — the existing `LlamaDense` engine (Qwen2 is
//     architecturally identical to Llama; QKV biases are auto-detected
//     by `loadLinear`). Text weights live under `language_model.*`.
//
// The three are joined by `VisionModel`'s cross-modal token splice: each
// image-placeholder token (id -200) in the prompt is replaced by one
// of the projected vision tokens.
//
// Weight layout notes:
//   All convolutions in the checkpoint are inference-mode reparameterized:
//   the multi-branch training-time conv is fused into a single conv at
//   save time (`reparam_conv` keys). BatchNorm parameters are also stored
//   (for the depthwise ConvFFN DW conv) and are folded into the conv
//   weight at load time.
//
// Spatial resolution path (1024px input):
//   image [1, 3, 1024, 1024]
//   → ConvStem (4× stride) → [B, H=256, W=256, C=96]  NHWC
//   → stage 0 (2 RepMixer) → [B, 256, 256, 96]
//   → PatchEmbed (stride 2) → [B, 128, 128, 192]
//   → stage 1 (12 RepMixer) → [B, 128, 128, 192]
//   → PatchEmbed             → [B, 64, 64, 384]
//   → stage 2 (24 RepMixer) → [B, 64, 64, 384]
//   → PatchEmbed             → [B, 32, 32, 768]
//   → CPE + stage 3 (4 Attn) → [B, 32, 32, 768]
//   → PatchEmbed             → [B, 16, 16, 1536]
//   → CPE + stage 4 (2 Attn) → [B, 16, 16, 1536]
//   → conv_exp (stride 1)    → [B, 16, 16, 3072]
//   → projector: reshape to [256, 3072] then 3072→hidden→hidden
//   → [256, text_hidden] vision tokens
//
// Conv convention:
//   All ops run in NHWC throughout the vision tower (natural for CPU
//   depthwise convolutions). Pointwise 1×1 convolutions use Ops.gemm on
//   the flattened [B*H*W, inC] matrix. Full-channel FFAI depthwise GPU
//   kernel is not available, so depthwise convolutions use a parallelized
//   CPU path with DispatchQueue.concurrentPerform over channels.
//
// Normalization: FastVLM uses mean=[0,0,0], std=[1,1,1] (pixels ÷ 255;
// no centering). FFAI's ImageNormalization supports custom mean/std.
//
// The vision tower internals (config structs, blocks, FastVLMVisionTower,
// FastVLMProjector, FastVLMComposedTower) live in
// `Models/Vision/FastVLMVision.swift`. This file is the family orchestrator
// (load entrypoint + the model-type / architecture identifiers).

import Foundation
import Metal

// ─── Errors ──────────────────────────────────────────────────────────

public enum FastVLMError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "FastVLM: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "FastVLM: checkpoint is missing tensor '\(name)'"
        }
    }
}

// ─── Family registry ─────────────────────────────────────────────────

public enum FastVLM {
    /// `model_type` values that identify a FastVLM checkpoint.
    public static let modelTypes: Set<String> = ["llava_qwen2"]

    /// Architecture strings used by FastVLM-family HF conversions.
    public static let architectures: Set<String> = ["LlavaQwen2ForCausalLM"]

    /// Default image placeholder token id. The tokenizer injects the
    /// `<image>` special token; preprocessing replaces it with id -200.
    /// Not stored in config.json — hardcoded in the reference processor.
    public static let defaultImageTokenId = -200

    /// Build a `VisionModel` from a FastVLM checkpoint: the FastViTHD tower
    /// + the mlp2x_gelu projector + the Qwen2 text backbone, joined by
    /// the cross-modal token splice.
    public static func load(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config")
        else { throw FastVLMError.missingConfig }

        // ── Text backbone ──
        // Qwen2-0.5B weights live under `language_model.*`. Merge
        // Qwen2/Llama defaults so the LlamaDense loader gets a complete
        // config (the VLM top-level config is sparse for text fields).
        let mergedTextConfig = ModelConfig(
            architecture: "Qwen2ForCausalLM",
            modelType: "qwen2",
            raw: fastVLMTextConfigWithDefaults(config.raw,
                                               vocabFallback: config.int("vocab_size")))
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try LlamaDense.loadModel(
            config: mergedTextConfig, weights: textWeights,
            options: options, device: device)

        // ── Vision tower ──
        // Vision weights live under `vision_tower.vision_model.*`.
        let visionWeights = weights.prefixed("vision_tower.vision_model.")
        let visionCfg = try FastVLMVisionConfig.decode(visionConfig)
        let tower = try FastVLMVisionTower.load(
            cfg: visionCfg, weights: visionWeights,
            dtype: textEngine.dtype, device: device)

        // ── Multi-modal projector ──
        // `mm_projector.{0,2}.*` — mlp2x_gelu is two linears with GELU.
        let projector = try FastVLMProjector.load(
            mmHidden: visionCfg.mmHiddenSize,
            textHidden: textEngine.hidden,
            weights: weights, device: device)

        // Image token count = spatial patches after all downsampling.
        // For 1024px with 4x stem + 4 PatchEmbeds (each 2x): 16x16 = 256.
        let imageTokenCount = tower.patchH * tower.patchW

        // Compose tower + projector behind a single VisionEncoder facade
        // so VisionModel's splice sees `[imageTokenCount, textHidden]` tokens.
        let composedTower = FastVLMComposedTower(
            tower: tower, projector: projector,
            imageTokenCount: imageTokenCount,
            textHidden: textEngine.hidden,
            dtype: textEngine.dtype)

        let imageTokenId = config.int("image_token_id")
            ?? config.int("image_token_index")
            ?? defaultImageTokenId

        // FastVLM uses mean=0, std=1 (simple ÷ 255 rescale). The CLIP
        // mean/std would over-normalize — use identity normalization.
        let normalization = ImageNormalization(
            mean: (0.0, 0.0, 0.0), std: (1.0, 1.0, 1.0))

        return try VisionModel(
            visionEncoder: composedTower.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: normalization,
            imageTokenCount: imageTokenCount)
    }
}

/// Merge FastVLM text-model defaults into the VLM's top-level config.
/// The `LlamaDense` loader needs a complete text config; the FastVLM
/// top-level config stores the Qwen2 text fields directly (no nested
/// `text_config`).
func fastVLMTextConfigWithDefaults(
    _ raw: [String: Any], vocabFallback: Int?
) -> [String: Any] {
    // Qwen2-0.5B text model defaults.
    var merged: [String: Any] = [
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocabFallback ?? 151936,
        "rope_theta": 1_000_000.0,
        "rope_traditional": false,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": true,
    ]
    // Checkpoint-declared fields override defaults. Exclude nested VLM
    // keys that LlamaDense doesn't understand (vision_config, mm_*).
    let excludeKeys: Set<String> = ["vision_config", "architectures",
                                    "model_type", "image_token_id",
                                    "image_token_index", "mm_projector_type",
                                    "mm_hidden_size"]
    for (k, v) in raw where !excludeKeys.contains(k) { merged[k] = v }
    return merged
}
