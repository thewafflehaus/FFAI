// Copyright 2026 Tom Turney (@TheTom)
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
// Step 3 vision tower — Perception-Encoder ViT + 2× strided
// downsamplers + projector, plus the Step-3 vision-language
// orchestrator (`enum Step3VL`).
//
// **Status:** WIP scaffold. This file declares the static shape
// (`Step3VLVisionConfig`) and the orchestrator entry point so a
// `Step3ForConditionalGeneration` checkpoint can be type-identified
// end-to-end. The tower forward path lands in follow-ups.
//
// ─── Architecture summary (from the upstream vision config) ──────────
//
// • **Perception-Encoder** — 47-layer ViT, hidden=1536, 16 heads,
//   head_dim=96. patch_size=14, image_size=728 → 52×52 = 2704 patches.
// • **Activation**: quick-GELU (`x * sigmoid(1.702 * x)`).
// • **Position**: 2D RoPE (`use_rope2d`) **plus** an additive
//   learned 2D absolute position embedding (`use_abs_posemb`).
// • **LayerScale (`ls_init_value=0.1`)** — a per-block scalar `γ` that
//   multiplies the residual branch output. Folded into `o_proj` /
//   `down_proj` at load time keeps the kernel surface flat.
// • **Pre-norm ViT** — `use_ln_pre: true`, `use_ln_post: false`. No
//   CLS token.
// • **Fused QKV** — `in_proj` projects hidden → 3×hidden in one matmul
//   (LXMERT-style ViT). Maps to `Ops.batchedQkvQmm` (or the dense
//   equivalent for fp16 weights).
// • **Post-encoder reduction**: two stride-2 Conv2d layers
//   (`vit_downsampler1` / `vit_downsampler2`) collapse the 52×52 grid
//   4× to 13×13 = 169 patch tokens, followed by a Linear projector
//   (`vit_large_projector`: width*4 → text.hidden = 4096) that maps
//   each surviving patch into the text-decoder hidden space.
// • **Splice**: each `image_token_id=128001` placeholder in the prompt
//   receives one of the 169 projected vision tokens. The standard
//   `VisionModel.cross-modal token splice` shape handles this directly.

import Foundation

// ─── Vision config ───────────────────────────────────────────────────

struct Step3VLVisionConfig {
    // ── ViT body ──
    let depth: Int           // 47
    let hidden: Int          // 1536
    let numHeads: Int        // 16
    let headDim: Int         // 96 (1536 / 16)
    let mlpRatio: Float      // 8960 / 1536 ≈ 5.833

    let patchSize: Int       // 14
    let imageSize: Int       // 728 → 52×52 patches

    let useAbsPosEmb: Bool   // true
    let useRope2d: Bool      // true
    let rope2dTheta: Float   // 10_000

    let useLnPre: Bool       // true
    let useLnPost: Bool      // false
    let layerScaleInit: Float?  // 0.1; nil disables

    // ── Downsamplers + projector ──
    /// Number of stride-2 Conv2d downsample stages after the encoder.
    /// Each one quarters the patch count.
    let downsampleStages: Int  // 2
    /// Final patch-token count after downsampling (169 = 13×13 for the
    /// default 728-image config).
    let outputTokenCount: Int  // 169
    /// Projector input multiplier: with `downsampleStages=2` the
    /// projector takes `hidden * 4` channels.
    let projectorInputMultiplier: Int  // 4

    static func decode(_ vc: ModelConfig) throws -> Step3VLVisionConfig {
        guard
            let depth = vc.int("layers"),
            let hidden = vc.int("width"),
            let numHeads = vc.int("heads"),
            let patchSize = vc.int("patch_size"),
            let imageSize = vc.int("image_size")
        else {
            throw Step3Error.missingConfig("vision_config")
        }
        let headDim = hidden / numHeads
        let mlpRatio = Float(vc.float("mlp_ratio") ?? 5.833)
        let useAbs = vc.bool("use_abs_posemb") ?? true
        let useRope = vc.bool("use_rope2d") ?? true
        let rope2dTheta = Float(vc.float("rope2d_theta") ?? 10_000)
        let useLnPre = vc.bool("use_ln_pre") ?? true
        let useLnPost = vc.bool("use_ln_post") ?? false
        let lsInit: Float? = (vc.float("ls_init_value")).map(Float.init)

        // The patch grid count is image_size / patch_size; both stride-2
        // downsamplers shrink it by 4×, so default output = (side/4)^2.
        let perSide = imageSize / patchSize
        let outputTokens = vc.int("image_token_len") ?? (perSide / 4) * (perSide / 4)

        return Step3VLVisionConfig(
            depth: depth, hidden: hidden, numHeads: numHeads, headDim: headDim,
            mlpRatio: mlpRatio,
            patchSize: patchSize, imageSize: imageSize,
            useAbsPosEmb: useAbs,
            useRope2d: useRope, rope2dTheta: rope2dTheta,
            useLnPre: useLnPre, useLnPost: useLnPost, layerScaleInit: lsInit,
            downsampleStages: 2, outputTokenCount: outputTokens,
            projectorInputMultiplier: 4)
    }
}

// ─── Step3VL — vision-language orchestrator ──────────────────────────

public enum Step3VL {
    /// `image_token_id` default for Step-3 vision-language checkpoints.
    public static let defaultImageTokenId = 128_001

    /// Architecture strings handled by this orchestrator — re-exported
    /// from the family entry point for direct loader use.
    public static let architectures: Set<String> = Step3.vlArchitectures

    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.imageIn])

    /// Build a `VisionModel` from a Step-3 vision-language checkpoint:
    /// the Perception-Encoder vision tower + 2× downsamplers +
    /// projector + Step-3 text backbone, joined by the cross-modal
    /// splice. **WIP** — both the vision tower forward and the text
    /// backbone load throw `Step3Error.notYetImplemented` today; the
    /// scaffold is in place so the loader dispatch type-checks.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Step3Error.missingConfig("vision_config")
        }
        _ = try Step3VLVisionConfig.decode(visionConfig)

        // Text backbone — Step-3 hybrid engine.
        let textVariant = try Step3.variant(for: config)
        _ = try textVariant.loadModel(
            config: config, weights: weights, options: options, device: device)

        throw Step3Error.notYetImplemented("Step3VL.load full wiring")
    }
}
