// Nemotron-VLM — vision tower internals.
//
// The multi-modal projector, composed encoder, and helper types for
// Nemotron Nano VL live here. The family orchestrator (load entrypoint,
// `NemotronVLError`, `NemotronVL` with constants and `load()`) lives in
// `Models/NemotronVL.swift`.
//
// Coherence-first port: the vision tower's bidirectional attention runs
// on the CPU (the shared `VisionEncoder` already does this — vision
// token counts are small, an O(n²·d) attention is cheap next to the GPU
// projection GEMMs and is unambiguously correct). The projector is CPU
// (it runs once per image). A head-dim-agnostic GPU vision SDPA and
// pixel-shuffle token reduction are later performance / fidelity passes.

import Foundation
import Metal

/// Load the ViT vision encoder into the shared `VisionEncoder`. The
/// checkpoint's `vision_model.*` ViT keys map onto
/// `VisionEncoder.parameters()`; the projection into the text hidden
/// dim is the separate `NemotronVLProjector`, so the encoder itself
/// carries no projection.
func nemotronVLLoadVisionEncoder(
    config: ModelConfig, weights: SafeTensorsBundle, device: Device
) throws -> VisionEncoder {
    guard let hidden = config.int("hidden_size"),
          let imageSize = config.int("image_size"),
          let patchSize = config.int("patch_size"),
          let intermediate = config.int("intermediate_size"),
          let nLayers = config.int("num_hidden_layers"),
          let nHeads = config.int("num_attention_heads")
    else {
        throw NemotronVLError.missingConfig
    }
    let eps = Float(config.float("layer_norm_eps") ?? 1e-6)
    let encConfig = VisionEncoderConfig(
        imageSize: imageSize, patchSize: patchSize, hidden: hidden,
        intermediate: intermediate, nLayers: nLayers, nHeads: nHeads,
        layerNormEps: eps, textHidden: hidden)

    // Probe the vision-tower weight prefix.
    let vm = weights.has("vision_model.embeddings.patch_embedding.weight")
        ? weights.prefixed("vision_model.")
        : weights.prefixed("model.vision_model.")

    // The mlx-converted checkpoint may store the patch-embed conv
    // weight in MLX's OHWI layout `[out_ch, kH, kW, in_ch]`;
    // `Ops.conv2d` expects PyTorch OIHW. Transpose when the trailing
    // dim is the channel count.
    let patchWRaw = try vm.tensor(named: "embeddings.patch_embedding.weight")
    let patchW = patchWRaw.shape.count == 4 && patchWRaw.shape[3] == 3
        ? transposeOHWItoOIHW(patchWRaw)
        : patchWRaw
    let patchB = try vm.tensor(named: "embeddings.patch_embedding.bias")
    let posEmb = try vm.tensor(named: "embeddings.position_embedding.weight")

    var layers: [VisionEncoderLayer] = []
    layers.reserveCapacity(nLayers)
    for i in 0..<nLayers {
        let p = "encoder.layers.\(i)"
        let ln1 = LayerNorm(
            weight: try vm.tensor(named: "\(p).layer_norm1.weight"),
            bias: try vm.tensor(named: "\(p).layer_norm1.bias"), eps: eps)
        let ln2 = LayerNorm(
            weight: try vm.tensor(named: "\(p).layer_norm2.weight"),
            bias: try vm.tensor(named: "\(p).layer_norm2.bias"), eps: eps)
        func lin(_ name: String) throws -> Linear {
            Linear(weight: try vm.tensor(named: "\(p).\(name).weight"),
                   bias: try? vm.tensor(named: "\(p).\(name).bias"))
        }
        layers.append(VisionEncoderLayer(
            layerNorm1: ln1,
            qProj: try lin("self_attn.q_proj"),
            kProj: try lin("self_attn.k_proj"),
            vProj: try lin("self_attn.v_proj"),
            oProj: try lin("self_attn.out_proj"),
            layerNorm2: ln2,
            fc1: try lin("mlp.fc1"), fc2: try lin("mlp.fc2"),
            hidden: hidden, nHeads: nHeads, intermediate: intermediate))
    }
    let postLN = LayerNorm(
        weight: try vm.tensor(named: "post_layernorm.weight"),
        bias: try vm.tensor(named: "post_layernorm.bias"), eps: eps)

    return VisionEncoder(
        config: encConfig, patchEmbedWeight: patchW, patchEmbedBias: patchB,
        positionEmbedding: posEmb, layers: layers,
        postLayerNorm: postLN, projection: nil, dtype: patchW.dtype)
}

// ─── Multi-modal projector ───────────────────────────────────────────

/// Nemotron-VLM's vision→text projector: a two-layer GELU MLP mapping
/// the encoder hidden into the text hidden dim. All CPU — it runs once
/// per image.
public final class NemotronVLProjector: @unchecked Sendable {
    /// First projection `[visionHidden] → [textHidden]`.
    let linear1: Linear
    /// Second projection `[textHidden] → [textHidden]`.
    let linear2: Linear
    let visionHidden: Int
    let textHidden: Int

    init(linear1: Linear, linear2: Linear, visionHidden: Int, textHidden: Int) {
        self.linear1 = linear1
        self.linear2 = linear2
        self.visionHidden = visionHidden
        self.textHidden = textHidden
    }

    static func load(
        visionConfig: ModelConfig, textHidden: Int,
        weights: SafeTensorsBundle, device: Device
    ) throws -> NemotronVLProjector {
        guard let visionHidden = visionConfig.int("hidden_size") else {
            throw NemotronVLError.missingConfig
        }
        // The projector is namespaced under `multi_modal_projector.`
        // (the HF convention) — probe both possible prefixes.
        let mp = weights.has("multi_modal_projector.linear_1.weight")
            ? weights.prefixed("multi_modal_projector.")
            : weights.prefixed("model.multi_modal_projector.")
        func lin(_ name: String) throws -> Linear {
            Linear(weight: try mp.tensor(named: "\(name).weight"),
                   bias: try? mp.tensor(named: "\(name).bias"))
        }
        return NemotronVLProjector(
            linear1: try lin("linear_1"), linear2: try lin("linear_2"),
            visionHidden: visionHidden, textHidden: textHidden)
    }

    /// Project `[numPatches, visionHidden]` raw encoder tokens into
    /// `[numPatches, textHidden]` — `linear_1` → GELU → `linear_2`.
    func project(encoderTokens: Tensor, device: Device) -> Tensor {
        let numTokens = encoderTokens.shape[0]
        let cmd = device.makeCommandBuffer()
        var x = Ops.gemm(weight: linear1.weight, input: encoderTokens,
                         nRows: numTokens, on: cmd)
        if let b = linear1.bias {
            x = Qwen25VLVisionModel.addRowBias(
                x, bias: b, nRows: numTokens,
                rowSize: linear1.weight.shape[0], on: cmd)
        }
        x = Ops.gelu(x, on: cmd)
        var y = Ops.gemm(weight: linear2.weight, input: x,
                         nRows: numTokens, on: cmd)
        if let b = linear2.bias {
            y = Qwen25VLVisionModel.addRowBias(
                y, bias: b, nRows: numTokens,
                rowSize: linear2.weight.shape[0], on: cmd)
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        return y
    }
}

// ─── Composed vision tower (encoder + projector) ─────────────────────

/// Couples the ViT `VisionEncoder` with the Nemotron-VLM projector so
/// the pair presents a single `VisionEncoder`-shaped surface to
/// `VLModel`. The composed tower's `encode` produces `[numPatches,
/// textHidden]` — the projected vision tokens.
final class NemotronVLVisionTower {
    let encoder: VisionEncoder
    let projector: NemotronVLProjector
    let textHidden: Int
    let dtype: DType

    init(encoder: VisionEncoder, projector: NemotronVLProjector,
         textHidden: Int, dtype: DType) {
        self.encoder = encoder
        self.projector = projector
        self.textHidden = textHidden
        self.dtype = dtype
    }

    /// Present the composed encoder+projector as a `VisionEncoder` whose
    /// `encode` runs the ViT forward + the projector.
    func asVisionEncoder() -> VisionEncoder {
        NemotronVLComposedEncoder(tower: self)
    }
}

/// A `VisionEncoder` subclass whose `encode` runs the ViT encoder then
/// the Nemotron-VLM projector — so `VLModel` sees one tower producing
/// `[numPatches, textHidden]` tokens.
final class NemotronVLComposedEncoder: VisionEncoder {
    let tower: NemotronVLVisionTower

    init(tower: NemotronVLVisionTower) {
        self.tower = tower
        let e = tower.encoder
        let projectedConfig = VisionEncoderConfig(
            imageSize: e.config.imageSize, patchSize: e.config.patchSize,
            hidden: e.config.hidden, intermediate: e.config.intermediate,
            nLayers: e.config.nLayers, nHeads: e.config.nHeads,
            layerNormEps: e.config.layerNormEps, textHidden: tower.textHidden)
        super.init(
            config: projectedConfig,
            patchEmbedWeight: e.patchEmbedWeight, patchEmbedBias: e.patchEmbedBias,
            positionEmbedding: e.positionEmbedding, layers: e.layers,
            postLayerNorm: e.postLayerNorm, projection: nil, dtype: tower.dtype)
    }

    /// Run the ViT encoder, then the projector. Returns
    /// `[numPatches, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        let raw = tower.encoder.encode(image: image, device: device)
        return tower.projector.project(encoderTokens: raw, device: device)
    }
}
