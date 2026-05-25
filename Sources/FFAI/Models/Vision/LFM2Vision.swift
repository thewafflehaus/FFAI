// LFM2 VL — vision tower internals.
//
// This file contains `LFM2VLProjector`, `LFM2VLComposedTower`, and all
// supporting CPU / GPU helpers for the SigLIP2 ViT + pixel-unshuffle
// projector used by LiquidAI's LFM2-VL checkpoints.
//
// Weight key layout:
//   vision_tower.embeddings.patch_embedding.{weight,bias}
//   vision_tower.embeddings.position_embedding.weight
//   vision_tower.encoder.layers.<i>.{layer_norm1,layer_norm2,self_attn,mlp}.*
//   vision_tower.post_layernorm.{weight,bias}
//   multi_modal_projector.layer_norm.{weight,bias}
//   multi_modal_projector.linear_1.{weight,bias}   (quantized in 4bit)
//   multi_modal_projector.linear_2.{weight,bias}   (quantized in 4bit)
//
// The family orchestrator (load entry-point + `image_token_index` default)
// lives in `Models/LFM2.swift`.

import Foundation
import Metal

// ─── Vision encoder loader ───────────────────────────────────────────

/// Load the SigLIP2 ViT into a `VisionEncoder`.
///
/// The patch embed is stored as a Linear weight `[hidden, channels*patch*patch]`;
/// `VisionEncoder` expects OIHW conv weight `[hidden, channels, patch, patch]`.
/// We reshape (not transpose — element order is the same) the 2D weight
/// into 4D in-place via a CPU copy.
func lfm2vlLoadVisionEncoder(
    config: ModelConfig,
    activeLayers: Int,
    weights: SafeTensorsBundle,
    device: Device
) throws -> VisionEncoder {
    guard let hidden       = config.int("hidden_size"),
          let intermediate = config.int("intermediate_size"),
          let nHeads       = config.int("num_attention_heads"),
          let numPatches   = config.int("num_patches"),
          let patchSize    = config.int("patch_size")
    else {
        throw LFM2VLError.missingConfig
    }
    let inChannels  = config.int("num_channels") ?? 3
    let layerNormEps = Float(config.float("layer_norm_eps") ?? 1e-6)

    // Derive the natural imageSize from num_patches and patch_size.
    // num_patches = (imageSize / patchSize)² → imageSize = sqrt(numPatches)*patchSize
    let patchesPerSide = Int(Double(numPatches).squareRoot().rounded())
    let imageSize = patchesPerSide * patchSize

    let encConfig = VisionEncoderConfig(
        inChannels: inChannels,
        imageSize: imageSize,
        patchSize: patchSize,
        hidden: hidden,
        intermediate: intermediate,
        nLayers: activeLayers,
        nHeads: nHeads,
        layerNormEps: layerNormEps,
        textHidden: hidden)   // no projection — projector handles text-hidden mapping

    // Patch embedding: stored as Linear [hidden, channels*patch*patch].
    // Conv2d OIHW requires [hidden, channels, patchH, patchW] — same
    // element count and order, different shape. A CPU view is sufficient.
    let patchWRaw = try weights.tensor(named: "embeddings.patch_embedding.weight")
    let patchW = reshapeLinearPatchEmbed(patchWRaw,
                                         hidden: hidden,
                                         channels: inChannels,
                                         patchSize: patchSize)
    let patchB = try weights.tensor(named: "embeddings.patch_embedding.bias")
    let posEmb = try weights.tensor(named: "embeddings.position_embedding.weight")

    // Encoder layers.
    var layers: [VisionEncoderLayer] = []
    layers.reserveCapacity(activeLayers)
    for i in 0..<activeLayers {
        let p = "encoder.layers.\(i)"
        let ln1 = LayerNorm(
            weight: try weights.tensor(named: "\(p).layer_norm1.weight"),
            bias:   try weights.tensor(named: "\(p).layer_norm1.bias"),
            eps: layerNormEps)
        let ln2 = LayerNorm(
            weight: try weights.tensor(named: "\(p).layer_norm2.weight"),
            bias:   try weights.tensor(named: "\(p).layer_norm2.bias"),
            eps: layerNormEps)
        func lin(_ name: String) throws -> Linear {
            Linear(weight: try weights.tensor(named: "\(p).\(name).weight"),
                   bias:   try weights.tensor(named: "\(p).\(name).bias"))
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
        weight: try weights.tensor(named: "post_layernorm.weight"),
        bias:   try weights.tensor(named: "post_layernorm.bias"),
        eps: layerNormEps)

    return VisionEncoder(
        config: encConfig,
        patchEmbedWeight: patchW,
        patchEmbedBias: patchB,
        positionEmbedding: posEmb,
        layers: layers,
        postLayerNorm: postLN,
        projection: nil,
        dtype: patchW.dtype)
}

/// Reshape a Linear patch-embed weight from `[hidden, channels*h*w]` to
/// `[hidden, channels, h, w]` (OIHW). The data order is unchanged — it
/// is a pure shape view, materialised into a fresh tensor so the result
/// is a standard `Tensor` backed by a Metal buffer.
private func reshapeLinearPatchEmbed(
    _ src: Tensor, hidden: Int, channels: Int, patchSize: Int
) -> Tensor {
    precondition(src.shape == [hidden, channels * patchSize * patchSize],
                 "reshapeLinearPatchEmbed: unexpected shape \(src.shape)")
    // Create a new tensor with the 4D OIHW shape; the underlying bytes are
    // identical, so we can copy the raw buffer content directly.
    let dst = Tensor.empty(
        shape: [hidden, channels, patchSize, patchSize],
        dtype: src.dtype)
    // Raw byte copy — element order is unchanged.
    let byteCount = src.byteCount
    memcpy(dst.buffer.contents().advanced(by: dst.offset),
           src.buffer.contents().advanced(by: src.offset),
           byteCount)
    return dst
}

// ─── Multi-modal projector ────────────────────────────────────────────

/// LFM2-VL multi-modal projector: pixel-unshuffle + LayerNorm + linear_1
/// (GELU) + linear_2.
///
/// The pixel-unshuffle collapses a `d × d` neighbourhood of adjacent ViT
/// output tokens (arranged in a 2D grid) into one super-patch, multiplying
/// the feature dim by `d²`. The LayerNorm + two-layer MLP then project the
/// super-patches into the text hidden dim.
///
/// All of this runs on the CPU (once per image) — the token count is at
/// most 256 (or 64 in our simplified pass), so the cost is negligible
/// next to the GPU projection GEMMs.
public final class LFM2VLProjector: @unchecked Sendable {
    let layerNorm: LayerNorm
    let linear1: AnyLinear
    let linear2: AnyLinear
    /// Explicit bias for linear_1 — loaded as a separate plain-float tensor.
    /// `loadLinear` embeds bias in `Linear` but not in `QuantizedLinear`, so
    /// quantized-checkpoint biases are stored and applied separately.
    let bias1: Tensor?
    /// Explicit bias for linear_2.
    let bias2: Tensor?
    let downsampleFactor: Int   // d — pixel-unshuffle factor
    let unshuffledDim: Int      // visionHidden * d * d
    let projHidden: Int
    let textHidden: Int

    init(layerNorm: LayerNorm, linear1: AnyLinear, linear2: AnyLinear,
         bias1: Tensor?, bias2: Tensor?,
         downsampleFactor: Int, unshuffledDim: Int, projHidden: Int, textHidden: Int) {
        self.layerNorm = layerNorm
        self.linear1 = linear1
        self.linear2 = linear2
        self.bias1 = bias1
        self.bias2 = bias2
        self.downsampleFactor = downsampleFactor
        self.unshuffledDim = unshuffledDim
        self.projHidden = projHidden
        self.textHidden = textHidden
    }

    static func load(
        unshuffledDim: Int, projHidden: Int, textHidden: Int,
        downsampleFactor: Int,
        weights: SafeTensorsBundle,
        quantization: ModelConfig.QuantizationConfig?,
        device: Device
    ) throws -> LFM2VLProjector {
        let ln = LayerNorm(
            weight: try weights.tensor(named: "layer_norm.weight"),
            bias:   try weights.tensor(named: "layer_norm.bias"),
            eps: 1e-5)
        let l1 = try loadLinear(base: "linear_1", in: weights, quantization: quantization)
        let l2 = try loadLinear(base: "linear_2", in: weights, quantization: quantization)
        // Load explicit biases: they are plain float tensors whether or not
        // the weight is quantized. `loadLinear` embeds the bias inside `Linear`
        // but NOT inside `QuantizedLinear`, so always load them separately here
        // and apply them externally to handle both checkpoint formats.
        let b1: Tensor? = weights.has("linear_1.bias")
            ? try weights.tensor(named: "linear_1.bias")
            : nil
        let b2: Tensor? = weights.has("linear_2.bias")
            ? try weights.tensor(named: "linear_2.bias")
            : nil
        return LFM2VLProjector(
            layerNorm: ln, linear1: l1, linear2: l2,
            bias1: b1, bias2: b2,
            downsampleFactor: downsampleFactor,
            unshuffledDim: unshuffledDim,
            projHidden: projHidden, textHidden: textHidden)
    }

    /// Project `[numPatches, visionHidden]` ViT tokens into
    /// `[numPatches / d², textHidden]` super-patch embeddings.
    ///
    /// Steps:
    ///   1. Reshape the 1D patch sequence into a 2D spatial grid.
    ///   2. Pixel-unshuffle: collapse each `d × d` neighbourhood into
    ///      one super-patch by concatenating the `d²` feature vectors.
    ///   3. LayerNorm each super-patch (dim = visionHidden * d²).
    ///   4. Linear_1 → GELU → Linear_2 → `[numSuperPatches, textHidden]`.
    ///
    /// All steps run on the CPU so the Metal command buffer stays pristine
    /// for the text backbone.
    func project(encoderTokens: Tensor, device: Device) -> Tensor {
        let numPatches = encoderTokens.shape[0]
        let visionHidden = encoderTokens.shape[1]
        let d = downsampleFactor

        // The patch sequence is in raster order: row-major over the 2D
        // grid of patches. patchesPerSide × patchesPerSide = numPatches.
        let pps = Int(Double(numPatches).squareRoot().rounded())
        precondition(pps * pps == numPatches,
                     "LFM2VLProjector: numPatches \(numPatches) must be a perfect square")
        precondition(pps % d == 0,
                     "LFM2VLProjector: patchesPerSide \(pps) must be divisible by downsampleFactor \(d)")

        let superPPS = pps / d          // super-patches per side
        let numSuperPatches = superPPS * superPPS
        let superPatchDim = visionHidden * d * d

        let src = encoderTokens.toFloatArray()

        // Pixel-unshuffle: for each (sy, sx) super-patch, concatenate the
        // d² constituent patch vectors in row-major (dy, dx) order.
        var unshuffled = [Float](repeating: 0, count: numSuperPatches * superPatchDim)
        for sy in 0..<superPPS {
            for sx in 0..<superPPS {
                let outRow = sy * superPPS + sx
                var col = 0
                for dy in 0..<d {
                    for dx in 0..<d {
                        let py = sy * d + dy
                        let px = sx * d + dx
                        let srcRow = py * pps + px
                        for c in 0..<visionHidden {
                            unshuffled[outRow * superPatchDim + col] =
                                src[srcRow * visionHidden + c]
                            col += 1
                        }
                    }
                }
            }
        }

        // Place into a Metal tensor for GPU norms + linear ops.
        let unshuffledT = Tensor.empty(
            shape: [numSuperPatches, superPatchDim], dtype: encoderTokens.dtype,
            device: device)
        ImagePreprocessing.copyFloats(unshuffled, into: unshuffledT)

        // LayerNorm over each super-patch row (GPU).
        let cmd1 = device.makeCommandBuffer()
        let normed = Ops.layerNorm(
            unshuffledT,
            weight: layerNorm.weight, bias: layerNorm.bias, eps: layerNorm.eps,
            nRows: numSuperPatches, rowSize: superPatchDim, on: cmd1)
        cmd1.commit()
        cmd1.waitUntilCompleted()

        // linear_1 → GELU → linear_2.
        // Use the multi-row helper so quantized (QuantizedLinear) and
        // plain (Linear) checkpoints both work correctly. Biases are
        // always stored as explicit separate tensors (bias1 / bias2) and
        // added externally — safe for both quantized and non-quantized.
        var h1 = lfm2vlApplyLinearRows(
            linear1, input: normed, nRows: numSuperPatches,
            outDim: projHidden, device: device)
        if let b1 = bias1 {
            h1 = lfm2vlBroadcastAddBias(h1, bias: b1,
                                         nRows: numSuperPatches, rowSize: projHidden,
                                         device: device)
        }

        // GELU.
        let cmd3 = device.makeCommandBuffer()
        let activated = Ops.gelu(h1, on: cmd3)
        cmd3.commit()
        cmd3.waitUntilCompleted()

        var after2 = lfm2vlApplyLinearRows(
            linear2, input: activated, nRows: numSuperPatches,
            outDim: textHidden, device: device)
        if let b2 = bias2 {
            after2 = lfm2vlBroadcastAddBias(after2, bias: b2,
                                             nRows: numSuperPatches, rowSize: textHidden,
                                             device: device)
        }
        return after2
    }
}

/// Apply an `AnyLinear` to a `[nRows, inDim]` tensor, returning
/// `[nRows, outDim]`. Biases are NOT added here — the caller adds them
/// separately via `lfm2vlBroadcastAddBias` so both quantized and plain
/// checkpoint formats work identically.
///
/// For plain `Linear` the work runs on the GPU via a single `Ops.gemm`
/// call. For `QuantizedLinear` each row is dispatched in a loop — 64
/// rows of 4608 elements is negligible next to the ViT forward pass.
private func lfm2vlApplyLinearRows(
    _ linear: AnyLinear, input: Tensor, nRows: Int, outDim: Int,
    device: Device
) -> Tensor {
    if let plain = linear.inner as? Linear {
        // GPU path: single tiled GEMM over all rows.
        let cmd = device.makeCommandBuffer()
        let out = Ops.gemm(weight: plain.weight, input: input,
                           nRows: nRows, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return out
    } else {
        // Quantized path: row-by-row dequant gemv. The row stride in the
        // flat `[nRows, inDim]` tensor is `inDim * byteSize`. Each row is
        // a `[inDim]` view; `AnyLinear` applies dequant+gemv to it.
        let inDim = input.shape[input.shape.count - 1]
        let rowBytes = inDim * input.dtype.byteSize
        var rows: [Tensor] = []
        rows.reserveCapacity(nRows)
        for r in 0..<nRows {
            let rowT = Tensor(buffer: input.buffer,
                              offset: input.offset + r * rowBytes,
                              shape: [inDim], dtype: input.dtype)
            let cmd = device.makeCommandBuffer()
            let outRow = linear(rowT, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            rows.append(outRow)
        }
        // Concatenate rows into `[nRows, outDim]`.
        let out = Tensor.empty(shape: [nRows, outDim], dtype: rows[0].dtype,
                               device: device)
        let outRowBytes = outDim * out.dtype.byteSize
        for (r, row) in rows.enumerated() {
            memcpy(out.buffer.contents().advanced(by: out.offset + r * outRowBytes),
                   row.buffer.contents().advanced(by: row.offset),
                   outRowBytes)
        }
        return out
    }
}

/// Broadcast-add a `[rowSize]` bias to each of `nRows` rows of a
/// `[nRows, rowSize]` tensor. Copies the bias once on the CPU, then uses
/// `Ops.add` (element-wise, same shape) on the GPU.
private func lfm2vlBroadcastAddBias(
    _ x: Tensor, bias: Tensor, nRows: Int, rowSize: Int, device: Device
) -> Tensor {
    let biasVals = bias.toFloatArray()
    var flat = [Float](repeating: 0, count: nRows * rowSize)
    for r in 0..<nRows {
        for c in 0..<rowSize { flat[r * rowSize + c] = biasVals[c] }
    }
    let biasT = Tensor.empty(shape: [nRows, rowSize], dtype: x.dtype, device: device)
    ImagePreprocessing.copyFloats(flat, into: biasT)
    let cmd = device.makeCommandBuffer()
    let out = Ops.add(x, biasT, on: cmd)
    cmd.commit()
    cmd.waitUntilCompleted()
    return out
}

// ─── Composed tower (encoder + projector) ────────────────────────────

/// Couples the SigLIP2 `VisionEncoder` with the LFM2-VL projector so the
/// pair presents a single `VisionEncoder`-shaped surface to `VLModel`.
/// The composed tower's `encode` produces
/// `[imageTokenCount, textHidden]` — the pixel-unshuffled, projected
/// vision tokens.
final class LFM2VLComposedTower {
    let encoder: VisionEncoder
    let projector: LFM2VLProjector
    let imageSize: Int      // encoder input resolution (256 for the simplified pass)
    let imageTokenCount: Int
    let textHidden: Int
    let dtype: DType

    init(encoder: VisionEncoder, projector: LFM2VLProjector,
         imageSize: Int, imageTokenCount: Int, textHidden: Int, dtype: DType) {
        self.encoder = encoder
        self.projector = projector
        self.imageSize = imageSize
        self.imageTokenCount = imageTokenCount
        self.textHidden = textHidden
        self.dtype = dtype
    }

    func asVisionEncoder() -> VisionEncoder {
        LFM2VLComposedEncoder(tower: self)
    }
}

/// A `VisionEncoder` subclass whose `encode` runs the SigLIP2 ViT then
/// the LFM2-VL pixel-unshuffle + projector — so `VLModel` sees one tower
/// producing `[imageTokenCount, textHidden]` tokens.
final class LFM2VLComposedEncoder: VisionEncoder {
    let tower: LFM2VLComposedTower

    init(tower: LFM2VLComposedTower) {
        self.tower = tower
        let e = tower.encoder
        // Override numPatches to the projected (post-pixel-unshuffle) count
        // so `VLModel.imageTokenCount` matches.
        let pooledConfig = VisionEncoderConfig(
            inChannels: e.config.inChannels,
            imageSize: tower.imageSize,
            patchSize: e.config.patchSize,
            hidden: e.config.hidden,
            intermediate: e.config.intermediate,
            nLayers: e.config.nLayers,
            nHeads: e.config.nHeads,
            layerNormEps: e.config.layerNormEps,
            textHidden: tower.textHidden)
        super.init(
            config: pooledConfig,
            patchEmbedWeight: e.patchEmbedWeight,
            patchEmbedBias: e.patchEmbedBias,
            positionEmbedding: e.positionEmbedding,
            layers: e.layers,
            postLayerNorm: e.postLayerNorm,
            projection: nil,
            dtype: tower.dtype)
    }

    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        // Run the SigLIP2 forward pass.
        let raw = tower.encoder.encode(image: image, device: device)
        // Pixel-unshuffle + project.
        return tower.projector.project(encoderTokens: raw, device: device)
    }
}

// ─── Quantized-aware LFM2 text-backbone loader ───────────────────────

/// Load the LFM2 text backbone with quantization support.
///
/// This is a parallel to `lfm2LoadModel` that accepts quantized checkpoints
/// by routing linear / embedding loads through `loadLinear` / `loadEmbedding`
/// with the checkpoint's quantization config. The standalone `lfm2LoadModel`
/// rejects quantized checkpoints (it is designed for raw bf16/f16 text-only
/// distributions); VL checkpoints are typically 4-bit to reduce VRAM
/// requirements.
///
/// The quantization guard from `lfm2LoadModel` is NOT present here by
/// design — this function is the VL-path equivalent.
func lfm2LoadModelQuantized(
    config: ModelConfig,
    weights: SafeTensorsBundle,
    quantization: ModelConfig.QuantizationConfig?,
    device: Device
) throws -> LFM2Model {
    guard let hidden = config.int("hidden_size") ?? config.int("block_dim"),
          let nLayers = config.int("num_hidden_layers"),
          let nHeads  = config.int("num_attention_heads") ?? config.int("num_heads"),
          let vocab   = config.int("vocab_size")
    else {
        throw LFM2VLError.missingConfig
    }
    let nKVHeads  = config.int("num_key_value_heads") ?? nHeads
    let headDim   = config.int("head_dim") ?? (hidden / nHeads)
    let eps       = Float(config.float("norm_eps") ?? config.float("block_norm_eps") ?? 1e-5)
    let theta     = Float(config.float("rope_theta") ?? 1_000_000.0)
    let maxSeq    = config.int("max_position_embeddings") ?? 128_000
    let convKernel = config.int("conv_L_cache") ?? 3
    let convBias   = config.bool("conv_bias") ?? false

    guard convKernel >= 2 else {
        throw LFM2VLError.unsupportedConfig(
            "conv_L_cache (\(convKernel)) must be ≥ 2")
    }
    guard headDim == 64 || headDim == 128 || headDim == 256 || headDim == 512 else {
        throw LFM2VLError.unsupportedConfig(
            "head_dim \(headDim) — Ops.sdpaDecode supports {64,128,256,512}")
    }

    // Layer schedule.
    let kinds = try lfm2LayerKinds(
        layerTypes: config.raw["layer_types"] as? [String],
        fullAttnIdxs: config.intArray("full_attn_idxs"),
        numLayers: nLayers)

    // Embedding (quantized-aware).
    let embedTokens = try loadEmbedding(
        base: "model.embed_tokens", in: weights,
        hidden: hidden, quantization: quantization)
    let activationDtype: DType
    switch embedTokens.weight.dtype {
    case .f32: activationDtype = .f32
    case .bf16: activationDtype = .bf16
    case .f16:  activationDtype = .f16
    default:    activationDtype = .bf16  // quantized packs → use bf16 for ops
    }

    // Per-layer construction.
    var layers: [any DecoderLayer] = []
    layers.reserveCapacity(nLayers)
    for (i, kind) in kinds.enumerated() {
        let p = "model.layers.\(i)"
        let operatorNorm = RMSNorm(
            weight: try weights.tensor(named: "\(p).operator_norm.weight"),
            eps: eps)
        let ffnNorm = RMSNorm(
            weight: try weights.tensor(named: "\(p).ffn_norm.weight"),
            eps: eps)

        let mixer: LFM2Mixer
        switch kind {
        case .conv:
            // in_proj and out_proj may be quantized.
            let inProj  = try loadLinear(base: "\(p).conv.in_proj",
                                         in: weights, quantization: quantization)
            let outProj = try loadLinear(base: "\(p).conv.out_proj",
                                         in: weights, quantization: quantization)
            // conv.conv.weight is NOT quantized (it is a small conv kernel).
            let convWSrc = try weights.tensor(named: "\(p).conv.conv.weight")
            precondition(convWSrc.elementCount == hidden * convKernel,
                         "LFM2VL: conv.conv.weight count \(convWSrc.elementCount) "
                         + "≠ hidden·kernel \(hidden * convKernel)")
            let convW = lfm2TransposeConvWeightQuantized(
                convWSrc, kernel: convKernel, channels: hidden,
                dtype: activationDtype, device: device)
            let convB: Tensor
            if convBias, weights.has("\(p).conv.conv.bias") {
                convB = lfm2CastVectorQuantized(
                    try weights.tensor(named: "\(p).conv.conv.bias"),
                    count: hidden, dtype: activationDtype, device: device)
            } else {
                convB = lfm2ZeroVectorQuantized(hidden, dtype: activationDtype, device: device)
            }
            mixer = .conv(LFM2ConvMixer(
                inProj: inProj, outProj: outProj,
                convW: convW, convB: convB,
                hidden: hidden, kernel: convKernel, dtype: activationDtype))

        case .attention:
            let qProj  = try loadLinear(base: "\(p).self_attn.q_proj",
                                         in: weights, quantization: quantization)
            let kProj  = try loadLinear(base: "\(p).self_attn.k_proj",
                                         in: weights, quantization: quantization)
            let vProj  = try loadLinear(base: "\(p).self_attn.v_proj",
                                         in: weights, quantization: quantization)
            let outProj = try loadLinear(base: "\(p).self_attn.out_proj",
                                          in: weights, quantization: quantization)
            let qNormW = lfm2CastVectorQuantized(
                try weights.tensor(named: "\(p).self_attn.q_layernorm.weight"),
                count: headDim, dtype: activationDtype, device: device)
            let kNormW = lfm2CastVectorQuantized(
                try weights.tensor(named: "\(p).self_attn.k_layernorm.weight"),
                count: headDim, dtype: activationDtype, device: device)
            mixer = .attention(LFM2AttentionMixer(
                qProj: qProj, kProj: kProj, vProj: vProj, outProj: outProj,
                qNormW: qNormW, kNormW: kNormW, normEps: eps,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                ropeTheta: theta))
        }

        // Feed-forward half — SwiGLU MLP (LFM2-VL is always dense; no MoE).
        let ffn = LFM2FFN.dense(LFM2MLP(
            w1: try loadLinear(base: "\(p).feed_forward.w1", in: weights, quantization: quantization),
            w3: try loadLinear(base: "\(p).feed_forward.w3", in: weights, quantization: quantization),
            w2: try loadLinear(base: "\(p).feed_forward.w2", in: weights, quantization: quantization)))

        layers.append(LFM2Layer(
            operatorNorm: operatorNorm, ffnNorm: ffnNorm,
            mixer: mixer, ffn: ffn, hidden: hidden))
    }

    let finalNorm = RMSNorm(
        weight: try weights.tensor(named: "model.embedding_norm.weight"),
        eps: eps)

    // LFM2 ties lm_head to embed_tokens when no standalone lm_head is stored.
    // For quantized checkpoints (e.g. mlx-community/LFM2-VL-1.6B-4bit) there
    // is no lm_head.weight in the bundle, so we fall back to the tied path.
    // When the embedding is quantized, embedTokens.weight is the PACKED uint32
    // table with shape [vocab, hidden/pack_factor]. Wrapping it in a plain
    // Linear and calling Ops.gemv crashes immediately at prewarm because
    // weight.shape[1] = hidden/8 (=256 for 2048-dim hidden) but the input
    // vector is [hidden] (=2048) — an 8× mismatch.
    //
    // Fix: when the embedding is a QuantizedEmbedding, reuse its
    // weight/scales/biases/bits/groupSize to build a QuantizedLinear. The
    // linear-to-logit projection uses the same dequant arithmetic as the
    // embedding gather (both are mlx affine-quantized rows), so this is
    // numerically correct in addition to being shape-correct.
    let lmHead: AnyLinear
    if weights.has("lm_head.weight") {
        lmHead = AnyLinear(Linear(
            weight: try weights.tensor(named: "lm_head.weight")))
    } else if let qe = embedTokens.inner as? QuantizedEmbedding {
        // Tied quantized lm_head: project via dequant gemv using the same
        // packed triplet as the embedding lookup. This avoids the gemv
        // in_dim mismatch that occurs when the packed uint32 weight
        // (shape [vocab, hidden/pack_factor]) is mistakenly treated as a
        // plain float weight of shape [vocab, hidden/pack_factor].
        lmHead = AnyLinear(QuantizedLinear(
            weight: qe.weight, scales: qe.scales, biases: qe.biases,
            bits: qe.bits, groupSize: qe.groupSize))
    } else {
        // Plain (unquantized) embedding — weight is already the full float
        // table with shape [vocab, hidden], safe to use as a Linear weight.
        lmHead = AnyLinear(Linear(weight: embedTokens.weight))
    }

    return LFM2Model(
        embedTokens: embedTokens, layers: layers,
        finalNorm: finalNorm, lmHead: lmHead,
        hidden: hidden, nLayers: nLayers,
        nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
        convDim: hidden, convKernel: convKernel,
        vocab: vocab, maxSeq: maxSeq, dtype: activationDtype)
}

// ─── Quantized-path helpers ──────────────────────────────────────────

/// Transpose an HF Conv1d weight `[channels, 1, kernel]` → `[kernel, channels]`
/// for the `conv1d_causal_step` kernel. Identical to `lfm2TransposeConvWeight`
/// in LFM2.swift but uses the module-local read helper name.
private func lfm2TransposeConvWeightQuantized(
    _ src: Tensor, kernel K: Int, channels C: Int, dtype: DType, device: Device
) -> Tensor {
    let floats = lfm2ReadFloats(src)
    precondition(floats.count == K * C, "LFM2VL: conv weight count mismatch")
    var dst = [Float](repeating: 0, count: K * C)
    for c in 0..<C { for k in 0..<K { dst[k * C + c] = floats[c * K + k] } }
    return lfm2vlWriteFloats(dst, shape: [K, C], dtype: dtype, device: device)
}

/// Cast a small per-channel / per-head vector to `dtype`.
private func lfm2CastVectorQuantized(
    _ src: Tensor, count: Int, dtype: DType, device: Device
) -> Tensor {
    if src.dtype == dtype { return src }
    let floats = lfm2ReadFloats(src)
    precondition(floats.count == count, "LFM2VL: vector size mismatch")
    return lfm2vlWriteFloats(floats, shape: [count], dtype: dtype, device: device)
}

/// Zero-filled `[n]` tensor in `dtype`.
private func lfm2ZeroVectorQuantized(_ n: Int, dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: [n], dtype: dtype, device: device)
    t.zero()
    return t
}

/// Write `[Float]` into a new tensor with the target dtype.
private func lfm2vlWriteFloats(_ values: [Float], shape: [Int],
                                dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
    switch dtype {
    case .f32:  t.copyIn(from: values)
    case .bf16: t.copyIn(from: values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) })
    case .f16:  t.copyIn(from: values.map { Float16($0) })
    default: fatalError("LFM2VL: unsupported dtype \(dtype)")
    }
    return t
}
