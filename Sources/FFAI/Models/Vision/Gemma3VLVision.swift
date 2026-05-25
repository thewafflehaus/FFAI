// Gemma 3 VL — vision tower internals.
//
// This file holds the SigLIP ViT encoder loader, the multi-modal
// projector, and the composed encoder facade for the Gemma 3 VL family.
// The family orchestrator (load entry-point + `<image>` token id) lives
// in `Models/Gemma3VL.swift`.

import Foundation
import Metal

/// Load the SigLIP ViT into a `VisionEncoder`. The checkpoint's
/// `vision_tower.vision_model.*` keys map 1:1 onto
/// `VisionEncoder.parameters()`; the encoder hidden differs from the
/// text hidden, but the projection into text-hidden is done by the
/// separate `Gemma3VLProjector`, so the `VisionEncoder` itself has
/// no projection (`textHidden == hidden`).
func gemma3vlLoadVisionEncoder(
    config: ModelConfig, textHidden: Int,
    weights: SafeTensorsBundle, device: Device
) throws -> VisionEncoder {
    guard let hidden = config.int("hidden_size"),
          let imageSize = config.int("image_size"),
          let patchSize = config.int("patch_size"),
          let intermediate = config.int("intermediate_size"),
          let nLayers = config.int("num_hidden_layers"),
          let nHeads = config.int("num_attention_heads")
    else {
        throw Gemma3Error.missingConfig
    }
    let eps = Float(config.float("layer_norm_eps") ?? 1e-6)
    let encConfig = VisionEncoderConfig(
        imageSize: imageSize, patchSize: patchSize, hidden: hidden,
        intermediate: intermediate, nLayers: nLayers, nHeads: nHeads,
        layerNormEps: eps, textHidden: hidden)

    // The mlx-converted checkpoint stores the patch-embed conv
    // weight in MLX's OHWI layout `[out_ch, kH, kW, in_ch]`;
    // `Ops.conv2d` expects PyTorch OIHW `[out_ch, in_ch, kH, kW]`.
    // Transpose if the trailing dim is the channel count (3).
    let patchWRaw = try weights.tensor(named: "embeddings.patch_embedding.weight")
    let patchW = patchWRaw.shape.count == 4 && patchWRaw.shape[3] == 3
        ? transposeOHWItoOIHW(patchWRaw)
        : patchWRaw
    let patchB = try weights.tensor(named: "embeddings.patch_embedding.bias")
    let posEmb = try weights.tensor(named: "embeddings.position_embedding.weight")

    var layers: [VisionEncoderLayer] = []
    layers.reserveCapacity(nLayers)
    for i in 0..<nLayers {
        let p = "encoder.layers.\(i)"
        let ln1 = LayerNorm(
            weight: try weights.tensor(named: "\(p).layer_norm1.weight"),
            bias: try weights.tensor(named: "\(p).layer_norm1.bias"), eps: eps)
        let ln2 = LayerNorm(
            weight: try weights.tensor(named: "\(p).layer_norm2.weight"),
            bias: try weights.tensor(named: "\(p).layer_norm2.bias"), eps: eps)
        func lin(_ name: String) throws -> Linear {
            Linear(weight: try weights.tensor(named: "\(p).\(name).weight"),
                   bias: try weights.tensor(named: "\(p).\(name).bias"))
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
        bias: try weights.tensor(named: "post_layernorm.bias"), eps: eps)

    return VisionEncoder(
        config: encConfig, patchEmbedWeight: patchW, patchEmbedBias: patchB,
        positionEmbedding: posEmb, layers: layers,
        postLayerNorm: postLN, projection: nil, dtype: patchW.dtype)
}

/// Merge Gemma 3 text-model defaults into a VLM's sparse `text_config`.
///
/// A `Gemma3ForConditionalGeneration` checkpoint stores only the
/// `text_config` fields that *differ* from the HF `Gemma3TextConfig`
/// class defaults — typically just `hidden_size`, `intermediate_size`,
/// `num_hidden_layers`, `sliding_window`, `rope_scaling`. The standalone
/// `Gemma3Dense` loader needs the full set, so the omitted fields are
/// filled here from the documented Gemma 3 defaults. Any field already
/// present in `raw` wins — only missing keys are added.
func gemma3TextConfigWithDefaults(
    _ raw: [String: Any], vocabFallback: Int?
) -> [String: Any] {
    // HF `transformers` Gemma3TextConfig defaults.
    var merged: [String: Any] = [
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocabFallback ?? 262_208,
        "query_pre_attn_scalar": 256,
        "rope_theta": 1_000_000.0,
        "rope_local_base_freq": 10_000.0,
        "sliding_window": 1024,
        "sliding_window_pattern": 6,
        "max_position_embeddings": 131_072,
        "tie_word_embeddings": true,
    ]
    // Checkpoint-declared fields override the defaults.
    for (k, v) in raw { merged[k] = v }
    return merged
}

// ─── Multi-modal projector ───────────────────────────────────────────

/// Gemma 3's vision→text projector: average-pool the encoder patch grid
/// down to `mm_tokens_per_image` tokens, GemmaRMSNorm, then project into
/// the text hidden dim. All CPU — it runs once per image.
public final class Gemma3VLProjector: @unchecked Sendable {
    /// `[vision_hidden, text_hidden]` projection matrix.
    let projectionWeight: Tensor
    /// GemmaRMSNorm over the pooled vision tokens (the `+1` fold is
    /// already baked into the loaded weight).
    let softEmbNorm: RMSNorm
    let visionHidden: Int
    let textHidden: Int
    /// Patches along one side of the encoder grid (`image / patch`).
    let patchesPerSide: Int
    /// Tokens along one side after pooling (`sqrt(mm_tokens_per_image)`).
    let tokensPerSide: Int
    /// Average-pool kernel / stride (`patchesPerSide / tokensPerSide`).
    let kernelSize: Int

    init(projectionWeight: Tensor, softEmbNorm: RMSNorm,
         visionHidden: Int, textHidden: Int,
         patchesPerSide: Int, tokensPerSide: Int) {
        self.projectionWeight = projectionWeight
        self.softEmbNorm = softEmbNorm
        self.visionHidden = visionHidden
        self.textHidden = textHidden
        self.patchesPerSide = patchesPerSide
        self.tokensPerSide = tokensPerSide
        self.kernelSize = patchesPerSide / tokensPerSide
    }

    static func load(
        visionConfig: ModelConfig, textHidden: Int, mmTokensPerImage: Int,
        weights: SafeTensorsBundle, device: Device
    ) throws -> Gemma3VLProjector {
        guard let visionHidden = visionConfig.int("hidden_size"),
              let imageSize = visionConfig.int("image_size"),
              let patchSize = visionConfig.int("patch_size")
        else {
            throw Gemma3Error.missingConfig
        }
        let patchesPerSide = imageSize / patchSize
        let tokensPerSide = Int(Double(mmTokensPerImage).squareRoot())
        let eps = visionConfig.float("layer_norm_eps") ?? 1e-6

        // mm_input_projection_weight is [vision_hidden, text_hidden].
        let projW = try weights.tensor(
            named: "multi_modal_projector.mm_input_projection_weight")
        // mm_soft_emb_norm is a GemmaRMSNorm — fold the +1 in.
        let normRaw = try weights.tensor(
            named: "multi_modal_projector.mm_soft_emb_norm.weight")
        let foldedNorm = foldGemmaRMSNormWeight(normRaw)

        return Gemma3VLProjector(
            projectionWeight: projW,
            softEmbNorm: RMSNorm(weight: foldedNorm, eps: Float(eps)),
            visionHidden: visionHidden, textHidden: textHidden,
            patchesPerSide: patchesPerSide, tokensPerSide: tokensPerSide)
    }

    /// Project `[numPatches, visionHidden]` raw encoder tokens into
    /// `[mmTokensPerImage, textHidden]`. CPU-driven: average-pool over
    /// the `kernelSize × kernelSize` patch neighbourhoods, RMSNorm each
    /// pooled token, then project.
    func project(encoderTokens: Tensor, device: Device) -> Tensor {
        let src = encoderTokens.toFloatArray()
        let pps = patchesPerSide
        let tps = tokensPerSide
        let k = kernelSize
        let vh = visionHidden

        // ── Average-pool the pps×pps grid → tps×tps ──
        var pooled = [Float](repeating: 0, count: tps * tps * vh)
        let kArea = Float(k * k)
        for ty in 0..<tps {
            for tx in 0..<tps {
                let outBase = (ty * tps + tx) * vh
                for c in 0..<vh {
                    var acc: Float = 0
                    for dy in 0..<k {
                        for dx in 0..<k {
                            let py = ty * k + dy
                            let px = tx * k + dx
                            acc += src[(py * pps + px) * vh + c]
                        }
                    }
                    pooled[outBase + c] = acc / kArea
                }
            }
        }
        let numTokens = tps * tps
        let pooledT = Tensor.empty(shape: [numTokens, vh], dtype: encoderTokens.dtype,
                                   device: device)
        ImagePreprocessing.copyFloats(pooled, into: pooledT)

        // ── GemmaRMSNorm each pooled token, then project ──
        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNormRows(
            pooledT, weight: softEmbNorm.weight, eps: softEmbNorm.eps,
            nRows: numTokens, rowSize: vh, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // projectionWeight is [visionHidden, textHidden] — projecting a
        // [vh] row means `out = norm @ W`, i.e. each output column is a
        // dot of the row with a column of W. Ops.gemm computes
        // `out = W' @ x` with W' = [outDim, inDim]; transpose W to
        // [textHidden, visionHidden] so the gemm yields [textHidden].
        let projT = transpose2D(projectionWeight, device: device)
        let cmd2 = device.makeCommandBuffer()
        let projected = Ops.gemm(weight: projT, input: normed,
                                 nRows: numTokens, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return projected
    }

    /// CPU transpose of a 2D `[r, c]` tensor → `[c, r]`.
    private func transpose2D(_ t: Tensor, device: Device) -> Tensor {
        let r = t.shape[0], c = t.shape[1]
        let src = t.toFloatArray()
        var dst = [Float](repeating: 0, count: r * c)
        for i in 0..<r {
            for j in 0..<c { dst[j * r + i] = src[i * c + j] }
        }
        let out = Tensor.empty(shape: [c, r], dtype: t.dtype, device: device)
        ImagePreprocessing.copyFloats(dst, into: out)
        return out
    }
}

// ─── Composed vision tower (encoder + projector) ─────────────────────

/// Couples the SigLIP `VisionEncoder` with the Gemma 3 projector so the
/// pair presents a single `VisionEncoder`-shaped surface to `VLModel`.
/// The composed tower's `encode` produces `[mmTokensPerImage,
/// textHidden]` — the pooled, projected vision tokens.
final class Gemma3VLVisionTower {
    let encoder: VisionEncoder
    let projector: Gemma3VLProjector
    let tokensPerImage: Int
    let textHidden: Int
    let dtype: DType

    init(encoder: VisionEncoder, projector: Gemma3VLProjector,
         tokensPerImage: Int, textHidden: Int, dtype: DType) {
        self.encoder = encoder
        self.projector = projector
        self.tokensPerImage = tokensPerImage
        self.textHidden = textHidden
        self.dtype = dtype
    }

    /// Present the composed encoder+projector as a `VisionEncoder` whose
    /// `numPatches` is the pooled token count and whose `encode` runs
    /// the SigLIP forward + the projector. Implemented by subclassing
    /// `VisionEncoder` so `VLModel` (which holds a `VisionEncoder`)
    /// transparently gets the pooled-and-projected output.
    func asVisionEncoder() -> VisionEncoder {
        Gemma3VLComposedEncoder(tower: self)
    }
}

/// A `VisionEncoder` subclass whose `encode` runs the SigLIP encoder
/// then the Gemma 3 projector — so `VLModel` sees one tower producing
/// `[mmTokensPerImage, textHidden]` tokens.
final class Gemma3VLComposedEncoder: VisionEncoder {
    let tower: Gemma3VLVisionTower

    init(tower: Gemma3VLVisionTower) {
        self.tower = tower
        // Re-expose the SigLIP encoder's geometry, but with numPatches
        // overridden (via config) to the pooled token count so
        // `VLModel.imageTokenCount` is correct.
        let e = tower.encoder
        let pooledConfig = VisionEncoderConfig(
            imageSize: e.config.imageSize, patchSize: e.config.patchSize,
            hidden: e.config.hidden, intermediate: e.config.intermediate,
            nLayers: e.config.nLayers, nHeads: e.config.nHeads,
            layerNormEps: e.config.layerNormEps, textHidden: tower.textHidden)
        super.init(
            config: pooledConfig,
            patchEmbedWeight: e.patchEmbedWeight, patchEmbedBias: e.patchEmbedBias,
            positionEmbedding: e.positionEmbedding, layers: e.layers,
            postLayerNorm: e.postLayerNorm, projection: nil, dtype: tower.dtype)
    }

    /// Run the SigLIP encoder, then pool + project. Returns
    /// `[mmTokensPerImage, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        let raw = tower.encoder.encode(image: image, device: device)
        return tower.projector.project(encoderTokens: raw, device: device)
    }
}

/// Transpose a conv2d weight from MLX's OHWI layout
/// `[out_ch, kH, kW, in_ch]` to PyTorch / `Ops.conv2d` OIHW
/// `[out_ch, in_ch, kH, kW]`. mlx-converted vision checkpoints ship
/// the OHWI layout; FFAI's conv kernel is OIHW-native.
func transposeOHWItoOIHW(_ w: Tensor) -> Tensor {
    precondition(w.shape.count == 4,
                 "transposeOHWItoOIHW: weight must be 4D, got \(w.shape)")
    let oc = w.shape[0], kh = w.shape[1], kw = w.shape[2], ic = w.shape[3]
    let src = w.toFloatArray()
    var dst = [Float](repeating: 0, count: oc * ic * kh * kw)
    for o in 0..<oc {
        for y in 0..<kh {
            for x in 0..<kw {
                for c in 0..<ic {
                    let srcIdx = ((o * kh + y) * kw + x) * ic + c
                    let dstIdx = ((o * ic + c) * kh + y) * kw + x
                    dst[dstIdx] = src[srcIdx]
                }
            }
        }
    }
    let out = Tensor.empty(shape: [oc, ic, kh, kw], dtype: w.dtype)
    ImagePreprocessing.copyFloats(dst, into: out)
    return out
}

/// Fold the GemmaRMSNorm `+1.0` offset into a raw norm weight — the
/// shared-with-`Gemma3.swift` recipe, replicated here because
/// `loadGemmaRMSNorm` is `private` to that file. Returns a fresh f32 /
/// f16 / bf16 tensor.
func foldGemmaRMSNormWeight(_ raw: Tensor) -> Tensor {
    precondition(raw.shape.count == 1,
                 "foldGemmaRMSNormWeight: weight must be 1D, got \(raw.shape)")
    let n = raw.elementCount
    let foldedBuf = Device.shared.makeBuffer(length: raw.byteCount)
    let folded = Tensor(buffer: foldedBuf, offset: 0, shape: raw.shape, dtype: raw.dtype)
    let dst = foldedBuf.contents()
    let src = raw.buffer.contents().advanced(by: raw.offset)
    switch raw.dtype {
    case .f32:
        let s = src.bindMemory(to: Float.self, capacity: n)
        let d = dst.bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { d[i] = s[i] + 1.0 }
    case .f16:
        let s = src.bindMemory(to: Float16.self, capacity: n)
        let d = dst.bindMemory(to: Float16.self, capacity: n)
        for i in 0..<n { d[i] = Float16(Float(s[i]) + 1.0) }
    case .bf16:
        let s = src.bindMemory(to: UInt16.self, capacity: n)
        let d = dst.bindMemory(to: UInt16.self, capacity: n)
        for i in 0..<n {
            let f = Float(bitPattern: UInt32(s[i]) << 16) + 1.0
            let bits = f.bitPattern
            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
            d[i] = UInt16(rounded >> 16)
        }
    default:
        fatalError("foldGemmaRMSNormWeight: unsupported dtype \(raw.dtype)")
    }
    return folded
}
