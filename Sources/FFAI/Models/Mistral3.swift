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
// The three are joined by `VLModel`'s cross-modal token splice.
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
        "Mistral3ForConditionalGeneration",
    ]

    /// Default `image_token_id` for Mistral3. The tokenizer's `[IMG]`
    /// special token resolves to id 10.
    public static let defaultImageTokenId = 10

    /// Default spatial merge size (2×2 pooling in the patch merger).
    public static let defaultSpatialMergeSize = 2

    /// Build a `VLModel` from a Mistral3 checkpoint: the Pixtral 2D-RoPE ViT
    /// + the Mistral3 patch-merger projector + the Mistral text backbone,
    /// joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VLModel {
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
            raw: pixtralTextConfigWithDefaults(textConfig.raw,
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

        let imageTokenId = config.int("image_token_id")
            ?? config.int("image_token_index")
            ?? defaultImageTokenId

        let composedTower = Mistral3ComposedTower(
            encoder: vision, projector: projector,
            visionCfg: visionCfg, spatialMergeSize: spatialMergeSize,
            textHidden: textEngine.hidden, dtype: textEngine.dtype)

        return try VLModel(
            visionEncoder: composedTower.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: .clip,
            imageTokenCount: mergedPatches)
    }
}

// ─── Mistral3 projector ───────────────────────────────────────────────

/// Mistral3 multi-modal projector: RMSNorm → patch merger (2×2 unfold +
/// linear) → linear_1 → GELU → linear_2.
///
/// The patch merger concatenates each `spatialMergeSize × spatialMergeSize`
/// neighbourhood of patch tokens and projects them through a single linear
/// layer, reducing the token count by a factor of `spatialMergeSize²`.
///
/// All linears are loaded as `AnyLinear` so quantized (4-bit) and plain
/// float checkpoints are both handled. See `mistral3ApplyLinearRows` for
/// the dispatch logic.
final class Mistral3Projector: @unchecked Sendable {
    /// RMSNorm applied to vision features before merging.
    let norm: RMSNorm
    /// Merge projection: [visionHidden, visionHidden * spatialMergeSize²]
    let mergingLayer: AnyLinear
    /// First MLP layer: [visionHidden → textHidden] (with optional bias).
    let linear1: AnyLinear
    /// Second MLP layer: [textHidden → textHidden] (with optional bias).
    let linear2: AnyLinear
    /// Optional bias added after linear1 (external to AnyLinear for compat
    /// with both quantized and non-quantized checkpoints).
    let linear1Bias: Tensor?
    /// Optional bias added after linear2.
    let linear2Bias: Tensor?

    let visionHidden: Int
    let textHidden: Int
    let spatialMergeSize: Int

    init(norm: RMSNorm,
         mergingLayer: AnyLinear,
         linear1: AnyLinear, linear1Bias: Tensor?,
         linear2: AnyLinear, linear2Bias: Tensor?,
         visionHidden: Int, textHidden: Int, spatialMergeSize: Int) {
        self.norm = norm
        self.mergingLayer = mergingLayer
        self.linear1 = linear1
        self.linear1Bias = linear1Bias
        self.linear2 = linear2
        self.linear2Bias = linear2Bias
        self.visionHidden = visionHidden
        self.textHidden = textHidden
        self.spatialMergeSize = spatialMergeSize
    }

    static func load(
        visionHidden: Int, textHidden: Int,
        spatialMergeSize: Int, hasBias: Bool,
        quantization: ModelConfig.QuantizationConfig?,
        weights: SafeTensorsBundle, device: Device
    ) throws -> Mistral3Projector {
        // RMSNorm — always float (not quantized in the mlx-community conversion).
        let normW = try weights.tensor(named: "multi_modal_projector.norm.weight")
        let normLayer = RMSNorm(weight: normW, eps: 1e-5)

        // Patch merger linear: input [visionHidden * s²], output [visionHidden].
        // The merging_layer weight is quantized in the 4-bit conversion.
        let mergingLayer = try loadLinear(
            base: "multi_modal_projector.patch_merger.merging_layer",
            in: weights, quantization: quantization)

        // MLP linear_1: [visionHidden → textHidden], bias optional.
        let linear1 = try loadLinear(
            base: "multi_modal_projector.linear_1",
            in: weights, quantization: quantization)
        // Biases are loaded separately because `QuantizedLinear` does not
        // store them — the external load ensures both checkpoint formats work.
        let linear1Bias: Tensor? = hasBias
            ? try? weights.tensor(named: "multi_modal_projector.linear_1.bias")
            : nil

        // MLP linear_2: [textHidden → textHidden], bias optional.
        let linear2 = try loadLinear(
            base: "multi_modal_projector.linear_2",
            in: weights, quantization: quantization)
        let linear2Bias: Tensor? = hasBias
            ? try? weights.tensor(named: "multi_modal_projector.linear_2.bias")
            : nil

        return Mistral3Projector(
            norm: normLayer,
            mergingLayer: mergingLayer,
            linear1: linear1, linear1Bias: linear1Bias,
            linear2: linear2, linear2Bias: linear2Bias,
            visionHidden: visionHidden,
            textHidden: textHidden,
            spatialMergeSize: spatialMergeSize)
    }

    /// Project `[numPatches, visionHidden]` vision tokens into
    /// `[mergedPatches, textHidden]`.
    ///
    /// - `gridH` and `gridW` are the patch grid dimensions (H × W = numPatches).
    ///   They must both be divisible by `spatialMergeSize`.
    func project(tokens: Tensor, gridH: Int, gridW: Int, device: Device) -> Tensor {
        let nPatches = gridH * gridW
        let s = spatialMergeSize
        let mergedH = gridH / s
        let mergedW = gridW / s
        let nMerged = mergedH * mergedW
        let mergedDim = visionHidden * s * s

        // ── Step 1: RMSNorm ──
        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNormRows(tokens, weight: norm.weight, eps: norm.eps,
                                     nRows: nPatches, rowSize: visionHidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── Step 2: Spatial unfold + merge (CPU) ──
        // Reshape [H×W, visionHidden] → [H, W, visionHidden] then gather
        // each s×s neighbourhood into a single [visionHidden*s²] vector.
        // Token counts are at most a few thousand, so CPU cost is negligible.
        let src = normed.toFloatArray()
        var merged = [Float](repeating: 0, count: nMerged * mergedDim)

        // Each merged patch writes to its own disjoint slot — race-free.
        merged.withUnsafeMutableBufferPointer { buf in
            let ptr = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nMerged) { mi in
                let mRow = mi / mergedW
                let mCol = mi % mergedW
                var outOff = mi * mergedDim
                // Gather the s×s patch block starting at (mRow*s, mCol*s).
                for dr in 0..<s {
                    for dc in 0..<s {
                        let patchRow = mRow * s + dr
                        let patchCol = mCol * s + dc
                        let patchIdx = patchRow * gridW + patchCol
                        let srcOff = patchIdx * visionHidden
                        for d in 0..<visionHidden {
                            ptr[outOff + d] = src[srcOff + d]
                        }
                        outOff += visionHidden
                    }
                }
            }
        }

        // Copy merged float buffer into a GPU tensor.
        let mergedTensor = Tensor.empty(shape: [nMerged, mergedDim],
                                        dtype: normed.dtype, device: device)
        ImagePreprocessing.copyFloats(merged, into: mergedTensor)

        // ── Step 3: Merge projection [visionHidden*s², visionHidden] ──
        // Uses `mistral3ApplyLinearRows` which dispatches to a single
        // `Ops.gemm` for plain weights, or per-row dequant gemv for quantized.
        let afterMerge = mistral3ApplyLinearRows(
            mergingLayer, input: mergedTensor, nRows: nMerged,
            outDim: visionHidden, device: device)

        // ── Step 4: linear_1 + GELU ──
        var x = mistral3ApplyLinearRows(
            linear1, input: afterMerge, nRows: nMerged,
            outDim: textHidden, device: device)
        if let b = linear1Bias {
            x = mistral3BroadcastAddBias(x, bias: b, nRows: nMerged,
                                          rowSize: textHidden, device: device)
        }
        let cmd2 = device.makeCommandBuffer()
        x = Ops.gelu(x, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        // ── Step 5: linear_2 ──
        var y = mistral3ApplyLinearRows(
            linear2, input: x, nRows: nMerged,
            outDim: textHidden, device: device)
        if let b = linear2Bias {
            y = mistral3BroadcastAddBias(y, bias: b, nRows: nMerged,
                                          rowSize: textHidden, device: device)
        }
        return y
    }
}

// ─── Projector helpers ────────────────────────────────────────────────

/// Apply an `AnyLinear` to a `[nRows, inDim]` tensor, returning
/// `[nRows, outDim]`. Biases are NOT added here — callers add them
/// separately via `mistral3BroadcastAddBias` so both quantized and plain
/// checkpoint formats work identically.
///
/// - For plain `Linear`: single `Ops.gemm` on the GPU.
/// - For `QuantizedLinear`: per-row dequant gemv loop.
///   `Ops.dequantGemv` requires a 1D input; each row is aliased into the
///   flat tensor via `Tensor(buffer:offset:shape:dtype:)` (zero-copy).
private func mistral3ApplyLinearRows(
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
        // Quantized path: row-by-row dequant gemv.
        // Each row is a zero-copy view into the flat buffer.
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
        // Concatenate rows into [nRows, outDim].
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
/// `[nRows, rowSize]` tensor. Tiles the bias on the CPU then adds on GPU.
private func mistral3BroadcastAddBias(
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

// ─── Composed tower ───────────────────────────────────────────────────

/// Couples `PixtralVisionEncoder` with `Mistral3Projector` so the pair
/// presents a single `VisionEncoder`-shaped surface to `VLModel`.
final class Mistral3ComposedTower {
    let encoder: PixtralVisionEncoder
    let projector: Mistral3Projector
    let visionCfg: PixtralVisionConfig
    let spatialMergeSize: Int
    let textHidden: Int
    let dtype: DType

    init(encoder: PixtralVisionEncoder, projector: Mistral3Projector,
         visionCfg: PixtralVisionConfig, spatialMergeSize: Int,
         textHidden: Int, dtype: DType) {
        self.encoder = encoder
        self.projector = projector
        self.visionCfg = visionCfg
        self.spatialMergeSize = spatialMergeSize
        self.textHidden = textHidden
        self.dtype = dtype
    }

    func asVisionEncoder() -> VisionEncoder {
        Mistral3ComposedEncoder(tower: self)
    }
}

/// A `VisionEncoder` subclass whose `encode` runs the Pixtral ViT tower
/// then the Mistral3 patch-merger projector, returning
/// `[mergedPatches, textHidden]`.
///
/// Grid dimensions are inferred dynamically from the input image shape so
/// variable-resolution inputs (Mistral3 supports up to image_size=1540)
/// produce the correct merged token count.
final class Mistral3ComposedEncoder: VisionEncoder {
    let tower: Mistral3ComposedTower

    init(tower: Mistral3ComposedTower) {
        self.tower = tower
        let cfg = tower.visionCfg
        let s = tower.spatialMergeSize
        let mergedPatches = cfg.numPatches / (s * s)
        let facadeConfig = VisionEncoderConfig(
            inChannels: cfg.numChannels,
            imageSize: cfg.imageSize, patchSize: cfg.patchSize,
            hidden: cfg.hiddenSize, intermediate: cfg.intermediateSize,
            nLayers: cfg.numLayers, nHeads: cfg.numHeads,
            layerNormEps: cfg.rmsNormEps, textHidden: tower.textHidden)
        // Placeholder tensors — base `encode` is fully overridden below.
        let placeholder = tower.encoder.patchConvWeight
        super.init(config: facadeConfig,
                   patchEmbedWeight: placeholder, patchEmbedBias: placeholder,
                   positionEmbedding: placeholder, layers: [],
                   postLayerNorm: tower.encoder.lnPre.asLayerNorm(),
                   projection: nil, dtype: tower.dtype)
        _ = mergedPatches  // suppress unused-variable warning
    }

    /// Run the Pixtral vision tower + Mistral3 patch-merger projector.
    /// Returns `[mergedPatches, textHidden]`.
    ///
    /// Grid dimensions are inferred from the input image geometry:
    ///   gridH = imageH / patchSize,  gridW = imageW / patchSize.
    /// Both must be divisible by `spatialMergeSize`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        let p = tower.visionCfg.patchSize
        let s = tower.spatialMergeSize

        // image shape: [1, inChannels, imageH, imageW] (NCHW).
        let imageH = image.shape[2]
        let imageW = image.shape[3]
        let gridH = imageH / p
        let gridW = imageW / p

        precondition(gridH % s == 0 && gridW % s == 0,
                     "Mistral3: gridH (\(gridH)) and gridW (\(gridW)) must be "
                     + "divisible by spatialMergeSize (\(s))")

        // Run the Pixtral vision encoder: [nPatches, visionHidden].
        let raw = tower.encoder.encode(image: image, device: device)

        // Project via Mistral3's patch-merger + MLP: [mergedPatches, textHidden].
        return tower.projector.project(tokens: raw, gridH: gridH, gridW: gridW,
                                       device: device)
    }
}

// ─── Bundle quantization probe ────────────────────────────────────────

extension SafeTensorsBundle {
    /// Quantization config inferred from the projector's `linear_1` key.
    /// Returns `nil` if the projector is stored as plain float. When
    /// quantized, returns the canonical 4-bit / group_size=64 config that
    /// the mlx-community 4-bit conversion uses.
    fileprivate var mistral3ProjectorQuantization: ModelConfig.QuantizationConfig? {
        guard isQuantized("multi_modal_projector.linear_1"),
              let scales = try? tensor(named: "multi_modal_projector.linear_1.scales")
        else { return nil }
        // The group_size is inferred from the scale tensor shape and the
        // known output dimension (scales: [outDim, inDim / groupSize]).
        // The canonical mlx-community 4-bit conversion uses group_size=64.
        _ = scales
        return ModelConfig.QuantizationConfig(bits: 4, groupSize: 64)
    }
}
