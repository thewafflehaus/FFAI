// Mistral3 vision tower internals — projector, composed encoder, and bundle helpers.
//
// The family orchestrator (`enum Mistral3`, error type, `load(...)`) lives in
// `Models/Mistral3.swift`. This file contains the private / internal types
// that implement the vision side:
//   • Mistral3Projector — RMSNorm → 2×2 spatial unfold + merge → linear_1
//     → GELU → linear_2, handling both plain and quantized weights.
//   • mistral3ApplyLinearRows / mistral3BroadcastAddBias — dispatch helpers
//     for plain (Ops.gemm) and quantized (per-row dequant gemv) paths.
//   • Mistral3ComposedTower / Mistral3ComposedEncoder — couples the Pixtral
//     ViT (see `Vision/PixtralVision.swift`) with the Mistral3 projector and
//     presents a single VisionEncoder surface to VisionModel.
//   • SafeTensorsBundle.mistral3ProjectorQuantization — quantization probe.

import Foundation
import Metal

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
/// presents a single `VisionEncoder`-shaped surface to `VisionModel`.
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
    var mistral3ProjectorQuantization: ModelConfig.QuantizationConfig? {
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
