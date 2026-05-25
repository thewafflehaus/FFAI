// Qwen 3-VL — vision tower internals.
//
// This file contains `Qwen3VLVisionConfig`, `Qwen3VLVisionBlock`,
// `Qwen3VLVisionModel`, `Qwen3VLComposedEncoder`, and all supporting
// CPU / GPU helpers for the dynamic-resolution ViT (Conv3d patch embed,
// LayerNorm pre-norms, learned position table, M-RoPE, full bidirectional
// attention, patch merger). The family orchestrator (load entry-point +
// token ids) lives in `Models/Qwen3.swift`.

import Foundation
import Metal

// ─── Qwen 3-VL vision tower ──────────────────────────────────────────

/// Static shape of the Qwen 3-VL vision tower, decoded from
/// `vision_config`.
struct Qwen3VLVisionConfig {
    let depth: Int
    let hidden: Int
    let intermediate: Int
    let outHidden: Int
    let numHeads: Int
    let patchSize: Int
    let spatialMergeSize: Int
    let temporalPatchSize: Int
    let numPositionEmbeddings: Int
    let inChannels: Int
    let layerNormEps: Float

    /// Per-head dimension.
    var headDim: Int { hidden / numHeads }
    /// `mergeSize²` — patches pooled into one merged token.
    var mergeUnit: Int { spatialMergeSize * spatialMergeSize }

    static func decode(_ c: ModelConfig) throws -> Qwen3VLVisionConfig {
        guard let depth = c.int("depth"),
              let hidden = c.int("hidden_size"),
              let numHeads = c.int("num_heads"),
              let patchSize = c.int("patch_size"),
              let mergeSize = c.int("spatial_merge_size")
        else {
            throw Qwen3VLError.missingConfig
        }
        let intermediate = c.int("intermediate_size") ?? hidden * 4
        let outHidden = c.int("out_hidden_size") ?? hidden
        // `num_position_embeddings` is a perfect square (the learned
        // position table covers a `√N × √N` grid).
        let numPos = c.int("num_position_embeddings") ?? (32 * 32)
        return Qwen3VLVisionConfig(
            depth: depth, hidden: hidden, intermediate: intermediate,
            outHidden: outHidden, numHeads: numHeads, patchSize: patchSize,
            spatialMergeSize: mergeSize,
            temporalPatchSize: c.int("temporal_patch_size") ?? 2,
            numPositionEmbeddings: numPos,
            inChannels: c.int("in_channels") ?? c.int("in_chans") ?? 3,
            layerNormEps: Float(c.float("layer_norm_eps") ?? 1e-6))
    }
}

/// One Qwen 3-VL vision block: LayerNorm → full MHA + M-RoPE → residual,
/// LayerNorm → GELU MLP → residual. Held as plain weight tensors; the
/// forward runs CPU attention + GPU GEMMs.
final class Qwen3VLVisionBlock {
    let norm1: LayerNorm
    let norm2: LayerNorm
    let qkv: Linear            // fused [3·hidden, hidden] (+ bias)
    let proj: Linear           // [hidden, hidden] (+ bias)
    let fc1: Linear            // GELU-MLP up   [paddedIntermediate, hidden]
    let fc2: Linear            // GELU-MLP down [hidden, paddedIntermediate]
    let cfg: Qwen3VLVisionConfig
    /// MLP intermediate dim rounded up to the GEMM K-tile width — `fc1`'s
    /// output rows are zero-extended and `fc2`'s input columns
    /// zero-padded to it, so the `fc2` projection's `inDim` is aligned.
    let paddedIntermediate: Int

    init(norm1: LayerNorm, norm2: LayerNorm, qkv: Linear, proj: Linear,
         fc1: Linear, fc2: Linear,
         paddedIntermediate: Int, cfg: Qwen3VLVisionConfig) {
        self.norm1 = norm1; self.norm2 = norm2
        self.qkv = qkv; self.proj = proj
        self.fc1 = fc1; self.fc2 = fc2
        self.paddedIntermediate = paddedIntermediate
        self.cfg = cfg
    }

    /// Forward `[nTokens, hidden]` activations through one block.
    /// `cosTable` / `sinTable` are the precomputed per-token rotary
    /// tables `[nTokens, headDim]`.
    func forward(_ h: Tensor, nTokens: Int,
                 cosTable: [Float], sinTable: [Float],
                 device: Device) -> Tensor {
        let hidden = cfg.hidden
        // ── Attention sub-block ──
        let cmd = device.makeCommandBuffer()
        let normed = Ops.layerNorm(h, weight: norm1.weight, bias: norm1.bias,
                                   eps: norm1.eps, nRows: nTokens,
                                   rowSize: hidden, on: cmd)
        let qkvOut = projectRows(qkv, normed, nTokens: nTokens,
                                 outDim: 3 * hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let attn = cpuAttention(qkv: qkvOut, nTokens: nTokens,
                                cosTable: cosTable, sinTable: sinTable,
                                device: device)

        // ── Residual + MLP sub-block ──
        let cmd2 = device.makeCommandBuffer()
        let attnProj = projectRows(proj, attn, nTokens: nTokens,
                                   outDim: hidden, on: cmd2)
        let postAttn = Ops.add(h, attnProj, on: cmd2)
        let normed2 = Ops.layerNorm(postAttn, weight: norm2.weight,
                                    bias: norm2.bias, eps: norm2.eps,
                                    nRows: nTokens, rowSize: hidden, on: cmd2)
        let up = projectRows(fc1, normed2, nTokens: nTokens,
                             outDim: paddedIntermediate, on: cmd2)
        let act = Ops.gelu(up, on: cmd2)
        let down = projectRows(fc2, act, nTokens: nTokens,
                               outDim: hidden, on: cmd2)
        let result = Ops.add(postAttn, down, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return result
    }

    /// CPU full bidirectional multi-head attention with M-RoPE. `qkv` is
    /// the fused `[nTokens, 3·hidden]` projection. Returns the context
    /// token-major `[nTokens, hidden]`.
    ///
    /// Two-stage parallelization with `DispatchQueue.concurrentPerform`:
    ///   1. Setup — M-RoPE: each `(head, token)` pair writes its own
    ///      `qH[head*nTokens+t]` / `kH[head*nTokens+t]` / `vH[head*nTokens+t]`
    ///      row — disjoint, race-free.
    ///   2. Attention — per `(head, query-row)`: writes to a disjoint
    ///      `[oBase, oBase + headDim)` output slice.
    private func cpuAttention(qkv: Tensor, nTokens: Int,
                              cosTable: [Float], sinTable: [Float],
                              device: Device) -> Tensor {
        let nHeads = cfg.numHeads
        let headDim = cfg.headDim
        let hidden = cfg.hidden
        let qkvA = qkv.toFloatArray()
        var out = [Float](repeating: 0, count: nTokens * hidden)
        let scale = 1.0 / Float(Double(headDim).squareRoot())
        let half = headDim / 2

        // Stage 1: Extract and RoPE every (head, token) slice.
        // Index layout: qH[head * nTokens + t] — each (head, t) pair owns
        // its slot, so concurrent writes are race-free.
        var qH = [[Float]](repeating: [], count: nHeads * nTokens)
        var kH = [[Float]](repeating: [], count: nHeads * nTokens)
        var vH = [[Float]](repeating: [], count: nHeads * nTokens)
        DispatchQueue.concurrentPerform(iterations: nHeads * nTokens) { work in
            let head = work / nTokens
            let t = work % nTokens
            let hOff = head * headDim
            let base = t * 3 * hidden
            var q = Array(qkvA[(base + hOff)..<(base + hOff + headDim)])
            var k = Array(qkvA[(base + hidden + hOff)..<(base + hidden + hOff + headDim)])
            let v = Array(qkvA[(base + 2 * hidden + hOff)..<(base + 2 * hidden + hOff + headDim)])
            // Apply M-RoPE inline. Qwen's rotate-half scheme:
            // out[d] = x[d]·cos − x[d±half]·sin. Defined inside the
            // closure to avoid @Sendable capture warnings.
            func applyRope(_ x: inout [Float], tok: Int) {
                let cb = tok * headDim
                for d in 0..<headDim {
                    let rotated = d < half ? -x[d + half] : x[d - half]
                    let c = cosTable[cb + d], s = sinTable[cb + d]
                    x[d] = x[d] * c + rotated * s
                }
            }
            applyRope(&q, tok: t)
            applyRope(&k, tok: t)
            qH[head * nTokens + t] = q
            kH[head * nTokens + t] = k
            vH[head * nTokens + t] = v
        }

        // Stage 2: Full attention — every token attends to every token.
        // Each (head, i) writes to a disjoint [oBase, oBase+headDim) slice.
        out.withUnsafeMutableBufferPointer { outBuf in
            let outPtr = outBuf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nHeads * nTokens) { work in
                let head = work / nTokens
                let i = work % nTokens
                let hOff = head * headDim
                var scores = [Float](repeating: 0, count: nTokens)
                var maxS = -Float.greatestFiniteMagnitude
                let qVec = qH[head * nTokens + i]
                for j in 0..<nTokens {
                    var dot: Float = 0
                    let kVec = kH[head * nTokens + j]
                    for d in 0..<headDim { dot += qVec[d] * kVec[d] }
                    let s = dot * scale
                    scores[j] = s
                    if s > maxS { maxS = s }
                }
                var sum: Float = 0
                for j in 0..<nTokens {
                    let e = exp(scores[j] - maxS)
                    scores[j] = e; sum += e
                }
                let inv = sum > 0 ? 1 / sum : 0
                let oBase = i * hidden + hOff
                for j in 0..<nTokens {
                    let w = scores[j] * inv
                    let vVec = vH[head * nTokens + j]
                    for d in 0..<headDim { outPtr[oBase + d] += w * vVec[d] }
                }
            }
        }
        let result = Tensor.empty(shape: [nTokens, hidden], dtype: qkv.dtype,
                                  device: device)
        ImagePreprocessing.copyFloats(out, into: result)
        return result
    }

    /// Apply a `Linear` to every row of `[nTokens, *]` via `Ops.gemm`,
    /// then broadcast-add the bias if present.
    private func projectRows(_ linear: Linear, _ x: Tensor, nTokens: Int,
                             outDim: Int, on cmd: MTLCommandBuffer) -> Tensor {
        let y = Ops.gemm(weight: linear.weight, input: x, nRows: nTokens, on: cmd)
        guard let bias = linear.bias else { return y }
        return addRowBias(y, bias: bias, nRows: nTokens,
                          rowSize: outDim, on: cmd)
    }
}

/// The Qwen 3-VL vision tower. Holds the patch-embed projection, the
/// learned position table, the block stack, and the patch-merger;
/// `encode` runs the full forward and returns merged tokens in the text
/// hidden dim.
final class Qwen3VLVisionModel: @unchecked Sendable {
    let cfg: Qwen3VLVisionConfig
    /// Flattened patch-embed weight `[hidden, patchDimPadded]`.
    let patchEmbedWeight: Tensor
    /// Patch-embed bias `[hidden]` (Qwen3-VL's Conv3d carries a bias).
    let patchEmbedBias: Tensor?
    /// `patchDim` padded up to a multiple of the GEMM K-tile width.
    let patchDimPadded: Int
    /// Learned position embedding table `[numPositionEmbeddings, hidden]`.
    let posEmbedTable: Tensor
    let blocks: [Qwen3VLVisionBlock]
    /// Patch-merger pre-shuffle LayerNorm over the pre-merge hidden.
    let mergerNorm: LayerNorm
    /// Patch-merger MLP: `[mergeUnit·hidden] → [mergeUnit·hidden] → [outHidden]`.
    let mergerFC1: Linear
    let mergerFC2: Linear
    let textHidden: Int
    let dtype: DType
    /// Patch grid side (square test image at the encoder's fixed size).
    let gridSide: Int

    /// Number of merged vision tokens one image contributes.
    var mergedTokenCount: Int {
        let llmSide = gridSide / cfg.spatialMergeSize
        return llmSide * llmSide
    }

    /// Number of merged vision tokens a video of `frameCount` frames
    /// contributes — `(T/tp) * (gridSide/m)²`. Used by callers to
    /// compute placeholder counts.
    func mergedTokenCount(frameCount: Int) -> Int {
        let llmSide = gridSide / cfg.spatialMergeSize
        let gridT = paddedFrameCount(frameCount) / cfg.temporalPatchSize
        return gridT * llmSide * llmSide
    }

    /// Round `frameCount` up to a multiple of `temporal_patch_size` by
    /// repeating the last frame (matches the mlx-vlm reference's
    /// `patchify` padding).
    func paddedFrameCount(_ frameCount: Int) -> Int {
        let tp = cfg.temporalPatchSize
        let mod = frameCount % tp
        return mod == 0 ? frameCount : frameCount + (tp - mod)
    }

    init(cfg: Qwen3VLVisionConfig, patchEmbedWeight: Tensor,
         patchEmbedBias: Tensor?, patchDimPadded: Int, posEmbedTable: Tensor,
         blocks: [Qwen3VLVisionBlock], mergerNorm: LayerNorm,
         mergerFC1: Linear, mergerFC2: Linear, textHidden: Int,
         dtype: DType, gridSide: Int) {
        self.cfg = cfg
        self.patchEmbedWeight = patchEmbedWeight
        self.patchEmbedBias = patchEmbedBias
        self.patchDimPadded = patchDimPadded
        self.posEmbedTable = posEmbedTable
        self.blocks = blocks
        self.mergerNorm = mergerNorm
        self.mergerFC1 = mergerFC1
        self.mergerFC2 = mergerFC2
        self.textHidden = textHidden
        self.dtype = dtype
        self.gridSide = gridSide
    }

    /// The patch grid side for a square image at the standard Qwen3-VL
    /// test resolution. Sized so the grid divides the merge size cleanly
    /// and matches the learned position table side (`√numPositionEmb`).
    static let defaultGridSide = 32

    static func load(
        visionConfig: ModelConfig, textHidden: Int,
        weights: SafeTensorsBundle, dtype: DType, device: Device
    ) throws -> Qwen3VLVisionModel {
        let cfg = try Qwen3VLVisionConfig.decode(visionConfig)

        // Patch-embed: a Conv3d weight `[hidden, in_ch, tPatch, p, p]`
        // (or MLX channel-last `[hidden, tPatch, p, p, in_ch]`). Flatten
        // to a 2D GEMM weight `[hidden, in_ch·tPatch·p·p]`.
        let rawPatch = try weights.tensor(named: "patch_embed.proj.weight")
        let patchDim = cfg.inChannels * cfg.temporalPatchSize
            * cfg.patchSize * cfg.patchSize
        let kTile = gemmKTileWidth
        let patchDimPadded = ((patchDim + kTile - 1) / kTile) * kTile
        let patchEmbedWeight = flattenPatchEmbed(
            rawPatch, hidden: cfg.hidden, patchDim: patchDim,
            patchDimPadded: patchDimPadded, device: device)
        let patchEmbedBias = try? weights.tensor(named: "patch_embed.proj.bias")

        // Learned position embedding table.
        let posEmbedTable = try weights.tensor(named: "pos_embed.weight")

        // Block stack — GELU-MLP intermediate padded to the K-tile width.
        let paddedIntermediate =
            ((cfg.intermediate + kTile - 1) / kTile) * kTile
        var blocks: [Qwen3VLVisionBlock] = []
        blocks.reserveCapacity(cfg.depth)
        for i in 0..<cfg.depth {
            let p = "blocks.\(i)"
            func lin(_ name: String) throws -> Linear {
                let w = try weights.tensor(named: "\(p).\(name).weight")
                let b = try? weights.tensor(named: "\(p).\(name).bias")
                return Linear(weight: w, bias: b)
            }
            func norm(_ name: String) throws -> LayerNorm {
                LayerNorm(weight: try weights.tensor(named: "\(p).\(name).weight"),
                          bias: try weights.tensor(named: "\(p).\(name).bias"),
                          eps: cfg.layerNormEps)
            }
            let fc1 = try lin("mlp.linear_fc1")
            let fc2 = try lin("mlp.linear_fc2")
            blocks.append(Qwen3VLVisionBlock(
                norm1: try norm("norm1"), norm2: try norm("norm2"),
                qkv: try lin("attn.qkv"), proj: try lin("attn.proj"),
                fc1: padLinearRows(
                    fc1, toRows: paddedIntermediate, device: device),
                fc2: padLinearCols(
                    fc2, toCols: paddedIntermediate, device: device),
                paddedIntermediate: paddedIntermediate, cfg: cfg))
        }

        // Patch-merger (`usePostShuffleNorm == false` → norm over hidden).
        let mergerNorm = LayerNorm(
            weight: try weights.tensor(named: "merger.norm.weight"),
            bias: try weights.tensor(named: "merger.norm.bias"),
            eps: cfg.layerNormEps)
        func mergerLin(_ name: String) throws -> Linear {
            let w = try weights.tensor(named: "merger.\(name).weight")
            let b = try? weights.tensor(named: "merger.\(name).bias")
            return Linear(weight: w, bias: b)
        }
        let mergerFC1 = try mergerLin("linear_fc1")
        let mergerFC2 = try mergerLin("linear_fc2")

        // Size the test grid to the learned position table side so the
        // position lookup is a direct (un-interpolated) gather.
        let posSide = Int(Double(cfg.numPositionEmbeddings).squareRoot())
        let gridSide = posSide > 0 && posSide % cfg.spatialMergeSize == 0
            ? posSide : defaultGridSide

        return Qwen3VLVisionModel(
            cfg: cfg, patchEmbedWeight: patchEmbedWeight,
            patchEmbedBias: patchEmbedBias, patchDimPadded: patchDimPadded,
            posEmbedTable: posEmbedTable, blocks: blocks,
            mergerNorm: mergerNorm, mergerFC1: mergerFC1, mergerFC2: mergerFC2,
            textHidden: textHidden, dtype: dtype, gridSide: gridSide)
    }

    /// Run the full vision forward on a preprocessed image. `image` is a
    /// normalized NCHW tensor `[1, inChannels, side, side]` where
    /// `side = gridSide · patchSize`. Returns `[mergedTokenCount,
    /// textHidden]`.
    ///
    /// Thin wrapper: delegates to the shared multi-frame path with
    /// `gridT=1` and a single frame replicated `temporalPatchSize` times
    /// in the unfold — identical to the Qwen 2.5-VL reference.
    func encode(image: Tensor, device: Device) -> Tensor {
        return runForward(framePixels: [image.toFloatArray()], gridT: 1,
                          device: device)
    }

    /// Run the full vision forward on `frames` preprocessed video frames
    /// (each a normalized NCHW tensor `[1, inChannels, side, side]`).
    /// Frames are padded to a multiple of `temporal_patch_size` by
    /// repeating the last frame. Returns
    /// `[mergedTokenCount(frameCount:), textHidden]`.
    ///
    /// Precondition: every frame must share `[1, inChannels, side, side]`
    /// with `side = gridSide · patchSize`.
    func encode(frames: [Tensor], device: Device) -> Tensor {
        precondition(!frames.isEmpty,
                     "Qwen3VL.encode(frames:): expected at least one frame")
        let side = gridSide * cfg.patchSize
        for (i, frame) in frames.enumerated() {
            precondition(
                frame.shape == [1, cfg.inChannels, side, side],
                "Qwen3VL.encode(frames:): frame[\(i)] shape \(frame.shape) "
                + "!= [1,\(cfg.inChannels),\(side),\(side)]")
        }

        // Pad frame count up to a multiple of `temporal_patch_size` by
        // repeating the last frame — same as mlx-vlm's `patchify`.
        let tp = cfg.temporalPatchSize
        let mod = frames.count % tp
        var framePixels: [[Float]] = frames.map { $0.toFloatArray() }
        if mod != 0 {
            let pad = tp - mod
            if let last = framePixels.last {
                for _ in 0..<pad { framePixels.append(last) }
            }
        }
        let gridT = framePixels.count / tp
        return runForward(framePixels: framePixels, gridT: gridT, device: device)
    }

    /// Shared forward path for image / video. `framePixels` has either
    /// 1 element (image, replicated `tp` times in the unfold) or
    /// `gridT * tp` elements (video, one per real frame in display order).
    /// `gridT` is the number of temporal patches the unfold will produce.
    ///
    /// Note on the learned position embedding: Qwen3-VL's bilinearly-
    /// interpolated `[numPositionEmbeddings, hidden]` table is a 2D
    /// spatial embedding. For video, the same spatial embedding is tiled
    /// per temporal group — there is no temporal axis in the table. This
    /// matches the reference mlx-vlm implementation where position ids
    /// cover only `(row, col)` and are replicated per frame. A true
    /// temporal position-embedding dimension would require extending the
    /// table, which is out of scope for this coherence-first port.
    private func runForward(framePixels: [[Float]], gridT: Int,
                            device: Device) -> Tensor {
        let side = gridSide
        let nPatches = gridT * side * side

        // ── Patch unfold + embed ──
        let unfolded = unfoldPatches(framePixels: framePixels, gridT: gridT)
        let cmd = device.makeCommandBuffer()
        var h = Ops.gemm(weight: patchEmbedWeight, input: unfolded,
                         nRows: nPatches, on: cmd)
        if let bias = patchEmbedBias {
            h = addRowBias(h, bias: bias, nRows: nPatches,
                           rowSize: cfg.hidden, on: cmd)
        }
        // ── Learned position embedding (tiled per temporal group) ──
        // The spatial embedding is repeated for each of the `gridT`
        // temporal groups — the per-frame patch count is `side * side`.
        let posEmb = mergeReorderedPositionEmbedding(gridT: gridT)
        h = Ops.add(h, posEmb, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── M-RoPE tables ──
        let (cosTable, sinTable) = ropeTables(gridT: gridT)

        // ── Block stack (full attention on every block) ──
        for block in blocks {
            h = block.forward(h, nTokens: nPatches,
                              cosTable: cosTable, sinTable: sinTable,
                              device: device)
        }

        // ── Patch merger ──
        return mergePatches(h, nPatches: nPatches, device: device)
    }

    /// Present the tower as a `VisionEncoder` so `VisionModel` accepts it.
    func asVisionEncoder() -> VisionEncoder {
        Qwen3VLComposedEncoder(tower: self)
    }

    // ── Patch unfold ──

    /// Unfold normalized NCHW frames into a `[nPatches, patchDimPadded]`
    /// tensor. `framePixels` is a list of length 1 (image, replicated
    /// `tp` times along the temporal axis) or `gridT · tp` (video, one
    /// entry per real frame in display order). Each row holds one
    /// `C · tPatch · patch · patch` patch in merge-block raster order.
    ///
    /// Row order is `(tGroup, mergeBlockRow, mergeBlockCol, intraRow,
    /// intraCol)` — temporal patches are blocked together so the merger's
    /// reshape lines up with one temporal patch group of spatial
    /// neighbourhoods.
    private func unfoldPatches(framePixels: [[Float]], gridT: Int) -> Tensor {
        let side = gridSide
        let p = cfg.patchSize
        let c = cfg.inChannels
        let tp = cfg.temporalPatchSize
        let m = cfg.spatialMergeSize
        let mergeBlocks = side / m
        let imgSide = side * p
        let nPatches = gridT * side * side
        // Image fast path: a single frame replicated `tp` times.
        let isImage = framePixels.count == 1

        var rows = [Float](repeating: 0, count: nPatches * patchDimPadded)

        // Outer loop: temporal groups, then merge-block raster.
        var patch = 0
        for tGroup in 0..<gridT {
            for br in 0..<mergeBlocks {
                for bc in 0..<mergeBlocks {
                    for ir in 0..<m {
                        for ic in 0..<m {
                            let pr = br * m + ir
                            let pc = bc * m + ic
                            var col = 0
                            // Layout: (temporal, channel, py, px).
                            for tWithin in 0..<tp {
                                let frameIdx = isImage ? 0 : tGroup * tp + tWithin
                                let pix = framePixels[frameIdx]
                                for ch in 0..<c {
                                    for py in 0..<p {
                                        let yy = pr * p + py
                                        for px in 0..<p {
                                            let xx = pc * p + px
                                            let v = pix[(ch * imgSide + yy) * imgSide + xx]
                                            rows[patch * patchDimPadded + col] = v
                                            col += 1
                                        }
                                    }
                                }
                            }
                            patch += 1
                        }
                    }
                }
            }
        }
        return ImagePreprocessing.makeTensor(
            from: rows, shape: [nPatches, patchDimPadded],
            dtype: dtype, device: .shared)
    }

    // ── Position embedding ──

    /// Build the per-patch learned position embedding `[nPatches, hidden]`.
    /// The learned table covers a `posSide × posSide` 2D grid; for video
    /// the same spatial embedding is tiled `gridT` times (one copy per
    /// temporal group). When the test grid equals `posSide` the lookup is
    /// a direct gather; otherwise nearest-neighbour (coherence-first).
    ///
    /// Patches are emitted in `(tGroup, merge-block raster)` order to
    /// match `unfoldPatches`.
    private func mergeReorderedPositionEmbedding(gridT: Int) -> Tensor {
        let side = gridSide
        let m = cfg.spatialMergeSize
        let mergeBlocks = side / m
        let hidden = cfg.hidden
        let posSide = Int(Double(cfg.numPositionEmbeddings).squareRoot())
        let table = posEmbedTable.toFloatArray()
        let perFrame = side * side
        let nPatches = gridT * perFrame
        var dst = [Float](repeating: 0, count: nPatches * hidden)

        // Build one frame's embedding in merge-block raster order, then
        // tile it `gridT` times.
        var frameDst = [Float](repeating: 0, count: perFrame * hidden)
        var idx = 0
        for br in 0..<mergeBlocks {
            for bc in 0..<mergeBlocks {
                for ir in 0..<m {
                    for ic in 0..<m {
                        let pr = br * m + ir
                        let pc = bc * m + ic
                        // Nearest grid cell in the learned table (a direct
                        // gather when side == posSide; nearest-neighbour
                        // otherwise — coherence-first).
                        let ty = posSide > 1
                            ? min(posSide - 1, pr * posSide / max(1, side)) : 0
                        let tx = posSide > 1
                            ? min(posSide - 1, pc * posSide / max(1, side)) : 0
                        let src = (ty * posSide + tx) * hidden
                        for d in 0..<hidden {
                            frameDst[idx * hidden + d] = table[src + d]
                        }
                        idx += 1
                    }
                }
            }
        }
        // Tile the per-frame embedding across all temporal groups.
        for t in 0..<gridT {
            let dstOff = t * perFrame * hidden
            for i in 0..<(perFrame * hidden) {
                dst[dstOff + i] = frameDst[i]
            }
        }
        return ImagePreprocessing.makeTensor(
            from: dst, shape: [nPatches, hidden], dtype: dtype,
            device: .shared)
    }

    // ── M-RoPE ──

    /// Build the per-token rotary `cos` / `sin` tables `[nPatches,
    /// headDim]`. The vision M-RoPE interleaves a height-rotary and a
    /// width-rotary half over `headDim`, each driven by the patch's
    /// `(h, w)` grid coordinate; positions are in merge-block order.
    ///
    /// For video (`gridT > 1`), the `(h, w)` pattern is tiled identically
    /// per temporal patch — the temporal axis is carried by the text
    /// decoder's M-RoPE 3-section split, not by the vision tower's rotary
    /// table. This matches the Qwen 2.5-VL reference implementation.
    private func ropeTables(gridT: Int) -> (cos: [Float], sin: [Float]) {
        let side = gridSide
        let nPatches = gridT * side * side
        let headDim = cfg.headDim
        let half = headDim / 2          // height half | width half
        let quarter = half / 2          // distinct rotary frequencies
        var invFreq = [Float](repeating: 0, count: quarter)
        for i in 0..<quarter {
            invFreq[i] = 1.0 / pow(10_000, Float(2 * i) / Float(half))
        }
        let (hPos, wPos) = mergeReorderedPositions()
        let perFrame = side * side

        var cosT = [Float](repeating: 0, count: nPatches * headDim)
        var sinT = [Float](repeating: 0, count: nPatches * headDim)
        for t in 0..<nPatches {
            // Spatial position is the same per-frame tile for all temporal
            // groups — only (h, w) coordinates drive the vision RoPE.
            let spatial = t % perFrame
            let base = t * headDim
            for i in 0..<quarter {
                let fh = Float(hPos[spatial]) * invFreq[i]
                let fw = Float(wPos[spatial]) * invFreq[i]
                for (off, f) in [(i, fh), (i + quarter, fh)] {
                    cosT[base + off] = cos(f); sinT[base + off] = sin(f)
                }
                for (off, f) in [(half + i, fw), (half + i + quarter, fw)] {
                    cosT[base + off] = cos(f); sinT[base + off] = sin(f)
                }
            }
        }
        return (cosT, sinT)
    }

    /// Per-patch `(h, w)` grid coordinates in merge-block raster order
    /// (matches `unfoldPatches` row order within one temporal group).
    private func mergeReorderedPositions() -> (h: [Int], w: [Int]) {
        let side = gridSide
        let m = cfg.spatialMergeSize
        let mergeBlocks = side / m
        var hPos = [Int](repeating: 0, count: side * side)
        var wPos = [Int](repeating: 0, count: side * side)
        var idx = 0
        for br in 0..<mergeBlocks {
            for bc in 0..<mergeBlocks {
                for ir in 0..<m {
                    for ic in 0..<m {
                        hPos[idx] = br * m + ir
                        wPos[idx] = bc * m + ic
                        idx += 1
                    }
                }
            }
        }
        return (hPos, wPos)
    }

    // ── Patch merger ──

    /// Pool each `mergeSize²` neighbourhood of post-encoder tokens into
    /// one token, then project into the text hidden dim. Tokens are in
    /// `(tGroup, merge-block raster)` order, so each consecutive run of
    /// `mergeUnit` tokens is one neighbourhood. Returns
    /// `[nPatches/mergeUnit, textHidden]`.
    private func mergePatches(_ h: Tensor, nPatches: Int, device: Device) -> Tensor {
        let hidden = cfg.hidden
        let mergeUnit = cfg.mergeUnit
        let merged = nPatches / mergeUnit

        // LayerNorm each token (pre-shuffle norm), then group `mergeUnit`
        // tokens into one row.
        let cmd = device.makeCommandBuffer()
        let normed = Ops.layerNorm(h, weight: mergerNorm.weight,
                                   bias: mergerNorm.bias, eps: mergerNorm.eps,
                                   nRows: nPatches, rowSize: hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let grouped = normed.reshaped(to: [merged, mergeUnit * hidden])

        let cmd2 = device.makeCommandBuffer()
        var x = Ops.gemm(weight: mergerFC1.weight, input: grouped,
                         nRows: merged, on: cmd2)
        if let b = mergerFC1.bias {
            x = addRowBias(
                x, bias: b, nRows: merged,
                rowSize: mergerFC1.weight.shape[0], on: cmd2)
        }
        x = Ops.gelu(x, on: cmd2)
        var y = Ops.gemm(weight: mergerFC2.weight, input: x,
                         nRows: merged, on: cmd2)
        if let b = mergerFC2.bias {
            y = addRowBias(
                y, bias: b, nRows: merged,
                rowSize: mergerFC2.weight.shape[0], on: cmd2)
        }
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return y
    }
}

/// A `VisionEncoder` subclass whose `encode` runs the Qwen 3-VL vision
/// tower — so `VisionModel` (which holds a `VisionEncoder`) transparently
/// gets the merged, projected vision tokens.
final class Qwen3VLComposedEncoder: VisionEncoder {
    let tower: Qwen3VLVisionModel

    init(tower: Qwen3VLVisionModel) {
        self.tower = tower
        let c = tower.cfg
        let side = tower.gridSide * c.patchSize
        let facadeConfig = VisionEncoderConfig(
            inChannels: c.inChannels, imageSize: side,
            patchSize: side / Int(Double(tower.mergedTokenCount).squareRoot()),
            hidden: c.hidden, intermediate: c.intermediate,
            nLayers: c.depth, nHeads: c.numHeads,
            layerNormEps: c.layerNormEps, textHidden: tower.textHidden)
        let placeholderW = tower.patchEmbedWeight
        super.init(
            config: facadeConfig,
            patchEmbedWeight: placeholderW, patchEmbedBias: placeholderW,
            positionEmbedding: placeholderW, layers: [],
            postLayerNorm: tower.mergerNorm,
            projection: nil, dtype: tower.dtype)
    }

    /// Run the Qwen 3-VL vision tower on a single preprocessed image.
    /// Returns `[mergedTokenCount, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        tower.encode(image: image, device: device)
    }

    /// Run the Qwen 3-VL vision tower on a sequence of preprocessed
    /// video frames. Returns
    /// `[(T/temporalPatchSize) · mergedTokenCount, textHidden]`.
    override func encode(frames: [Tensor], device: Device = .shared) throws -> Tensor {
        tower.encode(frames: frames, device: device)
    }
}
