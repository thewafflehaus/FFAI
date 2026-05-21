// Qwen 3-VL — Alibaba's Qwen3-VL vision-language model (the
// `Qwen3VLForConditionalGeneration` checkpoints).
//
// Composition:
//   • Qwen 3-VL vision tower — a dynamic-resolution ViT that, unlike the
//     Qwen 2.5-VL tower, uses:
//       – patch-embed via a flattened Conv3d (each input patch is a
//         `in_ch · temporalPatch · patch · patch` row, projected by a
//         single GEMM),
//       – LayerNorm (not RMSNorm) pre-norms,
//       – a learned, bilinearly-interpolated position embedding looked
//         up from a `[numPositionEmbeddings, hidden]` table,
//       – 2D rotary position embedding (M-RoPE) over the patch grid,
//       – full bidirectional attention on every block (no windowing —
//         Qwen3-VL dropped the windowed-attention schedule),
//       – a GELU-MLP feed-forward (`linear_fc1` / `linear_fc2`), not the
//         Qwen 2.5-VL SwiGLU,
//       – a patch-merger that pools each `mergeSize × mergeSize`
//         neighbourhood and projects into the text hidden dim.
//   • Qwen 3 text backbone — the existing `Qwen3Model` dense engine,
//     loaded from the `language_model.`-prefixed sub-tree (Qwen3-VL
//     stores text weights under `language_model.model.*` /
//     `language_model.lm_head.*`).
//
// The two are joined by `VLModel`'s cross-modal token splice: each
// `<|image_pad|>` placeholder (`image_token_id`) in the prompt takes one
// of the merged vision tokens.
//
// Coherence-first port: the vision tower's attention + M-RoPE run on the
// CPU (vision token counts are at most a few thousand, so an O(n²·d)
// attention is cheap next to the GPU projection GEMMs and is
// unambiguously correct). The text M-RoPE — Qwen's 3D position scheme —
// is approximated by `VLModel`'s sequential scalar positions; the splice
// itself is exact. The Qwen3-VL `deepstack` feature (injecting
// intermediate vision features into the text stack) is omitted in this
// coherence-first port — only the final merged tokens are spliced. A
// head-dim-agnostic GPU vision SDPA, true text M-RoPE, and deepstack are
// later performance / fidelity passes.

import Foundation
import Metal

public enum Qwen3VLError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "Qwen3VL: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "Qwen3VL: checkpoint is missing tensor '\(name)'"
        }
    }
}

public enum Qwen3VL {
    /// `image_token_id` default for Qwen 3-VL checkpoints.
    public static let defaultImageTokenId = 151_655

    /// Build a `VLModel` from a `Qwen3VLForConditionalGeneration`
    /// checkpoint: the dynamic-resolution vision tower + the Qwen 3
    /// text backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VLModel {
        guard let visionConfig = config.subConfig("vision_config"),
              let textConfigRaw = config.nested("text_config")
        else {
            throw Qwen3VLError.missingConfig
        }

        // ── Text backbone — Qwen 3 dense engine ──
        // Qwen3-VL stores text hyper-parameters under `text_config`; the
        // standalone `Qwen3Dense` loader reads top-level config keys, so
        // re-wrap the `text_config` sub-tree as a flat `ModelConfig`.
        let textConfig = ModelConfig(
            architecture: "Qwen3ForCausalLM",
            modelType: "qwen3",
            raw: textConfigRaw)
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try Qwen3Dense.loadModel(
            config: textConfig, weights: textWeights,
            options: options, device: device)

        // ── Vision tower ──
        // Vision weights are under `model.visual.*` on the mlx-community
        // Qwen3-VL conversion; the tower is loaded into a composed
        // `VisionEncoder` facade.
        let visionWeights = weights.prefixed("model.visual.")
        let vision = try Qwen3VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: visionWeights, dtype: textEngine.dtype, device: device)

        let imageTokenId = config.int("image_token_id")
            ?? config.int("image_token_index") ?? defaultImageTokenId
        return try VLModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: .clip,
            imageTokenCount: vision.mergedTokenCount)
    }
}

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

        // Apply M-RoPE to one head's `[headDim]` slice at token `tok`.
        // Qwen's rotate-half scheme: out[d] = x[d]·cos − x[d±half]·sin.
        func rope(_ x: inout [Float], tok: Int) {
            let cb = tok * headDim
            for d in 0..<headDim {
                let rotated = d < half ? -x[d + half] : x[d - half]
                let c = cosTable[cb + d], s = sinTable[cb + d]
                x[d] = x[d] * c + rotated * s
            }
        }

        for head in 0..<nHeads {
            let hOff = head * headDim
            var qH = [[Float]](repeating: [], count: nTokens)
            var kH = [[Float]](repeating: [], count: nTokens)
            var vH = [[Float]](repeating: [], count: nTokens)
            for t in 0..<nTokens {
                let base = t * 3 * hidden
                var q = Array(qkvA[(base + hOff)..<(base + hOff + headDim)])
                var k = Array(qkvA[(base + hidden + hOff)..<(base + hidden + hOff + headDim)])
                let v = Array(qkvA[(base + 2 * hidden + hOff)..<(base + 2 * hidden + hOff + headDim)])
                rope(&q, tok: t)
                rope(&k, tok: t)
                qH[t] = q; kH[t] = k; vH[t] = v
            }
            // Full attention — every token attends to every token.
            for i in 0..<nTokens {
                var scores = [Float](repeating: 0, count: nTokens)
                var maxS = -Float.greatestFiniteMagnitude
                for j in 0..<nTokens {
                    var dot: Float = 0
                    for d in 0..<headDim { dot += qH[i][d] * kH[j][d] }
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
                    for d in 0..<headDim { out[oBase + d] += w * vH[j][d] }
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
        return Qwen25VLVisionModel.addRowBias(y, bias: bias, nRows: nTokens,
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
        let kTile = Qwen25VLVisionModel.gemmKTile
        let patchDimPadded = ((patchDim + kTile - 1) / kTile) * kTile
        let patchEmbedWeight = Qwen25VLVisionModel.flattenPatchEmbed(
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
                fc1: Qwen25VLVisionModel.padLinearRows(
                    fc1, toRows: paddedIntermediate, device: device),
                fc2: Qwen25VLVisionModel.padLinearCols(
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
    func encode(image: Tensor, device: Device) -> Tensor {
        let side = gridSide
        let nPatches = side * side

        // ── Patch unfold + embed ──
        let unfolded = unfoldPatches(image: image)
        let cmd = device.makeCommandBuffer()
        var h = Ops.gemm(weight: patchEmbedWeight, input: unfolded,
                         nRows: nPatches, on: cmd)
        if let bias = patchEmbedBias {
            h = Qwen25VLVisionModel.addRowBias(h, bias: bias, nRows: nPatches,
                                               rowSize: cfg.hidden, on: cmd)
        }
        // ── Learned position embedding (direct gather in merge order) ──
        let posEmb = mergeReorderedPositionEmbedding()
        h = Ops.add(h, posEmb, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── M-RoPE tables ──
        let (cosTable, sinTable) = ropeTables()

        // ── Block stack (full attention on every block) ──
        for block in blocks {
            h = block.forward(h, nTokens: nPatches,
                              cosTable: cosTable, sinTable: sinTable,
                              device: device)
        }

        // ── Patch merger ──
        return mergePatches(h, device: device)
    }

    /// Present the tower as a `VisionEncoder` so `VLModel` accepts it.
    func asVisionEncoder() -> VisionEncoder {
        Qwen3VLComposedEncoder(tower: self)
    }

    // ── Patch unfold ──

    /// Unfold a normalized NCHW image into a `[nPatches, patchDimPadded]`
    /// tensor — each row one `C · tPatch · patch · patch` patch in
    /// merge-block raster order, the temporal axis a single frame
    /// repeated `temporalPatchSize` times.
    private func unfoldPatches(image: Tensor) -> Tensor {
        let side = gridSide
        let p = cfg.patchSize
        let c = cfg.inChannels
        let tp = cfg.temporalPatchSize
        let m = cfg.spatialMergeSize
        let blocks = side / m
        let pix = image.toFloatArray()        // [C, H, W]
        let imgSide = side * p
        var rows = [Float](repeating: 0, count: side * side * patchDimPadded)
        var patch = 0
        // Merge-block raster order: (blockRow, blockCol, intraRow, intraCol).
        for br in 0..<blocks {
            for bc in 0..<blocks {
                for ir in 0..<m {
                    for ic in 0..<m {
                        let pr = br * m + ir
                        let pc = bc * m + ic
                        var col = 0
                        for _ in 0..<tp {
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
        return ImagePreprocessing.makeTensor(
            from: rows, shape: [side * side, patchDimPadded],
            dtype: dtype, device: .shared)
    }

    // ── Position embedding ──

    /// Build the per-patch learned position embedding `[nPatches,
    /// hidden]`. The learned table covers a `posSide × posSide` grid;
    /// when the test grid equals `posSide` the lookup is a direct gather.
    /// Patches are emitted in merge-block raster order to match
    /// `unfoldPatches`.
    private func mergeReorderedPositionEmbedding() -> Tensor {
        let side = gridSide
        let m = cfg.spatialMergeSize
        let blocks = side / m
        let hidden = cfg.hidden
        let posSide = Int(Double(cfg.numPositionEmbeddings).squareRoot())
        let table = posEmbedTable.toFloatArray()
        var dst = [Float](repeating: 0, count: side * side * hidden)
        var patch = 0
        for br in 0..<blocks {
            for bc in 0..<blocks {
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
                            dst[patch * hidden + d] = table[src + d]
                        }
                        patch += 1
                    }
                }
            }
        }
        return ImagePreprocessing.makeTensor(
            from: dst, shape: [side * side, hidden], dtype: dtype,
            device: .shared)
    }

    // ── M-RoPE ──

    /// Build the per-token rotary `cos` / `sin` tables `[nPatches,
    /// headDim]`. The vision M-RoPE interleaves a height-rotary and a
    /// width-rotary half over `headDim`, each driven by the patch's
    /// `(h, w)` grid coordinate; positions are in merge-block order.
    private func ropeTables() -> (cos: [Float], sin: [Float]) {
        let side = gridSide
        let nPatches = side * side
        let headDim = cfg.headDim
        let half = headDim / 2          // height half | width half
        let quarter = half / 2          // distinct rotary frequencies
        var invFreq = [Float](repeating: 0, count: quarter)
        for i in 0..<quarter {
            invFreq[i] = 1.0 / pow(10_000, Float(2 * i) / Float(half))
        }
        let (hPos, wPos) = mergeReorderedPositions()

        var cosT = [Float](repeating: 0, count: nPatches * headDim)
        var sinT = [Float](repeating: 0, count: nPatches * headDim)
        for t in 0..<nPatches {
            let base = t * headDim
            for i in 0..<quarter {
                let fh = Float(hPos[t]) * invFreq[i]
                let fw = Float(wPos[t]) * invFreq[i]
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

    /// Per-patch `(h, w)` grid coordinates in merge-block raster order.
    private func mergeReorderedPositions() -> (h: [Int], w: [Int]) {
        let side = gridSide
        let m = cfg.spatialMergeSize
        let blocks = side / m
        var hPos = [Int](repeating: 0, count: side * side)
        var wPos = [Int](repeating: 0, count: side * side)
        var idx = 0
        for br in 0..<blocks {
            for bc in 0..<blocks {
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
    /// merge-block raster order, so each consecutive run of `mergeUnit`
    /// tokens is one neighbourhood. Returns `[mergedTokenCount,
    /// textHidden]`.
    private func mergePatches(_ h: Tensor, device: Device) -> Tensor {
        let nPatches = gridSide * gridSide
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
            x = Qwen25VLVisionModel.addRowBias(
                x, bias: b, nRows: merged,
                rowSize: mergerFC1.weight.shape[0], on: cmd2)
        }
        x = Ops.gelu(x, on: cmd2)
        var y = Ops.gemm(weight: mergerFC2.weight, input: x,
                         nRows: merged, on: cmd2)
        if let b = mergerFC2.bias {
            y = Qwen25VLVisionModel.addRowBias(
                y, bias: b, nRows: merged,
                rowSize: mergerFC2.weight.shape[0], on: cmd2)
        }
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return y
    }
}

/// A `VisionEncoder` subclass whose `encode` runs the Qwen 3-VL vision
/// tower — so `VLModel` (which holds a `VisionEncoder`) transparently
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

    /// Run the Qwen 3-VL vision tower. Returns
    /// `[mergedTokenCount, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        tower.encode(image: image, device: device)
    }
}
