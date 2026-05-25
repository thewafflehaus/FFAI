// Qwen 2-VL — Alibaba's Qwen2-VL vision-language model (the
// `Qwen2VLForConditionalGeneration` checkpoints).
//
// Composition:
//   • Qwen 2-VL vision tower — a dynamic-resolution ViT that is the
//     predecessor to Qwen 2.5-VL and Qwen 3-VL, with:
//       – patch-embed via a flattened Conv3d (each input patch is a
//         `in_ch · temporalPatch · patch · patch` row, projected by a
//         single GEMM),
//       – LayerNorm (not RMSNorm) pre-norms,
//       – 2D rotary position embedding (M-RoPE) over the patch grid,
//       – full bidirectional attention on every block (no windowing),
//       – a GELU-MLP feed-forward (`mlp.fc1` / `mlp.fc2`), not the
//         SwiGLU used in Qwen 2.5-VL,
//       – a patch-merger that pools each `mergeSize × mergeSize`
//         neighbourhood and projects into the text hidden dim.
//   • Qwen 2 text backbone — the existing LlamaDense engine (Qwen 2.x
//     routes through LlamaDense), loaded from the top-level config with
//     the `language_model.`-prefixed weight sub-tree.
//
// The two are joined by `VLModel`'s cross-modal token splice: each
// `<|image_pad|>` placeholder (`image_token_id`) in the prompt takes one
// of the merged vision tokens.
//
// Key differences from Qwen 2.5-VL:
//   – LayerNorm (with bias) not RMSNorm in the vision blocks.
//   – GELU MLP (`fc1`/`fc2`) not SwiGLU (`gate_proj`/`up_proj`/`down_proj`).
//   – No windowed attention schedule — all blocks use full attention.
//   – No learned position embedding table — pure M-RoPE.
//   – vision_config uses `embed_dim` (not `hidden_size`) and `mlp_ratio`
//     (not `intermediate_size`).
//   – Merger norm key is `merger.ln_q` (LayerNorm with bias, not RMSNorm).
//   – Merger MLP keys are `merger.mlp.0` / `merger.mlp.2`.
//
// Coherence-first port: vision attention + M-RoPE run on the CPU.
// The text M-RoPE is approximated by VLModel's sequential scalar positions;
// the splice itself is exact.

import Foundation
import Metal

public enum Qwen2VLError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "Qwen2VL: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "Qwen2VL: checkpoint is missing tensor '\(name)'"
        }
    }
}

public enum Qwen2VL {
    /// `image_token_id` default for Qwen 2-VL checkpoints.
    public static let defaultImageTokenId = 151_655
    /// `video_token_id` default for Qwen 2-VL checkpoints
    /// (`<|video_pad|>` — same id as Qwen 2.5-VL).
    public static let defaultVideoTokenId = 151_656

    /// Capabilities a Qwen 2-VL checkpoint declares to the loader.
    /// Text + image + video — the vision tower's Conv3d patch embed and
    /// temporal-patch unfold handle both single-image and multi-frame
    /// video paths.
    public static let availableCapabilities: Set<Capability> =
        Capability.textOnly.union([.visionIn, .videoIn])

    /// Build a `VLModel` from a `Qwen2VLForConditionalGeneration`
    /// checkpoint: the dynamic-resolution vision tower + the Qwen 2
    /// text backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VLModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Qwen2VLError.missingConfig
        }

        // ── Text backbone ──
        // Qwen2-VL stores text hyper-parameters at the top level (same
        // layout as a standalone Qwen 2 checkpoint). Text weights are
        // under `language_model.*`; the LlamaDense engine handles the
        // Qwen 2.x architecture (tied embeddings, SiLU MLP, GQA).
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try LlamaDense.loadModel(
            config: config, weights: textWeights, options: options, device: device)

        // ── Vision tower ──
        // Vision weights are under `vision_tower.*`.
        let visionWeights = weights.prefixed("vision_tower.")
        let vision = try Qwen2VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: visionWeights, dtype: textEngine.dtype, device: device)

        let imageTokenId = config.int("image_token_id") ?? defaultImageTokenId
        let videoTokenId = config.int("video_token_id") ?? defaultVideoTokenId
        return try VLModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            normalization: .clip,
            imageTokenCount: vision.mergedTokenCount)
    }
}

// ─── Qwen 2-VL vision tower ──────────────────────────────────────────

/// Static shape of the Qwen 2-VL vision tower, decoded from
/// `vision_config`.
struct Qwen2VLVisionConfig {
    let depth: Int
    /// Vision hidden dim (from `embed_dim` in the config).
    let hidden: Int
    /// MLP intermediate dim (`mlp_ratio × hidden`).
    let intermediate: Int
    /// Text projection output dim (`hidden_size` in the vision_config).
    let outHidden: Int
    let numHeads: Int
    let patchSize: Int
    let spatialMergeSize: Int
    let temporalPatchSize: Int
    let inChannels: Int
    let layerNormEps: Float

    /// Per-head dimension.
    var headDim: Int { hidden / numHeads }
    /// `mergeSize²` — patches pooled into one merged token.
    var mergeUnit: Int { spatialMergeSize * spatialMergeSize }

    static func decode(_ c: ModelConfig) throws -> Qwen2VLVisionConfig {
        // `embed_dim` is the vision hidden (Qwen2-VL naming convention).
        guard let depth = c.int("depth"),
              let hidden = c.int("embed_dim"),
              let numHeads = c.int("num_heads"),
              let patchSize = c.int("patch_size"),
              let mergeSize = c.int("spatial_merge_size")
        else {
            throw Qwen2VLError.missingConfig
        }
        // `mlp_ratio` scales the hidden dim to the MLP intermediate.
        let mlpRatio = c.float("mlp_ratio") ?? 4.0
        let intermediate = Int(Double(hidden) * mlpRatio)
        // `hidden_size` is the text backbone hidden (output of the merger).
        let outHidden = c.int("hidden_size") ?? hidden
        return Qwen2VLVisionConfig(
            depth: depth, hidden: hidden, intermediate: intermediate,
            outHidden: outHidden, numHeads: numHeads, patchSize: patchSize,
            spatialMergeSize: mergeSize,
            temporalPatchSize: c.int("temporal_patch_size") ?? 2,
            inChannels: c.int("in_chans") ?? c.int("in_channels") ?? 3,
            layerNormEps: Float(c.float("layer_norm_eps") ?? 1e-6))
    }
}

/// One Qwen 2-VL vision block: LayerNorm → full MHA + M-RoPE → residual,
/// LayerNorm → GELU MLP → residual. Held as plain weight tensors; the
/// forward runs CPU attention + GPU GEMMs.
final class Qwen2VLVisionBlock {
    let norm1: LayerNorm
    let norm2: LayerNorm
    let qkv: Linear            // fused [3·hidden, hidden] (+ bias)
    let proj: Linear           // [hidden, hidden] (+ bias)
    let fc1: Linear            // GELU-MLP up   [paddedIntermediate, hidden]
    let fc2: Linear            // GELU-MLP down [hidden, paddedIntermediate]
    let cfg: Qwen2VLVisionConfig
    /// MLP intermediate dim rounded up to the GEMM K-tile width — `fc1`'s
    /// output rows are zero-extended and `fc2`'s input columns zero-padded
    /// to it, so the `fc2` projection's `inDim` is aligned.
    let paddedIntermediate: Int

    init(norm1: LayerNorm, norm2: LayerNorm, qkv: Linear, proj: Linear,
         fc1: Linear, fc2: Linear,
         paddedIntermediate: Int, cfg: Qwen2VLVisionConfig) {
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
                for j in 0..<nTokens {
                    var dot: Float = 0
                    let qVec = qH[head * nTokens + i]
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
        return Qwen25VLVisionModel.addRowBias(y, bias: bias, nRows: nTokens,
                                              rowSize: outDim, on: cmd)
    }
}

/// The Qwen 2-VL vision tower. Holds the patch-embed projection, the block
/// stack, and the patch-merger; `encode` runs the full forward and returns
/// merged tokens in the text hidden dim.
final class Qwen2VLVisionModel: @unchecked Sendable {
    let cfg: Qwen2VLVisionConfig
    /// Flattened patch-embed weight `[hidden, patchDimPadded]`.
    let patchEmbedWeight: Tensor
    /// `patchDim` padded up to a multiple of the GEMM K-tile width.
    let patchDimPadded: Int
    let blocks: [Qwen2VLVisionBlock]
    /// Patch-merger LayerNorm over the pre-merge hidden (key: `merger.ln_q`).
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

    init(cfg: Qwen2VLVisionConfig, patchEmbedWeight: Tensor,
         patchDimPadded: Int,
         blocks: [Qwen2VLVisionBlock], mergerNorm: LayerNorm,
         mergerFC1: Linear, mergerFC2: Linear, textHidden: Int,
         dtype: DType, gridSide: Int) {
        self.cfg = cfg
        self.patchEmbedWeight = patchEmbedWeight
        self.patchDimPadded = patchDimPadded
        self.blocks = blocks
        self.mergerNorm = mergerNorm
        self.mergerFC1 = mergerFC1
        self.mergerFC2 = mergerFC2
        self.textHidden = textHidden
        self.dtype = dtype
        self.gridSide = gridSide
    }

    /// The patch grid side for a square image at the standard Qwen2-VL
    /// test resolution. 28×28 patches at patch-14 → 392×392 px; the grid
    /// divides the merge size cleanly (`28 % 2 == 0`).
    static let defaultGridSide = 28

    static func load(
        visionConfig: ModelConfig, textHidden: Int,
        weights: SafeTensorsBundle, dtype: DType, device: Device
    ) throws -> Qwen2VLVisionModel {
        let cfg = try Qwen2VLVisionConfig.decode(visionConfig)

        // Patch-embed: the checkpoint stores a Conv3d weight
        // `[hidden, in_ch, tPatch, patch, patch]` (PyTorch) or
        // `[hidden, tPatch, patch, patch, in_ch]` (MLX channel-last).
        // Flatten to a 2D GEMM weight `[hidden, patchDim]`.
        let rawPatch = try weights.tensor(named: "patch_embed.proj.weight")
        let patchDim = cfg.inChannels * cfg.temporalPatchSize
            * cfg.patchSize * cfg.patchSize
        let kTile = Qwen25VLVisionModel.gemmKTile
        let patchDimPadded = ((patchDim + kTile - 1) / kTile) * kTile
        let patchEmbedWeight = Qwen25VLVisionModel.flattenPatchEmbed(
            rawPatch, hidden: cfg.hidden, patchDim: patchDim,
            patchDimPadded: patchDimPadded, device: device)

        // Block stack — GELU-MLP intermediate padded to the K-tile width.
        let paddedIntermediate =
            ((cfg.intermediate + kTile - 1) / kTile) * kTile
        var blocks: [Qwen2VLVisionBlock] = []
        blocks.reserveCapacity(cfg.depth)
        for i in 0..<cfg.depth {
            let p = "blocks.\(i)"
            func lin(_ name: String) throws -> Linear {
                let w = try weights.tensor(named: "\(p).\(name).weight")
                let b = try? weights.tensor(named: "\(p).\(name).bias")
                return Linear(weight: w, bias: b)
            }
            // Qwen2-VL blocks use LayerNorm (weight + bias).
            func norm(_ name: String) throws -> LayerNorm {
                LayerNorm(
                    weight: try weights.tensor(named: "\(p).\(name).weight"),
                    bias: try weights.tensor(named: "\(p).\(name).bias"),
                    eps: cfg.layerNormEps)
            }
            // MLP uses `mlp.fc1` / `mlp.fc2` (GELU, not SwiGLU).
            let fc1 = try lin("mlp.fc1")
            let fc2 = try lin("mlp.fc2")
            blocks.append(Qwen2VLVisionBlock(
                norm1: try norm("norm1"), norm2: try norm("norm2"),
                qkv: try lin("attn.qkv"), proj: try lin("attn.proj"),
                fc1: Qwen25VLVisionModel.padLinearRows(
                    fc1, toRows: paddedIntermediate, device: device),
                fc2: Qwen25VLVisionModel.padLinearCols(
                    fc2, toCols: paddedIntermediate, device: device),
                paddedIntermediate: paddedIntermediate, cfg: cfg))
        }

        // Patch-merger: LayerNorm under `merger.ln_q` (with bias), then
        // a 2-layer MLP indexed as `merger.mlp.0` / `merger.mlp.2`.
        let mergerNorm = LayerNorm(
            weight: try weights.tensor(named: "merger.ln_q.weight"),
            bias: try weights.tensor(named: "merger.ln_q.bias"),
            eps: cfg.layerNormEps)
        func mergerLin(_ idx: Int) throws -> Linear {
            let w = try weights.tensor(named: "merger.mlp.\(idx).weight")
            let b = try? weights.tensor(named: "merger.mlp.\(idx).bias")
            return Linear(weight: w, bias: b)
        }
        let mergerFC1 = try mergerLin(0)
        let mergerFC2 = try mergerLin(2)

        return Qwen2VLVisionModel(
            cfg: cfg, patchEmbedWeight: patchEmbedWeight,
            patchDimPadded: patchDimPadded, blocks: blocks,
            mergerNorm: mergerNorm, mergerFC1: mergerFC1, mergerFC2: mergerFC2,
            textHidden: textHidden, dtype: dtype, gridSide: defaultGridSide)
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
                     "Qwen2VL.encode(frames:): expected at least one frame")
        let side = gridSide * cfg.patchSize
        for (i, frame) in frames.enumerated() {
            precondition(
                frame.shape == [1, cfg.inChannels, side, side],
                "Qwen2VL.encode(frames:): frame[\(i)] shape \(frame.shape) "
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
    private func runForward(framePixels: [[Float]], gridT: Int,
                            device: Device) -> Tensor {
        let side = gridSide
        let nPatches = gridT * side * side

        // ── Patch unfold + embed ──
        let unfolded = unfoldPatches(framePixels: framePixels, gridT: gridT)
        let cmd = device.makeCommandBuffer()
        let h = Ops.gemm(weight: patchEmbedWeight, input: unfolded,
                         nRows: nPatches, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── M-RoPE tables ──
        let (cosTable, sinTable) = ropeTables(gridT: gridT)

        // ── Block stack (full attention on every block) ──
        var hidden = h
        for block in blocks {
            hidden = block.forward(hidden, nTokens: nPatches,
                                   cosTable: cosTable, sinTable: sinTable,
                                   device: device)
        }

        // ── Patch merger ──
        return mergePatches(hidden, nPatches: nPatches, device: device)
    }

    /// Present the tower as a `VisionEncoder` so `VLModel` accepts it.
    func asVisionEncoder() -> VisionEncoder {
        Qwen2VLComposedEncoder(tower: self)
    }

    // ── Patch unfold ──

    /// Unfold normalized NCHW frames into a `[nPatches, patchDimPadded]`
    /// tensor. `framePixels` is a list of length 1 (image, replicated
    /// `tp` times along the temporal axis) or `gridT · tp` (video, one
    /// entry per real frame in display order). Each row holds one
    /// `C · tPatch · patch · patch` patch in merge-block raster order;
    /// the temporal dimension follows the layout expected by the flattened
    /// patch-embed weight.
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

        // Rows are zero-padded to `patchDimPadded`; the trailing columns
        // pair with the patch-embed weight's zero-pad columns.
        var rows = [Float](repeating: 0, count: nPatches * patchDimPadded)

        // Outer loop: temporal groups, then merge-block raster.
        // This keeps each group of `side*side` spatial patches for one
        // temporal patch contiguous, so the merger reshape is correct
        // for both images (gridT=1) and video (gridT>1).
        var patch = 0
        for tGroup in 0..<gridT {
            // Merge-block raster order within each temporal group.
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

    // ── M-RoPE ──

    /// Build the per-token rotary `cos` / `sin` tables `[nPatches,
    /// headDim]`. Qwen's vision M-RoPE interleaves a height-rotary and a
    /// width-rotary half over `headDim`, each driven by the patch's
    /// `(h, w)` grid coordinate. Positions are in merge-block raster order.
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
        // Inverse frequencies with theta 10000 (VisionRotaryEmbedding).
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
            // Height rotary fills [0, half); width rotary fills [half, 2·half).
            for i in 0..<quarter {
                let fh = Float(hPos[spatial]) * invFreq[i]
                let fw = Float(wPos[spatial]) * invFreq[i]
                // Frequencies tiled to half (duplicate for rotate-half).
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

        // LayerNorm each token, then group `mergeUnit` tokens into one row.
        let cmd = device.makeCommandBuffer()
        let normed = Ops.layerNorm(h, weight: mergerNorm.weight,
                                   bias: mergerNorm.bias, eps: mergerNorm.eps,
                                   nRows: nPatches, rowSize: hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // The merger MLP input dim is `mergeUnit · hidden`; tokens are
        // already contiguous per neighbourhood, so a plain reshape suffices.
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

/// A `VisionEncoder` subclass whose `encode` runs the Qwen 2-VL vision
/// tower — so `VLModel` (which holds a `VisionEncoder`) transparently
/// gets the merged, projected vision tokens.
final class Qwen2VLComposedEncoder: VisionEncoder {
    let tower: Qwen2VLVisionModel

    init(tower: Qwen2VLVisionModel) {
        self.tower = tower
        let c = tower.cfg
        let side = tower.gridSide * c.patchSize
        let facadeConfig = VisionEncoderConfig(
            inChannels: c.inChannels, imageSize: side,
            patchSize: side / Int(Double(tower.mergedTokenCount).squareRoot()),
            hidden: c.hidden, intermediate: c.intermediate,
            nLayers: c.depth, nHeads: c.numHeads,
            layerNormEps: c.layerNormEps, textHidden: tower.textHidden)
        // The patch-embed / position / layers are unused by the override
        // below; pass minimal placeholders.
        let placeholderW = tower.patchEmbedWeight
        super.init(
            config: facadeConfig,
            patchEmbedWeight: placeholderW, patchEmbedBias: placeholderW,
            positionEmbedding: placeholderW, layers: [],
            postLayerNorm: tower.mergerNorm,
            projection: nil, dtype: tower.dtype)
    }

    /// Run the Qwen 2-VL vision tower on a single preprocessed image.
    /// Returns `[mergedTokenCount, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        tower.encode(image: image, device: device)
    }

    /// Run the Qwen 2-VL vision tower on a sequence of preprocessed
    /// video frames. Returns
    /// `[(T/temporalPatchSize) · mergedTokenCount, textHidden]`.
    override func encode(frames: [Tensor], device: Device = .shared) throws -> Tensor {
        tower.encode(frames: frames, device: device)
    }
}
