// Qwen 2.5-VL — Alibaba's Qwen2.5-VL vision-language model (the
// `Qwen2_5_VLForConditionalGeneration` checkpoints).
//
// Composition:
//   • Qwen 2.5-VL vision tower — a dynamic-resolution ViT that differs
//     substantially from the shared SigLIP `VisionEncoder`:
//       – patch-embed via a flattened conv (each input patch is a
//         `in_ch · temporalPatch · patch · patch` row, projected by a
//         single GEMM),
//       – RMSNorm (not LayerNorm) pre-norms,
//       – 2D rotary position embedding (M-RoPE) over the patch grid,
//       – windowed attention on most blocks, full attention on a few
//         (`fullatt_block_indexes`),
//       – a patch-merger that pools each `mergeSize × mergeSize`
//         neighbourhood and projects into the text hidden dim.
//   • Qwen 2 / 2.5 text backbone — the existing Llama dense engine
//     (Qwen 2.x routes through `LlamaDense`), loaded from the
//     checkpoint's top-level (Qwen2.5-VL stores text weights under
//     `model.*` / `lm_head.*`, the standard text layout).
//
// The two are joined by `VLModel`'s cross-modal token splice: each
// `<|image_pad|>` placeholder (`image_token_id`) in the prompt takes one
// of the merged vision tokens.
//
// Coherence-first port: the vision tower's windowed attention + M-RoPE
// run on the CPU (vision token counts are at most a few thousand, so an
// O(n²·d) attention is cheap next to the GPU projection GEMMs and is
// unambiguously correct). The text M-RoPE — Qwen's 3D position scheme —
// is approximated by `VLModel`'s sequential scalar positions; the splice
// itself is exact. A head-dim-agnostic GPU vision SDPA + true text
// M-RoPE are later performance / fidelity passes.

import Foundation
import Metal

public enum Qwen25VLError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "Qwen25VL: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "Qwen25VL: checkpoint is missing tensor '\(name)'"
        }
    }
}

public enum Qwen25VL {
    /// `image_token_id` default for Qwen 2.5-VL checkpoints.
    public static let defaultImageTokenId = 151_655

    /// Build a `VLModel` from a `Qwen2_5_VLForConditionalGeneration`
    /// checkpoint: the dynamic-resolution vision tower + the Qwen 2.x
    /// text backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VLModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Qwen25VLError.missingConfig
        }

        // ── Text backbone ──
        // The mlx-community Qwen 2.5-VL conversion namespaces its text
        // weights under `language_model.model.*` / `language_model.norm`;
        // the text hyper-parameters live at the config top level. Route
        // the `language_model.`-prefixed sub-tree through the Llama dense
        // engine (Qwen 2.x == LlamaDense).
        let textWeights = weights.prefixed("language_model.")
        let textEngine = try LlamaDense.loadModel(
            config: config, weights: textWeights, options: options, device: device)

        // ── Vision tower ──
        // Vision weights are under `vision_tower.*` and (per the
        // checkpoint's `skip_vision` quantization flag) are not
        // quantized — plain f16 / bf16 tensors.
        let visionWeights = weights.prefixed("vision_tower.")
        let vision = try Qwen25VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: visionWeights, dtype: textEngine.dtype, device: device)

        let imageTokenId = config.int("image_token_id") ?? defaultImageTokenId
        // The vision tower decides its own merged-token count from the
        // dynamic image geometry; the encoder facade reports it.
        return try VLModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: .clip,
            imageTokenCount: vision.mergedTokenCount)
    }
}

// ─── Qwen 2.5-VL vision tower ────────────────────────────────────────

/// Static shape of the Qwen 2.5-VL vision tower, decoded from
/// `vision_config`.
struct Qwen25VLVisionConfig {
    let depth: Int
    let hidden: Int
    let intermediate: Int
    let outHidden: Int
    let numHeads: Int
    let patchSize: Int
    let spatialMergeSize: Int
    let temporalPatchSize: Int
    let windowSize: Int
    let fullattBlockIndexes: Set<Int>
    let inChannels: Int
    let rmsNormEps: Float

    /// Per-head dimension.
    var headDim: Int { hidden / numHeads }
    /// `mergeSize²` — patches pooled into one merged token.
    var mergeUnit: Int { spatialMergeSize * spatialMergeSize }

    static func decode(_ c: ModelConfig) throws -> Qwen25VLVisionConfig {
        guard let depth = c.int("depth"),
              let hidden = c.int("hidden_size"),
              let numHeads = c.int("num_heads"),
              let patchSize = c.int("patch_size"),
              let mergeSize = c.int("spatial_merge_size")
        else {
            throw Qwen25VLError.missingConfig
        }
        // `intermediate_size` / `out_hidden_size` are present on real
        // checkpoints; fall back to documented Qwen2.5-VL defaults.
        let intermediate = c.int("intermediate_size") ?? hidden * 4
        let outHidden = c.int("out_hidden_size") ?? hidden
        return Qwen25VLVisionConfig(
            depth: depth, hidden: hidden, intermediate: intermediate,
            outHidden: outHidden, numHeads: numHeads, patchSize: patchSize,
            spatialMergeSize: mergeSize,
            temporalPatchSize: c.int("temporal_patch_size") ?? 2,
            windowSize: c.int("window_size") ?? 112,
            fullattBlockIndexes: Set(c.intArray("fullatt_block_indexes") ?? []),
            inChannels: c.int("in_chans") ?? c.int("in_channels") ?? 3,
            rmsNormEps: Float(c.float("layer_norm_eps") ?? c.float("rms_norm_eps") ?? 1e-6))
    }
}

/// One Qwen 2.5-VL vision block: RMSNorm → windowed/full MHA + M-RoPE →
/// residual, RMSNorm → SwiGLU MLP → residual. Held as plain weight
/// tensors; the forward runs CPU attention + GPU GEMMs.
final class Qwen25VLVisionBlock {
    let norm1: RMSNorm
    let norm2: RMSNorm
    let qkv: Linear            // fused [3·hidden, hidden] (+ bias)
    let proj: Linear           // [hidden, hidden] (+ bias)
    let gate: Linear           // SwiGLU gate [paddedIntermediate, hidden]
    let up: Linear             // SwiGLU up   [paddedIntermediate, hidden]
    let down: Linear           // SwiGLU down [hidden, paddedIntermediate]
    let cfg: Qwen25VLVisionConfig
    /// SwiGLU intermediate dim rounded up to the GEMM K-tile width — the
    /// `gate`/`up` outputs are zero-extended and `down`'s input columns
    /// zero-padded to it, so the `down` projection's `inDim` is aligned.
    let paddedIntermediate: Int

    init(norm1: RMSNorm, norm2: RMSNorm, qkv: Linear, proj: Linear,
         gate: Linear, up: Linear, down: Linear,
         paddedIntermediate: Int, cfg: Qwen25VLVisionConfig) {
        self.norm1 = norm1; self.norm2 = norm2
        self.qkv = qkv; self.proj = proj
        self.gate = gate; self.up = up; self.down = down
        self.paddedIntermediate = paddedIntermediate
        self.cfg = cfg
    }

    /// Forward `[nTokens, hidden]` activations through one block.
    /// `cosTable` / `sinTable` are the precomputed per-token rotary
    /// tables `[nTokens, headDim]`; `windowGroups` partitions the tokens
    /// into attention windows (one group = full attention over the whole
    /// sequence for full-attention blocks).
    func forward(_ h: Tensor, nTokens: Int,
                 cosTable: [Float], sinTable: [Float],
                 windowGroups: [[Int]], device: Device) -> Tensor {
        let hidden = cfg.hidden
        // ── Attention sub-block ──
        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNormRows(h, weight: norm1.weight, eps: norm1.eps,
                                     nRows: nTokens, rowSize: hidden, on: cmd)
        let qkvOut = projectRows(qkv, normed, nTokens: nTokens,
                                 outDim: 3 * hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let attn = cpuAttention(qkv: qkvOut, nTokens: nTokens,
                                cosTable: cosTable, sinTable: sinTable,
                                windowGroups: windowGroups, device: device)

        // ── Residual + MLP sub-block ──
        let cmd2 = device.makeCommandBuffer()
        let attnProj = projectRows(proj, attn, nTokens: nTokens,
                                   outDim: hidden, on: cmd2)
        let postAttn = Ops.add(h, attnProj, on: cmd2)
        let normed2 = Ops.rmsNormRows(postAttn, weight: norm2.weight,
                                      eps: norm2.eps, nRows: nTokens,
                                      rowSize: hidden, on: cmd2)
        let g = projectRows(gate, normed2, nTokens: nTokens,
                            outDim: paddedIntermediate, on: cmd2)
        let u = projectRows(up, normed2, nTokens: nTokens,
                            outDim: paddedIntermediate, on: cmd2)
        let act = Ops.silu(g, on: cmd2)
        let gated = Ops.mul(act, u, on: cmd2)
        let d = projectRows(down, gated, nTokens: nTokens,
                            outDim: hidden, on: cmd2)
        let result = Ops.add(postAttn, d, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return result
    }

    /// CPU windowed multi-head attention with M-RoPE. `qkv` is the fused
    /// `[nTokens, 3·hidden]` projection. Returns the context token-major
    /// `[nTokens, hidden]`.
    ///
    /// Two-stage parallelization with `DispatchQueue.concurrentPerform`:
    ///   1. Setup — M-RoPE: each `(head, token)` pair writes its own
    ///      `qH[head*nTokens+t]` / `kH[head*nTokens+t]` / `vH[head*nTokens+t]`
    ///      row — disjoint, race-free.
    ///   2. Attention — parallelized over heads: within each head, window
    ///      groups are processed serially (each (head, i) writes to a
    ///      disjoint `[oBase, oBase + headDim)` output slice — race-free
    ///      across heads because different heads use different `hOff`).
    private func cpuAttention(qkv: Tensor, nTokens: Int,
                              cosTable: [Float], sinTable: [Float],
                              windowGroups: [[Int]], device: Device) -> Tensor {
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

        // Stage 2: Attention. The full-attention blocks (caller passes a
        // single group spanning every token — `fullattBlockIndexes`) run
        // through the GPU `Ops.sdpaBidirectional` kernel. The windowed
        // blocks keep the parallel CPU loop because `sdpaBidirectional`
        // is fully bidirectional and the kernel surface has no per-group
        // mask today.
        let isFullAttention = windowGroups.count == 1
            && windowGroups[0].count == nTokens
        if isFullAttention
            && OpsValidation.sdpaBidirectionalSupportedHeadDims.contains(headDim) {
            // GPU path. qH/kH/vH are already laid out as
            // [nHeads, nTokens, headDim] — kernel K/V layout natively.
            // Q needs a transpose into [nTokens, nHeads, headDim].
            var qFlat = [Float](repeating: 0, count: nTokens * nHeads * headDim)
            var kFlat = [Float](repeating: 0, count: nHeads * nTokens * headDim)
            var vFlat = [Float](repeating: 0, count: nHeads * nTokens * headDim)
            for head in 0..<nHeads {
                for t in 0..<nTokens {
                    let qRow = qH[head * nTokens + t]
                    let kRow = kH[head * nTokens + t]
                    let vRow = vH[head * nTokens + t]
                    let qDst = (t * nHeads + head) * headDim
                    let kvDst = (head * nTokens + t) * headDim
                    for d in 0..<headDim {
                        qFlat[qDst + d] = qRow[d]
                        kFlat[kvDst + d] = kRow[d]
                        vFlat[kvDst + d] = vRow[d]
                    }
                }
            }
            let qT = Tensor.empty(shape: [nTokens, nHeads, headDim], dtype: .f32,
                                  device: device)
            ImagePreprocessing.copyFloats(qFlat, into: qT)
            let kT = Tensor.empty(shape: [nHeads, nTokens, headDim], dtype: .f32,
                                  device: device)
            ImagePreprocessing.copyFloats(kFlat, into: kT)
            let vT = Tensor.empty(shape: [nHeads, nTokens, headDim], dtype: .f32,
                                  device: device)
            ImagePreprocessing.copyFloats(vFlat, into: vT)
            let cmd = device.makeCommandBuffer()
            let outT = Ops.sdpaBidirectional(
                q: qT, k: kT, v: vT,
                nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
                baseKV: 0, nQuery: nTokens, kvStride: nTokens,
                scale: scale, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            // outT is [nTokens, nHeads, headDim] — byte-identical to the
            // [nTokens, hidden] layout the patch-merger/proj expects.
            let outFlat = outT.toFloatArray()
            let result = Tensor.empty(shape: [nTokens, hidden], dtype: qkv.dtype,
                                      device: device)
            ImagePreprocessing.copyFloats(outFlat, into: result)
            return result
        }

        // CPU fallback path — windowed attention. Same loop as before,
        // parallelized over heads. Within a head, token `i` writes to
        // `out[i*hidden+hOff .. +headDim)` — disjoint across heads
        // (different `hOff`) and disjoint across tokens within a head
        // (different `i`).
        out.withUnsafeMutableBufferPointer { outBuf in
            let outPtr = outBuf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nHeads) { head in
                let hOff = head * headDim
                for group in windowGroups {
                    for i in group {
                        var scores = [Float](repeating: 0, count: group.count)
                        var maxS = -Float.greatestFiniteMagnitude
                        let qVec = qH[head * nTokens + i]
                        for (gi, j) in group.enumerated() {
                            var dot: Float = 0
                            let kVec = kH[head * nTokens + j]
                            for d in 0..<headDim { dot += qVec[d] * kVec[d] }
                            let s = dot * scale
                            scores[gi] = s
                            if s > maxS { maxS = s }
                        }
                        var sum: Float = 0
                        for gi in 0..<group.count {
                            let e = exp(scores[gi] - maxS)
                            scores[gi] = e; sum += e
                        }
                        let inv = sum > 0 ? 1 / sum : 0
                        let oBase = i * hidden + hOff
                        for (gi, j) in group.enumerated() {
                            let w = scores[gi] * inv
                            let vVec = vH[head * nTokens + j]
                            for d in 0..<headDim { outPtr[oBase + d] += w * vVec[d] }
                        }
                    }
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

/// The Qwen 2.5-VL vision tower. Holds the patch-embed projection, the
/// block stack, and the patch-merger; `encode` runs the full forward and
/// returns merged tokens in the text hidden dim.
final class Qwen25VLVisionModel: @unchecked Sendable {
    let cfg: Qwen25VLVisionConfig
    /// Flattened patch-embed weight `[hidden, patchDimPadded]` — the
    /// unfold `patchDim` rounded up to the GEMM K-tile width (16) with
    /// zero-pad columns, so the patch-embed projection is a single
    /// `Ops.gemm`.
    let patchEmbedWeight: Tensor
    /// `patchDim` padded up to a multiple of the GEMM K-tile width.
    let patchDimPadded: Int
    let blocks: [Qwen25VLVisionBlock]
    /// Patch-merger RMSNorm over the pre-merge hidden.
    let mergerNorm: RMSNorm
    /// Patch-merger MLP: `[mergeUnit·hidden] → [mergeUnit·hidden] → [outHidden]`.
    let mergerFC1: Linear
    let mergerFC2: Linear
    let textHidden: Int
    let dtype: DType
    /// Patch grid side (square test image at the encoder's fixed size).
    let gridSide: Int

    /// Number of merged vision tokens one image contributes — the
    /// `imageTokenCount` `VLModel` splices.
    var mergedTokenCount: Int {
        let llmSide = gridSide / cfg.spatialMergeSize
        return llmSide * llmSide
    }

    init(cfg: Qwen25VLVisionConfig, patchEmbedWeight: Tensor,
         patchDimPadded: Int,
         blocks: [Qwen25VLVisionBlock], mergerNorm: RMSNorm,
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

    /// The patch grid side for a square image at the standard Qwen2.5-VL
    /// test resolution. We size the test image so the grid divides the
    /// merge size cleanly and the window size; 28×28 patches at
    /// patch-14 → 392×392 px works for the default 112-px window.
    static let defaultGridSide = 28

    static func load(
        visionConfig: ModelConfig, textHidden: Int,
        weights: SafeTensorsBundle, dtype: DType, device: Device
    ) throws -> Qwen25VLVisionModel {
        let cfg = try Qwen25VLVisionConfig.decode(visionConfig)

        // Patch-embed: the checkpoint stores a Conv3d weight
        // `[hidden, in_ch, tPatch, patch, patch]`. We flatten it to a
        // 2D GEMM weight `[hidden, in_ch·tPatch·patch·patch]` so each
        // unfolded patch row projects with one matmul.
        let rawPatch = try weights.tensor(named: "patch_embed.proj.weight")
        let patchDim = cfg.inChannels * cfg.temporalPatchSize
            * cfg.patchSize * cfg.patchSize
        // Round the unfold dim up to the GEMM K-tile width so the
        // patch-embed projection dispatches as a single `Ops.gemm`.
        let patchDimPadded = ((patchDim + gemmKTile - 1) / gemmKTile) * gemmKTile
        let patchEmbedWeight = flattenPatchEmbed(
            rawPatch, hidden: cfg.hidden, patchDim: patchDim,
            patchDimPadded: patchDimPadded, device: device)

        // Block stack. The SwiGLU intermediate is rounded up to the
        // GEMM K-tile width so the `down` projection's `inDim` aligns;
        // `gate`/`up` output rows and `down` input columns are
        // zero-extended to the padded dim.
        let paddedIntermediate =
            ((cfg.intermediate + gemmKTile - 1) / gemmKTile) * gemmKTile
        var blocks: [Qwen25VLVisionBlock] = []
        blocks.reserveCapacity(cfg.depth)
        for i in 0..<cfg.depth {
            let p = "blocks.\(i)"
            func lin(_ name: String) throws -> Linear {
                let w = try weights.tensor(named: "\(p).\(name).weight")
                let b = try? weights.tensor(named: "\(p).\(name).bias")
                return Linear(weight: w, bias: b)
            }
            func norm(_ name: String) throws -> RMSNorm {
                RMSNorm(weight: try weights.tensor(named: "\(p).\(name).weight"),
                        eps: cfg.rmsNormEps)
            }
            let gate = try lin("mlp.gate_proj")
            let up = try lin("mlp.up_proj")
            let down = try lin("mlp.down_proj")
            blocks.append(Qwen25VLVisionBlock(
                norm1: try norm("norm1"), norm2: try norm("norm2"),
                qkv: try lin("attn.qkv"), proj: try lin("attn.proj"),
                gate: padLinearRows(gate, toRows: paddedIntermediate, device: device),
                up: padLinearRows(up, toRows: paddedIntermediate, device: device),
                down: padLinearCols(down, toCols: paddedIntermediate, device: device),
                paddedIntermediate: paddedIntermediate, cfg: cfg))
        }

        // Patch-merger.
        let mergerNorm = RMSNorm(
            weight: try weights.tensor(named: "merger.ln_q.weight"),
            eps: cfg.rmsNormEps)
        func mergerLin(_ idx: Int) throws -> Linear {
            let w = try weights.tensor(named: "merger.mlp.\(idx).weight")
            let b = try? weights.tensor(named: "merger.mlp.\(idx).bias")
            return Linear(weight: w, bias: b)
        }
        let mergerFC1 = try mergerLin(0)
        let mergerFC2 = try mergerLin(2)

        return Qwen25VLVisionModel(
            cfg: cfg, patchEmbedWeight: patchEmbedWeight,
            patchDimPadded: patchDimPadded, blocks: blocks,
            mergerNorm: mergerNorm, mergerFC1: mergerFC1, mergerFC2: mergerFC2,
            textHidden: textHidden, dtype: dtype, gridSide: defaultGridSide)
    }

    /// The `Ops.gemm` K-tile width — `inDim` must be a multiple of it.
    static let gemmKTile = 16

    /// Run the full vision forward on a preprocessed image. `image` is a
    /// normalized NCHW tensor `[1, inChannels, side, side]` where
    /// `side = gridSide · patchSize`. Returns `[mergedTokenCount,
    /// textHidden]`.
    func encode(image: Tensor, device: Device) -> Tensor {
        let side = gridSide
        let nPatches = side * side

        // ── Patch unfold + embed ──
        // Build the unfolded patch matrix `[nPatches, patchDim]`. The
        // temporal axis is a single frame repeated `temporalPatchSize`
        // times (mirrors the reference's temporal padding).
        let unfolded = unfoldPatches(image: image)
        let cmd = device.makeCommandBuffer()
        var h = Ops.gemm(weight: patchEmbedWeight, input: unfolded,
                         nRows: nPatches, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── M-RoPE tables + window groups ──
        let (cosTable, sinTable) = ropeTables()
        let windowGroups = windowAttentionGroups()
        let fullGroup = [Array(0..<nPatches)]

        // ── Block stack ──
        for (i, block) in blocks.enumerated() {
            let groups = cfg.fullattBlockIndexes.contains(i)
                ? fullGroup : windowGroups
            h = block.forward(h, nTokens: nPatches,
                              cosTable: cosTable, sinTable: sinTable,
                              windowGroups: groups, device: device)
        }

        // ── Patch merger ──
        return mergePatches(h, device: device)
    }

    /// Present the tower as a `VisionEncoder` so `VLModel` accepts it.
    func asVisionEncoder() -> VisionEncoder {
        Qwen25VLComposedEncoder(tower: self)
    }

    // ── Patch unfold ──

    /// Unfold a normalized NCHW image `[1, C, side·patch, side·patch]`
    /// into a `[nPatches, patchDim]` tensor. Each row holds one
    /// `C · tPatch · patch · patch` patch, the temporal axis being the
    /// single frame repeated `temporalPatchSize` times. Row order is
    /// raster `(patchRow, patchCol)` — matched by the merger's window
    /// reindex.
    private func unfoldPatches(image: Tensor) -> Tensor {
        let side = gridSide
        let p = cfg.patchSize
        let c = cfg.inChannels
        let tp = cfg.temporalPatchSize
        let pix = image.toFloatArray()        // [C, H, W]
        let imgSide = side * p
        // Rows are zero-padded to `patchDimPadded`; the trailing
        // columns pair with the patch-embed weight's zero-pad columns.
        var rows = [Float](repeating: 0, count: side * side * patchDimPadded)
        for pr in 0..<side {
            for pc in 0..<side {
                let patch = pr * side + pc
                var col = 0
                // Layout: (temporal, channel, py, px) — the conv weight
                // is flattened in the same order.
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
            }
        }
        return ImagePreprocessing.makeTensor(
            from: rows, shape: [side * side, patchDimPadded],
            dtype: dtype, device: .shared)
    }

    // ── M-RoPE ──

    /// Build the per-token rotary `cos` / `sin` tables `[nPatches,
    /// headDim]`. Qwen's vision M-RoPE interleaves a height-rotary and a
    /// width-rotary half over `headDim`, each driven by the patch's
    /// `(h, w)` grid coordinate. The merge-size block reorder of the
    /// position ids matches the reference `rot_pos_emb`.
    private func ropeTables() -> (cos: [Float], sin: [Float]) {
        let side = gridSide
        let nPatches = side * side
        let headDim = cfg.headDim
        let half = headDim / 2          // height half | width half
        let quarter = half / 2          // distinct rotary frequencies
        // inv_freq over quarter dims, theta 10000 — VisionRotaryEmbedding.
        var invFreq = [Float](repeating: 0, count: quarter)
        for i in 0..<quarter {
            invFreq[i] = 1.0 / pow(10_000, Float(2 * i) / Float(half))
        }
        // Per-patch (h, w) coordinates, reordered into merge-size blocks.
        let (hPos, wPos) = mergeReorderedPositions()

        var cosT = [Float](repeating: 0, count: nPatches * headDim)
        var sinT = [Float](repeating: 0, count: nPatches * headDim)
        for t in 0..<nPatches {
            let base = t * headDim
            // Height rotary fills [0, half); width rotary fills [half, 2·half).
            for i in 0..<quarter {
                let fh = Float(hPos[t]) * invFreq[i]
                let fw = Float(wPos[t]) * invFreq[i]
                // freqs are tiled to half (duplicate for rotate-half).
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

    /// Per-patch `(h, w)` grid coordinates, reordered into the
    /// merge-size block layout the reference's `rot_pos_emb` produces.
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

    /// Partition the patch tokens into window-attention groups. The
    /// patches are already in merge-block raster order; a window is a
    /// `windowGrid × windowGrid` block of merge-units. Tokens within a
    /// window attend only to each other.
    private func windowAttentionGroups() -> [[Int]] {
        let side = gridSide
        let m = cfg.spatialMergeSize
        let mergeBlocks = side / m                       // llm grid side
        // window in units of merge-blocks.
        let winBlocks = max(1, cfg.windowSize / (m * cfg.patchSize))
        let nWin = (mergeBlocks + winBlocks - 1) / winBlocks
        // token index = ((br*blocks)+bc)*m*m + (ir*m+ic)
        var groups: [[Int]] = []
        for wr in 0..<nWin {
            for wc in 0..<nWin {
                var group: [Int] = []
                for lbr in (wr * winBlocks)..<min((wr + 1) * winBlocks, mergeBlocks) {
                    for lbc in (wc * winBlocks)..<min((wc + 1) * winBlocks, mergeBlocks) {
                        let blockBase = (lbr * mergeBlocks + lbc) * m * m
                        for k in 0..<(m * m) { group.append(blockBase + k) }
                    }
                }
                if !group.isEmpty { groups.append(group) }
            }
        }
        return groups
    }

    // ── Patch merger ──

    /// Pool each `mergeSize²` neighbourhood of post-encoder tokens into
    /// one token, then project into the text hidden dim. The tokens are
    /// in merge-block raster order, so each consecutive run of
    /// `mergeUnit` tokens is one neighbourhood. Returns
    /// `[mergedTokenCount, textHidden]`.
    private func mergePatches(_ h: Tensor, device: Device) -> Tensor {
        let nPatches = gridSide * gridSide
        let hidden = cfg.hidden
        let mergeUnit = cfg.mergeUnit
        let merged = nPatches / mergeUnit

        // RMSNorm each token, then group `mergeUnit` tokens into one row.
        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNormRows(h, weight: mergerNorm.weight,
                                     eps: mergerNorm.eps, nRows: nPatches,
                                     rowSize: hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // The merger MLP input dim is `mergeUnit · hidden`; the tokens
        // are already contiguous per neighbourhood, so a plain reshape
        // of the buffer view to `[merged, mergeUnit·hidden]` suffices.
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

    // ── Static helpers ──

    /// Flatten the patch-embed conv weight into a 2D GEMM weight
    /// `[hidden, patchDimPadded]` whose first `patchDim` columns match
    /// `unfoldPatches`' row layout `(tP, in_ch, py, px)` and whose
    /// trailing `patchDimPadded - patchDim` columns are zero-pad.
    ///
    /// The mlx-community Qwen 2.5-VL conversion stores the Conv3d weight
    /// in MLX's channel-last layout `[hidden, tP, py, px, in_ch]`; a
    /// PyTorch checkpoint would store `[hidden, in_ch, tP, py, px]`.
    /// Both 5D layouts are detected (by the trailing dim) and repacked.
    static func flattenPatchEmbed(_ w: Tensor, hidden: Int, patchDim: Int,
                                  patchDimPadded: Int, device: Device) -> Tensor {
        precondition(w.shape.count == 5,
                     "Qwen25VL: patch-embed weight must be 5D Conv3d, "
                     + "got \(w.shape)")
        let src = w.toFloatArray()
        // Zero-initialized — the pad columns stay zero.
        var dst = [Float](repeating: 0, count: hidden * patchDimPadded)
        // dst column order: (((t·inCh + ch)·p + py)·p + px).
        let mlxLayout = w.shape[4] <= 4   // trailing dim is in_channels
        if mlxLayout {
            // src `[hidden, tP, py, px, inCh]` — channel last.
            let tP = w.shape[1], p = w.shape[2], inCh = w.shape[4]
            for o in 0..<hidden {
                for t in 0..<tP {
                    for py in 0..<p {
                        for px in 0..<p {
                            for ch in 0..<inCh {
                                let s = ((((o * tP + t) * p + py) * p + px) * inCh + ch)
                                let col = (((t * inCh + ch) * p + py) * p + px)
                                dst[o * patchDimPadded + col] = src[s]
                            }
                        }
                    }
                }
            }
        } else {
            // src `[hidden, inCh, tP, py, px]` — PyTorch channel-first.
            let inCh = w.shape[1], tP = w.shape[2], p = w.shape[3]
            for o in 0..<hidden {
                for ch in 0..<inCh {
                    for t in 0..<tP {
                        for py in 0..<p {
                            for px in 0..<p {
                                let s = ((((o * inCh + ch) * tP + t) * p + py) * p + px)
                                let col = (((t * inCh + ch) * p + py) * p + px)
                                dst[o * patchDimPadded + col] = src[s]
                            }
                        }
                    }
                }
            }
        }
        return ImagePreprocessing.makeTensor(
            from: dst, shape: [hidden, patchDimPadded], dtype: w.dtype,
            device: device)
    }

    /// Zero-extend a `Linear`'s output rows from `[outOld, inDim]` to
    /// `[toRows, inDim]` (and its bias to `[toRows]`). The extra rows
    /// are zero, so the extra outputs are zero — used to pad the
    /// SwiGLU `gate`/`up` outputs up to the K-tile-aligned intermediate.
    static func padLinearRows(_ linear: Linear, toRows: Int,
                              device: Device) -> Linear {
        let outOld = linear.weight.shape[0]
        let inDim = linear.weight.shape[1]
        if outOld == toRows { return linear }
        precondition(toRows >= outOld,
                     "Qwen25VL.padLinearRows: target \(toRows) < \(outOld)")
        let src = linear.weight.toFloatArray()
        var dst = [Float](repeating: 0, count: toRows * inDim)
        for r in 0..<outOld {
            for c in 0..<inDim { dst[r * inDim + c] = src[r * inDim + c] }
        }
        let w = ImagePreprocessing.makeTensor(
            from: dst, shape: [toRows, inDim], dtype: linear.weight.dtype,
            device: device)
        var b: Tensor?
        if let bias = linear.bias {
            let bs = bias.toFloatArray()
            var bd = [Float](repeating: 0, count: toRows)
            for i in 0..<outOld { bd[i] = bs[i] }
            b = ImagePreprocessing.makeTensor(
                from: bd, shape: [toRows], dtype: bias.dtype, device: device)
        }
        return Linear(weight: w, bias: b)
    }

    /// Zero-extend a `Linear`'s input columns from `[outDim, inOld]` to
    /// `[outDim, toCols]`. The extra columns are zero, so they
    /// contribute nothing — used to pad the SwiGLU `down` projection's
    /// `inDim` up to the K-tile-aligned intermediate.
    static func padLinearCols(_ linear: Linear, toCols: Int,
                              device: Device) -> Linear {
        let outDim = linear.weight.shape[0]
        let inOld = linear.weight.shape[1]
        if inOld == toCols { return linear }
        precondition(toCols >= inOld,
                     "Qwen25VL.padLinearCols: target \(toCols) < \(inOld)")
        let src = linear.weight.toFloatArray()
        var dst = [Float](repeating: 0, count: outDim * toCols)
        for r in 0..<outDim {
            for c in 0..<inOld { dst[r * toCols + c] = src[r * inOld + c] }
        }
        let w = ImagePreprocessing.makeTensor(
            from: dst, shape: [outDim, toCols], dtype: linear.weight.dtype,
            device: device)
        return Linear(weight: w, bias: linear.bias)
    }

    /// Broadcast-add a `[rowSize]` bias to each of `nRows` rows of a
    /// flat `[nRows, rowSize]` tensor. Shared by the vision blocks +
    /// the merger.
    static func addRowBias(_ x: Tensor, bias: Tensor, nRows: Int,
                           rowSize: Int, on cmd: MTLCommandBuffer) -> Tensor {
        let biasVals = bias.toFloatArray()
        var flat = [Float](repeating: 0, count: nRows * rowSize)
        for r in 0..<nRows {
            for c in 0..<rowSize { flat[r * rowSize + c] = biasVals[c] }
        }
        let tiled = Tensor.empty(shape: [nRows, rowSize], dtype: x.dtype)
        ImagePreprocessing.copyFloats(flat, into: tiled)
        return Ops.add(x, tiled, on: cmd)
    }
}

/// A `VisionEncoder` subclass whose `encode` runs the Qwen 2.5-VL vision
/// tower — so `VLModel` (which holds a `VisionEncoder`) transparently
/// gets the merged, projected vision tokens.
final class Qwen25VLComposedEncoder: VisionEncoder {
    let tower: Qwen25VLVisionModel

    init(tower: Qwen25VLVisionModel) {
        self.tower = tower
        let c = tower.cfg
        // The facade config reports the square image side the tower
        // expects (`gridSide · patchSize`) and the merged token count
        // as `numPatches` so `VLModel.imageTokenCount` is correct.
        let side = tower.gridSide * c.patchSize
        let facadeConfig = VisionEncoderConfig(
            inChannels: c.inChannels, imageSize: side,
            patchSize: side / Int(Double(tower.mergedTokenCount).squareRoot()),
            hidden: c.hidden, intermediate: c.intermediate,
            nLayers: c.depth, nHeads: c.numHeads,
            layerNormEps: c.rmsNormEps, textHidden: tower.textHidden)
        // The patch-embed / position / layers are unused by the override
        // below; pass minimal placeholders.
        let placeholderW = tower.patchEmbedWeight
        super.init(
            config: facadeConfig,
            patchEmbedWeight: placeholderW, patchEmbedBias: placeholderW,
            positionEmbedding: placeholderW, layers: [],
            postLayerNorm: tower.mergerNorm.asLayerNorm(),
            projection: nil, dtype: tower.dtype)
    }

    /// Run the Qwen 2.5-VL vision tower. Returns
    /// `[mergedTokenCount, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        tower.encode(image: image, device: device)
    }
}

extension RMSNorm {
    /// A throw-away `LayerNorm` view used only to satisfy the
    /// `VisionEncoder` initializer for the Qwen 2.5-VL facade (whose
    /// `encode` is fully overridden, so the post-norm is never invoked).
    func asLayerNorm() -> LayerNorm {
        let zeroBias = Tensor.empty(shape: weight.shape, dtype: weight.dtype)
        return LayerNorm(weight: weight, bias: zeroBias, eps: eps)
    }
}
