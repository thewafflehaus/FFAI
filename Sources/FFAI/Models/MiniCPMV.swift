// MiniCPM-V 4.6 — OpenBMB's `MiniCPMV4_6ForConditionalGeneration`
// checkpoint (`model_type: minicpmv4_6`). An image/video VLM composed
// of a **SigLIP2-400M** vision encoder + the **Qwen3.5** text backbone
// (`qwen3_5_text`, which routes to the in-tree `Qwen35` family),
// joined by a two-stage projector:
//
//   * `vit_merger` — a window cross-attention merger injected MID-stack
//     after encoder layer `insert_layer_id` (default 6). Reduces a
//     `(grid_h × grid_w)` patch grid by a `(2, 2)` factor and projects
//     back to the encoder hidden dim.
//   * `merger`     — the final 2×2 reduction + projection into the text
//     hidden dim (`MergerBlock`: LayerNorm → Linear → GELU → Linear).
//
// The vision tokens are spliced into the text embedding stream at every
// `image_token_id` (248056) position by the shared `VLModel`.
//
// ─── v1 scope ────────────────────────────────────────────────────────
//
// One 448×448 tile per image (single image, no LLaVA-UHD slicing). The
// SigLIP patch grid is 32×32 = 1024 patches; `vit_merger` (2,2) reduces
// it to 16×16 = 256; the final `merger` (2,2) reduces to 8×8 = 64
// tokens — matching the checkpoint's `query_num: 64`. The shipped
// `position_embedding` is `[4900, 1152]` (70×70 grid at the checkpoint
// `image_size: 980`); it is bilinearly interpolated to the runtime
// 32×32 grid once at load.
//
// The 4×-detail OCR mode (which skips `vit_merger`) and the multi-tile
// LLaVA-UHD path are deferred — they layer on top of the same building
// blocks.
//
// Reference: `mlx-vlm/mlx_vlm/models/minicpmv4_6/`
// (`minicpmv4_6.py`, `vision.py`, `processing_minicpmv4_6.py`).

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum MiniCPMV4_6 {
    public static let modelTypes: Set<String> = ["minicpmv4_6"]
    public static let architectures: Set<String> =
        ["MiniCPMV4_6ForConditionalGeneration"]

    /// `image_token_id` default for MiniCPM-V-4.6 checkpoints — the
    /// placeholder the chat template emits for each image.
    public static let defaultImageTokenId = 248056

    /// Runtime tile resolution for the v1 path. 448 = 32×32 patches at
    /// `patch_size: 14`; `vit_merger` (2,2) → 16×16; `merger` (2,2) →
    /// 8×8 = 64 tokens — matches `query_num: 64`.
    public static let runtimeImageSize = 448

    /// Build a `VLModel` from a `MiniCPMV4_6ForConditionalGeneration`
    /// checkpoint: SigLIP2-400M encoder + `vit_merger` + `merger` +
    /// Qwen3.5 text backbone, joined by `VLModel`'s cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VLModel {
        guard let visionConfig = config.subConfig("vision_config"),
              config.nested("text_config") != nil
        else {
            throw MiniCPMVError.missingConfig(
                "vision_config / text_config")
        }

        // ── Text backbone: Qwen3.5 from `text_config` ─────────────────
        // Raw checkpoint stores text weights as
        // `model.language_model.{embed_tokens, layers.*, norm}.weight`
        // (no inner `model.`). The Qwen3.5 loader reads `model.X` —
        // strip the outer namespace, then re-prepend `model.` so the
        // existing loader runs unchanged. Qwen3.5's loader pulls every
        // text hyper-parameter from `config.text_config`, so we hand it
        // the full MiniCPM root config.
        let textWeights = weights
            .prefixed("model.language_model.")
            .withAddedPrefix("model.")
        let textEngine = try Qwen35Hybrid.loadModel(
            config: config, weights: textWeights,
            options: options, device: device)

        // ── Vision tower: SigLIP2 encoder + vit_merger + merger ───────
        // All vision weights live under `model.vision_tower.*` and
        // `model.merger.*` — strip those outer prefixes.
        let visionWeights = weights.prefixed("model.vision_tower.")
        let mergerWeights = weights.prefixed("model.merger.")
        let composed = try MiniCPMVComposedEncoder.load(
            visionConfig: visionConfig,
            insertLayerId: config.int("insert_layer_id") ?? 6,
            mergerTimes: config.int("merger_times") ?? 1,
            runtimeImageSize: runtimeImageSize,
            textHidden: textEngine.hidden,
            visionWeights: visionWeights,
            mergerWeights: mergerWeights,
            device: device)

        let imageTokenId = config.int("image_token_id") ?? defaultImageTokenId
        return try VLModel(
            visionEncoder: composed, engine: textEngine,
            imageTokenId: imageTokenId, normalization: .siglip,
            imageTokenCount: composed.outputTokenCount)
    }
}

public enum MiniCPMVError: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "MiniCPM-V: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "MiniCPM-V: unsupported config: \(m)"
        }
    }
}

// ─── MiniCPMVViTMerger — window cross-attention merger ───────────────
//
// One pass of (2,2) grid merging with a window self-attention block,
// then a flat MLP, with two residual paths:
//
//   1. windows + window_self_attention(LN(windows))   [num_windows, 4, h]
//   2. window-mean(1) used as a residual on the merged-and-projected output
//
// Reference: `VitMerger` in `mlx-vlm/models/minicpmv4_6/minicpmv4_6.py`.

final class MiniCPMVViTMerger {
    /// Per-token LayerNorm applied before the window self-attention.
    let layerNorm1: LayerNorm
    let qProj, kProj, vProj, oProj: Linear

    /// LayerNorm over `4 · hidden` (the flattened-window dim).
    let preNorm: LayerNorm
    /// Linear `4·hidden → windowIntermediate`.
    let linear1: Linear
    /// Linear `windowIntermediate → hidden`.
    let linear2: Linear

    /// Encoder hidden dim and head count (matches the parent encoder).
    let hidden, nHeads, headDim: Int
    let scale: Float
    /// 4 — group_h × group_w (the (2,2) window).
    let groupTokens: Int = 4

    init(layerNorm1: LayerNorm,
         qProj: Linear, kProj: Linear, vProj: Linear, oProj: Linear,
         preNorm: LayerNorm, linear1: Linear, linear2: Linear,
         hidden: Int, nHeads: Int) {
        self.layerNorm1 = layerNorm1
        self.qProj = qProj; self.kProj = kProj
        self.vProj = vProj; self.oProj = oProj
        self.preNorm = preNorm
        self.linear1 = linear1; self.linear2 = linear2
        self.hidden = hidden; self.nHeads = nHeads
        self.headDim = hidden / nHeads
        self.scale = 1.0 / Float(Double(hidden / nHeads).squareRoot())
    }

    /// Load the merger from a sub-bundle prefixed at `model.vision_tower.
    /// vit_merger.`. Every linear ships a bias.
    static func load(weights: SafeTensorsBundle,
                     hidden: Int, nHeads: Int,
                     windowIntermediate: Int, eps: Float) throws -> MiniCPMVViTMerger {
        func lin(_ name: String) throws -> Linear {
            Linear(weight: try weights.tensor(named: "\(name).weight"),
                   bias: try weights.tensor(named: "\(name).bias"))
        }
        let ln1 = LayerNorm(
            weight: try weights.tensor(named: "layer_norm1.weight"),
            bias: try weights.tensor(named: "layer_norm1.bias"), eps: eps)
        let preNorm = LayerNorm(
            weight: try weights.tensor(named: "pre_norm.weight"),
            bias: try weights.tensor(named: "pre_norm.bias"), eps: eps)
        return MiniCPMVViTMerger(
            layerNorm1: ln1,
            qProj: try lin("self_attn.q_proj"),
            kProj: try lin("self_attn.k_proj"),
            vProj: try lin("self_attn.v_proj"),
            oProj: try lin("self_attn.out_proj"),
            preNorm: preNorm,
            linear1: try lin("linear_1"), linear2: try lin("linear_2"),
            hidden: hidden, nHeads: nHeads)
    }

    /// Forward `tokens` `[gridH·gridW, hidden]` (row-major over the
    /// patch grid) through one (2,2) window merge → window self-attn →
    /// flatten-and-project block. Returns `[(gridH/2)·(gridW/2), hidden]`
    /// in the same dtype. `gridH` and `gridW` must both be even.
    func forward(tokens: Tensor, gridH: Int, gridW: Int,
                 device: Device) -> Tensor {
        precondition(tokens.shape == [gridH * gridW, hidden],
                     "MiniCPMVViTMerger: tokens \(tokens.shape) ≠ "
                     + "[\(gridH * gridW), \(hidden)]")
        precondition(gridH % 2 == 0 && gridW % 2 == 0,
                     "MiniCPMVViTMerger: grid \(gridH)x\(gridW) must be (2,2)-divisible")
        let mergedH = gridH / 2, mergedW = gridW / 2
        let nWindows = mergedH * mergedW
        let nTokens = nWindows * groupTokens

        // ── 1. Reshape patch grid into (2,2) windows ──────────────────
        // Source: row-major [gridH, gridW, hidden]. Target: window-major
        // [nWindows, 4, hidden]. Done CPU-side — the encoder already
        // synchronises here (the per-layer attention core does).
        let windowed = reshapeIntoWindows(
            tokens: tokens, gridH: gridH, gridW: gridW, device: device)

        // ── 2. Per-token layer_norm1 over all nTokens patches ─────────
        let cmd = device.makeCommandBuffer()
        let normed = Ops.layerNorm(
            windowed, weight: layerNorm1.weight, bias: layerNorm1.bias,
            eps: layerNorm1.eps, nRows: nTokens, rowSize: hidden, on: cmd)

        // ── 3. Project Q / K / V over all nTokens (multi-row GEMM) ────
        let q = projectRows(qProj, normed, nRows: nTokens, outDim: hidden, on: cmd)
        let k = projectRows(kProj, normed, nRows: nTokens, outDim: hidden, on: cmd)
        let v = projectRows(vProj, normed, nRows: nTokens, outDim: hidden, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── 4. Per-window CPU self-attention ──────────────────────────
        let attnFlat = cpuWindowAttention(
            q: q, k: k, v: v, nWindows: nWindows, device: device)

        // ── 5. out_proj on all nTokens; window-residual add ───────────
        let cmd2 = device.makeCommandBuffer()
        let attnOut = projectRows(oProj, attnFlat, nRows: nTokens,
                                  outDim: hidden, on: cmd2)
        // Residual: original windowed tokens + attention output (token-wise).
        let postAttn = Ops.add(windowed, attnOut, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        // ── 6. mean over each window's 4 tokens → [nWindows, hidden] ──
        let meanResidual = meanOverWindowTokens(
            postAttn, nWindows: nWindows, hidden: hidden, device: device)

        // ── 7. Flatten windows: [nWindows, 4·hidden] ──────────────────
        // postAttn is [nWindows·4, hidden] window-major; the flattened
        // view is the same memory reinterpreted as [nWindows, 4·hidden]
        // — that is, `reshape(postAttn, [nWindows, groupTokens*hidden])`.
        let flat = postAttn.reshaped(to: [nWindows, groupTokens * hidden])
        let groupHidden = groupTokens * hidden

        // ── 8. pre_norm (LN over 4·hidden) → linear_1 → GELU → linear_2 ─
        let cmd3 = device.makeCommandBuffer()
        let normedFlat = Ops.layerNorm(
            flat, weight: preNorm.weight, bias: preNorm.bias,
            eps: preNorm.eps, nRows: nWindows, rowSize: groupHidden, on: cmd3)
        let ff1 = projectRows(linear1, normedFlat, nRows: nWindows,
                              outDim: linear1.weight.shape[0], on: cmd3)
        let act = Ops.gelu(ff1, on: cmd3)
        let ff2 = projectRows(linear2, act, nRows: nWindows,
                              outDim: hidden, on: cmd3)
        // ── 9. + mean residual → [nWindows, hidden] ──────────────────
        let result = Ops.add(ff2, meanResidual, on: cmd3)
        cmd3.commit()
        cmd3.waitUntilCompleted()
        return result
    }

    /// CPU reshape `[gridH·gridW, hidden]` row-major → `[nWindows·4,
    /// hidden]` window-major. Window order matches the python
    /// reference's `transpose(0, 2, 1, 3, 4)`: for each output row
    /// `(window_id, in_window)`, the source row is
    /// `((mh*2+dh)*gridW + (mw*2+dw))`.
    private func reshapeIntoWindows(tokens: Tensor, gridH: Int, gridW: Int,
                                    device: Device) -> Tensor {
        let src = tokens.toFloatArray()
        let mergedH = gridH / 2, mergedW = gridW / 2
        let nWindows = mergedH * mergedW
        var dst = [Float](repeating: 0, count: nWindows * 4 * hidden)
        for mh in 0..<mergedH {
            for mw in 0..<mergedW {
                let windowId = mh * mergedW + mw
                for dh in 0..<2 {
                    for dw in 0..<2 {
                        let inWindow = dh * 2 + dw
                        let srcRow = (mh * 2 + dh) * gridW + (mw * 2 + dw)
                        let dstRow = windowId * 4 + inWindow
                        for c in 0..<hidden {
                            dst[dstRow * hidden + c] = src[srcRow * hidden + c]
                        }
                    }
                }
            }
        }
        let out = Tensor.empty(shape: [nWindows * 4, hidden], dtype: tokens.dtype,
                               device: device)
        ImagePreprocessing.copyFloats(dst, into: out)
        return out
    }

    /// CPU per-window multi-head attention over `[nWindows·4, hidden]`
    /// Q/K/V buffers. Each window is 4 tokens; the (window, head)
    /// outer-product is `concurrentPerform`-fanned across cores. The
    /// returned tensor has the same `[nWindows·4, hidden]` layout.
    private func cpuWindowAttention(q: Tensor, k: Tensor, v: Tensor,
                                    nWindows: Int, device: Device) -> Tensor {
        let qa = q.toFloatArray()
        let ka = k.toFloatArray()
        let va = v.toFloatArray()
        let stride = nHeads * headDim
        precondition(stride == hidden,
                     "MiniCPMVViTMerger: nHeads·headDim must equal hidden")
        var out = [Float](repeating: 0, count: nWindows * 4 * stride)
        let hd = headDim, sc = scale, nh = nHeads
        out.withUnsafeMutableBufferPointer { outBuf in
            let outPtr = outBuf.baseAddress!
            qa.withUnsafeBufferPointer { qPtr in
            ka.withUnsafeBufferPointer { kPtr in
            va.withUnsafeBufferPointer { vPtr in
                let qb = qPtr.baseAddress!
                let kb = kPtr.baseAddress!
                let vb = vPtr.baseAddress!
                // Fan (window, head, query) across cores. 4 query tokens
                // per window × 16 heads × N windows — embarrassingly
                // parallel; each iteration writes a disjoint headDim slice.
                DispatchQueue.concurrentPerform(iterations: nWindows * nh * 4) { work in
                    let i = work % 4              // query token in window
                    let head = (work / 4) % nh
                    let win = work / (4 * nh)
                    let baseToken = win * 4
                    let hOff = head * hd
                    var scores = [Float](repeating: 0, count: 4)
                    var maxS = -Float.greatestFiniteMagnitude
                    let qBase = (baseToken + i) * stride + hOff
                    for j in 0..<4 {
                        var dot: Float = 0
                        let kBase = (baseToken + j) * stride + hOff
                        for d in 0..<hd { dot += qb[qBase + d] * kb[kBase + d] }
                        let s = dot * sc
                        scores[j] = s
                        if s > maxS { maxS = s }
                    }
                    var sumExp: Float = 0
                    for j in 0..<4 {
                        let e = exp(scores[j] - maxS)
                        scores[j] = e
                        sumExp += e
                    }
                    let inv = sumExp > 0 ? 1 / sumExp : 0
                    let oBase = (baseToken + i) * stride + hOff
                    for j in 0..<4 {
                        let w = scores[j] * inv
                        let vBase = (baseToken + j) * stride + hOff
                        for d in 0..<hd { outPtr[oBase + d] += w * vb[vBase + d] }
                    }
                }
            }}}
        }
        let result = Tensor.empty(shape: [nWindows * 4, stride], dtype: q.dtype,
                                  device: device)
        ImagePreprocessing.copyFloats(out, into: result)
        return result
    }

    /// Mean of each window's 4 tokens → `[nWindows, hidden]`.
    private func meanOverWindowTokens(_ x: Tensor, nWindows: Int,
                                      hidden: Int, device: Device) -> Tensor {
        let src = x.toFloatArray()
        var dst = [Float](repeating: 0, count: nWindows * hidden)
        for w in 0..<nWindows {
            for j in 0..<4 {
                let srcBase = (w * 4 + j) * hidden
                let dstBase = w * hidden
                for c in 0..<hidden { dst[dstBase + c] += src[srcBase + c] }
            }
            let inv: Float = 0.25
            let dstBase = w * hidden
            for c in 0..<hidden { dst[dstBase + c] *= inv }
        }
        let out = Tensor.empty(shape: [nWindows, hidden], dtype: x.dtype,
                               device: device)
        ImagePreprocessing.copyFloats(dst, into: out)
        return out
    }
}

// ─── MiniCPMVMerger — final (2,2) reduction + text-hidden projection ──
//
// `merger_times` rounds of `MergerBlock`. Each round groups (2,2) tokens
// into a flat `4·hidden` row, then runs `LayerNorm → Linear → GELU →
// Linear`. v1 ships `merger_times = 1` (the default), so one block does:
//
//   [num_windows, hidden] (16×16) → reshape → [num_windows/4, 4·hidden]
//                                  → pre_norm → linear_1 → GELU → linear_2
//                                  → [num_windows/4, text_hidden]   (8×8 = 64)

final class MiniCPMVMerger {
    /// One round per `merger_times`. Each block is consumed in order.
    let blocks: [MergerBlock]
    let mergeH: Int = 2
    let mergeW: Int = 2

    init(blocks: [MergerBlock]) {
        precondition(!blocks.isEmpty,
                     "MiniCPMVMerger: must have at least one MergerBlock")
        self.blocks = blocks
    }

    /// Load `mergerTimes` rounds from a bundle prefixed at
    /// `model.merger.`. Each round is `mlp.<i>.{pre_norm, linear_1,
    /// linear_2}.{weight, bias}`. The last round's `linear_2` projects
    /// into the text hidden dim.
    static func load(weights: SafeTensorsBundle,
                     hidden: Int, textHidden: Int, mergerTimes: Int,
                     eps: Float) throws -> MiniCPMVMerger {
        precondition(mergerTimes >= 1,
                     "MiniCPMVMerger: mergerTimes must be ≥ 1, got \(mergerTimes)")
        var blocks: [MergerBlock] = []
        for i in 0..<mergerTimes {
            let p = "mlp.\(i)"
            func lin(_ name: String) throws -> Linear {
                Linear(
                    weight: try weights.tensor(named: "\(p).\(name).weight"),
                    bias: try weights.tensor(named: "\(p).\(name).bias"))
            }
            let preNorm = LayerNorm(
                weight: try weights.tensor(named: "\(p).pre_norm.weight"),
                bias: try weights.tensor(named: "\(p).pre_norm.bias"),
                eps: eps)
            blocks.append(MergerBlock(
                preNorm: preNorm,
                linear1: try lin("linear_1"), linear2: try lin("linear_2")))
        }
        return MiniCPMVMerger(blocks: blocks)
    }

    /// Run all rounds. `tokens` is `[gridH·gridW, hidden]` row-major
    /// over the patch grid. Each round reduces the grid by (2,2). The
    /// final output is `[finalH·finalW, textHidden]`.
    func forward(tokens: Tensor, gridH: Int, gridW: Int,
                 device: Device) -> Tensor {
        var h = tokens
        var gh = gridH, gw = gridW
        for block in blocks {
            (h, gh, gw) = block.forwardOneRound(
                tokens: h, gridH: gh, gridW: gw,
                mergeH: mergeH, mergeW: mergeW, device: device)
        }
        return h
    }
}

/// One `MergerBlock`: LayerNorm → Linear → GELU → Linear with a (2,2)
/// window reshape on the input.
final class MergerBlock {
    let preNorm: LayerNorm
    let linear1, linear2: Linear

    init(preNorm: LayerNorm, linear1: Linear, linear2: Linear) {
        self.preNorm = preNorm
        self.linear1 = linear1
        self.linear2 = linear2
    }

    /// Group `tokens` `[gridH·gridW, in]` into `(mergeH × mergeW)`
    /// windows, run pre_norm → linear_1 → GELU → linear_2, return
    /// `[(gridH/mergeH)·(gridW/mergeW), outSize]`.
    func forwardOneRound(tokens: Tensor, gridH: Int, gridW: Int,
                         mergeH: Int, mergeW: Int,
                         device: Device) -> (Tensor, Int, Int) {
        precondition(gridH % mergeH == 0 && gridW % mergeW == 0,
                     "MergerBlock: grid \(gridH)x\(gridW) not divisible by "
                     + "\(mergeH)x\(mergeW)")
        let inDim = tokens.shape[1]
        let mh = gridH / mergeH, mw = gridW / mergeW
        let nWindows = mh * mw
        let groupSize = mergeH * mergeW
        let groupDim = inDim * groupSize

        // ── CPU window reshape: row-major grid → [nWindows, groupDim] ──
        let src = tokens.toFloatArray()
        var dst = [Float](repeating: 0, count: nWindows * groupDim)
        for h in 0..<mh {
            for w in 0..<mw {
                let windowId = h * mw + w
                var col = 0
                for dh in 0..<mergeH {
                    for dw in 0..<mergeW {
                        let srcRow = (h * mergeH + dh) * gridW + (w * mergeW + dw)
                        for c in 0..<inDim {
                            dst[windowId * groupDim + col] = src[srcRow * inDim + c]
                            col += 1
                        }
                    }
                }
            }
        }
        let grouped = Tensor.empty(shape: [nWindows, groupDim],
                                   dtype: tokens.dtype, device: device)
        ImagePreprocessing.copyFloats(dst, into: grouped)

        // ── LN → Linear → GELU → Linear ──────────────────────────────
        let cmd = device.makeCommandBuffer()
        let normed = Ops.layerNorm(
            grouped, weight: preNorm.weight, bias: preNorm.bias,
            eps: preNorm.eps, nRows: nWindows, rowSize: groupDim, on: cmd)
        let ff1 = applyLinear(linear1, normed, nRows: nWindows,
                              outDim: linear1.weight.shape[0], on: cmd)
        let act = Ops.gelu(ff1, on: cmd)
        let outDim = linear2.weight.shape[0]
        let ff2 = applyLinear(linear2, act, nRows: nWindows,
                              outDim: outDim, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return (ff2, mh, mw)
    }
}

// ─── MiniCPMVComposedEncoder ─────────────────────────────────────────
//
// A `VisionEncoder` subclass whose `encode` runs the SigLIP2 stack
// with `vit_merger` injected after `insertLayerId`, then `post_layernorm`,
// then `merger`. The exposed `numPatches` is the final token count
// (after both mergers), so `VLModel.imageTokenCount` is correct.

final class MiniCPMVComposedEncoder: VisionEncoder {
    let vitMerger: MiniCPMVViTMerger
    let merger: MiniCPMVMerger
    let insertLayerId: Int
    /// Initial patch grid (`runtimeImageSize / patchSize`).
    let initialGridSide: Int
    /// Final output tokens per image — the count `VLModel` splices.
    let outputTokenCount: Int

    init(config: VisionEncoderConfig,
         patchEmbedWeight: Tensor, patchEmbedBias: Tensor,
         positionEmbedding: Tensor, layers: [VisionEncoderLayer],
         postLayerNorm: LayerNorm,
         vitMerger: MiniCPMVViTMerger, merger: MiniCPMVMerger,
         insertLayerId: Int, initialGridSide: Int, outputTokenCount: Int,
         dtype: DType) {
        self.vitMerger = vitMerger
        self.merger = merger
        self.insertLayerId = insertLayerId
        self.initialGridSide = initialGridSide
        self.outputTokenCount = outputTokenCount
        super.init(
            config: config, patchEmbedWeight: patchEmbedWeight,
            patchEmbedBias: patchEmbedBias, positionEmbedding: positionEmbedding,
            layers: layers, postLayerNorm: postLayerNorm,
            projection: nil, dtype: dtype)
    }

    /// Load the composed tower. `visionWeights` is prefixed at
    /// `model.vision_tower.`; `mergerWeights` at `model.merger.`. The
    /// shipped `position_embedding` (`[stored²·hidden]`) is bilinearly
    /// interpolated to the runtime grid at load.
    static func load(
        visionConfig: ModelConfig, insertLayerId: Int,
        mergerTimes: Int, runtimeImageSize: Int, textHidden: Int,
        visionWeights: SafeTensorsBundle, mergerWeights: SafeTensorsBundle,
        device: Device
    ) throws -> MiniCPMVComposedEncoder {
        guard let hidden = visionConfig.int("hidden_size"),
              let storedImageSize = visionConfig.int("image_size"),
              let patchSize = visionConfig.int("patch_size"),
              let intermediate = visionConfig.int("intermediate_size"),
              let nLayers = visionConfig.int("num_hidden_layers"),
              let nHeads = visionConfig.int("num_attention_heads")
        else {
            throw MiniCPMVError.missingConfig(
                "vision_config hidden_size / image_size / patch_size / "
                + "intermediate_size / num_hidden_layers / num_attention_heads")
        }
        let eps = Float(visionConfig.float("layer_norm_eps") ?? 1e-6)

        precondition(runtimeImageSize % patchSize == 0,
                     "MiniCPM-V: runtimeImageSize \(runtimeImageSize) not "
                     + "divisible by patch_size \(patchSize)")
        let runtimeGridSide = runtimeImageSize / patchSize
        precondition(runtimeGridSide % 4 == 0,
                     "MiniCPM-V: runtime grid \(runtimeGridSide) must be "
                     + "(2,2)·(2,2) = 4-divisible for vit_merger + merger")

        // ── Patch embed + interpolated position embedding ─────────────
        // patch_embedding.weight ships [hidden, 3, patch, patch] (PyTorch
        // OIHW) — Ops.conv2d consumes it directly.
        let patchW = try visionWeights.tensor(
            named: "embeddings.patch_embedding.weight")
        let patchB = try visionWeights.tensor(
            named: "embeddings.patch_embedding.bias")
        let posEmbRaw = try visionWeights.tensor(
            named: "embeddings.position_embedding.weight")
        let storedGridSide = storedImageSize / patchSize
        let posEmb = interpolatePositionEmbedding(
            posEmbRaw, storedSide: storedGridSide,
            targetSide: runtimeGridSide, hidden: hidden, device: device)

        // ── SigLIP encoder layers — reuse VisionEncoderLayer ──────────
        var layers: [VisionEncoderLayer] = []
        layers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            let p = "encoder.layers.\(i)"
            let ln1 = LayerNorm(
                weight: try visionWeights.tensor(named: "\(p).layer_norm1.weight"),
                bias: try visionWeights.tensor(named: "\(p).layer_norm1.bias"),
                eps: eps)
            let ln2 = LayerNorm(
                weight: try visionWeights.tensor(named: "\(p).layer_norm2.weight"),
                bias: try visionWeights.tensor(named: "\(p).layer_norm2.bias"),
                eps: eps)
            func lin(_ name: String) throws -> Linear {
                Linear(
                    weight: try visionWeights.tensor(named: "\(p).\(name).weight"),
                    bias: try visionWeights.tensor(named: "\(p).\(name).bias"))
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
            weight: try visionWeights.tensor(named: "post_layernorm.weight"),
            bias: try visionWeights.tensor(named: "post_layernorm.bias"),
            eps: eps)

        // ── vit_merger — load from `model.vision_tower.vit_merger.` ───
        let vitMergerWeights = visionWeights.prefixed("vit_merger.")
        // window_intermediate_size: infer from the linear_1 weight shape
        // (the config also carries it but the weight is authoritative).
        let l1 = try vitMergerWeights.tensor(named: "linear_1.weight")
        let windowIntermediate = l1.shape[0]
        let vitMerger = try MiniCPMVViTMerger.load(
            weights: vitMergerWeights, hidden: hidden, nHeads: nHeads,
            windowIntermediate: windowIntermediate, eps: eps)

        // ── merger ────────────────────────────────────────────────────
        let merger = try MiniCPMVMerger.load(
            weights: mergerWeights, hidden: hidden,
            textHidden: textHidden, mergerTimes: mergerTimes, eps: eps)

        // ── Output token count ────────────────────────────────────────
        // vit_merger reduces (2,2); each merger round reduces (2,2).
        let postViTSide = runtimeGridSide / 2
        precondition(postViTSide >= (1 << mergerTimes),
                     "MiniCPM-V: post-vit_merger grid \(postViTSide) too small "
                     + "for \(mergerTimes) merger rounds")
        var finalSide = postViTSide
        for _ in 0..<mergerTimes { finalSide /= 2 }
        let outputTokens = finalSide * finalSide

        let cfg = VisionEncoderConfig(
            imageSize: runtimeImageSize, patchSize: patchSize, hidden: hidden,
            intermediate: intermediate, nLayers: nLayers, nHeads: nHeads,
            layerNormEps: eps, textHidden: textHidden)

        return MiniCPMVComposedEncoder(
            config: cfg, patchEmbedWeight: patchW, patchEmbedBias: patchB,
            positionEmbedding: posEmb, layers: layers, postLayerNorm: postLN,
            vitMerger: vitMerger, merger: merger,
            insertLayerId: insertLayerId, initialGridSide: runtimeGridSide,
            outputTokenCount: outputTokens, dtype: patchW.dtype)
    }

    /// Override `VisionEncoder.encode` to inject `vit_merger` mid-stack
    /// and apply the final `merger` after `post_layernorm`. Input is the
    /// preprocessed `[1, 3, imageSize, imageSize]` NCHW tensor; output is
    /// `[outputTokenCount, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        precondition(image.shape == [1, config.inChannels,
                                     config.imageSize, config.imageSize],
                     "MiniCPM-V encode: image \(image.shape) ≠ "
                     + "[1,3,\(config.imageSize),\(config.imageSize)]")
        let initSide = initialGridSide
        let initPatches = initSide * initSide

        // ── Patch embed + token-major reshape ─────────────────────────
        let cmd = device.makeCommandBuffer()
        let conv = Ops.conv2d(
            input: image, weight: patchEmbedWeight, bias: patchEmbedBias,
            strideH: config.patchSize, strideW: config.patchSize, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        var tokens = channelMajorToTokenMajor(
            conv, hidden: config.hidden, numPatches: initPatches,
            device: device)

        // ── + position embedding (already interpolated to runtime grid) ─
        let cmd2 = device.makeCommandBuffer()
        var h = Ops.add(tokens, positionEmbedding, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        // ── Walk encoder layers, inject vit_merger after insertLayerId ─
        var gridH = initSide, gridW = initSide
        var nTokens = initPatches
        for (i, layer) in layers.enumerated() {
            let layerCmd = device.makeCommandBuffer()
            h = layer.forward(h, nTokens: nTokens, device: device, on: layerCmd)
            if i == insertLayerId {
                h = vitMerger.forward(tokens: h, gridH: gridH, gridW: gridW,
                                      device: device)
                gridH /= 2; gridW /= 2
                nTokens = gridH * gridW
            }
        }

        // ── post_layernorm on the (post-vit_merger) grid ──────────────
        let cmdN = device.makeCommandBuffer()
        h = Ops.layerNorm(
            h, weight: postLayerNorm.weight, bias: postLayerNorm.bias,
            eps: postLayerNorm.eps, nRows: nTokens, rowSize: config.hidden,
            on: cmdN)
        cmdN.commit()
        cmdN.waitUntilCompleted()

        // ── merger: (2,2)·merger_times reduction + text-hidden project ─
        tokens = merger.forward(tokens: h, gridH: gridH, gridW: gridW,
                                device: device)
        return tokens
    }

    /// Reinterpret a conv2d `[1, hidden, P, P]` channel-major output as
    /// `[numPatches, hidden]` token-major patch tokens. CPU transpose
    /// (the conv2d kernel writes channel-major; we want token-major).
    private func channelMajorToTokenMajor(
        _ conv: Tensor, hidden: Int, numPatches: Int, device: Device
    ) -> Tensor {
        let src = conv.toFloatArray()
        var dst = [Float](repeating: 0, count: numPatches * hidden)
        for c in 0..<hidden {
            for p in 0..<numPatches {
                dst[p * hidden + c] = src[c * numPatches + p]
            }
        }
        let out = Tensor.empty(shape: [numPatches, hidden], dtype: dtype,
                               device: device)
        ImagePreprocessing.copyFloats(dst, into: out)
        return out
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

/// Apply a `Linear` `[outDim, inDim]` to every row of a `[nRows, inDim]`
/// tensor: `Ops.gemm` for the matmul, then broadcast-add the bias to
/// every row. Same shape as `VisionEncoderLayer.projectRows` — pulled
/// out here so `MiniCPMVViTMerger` / `MergerBlock` share the helper.
private func applyLinear(_ linear: Linear, _ x: Tensor, nRows: Int,
                         outDim: Int, on cmd: MTLCommandBuffer) -> Tensor {
    let y = Ops.gemm(weight: linear.weight, input: x, nRows: nRows, on: cmd)
    guard let bias = linear.bias else { return y }
    // Tile [outDim] bias into [nRows*outDim] for element-wise add.
    let tiled = Tensor.empty(shape: [nRows, outDim], dtype: x.dtype)
    let biasVals = bias.toFloatArray()
    var flat = [Float](repeating: 0, count: nRows * outDim)
    for r in 0..<nRows {
        for c in 0..<outDim { flat[r * outDim + c] = biasVals[c] }
    }
    ImagePreprocessing.copyFloats(flat, into: tiled)
    return Ops.add(y, tiled, on: cmd)
}

/// Internal alias so `MiniCPMVViTMerger` can call the shared
/// `applyLinear` from inside an instance method (Swift's name lookup
/// otherwise requires the helper to be a method).
private extension MiniCPMVViTMerger {
    func projectRows(_ linear: Linear, _ x: Tensor, nRows: Int,
                     outDim: Int, on cmd: MTLCommandBuffer) -> Tensor {
        applyLinear(linear, x, nRows: nRows, outDim: outDim, on: cmd)
    }
}

/// Bilinearly interpolate a `[storedSide², hidden]` position embedding
/// to `[targetSide², hidden]`. The python reference uses bicubic with
/// antialiasing; bilinear is the simpler, near-equivalent choice for the
/// modest 70→32 downsample MiniCPM-V-4.6 does at v1's 448×448 runtime
/// — a perf-vs-quality tradeoff we can revisit if a coherence regression
/// appears.
func interpolatePositionEmbedding(
    _ posEmb: Tensor, storedSide: Int, targetSide: Int, hidden: Int,
    device: Device
) -> Tensor {
    precondition(posEmb.elementCount == storedSide * storedSide * hidden,
                 "MiniCPM-V posEmb: \(posEmb.elementCount) elements ≠ "
                 + "\(storedSide)·\(storedSide)·\(hidden)")
    if storedSide == targetSide { return posEmb }
    let src = posEmb.toFloatArray()
    var dst = [Float](repeating: 0, count: targetSide * targetSide * hidden)
    let scale = Float(storedSide) / Float(targetSide)
    for ty in 0..<targetSide {
        // Half-pixel-centered sampling (align_corners = false).
        let srcY = (Float(ty) + 0.5) * scale - 0.5
        let y0 = max(0, min(storedSide - 1, Int(srcY.rounded(.down))))
        let y1 = min(storedSide - 1, y0 + 1)
        let wy = max(0, min(1, srcY - Float(y0)))
        for tx in 0..<targetSide {
            let srcX = (Float(tx) + 0.5) * scale - 0.5
            let x0 = max(0, min(storedSide - 1, Int(srcX.rounded(.down))))
            let x1 = min(storedSide - 1, x0 + 1)
            let wx = max(0, min(1, srcX - Float(x0)))
            let outBase = (ty * targetSide + tx) * hidden
            for c in 0..<hidden {
                let p00 = src[(y0 * storedSide + x0) * hidden + c]
                let p01 = src[(y0 * storedSide + x1) * hidden + c]
                let p10 = src[(y1 * storedSide + x0) * hidden + c]
                let p11 = src[(y1 * storedSide + x1) * hidden + c]
                let top = p00 * (1 - wx) + p01 * wx
                let bot = p10 * (1 - wx) + p11 * wx
                dst[outBase + c] = top * (1 - wy) + bot * wy
            }
        }
    }
    let out = Tensor.empty(shape: [targetSide * targetSide, hidden],
                           dtype: posEmb.dtype, device: device)
    ImagePreprocessing.copyFloats(dst, into: out)
    return out
}
