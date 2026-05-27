// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Gemma 4 VL — vision tower internals + family orchestrator.
//
// This file contains `Gemma4VLVisionConfig`, `Gemma4VLVisionBlock`,
// `Gemma4VLVisionModel`, `Gemma4VLComposedEncoder`, and all supporting
// CPU / GPU helpers for the bespoke Gemma 4 ViT (flattened linear
// patch-embed, learned 2D position tables, multi-dimensional RoPE MHA
// with per-projection q/k/v RMSNorms, GemmaRMSNorm zero-shift norms,
// SwiGLU MLP, attention pooling, multi-modal embedder). The Gemma 4 VL
// vision-language orchestrator (`enum Gemma4VL`) — which ties the
// Gemma 4 text backbone to this vision tower — lives at the bottom of
// this file. The family entry-point (`enum Gemma4`, `Gemma4Variant`,
// `Gemma4Error`) lives in `Models/Gemma4.swift`.

import Foundation
import Metal

// ─── Gemma 4 vision tower ────────────────────────────────────────────

/// Static shape of the Gemma 4 vision tower, decoded from
/// `vision_config`.
struct Gemma4VLVisionConfig {
    let depth: Int
    let hidden: Int
    let intermediate: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let patchSize: Int
    let rmsNormEps: Float
    let defaultOutputLength: Int
    let positionEmbeddingSize: Int
    let poolingKernelSize: Int
    let standardize: Bool
    let ropeTheta: Float

    static func decode(_ c: ModelConfig) throws -> Gemma4VLVisionConfig {
        guard let depth = c.int("num_hidden_layers"),
            let hidden = c.int("hidden_size"),
            let numHeads = c.int("num_attention_heads"),
            let patchSize = c.int("patch_size")
        else {
            throw Gemma4Error.missingConfig("vision_config")
        }
        let numKVHeads = c.int("num_key_value_heads") ?? numHeads
        let headDim = c.int("head_dim") ?? (hidden / numHeads)
        let intermediate = c.int("intermediate_size") ?? hidden * 4
        // The vision RoPE base frequency lives under `rope_parameters`;
        // default 100.0 per the Gemma 4 vision config.
        var ropeTheta: Float = 100.0
        if let rp = c.nested("rope_parameters") {
            if let t = rp["rope_theta"] as? Double {
                ropeTheta = Float(t)
            } else if let t = rp["rope_theta"] as? Int {
                ropeTheta = Float(t)
            }
        }
        return Gemma4VLVisionConfig(
            depth: depth, hidden: hidden, intermediate: intermediate,
            numHeads: numHeads, numKVHeads: numKVHeads, headDim: headDim,
            patchSize: patchSize,
            rmsNormEps: Float(c.float("rms_norm_eps") ?? 1e-6),
            defaultOutputLength: c.int("default_output_length") ?? 280,
            positionEmbeddingSize: c.int("position_embedding_size") ?? 10_240,
            poolingKernelSize: c.int("pooling_kernel_size") ?? 3,
            standardize: c.bool("standardize") ?? false,
            ropeTheta: ropeTheta)
    }
}

/// One Gemma 4 vision block: GemmaRMSNorm → RoPE MHA + q/k/v norms →
/// GemmaRMSNorm + residual, GemmaRMSNorm → SwiGLU MLP → GemmaRMSNorm +
/// residual. Held as plain weight tensors; the forward runs CPU
/// attention + GPU GEMMs.
final class Gemma4VLVisionBlock {
    /// The four per-block GemmaRMSNorms (the `+1` zero-shift is folded
    /// into the loaded weight).
    let inputNorm: RMSNorm
    let postAttnNorm: RMSNorm
    let preFFNorm: RMSNorm
    let postFFNorm: RMSNorm
    let qProj, kProj, vProj, oProj: Linear
    /// Per-head q / k RMSNorms (weighted) and the unweighted v-norm eps.
    let qNorm: RMSNorm
    let kNorm: RMSNorm
    let gate, up, down: Linear
    let cfg: Gemma4VLVisionConfig
    /// SwiGLU intermediate rounded up to the GEMM K-tile width.
    let paddedIntermediate: Int

    init(
        inputNorm: RMSNorm, postAttnNorm: RMSNorm, preFFNorm: RMSNorm,
        postFFNorm: RMSNorm, qProj: Linear, kProj: Linear, vProj: Linear,
        oProj: Linear, qNorm: RMSNorm, kNorm: RMSNorm,
        gate: Linear, up: Linear, down: Linear,
        paddedIntermediate: Int, cfg: Gemma4VLVisionConfig
    ) {
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
        self.preFFNorm = preFFNorm
        self.postFFNorm = postFFNorm
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.gate = gate
        self.up = up
        self.down = down
        self.paddedIntermediate = paddedIntermediate
        self.cfg = cfg
    }

    /// Forward `[nTokens, hidden]` activations through one block.
    /// `xPos` / `yPos` are the per-token grid coordinates driving the
    /// multi-dimensional vision RoPE.
    func forward(
        _ h: Tensor, nTokens: Int, xPos: [Int], yPos: [Int],
        device: Device
    ) -> Tensor {
        let hidden = cfg.hidden
        // ── Attention sub-block ──
        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNormRows(
            h, weight: inputNorm.weight,
            eps: inputNorm.eps, nRows: nTokens,
            rowSize: hidden, on: cmd)
        let q = projectRows(
            qProj, normed, nTokens: nTokens,
            outDim: cfg.numHeads * cfg.headDim, on: cmd)
        let k = projectRows(
            kProj, normed, nTokens: nTokens,
            outDim: cfg.numKVHeads * cfg.headDim, on: cmd)
        let v = projectRows(
            vProj, normed, nTokens: nTokens,
            outDim: cfg.numKVHeads * cfg.headDim, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let attn = cpuAttention(
            q: q, k: k, v: v, nTokens: nTokens,
            xPos: xPos, yPos: yPos, device: device)

        // ── Residual + MLP sub-block ──
        let cmd2 = device.makeCommandBuffer()
        let attnProj = projectRows(
            oProj, attn, nTokens: nTokens,
            outDim: hidden, on: cmd2)
        let postAttn = Ops.rmsNormRows(
            attnProj, weight: postAttnNorm.weight,
            eps: postAttnNorm.eps, nRows: nTokens,
            rowSize: hidden, on: cmd2)
        let afterAttn = Ops.add(h, postAttn, on: cmd2)
        let preFF = Ops.rmsNormRows(
            afterAttn, weight: preFFNorm.weight,
            eps: preFFNorm.eps, nRows: nTokens,
            rowSize: hidden, on: cmd2)
        let g = projectRows(
            gate, preFF, nTokens: nTokens,
            outDim: paddedIntermediate, on: cmd2)
        let u = projectRows(
            up, preFF, nTokens: nTokens,
            outDim: paddedIntermediate, on: cmd2)
        let act = Ops.gelu(g, on: cmd2)
        let gated = Ops.mul(act, u, on: cmd2)
        let d = projectRows(
            down, gated, nTokens: nTokens,
            outDim: hidden, on: cmd2)
        let postFF = Ops.rmsNormRows(
            d, weight: postFFNorm.weight,
            eps: postFFNorm.eps, nRows: nTokens,
            rowSize: hidden, on: cmd2)
        let result = Ops.add(afterAttn, postFF, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return result
    }

    /// CPU full bidirectional multi-head attention with per-head q / k
    /// RMSNorm + multi-dimensional RoPE. `q` / `k` / `v` are token-major
    /// `[nTokens, nHeads·headDim]` (q) / `[nTokens, nKVHeads·headDim]`
    /// (k, v). Returns the context token-major `[nTokens, nHeads·headDim]`.
    ///
    /// Two-stage parallelization with `DispatchQueue.concurrentPerform`:
    ///   1. Setup — per-token RMSNorm + M-RoPE: each token `t` writes to
    ///      its own `qH[t*nHeads+h]` / `kH[t*nKVHeads+h]` / `vH[t*nKVHeads+h]`
    ///      rows — disjoint across tokens, so race-free.
    ///   2. Attention — per-(head, query-row): each `(head, i)` pair writes
    ///      to a disjoint `[oBase, oBase + headDim)` output slice.
    private func cpuAttention(
        q: Tensor, k: Tensor, v: Tensor, nTokens: Int,
        xPos: [Int], yPos: [Int],
        device: Device
    ) -> Tensor {
        let nHeads = cfg.numHeads
        let nKVHeads = cfg.numKVHeads
        let headDim = cfg.headDim
        let qa = q.toFloatArray()
        let ka = k.toFloatArray()
        let va = v.toFloatArray()
        let qStride = nHeads * headDim
        let kvStride = nKVHeads * headDim
        // The vision attention uses scale 1.0 (no 1/√d) — see the
        // reference `gemma4EnsureFusedSDPA(..., scale: 1.0)`.
        let scale: Float = 1.0

        // Precompute RMSNorm / RoPE parameters so closures capture only
        // value types (no local funcs — avoids @Sendable capture warnings).
        let qWeight = qNorm.weight.toFloatArray()
        let kWeight = kNorm.weight.toFloatArray()
        let qNormEps = qNorm.eps
        let kNormEps = kNorm.eps
        let rmsNormEps = cfg.rmsNormEps
        let ropeTheta = cfg.ropeTheta
        // Multi-dimensional RoPE constants.
        let numDims = 2
        let chPerDim = 2 * (headDim / (2 * numDims))
        let halfPerDim = chPerDim / 2

        // Stage 1: Pre-norm + RoPE every q / k / v head.
        // Each token `t` writes only to rows `t*nHeads+h` / `t*nKVHeads+h`
        // of qH / kH / vH — disjoint across tokens, so no synchronization
        // is needed. Both the q-head loop and the kv-head loop are nested
        // inside a single per-token iteration to keep the index space flat.
        // `nonisolated(unsafe)`: each token writes disjoint slots.
        nonisolated(unsafe) var qH = [[Float]](repeating: [], count: nTokens * nHeads)
        nonisolated(unsafe) var kH = [[Float]](repeating: [], count: nTokens * nKVHeads)
        nonisolated(unsafe) var vH = [[Float]](repeating: [], count: nTokens * nKVHeads)
        DispatchQueue.concurrentPerform(iterations: nTokens) { t in
            // Inline RMSNorm: normalise an `[headDim]` slice in place,
            // optionally scaling by a per-dim weight vector.
            func applyRMSNorm(_ x: inout [Float], weight: [Float]?, eps: Float) {
                var ss: Float = 0
                for d in 0 ..< headDim { ss += x[d] * x[d] }
                let inv = 1.0 / (ss / Float(headDim) + eps).squareRoot()
                for d in 0 ..< headDim {
                    x[d] = x[d] * inv * (weight != nil ? weight![d] : 1.0)
                }
            }
            // Inline multi-dimensional RoPE: head dim splits into a per-axis
            // block (x then y); each block rotates by `position[axis]`.
            func applyRoPE(_ x: inout [Float], xp: Int, yp: Int) {
                for axis in 0 ..< numDims {
                    let start = axis * chPerDim
                    let pos = Float(axis == 0 ? xp : yp)
                    for i in 0 ..< halfPerDim {
                        let exponent = (2.0 / Float(chPerDim)) * Float(i)
                        let timescale = pow(ropeTheta, exponent)
                        let theta = pos / timescale
                        let c = cos(theta)
                        let s = sin(theta)
                        let a = x[start + i]
                        let b = x[start + halfPerDim + i]
                        // rotate-half: out_lo = a·c − b·s, out_hi = b·c + a·s.
                        x[start + i] = a * c - b * s
                        x[start + halfPerDim + i] = b * c + a * s
                    }
                }
            }
            for h in 0 ..< nHeads {
                var x = Array(qa[(t * qStride + h * headDim) ..< (t * qStride + (h + 1) * headDim)])
                applyRMSNorm(&x, weight: qWeight, eps: qNormEps)
                applyRoPE(&x, xp: xPos[t], yp: yPos[t])
                qH[t * nHeads + h] = x
            }
            for h in 0 ..< nKVHeads {
                var xk = Array(
                    ka[(t * kvStride + h * headDim) ..< (t * kvStride + (h + 1) * headDim)])
                applyRMSNorm(&xk, weight: kWeight, eps: kNormEps)
                applyRoPE(&xk, xp: xPos[t], yp: yPos[t])
                kH[t * nKVHeads + h] = xk
                var xv = Array(
                    va[(t * kvStride + h * headDim) ..< (t * kvStride + (h + 1) * headDim)])
                applyRMSNorm(&xv, weight: nil, eps: rmsNormEps)
                vH[t * nKVHeads + h] = xv
            }
        }

        // Stage 2: Full bidirectional attention, GQA head mapping. Pure
        // CPU per-(head, query) loop for correctness. The GPU path
        // (`Ops.sdpaBidirectional` d=64) was tried and pinned the GPU
        // at the GlmOcr production shape (nQuery=576, nQHeads=16) during
        // the 2026-05-27 bisect; Gemma 4 E2B/E4B's vision tower uses
        // the same d=64 kernel at similar-or-larger nTokens (4096+ at
        // 1024×1024 input) — same class of pin risk until the kernel
        // ships a paired GPU correctness test at production shape and
        // a fix to its inner-loop / bounds-select arithmetic.
        //
        // CPU dispatch is `DispatchQueue.concurrentPerform` over
        // (head · query) — same pattern Idefics3 / Mistral3 / FastVLM /
        // GlmOcr use. Slow at 4096+ tokens (the bisect timing the GPU
        // path was trying to fix), but safe: no machine reboot.
        let headsPerGroup = nHeads / nKVHeads
        let H = nHeads * headDim
        var outFlat = [Float](repeating: 0, count: nTokens * H)
        outFlat.withUnsafeMutableBufferPointer { outBuf in
            nonisolated(unsafe) let outPtr = outBuf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nHeads * nTokens) { work in
                let head = work / nTokens
                let i = work % nTokens
                let kvHead = head / headsPerGroup
                let qRow = qH[i * nHeads + head]
                var scores = [Float](repeating: 0, count: nTokens)
                var maxScore = -Float.greatestFiniteMagnitude
                for j in 0 ..< nTokens {
                    let kRow = kH[j * nKVHeads + kvHead]
                    var dot: Float = 0
                    for d in 0 ..< headDim { dot += qRow[d] * kRow[d] }
                    let s = dot * scale
                    scores[j] = s
                    if s > maxScore { maxScore = s }
                }
                var sumExp: Float = 0
                for j in 0 ..< nTokens {
                    let e = exp(scores[j] - maxScore)
                    scores[j] = e
                    sumExp += e
                }
                let invSum = 1.0 / sumExp
                let outBase = i * H + head * headDim
                for d in 0 ..< headDim { outPtr[outBase + d] = 0 }
                for j in 0 ..< nTokens {
                    let w = scores[j] * invSum
                    let vRow = vH[j * nKVHeads + kvHead]
                    for d in 0 ..< headDim {
                        outPtr[outBase + d] += w * vRow[d]
                    }
                }
            }
        }
        // Result is row-major [nTokens, nHeads·headDim] = [nTokens, qStride] —
        // byte-identical to what o_proj's GEMM consumes. Re-emit in the
        // input dtype.
        let result = Tensor.empty(
            shape: [nTokens, qStride], dtype: q.dtype,
            device: device)
        ImagePreprocessing.copyFloats(outFlat, into: result)
        return result
    }

    /// Apply a `Linear` to every row of `[nTokens, *]` via `Ops.gemm`,
    /// then broadcast-add the bias if present.
    private func projectRows(
        _ linear: Linear, _ x: Tensor, nTokens: Int,
        outDim: Int, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let y = Ops.gemm(weight: linear.weight, input: x, nRows: nTokens, on: cmd)
        guard let bias = linear.bias else { return y }
        return addRowBias(
            y, bias: bias, nRows: nTokens,
            rowSize: outDim, on: cmd)
    }
}

/// The Gemma 4 vision tower + multi-modal embedder. `encode` runs the
/// full forward and returns the pooled, projected soft tokens in the
/// text hidden dim.
final class Gemma4VLVisionModel: @unchecked Sendable {
    let cfg: Gemma4VLVisionConfig
    /// Patch-embed linear `[hidden, patchDimPadded]` (`3·patch·patch`
    /// columns, K-tile-padded).
    let patchEmbedWeight: Tensor
    let patchDimPadded: Int
    /// The two learned 2D position tables `[positionEmbeddingSize,
    /// hidden]` — index 0 is the x-axis, index 1 the y-axis.
    let posTableX: Tensor
    let posTableY: Tensor
    let blocks: [Gemma4VLVisionBlock]
    /// Optional standardization affine over the pooled tokens.
    let stdBias: Tensor?
    let stdScale: Tensor?
    /// Multi-modal embedder: GemmaRMSNorm (no-scale) + linear projection.
    /// Wrapped as `AnyLinear` so quantized checkpoints (4-bit e2b-it) work
    /// transparently — the weight is uint32-packed in those conversions.
    let embedderProjection: AnyLinear
    /// The no-scale norm's epsilon (the norm itself is unweighted).
    let embedderNormEps: Float
    let textHidden: Int
    let dtype: DType
    /// Patch grid side (square test image at the encoder's fixed size).
    let gridSide: Int
    /// Soft tokens one image contributes after pooling.
    let tokensPerImage: Int

    init(
        cfg: Gemma4VLVisionConfig, patchEmbedWeight: Tensor,
        patchDimPadded: Int, posTableX: Tensor, posTableY: Tensor,
        blocks: [Gemma4VLVisionBlock], stdBias: Tensor?, stdScale: Tensor?,
        embedderProjection: AnyLinear, embedderNormEps: Float,
        textHidden: Int, dtype: DType, gridSide: Int, tokensPerImage: Int
    ) {
        self.cfg = cfg
        self.patchEmbedWeight = patchEmbedWeight
        self.patchDimPadded = patchDimPadded
        self.posTableX = posTableX
        self.posTableY = posTableY
        self.blocks = blocks
        self.stdBias = stdBias
        self.stdScale = stdScale
        self.embedderProjection = embedderProjection
        self.embedderNormEps = embedderNormEps
        self.textHidden = textHidden
        self.dtype = dtype
        self.gridSide = gridSide
        self.tokensPerImage = tokensPerImage
    }

    static func load(
        visionConfig: ModelConfig, textHidden: Int,
        weights: SafeTensorsBundle, dtype: DType,
        quantization: ModelConfig.QuantizationConfig?,
        device: Device
    ) throws -> Gemma4VLVisionModel {
        let cfg = try Gemma4VLVisionConfig.decode(visionConfig)
        // The vision tower weights are namespaced under `vision_tower.`
        // (after `model.` stripping); the multi-modal embedder under
        // `embed_vision.`. Probe both possible prefixes.
        let vt =
            weights.has("vision_tower.patch_embedder.input_proj.weight")
            ? weights.prefixed("vision_tower.")
            : weights.prefixed("model.vision_tower.")
        let ev =
            weights.has("embed_vision.embedding_projection.weight")
            ? weights.prefixed("embed_vision.")
            : weights.prefixed("model.embed_vision.")

        // ── Patch-embed ──
        // `input_proj.weight` is `[hidden, 3·patch·patch]`; pad the
        // `inDim` to the GEMM K-tile width.
        let rawPatch = try vt.tensor(named: "patch_embedder.input_proj.weight")
        let patchDim = 3 * cfg.patchSize * cfg.patchSize
        let patchDimPadded =
            ((patchDim + gemmKTileWidth - 1) / gemmKTileWidth) * gemmKTileWidth
        let patchEmbedWeight = padLinearColsTo(
            rawPatch, toCols: patchDimPadded, device: device)

        // ── Position embedding tables ──
        // `position_embedding_table` is `[2, positionEmbeddingSize,
        // hidden]`; split into the per-axis tables.
        let posTableRaw = try vt.tensor(
            named: "patch_embedder.position_embedding_table")
        let (posX, posY) = splitPositionTable(posTableRaw, device: device)

        // ── Block stack ──
        let paddedIntermediate =
            ((cfg.intermediate + gemmKTileWidth - 1) / gemmKTileWidth)
            * gemmKTileWidth
        var blocks: [Gemma4VLVisionBlock] = []
        blocks.reserveCapacity(cfg.depth)
        for i in 0 ..< cfg.depth {
            let p = "encoder.layers.\(i)"
            func lin(_ name: String) throws -> Linear {
                // Gemma 4 vision linears are wrapped in a clippable
                // module — the projection weight is `.linear.weight`.
                // The leading `try` covers the whole `??` — the RHS
                // `vt.tensor(...)` throws, so the operator expression
                // must be marked `try` at its start.
                let w =
                    try (try? vt.tensor(named: "\(p).\(name).linear.weight"))
                    ?? vt.tensor(named: "\(p).\(name).weight")
                let b =
                    (try? vt.tensor(named: "\(p).\(name).linear.bias"))
                    ?? (try? vt.tensor(named: "\(p).\(name).bias"))
                return Linear(weight: w, bias: b)
            }
            // The four per-block norms are GemmaRMSNorm — fold the +1.
            func gemmaNorm(_ name: String) throws -> RMSNorm {
                let raw = try vt.tensor(named: "\(p).\(name).weight")
                return RMSNorm(
                    weight: foldGemmaRMSNormWeight(raw),
                    eps: cfg.rmsNormEps)
            }
            // The q / k head norms are plain (weighted, no +1) RMSNorms.
            func headNorm(_ name: String) throws -> RMSNorm {
                RMSNorm(
                    weight: try vt.tensor(named: "\(p).\(name).weight"),
                    eps: cfg.rmsNormEps)
            }
            let gate = try lin("mlp.gate_proj")
            let up = try lin("mlp.up_proj")
            let down = try lin("mlp.down_proj")
            blocks.append(
                Gemma4VLVisionBlock(
                    inputNorm: try gemmaNorm("input_layernorm"),
                    postAttnNorm: try gemmaNorm("post_attention_layernorm"),
                    preFFNorm: try gemmaNorm("pre_feedforward_layernorm"),
                    postFFNorm: try gemmaNorm("post_feedforward_layernorm"),
                    qProj: try lin("self_attn.q_proj"),
                    kProj: try lin("self_attn.k_proj"),
                    vProj: try lin("self_attn.v_proj"),
                    oProj: try lin("self_attn.o_proj"),
                    qNorm: try headNorm("self_attn.q_norm"),
                    kNorm: try headNorm("self_attn.k_norm"),
                    gate: padLinearRows(gate, toRows: paddedIntermediate, device: device),
                    up: padLinearRows(up, toRows: paddedIntermediate, device: device),
                    down: Linear(
                        weight: padLinearColsTo(
                            down.weight, toCols: paddedIntermediate,
                            device: device),
                        bias: down.bias),
                    paddedIntermediate: paddedIntermediate, cfg: cfg))
        }

        // ── Standardization affine (optional) ──
        let stdBias = try? vt.tensor(named: "std_bias")
        let stdScale = try? vt.tensor(named: "std_scale")

        // ── Multi-modal embedder ──
        // `embedding_projection` may be quantized in 4-bit checkpoints (e.g.
        // `gemma-4-e2b-it-4bit` — weight dtype is U32). Load through
        // `loadLinear` so a `QuantizedLinear` is returned for those cases
        // instead of a plain `Linear` with packed-uint32 data, which would
        // trip `Ops.gemm`'s element-count precondition (packed column count
        // 8× less than the actual input width).
        // `embedding_pre_projection_norm` is a GemmaRMSNorm-no-scale —
        // unweighted; only the eps matters.
        let embedderProjection = try loadLinear(
            base: "embedding_projection", in: ev, quantization: quantization)

        // Size the test grid so the patch count pools cleanly into the
        // soft-token grid. `defaultOutputLength` is the soft-token count;
        // pick a grid whose side, divided by the pooling kernel, gives a
        // near-square soft-token grid.
        let softSide = Int(Double(cfg.defaultOutputLength).squareRoot())
        let tokensPerImage = softSide * softSide
        let gridSide = softSide * cfg.poolingKernelSize

        return Gemma4VLVisionModel(
            cfg: cfg, patchEmbedWeight: patchEmbedWeight,
            patchDimPadded: patchDimPadded, posTableX: posX, posTableY: posY,
            blocks: blocks, stdBias: stdBias, stdScale: stdScale,
            embedderProjection: embedderProjection,
            embedderNormEps: cfg.rmsNormEps, textHidden: textHidden,
            dtype: dtype, gridSide: gridSide, tokensPerImage: tokensPerImage)
    }

    /// Run the full vision forward on a preprocessed image. `image` is a
    /// normalized NCHW tensor `[1, 3, side, side]` where `side =
    /// gridSide · patchSize`. Returns `[tokensPerImage, textHidden]`.
    func encode(image: Tensor, device: Device) -> Tensor {
        let side = gridSide
        let nPatches = side * side

        // ── Patch unfold + embed ──
        let unfolded = unfoldPatches(image: image)
        let cmd = device.makeCommandBuffer()
        var h = Ops.gemm(
            weight: patchEmbedWeight, input: unfolded,
            nRows: nPatches, on: cmd)
        // Add the learned 2D position embedding (x + y axis lookups).
        let posEmb = positionEmbedding()
        h = Ops.add(h, posEmb, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── Per-patch grid coordinates ──
        var xPos = [Int](repeating: 0, count: nPatches)
        var yPos = [Int](repeating: 0, count: nPatches)
        for y in 0 ..< side {
            for x in 0 ..< side {
                let idx = y * side + x
                xPos[idx] = x
                yPos[idx] = y
            }
        }

        // ── Block stack ──
        for block in blocks {
            h = block.forward(
                h, nTokens: nPatches, xPos: xPos, yPos: yPos,
                device: device)
        }

        // ── Attention pooling → soft tokens, then multi-modal embed ──
        let pooled = pool(h, device: device)
        return embed(pooled, device: device)
    }

    /// Present the tower as a `VisionEncoder` so `VisionModel` accepts it.
    func asVisionEncoder() -> VisionEncoder {
        Gemma4VLComposedEncoder(tower: self)
    }

    // ── Patch unfold ──

    /// Unfold a normalized NCHW image into a `[nPatches, patchDimPadded]`
    /// tensor. Each row holds one `3 · patch · patch` patch after the
    /// reference's `2·(x − 0.5)` re-centring; row order is raster
    /// `(patchRow, patchCol)`.
    private func unfoldPatches(image: Tensor) -> Tensor {
        let side = gridSide
        let p = cfg.patchSize
        let pix = image.toFloatArray()  // [3, H, W]
        let imgSide = side * p
        var rows = [Float](repeating: 0, count: side * side * patchDimPadded)
        for pr in 0 ..< side {
            for pc in 0 ..< side {
                let patch = pr * side + pc
                var col = 0
                // Layout: (channel, py, px) — matches the reference's
                // `(0,2,4,3,5,1)` transpose then channel-major flatten.
                for py in 0 ..< p {
                    let yy = pr * p + py
                    for px in 0 ..< p {
                        let xx = pc * p + px
                        for ch in 0 ..< 3 {
                            let v = pix[(ch * imgSide + yy) * imgSide + xx]
                            // Re-centre: 2·(v − 0.5).
                            rows[patch * patchDimPadded + col] = 2 * (v - 0.5)
                            col += 1
                        }
                    }
                }
            }
        }
        return ImagePreprocessing.makeTensor(
            from: rows, shape: [side * side, patchDimPadded],
            dtype: dtype, device: .shared)
    }

    // ── Position embedding ──

    /// Build the per-patch learned 2D position embedding `[nPatches,
    /// hidden]` — the x-axis table indexed by the patch column plus the
    /// y-axis table indexed by the patch row.
    private func positionEmbedding() -> Tensor {
        let side = gridSide
        let hidden = cfg.hidden
        let tableX = posTableX.toFloatArray()
        let tableY = posTableY.toFloatArray()
        let posSize = cfg.positionEmbeddingSize
        var dst = [Float](repeating: 0, count: side * side * hidden)
        for pr in 0 ..< side {
            for pc in 0 ..< side {
                let patch = pr * side + pc
                let xi = min(posSize - 1, pc)
                let yi = min(posSize - 1, pr)
                for d in 0 ..< hidden {
                    dst[patch * hidden + d] =
                        tableX[xi * hidden + d] + tableY[yi * hidden + d]
                }
            }
        }
        return ImagePreprocessing.makeTensor(
            from: dst, shape: [side * side, hidden], dtype: dtype,
            device: .shared)
    }

    // ── Attention pooling ──

    /// Pool the `gridSide × gridSide` patch tokens down to
    /// `tokensPerImage` soft tokens by averaging each
    /// `poolingKernelSize × poolingKernelSize` neighbourhood, then scale
    /// by `sqrt(hidden)` and apply the optional standardization affine.
    /// Returns `[tokensPerImage, hidden]`.
    private func pool(_ h: Tensor, device: Device) -> Tensor {
        let side = gridSide
        let k = cfg.poolingKernelSize
        let softSide = side / k
        let hidden = cfg.hidden
        let src = h.toFloatArray()
        let kArea = Float(k * k)
        let rootHidden = Float(Double(hidden).squareRoot())
        var pooled = [Float](repeating: 0, count: softSide * softSide * hidden)
        let stdB = stdBias?.toFloatArray()
        let stdS = stdScale?.toFloatArray()
        for sy in 0 ..< softSide {
            for sx in 0 ..< softSide {
                let outBase = (sy * softSide + sx) * hidden
                for c in 0 ..< hidden {
                    var acc: Float = 0
                    for dy in 0 ..< k {
                        for dx in 0 ..< k {
                            let py = sy * k + dy
                            let px = sx * k + dx
                            acc += src[(py * side + px) * hidden + c]
                        }
                    }
                    var val = (acc / kArea) * rootHidden
                    if let stdB, let stdS {
                        val = (val - stdB[c]) * stdS[c]
                    }
                    pooled[outBase + c] = val
                }
            }
        }
        let out = Tensor.empty(
            shape: [softSide * softSide, hidden],
            dtype: dtype, device: device)
        ImagePreprocessing.copyFloats(pooled, into: out)
        return out
    }

    // ── Multi-modal embedder ──

    /// GemmaRMSNorm (no-scale) each pooled soft token, then project into
    /// the text hidden dim. Returns `[tokensPerImage, textHidden]`.
    private func embed(_ pooled: Tensor, device: Device) -> Tensor {
        let nTokens = pooled.shape[0]
        let hidden = cfg.hidden
        let src = pooled.toFloatArray()
        // Unweighted RMSNorm — the embedder's pre-projection norm has no
        // learned scale.
        var normed = [Float](repeating: 0, count: nTokens * hidden)
        for t in 0 ..< nTokens {
            var ss: Float = 0
            for d in 0 ..< hidden { ss += src[t * hidden + d] * src[t * hidden + d] }
            let inv = 1.0 / (ss / Float(hidden) + embedderNormEps).squareRoot()
            for d in 0 ..< hidden { normed[t * hidden + d] = src[t * hidden + d] * inv }
        }
        let normedT = Tensor.empty(
            shape: [nTokens, hidden], dtype: dtype,
            device: device)
        ImagePreprocessing.copyFloats(normed, into: normedT)

        let cmd = device.makeCommandBuffer()
        // Use `callMany` so quantized weights (QuantizedLinear from a 4-bit
        // checkpoint) are handled correctly via the dequant-gemm path.
        let projected = embedderProjection.callMany(
            normedT, t: nTokens, on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        return projected
    }

    // ── Static helpers ──

    /// Split a `[2, posSize, hidden]` position table into the per-axis
    /// `[posSize, hidden]` tables.
    static func splitPositionTable(_ raw: Tensor, device: Device)
        -> (x: Tensor, y: Tensor)
    {
        precondition(
            raw.shape.count == 3 && raw.shape[0] == 2,
            "Gemma4VL: position table must be [2, posSize, hidden], "
                + "got \(raw.shape)")
        let posSize = raw.shape[1]
        let hidden = raw.shape[2]
        let src = raw.toFloatArray()
        var x = [Float](repeating: 0, count: posSize * hidden)
        var y = [Float](repeating: 0, count: posSize * hidden)
        let plane = posSize * hidden
        for i in 0 ..< plane {
            x[i] = src[i]
            y[i] = src[plane + i]
        }
        let xT = ImagePreprocessing.makeTensor(
            from: x, shape: [posSize, hidden], dtype: raw.dtype, device: device)
        let yT = ImagePreprocessing.makeTensor(
            from: y, shape: [posSize, hidden], dtype: raw.dtype, device: device)
        return (xT, yT)
    }

    // Shared `addRowBias`, `padLinearRows`, and `padLinearColsTo`
    // helpers live in `VisionTowerOps.swift`.
}

/// A `VisionEncoder` subclass whose `encode` runs the Gemma 4 vision
/// tower + multi-modal embedder — so `VisionModel` (which holds a
/// `VisionEncoder`) transparently gets the pooled, projected soft
/// tokens.
final class Gemma4VLComposedEncoder: VisionEncoder {
    let tower: Gemma4VLVisionModel

    init(tower: Gemma4VLVisionModel) {
        self.tower = tower
        let c = tower.cfg
        let side = tower.gridSide * c.patchSize
        let facadeConfig = VisionEncoderConfig(
            inChannels: 3, imageSize: side,
            patchSize: side / Int(Double(tower.tokensPerImage).squareRoot()),
            hidden: c.hidden, intermediate: c.intermediate,
            nLayers: c.depth, nHeads: c.numHeads,
            layerNormEps: c.rmsNormEps, textHidden: tower.textHidden)
        let placeholderW = tower.patchEmbedWeight
        let placeholderNorm = LayerNorm(
            weight: placeholderW, bias: placeholderW, eps: c.rmsNormEps)
        super.init(
            config: facadeConfig,
            patchEmbedWeight: placeholderW, patchEmbedBias: placeholderW,
            positionEmbedding: placeholderW, layers: [],
            postLayerNorm: placeholderNorm,
            projection: nil, dtype: tower.dtype)
    }

    /// Run the Gemma 4 vision tower. Returns
    /// `[tokensPerImage, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        tower.encode(image: image, device: device)
    }
}

// ─── Gemma 4 VL family orchestrator ──────────────────────────────────
//
// Composes the Gemma 4 vision tower above with the Gemma 4 text
// backbone (`Gemma4Dense` / `Gemma4E` / `Gemma4MoE` in
// `Models/Text/Gemma4Text.swift`) for the
// `Gemma4ForConditionalGeneration` checkpoints.
//
// Composition:
//   • Gemma 4 vision tower — a bespoke ViT that differs substantially
//     from the shared SigLIP `VisionEncoder`:
//       – patch-embed via a flattened linear projection (each
//         `3 · patch · patch` patch row is projected by one GEMM, after
//         a `2·(x − 0.5)` re-centring),
//       – a learned 2D position embedding: two `[positionEmbeddingSize,
//         hidden]` tables (one per spatial axis) summed by the patch's
//         `(x, y)` grid coordinate,
//       – RoPE attention with multi-dimensional positions (the head dim
//         is split into an x-rotary half and a y-rotary half) and
//         per-projection q / k / v RMSNorms,
//       – four GemmaRMSNorm "zero-shift" norms per block (input /
//         post-attention / pre-feedforward / post-feedforward) and a
//         GELU-gated SwiGLU MLP,
//       – an attention-pooling head that pools the patch grid down to
//         `default_output_length` soft tokens, then a `sqrt(hidden)`
//         scale and an optional standardization affine.
//   • Multi-modal embedder — a GemmaRMSNorm (no-scale) over the pooled
//     vision tokens followed by a linear projection into the text
//     hidden dim (`embed_vision.*`).
//   • Gemma 4 text backbone — the existing `Gemma4Model` engine, loaded
//     from the `language_model.`-prefixed sub-tree (the `Gemma4Loader`
//     probes the prefix and reads `text_config` itself).
//
// The three are joined by `VisionModel`'s cross-modal token splice: each
// `<image_soft_token>` placeholder (`image_token_id`) in the prompt
// takes one of the pooled, projected vision tokens.

public enum Gemma4VL {
    /// `image_token_id` default for Gemma 4 VL checkpoints.
    public static let defaultImageTokenId = 262_144

    /// Build a `VisionModel` from a `Gemma4ForConditionalGeneration`
    /// checkpoint: the Gemma 4 vision tower + multi-modal embedder +
    /// Gemma 4 text backbone, joined by the cross-modal splice.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> VisionModel {
        guard let visionConfig = config.subConfig("vision_config") else {
            throw Gemma4Error.missingConfig("vision_config")
        }

        // ── Text backbone — Gemma 4 engine ──
        // `Gemma4Loader` probes the weight prefix (`language_model.model.`
        // for a VLM conversion) and reads the text hyper-parameters from
        // the `text_config` sub-tree itself, so the whole `config` and
        // `weights` bundle is forwarded unchanged; the variant (dense /
        // E / MoE) is config-driven.
        let textVariant = try Gemma4.variant(for: config)
        let textEngine = try textVariant.loadModel(
            config: config, weights: weights, options: options, device: device)

        // ── Vision tower + multi-modal embedder ──
        let vision = try Gemma4VLVisionModel.load(
            visionConfig: visionConfig, textHidden: textEngine.hidden,
            weights: weights, dtype: textEngine.dtype,
            quantization: config.quantization, device: device)

        let imageTokenId =
            config.int("image_token_id")
            ?? config.int("image_token_index") ?? defaultImageTokenId
        return try VisionModel(
            visionEncoder: vision.asVisionEncoder(),
            engine: textEngine, imageTokenId: imageTokenId,
            normalization: .siglip,
            imageTokenCount: vision.tokensPerImage)
    }
}
