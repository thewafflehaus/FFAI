// Pixtral vision tower internals — config, 2D RoPE, attention blocks,
// encoder, projector, and composed tower.
//
// The family orchestrator (`enum PixtralError`, `enum Pixtral`, `load(...)`)
// lives in `Models/Pixtral.swift`. This file contains the private / internal
// types that implement the vision side:
//   • pixtralTextConfigWithDefaults — shared helper also used by Mistral3's
//     orchestrator to fill sparse VLM text_config from Mistral defaults.
//   • PixtralVisionConfig — decoded from `vision_config`.
//   • PixtralRoPE — precomputed 2D rotary position embedding tables.
//   • PixtralVisionBlock — one ViT block (RMSNorm + 2D-RoPE MHA + SiLU MLP).
//   • PixtralVisionEncoder — patch embed conv + block stack.
//   • PixtralProjector — two-layer GELU MLP projecting vision → text dim.
//   • PixtralComposedTower / PixtralComposedEncoder — single VisionEncoder
//     surface coupling the ViT with the projector for VLModel splice.

import Foundation
import Metal

/// Merge Pixtral text-model defaults into the VLM's sparse `text_config`.
/// The mlx-community VLM text_config only stores fields that differ from
/// the class defaults; the standalone `LlamaDense` loader needs the full
/// set for Mistral (especially `rms_norm_eps`, `rope_theta`, etc.).
func pixtralTextConfigWithDefaults(
    _ raw: [String: Any], vocabFallback: Int?
) -> [String: Any] {
    // Mistral-7B / Mistral-Small / Pixtral-12B text defaults.
    var merged: [String: Any] = [
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocabFallback ?? 131_072,
        "rope_theta": 1_000_000_000.0,
        "rope_traditional": false,
        "max_position_embeddings": 131_072,
        "tie_word_embeddings": false,
    ]
    // Checkpoint-declared fields override the defaults.
    for (k, v) in raw { merged[k] = v }
    return merged
}

// ─── Vision configuration ────────────────────────────────────────────

/// Static shape of the Pixtral vision tower, decoded from `vision_config`.
public struct PixtralVisionConfig {
    /// Number of ViT encoder blocks.
    let numLayers: Int
    /// Encoder hidden dimension.
    let hiddenSize: Int
    /// Per-head dimension (explicit in the config, defaults to 64).
    let headDim: Int
    /// SwiGLU MLP intermediate dimension.
    let intermediateSize: Int
    /// Number of attention heads.
    let numHeads: Int
    /// Square input image size the encoder expects (default 336 px).
    let imageSize: Int
    /// Patch side (default 14 px).
    let patchSize: Int
    /// Number of input channels (default 3).
    let numChannels: Int
    /// RMSNorm epsilon (default 1e-5).
    let rmsNormEps: Float
    /// RoPE base frequency (default 10_000).
    let ropeTheta: Float

    /// Patches along one axis.
    var patchesPerSide: Int { imageSize / patchSize }
    /// Total patch tokens an image produces.
    var numPatches: Int { patchesPerSide * patchesPerSide }

    static func decode(_ c: ModelConfig) throws -> PixtralVisionConfig {
        guard let numLayers = c.int("num_hidden_layers"),
              let hiddenSize = c.int("hidden_size"),
              let intermediateSize = c.int("intermediate_size"),
              let numHeads = c.int("num_attention_heads"),
              let patchSize = c.int("patch_size")
        else {
            throw PixtralError.missingConfig
        }
        let headDim = c.int("head_dim") ?? (hiddenSize / numHeads)
        return PixtralVisionConfig(
            numLayers: numLayers,
            hiddenSize: hiddenSize,
            headDim: headDim,
            intermediateSize: intermediateSize,
            numHeads: numHeads,
            imageSize: c.int("image_size") ?? 336,
            patchSize: patchSize,
            numChannels: c.int("num_channels") ?? 3,
            rmsNormEps: Float(c.float("rms_norm_eps") ?? 1e-5),
            ropeTheta: Float(c.float("rope_theta") ?? 10_000.0))
    }
}

// ─── 2D RoPE ─────────────────────────────────────────────────────────

/// Precomputed per-patch 2D rotary position embeddings for the Pixtral
/// vision tower.
///
/// Layout: given head-dim D, each D-dimensional RoPE vector is split
/// into two halves of size D/2:
///   • [0, D/2)   — driven by the patch row position (height RoPE)
///   • [D/2, D)   — driven by the patch column position (width RoPE)
///
/// Within each D/2 half, the standard rotate-half scheme is used with
/// `D/4` distinct frequency bands (each band spans 2 positions in the
/// half, duplicated for the cosine/sine pair). The base frequencies
/// alternate between even and odd indices of the full frequency set so
/// height and width use interleaved (not consecutive) bands, matching
/// the reference Python implementation.
///
/// `invFreqTable[patch, d]` gives the angle for position `d` of patch
/// `patch`. The `cos` and `sin` are precomputed for the default
/// `maxPatchesPerSide = imageSize / patchSize` square grid; during
/// forward, position ids index into the table to handle variable
/// image sizes.
final class PixtralRoPE {
    /// `[maxPatches², headDim]` precomputed inv_freq * position, stored as
    /// `[maxPatches², headDim]` — cos and sin arrays.
    let cosTable: [Float]   // [maxPositions, headDim]
    let sinTable: [Float]
    let maxPatchesPerSide: Int
    let headDim: Int

    init(cfg: PixtralVisionConfig) {
        let D = cfg.headDim
        let half = D / 2
        let quarter = half / 2  // number of distinct frequency bands per spatial axis
        let maxP = cfg.imageSize / cfg.patchSize
        self.maxPatchesPerSide = maxP
        self.headDim = D

        // inv_freq over D/2 positions: `1 / theta^(2i / D)` for i in 0..<D/2.
        // (D/2 not D because each half covers one spatial axis.)
        var invFreq = [Float](repeating: 0, count: half)
        for i in 0..<half {
            invFreq[i] = 1.0 / pow(cfg.ropeTheta, Float(2 * i) / Float(D))
        }

        // Pixtral interleaves the inv_freq across height and width halves:
        //   height uses even indices of inv_freq: [0, 2, 4, ..., D/2-2]
        //   width  uses odd  indices of inv_freq: [1, 3, 5, ..., D/2-1]
        // (Following the reference `RotaryEmbedding.__init__` in Pixtral.swift
        //  from mlx-swift-lm — the `indicesEven` / `indicesOdd` split.)
        var freqsH = [Float](repeating: 0, count: quarter)
        var freqsW = [Float](repeating: 0, count: quarter)
        for i in 0..<quarter {
            freqsH[i] = invFreq[2 * i]      // even indices → height
            freqsW[i] = invFreq[2 * i + 1]  // odd  indices → width
        }

        // Build per-position frequency tables: `[maxP, quarter]`.
        // freqsHPos[row, i] = row * freqsH[i]
        // freqsWPos[col, i] = col * freqsW[i]
        var freqsHPos = [Float](repeating: 0, count: maxP * quarter)
        var freqsWPos = [Float](repeating: 0, count: maxP * quarter)
        for pos in 0..<maxP {
            for i in 0..<quarter {
                freqsHPos[pos * quarter + i] = Float(pos) * freqsH[i]
                freqsWPos[pos * quarter + i] = Float(pos) * freqsW[i]
            }
        }

        // Build the per-patch (row, col) → [headDim] cos/sin tables.
        // For patch at grid (row, col), position id = row * maxP + col.
        // The [headDim] layout:
        //   [0..quarter)           freqsHPos[row, 0..quarter)   height band 1
        //   [quarter..half)        freqsHPos[row, 0..quarter)   height band 2 (duplicate for rotate-half)
        //   [half..half+quarter)   freqsWPos[col, 0..quarter)   width  band 1
        //   [half+quarter..D)      freqsWPos[col, 0..quarter)   width  band 2 (duplicate for rotate-half)
        let totalPositions = maxP * maxP
        var cosT = [Float](repeating: 0, count: totalPositions * D)
        var sinT = [Float](repeating: 0, count: totalPositions * D)
        for row in 0..<maxP {
            for col in 0..<maxP {
                let patchIdx = row * maxP + col
                let base = patchIdx * D
                for i in 0..<quarter {
                    let fh = freqsHPos[row * quarter + i]
                    let fw = freqsWPos[col * quarter + i]
                    // Height half: band1 [i] and band2 [i + quarter]
                    cosT[base + i]          = cos(fh)
                    sinT[base + i]          = sin(fh)
                    cosT[base + i + quarter] = cos(fh)
                    sinT[base + i + quarter] = sin(fh)
                    // Width half: band1 [half + i] and band2 [half + i + quarter]
                    cosT[base + half + i]           = cos(fw)
                    sinT[base + half + i]           = sin(fw)
                    cosT[base + half + i + quarter] = cos(fw)
                    sinT[base + half + i + quarter] = sin(fw)
                }
            }
        }
        self.cosTable = cosT
        self.sinTable = sinT
    }

    /// Retrieve the [D] cos and sin vectors for a patch at grid position
    /// `(row, col)`.
    func cosSin(row: Int, col: Int) -> (cos: ArraySlice<Float>, sin: ArraySlice<Float>) {
        let idx = (row * maxPatchesPerSide + col) * headDim
        return (cosTable[idx..<(idx + headDim)],
                sinTable[idx..<(idx + headDim)])
    }
}

// ─── Vision attention ─────────────────────────────────────────────────

/// One Pixtral vision encoder block: RMSNorm → 2D-RoPE MHA → residual,
/// then RMSNorm → SiLU-gated MLP → residual. Held as plain weight
/// tensors; the forward runs CPU attention + GPU GEMMs.
final class PixtralVisionBlock {
    /// Attention sub-block.
    let attNorm: RMSNorm
    let qProj: Linear    // [hidden, hidden] no bias
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear
    /// Feed-forward sub-block.
    let ffnNorm: RMSNorm
    let gateProj: Linear // [intermediate, hidden] no bias
    let upProj: Linear
    let downProj: Linear
    let cfg: PixtralVisionConfig

    init(attNorm: RMSNorm, qProj: Linear, kProj: Linear, vProj: Linear,
         oProj: Linear, ffnNorm: RMSNorm, gateProj: Linear, upProj: Linear,
         downProj: Linear, cfg: PixtralVisionConfig) {
        self.attNorm = attNorm
        self.qProj = qProj; self.kProj = kProj
        self.vProj = vProj; self.oProj = oProj
        self.ffnNorm = ffnNorm
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        self.cfg = cfg
    }

    /// Forward `[nPatches, hidden]` activations through one block.
    /// `rope` provides the 2D RoPE tables; `gridSide` is `√nPatches`
    /// (the patch grid is square).
    func forward(_ h: Tensor, nPatches: Int, gridSide: Int,
                 rope: PixtralRoPE, device: Device) -> Tensor {
        let hidden = cfg.hiddenSize

        // ── Attention sub-block ──
        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNormRows(h, weight: attNorm.weight, eps: attNorm.eps,
                                     nRows: nPatches, rowSize: hidden, on: cmd)
        let q = Ops.gemm(weight: qProj.weight, input: normed, nRows: nPatches, on: cmd)
        let k = Ops.gemm(weight: kProj.weight, input: normed, nRows: nPatches, on: cmd)
        let v = Ops.gemm(weight: vProj.weight, input: normed, nRows: nPatches, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // CPU 2D-RoPE + bidirectional multi-head attention.
        let attnOut = cpuAttention(q: q, k: k, v: v, nPatches: nPatches,
                                   gridSide: gridSide, rope: rope, device: device)

        // Output projection + residual.
        let cmd2 = device.makeCommandBuffer()
        let attnProj = Ops.gemm(weight: oProj.weight, input: attnOut,
                                nRows: nPatches, on: cmd2)
        let postAttn = Ops.add(h, attnProj, on: cmd2)

        // ── Feed-forward sub-block ──
        let normed2 = Ops.rmsNormRows(postAttn, weight: ffnNorm.weight,
                                      eps: ffnNorm.eps, nRows: nPatches,
                                      rowSize: hidden, on: cmd2)
        let gate = Ops.gemm(weight: gateProj.weight, input: normed2,
                            nRows: nPatches, on: cmd2)
        let up   = Ops.gemm(weight: upProj.weight, input: normed2,
                            nRows: nPatches, on: cmd2)
        let activated = Ops.silu(gate, on: cmd2)
        let gated = Ops.mul(activated, up, on: cmd2)
        let ffnOut = Ops.gemm(weight: downProj.weight, input: gated,
                              nRows: nPatches, on: cmd2)
        let result = Ops.add(postAttn, ffnOut, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return result
    }

    /// Bidirectional multi-head attention with 2D RoPE applied to Q
    /// and K. `q`, `k`, `v` are token-major `[nPatches, hidden]`. Returns
    /// the context, token-major `[nPatches, hidden]`.
    ///
    /// Stage 1 (CPU) extracts per-(head, patch) slices and applies 2D
    /// RoPE — the (head, patch) outer product is embarrassingly parallel
    /// and the writes are disjoint, so `concurrentPerform` is safe.
    /// Stage 2 (GPU) is one `Ops.sdpaBidirectional(headDim: 64)`
    /// dispatch — Pixtral SigLIP uses 1024/16 = 64.
    private func cpuAttention(q: Tensor, k: Tensor, v: Tensor,
                              nPatches: Int, gridSide: Int,
                              rope: PixtralRoPE, device: Device) -> Tensor {
        let nHeads = cfg.numHeads
        let headDim = cfg.headDim
        let hidden = cfg.hiddenSize
        let scale = 1.0 / Float(Double(headDim).squareRoot())
        let half = headDim / 2

        let qa = q.toFloatArray()
        let ka = k.toFloatArray()
        let va = v.toFloatArray()

        // Stage 1: extract per-head Q/K/V slices and apply 2D RoPE in
        // place. Layout target:
        //   Q  → [nPatches, nHeads, headDim] (kernel Q contract)
        //   KV → [nHeads, nPatches, headDim] (kernel K/V contract)
        // Each (head, patch) writes to its own disjoint slot — race-free
        // across concurrent iterations.
        // Per-head slice arrays — each (head, patch) writes to its own
        // disjoint element, so concurrent writes are race-free.
        var qH = [[Float]](repeating: [], count: nHeads * nPatches)
        var kH = [[Float]](repeating: [], count: nHeads * nPatches)
        var vH = [[Float]](repeating: [], count: nHeads * nPatches)

        DispatchQueue.concurrentPerform(iterations: nHeads * nPatches) { work in
            let head = work / nPatches
            let patch = work % nPatches
            let row = patch / gridSide
            let col = patch % gridSide
            let hOff = head * headDim
            let base = patch * hidden

            var qSlice = Array(qa[(base + hOff)..<(base + hOff + headDim)])
            var kSlice = Array(ka[(base + hOff)..<(base + hOff + headDim)])
            let vSlice = Array(va[(base + hOff)..<(base + hOff + headDim)])

            // Apply 2D RoPE to Q and K using the rotate-half scheme.
            // The [headDim] position encoding has height in [0, half) and
            // width in [half, D). rotate-half: out[d] = x[d]*cos - x[(d+half)%D]*sin
            // for the first half and x[d]*cos + x[d-half]*sin for the second.
            let (cosSlice, sinSlice) = rope.cosSin(row: row, col: col)
            applyRoPE2D(&qSlice, cos: cosSlice, sin: sinSlice, half: half)
            applyRoPE2D(&kSlice, cos: cosSlice, sin: sinSlice, half: half)

            qH[head * nPatches + patch] = qSlice
            kH[head * nPatches + patch] = kSlice
            vH[head * nPatches + patch] = vSlice
        }

        // Assemble kernel-layout flat buffers:
        //   Q  → [nPatches, nHeads, headDim]
        //   KV → [nHeads, nPatches, headDim]
        var qFlat = [Float](repeating: 0, count: nPatches * hidden)
        var kFlat = [Float](repeating: 0, count: nHeads * nPatches * headDim)
        var vFlat = [Float](repeating: 0, count: nHeads * nPatches * headDim)
        for head in 0..<nHeads {
            let hOff = head * headDim
            for patch in 0..<nPatches {
                let src = qH[head * nPatches + patch]
                let qDst = patch * hidden + hOff
                for d in 0..<headDim { qFlat[qDst + d] = src[d] }
                let kvDst = (head * nPatches + patch) * headDim
                let kSrc = kH[head * nPatches + patch]
                let vSrc = vH[head * nPatches + patch]
                for d in 0..<headDim {
                    kFlat[kvDst + d] = kSrc[d]
                    vFlat[kvDst + d] = vSrc[d]
                }
            }
        }

        // Stage 2: one GPU SDPA dispatch over all patches (full
        // bidirectional — no causal mask within one image).
        let qT = Tensor.empty(shape: [nPatches, nHeads, headDim], dtype: .f32, device: device)
        let kT = Tensor.empty(shape: [nHeads, nPatches, headDim], dtype: .f32, device: device)
        let vT = Tensor.empty(shape: [nHeads, nPatches, headDim], dtype: .f32, device: device)
        qT.copyIn(from: qFlat)
        kT.copyIn(from: kFlat)
        vT.copyIn(from: vFlat)
        let cmd = device.makeCommandBuffer()
        let outT = Ops.sdpaBidirectional(
            q: qT, k: kT, v: vT,
            nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
            baseKV: 0, nQuery: nPatches, kvStride: nPatches,
            scale: scale, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // Output is [nPatches, nHeads, headDim] = [nPatches, hidden] flat.
        // Copy into a tensor of the caller-expected dtype (e.g. bf16/f16
        // for downstream `Ops.gemm`).
        let result = Tensor.empty(shape: [nPatches, hidden], dtype: q.dtype,
                                  device: device)
        ImagePreprocessing.copyFloats(outT.toFloatArray(), into: result)
        return result
    }
}

/// Apply 2D RoPE (rotate-half) to a D-dimensional vector in-place.
/// The vector is split into two halves:
///   [0, D/2) — height-driven RoPE
///   [D/2, D) — width-driven RoPE
/// Within each half the rotate-half transform is:
///   out[i]     = x[i] * cos[i] - x[i + half/2] * sin[i]    (for i < half/2)
///   out[i+h/2] = x[i+half/2]*cos[i+half/2] + x[i]*sin[i+half/2]
/// where `half` = D/2 and `half/2` = D/4.
@inline(__always)
private func applyRoPE2D(
    _ x: inout [Float],
    cos: ArraySlice<Float>, sin: ArraySlice<Float>,
    half: Int
) {
    let D = x.count
    let quarter = half / 2
    // Height half [0, half): rotate-half within [0, half).
    for i in 0..<quarter {
        let xi = x[i]; let xi2 = x[i + quarter]
        let c = cos[cos.startIndex + i]; let s = sin[sin.startIndex + i]
        let c2 = cos[cos.startIndex + i + quarter]; let s2 = sin[sin.startIndex + i + quarter]
        x[i]          = xi * c  - xi2 * s
        x[i + quarter] = xi * s2 + xi2 * c2
    }
    // Width half [half, D): rotate-half within [half, D).
    for i in 0..<quarter {
        let xi = x[half + i]; let xi2 = x[half + i + quarter]
        let c = cos[cos.startIndex + half + i]; let s = sin[sin.startIndex + half + i]
        let c2 = cos[cos.startIndex + half + i + quarter]; let s2 = sin[sin.startIndex + half + i + quarter]
        x[half + i]          = xi * c  - xi2 * s
        x[half + i + quarter] = xi * s2 + xi2 * c2
    }
    _ = D  // suppress unused warning
}

// ─── Vision encoder ───────────────────────────────────────────────────

/// The Pixtral vision tower. Holds the patch-embed conv, the block stack,
/// and the precomputed 2D RoPE tables. `encode` runs the full forward and
/// returns patch tokens in the vision hidden dim (before the projector).
final class PixtralVisionEncoder: @unchecked Sendable {
    let cfg: PixtralVisionConfig
    /// Conv2d patch-embed weight `[hidden, inChannels, patchSize, patchSize]`.
    let patchConvWeight: Tensor
    let lnPre: RMSNorm
    let blocks: [PixtralVisionBlock]
    let rope: PixtralRoPE
    let dtype: DType

    init(cfg: PixtralVisionConfig, patchConvWeight: Tensor, lnPre: RMSNorm,
         blocks: [PixtralVisionBlock], rope: PixtralRoPE, dtype: DType) {
        self.cfg = cfg
        self.patchConvWeight = patchConvWeight
        self.lnPre = lnPre
        self.blocks = blocks
        self.rope = rope
        self.dtype = dtype
    }

    static func load(
        cfg: PixtralVisionConfig, weights: SafeTensorsBundle,
        dtype: DType, device: Device
    ) throws -> PixtralVisionEncoder {
        // Patch-embed Conv2d weight — mlx-community stores it as OHWI
        // `[out_ch, kH, kW, in_ch]`. `Ops.conv2d` expects OIHW.
        let patchRaw = try weights.tensor(named: "vision_model.patch_conv.weight")
        let patchW: Tensor
        if patchRaw.shape.count == 4 && patchRaw.shape[3] == cfg.numChannels {
            // MLX OHWI → OIHW transpose.
            patchW = transposeOHWItoOIHW(patchRaw)
        } else {
            patchW = patchRaw
        }

        // Pre-norm before the transformer stack (`ln_pre` in the reference).
        let lnPreW = try weights.tensor(named: "vision_model.ln_pre.weight")
        let lnPre = RMSNorm(weight: lnPreW, eps: cfg.rmsNormEps)

        // Build the block stack.
        var blocks: [PixtralVisionBlock] = []
        blocks.reserveCapacity(cfg.numLayers)
        for i in 0..<cfg.numLayers {
            let p = "vision_model.transformer.layers.\(i)"
            func norm(_ key: String) throws -> RMSNorm {
                RMSNorm(weight: try weights.tensor(named: "\(p).\(key).weight"),
                        eps: cfg.rmsNormEps)
            }
            func lin(_ key: String) throws -> Linear {
                Linear(weight: try weights.tensor(named: "\(p).\(key).weight"))
            }
            blocks.append(PixtralVisionBlock(
                attNorm: try norm("attention_norm"),
                qProj: try lin("attention.q_proj"),
                kProj: try lin("attention.k_proj"),
                vProj: try lin("attention.v_proj"),
                oProj: try lin("attention.o_proj"),
                ffnNorm: try norm("ffn_norm"),
                gateProj: try lin("feed_forward.gate_proj"),
                upProj:   try lin("feed_forward.up_proj"),
                downProj: try lin("feed_forward.down_proj"),
                cfg: cfg))
        }

        let rope = PixtralRoPE(cfg: cfg)
        return PixtralVisionEncoder(cfg: cfg, patchConvWeight: patchW,
                                    lnPre: lnPre, blocks: blocks,
                                    rope: rope, dtype: dtype)
    }

    /// Encode a preprocessed image `[1, inChannels, imageSize, imageSize]`
    /// (NCHW) through the Pixtral vision tower. Returns `[numPatches, hiddenSize]`.
    func encode(image: Tensor, device: Device = .shared) -> Tensor {
        let p = cfg.patchSize
        let gridSide = cfg.patchesPerSide
        let nPatches = gridSide * gridSide
        let hidden = cfg.hiddenSize

        // ── Conv2d patch embed ──
        // `Ops.conv2d` expects NCHW input and OIHW weight, stride = patchSize.
        // Pixtral's patch_conv has no bias — pass a zero-valued [hidden] bias.
        // Output: [1, hidden, gridSide, gridSide] (NCHW channel-major).
        let zeroBias = makeZeroBias(length: hidden, dtype: dtype, device: device)
        let cmd = device.makeCommandBuffer()
        let conv = Ops.conv2d(
            input: image, weight: patchConvWeight, bias: zeroBias,
            strideH: p, strideW: p, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // Transpose NCHW → [nPatches, hidden] token-major (CPU).
        var h = channelMajorToTokenMajor(conv, hidden: hidden,
                                          numPatches: nPatches, device: device)

        // ── Pre-norm ──
        let cmdN = device.makeCommandBuffer()
        h = Ops.rmsNormRows(h, weight: lnPre.weight, eps: lnPre.eps,
                             nRows: nPatches, rowSize: hidden, on: cmdN)
        cmdN.commit()
        cmdN.waitUntilCompleted()

        // ── Transformer block stack ──
        for block in blocks {
            h = block.forward(h, nPatches: nPatches, gridSide: gridSide,
                              rope: rope, device: device)
        }
        return h
    }

    /// Allocate a zero-valued `[length]` bias tensor. Used when the
    /// checkpoint's conv has no bias but `Ops.conv2d` requires one.
    private func makeZeroBias(length: Int, dtype: DType, device: Device) -> Tensor {
        let zeros = [Float](repeating: 0, count: length)
        let t = Tensor.empty(shape: [length], dtype: dtype, device: device)
        ImagePreprocessing.copyFloats(zeros, into: t)
        return t
    }

    /// Reinterpret a conv2d `[1, hidden, P, P]` channel-major output as
    /// `[numPatches, hidden]` token-major. CPU transpose.
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

// ─── Multi-modal projector ────────────────────────────────────────────

/// Pixtral multi-modal projector: two linear layers with GELU activation.
/// Maps `[numPatches, visionHidden]` encoder tokens into
/// `[numPatches, textHidden]` so the VLM splice can inject them.
final class PixtralProjector: @unchecked Sendable {
    let linear1: Linear   // [textHidden, visionHidden] — with bias
    let linear2: Linear   // [textHidden, textHidden]   — with bias
    let visionHidden: Int
    let textHidden: Int

    init(linear1: Linear, linear2: Linear, visionHidden: Int, textHidden: Int) {
        self.linear1 = linear1
        self.linear2 = linear2
        self.visionHidden = visionHidden
        self.textHidden = textHidden
    }

    static func load(
        visionHidden: Int, textHidden: Int,
        weights: SafeTensorsBundle, device: Device
    ) throws -> PixtralProjector {
        func lin(_ key: String) throws -> Linear {
            Linear(weight: try weights.tensor(named: "\(key).weight"),
                   bias: try? weights.tensor(named: "\(key).bias"))
        }
        let linear1 = try lin("multi_modal_projector.linear_1")
        let linear2 = try lin("multi_modal_projector.linear_2")
        return PixtralProjector(linear1: linear1, linear2: linear2,
                                visionHidden: visionHidden, textHidden: textHidden)
    }

    /// Project `[nTokens, visionHidden]` vision tokens into
    /// `[nTokens, textHidden]` via `GELU(linear1(x)) → linear2`.
    func project(tokens: Tensor, nTokens: Int, device: Device) -> Tensor {
        let cmd = device.makeCommandBuffer()
        var x = Ops.gemm(weight: linear1.weight, input: tokens,
                         nRows: nTokens, on: cmd)
        if let b = linear1.bias {
            x = addRowBias(x, bias: b, nRows: nTokens, rowSize: textHidden, on: cmd)
        }
        x = Ops.gelu(x, on: cmd)
        var y = Ops.gemm(weight: linear2.weight, input: x,
                         nRows: nTokens, on: cmd)
        if let b = linear2.bias {
            y = addRowBias(y, bias: b, nRows: nTokens, rowSize: textHidden, on: cmd)
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        return y
    }

    /// Broadcast-add a `[rowSize]` bias to each of `nRows` rows of a
    /// `[nRows, rowSize]` tensor.
    private func addRowBias(_ x: Tensor, bias: Tensor, nRows: Int,
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

// ─── Composed tower ───────────────────────────────────────────────────

/// Couples the `PixtralVisionEncoder` with the `PixtralProjector` so the
/// pair presents a single `VisionEncoder`-shaped surface to `VLModel`.
final class PixtralComposedTower {
    let encoder: PixtralVisionEncoder
    let projector: PixtralProjector
    let visionCfg: PixtralVisionConfig
    let textHidden: Int
    let dtype: DType

    init(encoder: PixtralVisionEncoder, projector: PixtralProjector,
         visionCfg: PixtralVisionConfig, textHidden: Int, dtype: DType) {
        self.encoder = encoder
        self.projector = projector
        self.visionCfg = visionCfg
        self.textHidden = textHidden
        self.dtype = dtype
    }

    func asVisionEncoder() -> VisionEncoder {
        PixtralComposedEncoder(tower: self)
    }
}

/// A `VisionEncoder` subclass whose `encode` runs the Pixtral vision tower
/// then the multi-modal projector, returning `[numPatches, textHidden]`.
final class PixtralComposedEncoder: VisionEncoder {
    let tower: PixtralComposedTower

    init(tower: PixtralComposedTower) {
        self.tower = tower
        let cfg = tower.visionCfg
        // The facade config uses imageSize and patchSize to expose the
        // patch count (`numPatches`) to `VLModel.imageTokenCount`.
        // textHidden is the projected output hidden dim.
        let facadeConfig = VisionEncoderConfig(
            inChannels: cfg.numChannels,
            imageSize: cfg.imageSize, patchSize: cfg.patchSize,
            hidden: cfg.hiddenSize, intermediate: cfg.intermediateSize,
            nLayers: cfg.numLayers, nHeads: cfg.numHeads,
            layerNormEps: cfg.rmsNormEps, textHidden: tower.textHidden)
        // Placeholder tensors — the base `encode` is fully overridden
        // below so these are never read.
        let placeholder = tower.encoder.patchConvWeight
        super.init(config: facadeConfig,
                   patchEmbedWeight: placeholder, patchEmbedBias: placeholder,
                   positionEmbedding: placeholder, layers: [],
                   postLayerNorm: tower.encoder.lnPre.asLayerNorm(),
                   projection: nil, dtype: tower.dtype)
    }

    /// Run the Pixtral vision tower + projector.
    /// Returns `[numPatches, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        let nPatches = tower.visionCfg.numPatches
        let raw = tower.encoder.encode(image: image, device: device)
        return tower.projector.project(tokens: raw, nTokens: nPatches, device: device)
    }
}
