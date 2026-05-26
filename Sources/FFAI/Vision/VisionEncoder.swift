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
// VisionEncoder — the vision-transformer (ViT) stack a VLM runs an image
// through before splicing the resulting tokens into the text stream.
//
// Declared in for the capability API; lit up here initially.
// The architecture is the SigLIP / CLIP ViT shape every shipped VLM
// vision tower uses:
//
//   image  ──conv2d patch embed──▶  [num_patches, hidden]
//          ──+ learned position embedding──▶
//          ──▶ N × { LayerNorm → MHA (bidirectional) → +residual
//                    LayerNorm → MLP (GELU)          → +residual }
//          ──post-LayerNorm──▶
//          ──projection──▶  [num_patches, text_hidden]
//
// The encoder processes ALL patch tokens at once (vision attention is
// bidirectional — no causal mask, no KV cache), so it uses the
// multi-query `Ops.sdpaMulti` path rather than the single-token decode
// SDPA the text backbone uses.
//
// Qwen-VL's windowed-attention / dynamic-resolution towers are a
// superset of this; the family files that need them layer the extra
// behaviour on top. This file is the shared SigLIP-style core.

import Foundation
import Metal

// ─── Configuration ───────────────────────────────────────────────────

/// Static shape + hyper-parameters of a ViT vision tower, decoded from
/// the checkpoint's `vision_config`.
public struct VisionEncoderConfig: Sendable {
    /// Channels of the input image (3 for RGB).
    public let inChannels: Int
    /// Square input resolution the encoder expects (e.g. 224, 896).
    public let imageSize: Int
    /// Square patch side (14 for SigLIP/Qwen-VL, 16 for CLIP/Gemma-VL).
    public let patchSize: Int
    /// Encoder hidden dimension.
    public let hidden: Int
    /// Encoder feed-forward intermediate dimension.
    public let intermediate: Int
    /// Number of transformer blocks.
    public let nLayers: Int
    /// Number of attention heads per block.
    public let nHeads: Int
    /// LayerNorm epsilon.
    public let layerNormEps: Float
    /// Text-model hidden dimension the encoder output is projected into.
    /// When equal to `hidden` the projection is identity / absent.
    public let textHidden: Int

    public init(
        inChannels: Int = 3, imageSize: Int, patchSize: Int,
        hidden: Int, intermediate: Int, nLayers: Int, nHeads: Int,
        layerNormEps: Float = 1e-6, textHidden: Int
    ) {
        self.inChannels = inChannels
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.hidden = hidden
        self.intermediate = intermediate
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.layerNormEps = layerNormEps
        self.textHidden = textHidden
    }

    /// Patches along one axis (`imageSize / patchSize`).
    public var patchesPerSide: Int { imageSize / patchSize }
    /// Total patch tokens an image produces (`patchesPerSide²`).
    public var numPatches: Int { patchesPerSide * patchesPerSide }
    /// Per-head dimension.
    public var headDim: Int { hidden / nHeads }
}

// ─── Encoder block ───────────────────────────────────────────────────

/// One pre-norm ViT transformer block: LayerNorm → bidirectional MHA →
/// residual, then LayerNorm → GELU MLP → residual.
public final class VisionEncoderLayer: Module {
    let layerNorm1: LayerNorm
    let qProj, kProj, vProj, oProj: Linear
    let layerNorm2: LayerNorm
    let fc1, fc2: Linear

    let hidden, nHeads, headDim, intermediate: Int
    let scale: Float

    init(
        layerNorm1: LayerNorm,
        qProj: Linear, kProj: Linear, vProj: Linear, oProj: Linear,
        layerNorm2: LayerNorm, fc1: Linear, fc2: Linear,
        hidden: Int, nHeads: Int, intermediate: Int
    ) {
        self.layerNorm1 = layerNorm1
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.layerNorm2 = layerNorm2
        self.fc1 = fc1
        self.fc2 = fc2
        self.hidden = hidden
        self.nHeads = nHeads
        self.headDim = hidden / nHeads
        self.intermediate = intermediate
        self.scale = 1.0 / Float(Double(hidden / nHeads).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in layerNorm1.parameters() { out.append(("layer_norm1.\(k)", v)) }
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.out_proj.\(k)", v)) }
        for (k, v) in layerNorm2.parameters() { out.append(("layer_norm2.\(k)", v)) }
        for (k, v) in fc1.parameters() { out.append(("mlp.fc1.\(k)", v)) }
        for (k, v) in fc2.parameters() { out.append(("mlp.fc2.\(k)", v)) }
        return out
    }

    /// Forward `[nTokens, hidden]` patch-token activations through one
    /// encoder block. The GEMM-heavy projections + norms run on the GPU
    /// queued on `cmd`; the bidirectional attention core then dispatches
    /// to a head-dim-matched GPU kernel where one exists, falling back
    /// to a parallel CPU pass only for head_dim values that have no
    /// kernel yet.
    ///
    /// Available kernels (FFAI/Ops.swift):
    ///   - head_dim ∈ {32, 64, 72, 80, 96} → `Ops.sdpaBidirectional`
    ///     (multi-query, non-causal)
    ///   - head_dim == 128 → `Ops.sdpaMulti(causal: false)`
    ///     (multi-query, hardcoded d=128)
    ///   - anything else   → CPU `cpuAttention` (parallel by
    ///                       (head, query-row), see below)
    ///
    /// At SigLIP-So400m's 4096 patches × 27 layers × 16 heads × O(n²)
    /// the CPU path used to take 25+ minutes per encoder pass; the GPU
    /// path collapses that into the per-layer GEMM cost.
    func forward(
        _ h: Tensor, nTokens: Int, device: Device,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        // ── Attention sub-block ──
        let normed = Ops.layerNorm(
            h, weight: layerNorm1.weight,
            bias: layerNorm1.bias, eps: layerNorm1.eps,
            nRows: nTokens, rowSize: hidden, on: cmd)
        // Q/K/V projections over every token, one multi-row GEMM each.
        let q = projectRows(qProj, normed, nTokens: nTokens, on: cmd)
        let k = projectRows(kProj, normed, nTokens: nTokens, on: cmd)
        let v = projectRows(vProj, normed, nTokens: nTokens, on: cmd)

        // Bidirectional multi-head attention. GPU path when the kernel
        // exists at this head_dim; CPU fallback otherwise.
        let attnFlat: Tensor
        if OpsValidation.sdpaBidirectionalSupportedHeadDims.contains(headDim) {
            // {32, 64, 72, 80, 96} — covers FastViT-HD, SigLIP-base/
            // CLIP-L/Mistral3/Gemma4-E2/Qwen3-VL-2B/4B (64),
            // SigLIP-So400m (72), Qwen2.5-VL (80), Qwen2-VL (96).
            attnFlat = gpuAttention(
                q: q, k: k, v: v, nTokens: nTokens,
                device: device, on: cmd)
        } else if headDim == 128 {
            // Pixtral, Mistral3 Pixtral-based, GlmOcr.
            attnFlat = gpuAttentionMulti(
                q: q, k: k, v: v, nTokens: nTokens,
                device: device, on: cmd)
        } else {
            // Flush the projection GEMMs so their results are CPU-readable
            // for the fallback attention core.
            cmd.commit()
            cmd.waitUntilCompleted()
            attnFlat = cpuAttention(
                q: q, k: k, v: v, nTokens: nTokens,
                device: device)
        }

        // ── Residual + MLP sub-block ──
        let cmd2 = device.makeCommandBuffer()
        let attnProj = projectRows(oProj, attnFlat, nTokens: nTokens, on: cmd2)
        let postAttn = Ops.add(h, attnProj, on: cmd2)
        let normed2 = Ops.layerNorm(
            postAttn, weight: layerNorm2.weight,
            bias: layerNorm2.bias, eps: layerNorm2.eps,
            nRows: nTokens, rowSize: hidden, on: cmd2)
        let ff1 = projectRows(
            fc1, normed2, nTokens: nTokens,
            outDim: intermediate, on: cmd2)
        let act = Ops.gelu(ff1, on: cmd2)
        let ff2 = projectRows(
            fc2, act, nTokens: nTokens,
            inDim: intermediate, outDim: hidden, on: cmd2)
        let result = Ops.add(postAttn, ff2, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return result
    }

    /// GPU bidirectional multi-head attention via
    /// `Ops.sdpaBidirectional(headDim:)`. head_dim must be one of
    /// {32, 64, 72} — the kernel surface FFAI ships today.
    ///
    /// Layout note: `Ops.sdpaBidirectional` takes
    ///   Q at `[nQuery, nQHeads, headDim]` (token-major, head-second)
    ///   K/V at `[nKVHeads, kvStride, headDim]` (head-major).
    ///
    /// The caller hands us Q/K/V as `[nTokens, nHeads*headDim]`. Q's
    /// memory layout is byte-identical to the kernel's
    /// `[nQuery, nQHeads, headDim]` (same row-major tensor with a
    /// different shape annotation). K/V need a `[nTokens, nHeads,
    /// headDim]` → `[nHeads, nTokens, headDim]` transpose; we do it
    /// CPU-side (one readback + copy) for the same reason the
    /// Paligemma / SmolVLM2 migrations do — there is no
    /// `Ops.transpose` wrapper. The transpose cost is dwarfed by the
    /// O(n²·d) attention kernel, and we get a fully GPU-resident
    /// softmax / weighted-V loop in exchange.
    private func gpuAttention(
        q: Tensor, k: Tensor, v: Tensor,
        nTokens: Int, device: Device,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        // Flush Q/K/V projections so we can read K/V back for the
        // CPU-side head-major transpose.
        cmd.commit()
        cmd.waitUntilCompleted()

        let kHeadMajor = transposeTokenToHeadMajor(
            k, nTokens: nTokens, device: device)
        let vHeadMajor = transposeTokenToHeadMajor(
            v, nTokens: nTokens, device: device)
        // Q in its current `[nTokens, nHeads*headDim]` layout is
        // byte-identical to `[nQuery, nQHeads, headDim]` — no copy.
        let qReshaped = q.reshaped(to: [nTokens, nHeads, headDim])

        let cmd2 = device.makeCommandBuffer()
        let attn = Ops.sdpaBidirectional(
            q: qReshaped, k: kHeadMajor, v: vHeadMajor,
            nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
            baseKV: 0, nQuery: nTokens, kvStride: nTokens,
            scale: scale, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        // Kernel output is `[nQuery, nQHeads, headDim]` — byte-identical
        // to the `[nTokens, nHeads*headDim]` flat layout o_proj expects.
        return attn.reshaped(to: [nTokens, nHeads * headDim])
    }

    /// GPU bidirectional MHA dispatched via `Ops.sdpaMulti(causal:false)`
    /// — the head_dim=128 variant. Same layout contract as
    /// `gpuAttention`. Used when this tower's head_dim is 128.
    private func gpuAttentionMulti(
        q: Tensor, k: Tensor, v: Tensor,
        nTokens: Int, device: Device,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        cmd.commit()
        cmd.waitUntilCompleted()

        let kHeadMajor = transposeTokenToHeadMajor(
            k, nTokens: nTokens, device: device)
        let vHeadMajor = transposeTokenToHeadMajor(
            v, nTokens: nTokens, device: device)
        let qReshaped = q.reshaped(to: [nTokens, nHeads, headDim])

        let cmd2 = device.makeCommandBuffer()
        let attn = Ops.sdpaMulti(
            q: qReshaped, k: kHeadMajor, v: vHeadMajor,
            nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
            baseKV: 0, nQuery: nTokens, kvStride: nTokens,
            causal: false, scale: scale, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        return attn.reshaped(to: [nTokens, nHeads * headDim])
    }

    /// `[nTokens, nHeads*headDim]` (token-major) →
    /// `[nHeads, nTokens, headDim]` (head-major) via a CPU pass.
    /// Used to repack K/V into the layout the sdpa kernels expect.
    private func transposeTokenToHeadMajor(
        _ t: Tensor, nTokens: Int, device: Device
    ) -> Tensor {
        let src = t.toFloatArray()
        var dst = [Float](repeating: 0, count: nTokens * nHeads * headDim)
        let stride = nHeads * headDim
        // src[tok, h, d] = src[tok*stride + h*headDim + d]
        // dst[h, tok, d] = dst[h*nTokens*headDim + tok*headDim + d]
        for tok in 0 ..< nTokens {
            let srcRow = tok * stride
            for h in 0 ..< nHeads {
                let srcH = srcRow + h * headDim
                let dstH = h * nTokens * headDim + tok * headDim
                for d in 0 ..< headDim { dst[dstH + d] = src[srcH + d] }
            }
        }
        let out = Tensor.empty(
            shape: [nHeads, nTokens, headDim],
            dtype: t.dtype, device: device)
        ImagePreprocessing.copyFloats(dst, into: out)
        return out
    }

    /// CPU bidirectional multi-head attention over `nTokens` patch
    /// tokens. `q` / `k` / `v` are token-major `[nTokens, nHeads*headDim]`.
    /// Returns the context, token-major `[nTokens, nHeads*headDim]`,
    /// ready for the output projection.
    ///
    /// The work is O(`nHeads · nTokens² · headDim`). SigLIP-896 vision
    /// towers produce `nTokens` = 4096 (64×64 patch grid), so a naive
    /// single-threaded scalar pass is ~10¹² float ops per encoder forward
    /// — minutes of wall time. Each `(head, query-row)` pair writes a
    /// disjoint `headDim`-wide slice of `out`, so the outer
    /// `nHeads · nTokens` index space is embarrassingly parallel: it is
    /// fanned out across cores with `concurrentPerform`, which is
    /// race-free precisely because the write targets never overlap.
    private func cpuAttention(
        q: Tensor, k: Tensor, v: Tensor,
        nTokens: Int, device: Device
    ) -> Tensor {
        let qa = q.toFloatArray()
        let ka = k.toFloatArray()
        let va = v.toFloatArray()
        let stride = nHeads * headDim
        var out = [Float](repeating: 0, count: nTokens * stride)

        // Fan the (head, query-row) index space across cores. `out` is
        // mutated through an unsafe pointer because each iteration owns a
        // disjoint `[oBase, oBase+headDim)` slice — no two iterations
        // touch the same element, so the writes need no synchronization.
        let nHeadsLocal = nHeads
        let headDimLocal = headDim
        let scaleLocal = scale
        out.withUnsafeMutableBufferPointer { outBuf in
            let outPtr = outBuf.baseAddress!
            qa.withUnsafeBufferPointer { qPtr in
                ka.withUnsafeBufferPointer { kPtr in
                    va.withUnsafeBufferPointer { vPtr in
                        let qb = qPtr.baseAddress!
                        let kb = kPtr.baseAddress!
                        let vb = vPtr.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: nHeadsLocal * nTokens) { work in
                            let head = work / nTokens
                            let i = work % nTokens
                            let hOff = head * headDimLocal
                            // Scaled dot-product scores against every token.
                            var scores = [Float](repeating: 0, count: nTokens)
                            var maxScore = -Float.greatestFiniteMagnitude
                            let qBase = i * stride + hOff
                            for j in 0 ..< nTokens {
                                var dot: Float = 0
                                let kBase = j * stride + hOff
                                for d in 0 ..< headDimLocal {
                                    dot += qb[qBase + d] * kb[kBase + d]
                                }
                                let s = dot * scaleLocal
                                scores[j] = s
                                if s > maxScore { maxScore = s }
                            }
                            // Softmax (numerically stable).
                            var sumExp: Float = 0
                            for j in 0 ..< nTokens {
                                let e = exp(scores[j] - maxScore)
                                scores[j] = e
                                sumExp += e
                            }
                            let inv = sumExp > 0 ? 1 / sumExp : 0
                            // Weighted sum of V into this row's disjoint slice.
                            let oBase = i * stride + hOff
                            for j in 0 ..< nTokens {
                                let w = scores[j] * inv
                                let vBase = j * stride + hOff
                                for d in 0 ..< headDimLocal {
                                    outPtr[oBase + d] += w * vb[vBase + d]
                                }
                            }
                        }
                    }
                }
            }
        }
        let result = Tensor.empty(
            shape: [nTokens, stride], dtype: q.dtype,
            device: device)
        ImagePreprocessing.copyFloats(out, into: result)
        return result
    }

    /// Apply a `Linear` to every row of a `[nTokens, *]` tensor via
    /// `Ops.gemm` (one tiled multi-row GEMM), then broadcast-add the
    /// bias to every row. The vision Linears always carry a bias.
    /// `outDim` defaults to the block hidden; pass it for the MLP's
    /// `hidden → intermediate → hidden` shape changes.
    private func projectRows(
        _ linear: Linear, _ x: Tensor, nTokens: Int,
        inDim: Int? = nil, outDim: Int? = nil,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        _ = inDim  // kept for call-site documentation symmetry with outDim
        let outD = outDim ?? hidden
        let y = Ops.gemm(weight: linear.weight, input: x, nRows: nTokens, on: cmd)
        guard let bias = linear.bias else { return y }
        return addRowBias(y, bias: bias, nRows: nTokens, rowSize: outD, on: cmd)
    }

    /// Add a `[rowSize]` bias to each of `nRows` rows of a flat
    /// `[nRows, rowSize]` tensor. `Ops.add` is element-wise same-shape,
    /// so the bias is tiled into a full `[nRows*rowSize]` buffer once.
    private func addRowBias(
        _ x: Tensor, bias: Tensor, nRows: Int,
        rowSize: Int, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let tiled = Tensor.empty(shape: [nRows, rowSize], dtype: x.dtype)
        let biasVals = bias.toFloatArray()
        var flat = [Float](repeating: 0, count: nRows * rowSize)
        for r in 0 ..< nRows {
            for c in 0 ..< rowSize { flat[r * rowSize + c] = biasVals[c] }
        }
        ImagePreprocessing.copyFloats(flat, into: tiled)
        return Ops.add(x, tiled, on: cmd)
    }
}

// ─── Vision encoder ──────────────────────────────────────────────────

/// A SigLIP / CLIP-style ViT vision tower. Holds the patch-embed conv
/// weights, the learned position embedding, the encoder block stack,
/// the post-LayerNorm, and the projection into the text hidden dim.
// Non-final so family files can subclass it — e.g. Gemma 3 VL's
// composed encoder+projector tower (`Gemma3VLComposedEncoder`) overrides
// `encode` to append its multi-modal projector.
public class VisionEncoder: Module {
    public let config: VisionEncoderConfig

    /// conv2d patch-embed weight `[hidden, inChannels, patchSize, patchSize]`.
    public let patchEmbedWeight: Tensor
    /// conv2d patch-embed bias `[hidden]`.
    public let patchEmbedBias: Tensor
    /// Learned position embedding `[numPatches, hidden]`.
    public let positionEmbedding: Tensor
    /// Encoder blocks.
    public let layers: [VisionEncoderLayer]
    /// Post-encoder LayerNorm.
    public let postLayerNorm: LayerNorm
    /// Optional projection into the text hidden dim. `nil` when the
    /// encoder hidden already equals the text hidden.
    public let projection: Linear?

    public let dtype: DType

    public init(
        config: VisionEncoderConfig,
        patchEmbedWeight: Tensor, patchEmbedBias: Tensor,
        positionEmbedding: Tensor, layers: [VisionEncoderLayer],
        postLayerNorm: LayerNorm, projection: Linear?,
        dtype: DType
    ) {
        self.config = config
        self.patchEmbedWeight = patchEmbedWeight
        self.patchEmbedBias = patchEmbedBias
        self.positionEmbedding = positionEmbedding
        self.layers = layers
        self.postLayerNorm = postLayerNorm
        self.projection = projection
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        out.append(("embeddings.patch_embedding.weight", patchEmbedWeight))
        out.append(("embeddings.patch_embedding.bias", patchEmbedBias))
        out.append(("embeddings.position_embedding.weight", positionEmbedding))
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("encoder.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in postLayerNorm.parameters() { out.append(("post_layernorm.\(k)", v)) }
        if let proj = projection {
            for (k, v) in proj.parameters() { out.append(("projection.\(k)", v)) }
        }
        return out
    }

    /// Encode a preprocessed image into patch-token embeddings in the
    /// text hidden dim.
    ///
    /// `image` is a normalized NCHW tensor `[1, inChannels, imageSize,
    /// imageSize]` (the `ImagePreprocessing.preprocess` output).
    /// Returns `[numPatches, textHidden]` — the tokens a VLM splices
    /// into its text stream. All GPU work is queued on a private command
    /// buffer that is committed + waited before returning, since callers
    /// consume the result on the CPU (the cross-modal splice).
    public func encode(image: Tensor, device: Device = .shared) -> Tensor {
        precondition(
            image.shape == [
                1, config.inChannels,
                config.imageSize, config.imageSize,
            ],
            "VisionEncoder.encode: image shape \(image.shape) != "
                + "[1,\(config.inChannels),\(config.imageSize),\(config.imageSize)]")
        let cmd = device.makeCommandBuffer()

        // Patch embedding — fused conv (one thread per output element).
        // conv2d output is NCHW [1, hidden, patchesPerSide, patchesPerSide];
        // flatten the spatial grid to [numPatches, hidden] token-major.
        let conv = Ops.conv2d(
            input: image, weight: patchEmbedWeight, bias: patchEmbedBias,
            strideH: config.patchSize, strideW: config.patchSize, on: cmd)
        // conv is [1, hidden, P, P] channel-major; we need [numPatches, hidden]
        // token-major — transpose on the CPU after the conv completes.
        cmd.commit()
        cmd.waitUntilCompleted()

        var tokens = channelMajorToTokenMajor(
            conv, hidden: config.hidden, numPatches: config.numPatches,
            device: device)

        // Add the learned position embedding.
        let cmd2 = device.makeCommandBuffer()
        var h = Ops.add(tokens, positionEmbedding, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        // Run the encoder block stack. Each block commits its own work
        // internally (its attention core needs a CPU sync point).
        for layer in layers {
            let cmd = device.makeCommandBuffer()
            h = layer.forward(
                h, nTokens: config.numPatches,
                device: device, on: cmd)
        }

        // Post-encoder LayerNorm.
        let cmdN = device.makeCommandBuffer()
        h = Ops.layerNorm(
            h, weight: postLayerNorm.weight,
            bias: postLayerNorm.bias, eps: postLayerNorm.eps,
            nRows: config.numPatches, rowSize: config.hidden,
            on: cmdN)
        cmdN.commit()
        cmdN.waitUntilCompleted()
        tokens = h

        // Project into the text hidden dim if the encoder hidden differs.
        guard let proj = projection else { return tokens }
        let cmd3 = device.makeCommandBuffer()
        var projected = Ops.gemm(
            weight: proj.weight, input: tokens,
            nRows: config.numPatches, on: cmd3)
        if let bias = proj.bias {
            let biasVals = bias.toFloatArray()
            var flat = [Float](
                repeating: 0,
                count: config.numPatches * config.textHidden)
            for r in 0 ..< config.numPatches {
                for c in 0 ..< config.textHidden {
                    flat[r * config.textHidden + c] = biasVals[c]
                }
            }
            let biasTiled = Tensor.empty(
                shape: [config.numPatches, config.textHidden], dtype: dtype)
            ImagePreprocessing.copyFloats(flat, into: biasTiled)
            projected = Ops.add(projected, biasTiled, on: cmd3)
        }
        cmd3.commit()
        cmd3.waitUntilCompleted()
        return projected
    }

    /// Encode a sequence of frames as a temporally-batched token stream.
    /// Default implementation throws `VisionEncoderError.videoUnsupported`
    /// — towers that wire real video patches (Qwen 2/2.5/3 VL,
    /// MiniCPM-V, SmolVLM2, QwenOmni) override this with the temporal-
    /// patch unfold path. Per-model overrides decide how frames map to
    /// the temporal axis (typically `temporal_patch_size`-wide groups).
    public func encode(frames: [Tensor], device: Device = .shared) throws -> Tensor {
        throw VisionEncoderError.videoUnsupported(
            family: String(describing: type(of: self)))
    }

    /// Reinterpret a conv2d `[1, hidden, P, P]` channel-major output as
    /// `[numPatches, hidden]` token-major patch tokens. CPU transpose —
    /// patch counts are at most a few thousand.
    private func channelMajorToTokenMajor(
        _ conv: Tensor, hidden: Int, numPatches: Int, device: Device
    ) -> Tensor {
        let src = conv.toFloatArray()
        var dst = [Float](repeating: 0, count: numPatches * hidden)
        for c in 0 ..< hidden {
            for p in 0 ..< numPatches {
                dst[p * hidden + c] = src[c * numPatches + p]
            }
        }
        let out = Tensor.empty(
            shape: [numPatches, hidden], dtype: dtype,
            device: device)
        ImagePreprocessing.copyFloats(dst, into: out)
        return out
    }
}

extension ImagePreprocessing {
    /// Write a `[Float]` array into an existing `Tensor`, converting to
    /// the tensor's storage dtype. Shared by the vision encoder for its
    /// CPU-side transpose / bias-tile staging.
    static func copyFloats(_ values: [Float], into tensor: Tensor) {
        precondition(
            values.count == tensor.elementCount,
            "copyFloats: count mismatch \(values.count) vs \(tensor.elementCount)")
        switch tensor.dtype {
        case .f32:
            tensor.copyIn(from: values)
        case .f16:
            tensor.copyIn(from: values.map { Float16($0) })
        case .bf16:
            tensor.copyIn(
                from: values.map { v -> UInt16 in
                    let bits = v.bitPattern
                    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                    return UInt16(rounded >> 16)
                })
        default:
            fatalError("copyFloats: unsupported dtype \(tensor.dtype)")
        }
    }
}

public enum VisionEncoderError: Error, CustomStringConvertible {
    /// The base `VisionEncoder.encode(frames:)` default throws this.
    /// Override on towers that wire the multi-frame temporal-patch path.
    case videoUnsupported(family: String)

    public var description: String {
        switch self {
        case .videoUnsupported(let family):
            return "VisionEncoder: \(family) does not implement video encode(frames:)."
                + " The single-image encode(image:) path is fine; video requires the"
                + " multi-frame temporal-patch unfold to be wired per-family."
        }
    }
}
