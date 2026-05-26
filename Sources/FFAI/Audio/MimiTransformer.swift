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
// MimiTransformer — the latent Transformer and split residual-VQ
// quantizer for the Mimi codec.
//
// Companion to Mimi.swift / MimiBlocks.swift. Mimi inserts an 8-layer
// pre-norm Transformer (RoPE attention + LayerScale + a no-gating GELU
// MLP) between the SEANet stack and the quantizer. The reference is
// streaming with a KV cache; this port runs the whole latent sequence
// in one pass with a causal mask, which is functionally identical for
// offline encode/decode.
//
// All math is CPU-native — see AudioPrimitives.swift.

import Foundation

// MARK: - RoPE

/// Traditional (interleaved-pair) rotary position embedding, applied to
/// a single head's `[T, headDim]` slice. Mirrors MLX's `RoPE(traditional:
/// true)`.
enum MimiRoPE {
    /// Rotate `x` `[T, headDim]` in place-style and return the result.
    static func apply(
        _ x: [Float], t: Int, headDim: Int,
        base: Float
    ) -> [Float] {
        let half = headDim / 2
        var out = x
        for pos in 0 ..< t {
            let rowBase = pos * headDim
            for i in 0 ..< half {
                // Traditional layout pairs dims (2i, 2i+1).
                let freq = powf(base, -2.0 * Float(i) / Float(headDim))
                let theta = Float(pos) * freq
                let (cs, sn) = (cosf(theta), sinf(theta))
                let a = x[rowBase + 2 * i]
                let b = x[rowBase + 2 * i + 1]
                out[rowBase + 2 * i] = a * cs - b * sn
                out[rowBase + 2 * i + 1] = a * sn + b * cs
            }
        }
        return out
    }
}

// MARK: - Attention

/// A single Mimi self-attention block. `in_proj` is a fused QKV
/// projection (output dim `3 * dModel`); attention is causal with a
/// bounded context window.
struct MimiAttention {
    let inProjW: [Float]  // [3*dModel, dModel]
    let outProjW: [Float]  // [dModel, dModel]
    let numHeads: Int
    let headDim: Int
    let dModel: Int
    let context: Int
    let maxPeriod: Float

    init(weights w: MimiWeights, prefix: String, config c: MimiConfig) throws {
        // Sanitized key: ....self_attn.in_proj.weight
        self.inProjW = try w.floats("\(prefix).in_proj.weight")
        self.outProjW = try w.floats("\(prefix).out_proj.weight")
        self.numHeads = c.numHeads
        self.headDim = c.headDim
        self.dModel = c.dModel
        self.context = c.context
        self.maxPeriod = Float(c.maxPeriod)
    }

    /// Run attention over a `[T, dModel]` sequence (batch 1). Causal,
    /// each query attends to at most `context` past keys.
    func callAsFunction(_ x: [Float], t: Int) -> [Float] {
        // Fused QKV: [T, 3*dModel].
        let qkv = AudioMath.linear(
            x, rows: t, inDim: dModel,
            weight: inProjW, outDim: 3 * dModel,
            bias: nil)
        // Split + per-head RoPE. Keep q/k/v as [head][T, headDim].
        var q = [[Float]](repeating: [], count: numHeads)
        var k = [[Float]](repeating: [], count: numHeads)
        var v = [[Float]](repeating: [], count: numHeads)
        for h in 0 ..< numHeads {
            var qh = [Float](repeating: 0, count: t * headDim)
            var kh = [Float](repeating: 0, count: t * headDim)
            var vh = [Float](repeating: 0, count: t * headDim)
            for pos in 0 ..< t {
                let rowBase = pos * 3 * dModel
                let hOff = h * headDim
                for d in 0 ..< headDim {
                    qh[pos * headDim + d] = qkv[rowBase + hOff + d]
                    kh[pos * headDim + d] = qkv[rowBase + dModel + hOff + d]
                    vh[pos * headDim + d] = qkv[rowBase + 2 * dModel + hOff + d]
                }
            }
            q[h] = MimiRoPE.apply(qh, t: t, headDim: headDim, base: maxPeriod)
            k[h] = MimiRoPE.apply(kh, t: t, headDim: headDim, base: maxPeriod)
            v[h] = vh
        }
        // Scaled dot-product attention per head, causal + windowed.
        let scale = 1.0 / sqrtf(Float(headDim))
        var attnOut = [Float](repeating: 0, count: t * dModel)
        for h in 0 ..< numHeads {
            let qh = q[h]
            let kh = k[h]
            let vh = v[h]
            for i in 0 ..< t {
                // Key window: [max(0, i-context+1), i].
                let lo = max(0, i - context + 1)
                var scores = [Float](repeating: 0, count: i - lo + 1)
                var mx = -Float.greatestFiniteMagnitude
                for j in lo ... i {
                    var dot: Float = 0
                    for d in 0 ..< headDim {
                        dot += qh[i * headDim + d] * kh[j * headDim + d]
                    }
                    let s = dot * scale
                    scores[j - lo] = s
                    if s > mx { mx = s }
                }
                var sum: Float = 0
                for n in 0 ..< scores.count {
                    let e = expf(scores[n] - mx)
                    scores[n] = e
                    sum += e
                }
                let invSum = 1.0 / sum
                let outBase = i * dModel + h * headDim
                for j in lo ... i {
                    let wgt = scores[j - lo] * invSum
                    for d in 0 ..< headDim {
                        attnOut[outBase + d] += wgt * vh[j * headDim + d]
                    }
                }
            }
        }
        // Output projection.
        return AudioMath.linear(
            attnOut, rows: t, inDim: dModel,
            weight: outProjW, outDim: dModel, bias: nil)
    }
}

// MARK: - Transformer layer

/// One pre-norm Mimi transformer layer: LayerNorm → attention →
/// LayerScale residual, then LayerNorm → GELU-MLP → LayerScale residual.
struct MimiTransformerLayer {
    let norm1W: [Float], norm1B: [Float]?
    let norm2W: [Float], norm2B: [Float]?
    let attn: MimiAttention
    let mlpW1: [Float], mlpW2: [Float]
    let scale1: [Float], scale2: [Float]
    let dModel: Int
    let dimFF: Int

    init(weights w: MimiWeights, prefix: String, config c: MimiConfig) throws {
        self.dModel = c.dModel
        self.dimFF = c.dimFeedforward
        self.norm1W = try w.floats("\(prefix).norm1.weight")
        self.norm1B =
            w.has("\(prefix).norm1.bias")
            ? try w.floats("\(prefix).norm1.bias") : nil
        self.norm2W = try w.floats("\(prefix).norm2.weight")
        self.norm2B =
            w.has("\(prefix).norm2.bias")
            ? try w.floats("\(prefix).norm2.bias") : nil
        self.attn = try MimiAttention(
            weights: w, prefix: "\(prefix).self_attn", config: c)
        // No-gating MLP: gating.linear1 / gating.linear2.
        self.mlpW1 = try w.floats("\(prefix).gating.linear1.weight")
        self.mlpW2 = try w.floats("\(prefix).gating.linear2.weight")
        self.scale1 = try w.floats("\(prefix).layer_scale_1.scale")
        self.scale2 = try w.floats("\(prefix).layer_scale_2.scale")
    }

    func callAsFunction(_ x: [Float], t: Int) -> [Float] {
        var out = x
        // ── Attention sub-block ──
        let n1 = AudioMath.layerNorm(
            out, rows: t, dim: dModel,
            weight: norm1W, bias: norm1B)
        let a = attn(n1, t: t)
        for pos in 0 ..< t {
            for d in 0 ..< dModel {
                out[pos * dModel + d] += a[pos * dModel + d] * scale1[d]
            }
        }
        // ── MLP sub-block ──
        let n2 = AudioMath.layerNorm(
            out, rows: t, dim: dModel,
            weight: norm2W, bias: norm2B)
        let h1 = AudioMath.linear(
            n2, rows: t, inDim: dModel,
            weight: mlpW1, outDim: dimFF, bias: nil)
        let act = AudioMath.gelu(h1)
        let h2 = AudioMath.linear(
            act, rows: t, inDim: dimFF,
            weight: mlpW2, outDim: dModel, bias: nil)
        for pos in 0 ..< t {
            for d in 0 ..< dModel {
                out[pos * dModel + d] += h2[pos * dModel + d] * scale2[d]
            }
        }
        return out
    }
}

// MARK: - Projected transformer

/// Mimi's `ProjectedTransformer` — the latent transformer with optional
/// input/output projections. For the standard preset `inputDim ==
/// dModel`, so both projections are absent; the stack is purely the
/// transformer layers.
struct MimiProjectedTransformer {
    let layers: [MimiTransformerLayer]
    let dModel: Int

    init(weights w: MimiWeights, config c: MimiConfig, prefix: String) throws {
        self.dModel = c.dModel
        var ls: [MimiTransformerLayer] = []
        for l in 0 ..< c.numLayers {
            ls.append(
                try MimiTransformerLayer(
                    weights: w,
                    prefix: "\(prefix).transformer.layers.\(l)", config: c))
        }
        self.layers = ls
    }

    /// Run the transformer over an NCL latent `[1, dModel, T]`. Returns
    /// the same NCL layout so it slots between the conv stages.
    func forward(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        let (n, c, t) = (shape[0], shape[1], shape[2])
        precondition(
            n == 1 && c == dModel,
            "MimiProjectedTransformer: expected [1, dModel, T]")
        // NCL -> [T, dModel] sequence.
        var seq = [Float](repeating: 0, count: t * dModel)
        for ch in 0 ..< dModel {
            for pos in 0 ..< t { seq[pos * dModel + ch] = x[ch * t + pos] }
        }
        for layer in layers { seq = layer(seq, t: t) }
        // [T, dModel] -> NCL.
        var out = [Float](repeating: 0, count: x.count)
        for ch in 0 ..< dModel {
            for pos in 0 ..< t { out[ch * t + pos] = seq[pos * dModel + ch] }
        }
        return (out, shape)
    }
}

// MARK: - Quantizer

/// A single Euclidean-distance codebook. Mimi stores the codebook as
/// `embedding_sum` + `cluster_usage`; the effective embedding is
/// `embedding_sum / max(cluster_usage, eps)`.
struct MimiCodebook {
    let embedding: [Float]  // [codebookSize, dim]
    let c2: [Float]  // ½‖embedding‖² per entry
    let codebookSize: Int
    let dim: Int

    private static let epsilon: Float = 1e-5

    init(weights w: MimiWeights, prefix: String, dim: Int) throws {
        let embSum = try w.floats("\(prefix).embedding_sum")
        let usage = try w.floats("\(prefix).cluster_usage")
        self.init(embeddingSum: embSum, clusterUsage: usage, dim: dim)
    }

    /// Build a codebook directly from `embedding_sum` / `cluster_usage`
    /// tables, reconstructing the effective embedding `embedding_sum /
    /// max(cluster_usage, eps)`.
    init(embeddingSum embSum: [Float], clusterUsage usage: [Float], dim: Int) {
        let size = usage.count
        var emb = [Float](repeating: 0, count: embSum.count)
        var c2v = [Float](repeating: 0, count: size)
        for c in 0 ..< size {
            let denom = max(usage[c], Self.epsilon)
            var ss: Float = 0
            for d in 0 ..< dim {
                let v = embSum[c * dim + d] / denom
                emb[c * dim + d] = v
                ss += v * v
            }
            c2v[c] = ss / 2.0
        }
        self.embedding = emb
        self.c2 = c2v
        self.codebookSize = size
        self.dim = dim
    }

    /// Nearest-codebook lookup over a `[T, dim]` matrix. The reference
    /// scores `c2 - x·emb` and takes the argmin — `‖x‖²` is shared
    /// across entries so it drops out.
    func encode(_ x: [Float], rows t: Int) -> [Int32] {
        var indices = [Int32](repeating: 0, count: t)
        for i in 0 ..< t {
            let xBase = i * dim
            var best: Float = .greatestFiniteMagnitude
            var bestIdx = 0
            for c in 0 ..< codebookSize {
                let cBase = c * dim
                var dot: Float = 0
                for d in 0 ..< dim { dot += x[xBase + d] * embedding[cBase + d] }
                let dist = c2[c] - dot
                if dist < best {
                    best = dist
                    bestIdx = c
                }
            }
            indices[i] = Int32(bestIdx)
        }
        return indices
    }

    /// Reconstruct a `[T, dim]` matrix from codes.
    func decode(codes: [Int32]) -> [Float] {
        var out = [Float](repeating: 0, count: codes.count * dim)
        for (i, code) in codes.enumerated() {
            let cBase = Int(code) * dim
            for d in 0 ..< dim { out[i * dim + d] = embedding[cBase + d] }
        }
        return out
    }
}

/// One `VectorQuantization` codebook with its optional project-in /
/// project-out linears. Mimi's codebook dim differs from the latent dim,
/// so the projections are always present.
struct MimiVQ {
    let codebook: MimiCodebook
    let projInW: [Float]?  // [codebookDim, dim]
    let projInDim: (in: Int, out: Int)?
    let projOutW: [Float]?  // [dim, codebookDim]
    let projOutDim: (in: Int, out: Int)?

    init(
        weights w: MimiWeights, prefix: String,
        dim: Int, codebookDim: Int
    ) throws {
        self.codebook = try MimiCodebook(
            weights: w, prefix: "\(prefix).codebook", dim: codebookDim)
        if dim != codebookDim, w.has("\(prefix).project_in.weight") {
            self.projInW = try w.floats("\(prefix).project_in.weight")
            self.projInDim = (dim, codebookDim)
            self.projOutW = try w.floats("\(prefix).project_out.weight")
            self.projOutDim = (codebookDim, dim)
        } else {
            self.projInW = nil
            self.projInDim = nil
            self.projOutW = nil
            self.projOutDim = nil
        }
    }

    /// Encode a `[T, dim]` matrix into codes.
    func encode(_ x: [Float], rows t: Int) -> [Int32] {
        var feat = x
        if let pw = projInW, let pd = projInDim {
            feat = AudioMath.linear(
                x, rows: t, inDim: pd.in,
                weight: pw, outDim: pd.out, bias: nil)
        }
        return codebook.encode(feat, rows: t)
    }

    /// Decode codes back into a `[T, dim]` matrix.
    func decode(codes: [Int32]) -> [Float] {
        let q = codebook.decode(codes: codes)
        if let pw = projOutW, let pd = projOutDim {
            return AudioMath.linear(
                q, rows: codes.count, inDim: pd.in,
                weight: pw, outDim: pd.out, bias: nil)
        }
        return q
    }
}

/// A `ResidualVectorQuantizer` — a stack of `MimiVQ` codebooks applied
/// to successive residuals, wrapped by 1×1 input/output projection
/// convs.
struct MimiResidualVQ {
    let layers: [MimiVQ]
    let inputProjW: [Float]?  // [dim, inputDim, 1] -> flat [dim*inputDim]
    let inputProjDims: (inDim: Int, outDim: Int)?
    let outputProjW: [Float]?
    let outputProjDims: (inDim: Int, outDim: Int)?

    init(
        weights w: MimiWeights, prefix: String, config c: MimiConfig,
        nq: Int
    ) throws {
        var ls: [MimiVQ] = []
        for q in 0 ..< nq {
            ls.append(
                try MimiVQ(
                    weights: w, prefix: "\(prefix).vq.layers.\(q)",
                    dim: c.seanetDim, codebookDim: c.quantizerDim))
        }
        self.layers = ls
        // input_proj / output_proj are 1×1 convs (bias-free). Their
        // weight ships as [Cout, 1, Cin] (MLX NLC, k=1); a 1×1 conv is
        // just a linear, so flatten to [Cout, Cin].
        if w.has("\(prefix).input_proj.weight") {
            let raw = try w.floats("\(prefix).input_proj.weight")
            let s = try w.shape("\(prefix).input_proj.weight")
            self.inputProjW = MimiResidualVQ.conv1x1ToLinear(raw, shape: s)
            self.inputProjDims = (s[2], s[0])
        } else {
            self.inputProjW = nil
            self.inputProjDims = nil
        }
        if w.has("\(prefix).output_proj.weight") {
            let raw = try w.floats("\(prefix).output_proj.weight")
            let s = try w.shape("\(prefix).output_proj.weight")
            self.outputProjW = MimiResidualVQ.conv1x1ToLinear(raw, shape: s)
            self.outputProjDims = (s[2], s[0])
        } else {
            self.outputProjW = nil
            self.outputProjDims = nil
        }
    }

    /// Flatten an MLX `[Cout, K=1, Cin]` 1×1-conv weight to a `[Cout,
    /// Cin]` linear weight.
    private static func conv1x1ToLinear(_ w: [Float], shape: [Int]) -> [Float] {
        // K is 1, so [Cout, 1, Cin] is already [Cout, Cin] in row order.
        precondition(
            shape.count == 3 && shape[1] == 1,
            "MimiResidualVQ: expected a 1×1 conv weight")
        return w
    }

    /// Encode a `[T, dim]` residual matrix into `nq` code streams.
    func encode(_ x: [Float], rows t: Int) -> [[Int32]] {
        // Apply the 1×1 input projection (if any).
        var feat = x
        if let pw = inputProjW, let pd = inputProjDims {
            feat = AudioMath.linear(
                x, rows: t, inDim: pd.inDim,
                weight: pw, outDim: pd.outDim, bias: nil)
        }
        var residual = feat
        var codes: [[Int32]] = []
        for layer in layers {
            let idx = layer.encode(residual, rows: t)
            let q = layer.decode(codes: idx)
            for i in 0 ..< residual.count { residual[i] -= q[i] }
            codes.append(idx)
        }
        return codes
    }

    /// Decode `nq` code streams into a `[T, dim]` matrix.
    func decode(codes: [[Int32]]) -> [Float] {
        guard let first = codes.first else { return [] }
        let t = first.count
        var acc = layers[0].decode(codes: first)
        for q in 1 ..< codes.count {
            let d = layers[q].decode(codes: codes[q])
            for i in 0 ..< acc.count { acc[i] += d[i] }
        }
        if let pw = outputProjW, let pd = outputProjDims {
            return AudioMath.linear(
                acc, rows: t, inDim: pd.inDim,
                weight: pw, outDim: pd.outDim, bias: nil)
        }
        return acc
    }
}

/// Mimi's `SplitResidualVectorQuantizer` — a 1-codebook "semantic" RVQ
/// plus an `(nq-1)`-codebook "acoustic" RVQ. The two streams are
/// concatenated; decoding sums their reconstructions.
struct MimiSplitRVQ {
    let rvqFirst: MimiResidualVQ
    let rvqRest: MimiResidualVQ
    let nq: Int
    let dim: Int

    init(weights w: MimiWeights, config c: MimiConfig) throws {
        self.nq = c.quantizerNQ
        self.dim = c.seanetDim
        self.rvqFirst = try MimiResidualVQ(
            weights: w, prefix: "quantizer.rvq_first", config: c, nq: 1)
        self.rvqRest = try MimiResidualVQ(
            weights: w, prefix: "quantizer.rvq_rest", config: c,
            nq: max(c.quantizerNQ - 1, 0))
    }

    /// Encode an NCL latent `[1, dim, T]` into `nq` code streams.
    func encode(_ z: [Float], shape: [Int]) throws -> [[Int32]] {
        let (n, c, t) = (shape[0], shape[1], shape[2])
        precondition(
            n == 1 && c == dim,
            "MimiSplitRVQ.encode: expected [1, dim, T]")
        // NCL -> [T, dim].
        var seq = [Float](repeating: 0, count: t * dim)
        for ch in 0 ..< dim {
            for pos in 0 ..< t { seq[pos * dim + ch] = z[ch * t + pos] }
        }
        var codes = rvqFirst.encode(seq, rows: t)
        if nq > 1 {
            codes.append(contentsOf: rvqRest.encode(seq, rows: t))
        }
        return codes
    }

    /// Decode `nq` code streams into an NCL latent `[1, dim, T]`.
    func decode(codes: [[Int32]]) throws -> (data: [Float], shape: [Int]) {
        guard let first = codes.first else {
            throw MimiError.shapeMismatch("empty code list")
        }
        let t = first.count
        var acc = rvqFirst.decode(codes: Array(codes.prefix(1)))  // [T, dim]
        if nq > 1 && codes.count > 1 {
            let rest = rvqRest.decode(codes: Array(codes.dropFirst()))
            for i in 0 ..< acc.count { acc[i] += rest[i] }
        }
        // [T, dim] -> NCL.
        var out = [Float](repeating: 0, count: dim * t)
        for pos in 0 ..< t {
            for ch in 0 ..< dim { out[ch * t + pos] = acc[pos * dim + ch] }
        }
        return (out, [1, dim, t])
    }
}
