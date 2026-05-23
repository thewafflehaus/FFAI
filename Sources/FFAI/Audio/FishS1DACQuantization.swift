// FishS1DACQuantization — RVQ quantizer, downsample/upsample stages, and
// the window-limited transformer (pre/post-module) used by FishS1DAC.
//
// Port of the quantization stack in:
//   mlx-audio-swift/Sources/MLXAudioCodecs/FishS1DAC/FishS1DACQuantization.swift
//   mlx-audio-swift/Sources/MLXAudioCodecs/FishS1DAC/FishS1DACTransformer.swift
//
// This file implements the DECODE path only — the primary operation is
// `FishS1DACDownsampleRVQ.decode(indices:config:weights:)`.
//
// CPU fallback note: the transformer and ConvNeXt blocks run on CPU via
// `AudioMath` helpers. A metaltile kernel port for the dilated depthwise
// convolutions would be the natural next step (see TODO below).
// TODO(metaltile): port FishS1ConvNeXt depthwise dilated Conv1d kernel.

import Foundation

// MARK: - Weight accessor

/// Thin wrapper over `SafeTensorsBundle` providing `[Float]` reads with
/// the same interface used by SNAC / DescriptDAC.
struct FishS1DACWeights {
    let bundle: SafeTensorsBundle

    func floats(_ key: String) throws -> [Float] {
        AudioMath.floats(try bundle.tensor(named: key))
    }

    func shape(_ key: String) throws -> [Int] {
        try bundle.tensor(named: key).shape
    }

    func has(_ key: String) -> Bool { bundle.has(key) }

    /// Reconstruct a weight-normalized Conv1d at `prefix`.
    /// PyTorch weight_norm stores `weight_g` (magnitude, shape [Cout,1,1])
    /// and `weight_v` (direction, same shape as the full weight).
    func wnConv1d(prefix: String, stride: Int, padding: Int,
                  dilation: Int, groups: Int) throws -> SNACWNConv1d {
        let gKey = "\(prefix).weight_g"
        let vKey = "\(prefix).weight_v"
        guard has(gKey), has(vKey) else {
            throw FishS1DACError.missingWeights("\(prefix) weight_g/weight_v")
        }
        let g = try floats(gKey)
        let v = try floats(vKey)
        let vShape = try shape(vKey)
        let weight = WeightNorm.effectiveWeight(g: g, v: v, shape: vShape, exceptDim: 0)
        let bias = has("\(prefix).bias") ? try floats("\(prefix).bias") : nil
        return SNACWNConv1d(weight: weight, wShape: vShape, bias: bias,
                            stride: stride, padding: padding,
                            dilation: dilation, groups: groups)
    }

    /// Reconstruct a weight-normalized transposed Conv1d at `prefix`.
    func wnConvTranspose1d(prefix: String, stride: Int, padding: Int,
                           outputPadding: Int, groups: Int) throws -> SNACWNConvTranspose1d {
        let gKey = "\(prefix).weight_g"
        let vKey = "\(prefix).weight_v"
        guard has(gKey), has(vKey) else {
            throw FishS1DACError.missingWeights("\(prefix) weight_g/weight_v")
        }
        let g = try floats(gKey)
        let v = try floats(vKey)
        let vShape = try shape(vKey)
        let weight = WeightNorm.effectiveWeight(g: g, v: v, shape: vShape, exceptDim: 0)
        let bias = has("\(prefix).bias") ? try floats("\(prefix).bias") : nil
        return SNACWNConvTranspose1d(weight: weight, wShape: vShape, bias: bias,
                                     stride: stride, padding: padding,
                                     outputPadding: outputPadding, groups: groups)
    }
}

// MARK: - RoPE

/// Non-interleaved (pair-split) RoPE used by the FishS1 transformer.
/// Mirrors `fishS1ApplyRotaryEmb` in the reference.
enum FishS1RoPE {
    /// Apply RoPE to a `[T, nHead, headDim]` tensor (row-major).
    /// `freqs` is `[T, headDim/2]` containing (cos, sin) interleaved as
    /// precomputed from `fishS1PrecomputeFreqsCis`.
    ///
    /// Layout: for each (pos, head, dim) pair, the halved indices are
    /// rotated. Real + imaginary parts are split into first and second half
    /// of headDim.
    static func apply(_ x: [Float], t: Int, nHead: Int, headDim: Int,
                      base: Double) -> [Float] {
        let half = headDim / 2
        var out = x
        for pos in 0..<t {
            for h in 0..<nHead {
                let rowBase = (pos * nHead + h) * headDim
                for i in 0..<half {
                    let freq = 1.0 / pow(base, Double(2 * i) / Double(headDim))
                    let theta = Float(Double(pos) * freq)
                    let cs = cosf(theta)
                    let sn = sinf(theta)
                    let a = x[rowBase + i]
                    let b = x[rowBase + half + i]
                    out[rowBase + i]        = a * cs - b * sn
                    out[rowBase + half + i] = a * sn + b * cs
                }
            }
        }
        return out
    }
}

// MARK: - ConvNeXt block (CPU)

/// FishS1 ConvNeXt block used in the downsample/upsample stages of the
/// quantizer. Architecture: depthwise causal Conv1d → LayerNorm →
/// pw-expand Linear → GELU → pw-reduce Linear → (optional γ scale) →
/// residual.
///
/// CPU implementation. All inner loops run on the calling thread because
/// these blocks are part of a one-shot decode, not an autoregressive loop.
/// TODO(metaltile): replace with a GPU kernel once the depthwise dilated
/// Conv1d kernel is available in metaltile-std.
struct FishS1DACConvNeXtBlock {
    let dwConvW: [Float]       // [dim, 1, kernelSize] (depthwise, groups=dim)
    let dwConvWShape: [Int]
    let dwConvBias: [Float]?
    let normW: [Float]         // [dim]
    let normB: [Float]?        // [dim] or nil
    let pw1W: [Float]          // [4*dim, dim]
    let pw1B: [Float]?
    let pw2W: [Float]          // [dim, 4*dim]
    let pw2B: [Float]?
    let gamma: [Float]?        // [dim] layer scale
    let dim: Int
    let kernelSize: Int

    init(weights w: FishS1DACWeights, prefix: String, dim: Int) throws {
        self.dim = dim
        // Depthwise causal Conv1d: conv.weight + bias
        // Key layout: <prefix>.dwconv.conv.weight / .bias
        let dwPrefix = "\(prefix).dwconv.conv"
        self.dwConvW     = try w.floats("\(dwPrefix).weight")
        self.dwConvWShape = try w.shape("\(dwPrefix).weight")
        self.dwConvBias  = w.has("\(dwPrefix).bias") ? try w.floats("\(dwPrefix).bias") : nil
        self.kernelSize  = dwConvWShape.last ?? 7

        // LayerNorm
        self.normW = try w.floats("\(prefix).norm.weight")
        self.normB = w.has("\(prefix).norm.bias") ? try w.floats("\(prefix).norm.bias") : nil

        // Pointwise convs serialized as Linear
        let mlpRatio = 4
        self.pw1W = try w.floats("\(prefix).pwconv1.weight")
        self.pw1B = w.has("\(prefix).pwconv1.bias") ? try w.floats("\(prefix).pwconv1.bias") : nil
        self.pw2W = try w.floats("\(prefix).pwconv2.weight")
        self.pw2B = w.has("\(prefix).pwconv2.bias") ? try w.floats("\(prefix).pwconv2.bias") : nil
        self.gamma = w.has("\(prefix).gamma") ? try w.floats("\(prefix).gamma") : nil

        _ = mlpRatio
    }

    /// Forward: input is NCL `[1, dim, T]`, output is `[1, dim, T]`.
    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        let (n, c, l) = (shape[0], shape[1], shape[2])
        precondition(n == 1 && c == dim, "FishS1DACConvNeXtBlock: unexpected shape \(shape)")

        // Causal depthwise Conv1d (same-length output via causal padding).
        // kernel is [dim, 1, K] in PyTorch / [dim, K] effectively.
        // We treat it as groups=dim conv1d with causal (left) padding.
        let k = kernelSize
        let leftPad = k - 1   // causal: pad left by (K-1)
        let paddedL = l + leftPad
        var padded = [Float](repeating: 0, count: c * paddedL)
        for ch in 0..<c {
            let src = ch * l
            let dst = ch * paddedL + leftPad
            for t in 0..<l { padded[dst + t] = x[src + t] }
        }

        // Apply depthwise conv: each channel uses its own kernel row.
        // dwConvW shape is [dim, 1, K] -> weight for channel ch is at
        // [ch * K .. ch * K + K).
        var dw = [Float](repeating: 0, count: c * l)
        let biasVec = dwConvBias ?? [Float](repeating: 0, count: c)
        for ch in 0..<c {
            let wBase = ch * k
            let pBase = ch * paddedL
            let oBase = ch * l
            for t in 0..<l {
                var acc = biasVec[ch]
                for kk in 0..<k { acc += padded[pBase + t + kk] * dwConvW[wBase + kk] }
                dw[oBase + t] = acc
            }
        }

        // Transpose NCL -> NLC for LayerNorm + Linear ops: [1, T, dim].
        var seq = [Float](repeating: 0, count: l * c)
        for ch in 0..<c {
            for t in 0..<l { seq[t * c + ch] = dw[ch * l + t] }
        }

        // LayerNorm over last dim.
        seq = AudioMath.layerNorm(seq, rows: l, dim: c, weight: normW, bias: normB)

        // PW Linear 1: [T, dim] -> [T, 4*dim], then GELU.
        let expandedDim = pw1W.count / c  // infer from weight shape
        var h = AudioMath.linear(seq, rows: l, inDim: c,
                                 weight: pw1W, outDim: expandedDim, bias: pw1B)
        h = AudioMath.gelu(h)

        // PW Linear 2: [T, 4*dim] -> [T, dim].
        var out = AudioMath.linear(h, rows: l, inDim: expandedDim,
                                   weight: pw2W, outDim: c, bias: pw2B)

        // Optional layer scale.
        if let g = gamma {
            for t in 0..<l {
                for ch in 0..<c { out[t * c + ch] *= g[ch] }
            }
        }

        // Add residual (still in NLC layout).
        var residualNLC = [Float](repeating: 0, count: l * c)
        for ch in 0..<c {
            for t in 0..<l { residualNLC[t * c + ch] = x[ch * l + t] }
        }
        for i in 0..<out.count { out[i] += residualNLC[i] }

        // Transpose back to NCL [1, dim, T].
        var outNCL = [Float](repeating: 0, count: c * l)
        for t in 0..<l {
            for ch in 0..<c { outNCL[ch * l + t] = out[t * c + ch] }
        }
        return (outNCL, shape)
    }
}

// MARK: - Quantizer transformer (window-limited, causal)

/// RMS norm as used in the FishS1 transformer. Operates on `[T, dim]`
/// row-major data.
struct FishS1DACRMSNorm {
    let weight: [Float]   // [dim]
    let eps: Float

    func callAsFunction(_ x: [Float], rows t: Int, dim: Int) -> [Float] {
        var out = [Float](repeating: 0, count: x.count)
        for i in 0..<t {
            let base = i * dim
            var ss: Float = 0
            for d in 0..<dim { ss += x[base + d] * x[base + d] }
            let rms = sqrtf(ss / Float(dim) + eps)
            for d in 0..<dim { out[base + d] = (x[base + d] / rms) * weight[d] }
        }
        return out
    }
}

/// SiLU (swish) gated MLP used in FishS1 transformer blocks.
struct FishS1DACMLP {
    let w1: [Float]           // [intermediateSize, dim]
    let w2: [Float]           // [dim, intermediateSize]
    let w3: [Float]           // [intermediateSize, dim] (gate)
    let dim: Int
    let intermediateSize: Int

    func callAsFunction(_ x: [Float], rows t: Int) -> [Float] {
        let h1 = AudioMath.linear(x, rows: t, inDim: dim,
                                   weight: w1, outDim: intermediateSize, bias: nil)
        let h3 = AudioMath.linear(x, rows: t, inDim: dim,
                                   weight: w3, outDim: intermediateSize, bias: nil)
        // SiLU(h1) * h3
        let gated = zip(AudioMath.silu(h1), h3).map { $0 * $1 }
        return AudioMath.linear(gated, rows: t, inDim: intermediateSize,
                                weight: w2, outDim: dim, bias: nil)
    }
}

/// One FishS1 transformer block: RMSNorm → attention (RoPE, layer scale) →
/// residual, then RMSNorm → MLP (SiLU-gate) → layer scale → residual.
struct FishS1DACTransformerBlock {
    let attnNorm: FishS1DACRMSNorm
    let ffnNorm: FishS1DACRMSNorm
    let wqkv: [Float]              // fused [kvTotal, dim] where kvTotal = (nHead + 2*nKVHead)*headDim
    let wo: [Float]                // [dim, nHead*headDim]
    let mlp: FishS1DACMLP
    let attnLayerScale: [Float]    // [dim]
    let ffnLayerScale: [Float]     // [dim]
    let nHead: Int
    let nKVHead: Int
    let headDim: Int
    let dim: Int
    let ropeBase: Double

    func callAsFunction(_ x: [Float], t: Int, windowSize: Int) -> [Float] {
        // ─ Attention ─
        let n1 = attnNorm(x, rows: t, dim: dim)

        // QKV projection
        let kvSize = nKVHead * headDim
        let qSize  = nHead   * headDim
        let qkvTotalDim = qSize + 2 * kvSize
        let qkv = AudioMath.linear(n1, rows: t, inDim: dim,
                                   weight: wqkv, outDim: qkvTotalDim, bias: nil)

        // Split Q, K, V (non-interleaved)
        var q = [Float](repeating: 0, count: t * qSize)
        var k = [Float](repeating: 0, count: t * kvSize)
        var v = [Float](repeating: 0, count: t * kvSize)
        for pos in 0..<t {
            let src = pos * qkvTotalDim
            let qDst = pos * qSize
            let kDst = pos * kvSize
            let vDst = pos * kvSize
            for i in 0..<qSize  { q[qDst + i] = qkv[src + i] }
            for i in 0..<kvSize { k[kDst + i] = qkv[src + qSize + i] }
            for i in 0..<kvSize { v[vDst + i] = qkv[src + qSize + kvSize + i] }
        }

        // Reshape to [T, nHead/nKVHead, headDim] for RoPE, then apply
        // RoPE to Q and K.
        q = FishS1RoPE.apply(q, t: t, nHead: nHead,   headDim: headDim, base: ropeBase)
        k = FishS1RoPE.apply(k, t: t, nHead: nKVHead, headDim: headDim, base: ropeBase)

        // GQA repeat: expand K, V from nKVHead to nHead.
        let repeatFactor = nHead / nKVHead
        var kExp = [Float](repeating: 0, count: t * qSize)
        var vExp = [Float](repeating: 0, count: t * qSize)
        if repeatFactor == 1 {
            kExp = k
            vExp = v
        } else {
            for pos in 0..<t {
                for h in 0..<nHead {
                    let kvHead = h / repeatFactor
                    let src = pos * kvSize + kvHead * headDim
                    let dst = pos * qSize  + h * headDim
                    for d in 0..<headDim {
                        kExp[dst + d] = k[src + d]
                        vExp[dst + d] = v[src + d]
                    }
                }
            }
        }

        // Causal window-limited scaled dot-product attention.
        let scale = 1.0 / sqrtf(Float(headDim))
        var attnOut = [Float](repeating: 0, count: t * dim)
        for pos in 0..<t {
            let lo = max(0, pos - windowSize + 1)
            for h in 0..<nHead {
                var scores = [Float](repeating: 0, count: pos - lo + 1)
                var mx: Float = -.greatestFiniteMagnitude
                let qBase = pos * qSize + h * headDim
                for j in lo...pos {
                    let kBase = j * qSize + h * headDim
                    var dot: Float = 0
                    for d in 0..<headDim { dot += q[qBase + d] * kExp[kBase + d] }
                    let s = dot * scale
                    scores[j - lo] = s
                    if s > mx { mx = s }
                }
                var sum: Float = 0
                for n in 0..<scores.count {
                    let e = expf(scores[n] - mx)
                    scores[n] = e
                    sum += e
                }
                let invSum = 1.0 / sum
                let outBase = pos * dim + h * headDim
                for j in lo...pos {
                    let wgt = scores[j - lo] * invSum
                    let vBase = j * qSize + h * headDim
                    for d in 0..<headDim {
                        attnOut[outBase + d] += wgt * vExp[vBase + d]
                    }
                }
            }
        }

        // Output projection: [T, nHead*headDim] -> [T, dim]
        let proj = AudioMath.linear(attnOut, rows: t, inDim: qSize,
                                    weight: wo, outDim: dim, bias: nil)

        // Residual + layer scale
        var h = x
        for pos in 0..<t {
            let base = pos * dim
            for d in 0..<dim { h[base + d] += proj[base + d] * attnLayerScale[d] }
        }

        // ─ MLP ─
        let n2 = ffnNorm(h, rows: t, dim: dim)
        let mlpOut = mlp(n2, rows: t)

        // Residual + layer scale
        for pos in 0..<t {
            let base = pos * dim
            for d in 0..<dim { h[base + d] += mlpOut[base + d] * ffnLayerScale[d] }
        }
        return h
    }
}

/// Window-limited FishS1 transformer. Wraps N `FishS1DACTransformerBlock`s
/// plus optional input/output projections and a final RMSNorm. Operates
/// on NCL `[1, dim, T]` tensors (channels-first); internally transposes
/// to `[T, dim]` for the attention + MLP ops.
struct FishS1DACWindowLimitedTransformer {
    let layers: [FishS1DACTransformerBlock]
    let norm: FishS1DACRMSNorm
    let inputProjW: [Float]?     // [dim, inputDim], nil if inputDim == dim
    let outputProjW: [Float]?    // [inputDim, dim], nil if inputDim == dim
    let inputDim: Int
    let dim: Int
    let windowSize: Int

    /// Run the transformer over an NCL latent `[1, dim, T]`. Returns NCL.
    func forward(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        let (n, c, t) = (shape[0], shape[1], shape[2])
        precondition(n == 1 && c == inputDim,
                     "FishS1DACWindowLimitedTransformer: expected [1, \(inputDim), T], got \(shape)")

        // NCL [1, inputDim, T] -> [T, inputDim]
        var seq = [Float](repeating: 0, count: t * inputDim)
        for ch in 0..<inputDim {
            for pos in 0..<t { seq[pos * inputDim + ch] = x[ch * t + pos] }
        }

        // Optional input projection: [T, inputDim] -> [T, dim]
        var hidden: [Float]
        if let w = inputProjW {
            hidden = AudioMath.linear(seq, rows: t, inDim: inputDim,
                                      weight: w, outDim: dim, bias: nil)
        } else {
            hidden = seq
        }

        // Transformer blocks
        for layer in layers {
            hidden = layer(hidden, t: t, windowSize: windowSize)
        }

        // Final RMSNorm
        hidden = norm(hidden, rows: t, dim: dim)

        // Optional output projection: [T, dim] -> [T, inputDim]
        var outSeq: [Float]
        if let w = outputProjW {
            outSeq = AudioMath.linear(hidden, rows: t, inDim: dim,
                                       weight: w, outDim: inputDim, bias: nil)
        } else {
            outSeq = hidden
        }

        // [T, inputDim] -> NCL [1, inputDim, T]
        var outNCL = [Float](repeating: 0, count: inputDim * t)
        for ch in 0..<inputDim {
            for pos in 0..<t { outNCL[ch * t + pos] = outSeq[pos * inputDim + ch] }
        }
        return (outNCL, shape)
    }
}

// MARK: - Downsample / Upsample stages

/// FishS1 downsample stage: causal strided Conv1d + ConvNeXt block.
struct FishS1DACDownsampleStage {
    let conv: SNACWNConv1d   // strided causal conv (reuses SNACWNConv1d for CPU math)
    let block: FishS1DACConvNeXtBlock
    let inputDim: Int
    let outputDim: Int
    let factor: Int

    init(weights w: FishS1DACWeights, prefix: String,
         inputDim: Int, outputDim: Int, factor: Int) throws {
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.factor = factor

        // The causal Conv1d weight is stored under the CausalConvNet path
        // (prefix.conv.weight + prefix.conv.bias).
        let convKey = "\(prefix).0.conv"
        let rawW = try w.floats("\(convKey).weight")
        let rawWShape = try w.shape("\(convKey).weight")
        let bias = w.has("\(convKey).bias") ? try w.floats("\(convKey).bias") : nil
        // Causal padding applied externally: left pad = (factor - 1)
        self.conv = SNACWNConv1d(weight: rawW, wShape: rawWShape, bias: bias,
                                  stride: factor, padding: 0,
                                  dilation: 1, groups: 1)
        self.block = try FishS1DACConvNeXtBlock(
            weights: w, prefix: "\(prefix).1", dim: outputDim)
    }

    /// Forward: input NCL `[1, inputDim, T]` -> `[1, outputDim, T/factor]`.
    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        let (_, _, l) = (shape[0], shape[1], shape[2])
        let leftPad = factor - 1
        // Causal left-pad the input before the strided conv.
        let (padded, paddedShape) = AudioMath.zeroPad1d(x, shape: shape,
                                                         left: leftPad, right: 0)
        _ = l
        let (convOut, convShape) = conv(padded, shape: paddedShape)
        return block(convOut, shape: convShape)
    }
}

/// FishS1 upsample stage: causal transposed Conv1d + ConvNeXt block.
struct FishS1DACUpsampleStage {
    let conv: SNACWNConvTranspose1d
    let block: FishS1DACConvNeXtBlock
    let inputDim: Int
    let outputDim: Int
    let factor: Int

    init(weights w: FishS1DACWeights, prefix: String,
         inputDim: Int, outputDim: Int, factor: Int) throws {
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.factor = factor

        // Causal transposed Conv: weight at prefix.0.conv
        let convKey = "\(prefix).0.conv"
        let rawW = try w.floats("\(convKey).weight")
        let rawWShape = try w.shape("\(convKey).weight")
        let bias = w.has("\(convKey).bias") ? try w.floats("\(convKey).bias") : nil
        // outputPadding from the reference: stride > 1 → outputPadding = 0 for causal
        self.conv = SNACWNConvTranspose1d(weight: rawW, wShape: rawWShape, bias: bias,
                                          stride: factor, padding: 0,
                                          outputPadding: 0, groups: 1)
        self.block = try FishS1DACConvNeXtBlock(
            weights: w, prefix: "\(prefix).1", dim: outputDim)
    }

    /// Forward: input NCL `[1, inputDim, T]` -> `[1, outputDim, T*factor]`.
    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        let (transOut, transShape) = conv(x, shape: shape)
        // Causal trim: remove trailing (factor - 1) samples from transposed conv output.
        let trimRight = factor - 1
        let (_, c, lOut) = (transShape[0], transShape[1], transShape[2])
        let trimmedL = lOut - trimRight
        guard trimmedL > 0 else {
            return block(transOut, shape: transShape)
        }
        var trimmed = [Float](repeating: 0, count: c * trimmedL)
        for ch in 0..<c {
            let src = ch * lOut
            let dst = ch * trimmedL
            for t in 0..<trimmedL { trimmed[dst + t] = transOut[src + t] }
        }
        let trimmedShape = [transShape[0], c, trimmedL]
        return block(trimmed, shape: trimmedShape)
    }
}

// MARK: - Vector quantizer codebook

/// A single FishS1 VQ codebook with WN-Conv in/out projections.
/// Decode-only: reconstructs the quantized latent from integer codes.
struct FishS1DACVectorQuantize {
    let inProj: SNACWNConv1d    // [codebookDim, inputDim, 1]
    let outProj: SNACWNConv1d   // [inputDim, codebookDim, 1]
    let codebook: [Float]       // [codebookSize, codebookDim]
    let codebookSize: Int
    let codebookDim: Int

    init(weights w: FishS1DACWeights, prefix: String,
         codebookSize: Int, codebookDim: Int) throws {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.inProj  = try w.wnConv1d(prefix: "\(prefix).in_proj",
                                       stride: 1, padding: 0, dilation: 1, groups: 1)
        self.outProj = try w.wnConv1d(prefix: "\(prefix).out_proj",
                                       stride: 1, padding: 0, dilation: 1, groups: 1)
        // Codebook stored as an Embedding: weight shape [codebookSize, codebookDim].
        self.codebook = try w.floats("\(prefix).codebook.weight")
    }

    /// Decode an integer code stream into a quantized latent `[1, inputDim, T]`.
    func decode(codes: [Int32]) -> (data: [Float], shape: [Int]) {
        let t = codes.count
        // Look up codebook entries → [codebookDim, T] (NCL layout).
        var zLatent = [Float](repeating: 0, count: codebookDim * t)
        for i in 0..<t {
            let cBase = Int(codes[i]) * codebookDim
            for c in 0..<codebookDim { zLatent[c * t + i] = codebook[cBase + c] }
        }
        // out_proj: [1, codebookDim, T] -> [1, inputDim, T]
        return outProj(zLatent, shape: [1, codebookDim, t])
    }
}

// MARK: - Residual VQ

/// A residual vector quantizer stack (n codebooks, single temporal scale).
struct FishS1DACResidualVQ {
    let quantizers: [FishS1DACVectorQuantize]
    let nCodebooks: Int
    let codebookSize: Int

    init(weights w: FishS1DACWeights, prefix: String,
         nCodebooks: Int, codebookSize: Int, codebookDim: Int,
         inputDim: Int) throws {
        self.nCodebooks = nCodebooks
        self.codebookSize = codebookSize
        var qs: [FishS1DACVectorQuantize] = []
        for i in 0..<nCodebooks {
            qs.append(try FishS1DACVectorQuantize(
                weights: w, prefix: "\(prefix).quantizers.\(i)",
                codebookSize: codebookSize, codebookDim: codebookDim))
        }
        _ = inputDim
        self.quantizers = qs
    }

    /// Decode `nCodebooks` code streams (each `[T]`) into a summed latent
    /// `[1, inputDim, T]`.
    func fromCodes(codes: [[Int32]]) -> (data: [Float], shape: [Int]) {
        var zQ: [Float] = []
        var zShape: [Int] = []
        let count = min(quantizers.count, codes.count)
        for i in 0..<count {
            let (zQI, sI) = quantizers[i].decode(codes: codes[i])
            if zQ.isEmpty {
                zQ = zQI
                zShape = sI
            } else {
                for j in 0..<zQ.count { zQ[j] += zQI[j] }
            }
        }
        _ = zShape
        return (zQ, zShape.isEmpty ? [1, 0, 0] : zShape)
    }
}

// MARK: - Downsampled RVQ

/// The full FishS1 downsampling-RVQ quantizer with semantic + residual
/// codebooks, pre/post transformer modules, and upsample stages.
///
/// Decode path:
///   1. semantic codes (index 0)   → `semanticQuantizer.fromCodes`
///   2. residual codes (indices 1…) → `quantizer.fromCodes`
///   3. sum semantic + residual latents → `postModule` transformer → upsample stages
struct FishS1DACDownsampleRVQ {
    let semanticQuantizer: FishS1DACResidualVQ
    let quantizer: FishS1DACResidualVQ
    let postModule: FishS1DACWindowLimitedTransformer?
    let upsampleStages: [FishS1DACUpsampleStage]
    let nCodebooks: Int
    let semanticCodebookSize: Int
    let residualCodebookSize: Int

    init(weights w: FishS1DACWeights, config: FishS1DACConfig) throws {
        let latentDim = config.latentDim
        let resolvedDims: [Int]
        if let dd = config.downsampleDims {
            resolvedDims = dd
        } else {
            resolvedDims = Array(repeating: latentDim, count: config.downsampleFactor.count)
        }
        let allDims = [latentDim] + resolvedDims

        self.nCodebooks = config.nCodebooks
        self.semanticCodebookSize = config.semanticCodebookSize
        self.residualCodebookSize = config.codebookSize

        // Semantic quantizer: 1 codebook at semanticCodebookSize.
        self.semanticQuantizer = try FishS1DACResidualVQ(
            weights: w, prefix: "quantizer.semantic_quantizer",
            nCodebooks: 1, codebookSize: config.semanticCodebookSize,
            codebookDim: config.codebookDim, inputDim: latentDim)

        // Residual quantizer: nCodebooks codebooks.
        self.quantizer = try FishS1DACResidualVQ(
            weights: w, prefix: "quantizer.quantizer",
            nCodebooks: config.nCodebooks, codebookSize: config.codebookSize,
            codebookDim: config.codebookDim, inputDim: latentDim)

        // Post-module transformer (window-limited).
        let hasTF = config.quantizerTransformerLayers > 0
        if hasTF {
            self.postModule = try FishS1DACDownsampleRVQ.loadTransformer(
                weights: w, prefix: "quantizer.post_module",
                config: config, inputDim: latentDim)
        } else {
            self.postModule = nil
        }

        // Upsample stages (reversed downsample order).
        var ups: [FishS1DACUpsampleStage] = []
        for (idx, factor) in config.downsampleFactor.enumerated().reversed() {
            ups.append(try FishS1DACUpsampleStage(
                weights: w,
                prefix: "quantizer.upsample.\(config.downsampleFactor.count - 1 - idx)",
                inputDim: allDims[idx + 1],
                outputDim: allDims[idx],
                factor: factor))
        }
        self.upsampleStages = ups
    }

    /// Load the window-limited transformer from checkpoint.
    private static func loadTransformer(
        weights w: FishS1DACWeights,
        prefix: String,
        config: FishS1DACConfig,
        inputDim: Int
    ) throws -> FishS1DACWindowLimitedTransformer {
        let dim = config.quantizerTransformerDim
        let nLayer = config.quantizerTransformerLayers
        let nHead = config.quantizerTransformerHeads
        let headDim = config.quantizerTransformerHeadDim
        // nKVHead: FishS1 transformer uses full GQA with nLocalHeads == nHead
        // in the quantizer (no KV reduction).
        let nKVHead = nHead
        let ropeBase: Double = 10_000

        var layers: [FishS1DACTransformerBlock] = []
        for i in 0..<nLayer {
            let lp = "\(prefix).layers.\(i)"
            let attnNorm = FishS1DACRMSNorm(
                weight: try w.floats("\(lp).attention_norm.weight"),
                eps: 1e-5)
            let ffnNorm = FishS1DACRMSNorm(
                weight: try w.floats("\(lp).ffn_norm.weight"),
                eps: 1e-5)
            let wqkv = try w.floats("\(lp).attention.wqkv.weight")
            let wo   = try w.floats("\(lp).attention.wo.weight")
            let mlp = FishS1DACMLP(
                w1: try w.floats("\(lp).feed_forward.w1.weight"),
                w2: try w.floats("\(lp).feed_forward.w2.weight"),
                w3: try w.floats("\(lp).feed_forward.w3.weight"),
                dim: dim,
                intermediateSize: config.quantizerTransformerIntermediateSize)
            let attnScale = try w.floats("\(lp).attention_layer_scale.gamma")
            let ffnScale  = try w.floats("\(lp).ffn_layer_scale.gamma")

            layers.append(FishS1DACTransformerBlock(
                attnNorm: attnNorm,
                ffnNorm: ffnNorm,
                wqkv: wqkv,
                wo: wo,
                mlp: mlp,
                attnLayerScale: attnScale,
                ffnLayerScale: ffnScale,
                nHead: nHead,
                nKVHead: nKVHead,
                headDim: headDim,
                dim: dim,
                ropeBase: ropeBase))
        }

        let normW = try w.floats("\(prefix).norm.weight")
        let normFinal = FishS1DACRMSNorm(weight: normW, eps: 1e-5)

        // Optional input/output projections (when inputDim != dim).
        let inputProjW: [Float]?
        let outputProjW: [Float]?
        if inputDim != dim {
            inputProjW  = w.has("\(prefix).input_proj.weight")
                ? try w.floats("\(prefix).input_proj.weight") : nil
            outputProjW = w.has("\(prefix).output_proj.weight")
                ? try w.floats("\(prefix).output_proj.weight") : nil
        } else {
            inputProjW  = nil
            outputProjW = nil
        }

        return FishS1DACWindowLimitedTransformer(
            layers: layers,
            norm: normFinal,
            inputProjW: inputProjW,
            outputProjW: outputProjW,
            inputDim: inputDim,
            dim: dim,
            windowSize: config.quantizerWindowSize)
    }

    /// Decode codes `[[Int32]]` (shape `[numCodebooks, T]`) into a
    /// continuous latent NCL `[1, latentDim, T*upsampleFactor]`.
    ///
    /// The code layout matches what `FishSpeech.generateCodes` produces:
    ///   - codes[0]: semantic tokens (codebook 0, size `semanticCodebookSize`)
    ///   - codes[1…]: residual tokens (size `residualCodebookSize`)
    func decode(codes: [[Int32]]) throws -> (data: [Float], shape: [Int]) {
        guard !codes.isEmpty, let firstStream = codes.first else {
            throw FishS1DACError.shapeMismatch("empty code list")
        }
        let t = firstStream.count

        // Clamp semantic codes to [0, semanticCodebookSize).
        let semanticCodes = codes[0].map { code -> Int32 in
            max(0, min(code, Int32(semanticCodebookSize - 1)))
        }
        let (zQSemantic, semShape) = semanticQuantizer.fromCodes(codes: [semanticCodes])
        guard !zQSemantic.isEmpty else {
            throw FishS1DACError.shapeMismatch("semantic decode returned empty tensor")
        }
        _ = t

        // Residual codes: indices 1…
        var zQ = zQSemantic
        let zShape = semShape
        if codes.count > 1 {
            let residualCodes = Array(codes.dropFirst()).enumerated().map { (i, stream) in
                stream.map { code -> Int32 in
                    max(0, min(code, Int32(residualCodebookSize - 1)))
                }
            }
            let (zQResidual, _) = quantizer.fromCodes(codes: residualCodes)
            if !zQResidual.isEmpty {
                precondition(zQResidual.count == zQ.count,
                             "FishS1DACDownsampleRVQ.decode: latent size mismatch")
                for i in 0..<zQ.count { zQ[i] += zQResidual[i] }
            }
        }

        // Post-module (transformer).
        var (hidden, hShape) = (zQ, zShape)
        if let tf = postModule {
            (hidden, hShape) = tf.forward(hidden, shape: hShape)
        }

        // Upsample stages.
        for stage in upsampleStages {
            (hidden, hShape) = stage(hidden, shape: hShape)
        }
        return (hidden, hShape)
    }
}
