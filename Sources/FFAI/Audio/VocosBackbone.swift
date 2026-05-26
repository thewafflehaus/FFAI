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
// VocosBackbone — the ConvNeXt backbone and ISTFT head for Vocos.
//
// Companion to Vocos.swift. Mirrors the weight layout of the reference
// MLX Vocos so checkpoint keys line up exactly:
//
//   backbone.embed.{weight,bias}              : input conv
//   backbone.norm.{weight,bias}               : initial LayerNorm
//   backbone.convnext.{i}.dwconv.{weight,bias}: depthwise conv
//   backbone.convnext.{i}.norm.{weight,bias}  : per-block LayerNorm
//   backbone.convnext.{i}.pwconv{1,2}.{w,b}   : pointwise linears
//   backbone.convnext.{i}.gamma               : layer-scale
//   backbone.final_layer_norm.{weight,bias}
//   head.out.{weight,bias}                    : STFT projection
//
// The backbone math is CPU-native (see AudioPrimitives.swift); the ISTFT
// head reuses the fused GPU `Ops.vocoderISTFT` kernel.

import Foundation

// MARK: - Layout helpers

/// Transpose an NCL tensor `[1, C, T]` to a `[T, C]` row-major matrix.
private func nclToRows(_ x: [Float], c: Int, t: Int) -> [Float] {
    var out = [Float](repeating: 0, count: t * c)
    for ch in 0..<c {
        for pos in 0..<t { out[pos * c + ch] = x[ch * t + pos] }
    }
    return out
}

/// Transpose a `[T, C]` row-major matrix back to NCL `[1, C, T]`.
private func rowsToNcl(_ x: [Float], c: Int, t: Int) -> [Float] {
    var out = [Float](repeating: 0, count: c * t)
    for pos in 0..<t {
        for ch in 0..<c { out[ch * t + pos] = x[pos * c + ch] }
    }
    return out
}

// MARK: - ConvNeXt block

/// A Vocos ConvNeXt block: depthwise conv → LayerNorm → pointwise linear
/// → GELU → pointwise linear → layer-scale, with a residual connection.
struct VocosConvNeXtBlock {
    let dwWeight: [Float]      // [dim, 1, K]  (depthwise)
    let dwShape: [Int]
    let dwBias: [Float]?
    let normW: [Float], normB: [Float]?
    let pw1W: [Float], pw1B: [Float]?
    let pw2W: [Float], pw2B: [Float]?
    let gamma: [Float]?
    let dim: Int
    let interDim: Int

    init(weights w: VocosWeights, prefix: String, config: VocosConfig) throws {
        let (dw, ds) = try w.convWeight("\(prefix).dwconv.weight")
        self.dwWeight = dw
        self.dwShape = ds
        self.dwBias = w.has("\(prefix).dwconv.bias")
            ? try w.floats("\(prefix).dwconv.bias") : nil
        self.normW = try w.floats("\(prefix).norm.weight")
        self.normB = w.has("\(prefix).norm.bias")
            ? try w.floats("\(prefix).norm.bias") : nil
        self.pw1W = try w.floats("\(prefix).pwconv1.weight")
        self.pw1B = w.has("\(prefix).pwconv1.bias")
            ? try w.floats("\(prefix).pwconv1.bias") : nil
        self.pw2W = try w.floats("\(prefix).pwconv2.weight")
        self.pw2B = w.has("\(prefix).pwconv2.bias")
            ? try w.floats("\(prefix).pwconv2.bias") : nil
        self.gamma = w.has("\(prefix).gamma")
            ? try w.floats("\(prefix).gamma") : nil
        self.dim = config.dim
        self.interDim = config.intermediateDim
    }

    /// Apply the block to an NCL feature map `[1, dim, T]`.
    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        let t = shape[2]
        // Depthwise conv — groups == dim keeps channels independent.
        // Kernel size K, "same" padding K/2.
        let k = dwShape[2]
        let (h, hs) = AudioMath.conv1d(
            x: x, xShape: shape, weight: dwWeight, wShape: dwShape,
            bias: dwBias, stride: 1, padding: k / 2, dilation: 1,
            groups: dim)
        // LayerNorm over channels — done in [T, dim] row layout.
        var rows = nclToRows(h, c: dim, t: hs[2])
        rows = AudioMath.layerNorm(rows, rows: hs[2], dim: dim,
                                   weight: normW, bias: normB)
        // Pointwise convs (1×1) implemented as linears over [T, dim].
        var ff = AudioMath.linear(rows, rows: hs[2], inDim: dim,
                                  weight: pw1W, outDim: interDim, bias: pw1B)
        ff = AudioMath.gelu(ff)
        var out = AudioMath.linear(ff, rows: hs[2], inDim: interDim,
                                   weight: pw2W, outDim: dim, bias: pw2B)
        // Layer scale (per-channel gamma).
        if let g = gamma {
            for pos in 0..<hs[2] {
                for ch in 0..<dim { out[pos * dim + ch] *= g[ch] }
            }
        }
        // Residual add (x is NCL; out is [T, dim]).
        let outNcl = rowsToNcl(out, c: dim, t: hs[2])
        precondition(outNcl.count == x.count,
                     "VocosConvNeXtBlock: residual length mismatch")
        var sum = x
        for i in 0..<sum.count { sum[i] += outNcl[i] }
        _ = t
        return (sum, shape)
    }
}

// MARK: - Backbone

/// The Vocos ConvNeXt backbone — an input conv, an initial LayerNorm, a
/// stack of ConvNeXt blocks, and a final LayerNorm.
struct VocosBackbone {
    let embedWeight: [Float]   // [dim, inputChannels, K]
    let embedShape: [Int]
    let embedBias: [Float]?
    let normW: [Float], normB: [Float]?
    let blocks: [VocosConvNeXtBlock]
    let finalNormW: [Float], finalNormB: [Float]?
    let dim: Int

    init(weights w: VocosWeights, config c: VocosConfig) throws {
        let (ew, es) = try w.convWeight("backbone.embed.weight")
        self.embedWeight = ew
        self.embedShape = es
        self.embedBias = w.has("backbone.embed.bias")
            ? try w.floats("backbone.embed.bias") : nil
        self.normW = try w.floats("backbone.norm.weight")
        self.normB = w.has("backbone.norm.bias")
            ? try w.floats("backbone.norm.bias") : nil
        var bs: [VocosConvNeXtBlock] = []
        for i in 0..<c.numLayers {
            bs.append(try VocosConvNeXtBlock(
                weights: w, prefix: "backbone.convnext.\(i)", config: c))
        }
        self.blocks = bs
        self.finalNormW = try w.floats("backbone.final_layer_norm.weight")
        self.finalNormB = w.has("backbone.final_layer_norm.bias")
            ? try w.floats("backbone.final_layer_norm.bias") : nil
        self.dim = c.dim
    }

    /// Run the backbone over an NCL feature map `[1, inputChannels, T]`.
    /// Returns the refined `[1, dim, T]` feature map.
    func forward(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        // Input conv — "same" padding K/2.
        let k = embedShape[2]
        var (h, hs) = AudioMath.conv1d(
            x: x, xShape: shape, weight: embedWeight, wShape: embedShape,
            bias: embedBias, stride: 1, padding: k / 2, dilation: 1,
            groups: 1)
        // Initial LayerNorm over channels.
        var rows = nclToRows(h, c: dim, t: hs[2])
        rows = AudioMath.layerNorm(rows, rows: hs[2], dim: dim,
                                   weight: normW, bias: normB)
        h = rowsToNcl(rows, c: dim, t: hs[2])
        // ConvNeXt blocks.
        for block in blocks { (h, hs) = block(h, shape: hs) }
        // Final LayerNorm.
        rows = nclToRows(h, c: dim, t: hs[2])
        rows = AudioMath.layerNorm(rows, rows: hs[2], dim: dim,
                                   weight: finalNormW, bias: finalNormB)
        h = rowsToNcl(rows, c: dim, t: hs[2])
        return (h, hs)
    }
}

// MARK: - ISTFT head

/// The Vocos ISTFT head — projects backbone features to a complex STFT
/// (magnitude + phase) and reconstructs a waveform via inverse-STFT
/// overlap-add. The overlap-add reuses the fused GPU `Ops.vocoderISTFT`
/// kernel; the post-ISTFT centre-trim is done on the host.
struct VocosISTFTHead {
    let outWeight: [Float]    // [nFFT+2, dim]
    let outBias: [Float]?
    let nFFT: Int
    let hopLength: Int
    let dim: Int
    /// Magnitude is clipped to keep the iSTFT numerically stable.
    private static let magClip: Float = 1e2

    init(weights w: VocosWeights, config c: VocosConfig) throws {
        self.outWeight = try w.floats("head.out.weight")
        self.outBias = w.has("head.out.bias")
            ? try w.floats("head.out.bias") : nil
        self.nFFT = c.nFFT
        self.hopLength = c.hopLength
        self.dim = c.dim
    }

    /// Reconstruct a `[L]` waveform from an NCL backbone feature map
    /// `[1, dim, T]`.
    func synthesize(_ x: [Float], shape: [Int],
                    device: Device = .shared) -> Tensor {
        let t = shape[2]
        // Project each frame to STFT coefficients: [T, nFFT+2].
        let rows = nclToRows(x, c: dim, t: t)
        let coeffs = AudioMath.linear(rows, rows: t, inDim: dim,
                                      weight: outWeight, outDim: nFFT + 2,
                                      bias: outBias)
        // Split into magnitude (exp, clipped) and phase; build the
        // complex STFT plane `[T, nFreq]` with nFreq = nFFT/2 + 1.
        let nFreq = nFFT / 2 + 1
        var specRe = [Float](repeating: 0, count: t * nFreq)
        var specIm = [Float](repeating: 0, count: t * nFreq)
        for frame in 0..<t {
            let base = frame * (nFFT + 2)
            for f in 0..<nFreq {
                let mag = min(expf(coeffs[base + f]), Self.magClip)
                let phase = coeffs[base + nFreq + f]
                specRe[frame * nFreq + f] = mag * cosf(phase)
                specIm[frame * nFreq + f] = mag * sinf(phase)
            }
        }
        // Inverse-STFT overlap-add on the GPU.
        let reT = Tensor.empty(shape: [t, nFreq], dtype: .f32, device: device)
        reT.copyIn(from: specRe)
        let imT = Tensor.empty(shape: [t, nFreq], dtype: .f32, device: device)
        imT.copyIn(from: specIm)
        let win = AudioPreprocessing.hannWindow(nFFT)
        let winT = Tensor.empty(shape: [nFFT], dtype: .f32, device: device)
        winT.copyIn(from: win)

        let cmd = device.makeCommandBuffer()
        let waveform = Ops.vocoderISTFT(
            specRe: reT, specIm: imT, window: winT,
            nFrames: t, nFFT: nFFT, hopLength: hopLength, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // Vocos centre-pads STFT framing, so trim nFFT/2 from each end.
        let full = AudioMath.floats(waveform)
        let trim = nFFT / 2
        let lo = min(trim, full.count)
        let hi = max(full.count - trim, lo)
        let trimmed = Array(full[lo..<hi])
        let out = Tensor.empty(shape: [trimmed.count], dtype: .f32,
                               device: device)
        out.copyIn(from: trimmed)
        return out
    }
}
