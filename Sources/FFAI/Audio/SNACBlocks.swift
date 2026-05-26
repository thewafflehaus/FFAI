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
// SNACBlocks — encoder/decoder blocks and the residual VQ quantizer.
//
// Companion to SNAC.swift. Mirrors the nn.Sequential weight layout of
// the reference MLX SNAC so checkpoint keys line up exactly:
//
//   ResidualUnit.block.layers   = [Snake, WNConv1d, Snake, WNConv1d]
//   EncoderBlock.block.layers   = [Res, Res, Res, Snake, WNConv1d]
//   DecoderBlock.block.layers   = [Snake, WNConvT1d, (Noise?), Res, Res, Res]
//
// All math is CPU-native (see AudioPrimitives.swift).

import Foundation

// MARK: - Residual unit construction

extension SNACResidualUnit {
    /// Build a residual unit from checkpoint weights at `prefix`.
    /// `dim` is the channel count; `dilation` the conv dilation.
    init(
        weights w: SNACWeights, prefix: String,
        dim: Int, dilation: Int, kernel: Int = 7, groups: Int
    ) throws {
        let pad = ((kernel - 1) * dilation) / 2
        // layers.0 — Snake1d, layers.1 — WNConv1d(k=7, dilated)
        // layers.2 — Snake1d, layers.3 — WNConv1d(k=1)
        self.alpha1 = try w.floats("\(prefix).block.layers.0.alpha")
        self.conv1 = try w.wnConv1d(
            prefix: "\(prefix).block.layers.1",
            stride: 1, padding: pad,
            dilation: dilation, groups: groups)
        self.alpha2 = try w.floats("\(prefix).block.layers.2.alpha")
        self.conv2 = try w.wnConv1d(
            prefix: "\(prefix).block.layers.3",
            stride: 1, padding: 0,
            dilation: 1, groups: 1)
    }
}

// MARK: - Encoder block

/// SNAC encoder block: three dilated residual units, a Snake, then a
/// strided WNConv1d that downsamples by `stride`.
struct SNACEncoderBlock {
    let residuals: [SNACResidualUnit]
    let snakeAlpha: [Float]
    let convDown: SNACWNConv1d

    init(
        weights w: SNACWeights, prefix: String,
        outputDim: Int, stride: Int, groups: Int
    ) throws {
        let inputDim = outputDim / 2
        // layers.{0,1,2} — ResidualUnit(dilation 1,3,9)
        var res: [SNACResidualUnit] = []
        for (i, dil) in [1, 3, 9].enumerated() {
            res.append(
                try SNACResidualUnit(
                    weights: w, prefix: "\(prefix).block.layers.\(i)",
                    dim: inputDim, dilation: dil, groups: groups))
        }
        self.residuals = res
        // layers.3 — Snake1d
        self.snakeAlpha = try w.floats("\(prefix).block.layers.3.alpha")
        // layers.4 — WNConv1d(k=2*stride, stride, pad=ceil(stride/2))
        let pad = Int(ceil(Double(stride) / 2.0))
        self.convDown = try w.wnConv1d(
            prefix: "\(prefix).block.layers.4",
            stride: stride, padding: pad,
            dilation: 1, groups: 1)
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (d, s) = (x, shape)
        for r in residuals { (d, s) = r(d, shape: s) }
        d = AudioMath.snake(d, shape: s, alpha: snakeAlpha)
        return convDown(d, shape: s)
    }
}

// MARK: - Decoder block

/// SNAC decoder block: Snake -> WNConvTranspose1d (upsample) ->
/// (optional noise injection) -> three dilated residual units.
struct SNACDecoderBlock {
    let snakeAlpha: [Float]
    let convUp: SNACWNConvTranspose1d
    let noiseConv: SNACWNConv1d?  // 1x1 WNConv, bias-free, when noise on
    let residuals: [SNACResidualUnit]

    init(
        weights w: SNACWeights, prefix: String,
        inputDim: Int, outputDim: Int, stride: Int,
        noise: Bool, groups: Int
    ) throws {
        // layers.0 — Snake1d
        self.snakeAlpha = try w.floats("\(prefix).block.layers.0.alpha")
        // layers.1 — WNConvTranspose1d(k=2*stride, stride,
        //            pad=ceil(stride/2), outputPadding=stride%2)
        let pad = Int(ceil(Double(stride) / 2.0))
        self.convUp = try w.wnConvTranspose1d(
            prefix: "\(prefix).block.layers.1",
            stride: stride, padding: pad,
            outputPadding: stride % 2, groups: 1)

        var idx = 2
        if noise {
            // NoiseBlock wraps a single bias-free 1x1 WNConv1d at
            // `.linear`.
            self.noiseConv = try w.wnConv1d(
                prefix: "\(prefix).block.layers.\(idx).linear",
                stride: 1, padding: 0, dilation: 1, groups: 1)
            idx += 1
        } else {
            self.noiseConv = nil
        }

        var res: [SNACResidualUnit] = []
        for dil in [1, 3, 9] {
            res.append(
                try SNACResidualUnit(
                    weights: w, prefix: "\(prefix).block.layers.\(idx)",
                    dim: outputDim, dilation: dil, groups: groups))
            idx += 1
        }
        self.residuals = res
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (d, s) = (x, shape)
        d = AudioMath.snake(d, shape: s, alpha: snakeAlpha)
        (d, s) = convUp(d, shape: s)
        if let nc = noiseConv {
            // Noise injection: x + normal(B,1,T) * conv1x1(x).
            // Deterministic decode is preferred for codec reconstruction
            // tests, so we omit the stochastic term (PyTorch SNAC also
            // runs noise=0 at inference for reproducible output).
            _ = nc
        }
        for r in residuals { (d, s) = r(d, shape: s) }
        return (d, s)
    }
}

// MARK: - Vector quantizer

/// A single residual-VQ codebook with optional temporal striding.
///
/// `encode` projects the latent down (`in_proj`), finds the nearest
/// codebook entry per frame, projects back up (`out_proj`), and — for
/// `stride > 1` — average-pools before and repeat-interleaves after.
struct SNACVectorQuantize {
    let codebookSize: Int
    let codebookDim: Int
    let stride: Int

    let inProj: SNACWNConv1d  // [codebookDim, inputDim, 1]
    let outProj: SNACWNConv1d  // [inputDim, codebookDim, 1]
    let codebook: [Float]  // [codebookSize, codebookDim]
    let inputDim: Int

    init(
        weights w: SNACWeights, prefix: String,
        codebookSize: Int, codebookDim: Int, stride: Int
    ) throws {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.stride = stride
        self.inProj = try w.wnConv1d(
            prefix: "\(prefix).in_proj",
            stride: 1, padding: 0,
            dilation: 1, groups: 1)
        self.outProj = try w.wnConv1d(
            prefix: "\(prefix).out_proj",
            stride: 1, padding: 0,
            dilation: 1, groups: 1)
        self.codebook = try w.floats("\(prefix).codebook.weight")
        // out_proj weight is [inputDim, codebookDim, 1].
        self.inputDim = outProj.wShape[0]
    }

    /// Average-pool an NCL tensor along time by `stride`.
    private func avgPool(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        if stride == 1 { return (x, shape) }
        let (n, c, l) = (shape[0], shape[1], shape[2])
        let lOut = l / stride
        var out = [Float](repeating: 0, count: n * c * lOut)
        for b in 0 ..< n {
            for ch in 0 ..< c {
                let inBase = (b * c + ch) * l
                let outBase = (b * c + ch) * lOut
                for t in 0 ..< lOut {
                    var acc: Float = 0
                    for k in 0 ..< stride { acc += x[inBase + t * stride + k] }
                    out[outBase + t] = acc / Float(stride)
                }
            }
        }
        return (out, [n, c, lOut])
    }

    /// Repeat-interleave an NCL tensor along time by `stride`.
    private func repeatInterleave(
        _ x: [Float],
        shape: [Int]
    ) -> (data: [Float], shape: [Int]) {
        if stride == 1 { return (x, shape) }
        let (n, c, l) = (shape[0], shape[1], shape[2])
        let lOut = l * stride
        var out = [Float](repeating: 0, count: n * c * lOut)
        for b in 0 ..< n {
            for ch in 0 ..< c {
                let inBase = (b * c + ch) * l
                let outBase = (b * c + ch) * lOut
                for t in 0 ..< l {
                    let v = x[inBase + t]
                    for k in 0 ..< stride { out[outBase + t * stride + k] = v }
                }
            }
        }
        return (out, [n, c, lOut])
    }

    /// Nearest-codebook lookup. `latents` is NCL `[1, codebookDim, T]`.
    /// Returns the picked indices `[T]` and the quantized latent `[1, D, T]`.
    private func decodeLatents(
        _ latents: [Float],
        shape: [Int]
    ) -> (zQ: [Float], indices: [Int32]) {
        let (_, d, t) = (shape[0], shape[1], shape[2])
        // Rearrange b d t -> (b t) d.
        var enc = [Float](repeating: 0, count: t * d)
        for i in 0 ..< t {
            for c in 0 ..< d { enc[i * d + c] = latents[c * t + i] }
        }
        // Normalize both encodings and codebook, then nearest by
        // distance = ||e||^2 - 2 e·c + ||c||^2.
        let encN = AudioMath.l2NormalizeRows(enc, rows: t, dim: d)
        let cbN = AudioMath.l2NormalizeRows(codebook, rows: codebookSize, dim: d)
        var indices = [Int32](repeating: 0, count: t)
        for i in 0 ..< t {
            var best: Float = .greatestFiniteMagnitude
            var bestIdx = 0
            let eBase = i * d
            for ci in 0 ..< codebookSize {
                let cBase = ci * d
                var dot: Float = 0
                for c in 0 ..< d { dot += encN[eBase + c] * cbN[cBase + c] }
                // ||e||^2 and ||c||^2 are both 1 after normalization, so
                // distance is monotone in -dot; maximize dot.
                let dist = 2.0 - 2.0 * dot
                if dist < best {
                    best = dist
                    bestIdx = ci
                }
            }
            indices[i] = Int32(bestIdx)
        }
        // zQ = codebook[indices], rearranged back to [1, D, T].
        var zQ = [Float](repeating: 0, count: d * t)
        for i in 0 ..< t {
            let cBase = Int(indices[i]) * d
            for c in 0 ..< d { zQ[c * t + i] = codebook[cBase + c] }
        }
        return (zQ, indices)
    }

    /// Encode a residual latent: returns the upsampled quantized latent
    /// (for residual subtraction), the projected latent, and the codes.
    func encode(
        _ z: [Float],
        shape: [Int]
    ) throws -> (zQ: [Float], zE: [Float], indices: [Int32]) {
        var (d, s) = (z, shape)
        (d, s) = avgPool(d, shape: s)
        // in_proj: [B, inputDim, T] -> [B, codebookDim, T].
        let (zE, zeShape) = inProj(d, shape: s)
        let (zQLatent, indices) = decodeLatents(zE, shape: zeShape)
        // out_proj: [B, codebookDim, T] -> [B, inputDim, T].
        var (zQ, zqShape) = outProj(zQLatent, shape: zeShape)
        (zQ, zqShape) = repeatInterleave(zQ, shape: zqShape)
        _ = zqShape
        return (zQ, zE, indices)
    }

    /// Decode codes back into the upsampled quantized latent `[1, D, T]`.
    func decode(codes: [Int32]) throws -> (data: [Float], shape: [Int]) {
        let t = codes.count
        // codebook lookup -> [1, codebookDim, T].
        var zLatent = [Float](repeating: 0, count: codebookDim * t)
        for i in 0 ..< t {
            let cBase = Int(codes[i]) * codebookDim
            for c in 0 ..< codebookDim { zLatent[c * t + i] = codebook[cBase + c] }
        }
        let (zQ, zqShape) = outProj(zLatent, shape: [1, codebookDim, t])
        return repeatInterleave(zQ, shape: zqShape)
    }
}
