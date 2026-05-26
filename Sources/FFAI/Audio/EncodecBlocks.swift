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
// EncodecBlocks — SEANet encoder/decoder, LSTM bottleneck, and the
// residual-VQ quantizer for the EnCodec codec.
//
// Companion to Encodec.swift. Mirrors the `nn.Sequential` weight layout
// of the reference MLX EnCodec so checkpoint keys line up exactly:
//
//   encoder.layers.{i}.conv.{weight,bias}        : EncodecConv1d
//   encoder.layers.{i}.lstm.{l}.{Wx,Wh,bias}     : EncodecLSTM
//   encoder.layers.{i}.block.{j}.conv.{...}      : resnet inner conv
//   encoder.layers.{i}.shortcut.conv.{...}       : resnet shortcut conv
//   quantizer.layers.{i}.codebook.embed          : VQ codebook
//
// EnCodec checkpoints store conv weights in MLX's NLC layout
// `[Cout, K, Cin]`; `AudioMath.conv1d` wants PyTorch `[Cout, Cin, K]`,
// so weights are transposed once at load time.
//
// All math is CPU-native (see AudioPrimitives.swift).

import Foundation

// MARK: - Conv weight layout

/// Transpose an MLX NLC conv weight `[Cout, K, Cin]` to the PyTorch
/// `[Cout, Cin, K]` layout `AudioMath.conv1d` expects.
private func nlcToNcw(_ w: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
    let (cOut, k, cIn) = (shape[0], shape[1], shape[2])
    var out = [Float](repeating: 0, count: w.count)
    for o in 0..<cOut {
        for kk in 0..<k {
            for ic in 0..<cIn {
                // src: [o, kk, ic]  dst: [o, ic, kk]
                out[(o * cIn + ic) * k + kk] = w[(o * k + kk) * cIn + ic]
            }
        }
    }
    return (out, [cOut, cIn, k])
}

// MARK: - Causal / asymmetric padded conv

/// A 1-D conv with EnCodec's causal-or-asymmetric padding scheme. Pads
/// the time axis (reflect or zero) so the strided conv lands on integer
/// frame counts, then convolves.
struct EncodecConv1d {
    let weight: [Float]      // [Cout, Cin, K]
    let wShape: [Int]
    let bias: [Float]?
    let stride: Int
    let dilation: Int
    let causal: Bool
    let reflect: Bool

    /// `kernelSizeEffective` and `paddingTotal` as in the reference.
    var kernelSizeEffective: Int { (wShape[2] - 1) * dilation + 1 }
    var paddingTotal: Int { kernelSizeEffective - stride }

    init(weights w: EncodecWeights, prefix: String, stride: Int,
         dilation: Int, config: EncodecConfig) throws {
        let wKey = "\(prefix).weight"
        guard w.has(wKey) else { throw EncodecError.missingWeights(prefix) }
        let raw = try w.floats(wKey)
        let rawShape = try w.shape(wKey)
        // EnCodec stores conv weight as [Cout, K, Cin] (MLX NLC).
        let (tw, ts) = nlcToNcw(raw, shape: rawShape)
        self.weight = tw
        self.wShape = ts
        self.bias = w.has("\(prefix).bias") ? try w.floats("\(prefix).bias") : nil
        self.stride = stride
        self.dilation = dilation
        self.causal = config.useCausalConv
        self.reflect = config.padMode == "reflect"
    }

    /// Extra right-padding so the strided conv lands on whole frames.
    private func extraPadding(length: Int) -> Int {
        let nFrames = Float(length - kernelSizeEffective + paddingTotal)
            / Float(stride) + 1
        let nFramesInt = Int(ceil(Double(nFrames))) - 1
        let idealLength = nFramesInt * stride + kernelSizeEffective - paddingTotal
        return max(0, idealLength - length)
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        let length = shape[2]
        let extra = extraPadding(length: length)
        let (left, right): (Int, Int)
        if causal {
            (left, right) = (paddingTotal, extra)
        } else {
            let r = paddingTotal / 2
            (left, right) = (paddingTotal - r, r + extra)
        }
        var (padded, ps) = (x, shape)
        if left > 0 || right > 0 {
            if reflect && length > 1 {
                (padded, ps) = AudioMath.reflectionPad1d(
                    x, shape: shape, left: min(left, length - 1),
                    right: min(right, length - 1))
                // If the requested pad exceeds length-1, top up with zeros.
                let extraL = left - min(left, length - 1)
                let extraR = right - min(right, length - 1)
                if extraL > 0 || extraR > 0 {
                    (padded, ps) = AudioMath.zeroPad1d(
                        padded, shape: ps, left: extraL, right: extraR)
                }
            } else {
                (padded, ps) = AudioMath.zeroPad1d(
                    x, shape: shape, left: left, right: right)
            }
        }
        return AudioMath.conv1d(
            x: padded, xShape: ps, weight: weight, wShape: wShape,
            bias: bias, stride: stride, padding: 0,
            dilation: dilation, groups: 1)
    }
}

// MARK: - Causal / asymmetric padded transposed conv

/// A transposed 1-D conv with EnCodec's output-trimming scheme — the
/// decoder's upsampler.
struct EncodecConvTranspose1d {
    let weight: [Float]      // [Cin, Cout, K]
    let wShape: [Int]
    let bias: [Float]?
    let stride: Int
    let causal: Bool
    let trimRightRatio: Float

    var paddingTotal: Int { wShape[2] - stride }

    init(weights w: EncodecWeights, prefix: String, stride: Int,
         config: EncodecConfig) throws {
        let wKey = "\(prefix).weight"
        guard w.has(wKey) else { throw EncodecError.missingWeights(prefix) }
        let raw = try w.floats(wKey)
        let rawShape = try w.shape(wKey)
        // EnCodec transposed conv weight ships as [Cout, K, Cin] (MLX
        // NLC). convTransposed1d wants [Cin, Cout, K] — permute.
        let (cOut, k, cIn) = (rawShape[0], rawShape[1], rawShape[2])
        var permuted = [Float](repeating: 0, count: raw.count)
        for o in 0..<cOut {
            for kk in 0..<k {
                for ic in 0..<cIn {
                    permuted[(ic * cOut + o) * k + kk] = raw[(o * k + kk) * cIn + ic]
                }
            }
        }
        self.weight = permuted
        self.wShape = [cIn, cOut, k]
        self.bias = w.has("\(prefix).bias") ? try w.floats("\(prefix).bias") : nil
        self.stride = stride
        self.causal = config.useCausalConv
        self.trimRightRatio = config.trimRightRatio
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (out, s) = AudioMath.convTransposed1d(
            x: x, xShape: shape, weight: weight, wShape: wShape,
            bias: bias, stride: stride, padding: 0, dilation: 1,
            outputPadding: 0, groups: 1)
        // Trim the padding back off the time axis.
        let right: Int
        if causal {
            right = Int(ceil(Double(Float(paddingTotal) * trimRightRatio)))
        } else {
            right = paddingTotal / 2
        }
        let left = paddingTotal - right
        let end = s[2] - right
        if end > left {
            (out, s) = sliceTime(out, shape: s, start: left, end: end)
        }
        return (out, s)
    }

    /// Crop an NCL tensor to `[start, end)` on the time axis.
    private func sliceTime(_ x: [Float], shape: [Int],
                           start: Int, end: Int) -> (data: [Float], shape: [Int]) {
        let (n, c, l) = (shape[0], shape[1], shape[2])
        let lOut = end - start
        var out = [Float](repeating: 0, count: n * c * lOut)
        for b in 0..<n {
            for ch in 0..<c {
                let inBase = (b * c + ch) * l
                let outBase = (b * c + ch) * lOut
                for t in 0..<lOut { out[outBase + t] = x[inBase + start + t] }
            }
        }
        return (out, [n, c, lOut])
    }
}

// MARK: - LSTM bottleneck

/// A single LSTM layer. EnCodec runs the LSTM on the latent sequence;
/// the codec is not autoregressive so a plain CPU loop over time is fine.
struct EncodecLSTM {
    let hiddenSize: Int
    let inputSize: Int
    let wx: [Float]          // [4H, inputSize]
    let wh: [Float]          // [4H, H]
    let bias: [Float]?       // [4H]

    init(weights w: EncodecWeights, prefix: String,
         inputSize: Int, hiddenSize: Int) throws {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        // Weight keys follow the reference MLX EncodecLSTM module.
        self.wx = try w.floats("\(prefix).Wx")
        self.wh = try w.floats("\(prefix).Wh")
        self.bias = w.has("\(prefix).bias") ? try w.floats("\(prefix).bias") : nil
    }

    /// Run the LSTM over an `[T, inputSize]` sequence (batch 1). Returns
    /// the hidden-state sequence `[T, H]`.
    func callAsFunction(_ x: [Float], steps t: Int) -> [Float] {
        let h4 = 4 * hiddenSize
        // xProj = x · Wxᵀ + bias  ->  [T, 4H]
        let xProj = AudioMath.linear(x, rows: t, inDim: inputSize,
                                     weight: wx, outDim: h4, bias: bias)
        var hidden = [Float](repeating: 0, count: hiddenSize)
        var cell = [Float](repeating: 0, count: hiddenSize)
        var hasHidden = false
        var allHidden = [Float](repeating: 0, count: t * hiddenSize)
        for step in 0..<t {
            // hProj = hidden · Whᵀ  (zeros on the first step).
            let hProj: [Float]
            if hasHidden {
                hProj = AudioMath.linear(hidden, rows: 1, inDim: hiddenSize,
                                         weight: wh, outDim: h4, bias: nil)
            } else {
                hProj = [Float](repeating: 0, count: h4)
            }
            let gBase = step * h4
            for j in 0..<hiddenSize {
                let i = sigmoidf(xProj[gBase + j] + hProj[j])
                let f = sigmoidf(xProj[gBase + hiddenSize + j] + hProj[hiddenSize + j])
                let g = tanhf(xProj[gBase + 2 * hiddenSize + j] + hProj[2 * hiddenSize + j])
                let o = sigmoidf(xProj[gBase + 3 * hiddenSize + j] + hProj[3 * hiddenSize + j])
                cell[j] = f * cell[j] + i * g
                hidden[j] = o * tanhf(cell[j])
            }
            hasHidden = true
            for j in 0..<hiddenSize { allHidden[step * hiddenSize + j] = hidden[j] }
        }
        return allHidden
    }
}

private func sigmoidf(_ x: Float) -> Float { 1.0 / (1.0 + expf(-x)) }

/// The LSTM bottleneck block: `numLstmLayers` stacked LSTMs with a
/// residual connection over the whole stack.
struct EncodecLSTMBlock {
    let dimension: Int
    let layers: [EncodecLSTM]

    init(weights w: EncodecWeights, prefix: String,
         dimension: Int, numLayers: Int) throws {
        self.dimension = dimension
        var ls: [EncodecLSTM] = []
        for i in 0..<numLayers {
            ls.append(try EncodecLSTM(
                weights: w, prefix: "\(prefix).lstm.\(i)",
                inputSize: dimension, hiddenSize: dimension))
        }
        self.layers = ls
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        // Reference runs the LSTM in NLC; we hold NCL so transpose into
        // a [T, C] sequence, run, then transpose the result back.
        let (n, c, l) = (shape[0], shape[1], shape[2])
        precondition(n == 1, "EncodecLSTMBlock: batch must be 1")
        var seq = [Float](repeating: 0, count: l * c)
        for ch in 0..<c {
            for t in 0..<l { seq[t * c + ch] = x[ch * l + t] }
        }
        var h = seq
        for layer in layers { h = layer(h, steps: l) }
        // Residual add in [T, C] then transpose back to NCL.
        var out = [Float](repeating: 0, count: x.count)
        for ch in 0..<c {
            for t in 0..<l {
                out[ch * l + t] = h[t * c + ch] + seq[t * c + ch]
            }
        }
        return (out, shape)
    }
}

// MARK: - Resnet block

/// SEANet residual block: `[ELU, Conv(k=residualKernelSize, dilated),
/// ELU, Conv(k=1)]` plus an optional 1×1 conv shortcut.
struct EncodecResnetBlock {
    let conv1: EncodecConv1d
    let conv2: EncodecConv1d
    let shortcut: EncodecConv1d?

    init(weights w: EncodecWeights, prefix: String, dim: Int,
         dilation: Int, config: EncodecConfig) throws {
        // block.0 — ELU, block.1 — Conv(residualKernelSize, dilation)
        // block.2 — ELU, block.3 — Conv(k=1)
        self.conv1 = try EncodecConv1d(
            weights: w, prefix: "\(prefix).block.1",
            stride: 1, dilation: dilation, config: config)
        self.conv2 = try EncodecConv1d(
            weights: w, prefix: "\(prefix).block.3",
            stride: 1, dilation: 1, config: config)
        if config.useConvShortcut && w.has("\(prefix).shortcut.conv.weight") {
            self.shortcut = try EncodecConv1d(
                weights: w, prefix: "\(prefix).shortcut.conv",
                stride: 1, dilation: 1, config: config)
        } else {
            self.shortcut = nil
        }
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (h, s) = (AudioMath.elu(x), shape)
        (h, s) = conv1(h, shape: s)
        h = AudioMath.elu(h)
        (h, s) = conv2(h, shape: s)
        // Residual: shortcut conv (1×1) keeps length unchanged.
        let res: [Float]
        if let sc = shortcut {
            res = sc(x, shape: shape).data
        } else {
            res = x
        }
        precondition(res.count == h.count,
                     "EncodecResnetBlock: residual length mismatch")
        var out = h
        for i in 0..<out.count { out[i] += res[i] }
        return (out, s)
    }
}

// MARK: - SEANet encoder / decoder

/// A SEANet stack — used for both the EnCodec encoder and decoder. The
/// layout is built procedurally from the config so the per-layer
/// checkpoint indices match the reference `nn.Sequential`.
struct EncodecSEANet {
    /// A heterogeneous layer in the stack. Each case carries the work
    /// closure plus retains its underlying struct so weights stay alive.
    enum Layer {
        case conv(EncodecConv1d)
        case convT(EncodecConvTranspose1d)
        case resnet(EncodecResnetBlock)
        case lstm(EncodecLSTMBlock)
        case elu
    }

    let layers: [Layer]

    init(weights w: EncodecWeights, config: EncodecConfig,
         prefix: String, isDecoder: Bool) throws {
        var ls: [Layer] = []
        var idx = 0
        func conv(_ inC: Int, _ outC: Int, k: Int, stride: Int = 1,
                  dilation: Int = 1) throws -> EncodecConv1d {
            defer { idx += 1 }
            return try EncodecConv1d(
                weights: w, prefix: "\(prefix).layers.\(idx).conv",
                stride: stride, dilation: dilation, config: config)
        }
        func convT(stride: Int) throws -> EncodecConvTranspose1d {
            defer { idx += 1 }
            return try EncodecConvTranspose1d(
                weights: w, prefix: "\(prefix).layers.\(idx).conv",
                stride: stride, config: config)
        }
        func resnet(dim: Int, dilation: Int) throws -> EncodecResnetBlock {
            defer { idx += 1 }
            return try EncodecResnetBlock(
                weights: w, prefix: "\(prefix).layers.\(idx)",
                dim: dim, dilation: dilation, config: config)
        }
        func lstm(dim: Int) throws -> EncodecLSTMBlock {
            defer { idx += 1 }
            return try EncodecLSTMBlock(
                weights: w, prefix: "\(prefix).layers.\(idx)",
                dimension: dim, numLayers: config.numLstmLayers)
        }

        if !isDecoder {
            // ── Encoder ──
            ls.append(.conv(try conv(config.audioChannels, config.numFilters,
                                     k: config.kernelSize)))
            var scaling = 1
            for ratio in config.upsamplingRatios.reversed() {
                let scale = scaling * config.numFilters
                for j in 0..<config.numResidualLayers {
                    let dil = intPow(config.dilationGrowthRate, j)
                    ls.append(.resnet(try resnet(dim: scale, dilation: dil)))
                }
                ls.append(.elu)
                ls.append(.conv(try conv(scale, scale * 2,
                                         k: ratio * 2, stride: ratio)))
                scaling *= 2
            }
            ls.append(.lstm(try lstm(dim: scaling * config.numFilters)))
            ls.append(.elu)
            ls.append(.conv(try conv(scaling * config.numFilters,
                                     config.hiddenSize, k: config.lastKernelSize)))
        } else {
            // ── Decoder ──
            var scaling = 1 << config.upsamplingRatios.count
            ls.append(.conv(try conv(config.hiddenSize,
                                     scaling * config.numFilters,
                                     k: config.kernelSize)))
            ls.append(.lstm(try lstm(dim: scaling * config.numFilters)))
            for ratio in config.upsamplingRatios {
                let scale = scaling * config.numFilters
                ls.append(.elu)
                ls.append(.convT(try convT(stride: ratio)))
                for j in 0..<config.numResidualLayers {
                    let dil = intPow(config.dilationGrowthRate, j)
                    ls.append(.resnet(try resnet(dim: scale / 2, dilation: dil)))
                }
                scaling /= 2
            }
            ls.append(.elu)
            ls.append(.conv(try conv(config.numFilters, config.audioChannels,
                                     k: config.lastKernelSize)))
        }
        self.layers = ls
    }

    func forward(_ data: inout [Float],
                 shape: inout [Int]) -> (data: [Float], shape: [Int]) {
        var (d, s) = (data, shape)
        for layer in layers {
            switch layer {
            case .conv(let c):   (d, s) = c(d, shape: s)
            case .convT(let c):  (d, s) = c(d, shape: s)
            case .resnet(let r): (d, s) = r(d, shape: s)
            case .lstm(let l):   (d, s) = l(d, shape: s)
            case .elu:           d = AudioMath.elu(d)
            }
        }
        return (d, s)
    }
}

/// Integer power `base^exp`.
private func intPow(_ base: Int, _ exp: Int) -> Int {
    var r = 1
    for _ in 0..<exp { r *= base }
    return r
}

// MARK: - Residual VQ quantizer

/// A single Euclidean-distance VQ codebook.
struct EncodecVQCodebook {
    let embed: [Float]       // [codebookSize, codebookDim]
    let codebookSize: Int
    let codebookDim: Int

    init(weights w: EncodecWeights, prefix: String,
         codebookSize: Int, codebookDim: Int) throws {
        // Reference key: quantizer.layers.{i}.codebook.embed
        self.embed = try w.floats("\(prefix).embed")
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
    }

    /// Memberwise initializer for tests that supply an in-memory table.
    init(embed: [Float], codebookSize: Int, codebookDim: Int) {
        self.embed = embed
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
    }

    /// Nearest-codebook lookup over a `[T, codebookDim]` matrix.
    func encode(_ x: [Float], rows t: Int) -> [Int32] {
        var indices = [Int32](repeating: 0, count: t)
        // Precompute ||embed||^2 per entry.
        var embedSq = [Float](repeating: 0, count: codebookSize)
        for c in 0..<codebookSize {
            var ss: Float = 0
            let base = c * codebookDim
            for d in 0..<codebookDim { ss += embed[base + d] * embed[base + d] }
            embedSq[c] = ss
        }
        for i in 0..<t {
            let xBase = i * codebookDim
            var best: Float = .greatestFiniteMagnitude
            var bestIdx = 0
            for c in 0..<codebookSize {
                let cBase = c * codebookDim
                var dot: Float = 0
                for d in 0..<codebookDim { dot += x[xBase + d] * embed[cBase + d] }
                // distance ∝ ||x||^2 - 2 x·e + ||e||^2; ||x||^2 is shared.
                let dist = embedSq[c] - 2 * dot
                if dist < best { best = dist; bestIdx = c }
            }
            indices[i] = Int32(bestIdx)
        }
        return indices
    }

    /// Reconstruct a `[T, codebookDim]` matrix from codes.
    func decode(codes: [Int32]) -> [Float] {
        var out = [Float](repeating: 0, count: codes.count * codebookDim)
        for (i, code) in codes.enumerated() {
            let cBase = Int(code) * codebookDim
            let oBase = i * codebookDim
            for d in 0..<codebookDim { out[oBase + d] = embed[cBase + d] }
        }
        return out
    }
}

/// EnCodec's residual vector quantizer — a stack of Euclidean codebooks
/// applied to successive residuals. The number of *active* codebooks is
/// chosen from the requested bandwidth.
struct EncodecResidualVQ {
    let codebooks: [EncodecVQCodebook]
    let codebookSize: Int
    let frameRate: Int

    init(weights w: EncodecWeights, config: EncodecConfig) throws {
        self.codebookSize = config.codebookSize
        self.frameRate = config.frameRate
        let maxBandwidth = config.targetBandwidths.max() ?? 24.0
        // Split into explicitly-typed sub-expressions — the inline
        // mixed-literal form overwhelmed the type-checker.
        let bandwidthScaled = Double(maxBandwidth) * 1000.0
        let quantizerDenom = Double(frameRate * 10)
        let numQuantizers = Int(bandwidthScaled / quantizerDenom)
        var cbs: [EncodecVQCodebook] = []
        for i in 0..<max(numQuantizers, 1) {
            cbs.append(try EncodecVQCodebook(
                weights: w, prefix: "quantizer.layers.\(i).codebook",
                codebookSize: config.codebookSize,
                codebookDim: config.codebookDim))
        }
        self.codebooks = cbs
    }

    /// Active codebook count for a target bandwidth (kbps).
    func numCodebooks(forBandwidth bw: Float) -> Int {
        let bwPerQ = log2(Double(codebookSize)) * Double(frameRate)
        if bw > 0 {
            return min(codebooks.count,
                       max(1, Int(floor(Double(bw) * 1000.0 / bwPerQ))))
        }
        return codebooks.count
    }

    /// Encode an NCL latent `[1, codebookDim, T]` into code streams.
    func encode(_ z: [Float], shape: [Int],
                bandwidth: Float) throws -> [[Int32]] {
        let (n, c, t) = (shape[0], shape[1], shape[2])
        precondition(n == 1, "EncodecResidualVQ: batch must be 1")
        // Rearrange b c t -> (b t) c.
        var residual = [Float](repeating: 0, count: t * c)
        for i in 0..<t {
            for ch in 0..<c { residual[i * c + ch] = z[ch * t + i] }
        }
        let active = numCodebooks(forBandwidth: bandwidth)
        var codes: [[Int32]] = []
        for q in 0..<active {
            let cb = codebooks[q]
            let idx = cb.encode(residual, rows: t)
            let quantized = cb.decode(codes: idx)   // [T, codebookDim]
            for i in 0..<residual.count { residual[i] -= quantized[i] }
            codes.append(idx)
        }
        return codes
    }

    /// Decode code streams into an NCL latent `[1, codebookDim, T]`.
    func decode(codes: [[Int32]]) throws -> (data: [Float], shape: [Int]) {
        guard let first = codes.first else {
            throw EncodecError.shapeMismatch("empty code list")
        }
        let t = first.count
        let dim = codebooks.first?.codebookDim ?? 0
        var acc = [Float](repeating: 0, count: t * dim)   // [T, dim]
        for (q, stream) in codes.enumerated() {
            let quantized = codebooks[q].decode(codes: stream)
            for i in 0..<acc.count { acc[i] += quantized[i] }
        }
        // Rearrange (b t) c -> b c t.
        var out = [Float](repeating: 0, count: dim * t)
        for i in 0..<t {
            for ch in 0..<dim { out[ch * t + i] = acc[i * dim + ch] }
        }
        return (out, [1, dim, t])
    }
}
