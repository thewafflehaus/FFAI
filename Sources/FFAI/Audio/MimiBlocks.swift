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
// MimiBlocks — SEANet stack, latent Transformer, and the split
// residual-VQ quantizer for the Mimi codec.
//
// Companion to Mimi.swift. Mirrors the (sanitized) weight layout of the
// reference MLX Mimi so checkpoint keys line up exactly:
//
//   encoder.init_conv1d.conv.{weight,bias}
//   encoder.layers.{i}.residuals.0.block.{0,1}.conv.{weight,bias}
//   encoder.layers.{i}.downsample.conv.{weight,bias}
//   encoder.final_conv1d.conv.{weight,bias}
//   encoder_transformer.transformer.layers.{l}.{self_attn,gating,norm*}…
//   quantizer.rvq_{first,rest}.{input,output}_proj.weight
//   quantizer.rvq_*.vq.layers.{q}.codebook.{embedding_sum,cluster_usage}
//
// The reference is streaming; this port runs whole utterances in one
// pass (causal mask, no KV cache). All math is CPU-native — see
// AudioPrimitives.swift.

import Foundation

// MARK: - Streamable conv (whole-sequence, causal padding)

/// Mimi's `StreamableConv1d` collapsed to a single full-sequence call.
/// Pads the time axis (causal: all padding on the left), then convolves.
struct MimiStreamableConv1d {
    let weight: [Float]  // [Cout, Cin/groups, K]
    let wShape: [Int]
    let bias: [Float]?
    let stride: Int
    let dilation: Int
    let groups: Int
    let padMode: MimiPadMode

    var kEff: Int { (wShape[2] - 1) * dilation + 1 }
    var paddingTotal: Int { kEff - stride }

    init(
        weights w: MimiWeights, prefix: String, stride: Int,
        dilation: Int, groups: Int, padMode: MimiPadMode
    ) throws {
        let (cw, cs) = try w.convWeight("\(prefix).conv.weight")
        self.weight = cw
        self.wShape = cs
        self.bias =
            w.has("\(prefix).conv.bias")
            ? try w.floats("\(prefix).conv.bias") : nil
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padMode = padMode
    }

    /// Extra right-padding so the strided conv lands on whole frames.
    private func extraPadding(length: Int) -> Int {
        let nframes = max(length + paddingTotal - kEff, 0)
        let nf = Double(nframes) / Double(stride) + 1.0
        let idealLen = (Int(ceil(nf)) - 1) * stride + kEff - paddingTotal
        return max(0, idealLen - length)
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        let length = shape[2]
        let extra = extraPadding(length: length)
        // Causal: all base padding on the left, extra on the right.
        let (left, right) = (paddingTotal, extra)
        var (padded, ps) = (x, shape)
        if left > 0 || right > 0 {
            (padded, ps) = MimiPad.pad(
                x, shape: shape, left: left,
                right: right, mode: padMode)
        }
        return AudioMath.conv1d(
            x: padded, xShape: ps, weight: weight, wShape: wShape,
            bias: bias, stride: stride, padding: 0,
            dilation: dilation, groups: groups)
    }
}

/// Mimi's `StreamableConvTranspose1d` collapsed to a single call. After
/// the transposed conv, trims the trailing `paddingTotal` samples
/// (causal output trimming).
struct MimiStreamableConvTranspose1d {
    let weight: [Float]  // [Cin, Cout/groups, K]
    let wShape: [Int]
    let bias: [Float]?
    let stride: Int
    let groups: Int

    var ksize: Int { wShape[2] }
    var paddingTotal: Int { max(ksize - stride, 0) }

    init(
        weights w: MimiWeights, prefix: String, stride: Int,
        groups: Int
    ) throws {
        // Mimi stores transposed-conv weight as [Cout, K, Cin/groups]
        // (MLX NLC). convTransposed1d wants [Cin, Cout/groups, K].
        let raw = try w.floats("\(prefix).convtr.weight")
        let s = try w.shape("\(prefix).convtr.weight")
        let (cOut, k, cInPerG) = (s[0], s[1], s[2])
        if groups == cOut && cInPerG == 1 {
            // Depthwise: weight ships as [C, K, 1]; effective per-group
            // weight is [Cin=C, Cout/groups=1, K].
            var out = [Float](repeating: 0, count: raw.count)
            for c in 0 ..< cOut {
                for kk in 0 ..< k {
                    out[(c * 1 + 0) * k + kk] = raw[(c * k + kk) * 1 + 0]
                }
            }
            self.weight = out
            self.wShape = [cOut, 1, k]
        } else {
            // Regular: [Cout, K, Cin] -> [Cin, Cout, K].
            var out = [Float](repeating: 0, count: raw.count)
            for o in 0 ..< cOut {
                for kk in 0 ..< k {
                    for ic in 0 ..< cInPerG {
                        out[(ic * cOut + o) * k + kk] = raw[(o * k + kk) * cInPerG + ic]
                    }
                }
            }
            self.weight = out
            self.wShape = [cInPerG, cOut, k]
        }
        self.bias =
            w.has("\(prefix).convtr.bias")
            ? try w.floats("\(prefix).convtr.bias") : nil
        self.stride = stride
        self.groups = groups
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (out, s) = AudioMath.convTransposed1d(
            x: x, xShape: shape, weight: weight, wShape: wShape,
            bias: bias, stride: stride, padding: 0, dilation: 1,
            outputPadding: 0, groups: groups)
        // Causal output trimming: drop the trailing paddingTotal samples.
        if paddingTotal > 0 {
            (out, s) = MimiPad.sliceTime(
                out, shape: s, start: 0,
                end: s[2] - paddingTotal)
        }
        return (out, s)
    }
}

// MARK: - SEANet resnet block

/// Mimi's `SeanetResnetBlock`: `[ELU, Conv(residualKsize, dilated),
/// ELU, Conv(k=1)]` plus a `trueSkip` (identity) residual.
struct MimiSeanetResnetBlock {
    let conv0: MimiStreamableConv1d
    let conv1: MimiStreamableConv1d
    let shortcut: MimiStreamableConv1d?

    init(
        weights w: MimiWeights, prefix: String, config: MimiConfig,
        dilation: Int
    ) throws {
        // block.0 — Conv(residualKsize, dilation); block.1 — Conv(k=1).
        self.conv0 = try MimiStreamableConv1d(
            weights: w, prefix: "\(prefix).block.0", stride: 1,
            dilation: dilation, groups: 1, padMode: .constant)
        self.conv1 = try MimiStreamableConv1d(
            weights: w, prefix: "\(prefix).block.1", stride: 1,
            dilation: 1, groups: 1, padMode: .constant)
        if !config.trueSkip && w.has("\(prefix).shortcut.conv.weight") {
            self.shortcut = try MimiStreamableConv1d(
                weights: w, prefix: "\(prefix).shortcut", stride: 1,
                dilation: 1, groups: 1, padMode: .constant)
        } else {
            self.shortcut = nil
        }
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (h, s) = (AudioMath.elu(x), shape)
        (h, s) = conv0(h, shape: s)
        h = AudioMath.elu(h)
        (h, s) = conv1(h, shape: s)
        let res = shortcut != nil ? shortcut!(x, shape: shape).data : x
        precondition(
            res.count == h.count,
            "MimiSeanetResnetBlock: residual length mismatch")
        var out = h
        for i in 0 ..< out.count { out[i] += res[i] }
        return (out, s)
    }
}

// MARK: - SEANet stack

/// A SEANet encoder or decoder — built procedurally from the config so
/// the per-layer checkpoint indices match the sanitized reference keys.
struct MimiSeanet {
    enum Op {
        case conv(MimiStreamableConv1d)
        case convT(MimiStreamableConvTranspose1d)
        case resnet(MimiSeanetResnetBlock)
        case elu
    }
    let ops: [Op]

    init(
        weights w: MimiWeights, config c: MimiConfig,
        prefix: String, isDecoder: Bool
    ) throws {
        var list: [Op] = []
        if !isDecoder {
            // ── Encoder ──
            list.append(
                .conv(
                    try MimiStreamableConv1d(
                        weights: w, prefix: "\(prefix).init_conv1d", stride: 1,
                        dilation: 1, groups: 1, padMode: .constant)))
            var mult = 1
            for (layerIdx, ratio) in c.ratios.reversed().enumerated() {
                var dilation = 1
                for _ in 0 ..< c.nresidualLayers {
                    list.append(
                        .resnet(
                            try MimiSeanetResnetBlock(
                                weights: w,
                                prefix: "\(prefix).layers.\(layerIdx).residuals.0",
                                config: c, dilation: dilation)))
                    dilation *= c.dilationBase
                }
                list.append(.elu)
                list.append(
                    .conv(
                        try MimiStreamableConv1d(
                            weights: w,
                            prefix: "\(prefix).layers.\(layerIdx).downsample",
                            stride: ratio, dilation: 1, groups: 1, padMode: .constant)))
                mult *= 2
            }
            list.append(.elu)
            list.append(
                .conv(
                    try MimiStreamableConv1d(
                        weights: w, prefix: "\(prefix).final_conv1d", stride: 1,
                        dilation: 1, groups: 1, padMode: .constant)))
        } else {
            // ── Decoder ──
            list.append(
                .conv(
                    try MimiStreamableConv1d(
                        weights: w, prefix: "\(prefix).init_conv1d", stride: 1,
                        dilation: 1, groups: 1, padMode: .constant)))
            for (layerIdx, ratio) in c.ratios.enumerated() {
                list.append(.elu)
                list.append(
                    .convT(
                        try MimiStreamableConvTranspose1d(
                            weights: w,
                            prefix: "\(prefix).layers.\(layerIdx).upsample",
                            stride: ratio, groups: 1)))
                var dilation = 1
                for _ in 0 ..< c.nresidualLayers {
                    list.append(
                        .resnet(
                            try MimiSeanetResnetBlock(
                                weights: w,
                                prefix: "\(prefix).layers.\(layerIdx).residuals.0",
                                config: c, dilation: dilation)))
                    dilation *= c.dilationBase
                }
            }
            list.append(.elu)
            list.append(
                .conv(
                    try MimiStreamableConv1d(
                        weights: w, prefix: "\(prefix).final_conv1d", stride: 1,
                        dilation: 1, groups: 1, padMode: .constant)))
        }
        self.ops = list
    }

    func forward(
        _ data: inout [Float],
        shape: inout [Int]
    ) -> (data: [Float], shape: [Int]) {
        var (d, s) = (data, shape)
        for op in ops {
            switch op {
            case .conv(let c): (d, s) = c(d, shape: s)
            case .convT(let c): (d, s) = c(d, shape: s)
            case .resnet(let r): (d, s) = r(d, shape: s)
            case .elu: d = AudioMath.elu(d)
            }
        }
        return (d, s)
    }
}

// MARK: - Conv resampler (latent down/up sample)

/// The latent `ConvDownsample1d` / `ConvTrUpsample1d` wrappers — a
/// single (transposed) streamable conv with no bias.
struct MimiConvResample {
    private let conv: MimiStreamableConv1d?
    private let convT: MimiStreamableConvTranspose1d?

    init(
        weights w: MimiWeights, prefix: String, config: MimiConfig,
        stride: Int, transposed: Bool
    ) throws {
        if transposed {
            // ConvTrUpsample uses a depthwise (groups=dim) transposed
            // conv with ksize 2*stride.
            self.convT = try MimiStreamableConvTranspose1d(
                weights: w, prefix: prefix, stride: stride,
                groups: config.seanetDim)
            self.conv = nil
        } else {
            self.conv = try MimiStreamableConv1d(
                weights: w, prefix: prefix, stride: stride,
                dilation: 1, groups: 1, padMode: .edge)
            self.convT = nil
        }
    }

    func forward(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        if let c = conv { return c(x, shape: shape) }
        return convT!(x, shape: shape)
    }
}

// MARK: - Padding

/// Mimi conv padding modes.
enum MimiPadMode { case constant, edge }

/// Time-axis padding helpers for the Mimi conv stack.
enum MimiPad {
    /// Pad the time axis of an NCL tensor — zeros (`constant`) or
    /// edge-replication (`edge`).
    static func pad(
        _ x: [Float], shape: [Int], left: Int, right: Int,
        mode: MimiPadMode
    ) -> (data: [Float], shape: [Int]) {
        switch mode {
        case .constant:
            return AudioMath.zeroPad1d(x, shape: shape, left: left, right: right)
        case .edge:
            let (n, c, l) = (shape[0], shape[1], shape[2])
            let lOut = l + left + right
            var out = [Float](repeating: 0, count: n * c * lOut)
            for b in 0 ..< n {
                for ch in 0 ..< c {
                    let inBase = (b * c + ch) * l
                    let outBase = (b * c + ch) * lOut
                    for t in 0 ..< lOut {
                        let src = min(max(t - left, 0), l - 1)
                        out[outBase + t] = x[inBase + src]
                    }
                }
            }
            return (out, [n, c, lOut])
        }
    }

    /// Crop an NCL tensor to `[start, end)` on the time axis.
    static func sliceTime(
        _ x: [Float], shape: [Int], start: Int,
        end: Int
    ) -> (data: [Float], shape: [Int]) {
        let (n, c, l) = (shape[0], shape[1], shape[2])
        let lOut = max(end - start, 0)
        var out = [Float](repeating: 0, count: n * c * lOut)
        for b in 0 ..< n {
            for ch in 0 ..< c {
                let inBase = (b * c + ch) * l
                let outBase = (b * c + ch) * lOut
                for t in 0 ..< lOut { out[outBase + t] = x[inBase + start + t] }
            }
        }
        return (out, [n, c, lOut])
    }
}
