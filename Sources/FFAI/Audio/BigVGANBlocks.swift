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
// BigVGANBlocks — the AMP residual block and the anti-aliased periodic
// activation for the BigVGAN vocoder.
//
// Companion to BigVGAN.swift. Mirrors the weight layout of the
// reference MLX BigVGAN so checkpoint keys line up exactly:
//
//   resblocks.{i}.convs1.{j}.{weight_g,weight_v,bias}   (AMPBlock1)
//   resblocks.{i}.convs2.{j}.{weight_g,weight_v,bias}
//   resblocks.{i}.activations.{j}.act.{alpha,beta}
//   resblocks.{i}.convs.{j}…                            (AMPBlock2)
//   activation_post.act.{alpha,beta}
//
// The anti-aliasing low-pass filters (Kaiser-windowed sinc) are *not*
// checkpoint weights — they are derived from fixed cutoff / half-width
// constants, so they are computed at load time.
//
// All math is CPU-native — see AudioPrimitives.swift.

import Foundation

// MARK: - Kaiser-sinc anti-aliasing filter

/// Builds the fixed Kaiser-windowed sinc low-pass filters BigVGAN uses
/// for anti-aliased up/down-sampling around each periodic activation.
enum BigVGANFilter {
    /// Normalized sinc, `sin(πx)/(πx)`.
    private static func sinc(_ x: Double) -> Double {
        abs(x) < 1e-12 ? 1.0 : sin(.pi * x) / (.pi * x)
    }

    /// Modified Bessel function of the first kind, order 0.
    private static func besselI0(_ x: Double) -> Double {
        let y = (x * x) / 4.0
        var term = 1.0, sum = 1.0
        for k in 1...40 {
            let kk = Double(k)
            term *= y / (kk * kk)
            sum += term
            if term < 1e-12 * sum { break }
        }
        return sum
    }

    /// A Kaiser window of `size` samples with shape parameter `beta`.
    private static func kaiserWindow(size: Int, beta: Double) -> [Double] {
        if size <= 1 { return [1.0] }
        let denom = besselI0(beta)
        let half = Double(size - 1) / 2.0
        return (0..<size).map { idx in
            let ratio = (Double(idx) - half) / half
            return besselI0(beta * (max(0.0, 1.0 - ratio * ratio)).squareRoot()) / denom
        }
    }

    /// The Kaiser-sinc low-pass kernel `[kernelSize]`, sum-normalized.
    static func kaiserSinc(cutoff: Double, halfWidth: Double,
                           kernelSize: Int) -> [Float] {
        let even = kernelSize % 2 == 0
        let halfSize = kernelSize / 2
        let deltaF = 4.0 * halfWidth
        let a = 2.285 * Double(max(halfSize - 1, 0)) * .pi * deltaF + 7.95
        let beta: Double
        if a > 50.0 {
            beta = 0.1102 * (a - 8.7)
        } else if a >= 21.0 {
            beta = 0.5842 * pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0)
        } else {
            beta = 0.0
        }
        guard cutoff > 0 else {
            return [Float](repeating: 0, count: kernelSize)
        }
        let window = kaiserWindow(size: kernelSize, beta: beta)
        var filter = [Float](repeating: 0, count: kernelSize)
        for idx in 0..<kernelSize {
            let time = even ? Double(idx - halfSize) + 0.5
                            : Double(idx - halfSize)
            filter[idx] = Float(2.0 * cutoff * window[idx]
                                * sinc(2.0 * cutoff * time))
        }
        var sum: Float = 0
        for v in filter { sum += v }
        let inv = 1.0 / max(sum, 1e-12)
        for i in 0..<kernelSize { filter[i] *= inv }
        return filter
    }
}

// MARK: - Periodic (Snake / SnakeBeta) activation

/// BigVGAN's periodic nonlinearity: `x + (1/(β+eps)) · sin(αx)²`. For
/// plain Snake, `β == α`. With `snakeLogscale`, the stored `alpha`/
/// `beta` are log-domain (exponentiated at use).
struct BigVGANPeriodicActivation {
    let alpha: [Float]       // per-channel
    let beta: [Float]        // per-channel (== alpha for plain snake)
    let logscale: Bool

    init(weights w: BigVGANWeights, prefix: String, channels: Int,
         config: BigVGANConfig) throws {
        let a = try w.floats("\(prefix).alpha")
        let useBeta = config.activation == .snakebeta
        let b = (useBeta && w.has("\(prefix).beta"))
            ? try w.floats("\(prefix).beta") : a
        self.alpha = a
        self.beta = b
        self.logscale = config.snakeLogscale
    }

    /// Apply to an NCL tensor `[1, C, L]`.
    func callAsFunction(_ x: [Float], shape: [Int]) -> [Float] {
        let (n, c, l) = (shape[0], shape[1], shape[2])
        precondition(alpha.count == c, "BigVGANPeriodicActivation: channel count")
        var out = [Float](repeating: 0, count: x.count)
        for b in 0..<n {
            for ch in 0..<c {
                let a = logscale ? expf(alpha[ch]) : alpha[ch]
                let bt = logscale ? expf(beta[ch]) : beta[ch]
                let recip = 1.0 / (bt + 1e-9)
                let base = (b * c + ch) * l
                for i in 0..<l {
                    let v = x[base + i]
                    let s = sinf(a * v)
                    out[base + i] = v + recip * s * s
                }
            }
        }
        return out
    }
}

// MARK: - Anti-aliased activation

/// BigVGAN's anti-aliased activation — upsample 2× (Kaiser-sinc), apply
/// the periodic nonlinearity, then downsample 2× (Kaiser-sinc low-pass).
/// This suppresses the aliasing the nonlinearity would otherwise inject.
struct BigVGANActivation {
    let act: BigVGANPeriodicActivation
    /// Up/down anti-aliasing filters (per-channel depthwise kernels).
    private let upFilter: [Float]
    private let downFilter: [Float]
    private let upKernelSize: Int
    private let downKernelSize: Int
    private static let ratio = 2

    init(weights w: BigVGANWeights, prefix: String, channels: Int,
         config: BigVGANConfig) throws {
        self.act = try BigVGANPeriodicActivation(
            weights: w, prefix: prefix, channels: channels, config: config)
        // Fixed up/down filters (kernelSize 12, default ratios).
        let r = Double(Self.ratio)
        self.upKernelSize = 12
        self.downKernelSize = 12
        self.upFilter = BigVGANFilter.kaiserSinc(
            cutoff: 0.5 / r, halfWidth: 0.6 / r, kernelSize: upKernelSize)
        self.downFilter = BigVGANFilter.kaiserSinc(
            cutoff: 0.5 / r, halfWidth: 0.6 / r, kernelSize: downKernelSize)
    }

    /// Upsample an NCL tensor 2× with the Kaiser-sinc filter.
    private func upsample(_ x: [Float],
                          shape: [Int]) -> (data: [Float], shape: [Int]) {
        let (n, c, _) = (shape[0], shape[1], shape[2])
        let stride = Self.ratio
        let pad = upKernelSize / stride - 1
        let padLeft = pad * stride + (upKernelSize - stride) / 2
        let padRight = pad * stride + (upKernelSize - stride + 1) / 2
        // Edge-pad, then per-channel transposed conv (depthwise).
        let (padded, ps) = edgePad(x, shape: shape, left: pad, right: pad)
        // Depthwise weight: [C, 1, K].
        var w = [Float](repeating: 0, count: c * upKernelSize)
        for ch in 0..<c {
            for k in 0..<upKernelSize { w[ch * upKernelSize + k] = upFilter[k] }
        }
        var (out, os) = AudioMath.convTransposed1d(
            x: padded, xShape: ps, weight: w, wShape: [c, 1, upKernelSize],
            bias: nil, stride: stride, padding: 0, dilation: 1,
            outputPadding: 0, groups: c)
        // Scale by the ratio, then trim the transient padding.
        let ratioF = Float(Self.ratio)
        for i in 0..<out.count { out[i] *= ratioF }
        let end = os[2] - padRight
        if end > padLeft {
            (out, os) = sliceTime(out, shape: os, start: padLeft, end: end)
        }
        _ = n
        return (out, os)
    }

    /// Downsample an NCL tensor 2× with the Kaiser-sinc low-pass filter.
    private func downsample(_ x: [Float],
                            shape: [Int]) -> (data: [Float], shape: [Int]) {
        let (_, c, _) = (shape[0], shape[1], shape[2])
        let even = downKernelSize % 2 == 0
        let padLeft = downKernelSize / 2 - (even ? 1 : 0)
        let padRight = downKernelSize / 2
        let (padded, ps) = edgePad(x, shape: shape, left: padLeft, right: padRight)
        var w = [Float](repeating: 0, count: c * downKernelSize)
        for ch in 0..<c {
            for k in 0..<downKernelSize { w[ch * downKernelSize + k] = downFilter[k] }
        }
        return AudioMath.conv1d(
            x: padded, xShape: ps, weight: w, wShape: [c, 1, downKernelSize],
            bias: nil, stride: Self.ratio, padding: 0, dilation: 1,
            groups: c)
    }

    /// Edge-replication padding on the time axis of an NCL tensor.
    private func edgePad(_ x: [Float], shape: [Int], left: Int,
                         right: Int) -> (data: [Float], shape: [Int]) {
        let (n, c, l) = (shape[0], shape[1], shape[2])
        let lOut = l + left + right
        var out = [Float](repeating: 0, count: n * c * lOut)
        for b in 0..<n {
            for ch in 0..<c {
                let inBase = (b * c + ch) * l
                let outBase = (b * c + ch) * lOut
                for t in 0..<lOut {
                    let src = min(max(t - left, 0), l - 1)
                    out[outBase + t] = x[inBase + src]
                }
            }
        }
        return (out, [n, c, lOut])
    }

    /// Crop an NCL tensor to `[start, end)` on the time axis.
    private func sliceTime(_ x: [Float], shape: [Int], start: Int,
                           end: Int) -> (data: [Float], shape: [Int]) {
        let (n, c, l) = (shape[0], shape[1], shape[2])
        let lOut = max(end - start, 0)
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

    /// downsample(act(upsample(x))).
    func callAsFunction(_ x: [Float],
                        shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (h, s) = upsample(x, shape: shape)
        h = act(h, shape: s)
        return downsample(h, shape: s)
    }
}

// MARK: - AMP residual block

/// BigVGAN's multi-receptive-field "AMP" residual block. Variant 1 has
/// dilated conv pairs (`convs1` / `convs2`); variant 2 has a single
/// dilated conv per branch. Each branch is `activation → conv` and adds
/// back into the running sum.
struct BigVGANAMPBlock {
    /// Variant-1 branch: two activations + two convs.
    /// Variant-2 branch: one activation + one conv.
    struct Branch {
        let act1: BigVGANActivation
        let conv1: SNACWNConv1d
        let act2: BigVGANActivation?
        let conv2: SNACWNConv1d?
    }
    let branches: [Branch]

    init(weights w: BigVGANWeights, prefix: String, channels: Int,
         kernelSize: Int, dilations: [Int], config: BigVGANConfig) throws {
        var bs: [Branch] = []
        if config.resblock == .one {
            // AMPBlock1: convs1[j] (dilated) + convs2[j] (dilation 1).
            for (j, dil) in dilations.enumerated() {
                let pad1 = ((kernelSize - 1) * dil) / 2
                let conv1 = try w.wnConv1d(
                    prefix: "\(prefix).convs1.\(j)", stride: 1,
                    padding: pad1, dilation: dil, groups: 1)
                let conv2 = try w.wnConv1d(
                    prefix: "\(prefix).convs2.\(j)", stride: 1,
                    padding: (kernelSize - 1) / 2, dilation: 1, groups: 1)
                // activations[2j], activations[2j+1].
                let act1 = try BigVGANActivation(
                    weights: w, prefix: "\(prefix).activations.\(2 * j).act",
                    channels: channels, config: config)
                let act2 = try BigVGANActivation(
                    weights: w, prefix: "\(prefix).activations.\(2 * j + 1).act",
                    channels: channels, config: config)
                bs.append(Branch(act1: act1, conv1: conv1,
                                 act2: act2, conv2: conv2))
            }
        } else {
            // AMPBlock2: a single dilated conv per branch.
            for (j, dil) in dilations.enumerated() {
                let pad = ((kernelSize - 1) * dil) / 2
                let conv = try w.wnConv1d(
                    prefix: "\(prefix).convs.\(j)", stride: 1,
                    padding: pad, dilation: dil, groups: 1)
                let act = try BigVGANActivation(
                    weights: w, prefix: "\(prefix).activations.\(j).act",
                    channels: channels, config: config)
                bs.append(Branch(act1: act, conv1: conv,
                                 act2: nil, conv2: nil))
            }
        }
        self.branches = bs
    }

    func callAsFunction(_ x: [Float],
                        shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (out, s) = (x, shape)
        for branch in branches {
            var (h, hs) = branch.act1(out, shape: s)
            (h, hs) = branch.conv1(h, shape: hs)
            if let act2 = branch.act2, let conv2 = branch.conv2 {
                h = act2(h, shape: hs).data
                (h, hs) = conv2(h, shape: hs)
            }
            precondition(h.count == out.count,
                         "BigVGANAMPBlock: branch length mismatch")
            for i in 0..<out.count { out[i] += h[i] }
            _ = hs
        }
        return (out, s)
    }
}
