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
// BigVGAN — NVIDIA's BigVGAN neural vocoder.
//
// Port of `mlx-audio-swift/Sources/MLXAudioCodecs/BigVGAN`. BigVGAN is a
// *decode-only* GAN vocoder: a mel-spectrogram in, a waveform out. A
// `conv_pre` lifts the mel features, a stack of transposed-conv
// upsample stages interleaved with multi-receptive-field "AMP" residual
// blocks reconstructs the time-domain detail, and a `conv_post` mixes
// down to a single audio channel. Every activation is an anti-aliased
// periodic (Snake / SnakeBeta) nonlinearity.
//
// This is a genuine rewrite onto FFAI primitives — the MLX version is
// replaced by `AudioMath` CPU kernels, and the proven `WeightNorm`
// weight-normalized conv path is shared with SNAC / DAC. The vocoder
// runs once per utterance, so the CPU path is acceptable (see
// AudioPrimitives.swift for the rationale).
//
// Public surface:
//   BigVGAN.fromPretrained(directory:)   load weights from a HF snapshot
//   bigvgan.decode(mel:)                 -> Tensor   reconstructed audio
//
// Layout: feature/audio tensors are NCL `[batch, channels, length]`.

import Foundation

/// BigVGAN residual-block variant.
public enum BigVGANResBlockType: String, Codable, Sendable {
    case one = "1"
    case two = "2"
}

/// BigVGAN periodic-activation variant.
public enum BigVGANActivationType: String, Codable, Sendable {
    case snake
    case snakebeta
}

/// BigVGAN model configuration. Mirrors `config.json` of a BigVGAN
/// checkpoint.
public struct BigVGANConfig: Codable, Sendable {
    public var numMels: Int
    public var upsampleRates: [Int]
    public var upsampleKernelSizes: [Int]
    public var upsampleInitialChannel: Int
    public var resblock: BigVGANResBlockType
    public var resblockKernelSizes: [Int]
    public var resblockDilationSizes: [[Int]]
    public var activation: BigVGANActivationType
    public var snakeLogscale: Bool
    public var useBiasAtFinal: Bool
    public var useTanhAtFinal: Bool

    enum CodingKeys: String, CodingKey {
        case numMels = "num_mels"
        case upsampleRates = "upsample_rates"
        case upsampleKernelSizes = "upsample_kernel_sizes"
        case upsampleInitialChannel = "upsample_initial_channel"
        case resblock
        case resblockKernelSizes = "resblock_kernel_sizes"
        case resblockDilationSizes = "resblock_dilation_sizes"
        case activation
        case snakeLogscale = "snake_logscale"
        case useBiasAtFinal = "use_bias_at_final"
        case useTanhAtFinal = "use_tanh_at_final"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numMels = try c.decodeIfPresent(Int.self, forKey: .numMels) ?? 100
        upsampleRates =
            try c.decodeIfPresent([Int].self, forKey: .upsampleRates)
            ?? [4, 4, 2, 2, 2, 2]
        upsampleKernelSizes =
            try c.decodeIfPresent([Int].self, forKey: .upsampleKernelSizes)
            ?? [8, 8, 4, 4, 4, 4]
        upsampleInitialChannel =
            try c.decodeIfPresent(Int.self, forKey: .upsampleInitialChannel)
            ?? 1536
        resblock = try c.decodeIfPresent(BigVGANResBlockType.self, forKey: .resblock) ?? .one
        resblockKernelSizes =
            try c.decodeIfPresent([Int].self, forKey: .resblockKernelSizes)
            ?? [3, 7, 11]
        resblockDilationSizes =
            try c.decodeIfPresent([[Int]].self, forKey: .resblockDilationSizes)
            ?? [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        activation =
            try c.decodeIfPresent(BigVGANActivationType.self, forKey: .activation)
            ?? .snakebeta
        snakeLogscale = try c.decodeIfPresent(Bool.self, forKey: .snakeLogscale) ?? true
        useBiasAtFinal = try c.decodeIfPresent(Bool.self, forKey: .useBiasAtFinal) ?? true
        useTanhAtFinal = try c.decodeIfPresent(Bool.self, forKey: .useTanhAtFinal) ?? true
    }

    /// Number of distinct resblock kernel sizes per upsample stage.
    public var numKernels: Int { resblockKernelSizes.count }
    /// Number of upsample stages.
    public var numUpsamples: Int { upsampleRates.count }
}

public enum BigVGANError: Error, CustomStringConvertible {
    case missingWeights(String)
    case configNotFound(String)
    case shapeMismatch(String)

    public var description: String {
        switch self {
        case .missingWeights(let s): return "BigVGAN: missing weights — \(s)"
        case .configNotFound(let s): return "BigVGAN: config not found — \(s)"
        case .shapeMismatch(let s): return "BigVGAN: shape mismatch — \(s)"
        }
    }
}

// MARK: - Weight loading helper

/// Loads BigVGAN weights from a `SafeTensorsBundle`, reconstructing
/// weight-normalized convs on the fly. BigVGAN stores conv weight as
/// PyTorch `[Cout, Cin/groups, K]` with a `weight_g`/`weight_v` pair —
/// identical to SNAC, so the `WeightNorm` helper is reused.
struct BigVGANWeights {
    let bundle: SafeTensorsBundle

    func floats(_ key: String) throws -> [Float] {
        AudioMath.floats(try bundle.tensor(named: key))
    }

    func shape(_ key: String) throws -> [Int] {
        try bundle.tensor(named: key).shape
    }

    func has(_ key: String) -> Bool { bundle.has(key) }

    /// Reconstruct a weight-normalized conv at `prefix`.
    func wnConv1d(
        prefix: String, stride: Int, padding: Int,
        dilation: Int, groups: Int
    ) throws -> SNACWNConv1d {
        let gKey = "\(prefix).weight_g"
        let vKey = "\(prefix).weight_v"
        guard has(gKey), has(vKey) else {
            throw BigVGANError.missingWeights(prefix)
        }
        let g = try floats(gKey)
        let v = try floats(vKey)
        let vShape = try shape(vKey)
        let weight = WeightNorm.effectiveWeight(
            g: g, v: v, shape: vShape,
            exceptDim: 0)
        let bias = has("\(prefix).bias") ? try floats("\(prefix).bias") : nil
        return SNACWNConv1d(
            weight: weight, wShape: vShape, bias: bias,
            stride: stride, padding: padding,
            dilation: dilation, groups: groups)
    }

    /// Reconstruct a weight-normalized transposed conv at `prefix`.
    func wnConvTranspose1d(
        prefix: String, stride: Int, padding: Int,
        groups: Int
    ) throws -> SNACWNConvTranspose1d {
        let gKey = "\(prefix).weight_g"
        let vKey = "\(prefix).weight_v"
        guard has(gKey), has(vKey) else {
            throw BigVGANError.missingWeights(prefix)
        }
        let g = try floats(gKey)
        let v = try floats(vKey)
        let vShape = try shape(vKey)
        let weight = WeightNorm.effectiveWeight(
            g: g, v: v, shape: vShape,
            exceptDim: 0)
        let bias = has("\(prefix).bias") ? try floats("\(prefix).bias") : nil
        return SNACWNConvTranspose1d(
            weight: weight, wShape: vShape, bias: bias,
            stride: stride, padding: padding,
            outputPadding: 0, groups: groups)
    }
}

// MARK: - BigVGAN

/// BigVGAN neural vocoder — `conv_pre`, upsample + AMP-resblock stages,
/// `conv_post`.
public final class BigVGAN: @unchecked Sendable {
    public let config: BigVGANConfig

    private let convPre: SNACWNConv1d
    private let upsamples: [SNACWNConvTranspose1d]
    /// `numUpsamples * numKernels` AMP residual blocks, stage-major.
    private let resblocks: [BigVGANAMPBlock]
    private let activationPost: BigVGANActivation
    private let convPost: SNACWNConv1d

    /// Load a BigVGAN model from a Hugging Face snapshot directory
    /// containing `config.json` + a `*.safetensors` weights file.
    public static func fromPretrained(directory: URL) throws -> BigVGAN {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw BigVGANError.configNotFound(configURL.path)
        }
        let config = try JSONDecoder().decode(
            BigVGANConfig.self,
            from: Data(contentsOf: configURL))
        let bundle = try SafeTensorsBundle(directory: directory)
        return try BigVGAN(config: config, bundle: bundle)
    }

    init(config: BigVGANConfig, bundle: SafeTensorsBundle) throws {
        self.config = config
        let w = BigVGANWeights(bundle: bundle)

        // conv_pre — WNConv1d(numMels -> initialChannel, k=7, pad=3).
        self.convPre = try w.wnConv1d(
            prefix: "conv_pre", stride: 1,
            padding: 3, dilation: 1, groups: 1)

        // Upsample stages — WNConvTranspose1d, halving channels each step.
        var ups: [SNACWNConvTranspose1d] = []
        for (i, pair) in zip(
            config.upsampleRates,
            config.upsampleKernelSizes
        ).enumerated() {
            let (stride, ksize) = pair
            ups.append(
                try w.wnConvTranspose1d(
                    prefix: "ups.\(i).0", stride: stride,
                    padding: (ksize - stride) / 2, groups: 1))
        }
        self.upsamples = ups

        // AMP residual blocks — stage-major: stage s, kernel k.
        var blocks: [BigVGANAMPBlock] = []
        for s in 0 ..< config.numUpsamples {
            let channels = config.upsampleInitialChannel / (1 << (s + 1))
            for k in 0 ..< config.numKernels {
                let blockIdx = s * config.numKernels + k
                blocks.append(
                    try BigVGANAMPBlock(
                        weights: w, prefix: "resblocks.\(blockIdx)",
                        channels: channels,
                        kernelSize: config.resblockKernelSizes[k],
                        dilations: config.resblockDilationSizes[k],
                        config: config))
            }
        }
        self.resblocks = blocks

        // Tail — anti-aliased activation + WNConv1d(-> 1, k=7, pad=3).
        let finalChannels =
            config.upsampleInitialChannel
            / (1 << config.numUpsamples)
        self.activationPost = try BigVGANActivation(
            weights: w, prefix: "activation_post.act",
            channels: finalChannels, config: config)
        self.convPost = try w.wnConv1d(
            prefix: "conv_post", stride: 1,
            padding: 3, dilation: 1, groups: 1)
    }

    /// Number of input mel channels expected by `decode`.
    public var melChannels: Int { config.numMels }

    // ─── Decode ───────────────────────────────────────────────────

    /// Decode a mel-spectrogram into a waveform Tensor `[1, 1, L]`.
    ///
    /// - mel: an f32 Tensor of shape `[numMels, T]` or `[1, numMels, T]`.
    public func decode(mel: Tensor) throws -> Tensor {
        let raw = AudioMath.floats(mel)
        let shape: [Int]
        switch mel.shape.count {
        case 2: shape = [1, mel.shape[0], mel.shape[1]]
        case 3: shape = mel.shape
        default:
            throw BigVGANError.shapeMismatch(
                "decode expects a mel of rank 2 or 3, got \(mel.shape)")
        }

        var (h, s) = convPre(raw, shape: shape)
        for step in 0 ..< config.numUpsamples {
            (h, s) = upsamples[step](h, shape: s)
            // Sum the per-kernel resblocks for this stage, then average.
            let base = step * config.numKernels
            var (acc, accShape) = resblocks[base](h, shape: s)
            for k in 1 ..< config.numKernels {
                let (out, _) = resblocks[base + k](h, shape: s)
                precondition(
                    out.count == acc.count,
                    "BigVGAN.decode: resblock length mismatch")
                for i in 0 ..< acc.count { acc[i] += out[i] }
            }
            let inv = 1.0 / Float(config.numKernels)
            for i in 0 ..< acc.count { acc[i] *= inv }
            (h, s) = (acc, accShape)
        }

        (h, s) = activationPost(h, shape: s)
        (h, s) = convPost(h, shape: s)
        // Final nonlinearity: tanh, or clip to [-1, 1].
        if config.useTanhAtFinal {
            h = AudioMath.tanhAll(h)
        } else {
            for i in 0 ..< h.count { h[i] = min(max(h[i], -1.0), 1.0) }
        }
        let out = Tensor.empty(shape: s, dtype: .f32)
        out.copyIn(from: h)
        return out
    }
}
