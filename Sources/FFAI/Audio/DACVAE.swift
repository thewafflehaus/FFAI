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
// DACVAE — the VAE-style Descript Audio Codec used by SAM-Audio.
//
// Port of `mlx-audio-swift/Sources/MLXAudioCodecs/DACVAE`. Unlike the
// discrete-code DAC (`DescriptDAC.swift`), DACVAE has no residual-VQ: it
// is a *variational* autoencoder. The encoder produces a continuous
// latent (a VAE `mean`); the decoder reconstructs the waveform from that
// latent. There are no integer code streams.
//
// Public surface mirrors a continuous codec:
//   DACVAE.fromPretrained(directory:)        load from a HF snapshot
//   dacvae.encode(waveform:)   -> Tensor     latent  [1, codebookDim, T]
//   dacvae.decode(latents:)    -> Tensor     waveform [1, 1, L]
//
// This is a genuine rewrite onto FFAI primitives: the MLX path is
// replaced by `AudioMath` CPU kernels operating on FFAI `Tensor`s. The
// codec runs once per utterance, so the CPU path is acceptable (see
// AudioPrimitives.swift for the rationale).
//
// Layout note: DACVAE checkpoints follow the MLX convention — conv
// weights are stored `[Cout, K, Cin]` and tensors flow NLC. FFAI's
// `AudioMath` is PyTorch-native (`[Cout, Cin, K]`, NCL), so conv weights
// are transposed to PyTorch layout once at load time and all internal
// activations stay NCL.
//
// The optional audio-watermarking path (the `wm_model` subtree) is NOT
// ported — `DACVAE.encode` / `decode` exercise only the standard path.
//
// Layout: audio tensors are NCL `[batch, channels, length]`, f32.

import Foundation

/// DACVAE model configuration. Mirrors `config.json` of a DACVAE
/// checkpoint (SAM-Audio's 48 kHz codec).
public struct DACVAEConfig: Codable, Sendable {
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var latentDim: Int
    public var decoderDim: Int
    public var decoderRates: [Int]
    public var nCodebooks: Int
    public var codebookSize: Int
    public var codebookDim: Int
    public var sampleRate: Int

    enum CodingKeys: String, CodingKey {
        case encoderDim = "encoder_dim"
        case encoderRates = "encoder_rates"
        case latentDim = "latent_dim"
        case decoderDim = "decoder_dim"
        case decoderRates = "decoder_rates"
        case nCodebooks = "n_codebooks"
        case codebookSize = "codebook_size"
        case codebookDim = "codebook_dim"
        case sampleRate = "sample_rate"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        encoderDim = try c.decodeIfPresent(Int.self, forKey: .encoderDim) ?? 64
        encoderRates = try c.decodeIfPresent([Int].self, forKey: .encoderRates) ?? [2, 8, 10, 12]
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? 1024
        decoderDim = try c.decodeIfPresent(Int.self, forKey: .decoderDim) ?? 1536
        decoderRates = try c.decodeIfPresent([Int].self, forKey: .decoderRates) ?? [12, 10, 8, 2]
        nCodebooks = try c.decodeIfPresent(Int.self, forKey: .nCodebooks) ?? 16
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 1024
        codebookDim = try c.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 128
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 48_000
    }

    /// Hop length — total temporal downsampling of the encoder.
    public var hopLength: Int { encoderRates.reduce(1, *) }
}

public enum DACVAEError: Error, CustomStringConvertible {
    case missingWeights(String)
    case configNotFound(String)
    case shapeMismatch(String)

    public var description: String {
        switch self {
        case .missingWeights(let s): return "DACVAE: missing weights — \(s)"
        case .configNotFound(let s): return "DACVAE: config not found — \(s)"
        case .shapeMismatch(let s): return "DACVAE: shape mismatch — \(s)"
        }
    }
}

// MARK: - Weight loading helper

/// Loads DACVAE weights from a `SafeTensorsBundle`. DACVAE checkpoints
/// store conv weights in MLX `[Cout, K, Cin]` layout; this helper
/// transposes them to PyTorch `[Cout, Cin, K]` and reconstructs the
/// weight-normalized direction/magnitude pair so the rest of the codec
/// can reuse FFAI's `AudioMath` PyTorch-native conv kernels.
struct DACVAEWeights {
    let bundle: SafeTensorsBundle

    func floats(_ key: String) throws -> [Float] {
        AudioMath.floats(try bundle.tensor(named: key))
    }

    func shape(_ key: String) throws -> [Int] {
        try bundle.tensor(named: key).shape
    }

    func has(_ key: String) -> Bool { bundle.has(key) }

    /// Transpose an MLX `[Cout, K, Cin]` conv weight to PyTorch
    /// `[Cout, Cin, K]`.
    private static func mlxToPyTorch(_ w: [Float],
                                     mlxShape: [Int]) -> (data: [Float], shape: [Int]) {
        let (cOut, k, cIn) = (mlxShape[0], mlxShape[1], mlxShape[2])
        var out = [Float](repeating: 0, count: w.count)
        for o in 0..<cOut {
            for kk in 0..<k {
                for ci in 0..<cIn {
                    // src [o,kk,ci] -> dst [o,ci,kk]
                    out[(o * cIn + ci) * k + kk] = w[(o * k + kk) * cIn + ci]
                }
            }
        }
        return (out, [cOut, cIn, k])
    }

    /// Reconstruct a (possibly weight-normalized) conv at `prefix`,
    /// returning a PyTorch-layout `SNACWNConv1d`.
    func conv1d(prefix: String, stride: Int, padding: Int,
                dilation: Int = 1, groups: Int = 1) throws -> SNACWNConv1d {
        let gKey = "\(prefix).weight_g"
        let vKey = "\(prefix).weight_v"
        let plainKey = "\(prefix).weight"
        let mlxWeight: [Float]
        let mlxShape: [Int]
        if has(gKey), has(vKey) {
            // Weight-normalized: effective = g * v / ||v|| over dims 1,2.
            let g = try floats(gKey)
            let v = try floats(vKey)
            mlxShape = try shape(vKey)
            mlxWeight = WeightNorm.effectiveWeight(g: g, v: v, shape: mlxShape,
                                                   exceptDim: 0)
        } else if has(plainKey) {
            mlxWeight = try floats(plainKey)
            mlxShape = try shape(plainKey)
        } else {
            throw DACVAEError.missingWeights(prefix)
        }
        let (weight, ptShape) = DACVAEWeights.mlxToPyTorch(mlxWeight,
                                                           mlxShape: mlxShape)
        let bias = has("\(prefix).bias") ? try floats("\(prefix).bias") : nil
        return SNACWNConv1d(weight: weight, wShape: ptShape, bias: bias,
                            stride: stride, padding: padding,
                            dilation: dilation, groups: groups)
    }

    /// Transpose an MLX `[Cout, K, Cin]` convT weight to PyTorch
    /// `[Cin, Cout, K]` (PyTorch convTranspose1d layout).
    private static func mlxToPyTorchTransposed(_ w: [Float],
                                               mlxShape: [Int]) -> (data: [Float], shape: [Int]) {
        let (cOut, k, cIn) = (mlxShape[0], mlxShape[1], mlxShape[2])
        var out = [Float](repeating: 0, count: w.count)
        for o in 0..<cOut {
            for kk in 0..<k {
                for ci in 0..<cIn {
                    // src [o,kk,ci] -> dst [ci,o,kk]
                    out[(ci * cOut + o) * k + kk] = w[(o * k + kk) * cIn + ci]
                }
            }
        }
        return (out, [cIn, cOut, k])
    }

    /// Reconstruct a (possibly weight-normalized) transposed conv at
    /// `prefix`, returning a PyTorch-layout `SNACWNConvTranspose1d`.
    func convTranspose1d(prefix: String, stride: Int, padding: Int,
                         outputPadding: Int = 0,
                         groups: Int = 1) throws -> SNACWNConvTranspose1d {
        let gKey = "\(prefix).weight_g"
        let vKey = "\(prefix).weight_v"
        let plainKey = "\(prefix).weight"
        let mlxWeight: [Float]
        let mlxShape: [Int]
        if has(gKey), has(vKey) {
            let g = try floats(gKey)
            let v = try floats(vKey)
            mlxShape = try shape(vKey)
            mlxWeight = WeightNorm.effectiveWeight(g: g, v: v, shape: mlxShape,
                                                   exceptDim: 0)
        } else if has(plainKey) {
            mlxWeight = try floats(plainKey)
            mlxShape = try shape(plainKey)
        } else {
            throw DACVAEError.missingWeights(prefix)
        }
        let (weight, ptShape) =
            DACVAEWeights.mlxToPyTorchTransposed(mlxWeight, mlxShape: mlxShape)
        let bias = has("\(prefix).bias") ? try floats("\(prefix).bias") : nil
        return SNACWNConvTranspose1d(weight: weight, wShape: ptShape, bias: bias,
                                     stride: stride, padding: padding,
                                     outputPadding: outputPadding, groups: groups)
    }
}

// MARK: - Residual / encoder / decoder blocks

/// DACVAE residual unit — Snake → WNConv(k=7, dilated) → Snake →
/// WNConv(k=1), then a centre-cropped residual add. Reuses
/// `SNACResidualUnit` so the math path is shared with SNAC / DAC.
private func dacvaeResidualUnit(weights w: DACVAEWeights, prefix: String,
                                dilation: Int) throws -> SNACResidualUnit {
    // Padding for the k=7 dilated conv keeps DACVAE's `pad_mode="none"`
    // contract: pad = (kernel - stride) * dilation / 2.
    let pad = ((7 - 1) * dilation) / 2
    let alpha1 = try w.floats("\(prefix).act1.alpha")
    let conv1 = try w.conv1d(prefix: "\(prefix).conv1", stride: 1,
                             padding: pad, dilation: dilation)
    let alpha2 = try w.floats("\(prefix).act2.alpha")
    let conv2 = try w.conv1d(prefix: "\(prefix).conv2", stride: 1, padding: 0)
    return SNACResidualUnit(alpha1: alpha1, conv1: conv1,
                            alpha2: alpha2, conv2: conv2)
}

/// DACVAE encoder block — three dilated residual units, a Snake, then a
/// strided WNConv that downsamples by `stride`.
struct DACVAEEncoderBlock {
    let residuals: [SNACResidualUnit]
    let snakeAlpha: [Float]
    let convDown: SNACWNConv1d

    init(weights w: DACVAEWeights, prefix: String, stride: Int) throws {
        var res: [SNACResidualUnit] = []
        for (i, dil) in [1, 3, 9].enumerated() {
            res.append(try dacvaeResidualUnit(
                weights: w, prefix: "\(prefix).res\(i + 1)", dilation: dil))
        }
        self.residuals = res
        self.snakeAlpha = try w.floats("\(prefix).snake.alpha")
        let pad = Int(ceil(Double(stride) / 2.0))
        self.convDown = try w.conv1d(prefix: "\(prefix).conv",
                                     stride: stride, padding: pad)
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (d, s) = (x, shape)
        for r in residuals { (d, s) = r(d, shape: s) }
        d = AudioMath.snake(d, shape: s, alpha: snakeAlpha)
        return convDown(d, shape: s)
    }
}

/// DACVAE decoder block (standard, non-watermark path) — Snake →
/// WNConvTranspose1d (upsample) → three residual units (Snake,
/// dilations 1/3/9). Watermark sub-blocks (`block_2/3/6/7/10/11`) are
/// loaded by the reference but unused on the standard path; they are
/// not bound here.
struct DACVAEDecoderBlock {
    let snakeAlpha: [Float]
    let convUp: SNACWNConvTranspose1d
    let res1: SNACResidualUnit  // block_4, dilation 1
    let res2: SNACResidualUnit  // block_5, dilation 3
    let res3: SNACResidualUnit  // block_8, dilation 9

    init(weights w: DACVAEWeights, prefix: String, stride: Int) throws {
        self.snakeAlpha = try w.floats("\(prefix).block_0.alpha")
        // ConvTranspose with pad = (stride + 1) / 2 (DACVAE convT default).
        let pad = (stride + 1) / 2
        self.convUp = try w.convTranspose1d(prefix: "\(prefix).block_1",
                                            stride: stride, padding: pad)
        self.res1 = try dacvaeResidualUnit(
            weights: w, prefix: "\(prefix).block_4", dilation: 1)
        self.res2 = try dacvaeResidualUnit(
            weights: w, prefix: "\(prefix).block_5", dilation: 3)
        self.res3 = try dacvaeResidualUnit(
            weights: w, prefix: "\(prefix).block_8", dilation: 9)
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (d, s) = (x, shape)
        d = AudioMath.snake(d, shape: s, alpha: snakeAlpha)
        (d, s) = convUp(d, shape: s)
        (d, s) = res1(d, shape: s)
        (d, s) = res2(d, shape: s)
        (d, s) = res3(d, shape: s)
        return (d, s)
    }
}

// MARK: - DACVAE

/// DACVAE — a VAE-style Descript Audio Codec. The encoder maps a
/// waveform to a continuous latent (VAE mean in `codebookDim` space);
/// the decoder reconstructs the waveform from that latent.
public final class DACVAE: @unchecked Sendable {
    public let config: DACVAEConfig

    // Encoder.
    private let encoderConvIn: SNACWNConv1d
    private let encoderBlocks: [DACVAEEncoderBlock]
    private let encoderSnakeAlpha: [Float]
    private let encoderConvOut: SNACWNConv1d

    // VAE quantizer projections (1x1 convs).
    private let quantizerInProj: SNACWNConv1d   // latentDim -> 2*codebookDim
    private let quantizerOutProj: SNACWNConv1d  // codebookDim -> latentDim

    // Decoder.
    private let decoderConvIn: SNACWNConv1d
    private let decoderBlocks: [DACVAEDecoderBlock]
    private let decoderSnakeAlpha: [Float]
    private let decoderConvOut: SNACWNConv1d

    /// Load a DACVAE model from a Hugging Face snapshot directory
    /// containing `config.json` + `model.safetensors`.
    public static func fromPretrained(directory: URL) throws -> DACVAE {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DACVAEError.configNotFound(configURL.path)
        }
        let config = try JSONDecoder().decode(DACVAEConfig.self,
                                              from: Data(contentsOf: configURL))
        let bundle = try SafeTensorsBundle(directory: directory)
        return try DACVAE(config: config, bundle: bundle)
    }

    init(config: DACVAEConfig, bundle: SafeTensorsBundle) throws {
        self.config = config
        let w = DACVAEWeights(bundle: bundle)

        // ─── Encoder ──────────────────────────────────────────────
        // conv_in — WNConv1d(in=1, k=7, pad=3).
        self.encoderConvIn = try w.conv1d(prefix: "encoder.conv_in",
                                          stride: 1, padding: 3)
        var encBlocks: [DACVAEEncoderBlock] = []
        for (i, stride) in config.encoderRates.enumerated() {
            encBlocks.append(try DACVAEEncoderBlock(
                weights: w, prefix: "encoder.blocks.\(i)", stride: stride))
        }
        self.encoderBlocks = encBlocks
        self.encoderSnakeAlpha = try w.floats("encoder.snake_out.alpha")
        // conv_out — WNConv1d(out=latentDim, k=3, pad=1).
        self.encoderConvOut = try w.conv1d(prefix: "encoder.conv_out",
                                           stride: 1, padding: 1)

        // ─── Quantizer projections ────────────────────────────────
        // 1x1 convs; in_proj emits 2*codebookDim (VAE mean + logvar).
        self.quantizerInProj = try w.conv1d(prefix: "quantizer_in_proj",
                                            stride: 1, padding: 0)
        self.quantizerOutProj = try w.conv1d(prefix: "quantizer_out_proj",
                                             stride: 1, padding: 0)

        // ─── Decoder ──────────────────────────────────────────────
        // conv_in — WNConv1d(in=latentDim, k=7, pad=3).
        self.decoderConvIn = try w.conv1d(prefix: "decoder.conv_in",
                                          stride: 1, padding: 3)
        var decBlocks: [DACVAEDecoderBlock] = []
        for (i, stride) in config.decoderRates.enumerated() {
            decBlocks.append(try DACVAEDecoderBlock(
                weights: w, prefix: "decoder.blocks.\(i)", stride: stride))
        }
        self.decoderBlocks = decBlocks
        self.decoderSnakeAlpha = try w.floats("decoder.snake_out.alpha")
        // conv_out — WNConv1d(out=1, k=7, pad=3).
        self.decoderConvOut = try w.conv1d(prefix: "decoder.conv_out",
                                           stride: 1, padding: 3)
    }

    /// Audio sample rate of the codec.
    public var sampleRate: Int { config.sampleRate }

    /// Total temporal downsampling factor (samples per latent frame).
    public var hopLength: Int { config.hopLength }

    // ─── Padding ──────────────────────────────────────────────────

    /// Pad a waveform up to a multiple of `hopLength`.
    private func preprocess(_ data: [Float]) -> [Float] {
        let length = data.count
        let padded = Int(ceil(Double(length) / Double(hopLength))) * hopLength
        let rightPad = padded - length
        if rightPad == 0 { return data }
        return data + [Float](repeating: 0, count: rightPad)
    }

    // ─── Encode ───────────────────────────────────────────────────

    /// Encode a mono waveform into the continuous VAE latent
    /// `[1, codebookDim, T]` (the VAE `mean`).
    ///
    /// - waveform: an f32 Tensor of shape `[L]`, `[1, L]` or `[1, 1, L]`.
    public func encode(waveform: Tensor) throws -> Tensor {
        let raw = AudioMath.floats(waveform)
        let padded = preprocess(raw)
        var data = padded
        var shape = [1, 1, padded.count]

        // Encoder: conv stem + downsampling blocks + Snake + conv_out.
        var (d, s) = encoderConvIn(data, shape: shape)
        for block in encoderBlocks { (d, s) = block(d, shape: s) }
        d = AudioMath.snake(d, shape: s, alpha: encoderSnakeAlpha)
        (d, s) = encoderConvOut(d, shape: s)
        _ = (data, shape)

        // VAE in-projection emits [1, 2*codebookDim, T]; the first
        // `codebookDim` channels are the mean, the rest the log-variance.
        let (proj, projShape) = quantizerInProj(d, shape: s)
        let twoCb = projShape[1]
        let t = projShape[2]
        let cb = twoCb / 2
        var mean = [Float](repeating: 0, count: cb * t)
        for c in 0..<cb {
            let srcBase = c * t
            for i in 0..<t { mean[srcBase + i] = proj[srcBase + i] }
        }
        let out = Tensor.empty(shape: [1, cb, t], dtype: .f32)
        out.copyIn(from: mean)
        return out
    }

    // ─── Decode ───────────────────────────────────────────────────

    /// Decode a continuous latent `[1, codebookDim, T]` back into a
    /// waveform Tensor `[1, 1, L]`.
    public func decode(latents: Tensor) throws -> Tensor {
        let raw = AudioMath.floats(latents)
        guard latents.shape.count == 3 else {
            throw DACVAEError.shapeMismatch(
                "decode expects [1, codebookDim, T], got \(latents.shape)")
        }
        let s0 = latents.shape

        // Project codebookDim -> latentDim.
        let (emb, embShape) = quantizerOutProj(raw, shape: s0)

        // Decoder: conv stem + upsampling blocks + Snake + conv_out + tanh.
        var (d, s) = decoderConvIn(emb, shape: embShape)
        for block in decoderBlocks { (d, s) = block(d, shape: s) }
        d = AudioMath.snake(d, shape: s, alpha: decoderSnakeAlpha)
        (d, s) = decoderConvOut(d, shape: s)
        d = AudioMath.tanhAll(d)

        let out = Tensor.empty(shape: s, dtype: .f32)
        out.copyIn(from: d)
        return out
    }
}
