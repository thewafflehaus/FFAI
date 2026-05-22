// DescriptDAC — the Descript Audio Codec (DAC).
//
// Port of `mlx-audio-swift/Sources/MLXAudioCodecs/Descript`. DAC is a
// high-fidelity residual-VQ codec (the "44 kHz DAC" of descript-audio-
// codec). Structurally it is very close to SNAC — a Snake-activated
// convolutional encoder, a residual-VQ quantizer with per-codebook
// projection convs, and a mirrored decoder — but with a single temporal
// scale (no per-codebook striding) and L2-normalized codebook lookup.
//
// This is a genuine rewrite onto FFAI primitives — the MLX version is
// replaced by `AudioMath` CPU kernels operating on FFAI `Tensor`s, and
// the proven `WeightNorm` / weight-normalized conv path is shared with
// SNAC. The codec runs once per utterance, so the CPU path is
// acceptable (see AudioPrimitives.swift for the rationale).
//
// Public surface:
//   DescriptDAC.fromPretrained(directory:)   load from a HF snapshot
//   dac.encode(waveform:)                    -> [[Int32]]   code streams
//   dac.decode(codes:)                       -> Tensor      reconstructed
//
// Layout: audio tensors are NCL `[batch, channels, length]`, f32.

import Foundation

/// Descript DAC model configuration. Mirrors `config.json` of a DAC
/// checkpoint.
public struct DescriptDACConfig: Codable, Sendable {
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var latentDim: Int?
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
        encoderRates = try c.decodeIfPresent([Int].self, forKey: .encoderRates) ?? [2, 4, 5, 8]
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim)
        decoderDim = try c.decodeIfPresent(Int.self, forKey: .decoderDim) ?? 1536
        decoderRates = try c.decodeIfPresent([Int].self, forKey: .decoderRates) ?? [8, 5, 4, 2]
        nCodebooks = try c.decodeIfPresent(Int.self, forKey: .nCodebooks) ?? 12
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 1024
        codebookDim = try c.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 8
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16_000
    }

    /// Hop length — total temporal downsampling of the encoder.
    public var hopLength: Int { encoderRates.reduce(1, *) }

    /// Resolved latent (continuous-bottleneck) dimension.
    public var resolvedLatentDim: Int {
        latentDim ?? (encoderDim * (1 << encoderRates.count))
    }
}

public enum DescriptDACError: Error, CustomStringConvertible {
    case missingWeights(String)
    case configNotFound(String)
    case shapeMismatch(String)

    public var description: String {
        switch self {
        case .missingWeights(let s): return "DescriptDAC: missing weights — \(s)"
        case .configNotFound(let s): return "DescriptDAC: config not found — \(s)"
        case .shapeMismatch(let s): return "DescriptDAC: shape mismatch — \(s)"
        }
    }
}

// MARK: - Weight loading helper

/// Loads DAC weights from a `SafeTensorsBundle`, reconstructing
/// weight-normalized convs on the fly. DAC's `nn.Sequential` blocks are
/// stored under `.block.N` indices; this wrapper reconstructs each
/// `weight_g`/`weight_v` pair just like SNAC.
struct DescriptDACWeights {
    let bundle: SafeTensorsBundle

    func floats(_ key: String) throws -> [Float] {
        AudioMath.floats(try bundle.tensor(named: key))
    }

    func shape(_ key: String) throws -> [Int] {
        try bundle.tensor(named: key).shape
    }

    func has(_ key: String) -> Bool { bundle.has(key) }

    /// Reconstruct a weight-normalized conv at `prefix`. DAC stores conv
    /// weight as PyTorch `[Cout, Cin/groups, K]`; `weight_g` reduces over
    /// every axis except dim 0.
    func wnConv1d(prefix: String, stride: Int, padding: Int,
                  dilation: Int, groups: Int) throws -> SNACWNConv1d {
        let gKey = "\(prefix).weight_g"
        let vKey = "\(prefix).weight_v"
        guard has(gKey), has(vKey) else {
            throw DescriptDACError.missingWeights(prefix)
        }
        let g = try floats(gKey)
        let v = try floats(vKey)
        let vShape = try shape(vKey)
        let weight = WeightNorm.effectiveWeight(g: g, v: v, shape: vShape,
                                                exceptDim: 0)
        let bias = has("\(prefix).bias") ? try floats("\(prefix).bias") : nil
        return SNACWNConv1d(weight: weight, wShape: vShape, bias: bias,
                            stride: stride, padding: padding,
                            dilation: dilation, groups: groups)
    }

    /// Reconstruct a weight-normalized transposed conv at `prefix`.
    func wnConvTranspose1d(prefix: String, stride: Int, padding: Int,
                           outputPadding: Int, groups: Int) throws -> SNACWNConvTranspose1d {
        let gKey = "\(prefix).weight_g"
        let vKey = "\(prefix).weight_v"
        guard has(gKey), has(vKey) else {
            throw DescriptDACError.missingWeights(prefix)
        }
        let g = try floats(gKey)
        let v = try floats(vKey)
        let vShape = try shape(vKey)
        let weight = WeightNorm.effectiveWeight(g: g, v: v, shape: vShape,
                                                exceptDim: 0)
        let bias = has("\(prefix).bias") ? try floats("\(prefix).bias") : nil
        return SNACWNConvTranspose1d(weight: weight, wShape: vShape, bias: bias,
                                     stride: stride, padding: padding,
                                     outputPadding: outputPadding, groups: groups)
    }
}

// MARK: - Residual / encoder / decoder blocks

/// DAC residual unit — identical structure to SNAC's: Snake → WNConv
/// (k=7, dilated) → Snake → WNConv (k=1), then a centre-cropped residual
/// add. Reuses `SNACResidualUnit` so the math path is shared.
private func dacResidualUnit(weights w: DescriptDACWeights, prefix: String,
                             dilation: Int) throws -> SNACResidualUnit {
    let pad = ((7 - 1) * dilation) / 2
    // block.0 — Snake, block.1 — WNConv(k=7), block.2 — Snake,
    // block.3 — WNConv(k=1).
    let alpha1 = try w.floats("\(prefix).block.0.alpha")
    let conv1 = try w.wnConv1d(prefix: "\(prefix).block.1", stride: 1,
                               padding: pad, dilation: dilation, groups: 1)
    let alpha2 = try w.floats("\(prefix).block.2.alpha")
    let conv2 = try w.wnConv1d(prefix: "\(prefix).block.3", stride: 1,
                               padding: 0, dilation: 1, groups: 1)
    return SNACResidualUnit(alpha1: alpha1, conv1: conv1,
                            alpha2: alpha2, conv2: conv2)
}

/// DAC encoder block — three dilated residual units, a Snake, then a
/// strided WNConv that downsamples by `stride`.
struct DescriptDACEncoderBlock {
    let residuals: [SNACResidualUnit]
    let snakeAlpha: [Float]
    let convDown: SNACWNConv1d

    init(weights w: DescriptDACWeights, prefix: String, stride: Int) throws {
        // block.{0,1,2} — ResidualUnit(dilation 1,3,9).
        var res: [SNACResidualUnit] = []
        for (i, dil) in [1, 3, 9].enumerated() {
            res.append(try dacResidualUnit(
                weights: w, prefix: "\(prefix).block.\(i)", dilation: dil))
        }
        self.residuals = res
        // block.3 — Snake; block.4 — WNConv(k=2*stride, stride).
        self.snakeAlpha = try w.floats("\(prefix).block.3.alpha")
        let pad = Int(ceil(Double(stride) / 2.0))
        self.convDown = try w.wnConv1d(prefix: "\(prefix).block.4",
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

/// DAC decoder block — Snake → WNConvTranspose1d (upsample) → three
/// dilated residual units.
struct DescriptDACDecoderBlock {
    let snakeAlpha: [Float]
    let convUp: SNACWNConvTranspose1d
    let residuals: [SNACResidualUnit]

    init(weights w: DescriptDACWeights, prefix: String, stride: Int) throws {
        // block.0 — Snake; block.1 — WNConvT(k=2*stride, stride,
        // outputPadding=1).
        self.snakeAlpha = try w.floats("\(prefix).block.0.alpha")
        let pad = Int(ceil(Double(stride) / 2.0))
        self.convUp = try w.wnConvTranspose1d(
            prefix: "\(prefix).block.1", stride: stride, padding: pad,
            outputPadding: 1, groups: 1)
        // block.{2,3,4} — ResidualUnit(dilation 1,3,9).
        var res: [SNACResidualUnit] = []
        for (i, dil) in [1, 3, 9].enumerated() {
            res.append(try dacResidualUnit(
                weights: w, prefix: "\(prefix).block.\(i + 2)", dilation: dil))
        }
        self.residuals = res
    }

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (d, s) = (x, shape)
        d = AudioMath.snake(d, shape: s, alpha: snakeAlpha)
        (d, s) = convUp(d, shape: s)
        for r in residuals { (d, s) = r(d, shape: s) }
        return (d, s)
    }
}

// MARK: - Vector quantizer

/// A single DAC residual-VQ codebook. Projects the latent down
/// (`in_proj`), finds the nearest codebook entry by L2-normalized
/// distance, and projects back up (`out_proj`). DAC has a single
/// temporal scale, so there is no avg-pool / repeat-interleave.
struct DescriptDACVectorQuantize {
    let inProj: SNACWNConv1d        // [codebookDim, inputDim, 1]
    let outProj: SNACWNConv1d       // [inputDim, codebookDim, 1]
    let codebook: [Float]           // [codebookSize, codebookDim]
    let codebookSize: Int
    let codebookDim: Int

    init(weights w: DescriptDACWeights, prefix: String,
         codebookSize: Int, codebookDim: Int) throws {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        // Reference `sanitize` rewrites `in_proj` -> `inProj`; the
        // physical checkpoint keys are `in_proj` / `out_proj`.
        self.inProj = try w.wnConv1d(prefix: "\(prefix).in_proj",
                                     stride: 1, padding: 0,
                                     dilation: 1, groups: 1)
        self.outProj = try w.wnConv1d(prefix: "\(prefix).out_proj",
                                      stride: 1, padding: 0,
                                      dilation: 1, groups: 1)
        self.codebook = try w.floats("\(prefix).codebook.weight")
    }

    /// Nearest-codebook lookup. `latents` is NCL `[1, codebookDim, T]`.
    private func decodeLatents(_ latents: [Float],
                               shape: [Int]) -> (zQ: [Float], indices: [Int32]) {
        let (_, d, t) = (shape[0], shape[1], shape[2])
        // Rearrange b d t -> (b t) d.
        var enc = [Float](repeating: 0, count: t * d)
        for i in 0..<t {
            for c in 0..<d { enc[i * d + c] = latents[c * t + i] }
        }
        // L2-normalize both encodings and codebook, then nearest by
        // distance = 2 - 2·dot (monotone in -dot for unit vectors).
        let encN = AudioMath.l2NormalizeRows(enc, rows: t, dim: d)
        let cbN = AudioMath.l2NormalizeRows(codebook, rows: codebookSize, dim: d)
        var indices = [Int32](repeating: 0, count: t)
        for i in 0..<t {
            var best: Float = .greatestFiniteMagnitude
            var bestIdx = 0
            let eBase = i * d
            for ci in 0..<codebookSize {
                let cBase = ci * d
                var dot: Float = 0
                for c in 0..<d { dot += encN[eBase + c] * cbN[cBase + c] }
                let dist = 2.0 - 2.0 * dot
                if dist < best { best = dist; bestIdx = ci }
            }
            indices[i] = Int32(bestIdx)
        }
        // zQ = codebook[indices], rearranged back to [1, D, T].
        var zQ = [Float](repeating: 0, count: d * t)
        for i in 0..<t {
            let cBase = Int(indices[i]) * d
            for c in 0..<d { zQ[c * t + i] = codebook[cBase + c] }
        }
        return (zQ, indices)
    }

    /// Encode a residual latent: returns the upsampled quantized latent
    /// (for residual subtraction) and the codes.
    func encode(_ z: [Float],
                shape: [Int]) -> (zQ: [Float], indices: [Int32]) {
        // in_proj: [B, inputDim, T] -> [B, codebookDim, T].
        let (zE, zeShape) = inProj(z, shape: shape)
        let (zQLatent, indices) = decodeLatents(zE, shape: zeShape)
        // out_proj: [B, codebookDim, T] -> [B, inputDim, T].
        let (zQ, _) = outProj(zQLatent, shape: zeShape)
        return (zQ, indices)
    }

    /// Decode codes back into the projected latent `[1, inputDim, T]`.
    func decode(codes: [Int32]) -> (data: [Float], shape: [Int]) {
        let t = codes.count
        var zLatent = [Float](repeating: 0, count: codebookDim * t)
        for i in 0..<t {
            let cBase = Int(codes[i]) * codebookDim
            for c in 0..<codebookDim { zLatent[c * t + i] = codebook[cBase + c] }
        }
        return outProj(zLatent, shape: [1, codebookDim, t])
    }
}

// MARK: - DescriptDAC

/// Descript Audio Codec — Snake-conv encoder, residual-VQ quantizer,
/// mirrored decoder.
public final class DescriptDAC: @unchecked Sendable {
    public let config: DescriptDACConfig

    private let encoderConvIn: SNACWNConv1d
    private let encoderBlocks: [DescriptDACEncoderBlock]
    private let encoderSnakeAlpha: [Float]
    private let encoderConvOut: SNACWNConv1d

    private let quantizers: [DescriptDACVectorQuantize]

    private let decoderConvIn: SNACWNConv1d
    private let decoderBlocks: [DescriptDACDecoderBlock]
    private let decoderSnakeAlpha: [Float]
    private let decoderConvOut: SNACWNConv1d

    /// Load a DAC model from a Hugging Face snapshot directory containing
    /// `config.json` + `model.safetensors`.
    public static func fromPretrained(directory: URL) throws -> DescriptDAC {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DescriptDACError.configNotFound(configURL.path)
        }
        let config = try JSONDecoder().decode(DescriptDACConfig.self,
                                              from: Data(contentsOf: configURL))
        let bundle = try SafeTensorsBundle(directory: directory)
        return try DescriptDAC(config: config, bundle: bundle)
    }

    init(config: DescriptDACConfig, bundle: SafeTensorsBundle) throws {
        self.config = config
        let w = DescriptDACWeights(bundle: bundle)
        let latentDim = config.resolvedLatentDim

        // ─── Encoder ──────────────────────────────────────────────
        // block.0 — WNConv1d(in=1, k=7, pad=3).
        self.encoderConvIn = try w.wnConv1d(
            prefix: "encoder.block.0", stride: 1, padding: 3,
            dilation: 1, groups: 1)
        var encBlocks: [DescriptDACEncoderBlock] = []
        var blockIdx = 1
        for stride in config.encoderRates {
            encBlocks.append(try DescriptDACEncoderBlock(
                weights: w, prefix: "encoder.block.\(blockIdx)", stride: stride))
            blockIdx += 1
        }
        self.encoderBlocks = encBlocks
        // block.{N-1} — Snake; block.N — WNConv1d(k=3, pad=1).
        self.encoderSnakeAlpha = try w.floats("encoder.block.\(blockIdx).alpha")
        blockIdx += 1
        self.encoderConvOut = try w.wnConv1d(
            prefix: "encoder.block.\(blockIdx)", stride: 1, padding: 1,
            dilation: 1, groups: 1)

        // ─── Quantizer ────────────────────────────────────────────
        var qs: [DescriptDACVectorQuantize] = []
        for i in 0..<config.nCodebooks {
            qs.append(try DescriptDACVectorQuantize(
                weights: w, prefix: "quantizer.quantizers.\(i)",
                codebookSize: config.codebookSize,
                codebookDim: config.codebookDim))
        }
        self.quantizers = qs

        // ─── Decoder ──────────────────────────────────────────────
        // model.0 — WNConv1d(in=latentDim, k=7, pad=3).
        self.decoderConvIn = try w.wnConv1d(
            prefix: "decoder.model.0", stride: 1, padding: 3,
            dilation: 1, groups: 1)
        var decBlocks: [DescriptDACDecoderBlock] = []
        var decIdx = 1
        for stride in config.decoderRates {
            decBlocks.append(try DescriptDACDecoderBlock(
                weights: w, prefix: "decoder.model.\(decIdx)", stride: stride))
            decIdx += 1
        }
        self.decoderBlocks = decBlocks
        // model.{N-1} — Snake; model.N — WNConv1d(out=1, k=7, pad=3);
        // model.{N+1} — Tanh.
        self.decoderSnakeAlpha = try w.floats("decoder.model.\(decIdx).alpha")
        decIdx += 1
        self.decoderConvOut = try w.wnConv1d(
            prefix: "decoder.model.\(decIdx)", stride: 1, padding: 3,
            dilation: 1, groups: 1)
        _ = latentDim
    }

    /// Audio sample rate of the codec.
    public var sampleRate: Int { config.sampleRate }

    /// Total temporal downsampling factor (samples per code frame).
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

    /// Encode a mono waveform into `nCodebooks` integer code streams.
    ///
    /// - waveform: an f32 Tensor of shape `[L]`, `[1, L]` or `[1, 1, L]`.
    public func encode(waveform: Tensor) throws -> [[Int32]] {
        let raw = AudioMath.floats(waveform)
        let padded = preprocess(raw)
        var data = padded
        var shape = [1, 1, padded.count]
        let z = runEncoder(&data, shape: &shape)
        return quantize(z.data, shape: z.shape)
    }

    private func runEncoder(_ data: inout [Float],
                            shape: inout [Int]) -> (data: [Float], shape: [Int]) {
        var (d, s) = encoderConvIn(data, shape: shape)
        for block in encoderBlocks { (d, s) = block(d, shape: s) }
        d = AudioMath.snake(d, shape: s, alpha: encoderSnakeAlpha)
        return encoderConvOut(d, shape: s)
    }

    private func quantize(_ z: [Float], shape: [Int]) -> [[Int32]] {
        var residual = z
        var codes: [[Int32]] = []
        for q in quantizers {
            let (zQ, indices) = q.encode(residual, shape: shape)
            for i in 0..<residual.count { residual[i] -= zQ[i] }
            codes.append(indices)
        }
        return codes
    }

    // ─── Decode ───────────────────────────────────────────────────

    /// Decode integer code streams back into a waveform Tensor `[1,1,L]`.
    public func decode(codes: [[Int32]]) throws -> Tensor {
        // Reconstruct the quantized latent: sum of per-codebook
        // projections.
        var zQ: [Float] = []
        var zShape: [Int] = []
        for (i, q) in quantizers.enumerated() where i < codes.count {
            let (zQI, sI) = q.decode(codes: codes[i])
            if zQ.isEmpty {
                zQ = zQI
                zShape = sI
            } else {
                precondition(zQI.count == zQ.count,
                             "DescriptDAC.decode: quantizer latent length mismatch")
                for j in 0..<zQ.count { zQ[j] += zQI[j] }
            }
        }
        guard !zQ.isEmpty else {
            throw DescriptDACError.shapeMismatch("empty code list")
        }
        let (audio, audioShape) = runDecoder(zQ, shape: zShape)
        let out = Tensor.empty(shape: audioShape, dtype: .f32)
        out.copyIn(from: audio)
        return out
    }

    private func runDecoder(_ z: [Float],
                            shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (d, s) = decoderConvIn(z, shape: shape)
        for block in decoderBlocks { (d, s) = block(d, shape: s) }
        d = AudioMath.snake(d, shape: s, alpha: decoderSnakeAlpha)
        (d, s) = decoderConvOut(d, shape: s)
        d = AudioMath.tanhAll(d)
        return (d, s)
    }
}
