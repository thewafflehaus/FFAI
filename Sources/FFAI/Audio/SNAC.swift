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
// SNAC — Multi-Scale Neural Audio Codec.
//
// Port of `mlx-audio-swift/Sources/MLXAudioCodecs/SNAC`. SNAC is the
// codec behind Orpheus-style TTS: it turns a waveform into a small set
// of integer code streams (one per residual-VQ codebook, each at a
// different temporal stride) and back.
//
// This is a genuine rewrite onto FFAI primitives — the MLX version is
// replaced by `AudioMath` CPU kernels operating on FFAI `Tensor`s. The
// codec runs once per utterance, so the CPU path is acceptable (see
// AudioPrimitives.swift for the rationale).
//
// Public surface:
//   SNAC.fromPretrained(directory:)   load weights from a HF snapshot
//   snac.encode(waveform:)            -> [[Int32]]   code streams
//   snac.decode(codes:)               -> Tensor      reconstructed audio
//
// Layout: audio tensors are NCL `[batch, channels, length]`, f32.

import Foundation

/// SNAC model configuration. Mirrors `config.json` of a SNAC checkpoint.
public struct SNACConfig: Codable, Sendable {
    public var samplingRate: Int
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var latentDim: Int?
    public var decoderDim: Int
    public var decoderRates: [Int]
    public var attnWindowSize: Int?
    public var codebookSize: Int
    public var codebookDim: Int
    public var vqStrides: [Int]
    public var noise: Bool
    public var depthwise: Bool

    enum CodingKeys: String, CodingKey {
        case samplingRate = "sampling_rate"
        case encoderDim = "encoder_dim"
        case encoderRates = "encoder_rates"
        case latentDim = "latent_dim"
        case decoderDim = "decoder_dim"
        case decoderRates = "decoder_rates"
        case attnWindowSize = "attn_window_size"
        case codebookSize = "codebook_size"
        case codebookDim = "codebook_dim"
        case vqStrides = "vq_strides"
        case noise
        case depthwise
    }

    /// Hop length — total temporal downsampling factor of the encoder.
    public var hopLength: Int { encoderRates.reduce(1, *) }
}

public enum SNACError: Error, CustomStringConvertible {
    case missingWeights(String)
    case configNotFound(String)
    case shapeMismatch(String)

    public var description: String {
        switch self {
        case .missingWeights(let s): return "SNAC: missing weights — \(s)"
        case .configNotFound(let s): return "SNAC: config not found — \(s)"
        case .shapeMismatch(let s): return "SNAC: shape mismatch — \(s)"
        }
    }
}

// MARK: - Weight-normalized conv layers

/// A weight-normalized 1-D conv. Holds the *effective* (already
/// reconstructed from `weight_g`/`weight_v`) weight as a flat array.
struct SNACWNConv1d {
    let weight: [Float]  // [Cout, Cin/groups, K]
    let wShape: [Int]
    let bias: [Float]?
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        AudioMath.conv1d(
            x: x, xShape: shape, weight: weight, wShape: wShape,
            bias: bias, stride: stride, padding: padding,
            dilation: dilation, groups: groups)
    }
}

/// A weight-normalized transposed 1-D conv (used for upsampling in the
/// decoder).
struct SNACWNConvTranspose1d {
    let weight: [Float]  // [Cin, Cout/groups, K]
    let wShape: [Int]
    let bias: [Float]?
    let stride: Int
    let padding: Int
    let outputPadding: Int
    let groups: Int

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        AudioMath.convTransposed1d(
            x: x, xShape: shape, weight: weight,
            wShape: wShape, bias: bias, stride: stride,
            padding: padding, dilation: 1,
            outputPadding: outputPadding, groups: groups)
    }
}

// MARK: - Weight loading helpers

/// Loads SNAC weights from a `SafeTensorsBundle`, reconstructing
/// weight-normalized convs on the fly.
struct SNACWeights {
    let bundle: SafeTensorsBundle

    /// Load an ordinary tensor as `[Float]`.
    func floats(_ key: String) throws -> [Float] {
        let t = try bundle.tensor(named: key)
        return AudioMath.floats(t)
    }

    func shape(_ key: String) throws -> [Int] {
        try bundle.tensor(named: key).shape
    }

    func has(_ key: String) -> Bool { bundle.has(key) }

    /// Reconstruct a weight-normalized conv at `prefix`.
    /// `transposed` selects which axis the magnitude `weight_g` reduces.
    func wnConv1d(
        prefix: String, stride: Int, padding: Int,
        dilation: Int, groups: Int
    ) throws -> SNACWNConv1d {
        let gKey = "\(prefix).weight_g"
        let vKey = "\(prefix).weight_v"
        guard has(gKey), has(vKey) else {
            throw SNACError.missingWeights(prefix)
        }
        let g = try floats(gKey)
        let v = try floats(vKey)
        let vShape = try shape(vKey)
        // PyTorch conv1d weight is [Cout, Cin/groups, K]; weight_norm
        // keeps the same shape for v. g broadcasts over dim 0.
        let weight = WeightNorm.effectiveWeight(g: g, v: v, shape: vShape, exceptDim: 0)
        let bias = has("\(prefix).bias") ? try floats("\(prefix).bias") : nil
        return SNACWNConv1d(
            weight: weight, wShape: vShape, bias: bias,
            stride: stride, padding: padding,
            dilation: dilation, groups: groups)
    }

    /// Reconstruct a weight-normalized transposed conv at `prefix`.
    func wnConvTranspose1d(
        prefix: String, stride: Int, padding: Int,
        outputPadding: Int, groups: Int
    ) throws -> SNACWNConvTranspose1d {
        let gKey = "\(prefix).weight_g"
        let vKey = "\(prefix).weight_v"
        guard has(gKey), has(vKey) else {
            throw SNACError.missingWeights(prefix)
        }
        let g = try floats(gKey)
        let v = try floats(vKey)
        let vShape = try shape(vKey)
        // convTranspose1d weight is [Cin, Cout/groups, K]; weight_norm
        // magnitude still reduces over all-but-dim-0.
        let weight = WeightNorm.effectiveWeight(g: g, v: v, shape: vShape, exceptDim: 0)
        let bias = has("\(prefix).bias") ? try floats("\(prefix).bias") : nil
        return SNACWNConvTranspose1d(
            weight: weight, wShape: vShape, bias: bias,
            stride: stride, padding: padding,
            outputPadding: outputPadding, groups: groups)
    }
}

// MARK: - Residual / Encoder / Decoder blocks

/// SNAC residual unit: Snake -> WNConv(k=7,dilated) -> Snake -> WNConv(k=1),
/// then residual add (with the input centre-cropped to match).
struct SNACResidualUnit {
    let alpha1: [Float]
    let conv1: SNACWNConv1d
    let alpha2: [Float]
    let conv2: SNACWNConv1d

    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        var (y, s) = (x, shape)
        y = AudioMath.snake(y, shape: s, alpha: alpha1)
        (y, s) = conv1(y, shape: s)
        y = AudioMath.snake(y, shape: s, alpha: alpha2)
        (y, s) = conv2(y, shape: s)
        // Residual: centre-crop x down to y's length.
        let pad = (shape[2] - s[2]) / 2
        let (n, c, lOut) = (s[0], s[1], s[2])
        var out = [Float](repeating: 0, count: y.count)
        for b in 0 ..< n {
            for ch in 0 ..< c {
                let xBase = (b * shape[1] + ch) * shape[2]
                let yBase = (b * c + ch) * lOut
                for t in 0 ..< lOut {
                    out[yBase + t] = y[yBase + t] + x[xBase + pad + t]
                }
            }
        }
        return (out, s)
    }
}

// MARK: - SNAC

/// SNAC neural audio codec — encoder, residual VQ quantizer, decoder.
///
/// The current port supports the *non-attention* SNAC variants (e.g.
/// the 24 kHz Orpheus codec, `attn_window_size == nil`). Local-attention
/// variants load but `encode`/`decode` will throw if attention layers
/// are present, so callers can fall back gracefully.
public final class SNAC: @unchecked Sendable {
    public let config: SNACConfig

    private let weights: SNACWeights

    // Encoder layout (matches nn.Sequential serialization):
    //   block.layers.0                  : WNConv1d in=1
    //   block.layers.{1..S}             : EncoderBlock per stride
    //   block.layers.{S+1}              : optional LocalMHA
    //   block.layers.last               : WNConv1d
    private let encoderConvIn: SNACWNConv1d
    private let encoderBlocks: [SNACEncoderBlock]
    private let encoderConvOut: SNACWNConv1d
    private let encoderHasAttention: Bool

    // Quantizer: one VQ per vq_stride.
    private let quantizers: [SNACVectorQuantize]

    // Decoder layout:
    //   model.layers.0 (.1 if depthwise) : input conv(s)
    //   optional LocalMHA
    //   model.layers.{...}               : DecoderBlock per rate
    //   model.layers.last-2              : Snake alpha
    //   model.layers.last-1              : WNConv1d out
    //   model.layers.last                : Tanh
    private let decoderConvsIn: [SNACWNConv1d]
    private let decoderBlocks: [SNACDecoderBlock]
    private let decoderSnakeAlpha: [Float]
    private let decoderConvOut: SNACWNConv1d
    private let decoderHasAttention: Bool

    /// Load a SNAC model from a Hugging Face snapshot directory containing
    /// `config.json` + `model.safetensors`.
    public static func fromPretrained(directory: URL) throws -> SNAC {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw SNACError.configNotFound(configURL.path)
        }
        let config = try JSONDecoder().decode(
            SNACConfig.self,
            from: Data(contentsOf: configURL))
        let bundle = try SafeTensorsBundle(directory: directory)
        return try SNAC(config: config, bundle: bundle)
    }

    init(config: SNACConfig, bundle: SafeTensorsBundle) throws {
        self.config = config
        let w = SNACWeights(bundle: bundle)
        self.weights = w

        // ─── Encoder ──────────────────────────────────────────────
        // layers.0 — WNConv1d(in=1, out=encoderDim, k=7, pad=3)
        self.encoderConvIn = try w.wnConv1d(
            prefix: "encoder.block.layers.0",
            stride: 1, padding: 3, dilation: 1, groups: 1)

        var encBlocks: [SNACEncoderBlock] = []
        var dModel = config.encoderDim
        var layerIdx = 1
        for stride in config.encoderRates {
            dModel *= 2
            let groups = config.depthwise ? dModel / 2 : 1
            let block = try SNACEncoderBlock(
                weights: w, prefix: "encoder.block.layers.\(layerIdx)",
                outputDim: dModel, stride: stride, groups: groups)
            encBlocks.append(block)
            layerIdx += 1
        }
        self.encoderBlocks = encBlocks

        // Optional LocalMHA occupies one layer index.
        let attnPresent = config.attnWindowSize != nil
        self.encoderHasAttention = attnPresent
        if attnPresent { layerIdx += 1 }

        // Final encoder WNConv1d.
        let encOutGroups = config.depthwise ? dModel : 1
        self.encoderConvOut = try w.wnConv1d(
            prefix: "encoder.block.layers.\(layerIdx)",
            stride: 1, padding: 3, dilation: 1, groups: encOutGroups)

        // ─── Quantizer ────────────────────────────────────────────
        var qs: [SNACVectorQuantize] = []
        for (i, stride) in config.vqStrides.enumerated() {
            let q = try SNACVectorQuantize(
                weights: w, prefix: "quantizer.quantizers.\(i)",
                codebookSize: config.codebookSize,
                codebookDim: config.codebookDim, stride: stride)
            qs.append(q)
        }
        self.quantizers = qs

        // ─── Decoder ──────────────────────────────────────────────
        let latentDim =
            config.latentDim
            ?? (config.encoderDim * (1 << config.encoderRates.count))
        var decConvsIn: [SNACWNConv1d] = []
        var decIdx = 0
        if config.depthwise {
            decConvsIn.append(
                try w.wnConv1d(
                    prefix: "decoder.model.layers.0",
                    stride: 1, padding: 3, dilation: 1, groups: latentDim))
            decConvsIn.append(
                try w.wnConv1d(
                    prefix: "decoder.model.layers.1",
                    stride: 1, padding: 0, dilation: 1, groups: 1))
            decIdx = 2
        } else {
            decConvsIn.append(
                try w.wnConv1d(
                    prefix: "decoder.model.layers.0",
                    stride: 1, padding: 3, dilation: 1, groups: 1))
            decIdx = 1
        }
        self.decoderConvsIn = decConvsIn

        self.decoderHasAttention = attnPresent
        if attnPresent { decIdx += 1 }

        var decBlocks: [SNACDecoderBlock] = []
        let channels = config.decoderDim
        for (i, stride) in config.decoderRates.enumerated() {
            let inputDim = channels / (1 << i)
            let outputDim = channels / (1 << (i + 1))
            let groups = config.depthwise ? outputDim : 1
            let block = try SNACDecoderBlock(
                weights: w, prefix: "decoder.model.layers.\(decIdx)",
                inputDim: inputDim, outputDim: outputDim, stride: stride,
                noise: config.noise, groups: groups)
            decBlocks.append(block)
            decIdx += 1
        }
        self.decoderBlocks = decBlocks

        // Tail: Snake alpha + WNConv1d(out=1) + Tanh.
        // alpha ships as [1, C, 1]; the flat read is already length C.
        self.decoderSnakeAlpha = try w.floats("decoder.model.layers.\(decIdx).alpha")
        decIdx += 1
        self.decoderConvOut = try w.wnConv1d(
            prefix: "decoder.model.layers.\(decIdx)",
            stride: 1, padding: 3, dilation: 1, groups: 1)
    }

    /// Total temporal downsampling factor (samples per top-level code).
    public var hopLength: Int { config.hopLength }

    /// Audio sample rate of the codec.
    public var sampleRate: Int { config.samplingRate }

    // ─── Padding ──────────────────────────────────────────────────

    /// Pad a waveform up to a multiple compatible with the encoder
    /// strides and (if present) attention window. Returns the padded
    /// signal plus the original length so `decode` can trim back.
    private func preprocess(_ data: [Float], length: Int) -> [Float] {
        var lcmValue = config.vqStrides[0]
        for s in config.vqStrides.dropFirst() { lcmValue = lcm(lcmValue, s) }
        if let w = config.attnWindowSize { lcmValue = lcm(lcmValue, w) }
        let padTo = config.hopLength * lcmValue
        let rightPad = Int(ceil(Double(length) / Double(padTo))) * padTo - length
        if rightPad == 0 { return data }
        return data + [Float](repeating: 0, count: rightPad)
    }

    // ─── Encode ───────────────────────────────────────────────────

    /// Encode a mono waveform into integer code streams.
    ///
    /// - waveform: an f32 Tensor of shape `[L]`, `[1, L]` or `[1, 1, L]`.
    /// - Returns: one `[Int32]` array per residual-VQ codebook. Stream 0
    ///   has the most frames; deeper streams are temporally strided.
    public func encode(waveform: Tensor) throws -> [[Int32]] {
        guard !encoderHasAttention else {
            throw SNACError.shapeMismatch(
                "attention-variant SNAC encode not yet supported")
        }
        let raw = AudioMath.floats(waveform)
        let length = raw.count
        let padded = preprocess(raw, length: length)
        var data = padded
        var shape = [1, 1, padded.count]

        let z = runEncoder(&data, shape: &shape)
        let codes = try quantize(z.data, shape: z.shape)
        return codes
    }

    /// Encode and return the continuous latent as well (useful for tests).
    func encodeLatent(waveform: Tensor) throws -> (data: [Float], shape: [Int]) {
        let raw = AudioMath.floats(waveform)
        let padded = preprocess(raw, length: raw.count)
        var data = padded
        var shape = [1, 1, padded.count]
        return runEncoder(&data, shape: &shape)
    }

    private func runEncoder(
        _ data: inout [Float],
        shape: inout [Int]
    ) -> (data: [Float], shape: [Int]) {
        var (d, s) = encoderConvIn(data, shape: shape)
        for block in encoderBlocks {
            (d, s) = block(d, shape: s)
        }
        (d, s) = encoderConvOut(d, shape: s)
        return (d, s)
    }

    private func quantize(_ z: [Float], shape: [Int]) throws -> [[Int32]] {
        var residual = z
        let residualShape = shape
        var codes: [[Int32]] = []
        for q in quantizers {
            let (zQ, _, indices) = try q.encode(residual, shape: residualShape)
            // residual = residual - zQ
            for i in 0 ..< residual.count { residual[i] -= zQ[i] }
            codes.append(indices)
        }
        return codes
    }

    // ─── Decode ───────────────────────────────────────────────────

    /// Decode integer code streams back into a waveform Tensor `[1, 1, L]`.
    public func decode(codes: [[Int32]]) throws -> Tensor {
        guard !decoderHasAttention else {
            throw SNACError.shapeMismatch(
                "attention-variant SNAC decode not yet supported")
        }
        // Reconstruct the quantized latent from codes.
        var zQ: [Float] = []
        var zShape: [Int] = []
        for (i, q) in quantizers.enumerated() {
            let (zQI, sI) = try q.decode(codes: codes[i])
            if zQ.isEmpty {
                zQ = zQI
                zShape = sI
            } else {
                precondition(
                    zQI.count == zQ.count,
                    "SNAC.decode: quantizer latent length mismatch")
                for j in 0 ..< zQ.count { zQ[j] += zQI[j] }
            }
        }
        let (audio, audioShape) = runDecoder(zQ, shape: zShape)
        let out = Tensor.empty(shape: audioShape, dtype: .f32)
        out.copyIn(from: audio)
        return out
    }

    private func runDecoder(
        _ z: [Float],
        shape: [Int]
    ) -> (data: [Float], shape: [Int]) {
        var (d, s) = (z, shape)
        for conv in decoderConvsIn {
            (d, s) = conv(d, shape: s)
        }
        for block in decoderBlocks {
            (d, s) = block(d, shape: s)
        }
        d = AudioMath.snake(d, shape: s, alpha: decoderSnakeAlpha)
        (d, s) = decoderConvOut(d, shape: s)
        d = AudioMath.tanhAll(d)
        return (d, s)
    }
}

// MARK: - Small math helpers

func gcd(_ a: Int, _ b: Int) -> Int {
    var (a, b) = (a, b)
    while b != 0 { (a, b) = (b, a % b) }
    return abs(a)
}

func lcm(_ a: Int, _ b: Int) -> Int {
    a == 0 || b == 0 ? 0 : abs(a / gcd(a, b) * b)
}
