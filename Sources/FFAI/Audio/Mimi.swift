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
// Mimi — Kyutai's streaming neural audio codec (the Moshi codec).
//
// Port of `mlx-audio-swift/Sources/MLXAudioCodecs/Mimi`. Mimi pairs a
// SEANet convolutional encoder/decoder with an 8-layer Transformer in
// the latent bottleneck and a *split* residual-VQ quantizer (a single
// "semantic" codebook plus an "acoustic" residual stack). It is the
// codec behind Moshi-style TTS and several speech LMs.
//
// This is a genuine rewrite onto FFAI primitives — the MLX version is
// replaced by `AudioMath` CPU kernels operating on FFAI `Tensor`s. The
// reference is *streaming* (per-frame `step()` with conv/KV state); this
// port runs the **whole utterance in one pass** with a causal attention
// mask, which is functionally equivalent for offline encode/decode and
// far simpler. The codec runs once per utterance, so the CPU path is
// acceptable (see AudioPrimitives.swift for the rationale).
//
// Public surface:
//   Mimi.fromPretrained(directory:)   load weights from a HF snapshot
//   mimi.encode(waveform:)            -> [[Int32]]   nq code streams
//   mimi.decode(codes:)               -> Tensor      reconstructed audio
//
// Layout: audio tensors are NCL `[batch, channels, length]`, f32.

import Foundation

/// Mimi model configuration. The default values reproduce the
/// `mimi_202407` preset (24 kHz, 12.5 fps, 8-layer transformer) used by
/// every shipped Mimi/Moshi checkpoint, so a config file is optional.
public struct MimiConfig: Sendable {
    // ── Top level ──
    public var channels: Int
    public var sampleRate: Int
    public var frameRate: Double
    public var renormalize: Bool
    // ── SEANet ──
    public var seanetDim: Int
    public var nfilters: Int
    public var nresidualLayers: Int
    public var ratios: [Int]
    public var ksize: Int
    public var residualKsize: Int
    public var lastKsize: Int
    public var dilationBase: Int
    public var compress: Int
    public var trueSkip: Bool
    // ── Transformer ──
    public var dModel: Int
    public var numHeads: Int
    public var numLayers: Int
    public var dimFeedforward: Int
    public var context: Int
    public var maxPeriod: Int
    // ── Quantizer ──
    public var quantizerNQ: Int
    public var quantizerBins: Int
    public var quantizerDim: Int

    /// The standard `mimi_202407` preset.
    public static let mimi202407 = MimiConfig(
        channels: 1, sampleRate: 24_000, frameRate: 12.5, renormalize: true,
        seanetDim: 512, nfilters: 64, nresidualLayers: 1,
        ratios: [8, 6, 5, 4], ksize: 7, residualKsize: 3, lastKsize: 3,
        dilationBase: 2, compress: 2, trueSkip: true,
        dModel: 512, numHeads: 8, numLayers: 8, dimFeedforward: 2_048,
        context: 250, maxPeriod: 10_000,
        quantizerNQ: 32, quantizerBins: 2_048, quantizerDim: 256)

    /// Total temporal downsampling of the SEANet encoder.
    public var hopLength: Int { ratios.reduce(1, *) }

    /// Encoder frames-per-second before the downsample conv.
    public var encoderFPS: Double { Double(sampleRate) / Double(hopLength) }

    /// Stride of the latent downsample/upsample convs.
    public var downsampleStride: Int { Int(encoderFPS / frameRate) }

    public var headDim: Int { dModel / numHeads }
}

public enum MimiError: Error, CustomStringConvertible {
    case missingWeights(String)
    case shapeMismatch(String)

    public var description: String {
        switch self {
        case .missingWeights(let s): return "Mimi: missing weights — \(s)"
        case .shapeMismatch(let s): return "Mimi: shape mismatch — \(s)"
        }
    }
}

// MARK: - Weight loading helper

/// Loads Mimi weights from a `SafeTensorsBundle`. Mimi checkpoints store
/// conv weights in MLX NLC layout `[Cout, K, Cin]`; this wrapper
/// transposes them to PyTorch `[Cout, Cin, K]` on access. It also
/// applies the key-renaming the reference `sanitize()` performs so the
/// nested `encoder.N` / `decoder.N` `nn.Sequential` indices resolve.
struct MimiWeights {
    let bundle: SafeTensorsBundle
    /// raw-key → renamed-key map, built once at construction.
    private let renamed: [String: String]

    init(bundle: SafeTensorsBundle) {
        self.bundle = bundle
        var map: [String: String] = [:]
        for raw in bundle.index.keys {
            map[Self.sanitizeKey(raw)] = raw
        }
        self.renamed = map
    }

    /// Reproduce the reference `Mimi.sanitize` key rewriting so logical
    /// names like `encoder.layers.0.downsample.conv.weight` resolve.
    static func sanitizeKey(_ rawKey: String) -> String {
        var k = rawKey
            .split(separator: ".")
            .map { seg -> String in
                seg.hasPrefix("_") ? String(seg.dropFirst()) : String(seg)
            }
            .joined(separator: ".")
        if k.hasPrefix("encoder.model.") {
            k = k.replacingOccurrences(of: "encoder.model.", with: "encoder.")
        }
        if k.hasPrefix("decoder.model.") {
            k = k.replacingOccurrences(of: "decoder.model.", with: "decoder.")
        }
        if k.hasSuffix(".in_proj_weight") {
            k = k.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
        }
        let decIdx = [2, 5, 8, 11]
        for (layerIdx, decoderIdx) in decIdx.enumerated() {
            k = k.replacingOccurrences(
                of: "decoder.\(decoderIdx).",
                with: "decoder.layers.\(layerIdx).upsample.")
            k = k.replacingOccurrences(
                of: "decoder.\(decoderIdx + 1).",
                with: "decoder.layers.\(layerIdx).residuals.0.")
        }
        let encIdx = [1, 4, 7, 10]
        for (layerIdx, encoderIdx) in encIdx.enumerated() {
            k = k.replacingOccurrences(
                of: "encoder.\(encoderIdx).",
                with: "encoder.layers.\(layerIdx).residuals.0.")
            k = k.replacingOccurrences(
                of: "encoder.\(encoderIdx + 2).",
                with: "encoder.layers.\(layerIdx).downsample.")
        }
        k = k.replacingOccurrences(of: "decoder.0.", with: "decoder.init_conv1d.")
        k = k.replacingOccurrences(of: "decoder.14.", with: "decoder.final_conv1d.")
        k = k.replacingOccurrences(of: "encoder.0.", with: "encoder.init_conv1d.")
        k = k.replacingOccurrences(of: "encoder.14.", with: "encoder.final_conv1d.")
        k = k.replacingOccurrences(of: ".block.1.", with: ".block.0.")
        k = k.replacingOccurrences(of: ".block.3.", with: ".block.1.")
        return k
    }

    /// Physical key for a logical (sanitized) key.
    private func physical(_ key: String) -> String? { renamed[key] }

    func has(_ key: String) -> Bool { physical(key) != nil }

    func floats(_ key: String) throws -> [Float] {
        guard let p = physical(key) else { throw MimiError.missingWeights(key) }
        return AudioMath.floats(try bundle.tensor(named: p))
    }

    func shape(_ key: String) throws -> [Int] {
        guard let p = physical(key) else { throw MimiError.missingWeights(key) }
        return try bundle.tensor(named: p).shape
    }

    /// A conv weight, transposed from MLX NLC `[Cout, K, Cin]` to the
    /// PyTorch `[Cout, Cin, K]` layout `AudioMath.conv1d` expects.
    func convWeight(_ key: String) throws -> (data: [Float], shape: [Int]) {
        let raw = try floats(key)
        let s = try shape(key)
        let (cOut, k, cIn) = (s[0], s[1], s[2])
        var out = [Float](repeating: 0, count: raw.count)
        for o in 0..<cOut {
            for kk in 0..<k {
                for ic in 0..<cIn {
                    out[(o * cIn + ic) * k + kk] = raw[(o * k + kk) * cIn + ic]
                }
            }
        }
        return (out, [cOut, cIn, k])
    }
}

// MARK: - Mimi

/// Mimi neural audio codec — SEANet encoder + transformer + split
/// residual-VQ + transformer + SEANet decoder.
public final class Mimi: @unchecked Sendable {
    public let config: MimiConfig

    private let encoder: MimiSeanet
    private let decoder: MimiSeanet
    private let encoderTransformer: MimiProjectedTransformer
    private let decoderTransformer: MimiProjectedTransformer
    private let downsample: MimiConvResample
    private let upsample: MimiConvResample
    private let quantizer: MimiSplitRVQ

    /// Load a Mimi model from a Hugging Face snapshot directory holding a
    /// `*.safetensors` weights file. Mimi checkpoints carry no
    /// `config.json`; the `mimi_202407` preset is assumed (overridable).
    public static func fromPretrained(directory: URL,
                                      config: MimiConfig = .mimi202407) throws -> Mimi {
        let bundle = try SafeTensorsBundle(directory: directory)
        return try Mimi(config: config, bundle: bundle)
    }

    init(config: MimiConfig, bundle: SafeTensorsBundle) throws {
        self.config = config
        let w = MimiWeights(bundle: bundle)

        self.encoder = try MimiSeanet(weights: w, config: config,
                                      prefix: "encoder", isDecoder: false)
        self.decoder = try MimiSeanet(weights: w, config: config,
                                      prefix: "decoder", isDecoder: true)
        self.encoderTransformer = try MimiProjectedTransformer(
            weights: w, config: config, prefix: "encoder_transformer")
        self.decoderTransformer = try MimiProjectedTransformer(
            weights: w, config: config, prefix: "decoder_transformer")
        self.downsample = try MimiConvResample(
            weights: w, prefix: "downsample.conv", config: config,
            stride: config.downsampleStride, transposed: false)
        self.upsample = try MimiConvResample(
            weights: w, prefix: "upsample.convtr", config: config,
            stride: config.downsampleStride, transposed: true)
        self.quantizer = try MimiSplitRVQ(weights: w, config: config)
    }

    /// Audio sample rate of the codec.
    public var sampleRate: Int { config.sampleRate }

    /// Discrete code frame rate (frames per second).
    public var frameRate: Double { config.frameRate }

    // ─── Encode ───────────────────────────────────────────────────

    /// Encode a mono waveform into `nq` integer code streams.
    ///
    /// - waveform: an f32 Tensor of shape `[L]`, `[1, L]` or `[1, 1, L]`.
    /// - Returns: one `[Int32]` stream per VQ codebook (semantic first,
    ///   then the acoustic residual stack).
    public func encode(waveform: Tensor) throws -> [[Int32]] {
        let raw = AudioMath.floats(waveform)
        var data = raw
        var shape = [1, 1, raw.count]

        var z = encoder.forward(&data, shape: &shape)
        z = encoderTransformer.forward(z.data, shape: z.shape)
        z = downsample.forward(z.data, shape: z.shape)
        return try quantizer.encode(z.data, shape: z.shape)
    }

    // ─── Decode ───────────────────────────────────────────────────

    /// Decode `nq` integer code streams back into a waveform `[1, 1, L]`.
    public func decode(codes: [[Int32]]) throws -> Tensor {
        var z = try quantizer.decode(codes: codes)
        z = upsample.forward(z.data, shape: z.shape)
        z = decoderTransformer.forward(z.data, shape: z.shape)
        var data = z.data
        var shape = z.shape
        let out = decoder.forward(&data, shape: &shape)
        let t = Tensor.empty(shape: out.shape, dtype: .f32)
        t.copyIn(from: out.data)
        return t
    }
}
