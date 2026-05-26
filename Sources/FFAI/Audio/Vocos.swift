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
// Vocos — a ConvNeXt + ISTFT neural vocoder.
//
// Port of `mlx-audio-swift/Sources/MLXAudioCodecs/Vocos`. Vocos is a
// *decode-only* codec: it turns a feature sequence (a mel-spectrogram,
// or summed EnCodec codebook embeddings) into a waveform. A ConvNeXt
// backbone refines the features, an ISTFT head predicts a complex STFT,
// and inverse-STFT overlap-add reconstructs the audio. There is no
// encoder or quantizer — Vocos is the waveform tail of a TTS stack.
//
// This is a genuine rewrite onto FFAI primitives: the ConvNeXt backbone
// runs on `AudioMath` CPU kernels, and the ISTFT head reuses the fused
// GPU `Ops.vocoderISTFT` kernel for the inverse-STFT overlap-add. The
// vocoder runs once per utterance, so the CPU backbone path is
// acceptable (see AudioPrimitives.swift for the rationale).
//
// Public surface:
//   Vocos.fromPretrained(directory:)   load weights from a HF snapshot
//   vocos.decode(features:)            -> Tensor   reconstructed audio
//
// Layout: feature tensors are NCL `[batch, channels, length]`, f32.

import Foundation

/// Vocos model configuration. Decoded from the `backbone` / `head`
/// sections of a Vocos `config.json`. Vocos checkpoints vary in how the
/// config is nested; the decoder below accepts both a flat config and
/// the common nested `{ feature_extractor, backbone, head }` shape.
public struct VocosConfig: Sendable {
    // ── Backbone ──
    public var inputChannels: Int
    public var dim: Int
    public var intermediateDim: Int
    public var numLayers: Int
    public var adanormNumEmbeddings: Int?
    // ── ISTFT head ──
    public var nFFT: Int
    public var hopLength: Int

    /// `true` when the backbone uses adaptive (bandwidth-conditioned)
    /// layer norm — the EnCodec-feature Vocos variant.
    public var useAdaNorm: Bool { adanormNumEmbeddings != nil }
}

extension VocosConfig: Decodable {
    enum TopKeys: String, CodingKey {
        case backbone, head, featureExtractor = "feature_extractor"
        // flat fallbacks
        case inputChannels = "input_channels"
        case dim
        case intermediateDim = "intermediate_dim"
        case numLayers = "num_layers"
        case adanormNumEmbeddings = "adanorm_num_embeddings"
        case nFFT = "n_fft"
        case hopLength = "hop_length"
    }
    enum InitArgsKeys: String, CodingKey {
        case inputChannels = "input_channels"
        case dim
        case intermediateDim = "intermediate_dim"
        case numLayers = "num_layers"
        case adanormNumEmbeddings = "adanorm_num_embeddings"
        case nFFT = "n_fft"
        case hopLength = "hop_length"
    }
    enum SectionKeys: String, CodingKey {
        case initArgs = "init_args"
    }

    public init(from decoder: Swift.Decoder) throws {
        let top = try decoder.container(keyedBy: TopKeys.self)

        // Nested `backbone.init_args` / `head.init_args` (the common
        // Vocos checkpoint layout).
        if top.contains(.backbone) {
            let bSection = try top.nestedContainer(
                keyedBy: SectionKeys.self, forKey: .backbone)
            let b = try bSection.nestedContainer(
                keyedBy: InitArgsKeys.self, forKey: .initArgs)
            inputChannels = try b.decodeIfPresent(Int.self, forKey: .inputChannels) ?? 100
            dim = try b.decodeIfPresent(Int.self, forKey: .dim) ?? 512
            intermediateDim = try b.decodeIfPresent(Int.self, forKey: .intermediateDim) ?? 1536
            numLayers = try b.decodeIfPresent(Int.self, forKey: .numLayers) ?? 8
            adanormNumEmbeddings = try b.decodeIfPresent(Int.self, forKey: .adanormNumEmbeddings)

            let hSection = try top.nestedContainer(
                keyedBy: SectionKeys.self, forKey: .head)
            let h = try hSection.nestedContainer(
                keyedBy: InitArgsKeys.self, forKey: .initArgs)
            nFFT = try h.decodeIfPresent(Int.self, forKey: .nFFT) ?? 1024
            hopLength = try h.decodeIfPresent(Int.self, forKey: .hopLength) ?? 256
        } else {
            // Flat config.
            inputChannels = try top.decodeIfPresent(Int.self, forKey: .inputChannels) ?? 100
            dim = try top.decodeIfPresent(Int.self, forKey: .dim) ?? 512
            intermediateDim = try top.decodeIfPresent(Int.self, forKey: .intermediateDim) ?? 1536
            numLayers = try top.decodeIfPresent(Int.self, forKey: .numLayers) ?? 8
            adanormNumEmbeddings = try top.decodeIfPresent(Int.self, forKey: .adanormNumEmbeddings)
            nFFT = try top.decodeIfPresent(Int.self, forKey: .nFFT) ?? 1024
            hopLength = try top.decodeIfPresent(Int.self, forKey: .hopLength) ?? 256
        }
    }
}

public enum VocosError: Error, CustomStringConvertible {
    case missingWeights(String)
    case configNotFound(String)
    case unsupported(String)

    public var description: String {
        switch self {
        case .missingWeights(let s): return "Vocos: missing weights — \(s)"
        case .configNotFound(let s): return "Vocos: config not found — \(s)"
        case .unsupported(let s): return "Vocos: unsupported — \(s)"
        }
    }
}

// MARK: - Weight loading helper

/// Loads Vocos weights from a `SafeTensorsBundle`. Vocos checkpoints
/// store conv weights in MLX NLC layout `[Cout, K, Cin]`; this wrapper
/// transposes conv weights to PyTorch `[Cout, Cin, K]` on access.
struct VocosWeights {
    let bundle: SafeTensorsBundle

    func floats(_ key: String) throws -> [Float] {
        AudioMath.floats(try bundle.tensor(named: key))
    }

    func shape(_ key: String) throws -> [Int] {
        try bundle.tensor(named: key).shape
    }

    func has(_ key: String) -> Bool { bundle.has(key) }

    /// A conv weight transposed from MLX NLC `[Cout, K, Cin]` to the
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

// MARK: - Vocos

/// Vocos vocoder — a ConvNeXt backbone plus an ISTFT head.
public final class Vocos: @unchecked Sendable {
    public let config: VocosConfig

    private let backbone: VocosBackbone
    private let head: VocosISTFTHead

    /// Load a Vocos model from a Hugging Face snapshot directory
    /// containing `config.json` + a `*.safetensors` weights file.
    public static func fromPretrained(directory: URL) throws -> Vocos {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw VocosError.configNotFound(configURL.path)
        }
        let config = try JSONDecoder().decode(VocosConfig.self,
                                              from: Data(contentsOf: configURL))
        let bundle = try SafeTensorsBundle(directory: directory)
        return try Vocos(config: config, bundle: bundle)
    }

    init(config: VocosConfig, bundle: SafeTensorsBundle) throws {
        self.config = config
        guard !config.useAdaNorm else {
            throw VocosError.unsupported(
                "bandwidth-conditioned (AdaLayerNorm) Vocos not yet supported")
        }
        let w = VocosWeights(bundle: bundle)
        self.backbone = try VocosBackbone(weights: w, config: config)
        self.head = try VocosISTFTHead(weights: w, config: config)
    }

    /// ISTFT FFT length.
    public var nFFT: Int { config.nFFT }

    /// ISTFT hop length.
    public var hopLength: Int { config.hopLength }

    /// Number of input feature channels (e.g. mel bins).
    public var featureChannels: Int { config.inputChannels }

    // ─── Decode ───────────────────────────────────────────────────

    /// Decode a feature sequence into a waveform Tensor `[L]`.
    ///
    /// - features: an f32 Tensor of shape `[inputChannels, T]` or
    ///   `[1, inputChannels, T]` (e.g. a mel-spectrogram).
    public func decode(features: Tensor) throws -> Tensor {
        let raw = AudioMath.floats(features)
        let shape: [Int]
        switch features.shape.count {
        case 2: shape = [1, features.shape[0], features.shape[1]]
        case 3: shape = features.shape
        default:
            throw VocosError.unsupported(
                "decode expects features of rank 2 or 3, got \(features.shape)")
        }
        // ConvNeXt backbone — refines the [1, dim, T] feature map.
        let h = backbone.forward(raw, shape: shape)
        // ISTFT head — predict the complex STFT and reconstruct audio.
        return head.synthesize(h.data, shape: h.shape)
    }
}
