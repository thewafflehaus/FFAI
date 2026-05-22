// Encodec — Meta's neural audio codec.
//
// Port of `mlx-audio-swift/Sources/MLXAudioCodecs/Encodec`. EnCodec is
// the codec behind several TTS stacks (e.g. Bark, MusicGen): a SEANet
// encoder downsamples a waveform to a latent, a residual-VQ quantizer
// turns it into integer code streams, and a SEANet decoder reconstructs
// the audio. Two LSTM layers sit in the latent bottleneck of both the
// encoder and the decoder.
//
// This is a genuine rewrite onto FFAI primitives — the MLX version is
// replaced by `AudioMath` CPU kernels operating on FFAI `Tensor`s. The
// codec runs once per utterance, so the CPU path is acceptable (see
// AudioPrimitives.swift for the rationale).
//
// Public surface:
//   Encodec.fromPretrained(directory:)   load weights from a HF snapshot
//   encodec.encode(waveform:)            -> [[Int32]]   code streams
//   encodec.decode(codes:)               -> Tensor      reconstructed audio
//
// Layout: audio tensors are NCL `[batch, channels, length]`, f32. Note
// the reference MLX code uses NLC; this port stays NCL throughout to
// match `AudioMath`'s convolution layout.
//
// Scope: the single-frame (no-chunking) path — `chunk_length_s == nil`
// — which covers the standard 24 kHz and 48 kHz EnCodec checkpoints
// used for TTS. Normalization (`config.normalize`) is supported.

import Foundation

/// Encodec model configuration. Mirrors `config.json` of an EnCodec
/// checkpoint. Fields default to the 24 kHz EnCodec preset so a partial
/// config still decodes.
public struct EncodecConfig: Codable, Sendable {
    public var audioChannels: Int
    public var numFilters: Int
    public var kernelSize: Int
    public var numResidualLayers: Int
    public var dilationGrowthRate: Int
    public var codebookSize: Int
    public var codebookDim: Int
    public var hiddenSize: Int
    public var numLstmLayers: Int
    public var residualKernelSize: Int
    public var useCausalConv: Bool
    public var normalize: Bool
    public var padMode: String
    public var normType: String
    public var lastKernelSize: Int
    public var trimRightRatio: Float
    public var compress: Int
    public var upsamplingRatios: [Int]
    public var targetBandwidths: [Float]
    public var samplingRate: Int
    public var chunkLengthS: Float?
    public var overlap: Float?
    public var useConvShortcut: Bool

    enum CodingKeys: String, CodingKey {
        case audioChannels = "audio_channels"
        case numFilters = "num_filters"
        case kernelSize = "kernel_size"
        case numResidualLayers = "num_residual_layers"
        case dilationGrowthRate = "dilation_growth_rate"
        case codebookSize = "codebook_size"
        case codebookDim = "codebook_dim"
        case hiddenSize = "hidden_size"
        case numLstmLayers = "num_lstm_layers"
        case residualKernelSize = "residual_kernel_size"
        case useCausalConv = "use_causal_conv"
        case normalize
        case padMode = "pad_mode"
        case normType = "norm_type"
        case lastKernelSize = "last_kernel_size"
        case trimRightRatio = "trim_right_ratio"
        case compress
        case upsamplingRatios = "upsampling_ratios"
        case targetBandwidths = "target_bandwidths"
        case samplingRate = "sampling_rate"
        case chunkLengthS = "chunk_length_s"
        case overlap
        case useConvShortcut = "use_conv_shortcut"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        audioChannels = try c.decodeIfPresent(Int.self, forKey: .audioChannels) ?? 1
        numFilters = try c.decodeIfPresent(Int.self, forKey: .numFilters) ?? 32
        kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 7
        numResidualLayers = try c.decodeIfPresent(Int.self, forKey: .numResidualLayers) ?? 1
        dilationGrowthRate = try c.decodeIfPresent(Int.self, forKey: .dilationGrowthRate) ?? 2
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 1024
        codebookDim = try c.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 128
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 128
        numLstmLayers = try c.decodeIfPresent(Int.self, forKey: .numLstmLayers) ?? 2
        residualKernelSize = try c.decodeIfPresent(Int.self, forKey: .residualKernelSize) ?? 3
        useCausalConv = try c.decodeIfPresent(Bool.self, forKey: .useCausalConv) ?? true
        normalize = try c.decodeIfPresent(Bool.self, forKey: .normalize) ?? false
        padMode = try c.decodeIfPresent(String.self, forKey: .padMode) ?? "reflect"
        normType = try c.decodeIfPresent(String.self, forKey: .normType) ?? "weight_norm"
        lastKernelSize = try c.decodeIfPresent(Int.self, forKey: .lastKernelSize) ?? 7
        trimRightRatio = try c.decodeIfPresent(Float.self, forKey: .trimRightRatio) ?? 1.0
        compress = try c.decodeIfPresent(Int.self, forKey: .compress) ?? 2
        upsamplingRatios = try c.decodeIfPresent([Int].self, forKey: .upsamplingRatios) ?? [8, 5, 4, 2]
        targetBandwidths = try c.decodeIfPresent([Float].self, forKey: .targetBandwidths)
            ?? [1.5, 3.0, 6.0, 12.0, 24.0]
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 24000
        chunkLengthS = try c.decodeIfPresent(Float.self, forKey: .chunkLengthS)
        overlap = try c.decodeIfPresent(Float.self, forKey: .overlap)
        useConvShortcut = try c.decodeIfPresent(Bool.self, forKey: .useConvShortcut) ?? true
    }

    /// Hop length — total temporal downsampling of the encoder.
    public var hopLength: Int { upsamplingRatios.reduce(1, *) }

    /// Frames per second of the discrete code streams.
    public var frameRate: Int {
        Int(ceil(Double(samplingRate) / Double(hopLength)))
    }
}

public enum EncodecError: Error, CustomStringConvertible {
    case missingWeights(String)
    case configNotFound(String)
    case shapeMismatch(String)
    case unsupported(String)

    public var description: String {
        switch self {
        case .missingWeights(let s): return "Encodec: missing weights — \(s)"
        case .configNotFound(let s): return "Encodec: config not found — \(s)"
        case .shapeMismatch(let s): return "Encodec: shape mismatch — \(s)"
        case .unsupported(let s): return "Encodec: unsupported — \(s)"
        }
    }
}

// MARK: - Weight loading helper

/// Loads EnCodec weights from a `SafeTensorsBundle`. EnCodec does not use
/// weight-norm in its serialized checkpoint — convs ship plain
/// `weight`/`bias` tensors — so this is a thin typed wrapper around the
/// bundle.
struct EncodecWeights {
    let bundle: SafeTensorsBundle

    func floats(_ key: String) throws -> [Float] {
        let t = try bundle.tensor(named: key)
        return AudioMath.floats(t)
    }

    func shape(_ key: String) throws -> [Int] {
        try bundle.tensor(named: key).shape
    }

    func has(_ key: String) -> Bool { bundle.has(key) }
}

// MARK: - Encodec

/// EnCodec neural audio codec — SEANet encoder, residual-VQ quantizer,
/// SEANet decoder.
public final class Encodec: @unchecked Sendable {
    public let config: EncodecConfig

    private let encoder: EncodecSEANet
    private let decoder: EncodecSEANet
    private let quantizer: EncodecResidualVQ

    /// Load an EnCodec model from a Hugging Face snapshot directory
    /// containing `config.json` + `model.safetensors`.
    public static func fromPretrained(directory: URL) throws -> Encodec {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw EncodecError.configNotFound(configURL.path)
        }
        let config = try JSONDecoder().decode(EncodecConfig.self,
                                              from: Data(contentsOf: configURL))
        let bundle = try SafeTensorsBundle(directory: directory)
        return try Encodec(config: config, bundle: bundle)
    }

    init(config: EncodecConfig, bundle: SafeTensorsBundle) throws {
        self.config = config
        guard config.chunkLengthS == nil else {
            throw EncodecError.unsupported(
                "chunked EnCodec (chunk_length_s != nil) not yet supported")
        }
        let w = EncodecWeights(bundle: bundle)
        self.encoder = try EncodecSEANet(weights: w, config: config,
                                         prefix: "encoder", isDecoder: false)
        self.decoder = try EncodecSEANet(weights: w, config: config,
                                         prefix: "decoder", isDecoder: true)
        self.quantizer = try EncodecResidualVQ(weights: w, config: config)
    }

    /// Audio sample rate of the codec.
    public var sampleRate: Int { config.samplingRate }

    /// Total temporal downsampling factor (samples per top-level code).
    public var hopLength: Int { config.hopLength }

    // ─── Encode ───────────────────────────────────────────────────

    /// Encode a mono waveform into integer code streams (one per VQ
    /// codebook). Uses the highest target bandwidth by default.
    ///
    /// - waveform: an f32 Tensor of shape `[L]`, `[1, L]` or `[1, 1, L]`.
    /// - bandwidth: target kbps; defaults to `targetBandwidths.first`.
    /// - Returns: `codes` — one `[Int32]` stream per active codebook —
    ///   and `scale` (nil unless `config.normalize`).
    public func encode(waveform: Tensor,
                       bandwidth: Float? = nil) throws -> [[Int32]] {
        try encodeWithScale(waveform: waveform, bandwidth: bandwidth).codes
    }

    /// Encode and also return the per-utterance normalization scale so a
    /// later `decode` can undo it.
    public func encodeWithScale(
        waveform: Tensor, bandwidth: Float? = nil
    ) throws -> (codes: [[Int32]], scale: Float?) {
        let bw = bandwidth ?? (config.targetBandwidths.first ?? 24.0)
        var raw = AudioMath.floats(waveform)
        var shape = [1, 1, raw.count]

        var scale: Float? = nil
        if config.normalize {
            // mono RMS over the whole clip.
            var ss: Float = 0
            for v in raw { ss += v * v }
            let rms = sqrtf(ss / Float(max(raw.count, 1))) + 1e-8
            for i in 0..<raw.count { raw[i] /= rms }
            scale = rms
        }

        var data = raw
        let z = encoder.forward(&data, shape: &shape)
        let codes = try quantizer.encode(z.data, shape: z.shape, bandwidth: bw)
        return (codes, scale)
    }

    // ─── Decode ───────────────────────────────────────────────────

    /// Decode integer code streams back into a waveform Tensor `[1, 1, L]`.
    ///
    /// - codes: one `[Int32]` stream per codebook (output of `encode`).
    /// - scale: optional normalization scale from `encodeWithScale`.
    public func decode(codes: [[Int32]], scale: Float? = nil) throws -> Tensor {
        let z = try quantizer.decode(codes: codes)
        var data = z.data
        var shape = z.shape
        var out = decoder.forward(&data, shape: &shape)
        if let s = scale {
            for i in 0..<out.data.count { out.data[i] *= s }
        }
        let t = Tensor.empty(shape: out.shape, dtype: .f32)
        t.copyIn(from: out.data)
        return t
    }
}
