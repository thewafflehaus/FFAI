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
// DeepFilterNet — speech enhancement family (audio-in / audio-out).
//
// DeepFilterNet removes background noise from mono speech audio using a
// dual-pathway encoder-decoder architecture:
//   * ERB pathway — captures the perceptual energy envelope at ERB-scale.
//   * DF (deep filter) pathway — learns complex FIR gains in the STFT domain.
//
// Three model versions are supported (V1, V2, V3).  V2 and V3 share the same
// architecture; V1 uses grouped convolutions and GRUs.  This file implements
// offline batch enhancement (`enhance(waveform:)`).  For streaming, the
// reference mlx-audio-swift streamer can be ported in a follow-up.
//
// Weight source: `mlx-community/DeepFilterNet-mlx` (subfolders v1/v2/v3).
//
// Capability: `Capability.speechToSpeech` ([.audioIn, .audioOut]).
//
// Key entry points:
//   * `DeepFilterNetConfig`       — model hyper-parameters (mirrors config.json).
//   * `DeepFilterNetModel`        — loaded model with `enhance(waveform:)` API.
//   * `DeepFilterNetModel.load`   — local directory loader.
//   * `DeepFilterNetModel.fromPretrained` — HuggingFace download + load.
//
// CPU + Accelerate only (no Metal kernels required); the GRU recurrence and
// all conv/BN ops run on the CPU since DeepFilterNet is too small to justify
// GPU dispatch overhead.

import Accelerate
import Foundation

// MARK: - Errors

public enum DeepFilterNetError: Error, LocalizedError, CustomStringConvertible {
    case missingConfig(URL)
    case missingWeights(URL)
    case missingWeightKey(String)
    case invalidAudioShape

    public var errorDescription: String? { description }

    public var description: String {
        switch self {
        case .missingConfig(let dir):
            return "DeepFilterNet: config.json not found at \(dir.path)"
        case .missingWeights(let dir):
            return "DeepFilterNet: no .safetensors weights at \(dir.path)"
        case .missingWeightKey(let k):
            return "DeepFilterNet: required weight missing: \(k)"
        case .invalidAudioShape:
            return "DeepFilterNet: expected a 1-D mono waveform"
        }
    }
}

// MARK: - Config

/// DeepFilterNet model configuration — mirrors `config.json`.
public struct DeepFilterNetConfig: Codable, Sendable {
    // Audio / STFT
    public var sampleRate: Int = 48_000
    public var fftSize: Int = 960
    public var hopSize: Int = 480

    // ERB pathway
    public var nbErb: Int = 32
    public var minNbErbFreqs: Int = 2
    /// Optional explicit ERB band widths (overrides the computed schedule).
    public var erbWidths: [Int]? = nil

    // Deep-filter pathway
    public var nbDf: Int = 96
    public var dfOrder: Int = 5
    public var dfLookahead: Int = 2

    // Network dimensions
    public var convCh: Int = 64
    public var convLookahead: Int = 2
    public var embHiddenDim: Int = 256
    public var embNumLayers: Int = 3
    public var dfHiddenDim: Int = 256
    public var dfNumLayers: Int = 2
    public var gruGroups: Int = 8
    public var linearGroups: Int = 16
    public var encLinearGroups: Int = 32
    public var groupShuffle: Bool = false
    public var encConcat: Bool = false
    public var dfGruSkip: String = "groupedlinear"

    // Kernel sizes
    public var convKernel: [Int] = [1, 3]
    public var convtKernel: [Int] = [1, 3]
    public var convKernelInp: [Int] = [3, 3]
    public var dfPathwayKernelSizeT: Int = 5

    // LSNR range
    public var lsnrMax: Int = 35
    public var lsnrMin: Int = -15

    // Version discriminator
    public var modelVersion: String = "DeepFilterNet3"

    // Derived
    public var freqBins: Int { fftSize / 2 + 1 }
    public var isV1: Bool { modelVersion.lowercased() == "deepfilternet" }

    enum CodingKeys: String, CodingKey {
        case sampleRate, fftSize, hopSize
        case nbErb, minNbErbFreqs, erbWidths
        case nbDf, dfOrder, dfLookahead
        case convCh, convLookahead, embHiddenDim, embNumLayers
        case dfHiddenDim, dfNumLayers, gruGroups, linearGroups
        case encLinearGroups, groupShuffle, encConcat, dfGruSkip
        case convKernel, convtKernel, convKernelInp, dfPathwayKernelSizeT
        case lsnrMax, lsnrMin, modelVersion
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? sampleRate
        fftSize = try c.decodeIfPresent(Int.self, forKey: .fftSize) ?? fftSize
        hopSize = try c.decodeIfPresent(Int.self, forKey: .hopSize) ?? hopSize
        nbErb = try c.decodeIfPresent(Int.self, forKey: .nbErb) ?? nbErb
        minNbErbFreqs = try c.decodeIfPresent(Int.self, forKey: .minNbErbFreqs) ?? minNbErbFreqs
        erbWidths = try c.decodeIfPresent([Int].self, forKey: .erbWidths) ?? erbWidths
        nbDf = try c.decodeIfPresent(Int.self, forKey: .nbDf) ?? nbDf
        dfOrder = try c.decodeIfPresent(Int.self, forKey: .dfOrder) ?? dfOrder
        dfLookahead = try c.decodeIfPresent(Int.self, forKey: .dfLookahead) ?? dfLookahead
        convCh = try c.decodeIfPresent(Int.self, forKey: .convCh) ?? convCh
        convLookahead = try c.decodeIfPresent(Int.self, forKey: .convLookahead) ?? convLookahead
        embHiddenDim = try c.decodeIfPresent(Int.self, forKey: .embHiddenDim) ?? embHiddenDim
        embNumLayers = try c.decodeIfPresent(Int.self, forKey: .embNumLayers) ?? embNumLayers
        dfHiddenDim = try c.decodeIfPresent(Int.self, forKey: .dfHiddenDim) ?? dfHiddenDim
        dfNumLayers = try c.decodeIfPresent(Int.self, forKey: .dfNumLayers) ?? dfNumLayers
        gruGroups = try c.decodeIfPresent(Int.self, forKey: .gruGroups) ?? gruGroups
        linearGroups = try c.decodeIfPresent(Int.self, forKey: .linearGroups) ?? linearGroups
        encLinearGroups =
            try c.decodeIfPresent(Int.self, forKey: .encLinearGroups) ?? encLinearGroups
        groupShuffle = try c.decodeIfPresent(Bool.self, forKey: .groupShuffle) ?? groupShuffle
        encConcat = try c.decodeIfPresent(Bool.self, forKey: .encConcat) ?? encConcat
        dfGruSkip = try c.decodeIfPresent(String.self, forKey: .dfGruSkip) ?? dfGruSkip
        convKernel = try c.decodeIfPresent([Int].self, forKey: .convKernel) ?? convKernel
        convtKernel = try c.decodeIfPresent([Int].self, forKey: .convtKernel) ?? convtKernel
        convKernelInp = try c.decodeIfPresent([Int].self, forKey: .convKernelInp) ?? convKernelInp
        dfPathwayKernelSizeT =
            try c.decodeIfPresent(Int.self, forKey: .dfPathwayKernelSizeT) ?? dfPathwayKernelSizeT
        lsnrMax = try c.decodeIfPresent(Int.self, forKey: .lsnrMax) ?? lsnrMax
        lsnrMin = try c.decodeIfPresent(Int.self, forKey: .lsnrMin) ?? lsnrMin
        modelVersion = try c.decodeIfPresent(String.self, forKey: .modelVersion) ?? modelVersion
    }
}

// MARK: - Weight table

/// Thin wrapper over a `[String: [Float]]` weight dictionary, loaded from
/// the SafeTensors bundle at model-load time.  All weights are materialised
/// to `[Float]` arrays for CPU-side math.
struct DFNWeightTable: @unchecked Sendable {
    let table: [String: [Float]]
    let shapes: [String: [Int]]

    subscript(_ key: String) -> [Float]? { table[key] }

    func require(_ key: String) throws -> [Float] {
        guard let w = table[key] else { throw DeepFilterNetError.missingWeightKey(key) }
        return w
    }

    func shape(_ key: String) -> [Int]? { shapes[key] }
    func has(_ key: String) -> Bool { table[key] != nil }
}

// MARK: - Pre-computed BN affine coefficients

/// A fused BatchNorm layer: `y = x * scale + bias` applied channel-wise in
/// `BCHW` layout (`shape = [1, C, 1, 1]` each).
struct DFNBatchNorm: Sendable {
    let scale: [Float]  // length C
    let bias: [Float]  // length C

    /// Apply to `x` in `[B, C, H, W]` layout (C = scale.count).
    /// Returns a new array with the same layout.
    func apply(_ x: [Float], batch: Int, channels: Int, height: Int, width: Int) -> [Float] {
        precondition(channels == scale.count)
        var out = [Float](repeating: 0, count: x.count)
        let hw = height * width
        for b in 0 ..< batch {
            for c in 0 ..< channels {
                let s = scale[c]
                let bi = bias[c]
                let base = (b * channels + c) * hw
                for i in 0 ..< hw {
                    out[base + i] = x[base + i] * s + bi
                }
            }
        }
        return out
    }
}

// MARK: - DeepFilterNetModel

/// Loaded DeepFilterNet speech enhancement model.
///
/// **Offline enhance:**
/// ```swift
/// let model = try DeepFilterNetModel.load(from: URL(fileURLWithPath: "path/to/v3"))
/// let enhanced = try model.enhance(waveform: noisyPCM)
/// ```
///
/// Capability: `Capability.speechToSpeech` = `[.audioIn, .audioOut]`.
public final class DeepFilterNetModel: @unchecked Sendable {

    // ─── Model identification ──────────────────────────────────────────

    /// Model types recognised in `config.json` for this family.
    public static let modelTypes: Set<String> = [
        "deepfilternet", "deepfilternet2", "deepfilternet3",
        "deep_filter_net", "deepfilternet_v3",
    ]

    /// `config.json` architecture strings that map to this family.
    public static let architectures: Set<String> = [
        "DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3",
    ]

    /// Capability set exposed by this model.
    public static let capabilities: Set<Capability> = Capability.speechToSpeech

    // ─── Public properties ─────────────────────────────────────────────

    public let config: DeepFilterNetConfig
    public let modelDirectory: URL

    // ─── Internal weight storage ───────────────────────────────────────

    let weights: DFNWeightTable
    let erbBandWidths: [Int]
    let vorbisWindowData: [Float]
    let normAlpha: Float

    // Pre-computed BN affine maps (prefix → BN).
    let bnMap: [String: DFNBatchNorm]

    // Pre-transposed Conv2d weights: key → [outC, kT, kF, inC] (OHWI layout).
    // We store them already transposed for use in conv2d.
    let conv2dWeightsOHWI: [String: [Float]]
    let conv2dShapes: [String: (outC: Int, kT: Int, kF: Int, inC: Int)]

    // Pre-transposed GRU weights: wih/whh → transposed [I, 3H] / [H, 3H].
    let gruWeightIHT: [String: [Float]]  // transposed [I, 3H]
    let gruWeightHHT: [String: [Float]]  // transposed [H, 3H]

    // ─── Initialiser (private — use load() / fromPretrained()) ────────

    private init(
        config: DeepFilterNetConfig,
        modelDirectory: URL,
        weights: DFNWeightTable
    ) throws {
        self.config = config
        self.modelDirectory = modelDirectory
        self.weights = weights

        // ERB band widths.
        if let explicit = config.erbWidths,
            explicit.reduce(0, +) == config.freqBins
        {
            self.erbBandWidths = explicit
        } else {
            self.erbBandWidths = dfErbBandWidths(
                sampleRate: config.sampleRate,
                fftSize: config.fftSize,
                nbBands: config.nbErb,
                minNbFreqs: max(1, config.minNbErbFreqs)
            )
        }

        self.vorbisWindowData = DeepFilterNetSTFT.vorbisWindow(size: config.fftSize)
        self.normAlpha = Self.computeNormAlpha(
            hopSize: config.hopSize, sampleRate: config.sampleRate)

        // Pre-compute fused BN affine (γ, β, μ, σ²) → (scale, bias).
        self.bnMap = Self.buildBatchNormMap(weights: weights)

        // Pre-transpose Conv2d weights (PyTorch layout [outC, inC, kH, kW]
        // → OHWI [outC, kH, kW, inC] for vDSP-friendly access).
        var ohwiWeights = [String: [Float]]()
        var ohwiShapes = [String: (outC: Int, kT: Int, kF: Int, inC: Int)]()
        for (key, w) in weights.table where key.hasSuffix(".weight") {
            guard let shape = weights.shapes[key], shape.count == 4 else { continue }
            let (outC, inC, kT, kF) = (shape[0], shape[1], shape[2], shape[3])
            // Transpose [outC, inC, kT, kF] → [outC, kT, kF, inC].
            var transposed = [Float](repeating: 0, count: w.count)
            for o in 0 ..< outC {
                for t in 0 ..< kT {
                    for f in 0 ..< kF {
                        for i in 0 ..< inC {
                            // src: o*inC*kT*kF + i*kT*kF + t*kF + f
                            // dst: o*kT*kF*inC + t*kF*inC + f*inC + i
                            transposed[o * kT * kF * inC + t * kF * inC + f * inC + i] =
                                w[o * inC * kT * kF + i * kT * kF + t * kF + f]
                        }
                    }
                }
            }
            ohwiWeights[key] = transposed
            ohwiShapes[key] = (outC: outC, kT: kT, kF: kF, inC: inC)
        }
        self.conv2dWeightsOHWI = ohwiWeights
        self.conv2dShapes = ohwiShapes

        // Pre-transpose GRU weights.
        var wihT = [String: [Float]]()
        var whhT = [String: [Float]]()
        for (key, w) in weights.table where key.contains(".gru.weight_") && key.hasSuffix("_l0") {
            guard let shape = weights.shapes[key], shape.count == 2 else { continue }
            let rows = shape[0]
            let cols = shape[1]
            var t = [Float](repeating: 0, count: rows * cols)
            for r in 0 ..< rows {
                for c in 0 ..< cols {
                    t[c * rows + r] = w[r * cols + c]
                }
            }
            if key.contains(".weight_ih_") {
                wihT[key] = t
            } else {
                whhT[key] = t
            }
        }
        self.gruWeightIHT = wihT
        self.gruWeightHHT = whhT
    }

    // MARK: - Loading

    /// Load a DeepFilterNet checkpoint from a local directory.
    ///
    /// The directory must contain `config.json` and at least one `.safetensors` file.
    /// A `subfolder` (e.g. `"v3"`) is used when the directory contains multiple version
    /// subfolders.
    ///
    /// - Parameters:
    ///   - directory: Path to the model directory.
    ///   - subfolder: Optional subfolder override.
    ///   - device: Metal device (for the SafeTensors loader; weights are then copied
    ///     to host Float arrays).
    public static func load(
        from directory: URL,
        subfolder: String? = nil,
        device: Device = .shared
    ) throws -> DeepFilterNetModel {
        // Resolve the actual model directory (handle optional subfolder).
        var modelDir = directory
        let configURL = modelDir.appendingPathComponent("config.json")
        if !FileManager.default.fileExists(atPath: configURL.path),
            let sub = subfolder, !sub.isEmpty
        {
            modelDir = directory.appendingPathComponent(sub)
        }

        let resolvedConfigURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: resolvedConfigURL.path) else {
            throw DeepFilterNetError.missingConfig(modelDir)
        }

        // Decode config.
        let configData = try Data(contentsOf: resolvedConfigURL)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        var config = try decoder.decode(DeepFilterNetConfig.self, from: configData)
        if config.modelVersion.isEmpty { config.modelVersion = "DeepFilterNet3" }

        // Load SafeTensors.
        let bundle = try SafeTensorsBundle(directory: modelDir, device: device)
        guard !bundle.allKeys.isEmpty else {
            throw DeepFilterNetError.missingWeights(modelDir)
        }

        // Materialise weights to [Float] + record shapes.
        var table = [String: [Float]]()
        var shapes = [String: [Int]]()
        for key in bundle.allKeys {
            let tensor = try bundle.tensor(named: key)
            shapes[key] = tensor.shape
            // Convert to Float (handles F32, F16, BF16 via Tensor.toArray).
            switch tensor.dtype {
            case .f32:
                table[key] = tensor.toArray(as: Float.self)
            case .f16:
                let raw = tensor.toArray(as: Float16.self)
                table[key] = raw.map { Float($0) }
            case .bf16:
                let raw = tensor.toArray(as: UInt16.self)
                table[key] = raw.map { bits -> Float in
                    let floatBits = UInt32(bits) << 16
                    return Float(bitPattern: floatBits)
                }
            default:
                // Skip unsupported dtypes (e.g. metadata tensors).
                continue
            }
        }

        let wt = DFNWeightTable(table: table, shapes: shapes)
        return try DeepFilterNetModel(config: config, modelDirectory: modelDir, weights: wt)
    }

    /// Download (or hit cache) a DeepFilterNet checkpoint from HuggingFace and load it.
    ///
    /// - Parameters:
    ///   - idOrPath: HuggingFace repo id or local path (e.g. `"mlx-community/DeepFilterNet-mlx"`).
    ///   - subfolder: Subfolder within the repo (default `"v3"`).
    ///   - device: Metal device for SafeTensors loading.
    public static func fromPretrained(
        _ idOrPath: String = "mlx-community/DeepFilterNet-mlx",
        subfolder: String? = "v3",
        device: Device = .shared
    ) async throws -> DeepFilterNetModel {
        let locator = ModelLocator()
        let dir = try await locator.resolve(idOrPath: idOrPath)
        return try load(from: dir, subfolder: subfolder, device: device)
    }

    // MARK: - Detection

    /// Returns `true` if `config` describes a DeepFilterNet checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt.lowercased()) { return true }
        if let arch = config.architecture, architectures.contains(arch) { return true }
        return false
    }

    // MARK: - Public API

    /// Enhance mono speech audio by removing background noise.
    ///
    /// Runs a full offline enhancement pass:
    ///   1. STFT analysis
    ///   2. ERB energy + deep-filter feature extraction + normalization
    ///   3. Network forward (encoder → ERB decoder → DF decoder)
    ///   4. ERB gain masking + deep filtering
    ///   5. iSTFT synthesis + delay compensation
    ///
    /// - Parameter waveform: Mono PCM samples in `[-1, 1]` at `config.sampleRate`.
    /// - Returns: Denoised waveform with the same length as `waveform`.
    public func enhance(waveform: [Float]) throws -> [Float] {
        guard !waveform.isEmpty else { return [] }

        let origLen = waveform.count

        // 1. STFT analysis.
        let spec = DeepFilterNetSTFT.stft(
            audio: waveform,
            fftSize: config.fftSize,
            hopSize: config.hopSize,
            window: vorbisWindowData
        )
        let nT = spec.nFrames
        let nF = spec.freqBins

        // 2. ERB energy and features.
        // specMagSq: [nT, nF]
        var specMagSq = [Float](repeating: 0, count: nT * nF)
        for i in 0 ..< (nT * nF) {
            let r = spec.real[i]
            let im = spec.imag[i]
            specMagSq[i] = r * r + im * im
        }

        // ERB energies: [nT, nbErb]
        let erb = computeErbEnergies(specMagSq: specMagSq, nT: nT)
        // ERB dB: [nT, nbErb]
        var erbDB = [Float](repeating: 0, count: nT * config.nbErb)
        for i in 0 ..< erbDB.count {
            erbDB[i] = 10.0 * log10f(erb[i] + 1e-10)
        }

        // Feature normalization: [nT, nbErb]
        let featErb =
            config.isV1
            ? bandMeanNormExact(erbDB, bands: config.nbErb, nT: nT)
            : bandMeanNorm(erbDB, bands: config.nbErb, nT: nT)

        // Deep-filter feature normalization: [nT, nbDf, 2]
        let dfRe = (0 ..< nT).flatMap { t in
            (0 ..< config.nbDf).map { b in spec.real[t * nF + b] }
        }
        let dfIm = (0 ..< nT).flatMap { t in
            (0 ..< config.nbDf).map { b in spec.imag[t * nF + b] }
        }
        let (normDfRe, normDfIm) =
            config.isV1
            ? bandUnitNormExact(real: dfRe, imag: dfIm, freqs: config.nbDf, nT: nT)
            : bandUnitNorm(real: dfRe, imag: dfIm, freqs: config.nbDf, nT: nT)

        // 3. Network forward.
        let (specEnhReal, specEnhImag) = try networkForward(
            specReal: spec.real, specImag: spec.imag,
            featErb: featErb, dfRe: normDfRe, dfIm: normDfIm,
            nT: nT, nF: nF
        )

        // 4. iSTFT synthesis.
        let enhSpec = DeepFilterNetSpectrum(
            real: specEnhReal, imag: specEnhImag,
            nFrames: nT, freqBins: nF
        )
        return DeepFilterNetSTFT.istft(
            spectrum: enhSpec,
            fftSize: config.fftSize,
            hopSize: config.hopSize,
            window: vorbisWindowData,
            origLen: origLen
        )
    }

    // MARK: - Capability accessor

    public var capabilities: Set<Capability> { Self.capabilities }
}

// MARK: - Feature normalization

extension DeepFilterNetModel {

    /// EMA-based ERB mean normalization (vectorised, libDF-compatible).
    /// `x` is `[nT, bands]`, returns same shape.
    func bandMeanNorm(_ x: [Float], bands: Int, nT: Int) -> [Float] {
        let a = normAlpha
        let oneMinusA = 1.0 - a
        // EMA over time using the libDF init state.
        var state = dfLinspace(start: -60.0, end: -90.0, count: bands)
        var out = [Float](repeating: 0, count: nT * bands)
        for t in 0 ..< nT {
            let base = t * bands
            for e in 0 ..< bands {
                let xv = x[base + e]
                state[e] = xv * oneMinusA + state[e] * a
                out[base + e] = (xv - state[e]) / 40.0
            }
        }
        return out
    }

    /// Exact sequential ERB mean normalization (identical to bandMeanNorm for V1 parity).
    func bandMeanNormExact(_ x: [Float], bands: Int, nT: Int) -> [Float] {
        bandMeanNorm(x, bands: bands, nT: nT)
    }

    /// EMA-based complex unit normalization (vectorised).
    /// Inputs `real`/`imag` are `[nT, freqs]`, returns same shape each.
    func bandUnitNorm(
        real: [Float], imag: [Float],
        freqs: Int, nT: Int
    ) -> ([Float], [Float]) {
        let a = normAlpha
        let oneMinusA = 1.0 - a
        var state = dfLinspace(start: 0.001, end: 0.0001, count: freqs)
        var outR = [Float](repeating: 0, count: nT * freqs)
        var outI = [Float](repeating: 0, count: nT * freqs)
        for t in 0 ..< nT {
            let base = t * freqs
            for f in 0 ..< freqs {
                let idx = base + f
                let r = real[idx]
                let im = imag[idx]
                let mag = sqrtf(r * r + im * im)
                state[f] = mag * oneMinusA + state[f] * a
                let denom = sqrtf(max(state[f], 1e-12))
                outR[idx] = r / denom
                outI[idx] = im / denom
            }
        }
        return (outR, outI)
    }

    func bandUnitNormExact(
        real: [Float], imag: [Float],
        freqs: Int, nT: Int
    ) -> ([Float], [Float]) {
        bandUnitNorm(real: real, imag: imag, freqs: freqs, nT: nT)
    }

    /// Compute ERB energies using the stored band widths.
    /// `specMagSq` is `[nT, freqBins]`, returns `[nT, nbErb]`.
    func computeErbEnergies(specMagSq: [Float], nT: Int) -> [Float] {
        let nF = config.freqBins
        let nbErb = config.nbErb
        var out = [Float](repeating: 0, count: nT * nbErb)
        for t in 0 ..< nT {
            let srcBase = t * nF
            let dstBase = t * nbErb
            var binStart = 0
            for e in 0 ..< nbErb {
                let width = erbBandWidths[e]
                let binEnd = min(binStart + width, nF)
                var sum: Float = 0
                let count = max(1, binEnd - binStart)
                for b in binStart ..< binEnd { sum += specMagSq[srcBase + b] }
                out[dstBase + e] = sum / Float(count)
                binStart = binEnd
            }
        }
        return out
    }
}

// MARK: - Norm alpha

extension DeepFilterNetModel {
    /// Compute the EMA alpha from hop / sample-rate (libDF convention).
    static func computeNormAlpha(hopSize: Int, sampleRate: Int) -> Float {
        let aRaw = exp(-Float(hopSize) / Float(sampleRate))
        var precision = 3
        var a: Float = 1.0
        while a >= 1.0 {
            let scale = powf(10, Float(precision))
            a = (aRaw * scale).rounded() / scale
            precision += 1
        }
        return a
    }
}

// MARK: - Batch-norm map construction

extension DeepFilterNetModel {
    static func buildBatchNormMap(weights: DFNWeightTable) -> [String: DFNBatchNorm] {
        var map = [String: DFNBatchNorm]()
        for (key, mean) in weights.table where key.hasSuffix(".running_mean") {
            let prefix = String(key.dropLast(".running_mean".count))
            guard let gamma = weights["\(prefix).weight"],
                let beta = weights["\(prefix).bias"],
                let variance = weights["\(prefix).running_var"]
            else { continue }
            let scale = zip(gamma, variance).map { g, v in g / sqrtf(v + 1e-5) }
            let bias = zip(zip(beta, mean), scale).map { pair, s in pair.0 - pair.1 * s }
            map[prefix] = DFNBatchNorm(scale: scale, bias: bias)
        }
        return map
    }
}

// MARK: - Network forward

extension DeepFilterNetModel {

    /// Full offline forward pass for V2/V3 (shared architecture).
    /// Returns `(enhReal [nT, nF], enhImag [nT, nF])`.
    func networkForward(
        specReal: [Float], specImag: [Float],
        featErb: [Float], dfRe: [Float], dfIm: [Float],
        nT: Int, nF: Int
    ) throws -> ([Float], [Float]) {
        // Layout note: the V2/V3 network processes 4-D tensors in BCHW
        // (batch=1, channels, time, freq).  We represent them as flat
        // [B*C*T*F] arrays and pass shape tuples explicitly.  B=1 throughout.
        let B = 1

        // Shape for ERB input: [1, 1, nT, nbErb] → channels=1.
        let erbIn4D = featErb  // already [nT*nbErb] as [T*F]

        // Shape for DF input: [1, 2, nT, nbDf] (real + imag as 2 channels).
        var dfIn4D = [Float](repeating: 0, count: 2 * nT * config.nbDf)
        for t in 0 ..< nT {
            for b in 0 ..< config.nbDf {
                dfIn4D[0 * nT * config.nbDf + t * config.nbDf + b] = dfRe[t * config.nbDf + b]
                dfIn4D[1 * nT * config.nbDf + t * config.nbDf + b] = dfIm[t * config.nbDf + b]
            }
        }

        // Apply convLookahead shift (shift the input forward in time by
        // convLookahead frames, padding the end with zeros).
        let lookahead = config.convLookahead
        let erbShifted =
            lookahead > 0
            ? applyLookaheadBCHW(erbIn4D, B: B, C: 1, T: nT, F: config.nbErb, l: lookahead)
            : erbIn4D
        let dfShifted =
            lookahead > 0
            ? applyLookaheadBCHW(dfIn4D, B: B, C: 2, T: nT, F: config.nbDf, l: lookahead) : dfIn4D

        // ── Encoder ──────────────────────────────────────────────────────
        // ERB branch.
        let e0 = try encConv(
            erbShifted, B: B, C: 1, T: nT, F: config.nbErb, prefix: "enc.erb_conv0", main: 1,
            pointwise: nil, bn: 2, fstride: 1)
        let e0T = nT
        let e0F = config.nbErb
        let e0C = convShape(prefix: "enc.erb_conv0", main: 1).outC
        let e1 = try encConv(
            e0, B: B, C: e0C, T: e0T, F: e0F, prefix: "enc.erb_conv1", main: 0, pointwise: 1, bn: 2,
            fstride: 2)
        let e1C = convShape(prefix: "enc.erb_conv1", main: 0).outC
        let e1F = e0F / 2
        let e2 = try encConv(
            e1, B: B, C: e1C, T: nT, F: e1F, prefix: "enc.erb_conv2", main: 0, pointwise: 1, bn: 2,
            fstride: 2)
        let e2C = convShape(prefix: "enc.erb_conv2", main: 0).outC
        let e2F = e1F / 2
        let e3 = try encConv(
            e2, B: B, C: e2C, T: nT, F: e2F, prefix: "enc.erb_conv3", main: 0, pointwise: 1, bn: 2,
            fstride: 1)
        let e3C = convShape(prefix: "enc.erb_conv3", main: 0).outC
        let e3F = e2F

        // DF branch.
        let c0 = try encConv(
            dfShifted, B: B, C: 2, T: nT, F: config.nbDf, prefix: "enc.df_conv0", main: 1,
            pointwise: 2, bn: 3, fstride: 1)
        let c0C = convShape(prefix: "enc.df_conv0", main: 1).outC
        let c1 = try encConv(
            c0, B: B, C: c0C, T: nT, F: config.nbDf, prefix: "enc.df_conv1", main: 0, pointwise: 1,
            bn: 2, fstride: 2)
        let c1C = convShape(prefix: "enc.df_conv1", main: 0).outC
        let c1F = config.nbDf / 2

        // Embedding.
        // cemb: flatten c1 from [1, c1C, nT, c1F] → [1, nT, c1C*c1F], then groupedLinear.
        var cemb = bchwTobtf(c1, B: B, C: c1C, T: nT, F: c1F)  // [nT, c1C*c1F]
        // The checkpoint stores this weight as "enc.df_fc_emb.0.weight" (shape [8, 384, 64]).
        // groupedLinear with prefix="enc.df_fc_emb.0" would append ".0.weight" giving the
        // wrong key "enc.df_fc_emb.0.0.weight", so we pass the full key directly.
        cemb = try groupedLinearRelu(cemb, T: nT, weightKey: "enc.df_fc_emb.0.weight")

        // emb: flatten e3 from [1, e3C, nT, e3F] → [1, nT, e3C*e3F].
        var emb = bchwTobtf(e3, B: B, C: e3C, T: nT, F: e3F)  // [nT, e3C*e3F]
        // Merge: add or concat.
        if config.encConcat {
            emb = zip(emb, cemb).map { $0 + $1 }  // placeholder — proper concat below
            // Proper concat along feature axis.
            let embDim = emb.count / nT
            let cDim = cemb.count / nT
            var merged = [Float](repeating: 0, count: nT * (embDim + cDim))
            for t in 0 ..< nT {
                for d in 0 ..< embDim { merged[t * (embDim + cDim) + d] = emb[t * embDim + d] }
                for d in 0 ..< cDim {
                    merged[t * (embDim + cDim) + embDim + d] = cemb[t * cDim + d]
                }
            }
            emb = merged
        } else {
            // element-wise add (requires same dim — typical for V2/V3).
            let embDim = emb.count / nT
            let cDim = min(embDim, cemb.count / nT)
            for t in 0 ..< nT {
                for d in 0 ..< cDim {
                    emb[t * embDim + d] += cemb[t * cDim + d]
                }
            }
        }

        // Embedding GRU.
        emb = try squeezedGRU(
            emb, T: nT, prefix: "enc.emb_gru", hiddenSize: config.embHiddenDim, linearOut: true)

        // ── ERB Decoder ───────────────────────────────────────────────────
        let embDec = try squeezedGRU(
            emb, T: nT, prefix: "erb_dec.emb_gru", hiddenSize: config.embHiddenDim, linearOut: true)
        let embDecDim = embDec.count / nT
        // Project to [1, embDecDim/e3F, nT, e3F] BCHW.
        let embDecC = embDecDim / max(1, e3F)
        let embDecBCHW = tfToBchw(embDec, T: nT, dim: embDecDim, newC: embDecC, F: e3F)

        var d3 = addBCHW(
            relu4D(try pathwayConv(e3, B: B, C: e3C, T: nT, F: e3F, prefix: "erb_dec.conv3p")),
            embDecBCHW, B: B, C: embDecC, T: nT, F: e3F)
        d3 = relu4D(try regularBlock(d3, B: B, C: embDecC, T: nT, F: e3F, prefix: "erb_dec.convt3"))

        let d3C = embDecC
        var d2 = addBCHW(
            relu4D(try pathwayConv(e2, B: B, C: e2C, T: nT, F: e2F, prefix: "erb_dec.conv2p")),
            alignTime(d3, C: d3C, T: nT, F: e2F), B: B, C: d3C, T: nT, F: e2F)
        d2 = relu4D(
            try transposeBlock(
                d2, B: B, C: d3C, T: nT, F: e2F, prefix: "erb_dec.convt2", fstride: 2))
        let d2C = d3C

        var d1 = addBCHW(
            relu4D(try pathwayConv(e1, B: B, C: e1C, T: nT, F: e1F, prefix: "erb_dec.conv1p")),
            alignTime(d2, C: d2C, T: nT, F: e1F), B: B, C: d2C, T: nT, F: e1F)
        d1 = relu4D(
            try transposeBlock(
                d1, B: B, C: d2C, T: nT, F: e1F, prefix: "erb_dec.convt1", fstride: 2))
        let d1C = d2C

        let d0 = addBCHW(
            relu4D(try pathwayConv(e0, B: B, C: e0C, T: nT, F: e0F, prefix: "erb_dec.conv0p")),
            alignTime(d1, C: d1C, T: nT, F: e0F), B: B, C: d1C, T: nT, F: e0F)
        var maskBCHW = try outputConv(d0, B: B, C: d1C, T: nT, F: e0F, prefix: "erb_dec.conv0_out")
        // Sigmoid.
        for i in maskBCHW.indices { maskBCHW[i] = 1.0 / (1.0 + expf(-maskBCHW[i])) }
        let maskC = convShape(prefix: "erb_dec.conv0_out", main: 0).outC

        // Apply ERB mask.
        // mask is [1, maskC, nT, e0F] where e0F == nbErb; maskC should be 1.
        // gains = mask @ erbInvFB → [nT, nF] gain per frequency bin.
        let maskFlat = maskBCHW  // [1*maskC*nT*nbErb]
        var specMaskedReal = specReal
        var specMaskedImag = specImag
        // Build inverse ERB filterbank gains.
        let gains = applyErbMask(maskFlat: maskFlat, nT: nT, maskC: maskC)
        for t in 0 ..< nT {
            for f in 0 ..< nF {
                let g = gains[t * nF + f]
                specMaskedReal[t * nF + f] *= g
                specMaskedImag[t * nF + f] *= g
            }
        }

        // ── DF Decoder ────────────────────────────────────────────────────
        var dfGruOut = try squeezedGRU(
            emb, T: nT, prefix: "df_dec.df_gru", hiddenSize: config.dfHiddenDim, linearOut: false)

        // Optional skip connection.
        if weights.has("df_dec.df_skip.weight") {
            let skip = try groupedLinear(
                emb, T: nT, prefix: nil, weightKey: "df_dec.df_skip.weight")
            let skipDim = skip.count / nT
            let dfDim = dfGruOut.count / nT
            let minDim = min(skipDim, dfDim)
            for t in 0 ..< nT {
                for d in 0 ..< minDim {
                    dfGruOut[t * dfDim + d] += skip[t * skipDim + d]
                }
            }
        }

        // df_convp branch from c0.
        let c0F = config.nbDf
        var c0p = try conv2dLayer(
            c0, B: B, C: c0C, T: nT, F: c0F, weightKey: "df_dec.df_convp.1.weight", bias: nil,
            fstride: 1, lookahead: 0)
        let c0p1C = convShape(prefix: "df_dec.df_convp", main: 1).outC
        c0p = try conv2dLayer(
            c0p, B: B, C: c0p1C, T: nT, F: c0F, weightKey: "df_dec.df_convp.2.weight", bias: nil,
            fstride: 1, lookahead: 0)
        let c0p2C = convShape(prefix: "df_dec.df_convp", main: 2).outC
        c0p = relu4D(
            try applyBatchNorm(c0p, B: B, C: c0p2C, T: nT, F: c0F, prefix: "df_dec.df_convp.3"))
        // c0p BCHW → [nT, c0p2C*c0F].
        let c0pFlat = bchwTobtf(c0p, B: B, C: c0p2C, T: nT, F: c0F)

        // df_out: tanh(groupedLinear(dfGruOut)) → [nT, nbDf * dfOrder * 2].
        var dfOut = try groupedLinear(
            dfGruOut, T: nT, prefix: nil, weightKey: "df_dec.df_out.0.weight")
        for i in dfOut.indices { dfOut[i] = tanhf(dfOut[i]) }
        // Reshape to [nT, nbDf, dfOrder*2].

        // Add c0pFlat (as [nT, nbDf, dfOrder*2] after proper reshape).
        let minLen = min(dfOut.count, c0pFlat.count)
        for i in 0 ..< minLen { dfOut[i] += c0pFlat[i] }

        // dfCoefs shape: [nT, dfOrder, nbDf, 2] (for deepFilter application).
        // dfOut is [nT, nbDf*dfOrder*2]; reshape to [nT, nbDf, dfOrder, 2] then transpose to [nT, dfOrder, nbDf, 2].
        var dfCoefs = [Float](repeating: 0, count: nT * config.dfOrder * config.nbDf * 2)
        for t in 0 ..< nT {
            for f in 0 ..< config.nbDf {
                for k in 0 ..< config.dfOrder {
                    let srcBase =
                        t * config.nbDf * config.dfOrder * 2 + f * config.dfOrder * 2 + k * 2
                    let dstBase = t * config.dfOrder * config.nbDf * 2 + k * config.nbDf * 2 + f * 2
                    if srcBase + 1 < dfOut.count {
                        dfCoefs[dstBase] = dfOut[srcBase]
                        dfCoefs[dstBase + 1] = dfOut[srcBase + 1]
                    }
                }
            }
        }

        // Deep filter: apply FIR filter with dfCoefs to the low-freq region.
        let (enhLowReal, enhLowImag) = deepFilter(
            specReal: specReal, specImag: specImag,
            dfCoefs: dfCoefs, nT: nT, nF: nF
        )

        // Combine: low-freq from deep filter, high-freq from ERB mask.
        var outReal = specMaskedReal
        var outImag = specMaskedImag
        for t in 0 ..< nT {
            for b in 0 ..< config.nbDf {
                outReal[t * nF + b] = enhLowReal[t * config.nbDf + b]
                outImag[t * nF + b] = enhLowImag[t * config.nbDf + b]
            }
        }

        return (outReal, outImag)
    }

    // MARK: - Deep filter

    /// Apply the deep FIR filter over the low-freq region.
    /// `dfCoefs` is `[nT, dfOrder, nbDf, 2]` (real + imag per coefficient).
    /// Returns `(real [nT*nbDf], imag [nT*nbDf])`.
    func deepFilter(
        specReal: [Float], specImag: [Float],
        dfCoefs: [Float], nT: Int, nF: Int
    ) -> ([Float], [Float]) {
        let nbDf = config.nbDf
        let dfOrder = config.dfOrder
        let padLeft = dfOrder - 1 - config.dfLookahead

        var outReal = [Float](repeating: 0, count: nT * nbDf)
        var outImag = [Float](repeating: 0, count: nT * nbDf)

        for t in 0 ..< nT {
            for b in 0 ..< nbDf {
                var accR: Float = 0
                var accI: Float = 0
                for k in 0 ..< dfOrder {
                    let srcT = t - padLeft + k
                    var sr: Float = 0
                    var si: Float = 0
                    if srcT >= 0, srcT < nT {
                        sr = specReal[srcT * nF + b]
                        si = specImag[srcT * nF + b]
                    }
                    let coefBase = t * dfOrder * nbDf * 2 + k * nbDf * 2 + b * 2
                    let cr = coefBase < dfCoefs.count ? dfCoefs[coefBase] : 0
                    let ci = (coefBase + 1) < dfCoefs.count ? dfCoefs[coefBase + 1] : 0
                    accR += sr * cr - si * ci
                    accI += sr * ci + si * cr
                }
                outReal[t * nbDf + b] = accR
                outImag[t * nbDf + b] = accI
            }
        }
        return (outReal, outImag)
    }

    // MARK: - ERB mask application

    /// Convert ERB-domain mask → per-bin gains and multiply into spectrum.
    /// `maskFlat` is `[maskC * nT * nbErb]` (C=1 typically).
    func applyErbMask(maskFlat: [Float], nT: Int, maskC: Int) -> [Float] {
        let nbErb = config.nbErb
        let nF = config.freqBins
        var gains = [Float](repeating: 0, count: nT * nF)
        for t in 0 ..< nT {
            var binStart = 0
            for e in 0 ..< nbErb {
                // Average across maskC channels (usually 1).
                var gain: Float = 0
                for c in 0 ..< maskC {
                    let idx = c * nT * nbErb + t * nbErb + e
                    if idx < maskFlat.count { gain += maskFlat[idx] }
                }
                gain /= Float(maskC)
                // Broadcast to all bins in this ERB band.
                let width = erbBandWidths[e]
                let binEnd = min(binStart + width, nF)
                for b in binStart ..< binEnd {
                    gains[t * nF + b] = gain
                }
                binStart = binEnd
            }
        }
        return gains
    }
}

// MARK: - Layout helpers

extension DeepFilterNetModel {

    /// Convert `[B, C, T, F]` to `[T, dim]` where `dim = C*F`.
    func bchwTobtf(_ x: [Float], B: Int, C: Int, T: Int, F: Int) -> [Float] {
        var out = [Float](repeating: 0, count: T * C * F)
        for b in 0 ..< B {
            for t in 0 ..< T {
                for c in 0 ..< C {
                    for f in 0 ..< F {
                        out[t * C * F + c * F + f] = x[b * C * T * F + c * T * F + t * F + f]
                    }
                }
            }
        }
        return out
    }

    /// Convert `[T, dim]` back to `[B, C, T, F]`.
    func tfToBchw(_ x: [Float], T: Int, dim: Int, newC: Int, F: Int) -> [Float] {
        var out = [Float](repeating: 0, count: newC * T * F)
        for t in 0 ..< T {
            for c in 0 ..< newC {
                for f in 0 ..< F {
                    let srcIdx = t * dim + c * F + f
                    if srcIdx < x.count {
                        out[c * T * F + t * F + f] = x[srcIdx]
                    }
                }
            }
        }
        return out
    }

    func addBCHW(_ a: [Float], _ b: [Float], B: Int, C: Int, T: Int, F: Int) -> [Float] {
        let n = min(a.count, b.count)
        var out = a
        for i in 0 ..< n { out[i] += b[i] }
        return out
    }

    func alignTime(_ x: [Float], C: Int, T: Int, F: Int) -> [Float] { x }

    func relu4D(_ x: [Float]) -> [Float] {
        x.map { max(0, $0) }
    }

    func applyLookaheadBCHW(
        _ x: [Float], B: Int, C: Int, T: Int, F: Int, l: Int
    ) -> [Float] {
        guard l > 0, T > l else { return x }
        // Shift time axis forward by `l` (drop first `l` frames, zero-pad end).
        var out = [Float](repeating: 0, count: B * C * T * F)
        for b in 0 ..< B {
            for c in 0 ..< C {
                for t in 0 ..< T {
                    let srcT = t + l
                    if srcT < T {
                        let srcBase = b * C * T * F + c * T * F + srcT * F
                        let dstBase = b * C * T * F + c * T * F + t * F
                        out.replaceSubrange(
                            dstBase ..< (dstBase + F), with: x[srcBase ..< (srcBase + F)])
                    }
                }
            }
        }
        return out
    }
}

// MARK: - Conv shape helpers

extension DeepFilterNetModel {

    struct ConvDims { var outC, kT, kF, inC: Int }

    func convShape(prefix: String, main: Int) -> ConvDims {
        let key = "\(prefix).\(main).weight"
        if let s = conv2dShapes[key] {
            return ConvDims(outC: s.outC, kT: s.kT, kF: s.kF, inC: s.inC)
        }
        if let shape = weights.shapes[key], shape.count == 4 {
            return ConvDims(outC: shape[0], kT: shape[2], kF: shape[3], inC: shape[1])
        }
        return ConvDims(outC: 1, kT: 1, kF: 1, inC: 1)
    }
}

// MARK: - Layer primitives

extension DeepFilterNetModel {

    // MARK: Conv2d

    /// 2-D convolution on a `[B, C, T, F]` tensor.
    /// Weight key points to a `[outC, inC, kT, kF]` (PyTorch) weight.
    func conv2dLayer(
        _ x: [Float], B: Int, C: Int, T: Int, F: Int,
        weightKey: String, bias: [Float]?, fstride: Int, lookahead: Int
    ) throws -> [Float] {
        guard let wOHWI = conv2dWeightsOHWI[weightKey],
            let shape = conv2dShapes[weightKey]
        else {
            // Fallback: load from weights dict.
            guard let wRaw = weights[weightKey],
                let rawShape = weights.shapes[weightKey], rawShape.count == 4
            else { throw DeepFilterNetError.missingWeightKey(weightKey) }
            let (outC, inC, kT, kF) = (rawShape[0], rawShape[1], rawShape[2], rawShape[3])
            // Transpose on the fly.
            var wOHWI2 = [Float](repeating: 0, count: outC * kT * kF * inC)
            for o in 0 ..< outC {
                for t in 0 ..< kT {
                    for f in 0 ..< kF {
                        for i in 0 ..< inC {
                            wOHWI2[o * kT * kF * inC + t * kF * inC + f * inC + i] =
                                wRaw[o * inC * kT * kF + i * kT * kF + t * kF + f]
                        }
                    }
                }
            }
            return conv2dCompute(
                x, B: B, C: C, T: T, F: F,
                weight: wOHWI2, outC: outC, kT: kT, kF: kF,
                bias: bias, fstride: fstride, lookahead: lookahead)
        }
        return conv2dCompute(
            x, B: B, C: C, T: T, F: F,
            weight: wOHWI, outC: shape.outC, kT: shape.kT, kF: shape.kF,
            bias: bias, fstride: fstride, lookahead: lookahead)
    }

    /// Core convolution computation.  Input/output in `[B, C, T, F]`.
    func conv2dCompute(
        _ x: [Float], B: Int, C: Int, T: Int, F: Int,
        weight: [Float], outC: Int, kT: Int, kF: Int,
        bias: [Float]?, fstride: Int, lookahead: Int
    ) -> [Float] {
        // True inPerGroup from weight shape.
        let inCPerGroup = weight.count / (outC * kT * kF)
        let groups = max(1, C / max(1, inCPerGroup))

        // Causal padding: kT-1 left (minus lookahead), lookahead right.
        let rawLeft = kT - 1 - lookahead
        let timePadLeft = max(0, rawLeft)
        let timePadRight = max(0, lookahead)
        let freqPad = kF / 2

        // Time crop for negative rawLeft (shouldn't happen for lookahead ≤ kT-1).
        let timeCrop = max(0, -rawLeft)
        let effectiveT = T - timeCrop
        let outT = max(0, effectiveT + timePadLeft + timePadRight - kT + 1)
        let outF = max(0, (F + 2 * freqPad - kF) / fstride + 1)

        // `nonisolated(unsafe)`: each `concurrentPerform` iteration writes a
        // disjoint `[b*outC*outT*outF + oc*outT*outF + ot*outF ..< +outF)`
        // slice of `out`; no overlap.
        nonisolated(unsafe) var out = [Float](repeating: 0, count: B * outC * outT * outF)

        // For each output position compute convolution over kernel window.
        // This is O(B·outC·outT·outF·kT·kF·inCPerGroup) — acceptable for the
        // small spatial dims of DeepFilterNet (nbErb=32, nbDf=96, outT=nT≤~200).
        let outCPerGroup = outC / groups
        DispatchQueue.concurrentPerform(iterations: B * outC * outT) { linearIdx in
            let b = linearIdx / (outC * outT)
            let rem = linearIdx % (outC * outT)
            let oc = rem / outT
            let ot = rem % outT
            let g = oc / outCPerGroup

            for of in 0 ..< outF {
                var acc: Float = 0
                for kt in 0 ..< kT {
                    for kf in 0 ..< kF {
                        let srcT = ot - timePadLeft + kt + timeCrop
                        let srcFRaw = of * fstride - freqPad + kf
                        guard srcT >= 0, srcT < T, srcFRaw >= 0, srcFRaw < F else { continue }
                        for ic in 0 ..< inCPerGroup {
                            let inC_idx = g * inCPerGroup + ic
                            let xVal = x[b * C * T * F + inC_idx * T * F + srcT * F + srcFRaw]
                            let wVal = weight[
                                oc * kT * kF * inCPerGroup + kt * kF * inCPerGroup + kf
                                    * inCPerGroup + ic]
                            acc += xVal * wVal
                        }
                    }
                }
                if let bias, oc < bias.count { acc += bias[oc] }
                out[b * outC * outT * outF + oc * outT * outF + ot * outF + of] = acc
            }
        }
        return out
    }

    // MARK: ConvTranspose2d (for decoder upsampling)

    /// Transposed conv used in the ERB decoder for freq upsampling.
    func convTranspose2dLayer(
        _ x: [Float], B: Int, C: Int, T: Int, F: Int,
        weightKey: String, fstride: Int
    ) throws -> [Float] {
        guard let wRaw = weights[weightKey],
            let rawShape = weights.shapes[weightKey], rawShape.count == 4
        else { throw DeepFilterNetError.missingWeightKey(weightKey) }
        let (inC, outCW, kT, kF) = (rawShape[0], rawShape[1], rawShape[2], rawShape[3])
        // groups = inC / outCW typically.
        let groups = max(1, inC / max(1, outCW))
        let inCPerGroup = max(1, inC / groups)
        let outCPerGroup = outCW
        let outC = groups * outCPerGroup

        let paddingT = kT - 1
        let paddingF = kF / 2
        let outT = T  // stride=1 for time
        let outF = (F - 1) * fstride - 2 * paddingF + kF + paddingF

        var out = [Float](repeating: 0, count: B * outC * outT * outF)

        for b in 0 ..< B {
            for g in 0 ..< groups {
                for ic in 0 ..< inCPerGroup {
                    let inC_idx = g * inCPerGroup + ic
                    for oc in 0 ..< outCPerGroup {
                        let outC_idx = g * outCPerGroup + oc
                        for it in 0 ..< T {
                            for inf2 in 0 ..< F {
                                let xVal = x[b * C * T * F + inC_idx * T * F + it * F + inf2]
                                for kt in 0 ..< kT {
                                    // Simplified: just add contribution.
                                    let ot2 = it + (kT - 1) - kt
                                    guard ot2 >= 0, ot2 < outT else { continue }
                                    for kf in 0 ..< kF {
                                        let of2 = inf2 * fstride - paddingF + kf
                                        guard of2 >= 0, of2 < outF else { continue }
                                        // weight layout: [inC, outCPerGroup, kT, kF]
                                        let wIdx =
                                            inC_idx * outCPerGroup * kT * kF
                                            + oc * kT * kF + kt * kF + kf
                                        if wIdx < wRaw.count {
                                            let outIdx =
                                                b * outC * outT * outF
                                                + outC_idx * outT * outF + ot2 * outF + of2
                                            out[outIdx] += xVal * wRaw[wIdx]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: BatchNorm

    func applyBatchNorm(_ x: [Float], B: Int, C: Int, T: Int, F: Int, prefix: String) throws
        -> [Float]
    {
        if let bn = bnMap[prefix] {
            return bn.apply(x, batch: B, channels: C, height: T, width: F)
        }
        // Build on the fly.
        guard let gamma = weights["\(prefix).weight"],
            let beta = weights["\(prefix).bias"],
            let mean = weights["\(prefix).running_mean"],
            let variance = weights["\(prefix).running_var"]
        else { throw DeepFilterNetError.missingWeightKey("\(prefix).weight") }
        let scale = zip(gamma, variance).map { g, v in g / sqrtf(v + 1e-5) }
        let bias = zip(zip(beta, mean), scale).map { pair, s in pair.0 - pair.1 * s }
        let bn = DFNBatchNorm(scale: scale, bias: bias)
        return bn.apply(x, batch: B, channels: C, height: T, width: F)
    }

    // MARK: Linear

    func linear(_ x: [Float], T: Int, weight: [Float], bias: [Float]?) -> [Float] {
        guard let shape = findLinearShape(weight: weight, inDim: x.count / T) else { return x }
        let (inDim, outDim) = shape
        var out = [Float](repeating: 0, count: T * outDim)
        for t in 0 ..< T {
            for o in 0 ..< outDim {
                var acc: Float = 0
                for i in 0 ..< inDim {
                    acc += x[t * inDim + i] * weight[o * inDim + i]
                }
                if let bias, o < bias.count { acc += bias[o] }
                out[t * outDim + o] = acc
            }
        }
        return out
    }

    private func findLinearShape(weight: [Float], inDim: Int) -> (Int, Int)? {
        guard inDim > 0 else { return nil }
        let outDim = weight.count / inDim
        guard outDim * inDim == weight.count else { return nil }
        return (inDim, outDim)
    }

    // MARK: Grouped linear

    /// Grouped linear: splits the feature axis into groups, applies per-group matmul.
    func groupedLinear(
        _ x: [Float], T: Int, prefix: String?, weightKey: String?
    ) throws -> [Float] {
        let key = weightKey ?? "\(prefix!).0.weight"
        guard let w = weights[key] else {
            throw DeepFilterNetError.missingWeightKey(key)
        }
        guard let shape = weights.shapes[key], shape.count == 3 else {
            // Fallback: treat as plain linear [outDim, inDim].
            return linear(x, T: T, weight: w, bias: nil)
        }
        // Shape: [groups, inPerGroup, outPerGroup]
        let (groups, inPG, outPG) = (shape[0], shape[1], shape[2])
        var out = [Float](repeating: 0, count: T * groups * outPG)
        for t in 0 ..< T {
            for g in 0 ..< groups {
                for o in 0 ..< outPG {
                    var acc: Float = 0
                    for i in 0 ..< inPG {
                        let xIdx = t * (groups * inPG) + g * inPG + i
                        let wIdx = g * inPG * outPG + i * outPG + o
                        if xIdx < x.count, wIdx < w.count {
                            acc += x[xIdx] * w[wIdx]
                        }
                    }
                    out[t * groups * outPG + g * outPG + o] = acc
                }
            }
        }
        return out
    }

    func groupedLinearRelu(_ x: [Float], T: Int, prefix: String) throws -> [Float] {
        var y = try groupedLinear(x, T: T, prefix: prefix, weightKey: nil)
        for i in y.indices { y[i] = max(0, y[i]) }
        return y
    }

    /// Grouped linear + ReLU, addressed by an explicit weight key.
    /// Use this when the caller already has the full key (avoids the ".0.weight" suffix
    /// that the prefix-based variant appends automatically).
    func groupedLinearRelu(_ x: [Float], T: Int, weightKey: String) throws -> [Float] {
        var y = try groupedLinear(x, T: T, prefix: nil, weightKey: weightKey)
        for i in y.indices { y[i] = max(0, y[i]) }
        return y
    }

    // MARK: GRU

    /// PyTorch GRU layer — CPU recurrence over time steps.
    /// `x` is `[T, inDim]`, returns `[T, hiddenSize]`.
    func gruLayer(
        _ x: [Float], T: Int, inDim: Int,
        wihT: [Float], whhT: [Float], bih: [Float], bhh: [Float],
        hiddenSize: Int
    ) -> [Float] {
        let h3 = 3 * hiddenSize
        // Batch project input: gxAll[T, 3H] = x @ wihT + bih.
        var gxAll = [Float](repeating: 0, count: T * h3)
        // matmul: [T, inDim] x [inDim, h3] → [T, h3]
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(T), Int32(h3), Int32(inDim),
            1.0, x, Int32(inDim),
            wihT, Int32(h3),
            0.0, &gxAll, Int32(h3)
        )
        // Add bias.
        for t in 0 ..< T {
            for j in 0 ..< h3 { gxAll[t * h3 + j] += bih[j] }
        }

        var output = [Float](repeating: 0, count: T * hiddenSize)
        var state = [Float](repeating: 0, count: hiddenSize)
        var gh = [Float](repeating: 0, count: h3)

        for t in 0 ..< T {
            // gh = state @ whhT.
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                Int32(h3), Int32(hiddenSize),
                1.0, whhT, Int32(hiddenSize),
                state, 1, 0.0, &gh, 1
            )
            for j in 0 ..< h3 { gh[j] += bhh[j] }
            // GRU gates.
            for k in 0 ..< hiddenSize {
                let xr = gxAll[t * h3 + k]
                let xz = gxAll[t * h3 + hiddenSize + k]
                let xn = gxAll[t * h3 + 2 * hiddenSize + k]
                let hr = gh[k]
                let hz = gh[hiddenSize + k]
                let hn = gh[2 * hiddenSize + k]
                let r = 1.0 / (1.0 + expf(-(xr + hr)))
                let z = 1.0 / (1.0 + expf(-(xz + hz)))
                let n = tanhf(xn + r * hn)
                state[k] = (1.0 - z) * n + z * state[k]
            }
            output.replaceSubrange((t * hiddenSize) ..< ((t + 1) * hiddenSize), with: state)
        }
        return output
    }

    /// SqueezedGRU block: linearIn → ReLU → GRU layers → (optional linearOut → ReLU).
    func squeezedGRU(
        _ x: [Float], T: Int, prefix: String,
        hiddenSize: Int, linearOut: Bool
    ) throws -> [Float] {
        // Linear in.
        guard weights["\(prefix).linear_in.0.weight"] != nil else {
            throw DeepFilterNetError.missingWeightKey("\(prefix).linear_in.0.weight")
        }
        var y = try groupedLinear(x, T: T, prefix: nil, weightKey: "\(prefix).linear_in.0.weight")
        for i in y.indices { y[i] = max(0, y[i]) }  // ReLU

        // GRU layers.
        var layer = 0
        while true {
            let wihKey = "\(prefix).gru.weight_ih_l\(layer)"
            guard weights[wihKey] != nil else { break }
            let bihKey = "\(prefix).gru.bias_ih_l\(layer)"
            let bhhKey = "\(prefix).gru.bias_hh_l\(layer)"
            guard let bih = weights[bihKey], let bhh = weights[bhhKey] else {
                throw DeepFilterNetError.missingWeightKey(bihKey)
            }
            let curDim = y.count / T
            let wihTKey = "\(prefix).gru.weight_ih_l\(layer)"
            let whhTKey = "\(prefix).gru.weight_hh_l\(layer)"
            let wihT =
                gruWeightIHT[wihTKey]
                ?? (weights[wihTKey].map { transpose2D($0, rows: 3 * hiddenSize, cols: curDim) }
                    ?? [])
            let whhT =
                gruWeightHHT[whhTKey]
                ?? (weights[whhTKey].map { transpose2D($0, rows: 3 * hiddenSize, cols: hiddenSize) }
                    ?? [])
            y = gruLayer(
                y, T: T, inDim: curDim, wihT: wihT, whhT: whhT, bih: bih, bhh: bhh,
                hiddenSize: hiddenSize)
            layer += 1
        }

        // Optional linear out.
        if linearOut, weights["\(prefix).linear_out.0.weight"] != nil {
            y = try groupedLinear(y, T: T, prefix: nil, weightKey: "\(prefix).linear_out.0.weight")
            for i in y.indices { y[i] = max(0, y[i]) }  // ReLU
        }
        return y
    }

    /// Transpose a 2-D matrix stored row-major: `[rows, cols]` → `[cols, rows]`.
    func transpose2D(_ m: [Float], rows: Int, cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * cols)
        for r in 0 ..< rows {
            for c in 0 ..< cols {
                out[c * rows + r] = m[r * cols + c]
            }
        }
        return out
    }

    // MARK: Encoder conv block

    func encConv(
        _ x: [Float], B: Int, C: Int, T: Int, F: Int,
        prefix: String, main: Int, pointwise: Int?, bn: Int, fstride: Int
    ) throws -> [Float] {
        var y = try conv2dLayer(
            x, B: B, C: C, T: T, F: F,
            weightKey: "\(prefix).\(main).weight",
            bias: nil, fstride: fstride, lookahead: 0)
        let outDims = convShape(prefix: prefix, main: main)
        let outC = outDims.outC
        let outF = (fstride == 1) ? F : F / fstride
        if let pw = pointwise {
            y = try conv2dLayer(
                y, B: B, C: outC, T: T, F: outF,
                weightKey: "\(prefix).\(pw).weight",
                bias: nil, fstride: 1, lookahead: 0)
        }
        let pwC = pointwise != nil ? convShape(prefix: prefix, main: pointwise!).outC : outC
        y = try applyBatchNorm(y, B: B, C: pwC, T: T, F: outF, prefix: "\(prefix).\(bn)")
        for i in y.indices { y[i] = max(0, y[i]) }  // ReLU
        return y
    }

    // MARK: Decoder blocks

    func pathwayConv(_ x: [Float], B: Int, C: Int, T: Int, F: Int, prefix: String) throws -> [Float] {
        var y = try conv2dLayer(
            x, B: B, C: C, T: T, F: F,
            weightKey: "\(prefix).0.weight", bias: nil, fstride: 1, lookahead: 0)
        let outC = convShape(prefix: prefix, main: 0).outC
        y = try applyBatchNorm(y, B: B, C: outC, T: T, F: F, prefix: "\(prefix).1")
        for i in y.indices { y[i] = max(0, y[i]) }  // ReLU
        return y
    }

    func regularBlock(_ x: [Float], B: Int, C: Int, T: Int, F: Int, prefix: String) throws
        -> [Float]
    {
        var y = try conv2dLayer(
            x, B: B, C: C, T: T, F: F,
            weightKey: "\(prefix).0.weight", bias: nil, fstride: 1, lookahead: 0)
        let c1 = convShape(prefix: prefix, main: 0).outC
        y = try conv2dLayer(
            y, B: B, C: c1, T: T, F: F,
            weightKey: "\(prefix).1.weight", bias: nil, fstride: 1, lookahead: 0)
        let c2 = convShape(prefix: prefix, main: 1).outC
        return try applyBatchNorm(y, B: B, C: c2, T: T, F: F, prefix: "\(prefix).2")
    }

    func transposeBlock(_ x: [Float], B: Int, C: Int, T: Int, F: Int, prefix: String, fstride: Int)
        throws -> [Float]
    {
        var y = try convTranspose2dLayer(
            x, B: B, C: C, T: T, F: F,
            weightKey: "\(prefix).0.weight", fstride: fstride)
        let outF = F * fstride
        // Infer channels from transpose-conv output.
        let outC: Int
        if let shape = weights.shapes["\(prefix).0.weight"], shape.count == 4 {
            outC = shape[1]  // [inC, outCPerGroup, kT, kF]
        } else {
            outC = C
        }
        y = try conv2dLayer(
            y, B: B, C: outC, T: T, F: outF,
            weightKey: "\(prefix).1.weight", bias: nil, fstride: 1, lookahead: 0)
        let c2 = convShape(prefix: prefix, main: 1).outC
        return try applyBatchNorm(y, B: B, C: c2, T: T, F: outF, prefix: "\(prefix).2")
    }

    func outputConv(_ x: [Float], B: Int, C: Int, T: Int, F: Int, prefix: String) throws -> [Float] {
        let y = try conv2dLayer(
            x, B: B, C: C, T: T, F: F,
            weightKey: "\(prefix).0.weight", bias: nil, fstride: 1, lookahead: 0)
        let outC = convShape(prefix: prefix, main: 0).outC
        return try applyBatchNorm(y, B: B, C: outC, T: T, F: F, prefix: "\(prefix).1")
    }
}
