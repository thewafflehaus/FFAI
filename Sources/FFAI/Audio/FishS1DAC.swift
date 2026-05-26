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
// FishS1DAC — FishAudio Stage-1 neural audio codec.
//
// Port of the MLX reference implementation at:
//   mlx-audio-swift/Sources/MLXAudioCodecs/FishS1DAC/
//
// Architecture (decode path only — this port targets TTS waveform output):
//   codes [[Int32]] (numCodebooks × T)
//     → FishS1DACDownsampleRVQ.decode
//         ↳ semantic quantizer (1 codebook, semanticCodebookSize)
//         ↳ residual quantizer (nCodebooks, codebookSize)
//         ↳ post-module transformer (window-limited, causal)
//         ↳ upsample stages (Conv1d-Transposed + ConvNeXt)
//     → FishS1DACDecoder (Snake+WNConvTranspose blocks → Tanh)
//     → Tensor [1, 1, L]   (mono waveform, f32)
//
// CPU fallback: all convolutions run via `AudioMath` (CPU). A metaltile
// kernel for dilated depthwise Conv1d / ConvTranspose1d would accelerate
// the decoder blocks. See FishS1DACQuantization.swift for the TODO.
//
// Public surface:
//   FishS1DAC.load(from:)           load from an HF snapshot directory
//   codec.decode(codes:)            [[Int32]] → Tensor [1, 1, L]
//   codec.sampleRate                Int (44_100 by default)
//   codec.hopLength                 Int (product of encoder strides)

import Foundation

// MARK: - Decoder blocks

/// FishS1 residual unit: Snake → WNConv(k=7, dilated) → Snake → WNConv(k=1),
/// then add residual (centre-cropped to output length).
/// Reuses `SNACResidualUnit` so the math path is shared.
private func fishS1DACResidualUnit(weights w: FishS1DACWeights,
                                    prefix: String,
                                    dilation: Int) throws -> SNACResidualUnit {
    // block.0 = Snake, block.1 = WNConv(k=7, dil), block.2 = Snake, block.3 = WNConv(k=1)
    let pad = ((7 - 1) * dilation) / 2
    let alpha1 = try w.floats("\(prefix).block.0.alpha")
    let conv1  = try w.wnConv1d(prefix: "\(prefix).block.1",
                                 stride: 1, padding: pad, dilation: dilation, groups: 1)
    let alpha2 = try w.floats("\(prefix).block.2.alpha")
    let conv2  = try w.wnConv1d(prefix: "\(prefix).block.3",
                                 stride: 1, padding: 0, dilation: 1, groups: 1)
    return SNACResidualUnit(alpha1: alpha1, conv1: conv1,
                            alpha2: alpha2, conv2: conv2)
}

/// FishS1 decoder block: Snake → WNConvTranspose1d (upsample by `stride`) →
/// three dilated residual units.
private struct FishS1DACDecoderBlock {
    let snakeAlpha: [Float]
    let convUp: SNACWNConvTranspose1d
    let residuals: [SNACResidualUnit]

    init(weights w: FishS1DACWeights, prefix: String, stride: Int) throws {
        // block.0 = Snake, block.1 = WNConvTranspose1d(k=2*stride, stride, pad=ceil(stride/2))
        self.snakeAlpha = try w.floats("\(prefix).block.0.alpha")
        let pad = Int(ceil(Double(stride) / 2.0))
        self.convUp = try w.wnConvTranspose1d(
            prefix: "\(prefix).block.1",
            stride: stride, padding: pad, outputPadding: 1, groups: 1)
        // block.{2,3,4} = ResidualUnit(dilation 1,3,9)
        var res: [SNACResidualUnit] = []
        for (i, dil) in [1, 3, 9].enumerated() {
            res.append(try fishS1DACResidualUnit(
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

// MARK: - FishS1DAC

/// FishAudio Stage-1 DAC neural audio codec. Decodes integer VQ code
/// streams produced by a FishSpeech model into a mono 44.1 kHz waveform.
///
/// Loading is checkpoint-driven: call `FishS1DAC.load(from:)` with a
/// Hugging Face snapshot directory that contains `codec.json` (or
/// `config.json`) and one of:
///   - `codec.safetensors`
///   - `model.safetensors`
///   - `pytorch_model.safetensors`
public final class FishS1DAC: @unchecked Sendable {
    public let config: FishS1DACConfig

    // Quantizer (semantic + residual + postModule + upsample)
    private let quantizer: FishS1DACDownsampleRVQ

    // Decoder layers
    private let decoderConvIn: SNACWNConv1d
    private let decoderBlocks: [FishS1DACDecoderBlock]
    private let decoderSnakeAlpha: [Float]
    private let decoderConvOut: SNACWNConv1d

    // MARK: - Load

    /// Load a FishS1DAC codec from an HF snapshot directory.
    ///
    /// The directory is expected to contain:
    ///   - `codec.json` or `config.json` — model configuration
    ///   - `codec.safetensors`, `model.safetensors`, or
    ///     `pytorch_model.safetensors` — weights
    ///
    /// The codec weights may live in a sub-folder (e.g. `codec/`) or in
    /// the same snapshot directory as the main TTS weights. Both locations
    /// are probed, parent directory first.
    public static func load(from directory: URL) throws -> FishS1DAC {
        // Probe parent directory then common sub-folder names.
        let candidates: [URL] = [
            directory,
            directory.appendingPathComponent("codec"),
            directory.appendingPathComponent("vocoder"),
        ]

        var lastError: Error = FishS1DACError.missingWeights(
            "No safetensors weights found under \(directory.path)")

        for candidate in candidates {
            guard hasWeights(in: candidate) else { continue }
            do {
                let config = try loadConfig(from: candidate)
                let bundle = try SafeTensorsBundle(directory: candidate)
                return try FishS1DAC(config: config, bundle: bundle)
            } catch {
                lastError = error
            }
        }
        throw lastError
    }

    /// Returns `true` if `dir` contains at least one known weights file name.
    private static func hasWeights(in dir: URL) -> Bool {
        let fm = FileManager.default
        for name in ["codec.safetensors", "model.safetensors", "pytorch_model.safetensors"] {
            if fm.fileExists(atPath: dir.appendingPathComponent(name).path) { return true }
        }
        return false
    }

    /// Load configuration from the codec directory (tries `codec.json` first,
    /// then `config.json`).
    private static func loadConfig(from dir: URL) throws -> FishS1DACConfig {
        let fm = FileManager.default
        for name in ["codec.json", "config.json"] {
            let url = dir.appendingPathComponent(name)
            if fm.fileExists(atPath: url.path) {
                let data = try Data(contentsOf: url)
                let decoder = JSONDecoder()
                if let cfg = try? decoder.decode(FishS1DACConfig.self, from: data) {
                    return cfg
                }
            }
        }
        // Fall back to defaults when no readable config is present.
        let data = Data("{}".utf8)
        return try JSONDecoder().decode(FishS1DACConfig.self, from: data)
    }

    // MARK: - Init

    init(config: FishS1DACConfig, bundle: SafeTensorsBundle) throws {
        self.config = config
        let w = FishS1DACWeights(bundle: bundle)

        // ─── Quantizer ────────────────────────────────────────────────────
        self.quantizer = try FishS1DACDownsampleRVQ(weights: w, config: config)

        // ─── Decoder ──────────────────────────────────────────────────────
        // model.0 — WNConv1d(in=latentDim, k=7, pad=3)
        self.decoderConvIn = try w.wnConv1d(
            prefix: "decoder.model.0",
            stride: 1, padding: 3, dilation: 1, groups: 1)

        var decBlocks: [FishS1DACDecoderBlock] = []
        var decIdx = 1
        for stride in config.decoderRates {
            decBlocks.append(try FishS1DACDecoderBlock(
                weights: w, prefix: "decoder.model.\(decIdx)", stride: stride))
            decIdx += 1
        }
        self.decoderBlocks = decBlocks

        // model.{N} — Snake; model.{N+1} — WNConv1d(out=1, k=7, pad=3); model.{N+2} — Tanh
        self.decoderSnakeAlpha = try w.floats("decoder.model.\(decIdx).alpha")
        decIdx += 1
        self.decoderConvOut = try w.wnConv1d(
            prefix: "decoder.model.\(decIdx)",
            stride: 1, padding: 3, dilation: 1, groups: 1)
    }

    // MARK: - Public interface

    /// Audio sample rate of the codec (Hz).
    public var sampleRate: Int { config.sampleRate }

    /// Total temporal downsampling factor (encoder strides × 1).
    /// Samples per code frame = hopLength.
    public var hopLength: Int { config.hopLength }

    /// Number of codebooks (semantic + residual).
    public var numCodebooks: Int { config.nCodebooks + 1 }

    // MARK: - Decode

    /// Decode integer VQ code streams into a mono waveform.
    ///
    /// - codes: `[[Int32]]` shaped `[numCodebooks, T]` where
    ///   `codes[0]` are semantic tokens and `codes[1…]` are residual tokens.
    ///
    /// - Returns: Tensor of shape `[1, 1, L]` containing f32 mono audio,
    ///   where `L ≈ T * hopLength`.
    public func decode(codes: [[Int32]]) throws -> Tensor {
        guard !codes.isEmpty else {
            throw FishS1DACError.shapeMismatch("empty code list")
        }

        // Step 1: decode VQ codes → quantized latent [1, latentDim, T']
        let (zQ, zShape) = try quantizer.decode(codes: codes)
        guard !zQ.isEmpty else {
            throw FishS1DACError.shapeMismatch("quantizer returned empty latent")
        }

        // Step 2: decoder network → waveform
        let (audio, audioShape) = runDecoder(zQ, shape: zShape)

        let out = Tensor.empty(shape: audioShape, dtype: .f32)
        out.copyIn(from: audio)
        return out
    }

    // MARK: - Decoder forward

    private func runDecoder(_ z: [Float],
                            shape: [Int]) -> (data: [Float], shape: [Int]) {
        // Input conv: [1, latentDim, T] → [1, decoderDim, T]
        var (d, s) = decoderConvIn(z, shape: shape)

        // Decoder blocks (each upsamples by its stride)
        for block in decoderBlocks { (d, s) = block(d, shape: s) }

        // Tail: Snake → WNConv1d(out=1) → Tanh
        d = AudioMath.snake(d, shape: s, alpha: decoderSnakeAlpha)
        (d, s) = decoderConvOut(d, shape: s)
        d = AudioMath.tanhAll(d)
        return (d, s)
    }
}

