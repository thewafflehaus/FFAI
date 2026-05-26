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
// FishS1DACConfig — configuration types for the FishS1DAC neural audio codec.
//
// Mirrors `FishS1DACBuildConfig` from the reference MLX implementation at
// mlx-audio-swift/Sources/MLXAudioCodecs/FishS1DAC/FishS1DACConfig.swift.
//
// Used by FishS1DAC.swift to reconstruct the model architecture from the
// codec's `config.json` when loading a pretrained checkpoint.

import Foundation

/// Full FishS1DAC configuration, decoded from the codec `config.json`.
/// All fields default to the standard FishAudio S2 preset so a missing or
/// sparse config still produces a usable model.
public struct FishS1DACConfig: Codable, Sendable {
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var latentDim: Int
    public var decoderDim: Int
    public var decoderRates: [Int]
    public var nCodebooks: Int
    public var codebookSize: Int
    public var codebookDim: Int
    public var semanticCodebookSize: Int
    public var downsampleFactor: [Int]
    public var downsampleDims: [Int]?
    public var sampleRate: Int

    // Quantizer transformer (pre/post module)
    public var quantizerTransformerLayers: Int
    public var quantizerTransformerHeads: Int
    public var quantizerTransformerDim: Int
    public var quantizerTransformerIntermediateSize: Int
    public var quantizerTransformerHeadDim: Int
    public var quantizerWindowSize: Int

    enum CodingKeys: String, CodingKey {
        case encoderDim                          = "encoder_dim"
        case encoderRates                        = "encoder_rates"
        case latentDim                           = "latent_dim"
        case decoderDim                          = "decoder_dim"
        case decoderRates                        = "decoder_rates"
        case nCodebooks                          = "n_codebooks"
        case codebookSize                        = "codebook_size"
        case codebookDim                         = "codebook_dim"
        case semanticCodebookSize                = "semantic_codebook_size"
        case downsampleFactor                    = "downsample_factor"
        case downsampleDims                      = "downsample_dims"
        case sampleRate                          = "sample_rate"
        case quantizerTransformerLayers          = "quantizer_transformer_layers"
        case quantizerTransformerHeads           = "quantizer_transformer_heads"
        case quantizerTransformerDim             = "quantizer_transformer_dim"
        case quantizerTransformerIntermediateSize = "quantizer_transformer_intermediate_size"
        case quantizerTransformerHeadDim         = "quantizer_transformer_head_dim"
        case quantizerWindowSize                 = "quantizer_window_size"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        encoderDim                          = try c.decodeIfPresent(Int.self,    forKey: .encoderDim)                          ?? 64
        encoderRates                        = try c.decodeIfPresent([Int].self,  forKey: .encoderRates)                        ?? [2, 4, 8, 8]
        latentDim                           = try c.decodeIfPresent(Int.self,    forKey: .latentDim)                           ?? 1024
        decoderDim                          = try c.decodeIfPresent(Int.self,    forKey: .decoderDim)                          ?? 1536
        decoderRates                        = try c.decodeIfPresent([Int].self,  forKey: .decoderRates)                        ?? [8, 8, 4, 2]
        nCodebooks                          = try c.decodeIfPresent(Int.self,    forKey: .nCodebooks)                          ?? 9
        codebookSize                        = try c.decodeIfPresent(Int.self,    forKey: .codebookSize)                        ?? 1024
        codebookDim                         = try c.decodeIfPresent(Int.self,    forKey: .codebookDim)                         ?? 8
        semanticCodebookSize                = try c.decodeIfPresent(Int.self,    forKey: .semanticCodebookSize)                ?? 4096
        downsampleFactor                    = try c.decodeIfPresent([Int].self,  forKey: .downsampleFactor)                    ?? [2, 2]
        downsampleDims                      = try c.decodeIfPresent([Int]?.self, forKey: .downsampleDims)                      ?? nil
        sampleRate                          = try c.decodeIfPresent(Int.self,    forKey: .sampleRate)                          ?? 44_100
        quantizerTransformerLayers          = try c.decodeIfPresent(Int.self,    forKey: .quantizerTransformerLayers)          ?? 8
        quantizerTransformerHeads           = try c.decodeIfPresent(Int.self,    forKey: .quantizerTransformerHeads)           ?? 16
        quantizerTransformerDim             = try c.decodeIfPresent(Int.self,    forKey: .quantizerTransformerDim)             ?? 1024
        quantizerTransformerIntermediateSize = try c.decodeIfPresent(Int.self,   forKey: .quantizerTransformerIntermediateSize) ?? 3072
        quantizerTransformerHeadDim         = try c.decodeIfPresent(Int.self,    forKey: .quantizerTransformerHeadDim)         ?? 64
        quantizerWindowSize                 = try c.decodeIfPresent(Int.self,    forKey: .quantizerWindowSize)                 ?? 128
    }

    /// Total temporal downsampling factor (encoder strides product).
    public var hopLength: Int { encoderRates.reduce(1, *) }

    /// Total upsample factor applied by the quantizer upsample stages.
    public var quantizerUpsampleFactor: Int { downsampleFactor.reduce(1, *) }

    /// Frame length used for padding during encode (4× hopLength in the reference).
    public var frameLength: Int { hopLength * 4 }
}

public enum FishS1DACError: Error, CustomStringConvertible {
    case missingWeights(String)
    case configNotFound(String)
    case shapeMismatch(String)

    public var description: String {
        switch self {
        case .missingWeights(let s):  return "FishS1DAC: missing weights — \(s)"
        case .configNotFound(let s):  return "FishS1DAC: config not found — \(s)"
        case .shapeMismatch(let s):   return "FishS1DAC: shape mismatch — \(s)"
        }
    }
}
