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
// PocketTTS — Kyutai's streaming text-to-speech family.
//
// PocketTTS is a flow-matching TTS pipeline:
//
//   text ──sentencepiece tokenizer──▶ token embeddings
//        ──streaming transformer (LUT conditioner)──▶ conditioning
//   conditioning + speaker embedding ──flow LM──▶ latent frames
//   latent frames ──Mimi codec decoder (SEANet + quantizer)──▶ 24 kHz waveform
//
// The flow LM uses a latent-space diffusion approach (LSD decode):
//   • A `SimpleMLPAdaLN` flow net maps noise + condition → predicted flow
//   • Iterative Euler integration over `lsd_decode_steps` steps
//   • EOS detection via a small linear head on the transformer output
//
// Architecture:
//   • Flow LM transformer: 6 layers, d_model=1024, 16 heads, 6144 FF dim
//   • Flow net: 512 dim, 6 depth residual MLP
//   • Mimi codec: SEANet encoder/decoder (ratios=[6,5,4]), 2-layer transformer,
//     quantizer dimension=32 → output 512
//   • Sample rate: 24 kHz, frame rate: 12.5 fps
//   • model_type: "pocket_tts"
//
// ## Scope note — STAGED PORT
//
// Stage 1 ships: config decoding, `AudioModelRegistry` detection,
// checkpoint weight-bundle retention, and the `PocketTTSModel` scaffold.
// The full synthesis pipeline — flow LM autoregressive decoding, sentencepiece
// tokenization, and Mimi codec decode — is a follow-on stage. Until it
// lands, `synthesize` throws `PocketTTSError.synthesisNotWired`.
//
// Reference implementation:
//   ~/Development/personal/ai/mlx-audio-swift/Sources/MLXAudioTTS/Models/PocketTTS/
// Checkpoint: mlx-community/pocket-tts

import Foundation

// ─── Errors ──────────────────────────────────────────────────────────

public enum PocketTTSError: Error, CustomStringConvertible {
    /// Full synthesis pipeline not wired yet. Config + detection + weight
    /// loading land first; the flow LM autoregressive decode, sentencepiece
    /// tokenization, and Mimi codec decode are follow-on stages.
    case synthesisNotWired
    /// A required config field is missing from `config.json`.
    case missingConfig(String)
    /// Voice embedding or audio prompt could not be resolved.
    case missingVoice(String)

    public var description: String {
        switch self {
        case .synthesisNotWired:
            return "PocketTTS: the flow LM autoregressive decode, sentencepiece "
                + "tokenization, and Mimi codec decoder are not yet wired in "
                + "this build. Stage 1 ships config decoding + registry detection. "
                + "Follow-on stages will wire the full synthesis pipeline "
                + "(flow LM forward + Mimi codec decode → waveform)."
        case .missingConfig(let field):
            return "PocketTTS: required config field missing: \(field)"
        case .missingVoice(let name):
            return "PocketTTS: voice embedding not found: \(name)"
        }
    }
}

// ─── Configuration ───────────────────────────────────────────────────

// MARK: - Flow LM sub-configs

/// Flow network (SimpleMLPAdaLN) hyper-parameters.
/// Maps to `flow` in `flow_lm` in `config.json`.
public struct PocketTTSFlowConfig: Sendable {
    /// Hidden dimension of the flow MLP.
    public let dim: Int
    /// Number of residual blocks in the flow MLP.
    public let depth: Int

    public static func from(_ raw: [String: Any]) -> PocketTTSFlowConfig {
        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }
        return PocketTTSFlowConfig(dim: i("dim", 512), depth: i("depth", 6))
    }
}

/// Streaming transformer hyper-parameters nested under `flow_lm.transformer`.
public struct PocketTTSTransformerConfig: Sendable {
    /// Multiplier for feed-forward hidden dimension (ff_dim = hidden_scale * d_model).
    public let hiddenScale: Int
    /// Max period for rotary positional embedding.
    public let maxPeriod: Double
    /// Model (hidden) dimension.
    public let dModel: Int
    /// Number of attention heads.
    public let numHeads: Int
    /// Number of transformer layers.
    public let numLayers: Int

    /// Feed-forward dimension derived from hidden_scale × d_model.
    public var dimFeedforward: Int { hiddenScale * dModel }

    public static func from(_ raw: [String: Any]) -> PocketTTSTransformerConfig {
        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }
        func d(_ k: String, _ def: Double) -> Double {
            if let v = raw[k] as? Double { return v }
            if let v = raw[k] as? Int { return Double(v) }
            return def
        }
        return PocketTTSTransformerConfig(
            hiddenScale: i("hidden_scale", 4),
            maxPeriod: d("max_period", 10_000),
            dModel: i("d_model", 1_024),
            numHeads: i("num_heads", 16),
            numLayers: i("num_layers", 6)
        )
    }
}

/// Lookup-table text conditioner config (`flow_lm.lookup_table`).
/// Describes the sentencepiece tokenizer + embedding dimensions.
public struct PocketTTSLookupTableConfig: Sendable {
    /// Embedding dimension (before optional projection to d_model).
    public let dim: Int
    /// Number of sentencepiece bins (vocabulary size).
    public let nBins: Int
    /// Tokenizer type identifier (always `"sentencepiece"`).
    public let tokenizer: String
    /// Path hint for the tokenizer model file (may be an HF URI).
    public let tokenizerPath: String

    public static func from(_ raw: [String: Any]) -> PocketTTSLookupTableConfig {
        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }
        return PocketTTSLookupTableConfig(
            dim: i("dim", 1_024),
            nBins: i("n_bins", 4_000),
            tokenizer: raw["tokenizer"] as? String ?? "sentencepiece",
            tokenizerPath: raw["tokenizer_path"] as? String ?? ""
        )
    }
}

/// Combined flow LM config (`flow_lm` block in `config.json`).
public struct PocketTTSFlowLMConfig: Sendable {
    public let dtype: String?
    public let flow: PocketTTSFlowConfig
    public let transformer: PocketTTSTransformerConfig
    public let lookupTable: PocketTTSLookupTableConfig
    public let weightsPath: String?

    public static func from(_ raw: [String: Any]) -> PocketTTSFlowLMConfig {
        let flowRaw = raw["flow"] as? [String: Any] ?? [:]
        let xfRaw = raw["transformer"] as? [String: Any] ?? [:]
        let lutRaw = raw["lookup_table"] as? [String: Any] ?? [:]
        return PocketTTSFlowLMConfig(
            dtype: raw["dtype"] as? String,
            flow: PocketTTSFlowConfig.from(flowRaw),
            transformer: PocketTTSTransformerConfig.from(xfRaw),
            lookupTable: PocketTTSLookupTableConfig.from(lutRaw),
            weightsPath: raw["weights_path"] as? String
        )
    }
}

// MARK: - Mimi codec sub-configs

/// SEANet encoder/decoder config (`mimi.seanet`).
public struct PocketTTSSeanetConfig: Sendable {
    public let dimension: Int
    public let channels: Int
    public let nFilters: Int
    public let nResidualLayers: Int
    /// Temporal downsampling factors (encoder strides).
    public let ratios: [Int]
    public let kernelSize: Int
    public let residualKernelSize: Int
    public let lastKernelSize: Int
    public let dilationBase: Int
    public let padMode: String
    public let compress: Int

    /// Total temporal downsampling factor (product of ratios).
    public var hopLength: Int { ratios.reduce(1, *) }

    public static func from(_ raw: [String: Any]) -> PocketTTSSeanetConfig {
        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }
        let ratios = raw["ratios"] as? [Int] ?? [6, 5, 4]
        return PocketTTSSeanetConfig(
            dimension: i("dimension", 512),
            channels: i("channels", 1),
            nFilters: i("n_filters", 64),
            nResidualLayers: i("n_residual_layers", 1),
            ratios: ratios,
            kernelSize: i("kernel_size", 7),
            residualKernelSize: i("residual_kernel_size", 3),
            lastKernelSize: i("last_kernel_size", 3),
            dilationBase: i("dilation_base", 2),
            padMode: raw["pad_mode"] as? String ?? "constant",
            compress: i("compress", 2)
        )
    }
}

/// Mimi transformer config (`mimi.transformer`).
public struct PocketTTSMimiTransformerConfig: Sendable {
    public let dModel: Int
    public let inputDimension: Int
    public let outputDimensions: [Int]
    public let numHeads: Int
    public let numLayers: Int
    public let layerScale: Double
    public let context: Int
    public let dimFeedforward: Int
    public let maxPeriod: Double

    public static func from(_ raw: [String: Any]) -> PocketTTSMimiTransformerConfig {
        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }
        func d(_ k: String, _ def: Double) -> Double {
            if let v = raw[k] as? Double { return v }
            if let v = raw[k] as? Int { return Double(v) }
            return def
        }
        let outDims = raw["output_dimensions"] as? [Int] ?? [512]
        return PocketTTSMimiTransformerConfig(
            dModel: i("d_model", 512),
            inputDimension: i("input_dimension", 512),
            outputDimensions: outDims,
            numHeads: i("num_heads", 8),
            numLayers: i("num_layers", 2),
            layerScale: d("layer_scale", 0.01),
            context: i("context", 250),
            dimFeedforward: i("dim_feedforward", 2_048),
            maxPeriod: d("max_period", 10_000)
        )
    }
}

/// Mimi quantizer config (`mimi.quantizer`).
public struct PocketTTSQuantizerConfig: Sendable {
    /// Codebook embedding dimension (32 for pocket-tts).
    public let dimension: Int
    /// Projected output dimension (512 for pocket-tts).
    public let outputDimension: Int

    public static func from(_ raw: [String: Any]) -> PocketTTSQuantizerConfig {
        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }
        return PocketTTSQuantizerConfig(
            dimension: i("dimension", 32),
            outputDimension: i("output_dimension", 512)
        )
    }
}

/// Mimi codec config (`mimi` block in `config.json`).
public struct PocketTTSMimiConfig: Sendable {
    public let dtype: String?
    public let sampleRate: Int
    public let channels: Int
    public let frameRate: Double
    public let seanet: PocketTTSSeanetConfig
    public let transformer: PocketTTSMimiTransformerConfig
    public let quantizer: PocketTTSQuantizerConfig
    public let weightsPath: String?

    /// Encoder frame rate before the latent downsample step.
    public var encoderFrameRate: Double {
        Double(sampleRate) / Double(seanet.hopLength)
    }

    public static func from(_ raw: [String: Any]) -> PocketTTSMimiConfig {
        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }
        func d(_ k: String, _ def: Double) -> Double {
            if let v = raw[k] as? Double { return v }
            if let v = raw[k] as? Int { return Double(v) }
            return def
        }
        let seanetRaw = raw["seanet"] as? [String: Any] ?? [:]
        let xfRaw = raw["transformer"] as? [String: Any] ?? [:]
        let qRaw = raw["quantizer"] as? [String: Any] ?? [:]
        return PocketTTSMimiConfig(
            dtype: raw["dtype"] as? String,
            sampleRate: i("sample_rate", 24_000),
            channels: i("channels", 1),
            frameRate: d("frame_rate", 12.5),
            seanet: PocketTTSSeanetConfig.from(seanetRaw),
            transformer: PocketTTSMimiTransformerConfig.from(xfRaw),
            quantizer: PocketTTSQuantizerConfig.from(qRaw),
            weightsPath: raw["weights_path"] as? String
        )
    }
}

// MARK: - Top-level config

/// PocketTTS model configuration. Decoded from the top-level `config.json`.
/// Matches `model_type = "pocket_tts"`.
public struct PocketTTSConfig: Sendable {
    /// Always `"pocket_tts"`.
    public let modelType: String
    /// Flow language model sub-config.
    public let flowLM: PocketTTSFlowLMConfig
    /// Mimi codec sub-config.
    public let mimi: PocketTTSMimiConfig
    /// Optional path for the combined weights file (may be null / HF URI).
    public let weightsPath: String?
    /// Optional path for the voice-cloning-free weights (may be null / HF URI).
    public let weightsPathWithoutVoiceCloning: String?

    /// Convenience: sample rate from the Mimi sub-config.
    public var sampleRate: Int { mimi.sampleRate }

    /// Decode from a top-level `ModelConfig`. Returns `nil` if `model_type`
    /// is not `"pocket_tts"` and neither `flow_lm` nor `mimi` are present.
    public static func from(_ config: ModelConfig) -> PocketTTSConfig? {
        let raw = config.raw
        // Require at least one of the canonical markers.
        guard
            config.modelType == "pocket_tts"
                || raw["flow_lm"] != nil
                || raw["mimi"] != nil
        else { return nil }

        let flowRaw = raw["flow_lm"] as? [String: Any] ?? [:]
        let mimiRaw = raw["mimi"] as? [String: Any] ?? [:]

        return PocketTTSConfig(
            modelType: config.modelType ?? "pocket_tts",
            flowLM: PocketTTSFlowLMConfig.from(flowRaw),
            mimi: PocketTTSMimiConfig.from(mimiRaw),
            weightsPath: raw["weights_path"] as? String,
            weightsPathWithoutVoiceCloning: raw["weights_path_without_voice_cloning"] as? String
        )
    }
}

// ─── Model ───────────────────────────────────────────────────────────

/// A loaded PocketTTS model. Owns the decoded config and the safetensors
/// weight bundle.
///
/// The full synthesis pipeline — sentencepiece tokenisation, flow LM
/// autoregressive decode, and Mimi codec decode — is a follow-on stage.
/// `synthesize` throws `PocketTTSError.synthesisNotWired` until it lands.
///
/// Architecture summary (from `mlx-community/pocket-tts`):
///   • Flow LM transformer: 6 layers, d_model=1024, 16 heads, 6144 ff
///   • Flow net: 512 dim, 6-depth SimpleMLPAdaLN
///   • Mimi: SEANet ratios=[6,5,4], 2-layer causal transformer, quantizer dim=32
///   • sample_rate=24000, frame_rate=12.5 fps
public final class PocketTTSModel: @unchecked Sendable {

    // Default generation hyper-parameters (match the reference impl).
    public static let defaultTemperature: Float = 0.7
    public static let defaultLsdDecodeSteps: Int = 1
    public static let defaultEosThreshold: Float = -4.0
    public static let defaultAudioPrompt: String = "alba"

    /// Decoded configuration.
    public let config: PocketTTSConfig

    /// Retained safetensors bundle — available for weight inspection.
    public let weights: SafeTensorsBundle

    /// Output waveform sample rate (24000 Hz for pocket-tts).
    public var sampleRate: Int { config.sampleRate }

    public init(config: PocketTTSConfig, weights: SafeTensorsBundle) {
        self.config = config
        self.weights = weights
    }

    /// Synthesise speech from `text`. Throws `PocketTTSError.synthesisNotWired`
    /// until the flow LM autoregressive decode, sentencepiece tokenisation,
    /// and Mimi codec decode stages land.
    ///
    /// Expected signature for follow-on stages:
    ///   - Stage 2: wire sentencepiece tokeniser + flow LM transformer forward.
    ///   - Stage 3: wire Mimi codec decode (SEANet decoder + quantizer projection).
    ///
    /// `device` is accepted for forward-compatibility but unused in stage 1.
    public func synthesize(
        text: String,
        voice: String = PocketTTSModel.defaultAudioPrompt,
        temperature: Float = PocketTTSModel.defaultTemperature,
        device: Device = .shared
    ) throws -> [Float] {
        _ = text
        _ = voice
        _ = temperature
        _ = device
        throw PocketTTSError.synthesisNotWired
    }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension PocketTTSModel {

    /// `model_type` values this family handles.
    public static let modelTypes: Set<String> = ["pocket_tts"]

    /// Whether a decoded `config.json` describes a PocketTTS checkpoint.
    ///
    /// Detection strategy:
    ///   1. `model_type == "pocket_tts"` — canonical marker.
    ///   2. Structural: top-level `flow_lm` block with a nested `transformer`
    ///      plus a `mimi` block with `frame_rate` — the structural fingerprint
    ///      of Kyutai's PocketTTS config layout.
    public static func handles(_ config: ModelConfig) -> Bool {
        // Canonical: model_type field.
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        // Structural: flow_lm + mimi co-presence.
        let hasFlowLM = config.raw["flow_lm"] is [String: Any]
        let hasMimi = config.raw["mimi"] is [String: Any]
        if hasFlowLM && hasMimi { return true }
        return false
    }

    /// Load a PocketTTS checkpoint from a resolved snapshot directory.
    ///
    /// Decodes `config.json`, retains the safetensors weight bundle, and
    /// returns a `PocketTTSModel`. The synthesis pipeline is not wired yet
    /// — `synthesize` will throw `PocketTTSError.synthesisNotWired`.
    public static func load(
        directory: URL,
        device: Device = .shared
    ) throws -> PocketTTSModel {
        let modelConfig = try ModelConfig.load(from: directory)
        guard let config = PocketTTSConfig.from(modelConfig) else {
            throw PocketTTSError.missingConfig("flow_lm or mimi")
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return PocketTTSModel(config: config, weights: bundle)
    }
}
