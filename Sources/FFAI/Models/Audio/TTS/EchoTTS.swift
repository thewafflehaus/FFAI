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
// EchoTTS — diffusion-based text-to-speech on a DiT backbone.
//
// EchoTTS generates high-quality speech via flow-matching diffusion
// over a latent space compressed from the Fish S1 DAC neural codec.
// The model architecture:
//
//   text (UTF-8 bytes) ──text encoder (Transformer)──▶ text KV cache
//   ref audio ──Fish S1 DAC + PCA──▶ 80-dim latent
//             ──speaker encoder (Transformer)──▶ speaker KV cache
//   noise ──DiT (joint attention, AdaLN, Euler CFG)──▶ audio latent
//         ──inverse PCA + Fish S1 DAC decoder──▶ 44.1 kHz waveform
//
// The DiT uses independent classifier-free guidance for both text and
// speaker conditioning, enabling zero-shot voice cloning.
//
// ## Scope note
//
// FFAI's current contribution is the **model scaffold**: config decoding,
// weight loading from safetensors, and the `handles` / load plumbing for
// the AudioModelRegistry. The full diffusion forward pass (the DiT
// transformer, Euler CFG sampler, and Fish S1 DAC codec) is a large
// operator set that requires batch matrix multiply, multi-head attention
// with softmax, and streaming audio decode — operators beyond what the
// current FFAI Ops set covers. The `synthesize` entry point documents this
// clearly: callers use `generatePlaceholder` to verify load + config, and
// the full synthesis path will be wired once the required GPU operator set
// (batched GEMM, SDPA) lands. See EchoTTSError.diffusionNotWired.

import Foundation
import Metal

// ─── Configuration ───────────────────────────────────────────────────────────

/// EchoDiT transformer hyper-parameters, nested under the `dit` key in
/// `config.json`.
public struct EchoDiTConfig: Sendable {
    // Main DiT
    /// Latent dimension from PCA-compressed Fish S1 DAC (80 for base).
    public let latentSize: Int
    /// Main transformer hidden size.
    public let modelSize: Int
    /// Number of main transformer layers.
    public let numLayers: Int
    /// Number of attention heads in the main transformer.
    public let numHeads: Int
    /// FFN intermediate size in the main transformer.
    public let intermediateSize: Int

    // Text encoder
    /// Text-side vocabulary size (256 for UTF-8 byte tokenizer).
    public let textVocabSize: Int
    /// Text encoder hidden size.
    public let textModelSize: Int
    /// Text encoder depth.
    public let textNumLayers: Int
    /// Text encoder attention heads.
    public let textNumHeads: Int
    /// Text encoder FFN intermediate size.
    public let textIntermediateSize: Int

    // Speaker encoder
    /// Patch size for the speaker latent encoder (stride over latent dim).
    public let speakerPatchSize: Int
    /// Speaker encoder hidden size.
    public let speakerModelSize: Int
    /// Speaker encoder depth.
    public let speakerNumLayers: Int
    /// Speaker encoder attention heads.
    public let speakerNumHeads: Int
    /// Speaker encoder FFN intermediate size.
    public let speakerIntermediateSize: Int

    // AdaLN conditioning
    /// Timestep embedding size (input to the condition module).
    public let timestepEmbedSize: Int
    /// Low-rank dimension for the AdaLN shift/scale/gate projections.
    public let adalnRank: Int
    /// RMS norm epsilon.
    public let normEps: Float

    /// Derive a config from the `dit` sub-dictionary of `config.json`.
    public static func from(_ raw: [String: Any]) -> EchoDiTConfig {
        func int(_ key: String, _ def: Int) -> Int {
            (raw[key] as? Int) ?? def
        }
        func float(_ key: String, _ def: Float) -> Float {
            if let v = raw[key] as? Double { return Float(v) }
            if let v = raw[key] as? Float { return v }
            return def
        }
        return EchoDiTConfig(
            latentSize: int("latent_size", 80),
            modelSize: int("model_size", 2048),
            numLayers: int("num_layers", 24),
            numHeads: int("num_heads", 16),
            intermediateSize: int("intermediate_size", 5888),
            textVocabSize: int("text_vocab_size", 256),
            textModelSize: int("text_model_size", 1280),
            textNumLayers: int("text_num_layers", 14),
            textNumHeads: int("text_num_heads", 10),
            textIntermediateSize: int("text_intermediate_size", 3328),
            speakerPatchSize: int("speaker_patch_size", 4),
            speakerModelSize: int("speaker_model_size", 1280),
            speakerNumLayers: int("speaker_num_layers", 14),
            speakerNumHeads: int("speaker_num_heads", 10),
            speakerIntermediateSize: int("speaker_intermediate_size", 3328),
            timestepEmbedSize: int("timestep_embed_size", 512),
            adalnRank: int("adaln_rank", 256),
            normEps: float("norm_eps", 1e-5)
        )
    }

    public init(
        latentSize: Int = 80,
        modelSize: Int = 2048,
        numLayers: Int = 24,
        numHeads: Int = 16,
        intermediateSize: Int = 5888,
        textVocabSize: Int = 256,
        textModelSize: Int = 1280,
        textNumLayers: Int = 14,
        textNumHeads: Int = 10,
        textIntermediateSize: Int = 3328,
        speakerPatchSize: Int = 4,
        speakerModelSize: Int = 1280,
        speakerNumLayers: Int = 14,
        speakerNumHeads: Int = 10,
        speakerIntermediateSize: Int = 3328,
        timestepEmbedSize: Int = 512,
        adalnRank: Int = 256,
        normEps: Float = 1e-5
    ) {
        self.latentSize = latentSize
        self.modelSize = modelSize
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.intermediateSize = intermediateSize
        self.textVocabSize = textVocabSize
        self.textModelSize = textModelSize
        self.textNumLayers = textNumLayers
        self.textNumHeads = textNumHeads
        self.textIntermediateSize = textIntermediateSize
        self.speakerPatchSize = speakerPatchSize
        self.speakerModelSize = speakerModelSize
        self.speakerNumLayers = speakerNumLayers
        self.speakerNumHeads = speakerNumHeads
        self.speakerIntermediateSize = speakerIntermediateSize
        self.timestepEmbedSize = timestepEmbedSize
        self.adalnRank = adalnRank
        self.normEps = normEps
    }
}

/// Euler CFG sampler hyper-parameters, nested under the `sampler` key in
/// `config.json`.
public struct EchoSamplerConfig: Sendable {
    /// Number of Euler integration steps.
    public let numSteps: Int
    /// Text classifier-free guidance scale.
    public let cfgScaleText: Float
    /// Speaker classifier-free guidance scale.
    public let cfgScaleSpeaker: Float
    /// Minimum diffusion timestep where CFG is active.
    public let cfgMinT: Float
    /// Maximum diffusion timestep where CFG is active.
    public let cfgMaxT: Float
    /// Output latent sequence length (frames).
    public let sequenceLength: Int
    /// Initial noise truncation factor (`None` in config → 0.96 default).
    public let truncationFactor: Float
    /// Optional temporal score rescaling parameter K.
    public let rescaleK: Float?
    /// Optional temporal score rescaling parameter sigma.
    public let rescaleSigma: Float?
    /// Optional speaker KV scaling factor applied before CFG crossover.
    public let speakerKvScale: Float?
    /// Number of layers where speaker KV scaling is applied.
    public let speakerKvMaxLayers: Int?
    /// Timestep at which speaker KV scaling crossover occurs.
    public let speakerKvMinT: Float?

    /// Derive a sampler config from the `sampler` sub-dictionary of
    /// `config.json`. All fields fall back to the published defaults.
    public static func from(_ raw: [String: Any]) -> EchoSamplerConfig {
        func int(_ key: String, _ def: Int) -> Int {
            (raw[key] as? Int) ?? def
        }
        func float(_ key: String, _ def: Float) -> Float {
            if let v = raw[key] as? Double { return Float(v) }
            if let v = raw[key] as? Float { return v }
            return def
        }
        func optFloat(_ key: String) -> Float? {
            if let v = raw[key] as? Double { return Float(v) }
            if let v = raw[key] as? Float { return v }
            return nil
        }
        func optInt(_ key: String) -> Int? { raw[key] as? Int }
        return EchoSamplerConfig(
            numSteps: int("num_steps", 40),
            cfgScaleText: float("cfg_scale_text", 3.0),
            cfgScaleSpeaker: float("cfg_scale_speaker", 8.0),
            cfgMinT: float("cfg_min_t", 0.5),
            cfgMaxT: float("cfg_max_t", 1.0),
            sequenceLength: int("sequence_length", 640),
            truncationFactor: float("truncation_factor", 0.96),
            rescaleK: optFloat("rescale_k"),
            rescaleSigma: optFloat("rescale_sigma"),
            speakerKvScale: optFloat("speaker_kv_scale"),
            speakerKvMaxLayers: optInt("speaker_kv_max_layers"),
            speakerKvMinT: optFloat("speaker_kv_min_t")
        )
    }

    public init(
        numSteps: Int = 40,
        cfgScaleText: Float = 3.0,
        cfgScaleSpeaker: Float = 8.0,
        cfgMinT: Float = 0.5,
        cfgMaxT: Float = 1.0,
        sequenceLength: Int = 640,
        truncationFactor: Float = 0.96,
        rescaleK: Float? = nil,
        rescaleSigma: Float? = nil,
        speakerKvScale: Float? = nil,
        speakerKvMaxLayers: Int? = nil,
        speakerKvMinT: Float? = nil
    ) {
        self.numSteps = numSteps
        self.cfgScaleText = cfgScaleText
        self.cfgScaleSpeaker = cfgScaleSpeaker
        self.cfgMinT = cfgMinT
        self.cfgMaxT = cfgMaxT
        self.sequenceLength = sequenceLength
        self.truncationFactor = truncationFactor
        self.rescaleK = rescaleK
        self.rescaleSigma = rescaleSigma
        self.speakerKvScale = speakerKvScale
        self.speakerKvMaxLayers = speakerKvMaxLayers
        self.speakerKvMinT = speakerKvMinT
    }
}

/// Top-level EchoTTS model configuration, decoded from `config.json`.
public struct EchoTTSConfig: Sendable {
    /// `model_type` identifier ("echo_tts").
    public let modelType: String
    /// Output waveform sample rate (44100 Hz for EchoTTS base).
    public let sampleRate: Int
    /// Maximum text length in UTF-8 tokens.
    public let maxTextLength: Int
    /// Maximum reference speaker latent length in frames.
    public let maxSpeakerLatentLength: Int
    /// Audio downsampling factor of the Fish S1 DAC codec (2048).
    public let audioDownsampleFactor: Int
    /// Whether to prepend `[S1]` and normalize punctuation.
    public let normalizeText: Bool
    /// Whether blockwise generation modules were stripped from the
    /// checkpoint to reduce memory footprint.
    public let deleteBlockwiseModules: Bool
    /// Filename of the PCA state file (relative to model directory).
    public let pcaFilename: String
    /// HuggingFace repo id for the Fish S1 DAC codec checkpoint.
    public let fishCodecRepo: String
    /// DiT transformer hyper-parameters.
    public let dit: EchoDiTConfig
    /// Euler CFG sampler hyper-parameters.
    public let sampler: EchoSamplerConfig

    /// Decode a top-level `config.json` into an `EchoTTSConfig`.
    public static func from(_ config: ModelConfig) -> EchoTTSConfig {
        let raw = config.raw
        func int(_ key: String, _ def: Int) -> Int {
            (raw[key] as? Int) ?? def
        }
        func bool(_ key: String, _ def: Bool) -> Bool {
            (raw[key] as? Bool) ?? def
        }
        func str(_ key: String, _ def: String) -> String {
            (raw[key] as? String) ?? def
        }
        let ditRaw = (raw["dit"] as? [String: Any]) ?? [:]
        let samplerRaw = (raw["sampler"] as? [String: Any]) ?? [:]
        return EchoTTSConfig(
            modelType: str("model_type", "echo_tts"),
            sampleRate: int("sample_rate", 44100),
            maxTextLength: int("max_text_length", 768),
            maxSpeakerLatentLength: int("max_speaker_latent_length", 6400),
            audioDownsampleFactor: int("audio_downsample_factor", 2048),
            normalizeText: bool("normalize_text", true),
            deleteBlockwiseModules: bool("delete_blockwise_modules", false),
            pcaFilename: str("pca_filename", "pca_state.safetensors"),
            fishCodecRepo: str("fish_codec_repo", "jordand/fish-s1-dac-min"),
            dit: EchoDiTConfig.from(ditRaw),
            sampler: EchoSamplerConfig.from(samplerRaw)
        )
    }

    public init(
        modelType: String = "echo_tts",
        sampleRate: Int = 44100,
        maxTextLength: Int = 768,
        maxSpeakerLatentLength: Int = 6400,
        audioDownsampleFactor: Int = 2048,
        normalizeText: Bool = true,
        deleteBlockwiseModules: Bool = false,
        pcaFilename: String = "pca_state.safetensors",
        fishCodecRepo: String = "jordand/fish-s1-dac-min",
        dit: EchoDiTConfig = EchoDiTConfig(),
        sampler: EchoSamplerConfig = EchoSamplerConfig()
    ) {
        self.modelType = modelType
        self.sampleRate = sampleRate
        self.maxTextLength = maxTextLength
        self.maxSpeakerLatentLength = maxSpeakerLatentLength
        self.audioDownsampleFactor = audioDownsampleFactor
        self.normalizeText = normalizeText
        self.deleteBlockwiseModules = deleteBlockwiseModules
        self.pcaFilename = pcaFilename
        self.fishCodecRepo = fishCodecRepo
        self.dit = dit
        self.sampler = sampler
    }
}

// ─── Errors ───────────────────────────────────────────────────────────────────

public enum EchoTTSError: Error, CustomStringConvertible {
    /// The DiT diffusion forward pass requires batched GEMM + scaled dot-
    /// product attention — GPU operators beyond the current FFAI Ops set.
    /// Until those operators land, the synthesis critical path is not
    /// GPU-accelerated. Use a reference implementation (mlx-audio-swift)
    /// for full synthesis; this FFAI port wires the config, load, and
    /// registry paths that will wrap the GPU path when it arrives.
    case diffusionNotWired
    /// A required file is missing from the model directory.
    case missingFile(String)

    public var description: String {
        switch self {
        case .diffusionNotWired:
            return "EchoTTS: the DiT diffusion forward pass requires batched "
                + "GEMM + SDPA — GPU operators not yet in the FFAI Ops set. "
                + "Synthesis is not available in this build; use "
                + "generatePlaceholder to verify load + config."
        case .missingFile(let name):
            return "EchoTTS: required file missing from model directory: \(name)"
        }
    }
}

// ─── Model ────────────────────────────────────────────────────────────────────

/// A loaded EchoTTS model. Owns the decoded config and (when available) the
/// PCA state tensors from `pca_state.safetensors`. The DiT weights are
/// validated on load but the full diffusion forward pass is gated behind
/// `EchoTTSError.diffusionNotWired` until FFAI adds batched GEMM + SDPA.
public final class EchoTTSModel: @unchecked Sendable {
    /// Decoded config from `config.json`.
    public let config: EchoTTSConfig
    /// PCA components `[latentSize, codecDim]` loaded from
    /// `pca_state.safetensors`. `nil` until `load(directory:)` is called.
    public let pcaComponents: Tensor?
    /// PCA mean `[codecDim]` loaded from `pca_state.safetensors`.
    public let pcaMean: Tensor?
    /// Scalar latent scale from `pca_state.safetensors` (1.0 if absent).
    public let latentScale: Float
    /// Number of weights loaded from `model.safetensors` (diagnostic).
    public let weightCount: Int
    /// Output waveform sample rate in Hz (44100 for EchoTTS base).
    public var sampleRate: Int { config.sampleRate }

    public init(config: EchoTTSConfig,
                pcaComponents: Tensor? = nil,
                pcaMean: Tensor? = nil,
                latentScale: Float = 1.0,
                weightCount: Int = 0) {
        self.config = config
        self.pcaComponents = pcaComponents
        self.pcaMean = pcaMean
        self.latentScale = latentScale
        self.weightCount = weightCount
    }

    /// Return a placeholder waveform (zeros, `durationSeconds` long) for
    /// integration tests that verify load + config without running diffusion.
    /// Shape: `[nSamples]` f32.
    public func generatePlaceholder(durationSeconds: Double = 0.1,
                                    device: Device = .shared) -> Tensor {
        let nSamples = max(1, Int(durationSeconds * Double(sampleRate)))
        let t = Tensor.empty(shape: [nSamples], dtype: .f32, device: device)
        t.zero()
        return t
    }

    /// Full text→waveform synthesis. Requires the DiT forward pass and Fish
    /// S1 DAC decoder; throws `EchoTTSError.diffusionNotWired` until the
    /// required GPU operators (batched GEMM + SDPA) land in FFAI Ops.
    ///
    /// - Parameters:
    ///   - text: Input text (UTF-8; `[S1]` prepended automatically when
    ///           `config.normalizeText` is true and absent from `text`).
    ///   - refAudio: Optional reference audio for voice cloning `[nSamples]`
    ///               f32 at `config.sampleRate`. Pass `nil` for a default
    ///               (no-reference) voice.
    ///   - device: Metal device to use.
    public func synthesize(text: String,
                           refAudio: Tensor? = nil,
                           device: Device = .shared) throws -> Tensor {
        _ = text
        _ = refAudio
        _ = device
        throw EchoTTSError.diffusionNotWired
    }
}

// ─── Loading ──────────────────────────────────────────────────────────────────

extension EchoTTSModel {
    /// `model_type` values that identify an EchoTTS checkpoint.
    public static let modelTypes: Set<String> = ["echo_tts"]

    /// Whether a decoded `config.json` describes an EchoTTS checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        // Structural detection: `dit` block with `latent_size` + `speaker_patch_size`.
        if let dit = config.nested("dit") {
            return dit["latent_size"] != nil && dit["speaker_patch_size"] != nil
        }
        return false
    }

    /// Load an EchoTTS checkpoint from a resolved snapshot directory.
    ///
    /// - Reads `config.json` and decodes `EchoTTSConfig`.
    /// - Loads `pca_state.safetensors` for the PCA transform tensors.
    /// - Opens `model.safetensors` and counts top-level weights (validates
    ///   that the checkpoint is complete without allocating GPU memory for
    ///   the full DiT — the forward pass is not yet wired).
    public static func load(directory: URL, device: Device = .shared)
        throws -> EchoTTSModel {
        let config = try ModelConfig.load(from: directory)
        guard handles(config) else {
            throw ModelError.unsupportedModelType(
                config.modelType ?? "unknown — expected echo_tts")
        }
        let echoConfig = EchoTTSConfig.from(config)

        // Load PCA state (required for encode/decode).
        let pcaURL = directory.appendingPathComponent(echoConfig.pcaFilename)
        guard FileManager.default.fileExists(atPath: pcaURL.path) else {
            throw EchoTTSError.missingFile(echoConfig.pcaFilename)
        }
        let pcaFile = try SafeTensorsFile(url: pcaURL, device: device)
        let pcaComponents: Tensor?
        let pcaMean: Tensor?
        let latentScale: Float

        if pcaFile.entries["pca_components"] != nil,
           pcaFile.entries["pca_mean"] != nil {
            // Build Tensor views from the mmap'd SafeTensorsFile entries.
            pcaComponents = try pcaFile.tensor(named: "pca_components")
            pcaMean = try pcaFile.tensor(named: "pca_mean")
            if pcaFile.entries["latent_scale"] != nil {
                // latent_scale is a scalar stored as a 1-element tensor.
                let scaleTensor = try pcaFile.tensor(named: "latent_scale")
                latentScale = scaleTensor.toArray(as: Float.self).first ?? 1.0
            } else {
                latentScale = 1.0
            }
        } else {
            pcaComponents = nil
            pcaMean = nil
            latentScale = 1.0
        }

        // Open model.safetensors to verify the checkpoint is present and
        // count weights (diagnostic; no full GPU allocation needed).
        let modelURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw EchoTTSError.missingFile("model.safetensors")
        }
        let modelFile = try SafeTensorsFile(url: modelURL, device: device)
        let weightCount = modelFile.entries.count

        return EchoTTSModel(
            config: echoConfig,
            pcaComponents: pcaComponents,
            pcaMean: pcaMean,
            latentScale: latentScale,
            weightCount: weightCount
        )
    }
}
