// QwenOmni — Qwen's omni-modal family (Qwen2.5-Omni / Qwen3-Omni):
// text + vision + audio in, text (and optionally audio) out. The model
// is a Qwen text backbone fronted by per-modality encoders whose
// outputs are spliced into the text token stream.
//
//   audio   ──audio encoder──▶ audio tokens ──┐
//   image   ──vision encoder──▶ image tokens ─┼─▶ text backbone ──▶ text
//   text    ──token embed──────▶ text tokens ─┘
//
// ## Scope (Phase 7)
//
// This file wires the **audio-in path**: a Whisper-style `AudioEncoder`
// turns a waveform into audio feature tokens in the backbone's hidden
// dim. The vision path is Phase 6.5 infrastructure (`VisionEncoder`)
// and is scaffolded but not spliced here — Qwen-VL's dynamic-resolution
// tower is a separate port. The text backbone is the existing Qwen3
// engine; QwenOmni composes it rather than reimplementing it.
//
// `QwenOmni.encodeAudio` is the supported entry point: it produces the
// `[nAudioTokens, textHidden]` features a caller splices into a Qwen3
// prompt's embedding stream.

import Foundation
import Metal

// ─── Configuration ───────────────────────────────────────────────────

/// QwenOmni audio-tower hyper-parameters, decoded from the checkpoint's
/// nested `audio_config` (a Whisper-style encoder config) plus the
/// backbone text hidden dim.
public struct QwenOmniAudioConfig: Sendable {
    /// Mel bins the audio front-end produces.
    public let nMels: Int
    /// Audio-encoder hidden dim (`d_model` of the audio tower).
    public let encoderHidden: Int
    /// Audio-encoder feed-forward intermediate dim.
    public let encoderIntermediate: Int
    /// Audio-encoder transformer blocks.
    public let encoderLayers: Int
    /// Audio-encoder attention heads.
    public let encoderHeads: Int
    /// Maximum audio-context length.
    public let maxAudioCtx: Int
    /// The text backbone's hidden dim — the audio features are
    /// projected into this so they can be spliced into the text stream.
    public let textHidden: Int

    public init(nMels: Int, encoderHidden: Int, encoderIntermediate: Int,
                encoderLayers: Int, encoderHeads: Int, maxAudioCtx: Int,
                textHidden: Int) {
        self.nMels = nMels
        self.encoderHidden = encoderHidden
        self.encoderIntermediate = encoderIntermediate
        self.encoderLayers = encoderLayers
        self.encoderHeads = encoderHeads
        self.maxAudioCtx = maxAudioCtx
        self.textHidden = textHidden
    }

    /// The `audio_config` block in a Qwen-Omni `config.json`. Qwen2.5-
    /// Omni nests it under `thinker_config.audio_config`; some Qwen3-
    /// Omni / mlx conversions hoist it to the top level. Returns the
    /// `(audio_config, text_config)` pair, looking in both places.
    static func locateConfigs(_ config: ModelConfig)
        -> (audio: [String: Any], text: [String: Any]?)? {
        // Top-level (hoisted) layout.
        if let audio = config.nested("audio_config") {
            return (audio, config.nested("text_config"))
        }
        // Qwen2.5-Omni nested-under-thinker layout.
        if let thinker = config.nested("thinker_config"),
           let audio = thinker["audio_config"] as? [String: Any] {
            return (audio, thinker["text_config"] as? [String: Any])
        }
        return nil
    }

    /// Build from a decoded `config.json`. Qwen-Omni nests the audio
    /// tower under `audio_config` (a Whisper-style block) and the text
    /// backbone under `text_config`, both inside `thinker_config` for
    /// Qwen2.5-Omni.
    public static func from(_ config: ModelConfig) -> QwenOmniAudioConfig? {
        guard let located = locateConfigs(config) else { return nil }
        let audio = located.audio
        func ai(_ k: String) -> Int? {
            if let v = audio[k] as? Int { return v }
            if let v = audio[k] as? Double { return Int(v) }
            return nil
        }
        guard let encHidden = ai("d_model") else { return nil }
        // Whisper-style configs name the layer count `encoder_layers`;
        // Qwen-Omni's audio block also carries `num_hidden_layers`.
        guard let encLayers = ai("encoder_layers") ?? ai("num_hidden_layers"),
              let encHeads = ai("encoder_attention_heads") else { return nil }
        let nMels = ai("num_mel_bins") ?? 128
        let encInter = ai("encoder_ffn_dim") ?? (4 * encHidden)
        let maxAud = ai("max_source_positions") ?? 1500
        // The text hidden dim — the audio features project into it.
        // `output_dim` on the audio block is the projection target;
        // fall back to the text config's hidden_size.
        let textHidden = ai("output_dim")
            ?? (located.text?["hidden_size"] as? Int)
            ?? config.hiddenSize
            ?? encHidden
        return QwenOmniAudioConfig(
            nMels: nMels, encoderHidden: encHidden,
            encoderIntermediate: encInter, encoderLayers: encLayers,
            encoderHeads: encHeads, maxAudioCtx: maxAud,
            textHidden: textHidden)
    }

    /// The front-end config the audio tower expects (16 kHz log-Mel).
    public var frontEnd: AudioFrontEndConfig {
        AudioFrontEndConfig(sampleRate: 16_000, nFFT: 400, hopLength: 160,
                            nMels: nMels)
    }
}

// ─── QwenOmni model ──────────────────────────────────────────────────

public enum QwenOmniError: Error, CustomStringConvertible {
    /// The text backbone is not loaded — `QwenOmni` was constructed
    /// audio-only. `encodeAudio` works regardless.
    case textBackboneUnavailable

    public var description: String {
        switch self {
        case .textBackboneUnavailable:
            return "QwenOmni: the text backbone is not loaded in this "
                + "audio-only build; use encodeAudio to obtain audio "
                + "feature tokens and run them through a Qwen3 model"
        }
    }
}

/// A loaded QwenOmni model. Owns the audio encoder (always available —
/// the FFAI Phase 7 contribution) and, when wired, the text backbone +
/// vision tower.
public final class QwenOmniModel: @unchecked Sendable {
    public let config: QwenOmniAudioConfig
    /// The Whisper-style audio encoder.
    public let audioEncoder: AudioEncoder
    /// Projection from the audio-encoder hidden dim into the text
    /// backbone hidden dim. `nil` when they already match.
    public let audioProjection: Linear?
    /// The text backbone — `nil` in an audio-only build.
    public let textBackbone: Qwen3Model?
    let dtype: DType

    public init(config: QwenOmniAudioConfig, audioEncoder: AudioEncoder,
                audioProjection: Linear?, textBackbone: Qwen3Model?,
                dtype: DType) {
        self.config = config
        self.audioEncoder = audioEncoder
        self.audioProjection = audioProjection
        self.textBackbone = textBackbone
        self.dtype = dtype
    }

    /// Encode a waveform into audio feature tokens in the text backbone
    /// hidden dim. Returns `[nAudioTokens, textHidden]` — the tokens a
    /// caller splices into a Qwen3 prompt's embedding stream.
    public func encodeAudio(waveform: [Float], device: Device = .shared)
        -> Tensor {
        // The mel_spectrogram kernel only emits f32 / f16; a bf16 model
        // gets the front-end run in f32, then the spectrogram cast to
        // the model's activation dtype before the conv stem.
        let melDtype: DType = dtype == .f16 ? .f16 : .f32
        let cmd = device.makeCommandBuffer()
        // `whisperNormalize: true` — `logMelSpectrogram` commits + waits
        // on `cmd` itself (it normalises the kernel result on the CPU),
        // so the result is already CPU-synced; do NOT re-commit `cmd`.
        let melRaw = AudioPreprocessing.logMelSpectrogram(
            waveform: waveform, cfg: config.frontEnd, dtype: melDtype,
            whisperNormalize: true, device: device, on: cmd)
        let mel = AudioPreprocessing.castTensor(melRaw, to: dtype,
                                                device: device)

        // Run the Whisper-style encoder over the log-Mel.
        var features = audioEncoder.encode(mel: mel, melFrameMajor: true,
                                           device: device)

        // Project into the text hidden dim if the dims differ.
        guard let proj = audioProjection else { return features }
        let nTokens = features.shape[0]
        let cmd2 = device.makeCommandBuffer()
        var projected = Ops.gemm(weight: proj.weight, input: features,
                                 nRows: nTokens, on: cmd2)
        if let bias = proj.bias {
            projected = AudioEncoder.addRowBias(
                projected, bias: bias, nRows: nTokens,
                rowSize: config.textHidden, on: cmd2)
        }
        cmd2.commit()
        cmd2.waitUntilCompleted()
        features = projected
        return features
    }

    /// Whether this build has a text backbone wired.
    public var hasTextBackbone: Bool { textBackbone != nil }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension QwenOmniModel {
    public static let modelTypes: Set<String> = [
        "qwen2_5_omni", "qwen3_omni", "qwen2_5_omni_thinker",
    ]
    public static let architectures: Set<String> = [
        "Qwen2_5OmniForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
    ]

    /// Whether a decoded `config.json` describes a QwenOmni checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        // Fall back to structural detection — an `audio_config` block,
        // top-level or nested under `thinker_config`.
        return QwenOmniAudioConfig.locateConfigs(config) != nil
    }

    /// Load the QwenOmni audio path from a resolved snapshot directory.
    /// The audio encoder is always constructed; the text backbone is
    /// loaded when the checkpoint's text weights are present and the
    /// vision path is left for the Qwen-VL port (see the scope note).
    public static func load(directory: URL, device: Device = .shared)
        throws -> QwenOmniModel {
        let config = try ModelConfig.load(from: directory)
        guard let qc = QwenOmniAudioConfig.from(config) else {
            throw ModelError.unsupportedModelType(
                "config.json has no audio_config — not a QwenOmni checkpoint")
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        return try build(config: qc, bundle: bundle)
    }

    /// Assemble a `QwenOmniModel` audio path from a decoded config + a
    /// weight bundle. Factored out so tests can drive it directly.
    public static func build(config qc: QwenOmniAudioConfig,
                             bundle: SafeTensorsBundle) throws -> QwenOmniModel {
        // Qwen-Omni prefixes the audio tower weights; conversions vary
        // (`audio_tower.` / `thinker.audio_tower.`). Detect the prefix.
        let prefixes = ["thinker.audio_tower.", "audio_tower.",
                        "model.audio_tower.", ""]
        guard let prefix = prefixes.first(where: {
            bundle.has("\($0)conv1.weight")
        }) else {
            throw ModelError.unsupportedModelType(
                "QwenOmni: no audio_tower weights found in the checkpoint")
        }
        let dtype = try bundle.tensor(named: "\(prefix)conv1.weight").dtype

        func t(_ name: String) throws -> Tensor {
            try bundle.tensor(named: prefix + name)
        }
        func ln(_ base: String) throws -> LayerNorm {
            LayerNorm(weight: try t("\(base).weight"),
                      bias: try t("\(base).bias"), eps: 1e-5)
        }
        func linear(_ base: String, hasBias: Bool = true) throws -> Linear {
            let w = try t("\(base).weight")
            let b = hasBias && bundle.has(prefix + "\(base).bias")
                ? try t("\(base).bias") : nil
            return Linear(weight: w, bias: b)
        }

        var encLayers: [AudioEncoderLayer] = []
        for i in 0..<qc.encoderLayers {
            let base = "layers.\(i)"
            encLayers.append(AudioEncoderLayer(
                layerNorm1: try ln("\(base).self_attn_layer_norm"),
                qProj: try linear("\(base).self_attn.q_proj"),
                kProj: try linear("\(base).self_attn.k_proj", hasBias: false),
                vProj: try linear("\(base).self_attn.v_proj"),
                oProj: try linear("\(base).self_attn.out_proj"),
                layerNorm2: try ln("\(base).final_layer_norm"),
                fc1: try linear("\(base).fc1"),
                fc2: try linear("\(base).fc2"),
                hidden: qc.encoderHidden, nHeads: qc.encoderHeads,
                intermediate: qc.encoderIntermediate))
        }
        let encoderConfig = AudioEncoderConfig(
            nMels: qc.nMels, hidden: qc.encoderHidden,
            intermediate: qc.encoderIntermediate, nLayers: qc.encoderLayers,
            nHeads: qc.encoderHeads, maxAudioCtx: qc.maxAudioCtx,
            layerNormEps: 1e-5)

        // Qwen-Omni's audio encoder uses a fixed sinusoidal position
        // encoding computed at runtime — unlike Whisper it does NOT
        // store `embed_positions.weight` in the checkpoint. Synthesize
        // the table when the weight is absent.
        let posEmbedding: Tensor
        if bundle.has(prefix + "embed_positions.weight") {
            posEmbedding = try t("embed_positions.weight")
        } else {
            let table = AudioPreprocessing.sinusoidalPositions(
                length: qc.maxAudioCtx, dim: qc.encoderHidden)
            let pe = Tensor.empty(shape: [qc.maxAudioCtx, qc.encoderHidden],
                                  dtype: dtype)
            AudioPreprocessing.copyFloats(table, into: pe)
            posEmbedding = pe
        }

        let encoder = AudioEncoder(
            config: encoderConfig,
            conv1Weight: try t("conv1.weight"),
            conv1Bias: try t("conv1.bias"),
            conv2Weight: try t("conv2.weight"),
            conv2Bias: try t("conv2.bias"),
            positionEmbedding: posEmbedding,
            layers: encLayers,
            postLayerNorm: try ln("ln_post"),
            dtype: dtype)

        // Audio→text projection, if the checkpoint carries one and the
        // dims differ.
        var audioProjection: Linear? = nil
        if qc.encoderHidden != qc.textHidden {
            for projBase in ["proj", "audio_bos_eos_token.proj",
                             "multi_modal_projector"] {
                if bundle.has(prefix + "\(projBase).weight") {
                    audioProjection = try linear(projBase)
                    break
                }
            }
        }

        return QwenOmniModel(
            config: qc, audioEncoder: encoder,
            audioProjection: audioProjection, textBackbone: nil,
            dtype: dtype)
    }
}
