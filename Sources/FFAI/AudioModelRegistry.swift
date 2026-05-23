// AudioModelRegistry — config-driven detection + loading for the audio
// model families (Whisper STT, Kokoro TTS, QwenOmni audio-in).
//
// Audio models do NOT fit `LanguageModel` / `ModelRegistry` — those
// describe a pure text-in / text-out causal decoder. An STT model is
// audio-in / text-out, a TTS model is text-in / audio-out, and an omni
// model is multi-modal. This registry is the audio-side counterpart of
// `ModelRegistry.dispatchAndLoad`: it inspects a decoded `config.json`,
// picks the family, and reports the `Capability` set the model exposes.

import Foundation

/// A loaded audio model — one of the three family types — together
/// with the capability set it supports. `Model`-style callers branch on
/// the enum; tests pull the concrete type out.
public enum LoadedAudioModel: @unchecked Sendable {
    case whisper(WhisperModel)
    case senseVoice(SenseVoiceModel)
    case kokoro(KokoroModel)
    case qwenOmni(QwenOmniModel)
    case llamaTTS(LlamaTTSModel)
    case marvis(MarvisModel)
    case qwen3TTS(Qwen3TTSModel)
    case echoTTS(EchoTTSModel)
    case qwen3TTSBase(Qwen3TTSBaseModel)
    case parakeet(ParakeetModel)
    case chatterbox(ChatterboxModel)
    case qwen3ASR(Qwen3ASRModel)
    case fireRedASR2(FireRedASR2Model)

    /// The capabilities this model exposes — `audioIn` for STT / omni,
    /// `audioOut` for TTS.
    public var capabilities: Set<Capability> {
        switch self {
        case .whisper: return Capability.speechToText
        case .senseVoice: return Capability.speechToText
        case .kokoro: return Capability.textToSpeech
        case .qwenOmni: return Capability.omniAudio
        case .llamaTTS: return Capability.textToSpeech
        case .marvis: return Capability.textToSpeech
        case .qwen3TTS: return Capability.textToSpeech
        case .echoTTS: return Capability.textToSpeech
        case .qwen3TTSBase: return Capability.textToSpeech
        case .parakeet: return Capability.speechToText
        case .chatterbox: return Capability.textToSpeech
        case .qwen3ASR: return Capability.speechToText
        case .fireRedASR2: return Capability.speechToText
        }
    }
}

/// Routes an audio-model `config.json` to the right family loader.
public enum AudioModelRegistry {

    /// Whether `config` describes any audio model this registry handles.
    /// Lets a caller decide between `ModelRegistry` (text) and this
    /// registry (audio) before committing to a load path.
    public static func handles(_ config: ModelConfig) -> Bool {
        WhisperModel.handles(config)
            || SenseVoiceModel.handles(config)
            || KokoroModel.handles(config)
            || QwenOmniModel.handles(config)
            || LlamaTTSModel.handles(config)
            || MarvisModel.handles(config)
            || Qwen3TTSModel.handles(config)
            || EchoTTSModel.handles(config)
            || Qwen3TTSBaseModel.handles(config)
            || ParakeetModel.handles(config)
            || ChatterboxModel.handles(config)
            || Qwen3ASRModel.handles(config)
            || FireRedASR2Model.handles(config)
    }

    /// The capability set a checkpoint at `directory` would expose,
    /// without loading its weights. `nil` when it is not an audio model.
    public static func capabilities(forConfigAt directory: URL)
        -> Set<Capability>? {
        guard let config = try? ModelConfig.load(from: directory) else {
            return nil
        }
        return capabilities(for: config)
    }

    /// The capability set a decoded config would expose. `nil` when the
    /// config is not an audio model. QwenOmni is checked before Whisper
    /// because an omni checkpoint nests a Whisper-style `audio_config`.
    public static func capabilities(for config: ModelConfig)
        -> Set<Capability>? {
        if Qwen3TTSModel.handles(config) { return Capability.textToSpeech }
        if QwenOmniModel.handles(config) { return Capability.omniAudio }
        if WhisperModel.handles(config) { return Capability.speechToText }
        if SenseVoiceModel.handles(config) { return Capability.speechToText }
        if KokoroModel.handles(config) { return Capability.textToSpeech }
        if MarvisModel.handles(config) { return Capability.textToSpeech }
        if LlamaTTSModel.handles(config) { return Capability.textToSpeech }
        if EchoTTSModel.handles(config) { return Capability.textToSpeech }
        if Qwen3TTSBaseModel.handles(config) { return Capability.textToSpeech }
        if ParakeetModel.handles(config) { return Capability.speechToText }
        if ChatterboxModel.handles(config) { return Capability.textToSpeech }
        if Qwen3ASRModel.handles(config) { return Capability.speechToText }
        if FireRedASR2Model.handles(config) { return Capability.speechToText }
        return nil
    }

    /// Load the audio model at a resolved snapshot `directory`.
    /// QwenOmni is checked first (its config nests a Whisper-style
    /// `audio_config`, which would otherwise be mistaken for Whisper).
    /// LlamaTTS is checked before Whisper / Kokoro because an Orpheus
    /// checkpoint can carry a plain `LlamaForCausalLM` architecture and
    /// its loader needs the tokenizer (hence the async signature).
    public static func load(directory: URL, device: Device = .shared)
        async throws -> LoadedAudioModel {
        let config = try ModelConfig.load(from: directory)
        // Qwen3TTS is checked before QwenOmni: both are Qwen-family
        // audio models, but Qwen3TTS's `talker_config` is its own marker.
        if Qwen3TTSModel.handles(config) {
            return .qwen3TTS(try Qwen3TTSModel.load(directory: directory,
                                                    device: device))
        }
        if QwenOmniModel.handles(config) {
            return .qwenOmni(try QwenOmniModel.load(directory: directory,
                                                    device: device))
        }
        if MarvisModel.handles(config) {
            return .marvis(try await MarvisModel.load(
                directory: directory, device: device))
        }
        if LlamaTTSModel.handles(config) {
            return .llamaTTS(try await LlamaTTSModel.load(
                directory: directory, device: device))
        }
        if WhisperModel.handles(config) {
            return .whisper(try WhisperModel.load(directory: directory,
                                                  device: device))
        }
        if SenseVoiceModel.handles(config) {
            return .senseVoice(try SenseVoiceModel.load(directory: directory,
                                                        device: device))
        }
        if KokoroModel.handles(config) {
            return .kokoro(try KokoroModel.load(directory: directory,
                                                device: device))
        }
        if EchoTTSModel.handles(config) {
            return .echoTTS(try EchoTTSModel.load(directory: directory,
                                                  device: device))
        }
        if Qwen3TTSBaseModel.handles(config) {
            return .qwen3TTSBase(try await Qwen3TTSBaseModel.load(
                directory: directory, device: device))
        }
        if ParakeetModel.handles(config) {
            return .parakeet(try ParakeetModel.load(directory: directory,
                                                    device: device))
        }
        if ChatterboxModel.handles(config) {
            return .chatterbox(try ChatterboxModel.load(directory: directory,
                                                        device: device))
        }
        if Qwen3ASRModel.handles(config) {
            return .qwen3ASR(try Qwen3ASRModel.load(directory: directory,
                                                    device: device))
        }
        if FireRedASR2Model.handles(config) {
            return .fireRedASR2(try FireRedASR2Model.load(
                directory: directory, device: device))
        }
        throw ModelError.unsupportedArchitecture(
            config.architecture ?? config.modelType ?? "<unknown audio model>")
    }
}
