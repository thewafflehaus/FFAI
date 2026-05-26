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
    case mossTTS(MossTTSModel)
    case mossTTSNano(MossTTSNanoModel)
    case lfmAudio(LFMAudioModel)
    case voxtralRealtime(VoxtralRealtimeModel)
    case glmASR(GLMASRModel)
    case pocketTTS(PocketTTSModel)
    case soprano(SopranoModel)
    case styleTTS2(StyleTTS2Model)
    case graniteSpeech(GraniteSpeechModel)
    case deepFilterNet(DeepFilterNetModel)
    case cohereTranscribe(CohereTranscribeModel)
    case mossFormer2SE(MossFormer2SEModel)
    case samAudio(SAMAudioModel)

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
        case .mossTTS: return Capability.textToSpeech
        case .mossTTSNano: return Capability.textToSpeech
        case .lfmAudio: return Capability.omniAudio
        case .voxtralRealtime: return Capability.speechToText
        case .glmASR: return Capability.speechToText
        case .pocketTTS: return Capability.textToSpeech
        case .soprano: return Capability.textToSpeech
        case .styleTTS2: return Capability.textToSpeech
        case .graniteSpeech: return Capability.speechToText
        case .deepFilterNet: return Capability.speechToSpeech
        case .cohereTranscribe: return Capability.speechToText
        case .mossFormer2SE: return Capability.speechToSpeech
        case .samAudio: return Capability.speechToSpeech
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
            || MossTTSNanoModel.handles(config)
            || MossTTSModel.handles(config)
            || LFMAudioModel.handles(config)
            || VoxtralRealtimeModel.handles(config)
            || GLMASRModel.handles(config)
            || PocketTTSModel.handles(config)
            || SopranoModel.handles(config)
            || StyleTTS2Model.handles(config)
            || GraniteSpeech.handles(config)
            || DeepFilterNetModel.handles(config)
            || CohereTranscribeModel.handles(config)
            || MossFormer2SEModel.handles(config)
            || SAMAudio.handles(config)
    }

    /// The capability set a checkpoint at `directory` would expose,
    /// without loading its weights. `nil` when it is not an audio model.
    public static func capabilities(forConfigAt directory: URL)
        -> Set<Capability>?
    {
        guard let config = try? ModelConfig.load(from: directory) else {
            return nil
        }
        return capabilities(for: config)
    }

    /// The capability set a decoded config would expose. `nil` when the
    /// config is not an audio model. QwenOmni is checked before Whisper
    /// because an omni checkpoint nests a Whisper-style `audio_config`.
    /// Qwen3ASR is checked BEFORE QwenOmni because QwenOmni's structural
    /// fallback (`audio_config` nested anywhere) also matches a Qwen3-ASR
    /// config — Qwen3-ASR's distinctive `qwen3_asr` model_type lets us
    /// route it precisely if we check first.
    public static func capabilities(for config: ModelConfig)
        -> Set<Capability>?
    {
        if Qwen3TTSModel.handles(config) { return Capability.textToSpeech }
        if Qwen3ASRModel.handles(config) { return Capability.speechToText }
        if QwenOmniModel.handles(config) { return Capability.omniAudio }
        if WhisperModel.handles(config) { return Capability.speechToText }
        if SenseVoiceModel.handles(config) { return Capability.speechToText }
        // StyleTTS2 / KittenTTS before Kokoro: Kokoro's structural fallback
        // (`istftnet + n_token`) also matches KittenTTS's `kitten_tts`
        // model_type. StyleTTS2's stricter structural check
        // (`n_token + istftnet + plbert`) routes Kitten precisely.
        if StyleTTS2Model.handles(config) { return Capability.textToSpeech }
        if KokoroModel.handles(config) { return Capability.textToSpeech }
        if MarvisModel.handles(config) { return Capability.textToSpeech }
        if LlamaTTSModel.handles(config) { return Capability.textToSpeech }
        if EchoTTSModel.handles(config) { return Capability.textToSpeech }
        if Qwen3TTSBaseModel.handles(config) { return Capability.textToSpeech }
        if ParakeetModel.handles(config) { return Capability.speechToText }
        if ChatterboxModel.handles(config) { return Capability.textToSpeech }
        if FireRedASR2Model.handles(config) { return Capability.speechToText }
        if MossTTSNanoModel.handles(config) { return Capability.textToSpeech }
        if MossTTSModel.handles(config) { return Capability.textToSpeech }
        if LFMAudioModel.handles(config) { return Capability.omniAudio }
        if VoxtralRealtimeModel.handles(config) { return Capability.speechToText }
        if GLMASRModel.handles(config) { return Capability.speechToText }
        if PocketTTSModel.handles(config) { return Capability.textToSpeech }
        if SopranoModel.handles(config) { return Capability.textToSpeech }
        if GraniteSpeech.handles(config) { return Capability.speechToText }
        if DeepFilterNetModel.handles(config) { return Capability.speechToSpeech }
        if CohereTranscribeModel.handles(config) { return Capability.speechToText }
        if MossFormer2SEModel.handles(config) { return Capability.speechToSpeech }
        if SAMAudio.handles(config) { return Capability.speechToSpeech }
        return nil
    }

    /// Load the audio model at a resolved snapshot `directory`.
    /// QwenOmni is checked first (its config nests a Whisper-style
    /// `audio_config`, which would otherwise be mistaken for Whisper).
    /// LlamaTTS is checked before Whisper / Kokoro because an Orpheus
    /// checkpoint can carry a plain `LlamaForCausalLM` architecture and
    /// its loader needs the tokenizer (hence the async signature).
    public static func load(directory: URL, device: Device = .shared)
        async throws -> LoadedAudioModel
    {
        let config = try ModelConfig.load(from: directory)
        // Qwen3TTS is checked before QwenOmni: both are Qwen-family
        // audio models, but Qwen3TTS's `talker_config` is its own marker.
        if Qwen3TTSModel.handles(config) {
            return .qwen3TTS(
                try Qwen3TTSModel.load(
                    directory: directory,
                    device: device))
        }
        // Qwen3ASR before QwenOmni — see `capabilities(for:)` for the
        // routing-precedence reasoning (Qwen3-ASR's `audio_config` is
        // also matched by QwenOmni's structural fallback).
        if Qwen3ASRModel.handles(config) {
            return .qwen3ASR(
                try Qwen3ASRModel.load(
                    directory: directory,
                    device: device))
        }
        if QwenOmniModel.handles(config) {
            return .qwenOmni(
                try QwenOmniModel.load(
                    directory: directory,
                    device: device))
        }
        if MarvisModel.handles(config) {
            return .marvis(
                try await MarvisModel.load(
                    directory: directory, device: device))
        }
        if LlamaTTSModel.handles(config) {
            return .llamaTTS(
                try await LlamaTTSModel.load(
                    directory: directory, device: device))
        }
        if WhisperModel.handles(config) {
            return .whisper(
                try WhisperModel.load(
                    directory: directory,
                    device: device))
        }
        if SenseVoiceModel.handles(config) {
            return .senseVoice(
                try SenseVoiceModel.load(
                    directory: directory,
                    device: device))
        }
        // StyleTTS2 / KittenTTS before Kokoro — see `capabilities(for:)`
        // routing-precedence note. Kitten's `kitten_tts` model_type would
        // otherwise be claimed by Kokoro's structural fallback.
        if StyleTTS2Model.handles(config) {
            return .styleTTS2(
                try StyleTTS2Model.load(
                    directory: directory, device: device))
        }
        if KokoroModel.handles(config) {
            return .kokoro(
                try KokoroModel.load(
                    directory: directory,
                    device: device))
        }
        if EchoTTSModel.handles(config) {
            return .echoTTS(
                try EchoTTSModel.load(
                    directory: directory,
                    device: device))
        }
        if Qwen3TTSBaseModel.handles(config) {
            return .qwen3TTSBase(
                try await Qwen3TTSBaseModel.load(
                    directory: directory, device: device))
        }
        if ParakeetModel.handles(config) {
            return .parakeet(
                try ParakeetModel.load(
                    directory: directory,
                    device: device))
        }
        if ChatterboxModel.handles(config) {
            return .chatterbox(
                try ChatterboxModel.load(
                    directory: directory,
                    device: device))
        }
        if FireRedASR2Model.handles(config) {
            return .fireRedASR2(
                try FireRedASR2Model.load(
                    directory: directory, device: device))
        }
        // Nano checked before MossTTS-8B — both share the Moss family but
        // Nano has a distinctive `gpt2_config` block.
        if MossTTSNanoModel.handles(config) {
            return .mossTTSNano(
                try MossTTSNanoModel.load(
                    directory: directory, device: device))
        }
        if MossTTSModel.handles(config) {
            return .mossTTS(
                try MossTTSModel.load(
                    directory: directory, device: device))
        }
        if LFMAudioModel.handles(config) {
            return .lfmAudio(
                try LFMAudioModel.load(
                    directory: directory, device: device))
        }
        if VoxtralRealtimeModel.handles(config) {
            return .voxtralRealtime(
                try VoxtralRealtimeModel.load(
                    directory: directory, device: device))
        }
        if GLMASRModel.handles(config) {
            return .glmASR(
                try GLMASRModel.load(
                    directory: directory, device: device))
        }
        if PocketTTSModel.handles(config) {
            return .pocketTTS(
                try PocketTTSModel.load(
                    directory: directory, device: device))
        }
        if SopranoModel.handles(config) {
            return .soprano(
                try await SopranoModel.load(
                    directory: directory, device: device))
        }
        if GraniteSpeech.handles(config) {
            return .graniteSpeech(
                try await GraniteSpeech.load(
                    directory: directory, device: device))
        }
        if DeepFilterNetModel.handles(config) {
            return .deepFilterNet(
                try DeepFilterNetModel.load(
                    from: directory, device: device))
        }
        if CohereTranscribeModel.handles(config) {
            return .cohereTranscribe(
                try await CohereTranscribeModel.load(
                    directory: directory, device: device))
        }
        if MossFormer2SEModel.handles(config) {
            return .mossFormer2SE(
                try MossFormer2SEModel.load(
                    directory: directory, device: device))
        }
        if SAMAudio.handles(config) {
            let samConfig = SAMAudioModel.loadConfig(from: directory)
            let bundle = try SafeTensorsBundle(directory: directory, device: device)
            let variant = try SAMAudio.variant(for: config)
            let model = try variant.loadModel(
                directory: directory, config: samConfig,
                weights: bundle, device: device)
            return .samAudio(model)
        }
        throw ModelError.unsupportedArchitecture(
            config.architecture ?? config.modelType ?? "<unknown audio model>")
    }
}
