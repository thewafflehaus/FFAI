// KokoroTests — unit tests for the Kokoro TTS family: config parsing,
// the iSTFTNet vocoder configuration, the voice catalogue, family
// detection, and registry routing.
//
// Validates:
//   * KokoroConfig.from(_:) decodes the canonical kokoro / style_tts2
//     config layout (including the nested istftnet block).
//   * KokoroModel.handles(_:) accepts both kokoro and style_tts2
//     model_types and structural fallback (istftnet + n_token).
//   * AudioModelRegistry routes Kokoro configs to Capability.textToSpeech.
//   * KokoroModel.availableVoices catalogue has the expected default and
//     covers multiple language prefixes.
//   * KokoroError.description renders the expected family prefix.

import Foundation
import Testing
@testable import FFAI

@Suite("Kokoro")
struct KokoroTests {

    // ─── Config parsing ──────────────────────────────────────────────────

    @Test("KokoroConfig.from — decodes canonical config with nested istftnet")
    func configDecodesCanonical() {
        let raw: [String: Any] = [
            "model_type": "kokoro",
            "n_token": 178,
            "hidden_dim": 512,
            "n_mels": 80,
            "sample_rate": 24_000,
            "istftnet": [
                "gen_istft_n_fft": 20,
                "gen_istft_hop_size": 5,
            ] as [String: Any],
        ]
        let config = ModelConfig(architecture: nil, modelType: "kokoro", raw: raw)
        let kc = KokoroConfig.from(config)
        #expect(kc != nil)
        #expect(kc?.nToken == 178)
        #expect(kc?.hidden == 512)
        #expect(kc?.nMels == 80)
        #expect(kc?.sampleRate == 24_000)
        #expect(kc?.istftNFFT == 20)
        #expect(kc?.istftHop == 5)
    }

    @Test("KokoroConfig.from — fills istftnet defaults when block absent")
    func configFillsIstftDefaults() {
        let raw: [String: Any] = [
            "model_type": "kokoro",
            "n_token": 178,
            "hidden_dim": 512,
        ]
        let config = ModelConfig(architecture: nil, modelType: "kokoro", raw: raw)
        let kc = KokoroConfig.from(config)
        #expect(kc?.istftNFFT == 20)
        #expect(kc?.istftHop == 5)
        #expect(kc?.sampleRate == 24_000)
        #expect(kc?.nMels == 80)
    }

    @Test("KokoroConfig.from — returns nil when required fields are missing")
    func configReturnsNilForMissing() {
        // Missing n_token + hidden_dim.
        let config = ModelConfig(architecture: nil, modelType: "kokoro",
                                 raw: ["model_type": "kokoro"])
        #expect(KokoroConfig.from(config) == nil)
    }

    // ─── Registry detection ──────────────────────────────────────────────

    @Test("KokoroModel.modelTypes — contains kokoro and style_tts2")
    func modelTypesContents() {
        let types = KokoroModel.modelTypes
        #expect(types.contains("kokoro"))
        #expect(types.contains("style_tts2"))
    }

    @Test("KokoroModel.handles — true for kokoro model_type")
    func handlesByModelType() {
        let config = ModelConfig(architecture: nil, modelType: "kokoro",
                                 raw: ["model_type": "kokoro"])
        #expect(KokoroModel.handles(config))
    }

    @Test("KokoroModel.handles — structural fallback via istftnet + n_token")
    func handlesStructural() {
        let raw: [String: Any] = [
            "istftnet": [:] as [String: Any],
            "n_token": 178,
        ]
        let config = ModelConfig(architecture: nil, modelType: nil, raw: raw)
        #expect(KokoroModel.handles(config))
    }

    @Test("KokoroModel.handles — false for unrelated text model")
    func handlesFalseForTextModel() {
        let config = ModelConfig(architecture: "LlamaForCausalLM",
                                 modelType: "llama",
                                 raw: ["model_type": "llama"])
        #expect(!KokoroModel.handles(config))
    }

    // ─── Voice catalogue ────────────────────────────────────────────────

    @Test("KokoroModel.defaultVoice — is af_heart")
    func defaultVoiceConstant() {
        #expect(KokoroModel.defaultVoice == "af_heart")
    }

    @Test("KokoroModel.availableVoices — non-empty + defaultVoice is in the list")
    func availableVoicesContents() {
        let voices = KokoroModel.availableVoices
        #expect(voices.count > 0)
        #expect(voices.contains(KokoroModel.defaultVoice))
        // Every voice id is non-empty.
        for v in voices {
            #expect(!v.isEmpty)
        }
    }

    @Test("KokoroModel.availableVoices — covers multiple language prefixes")
    func availableVoicesLanguageCoverage() {
        let voices = KokoroModel.availableVoices
        // The catalogue mixes American / British English with other
        // language prefixes (bf, hf, jf, zf, …). Verify at least three
        // distinct families are present.
        let prefixes = Set(voices.compactMap { $0.split(separator: "_").first.map(String.init) })
        #expect(prefixes.count >= 3)
    }

    // ─── AudioModelRegistry routing ──────────────────────────────────────

    @Test("AudioModelRegistry.capabilities — Kokoro maps to textToSpeech")
    func registryCapability() {
        let config = ModelConfig(architecture: nil, modelType: "kokoro",
                                 raw: ["model_type": "kokoro"])
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.textToSpeech)
    }

    // ─── Error stringification ───────────────────────────────────────────

    @Test("KokoroError.description — acousticFrontEndUnavailable mentions Kokoro")
    func errorDescription() {
        let err = KokoroError.acousticFrontEndUnavailable
        let desc = err.description
        #expect(desc.contains("Kokoro"))
        #expect(desc.contains("acoustic") || desc.contains("spectrogram"))
    }
}
