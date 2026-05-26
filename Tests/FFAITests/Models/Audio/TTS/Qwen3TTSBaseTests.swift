// Unit tests for Qwen3TTSBase config parsing, family detection, and
// registry routing. No GPU / HF-hub access required — all tests run
// from synthetic JSON documents.
//
// Key checks:
//  • Config parses required Qwen3 transformer fields.
//  • `handles` correctly accepts VyvoTTS-style checkpoints and rejects
//    plain text Qwen3 checkpoints and the 12Hz Flash variant.
//  • `AudioModelRegistry.handles` routes consistently with the model's
//    own `handles`.
//  • Detection does NOT collide with the Qwen3TTS (12Hz Flash) family,
//    which is distinguished by `talker_config` + `speaker_encoder_config`.

import Foundation
import Testing
@testable import FFAI

// ─── Helpers ─────────────────────────────────────────────────────────

private func makeConfig(_ json: String) throws -> ModelConfig {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("qwen3-tts-base-cfg-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }
    try FileManager.default.createDirectory(at: dir,
                                            withIntermediateDirectories: true)
    try json.write(to: dir.appendingPathComponent("config.json"),
                   atomically: true, encoding: .utf8)
    return try ModelConfig.load(from: dir)
}

/// Minimal config for a VyvoTTS / Qwen3 base TTS checkpoint.
/// Vocabulary size of 163 840 (> 151 936) + sample_rate = audio.
private let vyvoTTSJSON = """
{
  "model_type": "qwen3",
  "architectures": ["Qwen3ForCausalLM"],
  "hidden_size": 2048,
  "intermediate_size": 11008,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "rms_norm_eps": 1e-6,
  "rope_theta": 1000000,
  "vocab_size": 163840,
  "max_position_embeddings": 32768,
  "sample_rate": 24000,
  "tie_word_embeddings": false
}
"""

/// Plain Qwen3 text checkpoint — same arch but base vocab and no sample_rate.
private let plainQwen3JSON = """
{
  "model_type": "qwen3",
  "architectures": ["Qwen3ForCausalLM"],
  "hidden_size": 2048,
  "intermediate_size": 11008,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "rms_norm_eps": 1e-6,
  "vocab_size": 151936,
  "tie_word_embeddings": false
}
"""

/// Qwen3TTS Flash / 12Hz variant — has talker_config + speaker_encoder_config.
/// Must NOT be routed to Qwen3TTSBase.
private let qwen3TTSFlashJSON = """
{
  "model_type": "qwen3_tts",
  "architectures": ["Qwen3TTSForConditionalGeneration"],
  "talker_config": {
    "vocab_size": 3072,
    "hidden_size": 1024
  },
  "speaker_encoder_config": {
    "channels": 512
  },
  "sample_rate": 24000,
  "vocab_size": 151936,
  "hidden_size": 1024,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 64,
  "intermediate_size": 4096,
  "rms_norm_eps": 1e-6
}
"""

/// Config with the explicit Qwen3TTSBase model_type.
private let explicitQwen3TTSBaseJSON = """
{
  "model_type": "qwen3_tts_base",
  "architectures": ["Qwen3TTSBaseForConditionalGeneration"],
  "hidden_size": 2048,
  "intermediate_size": 11008,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "rms_norm_eps": 1e-6,
  "vocab_size": 163840,
  "max_position_embeddings": 32768,
  "sample_rate": 24000,
  "tie_word_embeddings": false
}
"""

// ─── Tests ───────────────────────────────────────────────────────────

@Suite("Qwen3TTSBase")
struct Qwen3TTSBaseTests {

    // ── Config parsing ────────────────────────────────────────────────

    @Test("config parses required transformer fields from VyvoTTS JSON")
    func config_parsesRequiredFields() throws {
        let cfg = try makeConfig(vyvoTTSJSON)
        guard let tts = Qwen3TTSBaseConfig.from(cfg) else {
            Issue.record("Qwen3TTSBaseConfig.from returned nil for a valid VyvoTTS config")
            return
        }
        #expect(tts.hidden == 2048)
        #expect(tts.intermediate == 11008)
        #expect(tts.nLayers == 28)
        #expect(tts.nHeads == 16)
        #expect(tts.nKVHeads == 8)
        #expect(tts.headDim == 128)
        #expect(tts.vocabSize == 163_840)
        #expect(tts.sampleRate == 24_000)
        #expect(tts.tieWordEmbeddings == false)
    }

    @Test("config returns nil for missing required fields")
    func config_nilOnMissingFields() throws {
        // Missing hidden_size — must return nil.
        let cfg = try makeConfig(#"{"model_type":"qwen3","vocab_size":163840,"sample_rate":24000}"#)
        let result = Qwen3TTSBaseConfig.from(cfg)
        #expect(result == nil)
    }

    @Test("config uses sample_rate default of 24000 when absent")
    func config_defaultSampleRate() throws {
        let json = """
        {
          "model_type": "qwen3_tts_base",
          "hidden_size": 2048,
          "intermediate_size": 11008,
          "num_hidden_layers": 28,
          "num_attention_heads": 16,
          "head_dim": 128,
          "rms_norm_eps": 1e-6,
          "vocab_size": 163840
        }
        """
        let cfg = try makeConfig(json)
        let tts = Qwen3TTSBaseConfig.from(cfg)
        // No sample_rate key → default 24000
        #expect(tts?.sampleRate == 24_000)
    }

    // ── Detection ─────────────────────────────────────────────────────

    @Test("handles — VyvoTTS structural config is accepted")
    func handles_vyvoTTSStructural() throws {
        let cfg = try makeConfig(vyvoTTSJSON)
        #expect(Qwen3TTSBaseModel.handles(cfg) == true)
    }

    @Test("handles — explicit model_type qwen3_tts_base is accepted")
    func handles_explicitModelType() throws {
        let cfg = try makeConfig(explicitQwen3TTSBaseJSON)
        #expect(Qwen3TTSBaseModel.handles(cfg) == true)
    }

    @Test("handles — plain Qwen3 text checkpoint (base vocab, no sample_rate) is rejected")
    func handles_plainQwen3Rejected() throws {
        let cfg = try makeConfig(plainQwen3JSON)
        #expect(Qwen3TTSBaseModel.handles(cfg) == false)
    }

    @Test("handles — Qwen3TTS Flash (talker_config + speaker_encoder_config) is rejected")
    func handles_qwen3TTSFlashRejected() throws {
        let cfg = try makeConfig(qwen3TTSFlashJSON)
        // Must NOT match — the 12Hz Flash variant has talker_config.
        #expect(Qwen3TTSBaseModel.handles(cfg) == false)
    }

    @Test("handles — Qwen3 with extended audio vocab is accepted even without sample_rate")
    func handles_audioVocabAcceptedWithoutSampleRate() throws {
        // VyvoTTS-EN-Beta-4bit ships without a top-level `sample_rate`
        // field. The extended vocabulary (> 151 936) is the distinguishing
        // codec-token marker; the loader defaults the rate to 24 kHz when
        // the config omits it. Detection must accept this shape.
        let json = """
        {
          "model_type": "qwen3",
          "architectures": ["Qwen3ForCausalLM"],
          "hidden_size": 2048,
          "intermediate_size": 11008,
          "num_hidden_layers": 28,
          "num_attention_heads": 16,
          "head_dim": 128,
          "rms_norm_eps": 1e-6,
          "vocab_size": 163840
        }
        """
        let cfg = try makeConfig(json)
        #expect(Qwen3TTSBaseModel.handles(cfg) == true)
    }

    // ── Registry ──────────────────────────────────────────────────────

    @Test("AudioModelRegistry.handles returns true for VyvoTTS config")
    func registry_handles_vyvoTTS() throws {
        let cfg = try makeConfig(vyvoTTSJSON)
        #expect(AudioModelRegistry.handles(cfg) == true)
    }

    @Test("AudioModelRegistry.handles returns false for plain text Qwen3")
    func registry_handles_plainQwen3Rejected() throws {
        let cfg = try makeConfig(plainQwen3JSON)
        #expect(AudioModelRegistry.handles(cfg) == false)
    }

    @Test("AudioModelRegistry.capabilities returns textToSpeech for VyvoTTS")
    func registry_capabilities_textToSpeech() throws {
        let cfg = try makeConfig(vyvoTTSJSON)
        let caps = AudioModelRegistry.capabilities(for: cfg)
        #expect(caps == Capability.textToSpeech)
    }

    @Test("AudioModelRegistry.capabilities returns nil for plain Qwen3 text model")
    func registry_capabilities_nilForTextModel() throws {
        let cfg = try makeConfig(plainQwen3JSON)
        let caps = AudioModelRegistry.capabilities(for: cfg)
        #expect(caps == nil)
    }

    // ── Special-token constants ────────────────────────────────────────

    @Test("Qwen3TTSBaseTokens — audioTokensStart is 10 beyond the base vocab size")
    func tokens_audioStart() {
        #expect(Qwen3TTSBaseTokens.audioTokensStart
                == Qwen3TTSBaseTokens.baseVocabSize + 10)
    }

    @Test("Qwen3TTSBaseTokens — endOfSpeech is distinct from startOfSpeech")
    func tokens_speechMarkersDistinct() {
        #expect(Qwen3TTSBaseTokens.startOfSpeech
                != Qwen3TTSBaseTokens.endOfSpeech)
        #expect(Qwen3TTSBaseTokens.endOfSpeech
                > Qwen3TTSBaseTokens.startOfSpeech)
    }

    // ── Code de-interleaving ──────────────────────────────────────────

    @Test("deinterleaveSNACCodes — three frames yield correct plane lengths")
    func deinterleave_planeShapes() {
        // 3 frames × 7 = 21 tokens → layer1 = 3, layer2 = 6, layer3 = 12
        let tokens = Array(0..<21)
        let planes = Qwen3TTSBaseModel.deinterleaveSNACCodes(tokens)
        #expect(planes.count == 3)
        #expect(planes[0].count == 3)
        #expect(planes[1].count == 6)
        #expect(planes[2].count == 12)
    }

    @Test("deinterleaveSNACCodes — partial trailing frame is dropped")
    func deinterleave_partialFrameDropped() {
        // 25 tokens → 3 full frames (21), 4 leftover dropped
        let tokens = Array(0..<25)
        let planes = Qwen3TTSBaseModel.deinterleaveSNACCodes(tokens)
        #expect(planes[0].count == 3)
        #expect(planes[1].count == 6)
        #expect(planes[2].count == 12)
    }

    @Test("deinterleaveSNACCodes — empty input returns empty planes")
    func deinterleave_empty() {
        let planes = Qwen3TTSBaseModel.deinterleaveSNACCodes([])
        #expect(planes.count == 3)
        #expect(planes[0].isEmpty)
        #expect(planes[1].isEmpty)
        #expect(planes[2].isEmpty)
    }

    @Test("deinterleaveSNACCodes — SNAC up-sampling invariant: layer2 = 2x, layer3 = 4x layer1")
    func deinterleave_upSamplingInvariant() {
        let tokens = Array(0..<70)  // 10 complete frames
        let planes = Qwen3TTSBaseModel.deinterleaveSNACCodes(tokens)
        #expect(planes[1].count == 2 * planes[0].count)
        #expect(planes[2].count == 4 * planes[0].count)
    }

    // ── Error descriptions ────────────────────────────────────────────

    @Test("Qwen3TTSBaseError descriptions are non-empty and meaningful")
    func errorDescriptions() {
        let codecErr = Qwen3TTSBaseError.codecUnavailable
        let noAudio  = Qwen3TTSBaseError.noAudioCodes
        let missCfg  = Qwen3TTSBaseError.missingConfig
        #expect(String(describing: codecErr).contains("SNAC"))
        #expect(String(describing: noAudio).contains("audio-code"))
        #expect(String(describing: missCfg).contains("config"))
    }
}
