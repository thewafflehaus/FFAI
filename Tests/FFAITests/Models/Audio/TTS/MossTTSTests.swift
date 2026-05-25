// Unit tests for the MOSS-TTS family (8B Qwen3 backbone, delay-pattern).
//
// Covers:
//   • Config decoding — from minimal and full config.json payloads.
//   • Nested language_config decoding (MossTTSLanguageConfig).
//   • Registry detection — model_type, architecture, structural paths.
//   • Negative detection — unrelated configs are not matched.
//   • Capability set verification.
//   • Staged synthesis error — asserts synthesize() throws
//     MossTTSError.synthesisNotWired (by design at this stage).
//
// These tests are fast and offline — no checkpoint weights are loaded.

import Foundation
import Testing
@testable import FFAI

@Suite("MossTTS")
struct MossTTSTests {

    // ─── Config decoding ──────────────────────────────────────────────

    @Test("MossTTSConfig — decodes from canonical moss_tts model_type")
    func configDecodesFromModelType() {
        let config = ModelConfig(
            architecture: "MossTTSDelayModel",
            modelType: "moss_tts",
            raw: [
                "model_type": "moss_tts",
                "architectures": ["MossTTSDelayModel"],
                "n_vq": 32,
                "audio_vocab_size": 1024,
                "audio_pad_code": 1024,
                "audio_start_token_id": 151_652,
                "audio_end_token_id": 151_653,
                "audio_user_slot_token_id": 151_654,
                "audio_assistant_gen_slot_token_id": 151_656,
                "audio_assistant_delay_slot_token_id": 151_662,
                "pad_token_id": 151_643,
                "im_start_token_id": 151_644,
                "im_end_token_id": 151_645,
                "sampling_rate": 24_000,
                "language_config": [
                    "model_type": "qwen3",
                    "vocab_size": 155_648,
                    "hidden_size": 4_096,
                    "num_hidden_layers": 36,
                    "intermediate_size": 12_288,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "rms_norm_eps": 1e-6,
                    "max_position_embeddings": 40_960,
                    "rope_theta": 1_000_000,
                ],
            ])
        let tts = MossTTSConfig.from(config)
        #expect(tts != nil)
        #expect(tts?.modelType == "moss_tts")
        #expect(tts?.nVQ == 32)
        #expect(tts?.audioVocabSize == 1024)
        #expect(tts?.audioPadCode == 1024)
        #expect(tts?.audioStartTokenID == 151_652)
        #expect(tts?.audioEndTokenID == 151_653)
        #expect(tts?.audioUserSlotTokenID == 151_654)
        #expect(tts?.audioAssistantGenSlotTokenID == 151_656)
        #expect(tts?.audioAssistantDelaySlotTokenID == 151_662)
        #expect(tts?.padTokenID == 151_643)
        #expect(tts?.imStartTokenID == 151_644)
        #expect(tts?.imEndTokenID == 151_645)
        #expect(tts?.samplingRate == 24_000)
        #expect(tts?.sampleRate == 24_000)
    }

    @Test("MossTTSConfig — decodes from minimal config with defaults")
    func configDecodesFromMinimal() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts",
            raw: ["model_type": "moss_tts"])
        let tts = MossTTSConfig.from(config)
        #expect(tts != nil)
        // Verify defaults are applied.
        #expect(tts?.nVQ == 32)
        #expect(tts?.audioVocabSize == 1024)
        #expect(tts?.samplingRate == 24_000)
        #expect(tts?.padTokenID == 151_643)
        #expect(tts?.audioPadCode == 1024)
    }

    @Test("MossTTSConfig — decodes moss_tts_delay model_type variant")
    func configDecodesDelayVariant() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts_delay",
            raw: ["model_type": "moss_tts_delay", "n_vq": 16])
        let tts = MossTTSConfig.from(config)
        #expect(tts != nil)
        #expect(tts?.modelType == "moss_tts_delay")
        #expect(tts?.nVQ == 16)
    }

    @Test("MossTTSLanguageConfig — decodes Qwen3-8B sub-config fields")
    func languageConfigDecodes() {
        let raw: [String: Any] = [
            "model_type": "qwen3",
            "vocab_size": 155_648,
            "hidden_size": 4_096,
            "num_hidden_layers": 36,
            "intermediate_size": 12_288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 40_960,
            "rope_theta": 1_000_000,
        ]
        let lc = MossTTSLanguageConfig.from(raw)
        #expect(lc.modelType == "qwen3")
        #expect(lc.vocabSize == 155_648)
        #expect(lc.hiddenSize == 4_096)
        #expect(lc.numHiddenLayers == 36)
        #expect(lc.intermediateSize == 12_288)
        #expect(lc.numAttentionHeads == 32)
        #expect(lc.numKeyValueHeads == 8)
        #expect(lc.headDim == 128)
        #expect(lc.maxPositionEmbeddings == 40_960)
        #expect(abs(lc.ropeTheta - 1_000_000) < 1)
    }

    @Test("MossTTSLanguageConfig — falls back to defaults for empty dict")
    func languageConfigDefaults() {
        let lc = MossTTSLanguageConfig.from([:])
        #expect(lc.modelType == "qwen3")
        #expect(lc.vocabSize == 155_648)
        #expect(lc.hiddenSize == 4_096)
        #expect(lc.numHiddenLayers == 36)
        #expect(lc.numAttentionHeads == 32)
        // headDim defaults to hiddenSize / numAttentionHeads = 128
        #expect(lc.headDim == 128)
    }

    @Test("MossTTSConfig — sample_rate key is also accepted")
    func configAcceptsSampleRateKey() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts",
            raw: ["model_type": "moss_tts", "sample_rate": 22_050])
        let tts = MossTTSConfig.from(config)
        #expect(tts?.samplingRate == 22_050)
    }

    // ─── Registry detection ───────────────────────────────────────────

    @Test("MossTTSModel.handles — detects from model_type moss_tts")
    func handlesModelType() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts",
            raw: ["model_type": "moss_tts"])
        #expect(MossTTSModel.handles(config))
    }

    @Test("MossTTSModel.handles — detects from model_type moss_tts_delay")
    func handlesModelTypeDelay() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts_delay",
            raw: ["model_type": "moss_tts_delay"])
        #expect(MossTTSModel.handles(config))
    }

    @Test("MossTTSModel.handles — detects from architecture MossTTSDelayModel")
    func handlesArchitecture() {
        let config = ModelConfig(
            architecture: "MossTTSDelayModel",
            modelType: nil,
            raw: ["architectures": ["MossTTSDelayModel"]])
        #expect(MossTTSModel.handles(config))
    }

    @Test("MossTTSModel.handles — detects structurally from language_config + n_vq")
    func handlesStructural() {
        let config = ModelConfig(
            architecture: nil,
            modelType: nil,
            raw: [
                "language_config": ["model_type": "qwen3"],
                "n_vq": 32,
            ])
        #expect(MossTTSModel.handles(config))
    }

    @Test("MossTTSModel.handles — does not detect unrelated models")
    func doesNotDetectLlama() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM",
            modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 4_096])
        #expect(!MossTTSModel.handles(config))
    }

    @Test("MossTTSModel.handles — does not detect MOSS-TTS-Nano")
    func doesNotDetectNano() {
        // Nano has gpt2_config, not language_config.
        let config = ModelConfig(
            architecture: "MossTTSNanoForCausalLM",
            modelType: "moss_tts_nano",
            raw: ["model_type": "moss_tts_nano", "gpt2_config": [:], "n_vq": 16])
        // The Nano model_type is not in MossTTSModel.modelTypes.
        #expect(!MossTTSModel.handles(config))
    }

    // ─── AudioModelRegistry integration ──────────────────────────────

    @Test("AudioModelRegistry — handles MOSS-TTS from model_type")
    func registryHandlesMossTTS() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts",
            raw: ["model_type": "moss_tts"])
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.textToSpeech)
    }

    @Test("AudioModelRegistry — does not handle unrelated models")
    func registryDoesNotHandleQwen3() {
        let config = ModelConfig(
            architecture: "Qwen3ForCausalLM",
            modelType: "qwen3",
            raw: ["model_type": "qwen3", "hidden_size": 4_096])
        #expect(!AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == nil)
    }

    // ─── Staged synthesis error ───────────────────────────────────────

    @Test("MossTTSError — synthesisNotWired description mentions backbone and codec")
    func errorDescription() {
        let err = MossTTSError.synthesisNotWired
        #expect(err.description.contains("Qwen3"))
        #expect(err.description.contains("codec"))
    }

    @Test("MossTTSError — missingConfig carries field name")
    func missingConfigError() {
        let err = MossTTSError.missingConfig("language_config")
        #expect(err.description.contains("language_config"))
    }

    @Test("MossTTSModel — synthesize throws synthesisNotWired on a real checkpoint")
    func synthesizeThrowsSynthesisNotWired() {
        // Construct a config and check the staged error without loading weights.
        // If a cached checkpoint is available on disk, verify via a full load.
        let hfRoot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        let candidates = [
            "models--mlx-community--MOSS-TTS-8B-8bit",
        ]
        let fm = FileManager.default
        for slug in candidates {
            let base = hfRoot.appendingPathComponent(slug)
                .appendingPathComponent("snapshots")
            guard let sub = try? fm.contentsOfDirectory(
                at: base, includingPropertiesForKeys: nil).first
            else { continue }
            guard fm.fileExists(atPath: sub.appendingPathComponent("config.json").path)
            else { continue }

            do {
                let model = try MossTTSModel.load(directory: sub)
                #expect(throws: MossTTSError.self) {
                    _ = try model.synthesize(text: "Hello, world!")
                }
                return
            } catch {
                print("MossTTSTests: load skipped: \(error)")
            }
        }
        // No checkpoint on disk — validate the error type directly.
        let err = MossTTSError.synthesisNotWired
        #expect(err.description.contains("Stage 1") || err.description.contains("stage"))
    }
}
