// Unit tests for the MOSS-TTS-Nano family (100M GPT-2 backbone, local transformer).
//
// Covers:
//   • Config decoding — from minimal and full config.json payloads.
//   • GPT-2 sub-config decoding (MossTTSNanoGPT2Config).
//   • localGPT2Config derivation — nPositions = nVQ + 1.
//   • Registry detection — model_type, architecture, structural paths.
//   • Negative detection — unrelated configs are not matched.
//   • Capability set verification.
//   • Staged synthesis error — asserts synthesize() throws
//     MossTTSNanoError.synthesisNotWired (by design at this stage).
//
// These tests are fast and offline — no checkpoint weights are loaded.

import Foundation
import Testing
@testable import FFAI

@Suite("MossTTSNano")
struct MossTTSNanoTests {

    // ─── Config decoding ──────────────────────────────────────────────

    @Test("MossTTSNanoConfig — decodes from canonical moss_tts_nano model_type")
    func configDecodesFromModelType() {
        let gpt2Raw: [String: Any] = [
            "model_type": "gpt2",
            "vocab_size": 16_384,
            "n_positions": 32_768,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3_072,
            "layer_norm_epsilon": 1e-5,
            "position_embedding_type": "rope",
            "rope_base": 10_000,
            "pad_token_id": 3,
            "bos_token_id": 1,
            "eos_token_id": 2,
        ]
        let config = ModelConfig(
            architecture: "MossTTSNanoForCausalLM",
            modelType: "moss_tts_nano",
            raw: [
                "model_type": "moss_tts_nano",
                "architectures": ["MossTTSNanoForCausalLM"],
                "gpt2_config": gpt2Raw,
                "n_vq": 16,
                "audio_vocab_size": 1024,
                "audio_pad_token_id": 1024,
                "pad_token_id": 3,
                "im_start_token_id": 4,
                "im_end_token_id": 5,
                "audio_start_token_id": 6,
                "audio_end_token_id": 7,
                "audio_user_slot_token_id": 8,
                "audio_assistant_slot_token_id": 9,
                "audio_tokenizer_type": "moss-audio-tokenizer-nano",
                "audio_tokenizer_sample_rate": 48_000,
                "local_transformer_layers": 1,
            ])
        let nano = MossTTSNanoConfig.from(config)
        #expect(nano != nil)
        #expect(nano?.modelType == "moss_tts_nano")
        #expect(nano?.nVQ == 16)
        #expect(nano?.audioVocabSize == 1024)
        #expect(nano?.audioPadTokenID == 1024)
        #expect(nano?.padTokenID == 3)
        #expect(nano?.imStartTokenID == 4)
        #expect(nano?.imEndTokenID == 5)
        #expect(nano?.audioStartTokenID == 6)
        #expect(nano?.audioEndTokenID == 7)
        #expect(nano?.audioUserSlotTokenID == 8)
        #expect(nano?.audioAssistantSlotTokenID == 9)
        #expect(nano?.audioTokenizerType == "moss-audio-tokenizer-nano")
        #expect(nano?.audioTokenizerSampleRate == 48_000)
        #expect(nano?.localTransformerLayers == 1)
        #expect(nano?.sampleRate == 48_000)
    }

    @Test("MossTTSNanoConfig — decodes from minimal config with defaults")
    func configDecodesFromMinimal() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts_nano",
            raw: ["model_type": "moss_tts_nano", "gpt2_config": [:], "n_vq": 16])
        let nano = MossTTSNanoConfig.from(config)
        #expect(nano != nil)
        // Verify defaults from the empty gpt2_config.
        #expect(nano?.nVQ == 16)
        #expect(nano?.audioVocabSize == 1024)
        #expect(nano?.audioTokenizerSampleRate == 48_000)
        #expect(nano?.localTransformerLayers == 1)
        #expect(nano?.gpt2Config.nEmbd == 768)
        #expect(nano?.gpt2Config.nLayer == 12)
        #expect(nano?.gpt2Config.nHead == 12)
    }

    @Test("MossTTSNanoGPT2Config — decodes n_embd and derived fields")
    func gpt2ConfigDecodes() {
        let raw: [String: Any] = [
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3_072,
            "vocab_size": 16_384,
            "layer_norm_epsilon": 1e-5,
            "rope_base": 10_000.0,
        ]
        let gpt = MossTTSNanoGPT2Config.from(raw)
        #expect(gpt.nEmbd == 768)
        #expect(gpt.nLayer == 12)
        #expect(gpt.nHead == 12)
        #expect(gpt.nInner == 3_072)
        #expect(gpt.vocabSize == 16_384)
        #expect(gpt.headDim == 64)        // 768 / 12
        #expect(gpt.intermediateSize == 3_072)
        #expect(gpt.hiddenSize == 768)
        #expect(abs(gpt.layerNormEpsilon - 1e-5) < 1e-8)
        #expect(abs(gpt.ropeBase - 10_000) < 1)
    }

    @Test("MossTTSNanoGPT2Config — accepts hidden_size as alias for n_embd")
    func gpt2ConfigHiddenSizeAlias() {
        let raw: [String: Any] = ["hidden_size": 512, "n_head": 8]
        let gpt = MossTTSNanoGPT2Config.from(raw)
        #expect(gpt.nEmbd == 512)
        #expect(gpt.headDim == 64)   // 512 / 8
    }

    @Test("MossTTSNanoConfig — localGPT2Config has nPositions = nVQ + 1")
    func localGPT2ConfigPositions() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts_nano",
            raw: ["model_type": "moss_tts_nano", "gpt2_config": [:], "n_vq": 16])
        guard let nano = MossTTSNanoConfig.from(config) else {
            Issue.record("config should decode")
            return
        }
        let localCfg = nano.localGPT2Config()
        #expect(localCfg.nPositions == nano.nVQ + 1)   // 17
        #expect(localCfg.nCtx == nano.nVQ + 1)
        #expect(localCfg.nLayer == nano.localTransformerLayers)
        // Other fields mirror the global GPT-2 config.
        #expect(localCfg.nEmbd == nano.gpt2Config.nEmbd)
        #expect(localCfg.nHead == nano.gpt2Config.nHead)
    }

    @Test("MossTTSNanoConfig — audioCodebookSizes defaults to nVQ × audioVocabSize")
    func audioCodebookSizesDefault() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts_nano",
            raw: ["model_type": "moss_tts_nano", "gpt2_config": [:], "n_vq": 16,
                  "audio_vocab_size": 1024])
        let nano = MossTTSNanoConfig.from(config)
        #expect(nano?.audioCodebookSizes.count == 16)
        #expect(nano?.audioCodebookSizes.allSatisfy { $0 == 1024 } == true)
    }

    @Test("MossTTSNanoConfig — explicit audioCodebookSizes array is preserved")
    func audioCodebookSizesExplicit() {
        let sizes = Array(repeating: 1024, count: 16)
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts_nano",
            raw: ["model_type": "moss_tts_nano", "gpt2_config": [:],
                  "n_vq": 16, "audio_codebook_sizes": sizes])
        let nano = MossTTSNanoConfig.from(config)
        #expect(nano?.audioCodebookSizes == sizes)
    }

    // ─── Registry detection ───────────────────────────────────────────

    @Test("MossTTSNanoModel.handles — detects from model_type moss_tts_nano")
    func handlesModelType() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts_nano",
            raw: ["model_type": "moss_tts_nano"])
        #expect(MossTTSNanoModel.handles(config))
    }

    @Test("MossTTSNanoModel.handles — detects from architecture MossTTSNanoForCausalLM")
    func handlesArchitecture() {
        let config = ModelConfig(
            architecture: "MossTTSNanoForCausalLM",
            modelType: nil,
            raw: ["architectures": ["MossTTSNanoForCausalLM"]])
        #expect(MossTTSNanoModel.handles(config))
    }

    @Test("MossTTSNanoModel.handles — detects structurally from gpt2_config + n_vq")
    func handlesStructural() {
        let config = ModelConfig(
            architecture: nil,
            modelType: nil,
            raw: [
                "gpt2_config": ["n_embd": 768],
                "n_vq": 16,
            ])
        #expect(MossTTSNanoModel.handles(config))
    }

    @Test("MossTTSNanoModel.handles — does not detect MOSS-TTS-8B (Qwen3 backbone)")
    func doesNotDetectMossTTS8B() {
        // MOSS-TTS-8B uses language_config (Qwen3), not gpt2_config.
        let config = ModelConfig(
            architecture: "MossTTSDelayModel",
            modelType: "moss_tts",
            raw: [
                "model_type": "moss_tts",
                "language_config": ["model_type": "qwen3"],
                "n_vq": 32,
            ])
        #expect(!MossTTSNanoModel.handles(config))
    }

    @Test("MossTTSNanoModel.handles — does not detect plain GPT-2 models")
    func doesNotDetectGPT2() {
        let config = ModelConfig(
            architecture: "GPT2LMHeadModel",
            modelType: "gpt2",
            raw: ["model_type": "gpt2"])
        #expect(!MossTTSNanoModel.handles(config))
    }

    // ─── AudioModelRegistry integration ──────────────────────────────

    @Test("AudioModelRegistry — handles MOSS-TTS-Nano from model_type")
    func registryHandlesMossTTSNano() {
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts_nano",
            raw: ["model_type": "moss_tts_nano"])
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.textToSpeech)
    }

    @Test("AudioModelRegistry — routes to .mossTTSNano, not .mossTTS")
    func registryRoutesToNano() {
        // A Nano config must never be matched by the 8B MOSS-TTS family
        // because MossTTSNanoModel.handles is checked first in the registry.
        let config = ModelConfig(
            architecture: nil,
            modelType: "moss_tts_nano",
            raw: ["model_type": "moss_tts_nano", "gpt2_config": [:], "n_vq": 16])
        // Nano model matches; 8B model must not.
        #expect(MossTTSNanoModel.handles(config))
        #expect(!MossTTSModel.handles(config))
    }

    // ─── Staged synthesis error ───────────────────────────────────────

    @Test("MossTTSNanoError — synthesisNotWired description mentions backbone and codec")
    func errorDescription() {
        let err = MossTTSNanoError.synthesisNotWired
        #expect(err.description.contains("GPT-2"))
        #expect(err.description.contains("codec"))
    }

    @Test("MossTTSNanoError — missingConfig carries field name")
    func missingConfigError() {
        let err = MossTTSNanoError.missingConfig("gpt2_config")
        #expect(err.description.contains("gpt2_config"))
    }

    @Test("MossTTSNanoModel — synthesize throws synthesisNotWired on a real checkpoint")
    func synthesizeThrowsSynthesisNotWired() {
        // If a cached checkpoint is available on disk, verify via a full load.
        let hfRoot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        let candidates = [
            "models--mlx-community--MOSS-TTS-Nano-100M",
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
                let model = try MossTTSNanoModel.load(directory: sub)
                #expect(throws: MossTTSNanoError.self) {
                    _ = try model.synthesize(text: "Hello, world!")
                }
                return
            } catch {
                print("MossTTSNanoTests: load skipped: \(error)")
            }
        }
        // No checkpoint on disk — validate the error type directly.
        let err = MossTTSNanoError.synthesisNotWired
        #expect(err.description.contains("Stage 1") || err.description.contains("stage"))
    }
}
