// EchoTTSTests — unit tests for EchoTTS config parsing and registry
// detection. Exercises the config decode path and the AudioModelRegistry
// `handles` + `capabilities` surface on synthetic config dictionaries.
// All tests run offline (no checkpoint load, no GPU forward pass).

import Foundation
import Testing
@testable import FFAI

@Suite("EchoTTS")
struct EchoTTSTests {

    // ─── EchoDiTConfig ────────────────────────────────────────────────────

    @Test("EchoDiTConfig — decodes the published base model geometry")
    func ditConfigDecodes() {
        // Raw dict matching the `dit` sub-object from the mlx-community
        // echo-tts-base config.json.
        let raw: [String: Any] = [
            "latent_size": 80,
            "model_size": 2048,
            "num_layers": 24,
            "num_heads": 16,
            "intermediate_size": 5888,
            "text_vocab_size": 256,
            "text_model_size": 1280,
            "text_num_layers": 14,
            "text_num_heads": 10,
            "text_intermediate_size": 3328,
            "speaker_patch_size": 4,
            "speaker_model_size": 1280,
            "speaker_num_layers": 14,
            "speaker_num_heads": 10,
            "speaker_intermediate_size": 3328,
            "timestep_embed_size": 512,
            "adaln_rank": 256,
            "norm_eps": 1e-5,
        ]
        let dit = EchoDiTConfig.from(raw)
        #expect(dit.latentSize == 80)
        #expect(dit.modelSize == 2048)
        #expect(dit.numLayers == 24)
        #expect(dit.numHeads == 16)
        #expect(dit.intermediateSize == 5888)
        #expect(dit.textVocabSize == 256)
        #expect(dit.textModelSize == 1280)
        #expect(dit.textNumLayers == 14)
        #expect(dit.textNumHeads == 10)
        #expect(dit.textIntermediateSize == 3328)
        #expect(dit.speakerPatchSize == 4)
        #expect(dit.speakerModelSize == 1280)
        #expect(dit.speakerNumLayers == 14)
        #expect(dit.speakerNumHeads == 10)
        #expect(dit.speakerIntermediateSize == 3328)
        #expect(dit.timestepEmbedSize == 512)
        #expect(dit.adalnRank == 256)
        #expect(dit.normEps > 0)
    }

    @Test("EchoDiTConfig — falls back to defaults for an empty dict")
    func ditConfigDefaults() {
        let dit = EchoDiTConfig.from([:])
        #expect(dit.latentSize == 80)
        #expect(dit.modelSize == 2048)
        #expect(dit.numLayers == 24)
        #expect(dit.numHeads == 16)
        #expect(dit.speakerPatchSize == 4)
    }

    // ─── EchoSamplerConfig ────────────────────────────────────────────────

    @Test("EchoSamplerConfig — decodes the published sampler config")
    func samplerConfigDecodes() {
        let raw: [String: Any] = [
            "num_steps": 40,
            "cfg_scale_text": 3.0,
            "cfg_scale_speaker": 8.0,
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "sequence_length": 640,
        ]
        let sampler = EchoSamplerConfig.from(raw)
        #expect(sampler.numSteps == 40)
        #expect(sampler.cfgScaleText == 3.0)
        #expect(sampler.cfgScaleSpeaker == 8.0)
        #expect(sampler.cfgMinT == 0.5)
        #expect(sampler.cfgMaxT == 1.0)
        #expect(sampler.sequenceLength == 640)
        // Optional fields absent from dict → nil.
        #expect(sampler.rescaleK == nil)
        #expect(sampler.rescaleSigma == nil)
        #expect(sampler.speakerKvScale == nil)
        #expect(sampler.speakerKvMaxLayers == nil)
        #expect(sampler.speakerKvMinT == nil)
    }

    @Test("EchoSamplerConfig — decodes optional speaker KV scaling fields")
    func samplerConfigWithKVScale() {
        let raw: [String: Any] = [
            "num_steps": 10,
            "speaker_kv_scale": 1.1,
            "speaker_kv_max_layers": 12,
            "speaker_kv_min_t": 0.5,
        ]
        let sampler = EchoSamplerConfig.from(raw)
        #expect(sampler.numSteps == 10)
        #expect(sampler.speakerKvScale == 1.1)
        #expect(sampler.speakerKvMaxLayers == 12)
        #expect(sampler.speakerKvMinT == 0.5)
    }

    @Test("EchoSamplerConfig — falls back to defaults for an empty dict")
    func samplerConfigDefaults() {
        let sampler = EchoSamplerConfig.from([:])
        #expect(sampler.numSteps == 40)
        #expect(sampler.sequenceLength == 640)
        #expect(sampler.truncationFactor == 0.96)
    }

    // ─── EchoTTSConfig ────────────────────────────────────────────────────

    @Test("EchoTTSConfig — decodes a full config.json")
    func echoConfigDecodes() {
        let ditRaw: [String: Any] = [
            "latent_size": 80,
            "model_size": 2048,
            "num_layers": 24,
            "num_heads": 16,
            "intermediate_size": 5888,
            "text_vocab_size": 256,
            "text_model_size": 1280,
            "text_num_layers": 14,
            "text_num_heads": 10,
            "text_intermediate_size": 3328,
            "speaker_patch_size": 4,
            "speaker_model_size": 1280,
            "speaker_num_layers": 14,
            "speaker_num_heads": 10,
            "speaker_intermediate_size": 3328,
            "timestep_embed_size": 512,
            "adaln_rank": 256,
        ]
        let samplerRaw: [String: Any] = [
            "num_steps": 40,
            "cfg_scale_text": 3.0,
            "cfg_scale_speaker": 8.0,
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "sequence_length": 640,
        ]
        let raw: [String: Any] = [
            "model_type": "echo_tts",
            "sample_rate": 44100,
            "max_text_length": 768,
            "max_speaker_latent_length": 6400,
            "audio_downsample_factor": 2048,
            "normalize_text": true,
            "delete_blockwise_modules": false,
            "pca_filename": "pca_state.safetensors",
            "fish_codec_repo": "jordand/fish-s1-dac-min",
            "dit": ditRaw,
            "sampler": samplerRaw,
        ]
        let config = ModelConfig(architecture: nil, modelType: "echo_tts", raw: raw)
        let ec = EchoTTSConfig.from(config)
        #expect(ec.modelType == "echo_tts")
        #expect(ec.sampleRate == 44100)
        #expect(ec.maxTextLength == 768)
        #expect(ec.maxSpeakerLatentLength == 6400)
        #expect(ec.audioDownsampleFactor == 2048)
        #expect(ec.normalizeText == true)
        #expect(ec.deleteBlockwiseModules == false)
        #expect(ec.pcaFilename == "pca_state.safetensors")
        #expect(ec.fishCodecRepo == "jordand/fish-s1-dac-min")
        #expect(ec.dit.modelSize == 2048)
        #expect(ec.sampler.numSteps == 40)
    }

    @Test("EchoTTSConfig — falls back to defaults for a minimal config")
    func echoConfigDefaults() {
        let config = ModelConfig(
            architecture: nil, modelType: "echo_tts",
            raw: ["model_type": "echo_tts"])
        let ec = EchoTTSConfig.from(config)
        #expect(ec.sampleRate == 44100)
        #expect(ec.maxTextLength == 768)
        #expect(ec.dit.latentSize == 80)
        #expect(ec.sampler.numSteps == 40)
        #expect(ec.pcaFilename == "pca_state.safetensors")
    }

    // ─── Registry detection ───────────────────────────────────────────────

    @Test("EchoTTSModel.handles — detects echo_tts by model_type")
    func handlesDetectsByModelType() {
        let config = ModelConfig(
            architecture: nil, modelType: "echo_tts",
            raw: ["model_type": "echo_tts", "sample_rate": 44100,
                  "dit": ["latent_size": 80, "speaker_patch_size": 4],
                  "sampler": [:]])
        #expect(EchoTTSModel.handles(config))
    }

    @Test("EchoTTSModel.handles — detects echo_tts by structural keys")
    func handlesDetectsStructurally() {
        // A config without `model_type` but with the `dit` block is still
        // detected structurally (latent_size + speaker_patch_size).
        let config = ModelConfig(
            architecture: nil, modelType: nil,
            raw: ["dit": ["latent_size": 80, "speaker_patch_size": 4],
                  "sample_rate": 44100])
        #expect(EchoTTSModel.handles(config))
    }

    @Test("EchoTTSModel.handles — rejects a plain Llama config")
    func handlesRejectsTextModel() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 2048])
        #expect(!EchoTTSModel.handles(config))
    }

    @Test("AudioModelRegistry — detects EchoTTS and reports textToSpeech")
    func registryDetectsEchoTTS() {
        let config = ModelConfig(
            architecture: nil, modelType: "echo_tts",
            raw: ["model_type": "echo_tts", "sample_rate": 44100,
                  "dit": ["latent_size": 80, "speaker_patch_size": 4],
                  "sampler": [:]])
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.textToSpeech)
    }

    @Test("AudioModelRegistry — text-only config is not an audio model")
    func registryRejectsTextModel() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 2048])
        #expect(!AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == nil)
    }

    // ─── EchoTTSError ────────────────────────────────────────────────────

    @Test("EchoTTSError — diffusionNotWired has a descriptive message")
    func errorDiffusionNotWired() {
        let err = EchoTTSError.diffusionNotWired
        #expect(String(describing: err).contains("DiT"))
        #expect(String(describing: err).contains("SDPA"))
    }

    @Test("EchoTTSError — missingFile carries the filename")
    func errorMissingFile() {
        let err = EchoTTSError.missingFile("pca_state.safetensors")
        #expect(String(describing: err).contains("pca_state.safetensors"))
    }

    @Test("EchoTTSModel — synthesize throws diffusionNotWired")
    func synthesizeThrowsDiffusionNotWired() {
        let model = EchoTTSModel(config: EchoTTSConfig())
        #expect(throws: EchoTTSError.self) {
            _ = try model.synthesize(text: "Hello.")
        }
    }

    @Test("EchoTTSModel — generatePlaceholder returns a non-empty waveform")
    func generatePlaceholderIsNonEmpty() {
        let model = EchoTTSModel(config: EchoTTSConfig())
        let wav = model.generatePlaceholder(durationSeconds: 0.05)
        // 0.05s at 44100 Hz → 2205 samples.
        let expected = max(1, Int(0.05 * 44100))
        #expect(wav.shape == [expected])
        // generatePlaceholder returns zero-filled audio — all samples finite.
        let samples = wav.toArray(as: Float.self)
        #expect(samples.allSatisfy { $0.isFinite })
    }

    // ─── Capability extensions ────────────────────────────────────────────

    @Test("Capability — textToSpeech contains textIn and audioOut")
    func capabilityTextToSpeech() {
        #expect(Capability.textToSpeech.contains(.textIn))
        #expect(Capability.textToSpeech.contains(.audioOut))
        #expect(!Capability.textToSpeech.contains(.audioIn))
    }

    @Test("Capability — speechToText contains audioIn and textOut")
    func capabilitySpeechToText() {
        #expect(Capability.speechToText.contains(.audioIn))
        #expect(Capability.speechToText.contains(.textOut))
        #expect(!Capability.speechToText.contains(.audioOut))
    }
}
