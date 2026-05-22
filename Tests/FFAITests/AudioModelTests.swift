import Foundation
import Metal
import Testing
@testable import FFAI

// Unit tests for the audio model families — exercises construction,
// the GPU-accelerated paths (the Kokoro vocoder, the QwenOmni audio
// encoder), and the config-driven registry detection. The full
// checkpoint-load path is covered by the ModelTests integration suites;
// these tests run on synthetic data so they are fast + offline.
@Suite("Audio models")
struct AudioModelTests {

    private func randTensor(_ shape: [Int], scale: Float = 0.05,
                            seed: Int) -> Tensor {
        let n = shape.reduce(1, *)
        var data = [Float](repeating: 0, count: n)
        var s = UInt64(seed &+ 1)
        for i in 0..<n {
            s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let u = Float(s >> 40) / Float(1 << 24)
            data[i] = (u - 0.5) * 2 * scale
        }
        let t = Tensor.empty(shape: shape, dtype: .f32)
        t.copyIn(from: data)
        return t
    }

    // ─── Kokoro vocoder ──────────────────────────────────────────────

    @Test("KokoroVocoder — synthesizes a non-degenerate waveform")
    func kokoroVocoderWaveform() {
        autoreleasepool {
            // Kokoro's iSTFTNet head: tiny 20-sample FFT, hop 5.
            let cfg = KokoroConfig(nToken: 178, hidden: 512, nMels: 80)
            let model = KokoroModel.build(config: cfg)
            #expect(model.vocoder.nFFT == 20)
            #expect(model.vocoder.hopLength == 5)

            // A predicted complex spectrogram — frequency-sweep content
            // so the reconstruction is non-constant.
            let nFrames = 24
            let nFreq = cfg.istftNFFT / 2 + 1
            let specRe = randTensor([nFrames, nFreq], scale: 0.5, seed: 1)
            let specIm = randTensor([nFrames, nFreq], scale: 0.5, seed: 2)

            let waveform = model.synthesizeFromSpectrogram(
                specRe: specRe, specIm: specIm)
            // outLen = (nFrames - 1) * hop + nFFT = 23*5 + 20 = 135.
            #expect(waveform.shape == [135])
            let samples = waveform.toFloatArray()
            // Non-degenerate: finite, not silent, not constant.
            #expect(samples.allSatisfy { $0.isFinite })
            let energy = samples.map { $0 * $0 }.reduce(0, +)
            #expect(energy > 1e-6, "vocoder produced a silent waveform")
            #expect(Set(samples.map { ($0 * 1000).rounded() }).count > 1,
                    "vocoder produced a constant waveform")
        }
    }

    @Test("KokoroVocoder — DC spectrum reconstructs a steady signal")
    func kokoroVocoderDC() {
        autoreleasepool {
            let model = KokoroModel.build(
                config: KokoroConfig(nToken: 178, hidden: 256, nMels: 80))
            let nFrames = 10
            let nFreq = model.vocoder.nFFT / 2 + 1
            var re = [Float](repeating: 0, count: nFrames * nFreq)
            for f in 0..<nFrames {
                re[f * nFreq] = Float(model.vocoder.nFFT)  // DC = nFFT
            }
            let reT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
            reT.copyIn(from: re)
            let imT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
            imT.zero()
            let wav = model.synthesizeFromSpectrogram(specRe: reT, specIm: imT)
            // A pure-DC spectrum inverse-transforms to a constant frame
            // value; after Hann-windowed COLA the interior samples are
            // steady (each interior sample sees the same set of window
            // taps, so the COLA ratio is hop-periodic and bounded).
            let samples = wav.toFloatArray()
            #expect(samples.allSatisfy { $0.isFinite })
            let interior = Array(samples[(model.vocoder.nFFT)..<(samples.count
                                                          - model.vocoder.nFFT)])
            // Every interior sample is a positive, bounded reconstruction
            // of the same DC level — no NaN, no blow-up, no zero.
            #expect(interior.allSatisfy { $0 > 0.5 && $0 < 2.0 })
            // hop-periodic: samples `hop` apart reconstruct identically.
            let hop = model.vocoder.hopLength
            for i in hop..<(interior.count - hop) {
                #expect(abs(interior[i] - interior[i - hop]) < 1e-3)
            }
        }
    }

    @Test("Kokoro — synthesize(phonemeIds:) reports front-end unavailable")
    func kokoroFrontEndUnavailable() {
        let model = KokoroModel.build(
            config: KokoroConfig(nToken: 178, hidden: 256, nMels: 80))
        #expect(throws: KokoroError.self) {
            _ = try model.synthesize(phonemeIds: [1, 2, 3])
        }
    }

    @Test("Kokoro — phonemeIds maps via the config vocab")
    func kokoroPhonemeIds() {
        let vocab = ["h": 5, "ɛ": 6, "l": 7, "o": 8]
        let model = KokoroModel.build(
            config: KokoroConfig(nToken: 178, hidden: 256, nMels: 80),
            phonemeVocab: vocab)
        // "hɛlo" → [5,6,7,8]; an unknown phoneme is dropped.
        #expect(model.phonemeIds(for: "hɛlo") == [5, 6, 7, 8])
        #expect(model.phonemeIds(for: "hzo") == [5, 8])
    }

    // ─── Registry detection ──────────────────────────────────────────

    @Test("AudioModelRegistry — detects Whisper from model_type")
    func registryDetectsWhisper() {
        let config = ModelConfig(
            architecture: "WhisperForConditionalGeneration",
            modelType: "whisper",
            raw: ["model_type": "whisper", "d_model": 384,
                  "encoder_layers": 4, "encoder_attention_heads": 6,
                  "decoder_layers": 4, "decoder_attention_heads": 6,
                  "vocab_size": 51865])
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.speechToText)
        #expect(WhisperModel.handles(config))
    }

    @Test("AudioModelRegistry — detects Kokoro from istftnet block")
    func registryDetectsKokoro() {
        let config = ModelConfig(
            architecture: nil, modelType: "kokoro",
            raw: ["model_type": "kokoro", "n_token": 178,
                  "hidden_dim": 512, "n_mels": 80,
                  "istftnet": ["gen_istft_n_fft": 20,
                               "gen_istft_hop_size": 5]])
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.textToSpeech)
    }

    @Test("AudioModelRegistry — detects QwenOmni from audio_config")
    func registryDetectsQwenOmni() {
        let config = ModelConfig(
            architecture: "Qwen2_5OmniForConditionalGeneration",
            modelType: "qwen2_5_omni",
            raw: ["model_type": "qwen2_5_omni",
                  "audio_config": ["d_model": 1280, "encoder_layers": 32,
                                   "encoder_attention_heads": 20,
                                   "num_mel_bins": 128],
                  "text_config": ["hidden_size": 3584]])
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.omniAudio)
    }

    @Test("AudioModelRegistry — detects LlamaTTS from the orpheus type")
    func registryDetectsLlamaTTS() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "orpheus",
            raw: ["model_type": "orpheus", "hidden_size": 3072,
                  "num_hidden_layers": 28, "num_attention_heads": 24,
                  "vocab_size": 156_940, "sample_rate": 24_000])
        #expect(AudioModelRegistry.handles(config))
        #expect(LlamaTTSModel.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.textToSpeech)
    }

    @Test("AudioModelRegistry — detects LlamaTTS structurally")
    func registryDetectsLlamaTTSStructural() {
        // A plain LlamaForCausalLM with the Orpheus-extended vocabulary
        // and a TTS sample_rate is detected even without an explicit
        // model_type.
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 3072,
                  "vocab_size": 156_940, "sample_rate": 24_000])
        #expect(LlamaTTSModel.handles(config))
    }

    @Test("AudioModelRegistry — detects Marvis / CSM")
    func registryDetectsMarvis() {
        let config = ModelConfig(
            architecture: "CSMForConditionalGeneration", modelType: "csm",
            raw: ["model_type": "csm", "hidden_size": 1024,
                  "num_hidden_layers": 16, "num_attention_heads": 16,
                  "intermediate_size": 8192, "audio_vocab_size": 2051,
                  "audio_num_codebooks": 32,
                  "depth_decoder_config": ["hidden_size": 1024,
                                           "num_hidden_layers": 4,
                                           "num_attention_heads": 8,
                                           "intermediate_size": 8192]])
        #expect(AudioModelRegistry.handles(config))
        #expect(MarvisModel.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.textToSpeech)
    }

    @Test("AudioModelRegistry — detects Qwen3TTS from talker_config")
    func registryDetectsQwen3TTS() {
        let config = ModelConfig(
            architecture: "Qwen3TTSForConditionalGeneration",
            modelType: "qwen3_tts",
            raw: ["model_type": "qwen3_tts",
                  "talker_config": ["hidden_size": 1024,
                                    "num_hidden_layers": 28,
                                    "vocab_size": 3072],
                  "speaker_encoder_config": ["enc_dim": 1024],
                  "sample_rate": 24_000])
        #expect(AudioModelRegistry.handles(config))
        #expect(Qwen3TTSModel.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.textToSpeech)
    }

    @Test("MarvisConfig — decodes the dual-transformer shape")
    func marvisConfigDecodes() {
        let config = ModelConfig(
            architecture: nil, modelType: "csm",
            raw: ["model_type": "csm", "hidden_size": 2048,
                  "num_hidden_layers": 16, "num_attention_heads": 32,
                  "num_key_value_heads": 8, "head_dim": 64,
                  "intermediate_size": 8192, "audio_vocab_size": 2051,
                  "audio_num_codebooks": 32, "text_vocab_size": 128_256,
                  "depth_decoder_config": ["hidden_size": 1024,
                                           "num_hidden_layers": 4,
                                           "num_attention_heads": 8,
                                           "intermediate_size": 8192]])
        let mc = MarvisConfig.from(config)
        #expect(mc != nil)
        #expect(mc?.backbone.nLayers == 16)
        #expect(mc?.decoder.nLayers == 4)
        #expect(mc?.audioNumCodebooks == 32)
    }

    @Test("Qwen3TTSConfig — decodes the nested talker config")
    func qwen3TTSConfigDecodes() {
        let config = ModelConfig(
            architecture: nil, modelType: "qwen3_tts",
            raw: ["model_type": "qwen3_tts",
                  "talker_config": ["hidden_size": 1024,
                                    "num_hidden_layers": 28,
                                    "num_attention_heads": 16,
                                    "vocab_size": 3072,
                                    "codec_eos_token_id": 2150],
                  "speaker_encoder_config": [:],
                  "sample_rate": 24_000])
        let qc = Qwen3TTSConfig.from(config)
        #expect(qc != nil)
        #expect(qc?.talker.nLayers == 28)
        #expect(qc?.talker.hidden == 1024)
        #expect(qc?.codecEosTokenId == 2150)
    }

    @Test("LlamaTTS — SNAC code de-interleave plane lengths")
    func llamaTTSDeinterleave() {
        // 14 tokens = 2 SNAC frames → layer1:2, layer2:4, layer3:8.
        let planes = LlamaTTSModel.deinterleaveSNACCodes(Array(0..<14))
        #expect(planes[0].count == 2)
        #expect(planes[1].count == 4)
        #expect(planes[2].count == 8)
    }

    @Test("AudioModelRegistry — text-only config is not an audio model")
    func registryRejectsTextModel() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 2048])
        #expect(!AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == nil)
    }

    // ─── Config parsing ──────────────────────────────────────────────

    @Test("WhisperConfig — derives geometry from a Whisper config")
    func whisperConfigParse() {
        let config = ModelConfig(
            architecture: "WhisperForConditionalGeneration",
            modelType: "whisper",
            raw: ["d_model": 512, "encoder_layers": 6,
                  "encoder_attention_heads": 8, "decoder_layers": 6,
                  "decoder_attention_heads": 8, "vocab_size": 51865,
                  "num_mel_bins": 80])
        let wc = WhisperConfig.from(config)
        #expect(wc != nil)
        #expect(wc?.hidden == 512)
        #expect(wc?.decoderHeadDim == 64)
        #expect(wc?.frontEnd.nMels == 80)
    }

    @Test("QwenOmniAudioConfig — pulls text hidden from text_config")
    func qwenOmniConfigParse() {
        let config = ModelConfig(
            architecture: nil, modelType: "qwen3_omni",
            raw: ["audio_config": ["d_model": 1280, "encoder_layers": 32,
                                   "encoder_attention_heads": 20,
                                   "num_mel_bins": 128,
                                   "encoder_ffn_dim": 5120],
                  "text_config": ["hidden_size": 2048]])
        let qc = QwenOmniAudioConfig.from(config)
        #expect(qc != nil)
        #expect(qc?.encoderHidden == 1280)
        #expect(qc?.textHidden == 2048)
        #expect(qc?.nMels == 128)
    }
}
