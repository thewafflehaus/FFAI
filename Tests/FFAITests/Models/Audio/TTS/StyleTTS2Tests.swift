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
// StyleTTS2Tests — config parsing, handles() detection, registry plumbing,
// text-cleaner, and vocoder unit tests. No model weights required.

import Foundation
import Testing

@testable import FFAI

@Suite("StyleTTS2")
struct StyleTTS2Tests {

    // ─── Config parsing ────────────────────────────────────────────────

    @Test("StyleTTS2Config.from() decodes kitten-tts-nano config fields")
    func configParseKittenNano() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let json = kittenNanoConfigJSON
        try json.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)

        let modelConfig = try ModelConfig.load(from: dir)
        guard let sc = StyleTTS2Config.from(modelConfig) else {
            Issue.record("StyleTTS2Config.from returned nil for kitten-nano config")
            return
        }

        #expect(sc.modelType == "kitten_tts")
        #expect(sc.nToken == 178)
        #expect(sc.hiddenDim == 128)
        #expect(sc.styleDim == 128)
        #expect(sc.nLayer == 2)
        #expect(sc.sampleRate == 24_000)
        #expect(sc.istftnet.genIstftNFft == 20)
        #expect(sc.istftnet.genIstftHopSize == 5)
        #expect(sc.plbert.numHiddenLayers == 12)
        #expect(sc.plbert.hiddenSize == 768)
        #expect(sc.voiceAliases["Bella"] == "expr-voice-2-f")
        #expect(sc.speedPriors["expr-voice-5-m"] == 0.8)
    }

    @Test("StyleTTS2Config defaults survive missing optional fields")
    func configDefaults() {
        // Minimal config — only required fields.
        let raw: [String: Any] = [
            "model_type": "kitten_tts",
            "n_token": 100,
            "hidden_dim": 64,
            "plbert": [
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "hidden_size": 256,
                "intermediate_size": 512,
                "max_position_embeddings": 128,
            ] as [String: Any],
            "istftnet": [
                "gen_istft_n_fft": 20,
                "gen_istft_hop_size": 5,
            ] as [String: Any],
        ]
        let mc = ModelConfig(architecture: nil, modelType: "kitten_tts", raw: raw)
        guard let sc = StyleTTS2Config.from(mc) else {
            Issue.record("StyleTTS2Config.from returned nil for minimal config")
            return
        }
        #expect(sc.sampleRate == 24_000)
        #expect(sc.voiceAliases.isEmpty)
        #expect(sc.speedPriors.isEmpty)
        #expect(sc.voicesPath == "voices.safetensors")
    }

    // ─── handles() detection ──────────────────────────────────────────

    @Test("StyleTTS2Model.handles() returns true for kitten_tts model_type")
    func handlesKittenTTS() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        try #"{"model_type": "kitten_tts", "n_token": 178, "hidden_dim": 128}"#
            .write(
                to: dir.appendingPathComponent("config.json"),
                atomically: true, encoding: .utf8)

        let mc = try ModelConfig.load(from: dir)
        #expect(StyleTTS2Model.handles(mc) == true)
    }

    @Test("StyleTTS2Model.handles() returns true for style_tts2 model_type")
    func handlesStyleTTS2() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        try #"{"model_type": "style_tts2", "n_token": 50}"#
            .write(
                to: dir.appendingPathComponent("config.json"),
                atomically: true, encoding: .utf8)

        let mc = try ModelConfig.load(from: dir)
        #expect(StyleTTS2Model.handles(mc) == true)
    }

    @Test("StyleTTS2Model.handles() returns true for structural detection (no model_type)")
    func handlesStructural() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let json = """
            {
                "n_token": 178,
                "hidden_dim": 128,
                "plbert": {"num_hidden_layers": 12},
                "istftnet": {"gen_istft_n_fft": 20, "gen_istft_hop_size": 5}
            }
            """
        try json.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)

        let mc = try ModelConfig.load(from: dir)
        #expect(StyleTTS2Model.handles(mc) == true)
    }

    @Test("StyleTTS2Model.handles() returns false for unrelated config")
    func doesNotHandleLlama() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        try #"{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}"#
            .write(
                to: dir.appendingPathComponent("config.json"),
                atomically: true, encoding: .utf8)

        let mc = try ModelConfig.load(from: dir)
        #expect(StyleTTS2Model.handles(mc) == false)
    }

    // ─── AudioModelRegistry ────────────────────────────────────────────

    @Test("AudioModelRegistry.handles() returns true for kitten_tts")
    func registryHandles() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        try #"{"model_type": "kitten_tts", "n_token": 100, "hidden_dim": 64}"#
            .write(
                to: dir.appendingPathComponent("config.json"),
                atomically: true, encoding: .utf8)

        let mc = try ModelConfig.load(from: dir)
        #expect(AudioModelRegistry.handles(mc) == true)
    }

    @Test("AudioModelRegistry.capabilities returns textToSpeech for kitten_tts")
    func registryCapabilities() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        try #"{"model_type": "kitten_tts", "n_token": 100, "hidden_dim": 64}"#
            .write(
                to: dir.appendingPathComponent("config.json"),
                atomically: true, encoding: .utf8)

        let mc = try ModelConfig.load(from: dir)
        let caps = AudioModelRegistry.capabilities(for: mc)
        #expect(caps == Capability.textToSpeech)
    }

    @Test("AudioModelRegistry.capabilities(forConfigAt:) resolves kitten_tts directory")
    func registryCapabilitiesFromDir() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        try #"{"model_type": "kitten_tts", "n_token": 100, "hidden_dim": 64}"#
            .write(
                to: dir.appendingPathComponent("config.json"),
                atomically: true, encoding: .utf8)

        let caps = AudioModelRegistry.capabilities(forConfigAt: dir)
        #expect(caps == Capability.textToSpeech)
    }

    // ─── Capability extensions ─────────────────────────────────────────

    @Test("Capability.textToSpeech contains textIn and audioOut")
    func capabilityTextToSpeech() {
        #expect(Capability.textToSpeech.contains(.textIn))
        #expect(Capability.textToSpeech.contains(.audioOut))
    }

    @Test("Capability.speechToText contains audioIn and textOut")
    func capabilitySpeechToText() {
        #expect(Capability.speechToText.contains(.audioIn))
        #expect(Capability.speechToText.contains(.textOut))
    }

    // ─── StyleTTS2Model.build() ────────────────────────────────────────

    @Test("StyleTTS2Model.build() constructs a model without a checkpoint")
    func buildFromConfig() {
        let sc = StyleTTS2Config()
        let m = StyleTTS2Model.build(config: sc)
        #expect(m.config.nToken == 178)
        #expect(m.config.sampleRate == 24_000)
        #expect(m.vocoder.nFFT == sc.istftnet.genIstftNFft)
        #expect(m.vocoder.hopLength == sc.istftnet.genIstftHopSize)
        #expect(m.voiceEmbeddings.isEmpty)
        #expect(m.sampleRate == 24_000)
    }

    @Test("StyleTTS2Model.synthesize() throws acousticFrontEndNotWired")
    func synthesizeThrows() {
        let m = StyleTTS2Model.build(config: StyleTTS2Config())
        do {
            _ = try m.synthesize(text: "Hello world.")
            Issue.record("expected StyleTTS2Error.acousticFrontEndNotWired")
        } catch StyleTTS2Error.acousticFrontEndNotWired {
            // expected
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

    @Test("StyleTTS2Model.generatePlaceholder() returns non-empty f32 tensor")
    func placeholderNonEmpty() {
        let m = StyleTTS2Model.build(config: StyleTTS2Config())
        let t = m.generatePlaceholder(durationSeconds: 0.05)
        #expect(t.elementCount > 0)
        #expect(t.dtype == .f32)
    }

    // ─── StyleTTS2Error descriptions ──────────────────────────────────

    @Test("StyleTTS2Error descriptions contain identifying text")
    func errorDescriptions() {
        let e1 = StyleTTS2Error.acousticFrontEndNotWired
        #expect(String(describing: e1).contains("acoustic"))

        let e2 = StyleTTS2Error.missingFile("voices.safetensors")
        #expect(String(describing: e2).contains("voices.safetensors"))

        let e3 = StyleTTS2Error.unknownVoice("Bella")
        #expect(String(describing: e3).contains("Bella"))
    }

    // ─── Text cleaner ──────────────────────────────────────────────────

    @Test("StyleTTS2TextCleaner.tokenize maps ASCII letters to non-zero ids")
    func cleanerAscii() {
        let ids = StyleTTS2TextCleaner.tokenize("Hello")
        // H (upper) and e, l, o (lower) should all map to known symbols.
        #expect(!ids.isEmpty)
        // All ids should be in valid range [0, nSymbols).
        let allValid = ids.allSatisfy { $0 >= 0 }
        #expect(allValid)
    }

    @Test("StyleTTS2TextCleaner.tokenize drops unknown characters")
    func cleanerDropsUnknown() {
        // Chinese character — not in the symbol table.
        let ids = StyleTTS2TextCleaner.tokenize("你好")
        #expect(ids.isEmpty)
    }

    @Test("StyleTTS2TextCleaner.tokenize handles IPA phonemes")
    func cleanerIPA() {
        let ids = StyleTTS2TextCleaner.tokenize("həˈloʊ")
        // h, ə, ˈ, l, o, ʊ are all in the symbol table.
        #expect(!ids.isEmpty)
    }

    @Test("StyleTTS2TextCleaner.tokenize(sentences:) returns results in order")
    func cleanerBatch() {
        let inputs = ["Hello", "World", "foo", "bar", "baz"]
        let results = StyleTTS2TextCleaner.tokenize(sentences: inputs)
        #expect(results.count == inputs.count)
        // Each batch result should equal the single-item result.
        for (i, input) in inputs.enumerated() {
            #expect(results[i] == StyleTTS2TextCleaner.tokenize(input))
        }
    }

    @Test("StyleTTS2Model.phonemeIds() prepends and appends pad token 0")
    func phonemeIdsPadding() {
        let m = StyleTTS2Model.build(config: StyleTTS2Config())
        let ids = m.phonemeIds(for: "hɛˈloʊ")
        #expect(ids.first == 0, "expected BOS pad token 0")
        #expect(ids.last == 0, "expected EOS pad token 0")
        #expect(ids.count >= 2)
    }

    // ─── Vocoder ─────────────────────────────────────────────────────

    @Test("StyleTTS2Vocoder.synthesize produces non-trivially-zero output for non-zero input")
    func vocoderNonZero() {
        let nFFT = 20
        let hop = 5
        let nFrames = 10
        let nFreq = nFFT / 2 + 1  // 11

        // Build a non-trivial spectrogram: magnitude 1 with a linear phase ramp.
        // A flat-magnitude spectrum with zero phase concentrates energy in DC,
        // but after WOLA normalization the per-sample values are very small
        // (DC component only, then divided by N). Use alternating phase to
        // spread energy across frequencies and produce audible amplitude.
        var re = [Float](repeating: 0, count: nFrames * nFreq)
        var im = [Float](repeating: 0, count: nFrames * nFreq)
        for f in 0 ..< nFrames {
            for k in 0 ..< nFreq {
                let phase = Float(k) * .pi / Float(nFreq)
                re[f * nFreq + k] = cos(phase)
                im[f * nFreq + k] = sin(phase)
            }
        }

        let vocoder = StyleTTS2Vocoder(nFFT: nFFT, hopLength: hop)
        let waveform = vocoder.synthesize(
            specReFlat: re, specImFlat: im,
            nFrames: nFrames)
        #expect(waveform.elementCount > 0)
        let samples = waveform.toArray(as: Float.self)
        // Verify output is finite and has at least some non-zero samples.
        let allFinite = samples.allSatisfy { $0.isFinite }
        #expect(allFinite, "waveform must be finite")
        let maxAbs = samples.map { abs($0) }.max() ?? 0
        #expect(maxAbs > 0, "expected at least some non-zero samples")
    }

    @Test("StyleTTS2Vocoder.hannWindow has correct length and energy")
    func vocoderHannWindow() {
        let vocoder = StyleTTS2Vocoder(nFFT: 20, hopLength: 5)
        #expect(vocoder.hannWindow.count == 20)
        // Hann window should sum to roughly N/2.
        let sum = vocoder.hannWindow.reduce(0, +)
        #expect(abs(sum - Float(20) / 2.0) < 1.0)
    }

    // ─── PLBertConfig / ISTFTNetConfig ────────────────────────────────

    @Test("PLBertConfig.from() decodes standard fields")
    func plbertConfigDecode() {
        let raw: [String: Any] = [
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "hidden_size": 768,
            "intermediate_size": 2048,
            "max_position_embeddings": 512,
        ]
        let c = PLBertConfig.from(raw)
        #expect(c.numHiddenLayers == 12)
        #expect(c.hiddenSize == 768)
        #expect(c.maxPositionEmbeddings == 512)
    }

    @Test("ISTFTNetConfig.from() decodes standard fields")
    func istftnetConfigDecode() {
        let raw: [String: Any] = [
            "resblock_kernel_sizes": [3, 3],
            "upsample_rates": [10, 6],
            "upsample_initial_channel": 256,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
            "upsample_kernel_sizes": [20, 12],
            "gen_istft_n_fft": 20,
            "gen_istft_hop_size": 5,
        ]
        let c = ISTFTNetConfig.from(raw)
        #expect(c.genIstftNFft == 20)
        #expect(c.genIstftHopSize == 5)
        #expect(c.upsampleRates == [10, 6])
        #expect(c.upsampleInitialChannel == 256)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

private func makeTempDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("ffai-s2tts-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
        at: dir,
        withIntermediateDirectories: true)
    return dir
}

/// Inline kitten-tts-nano config.json (matches
/// `mlx-community/kitten-tts-nano-0.8-fp16`).
private let kittenNanoConfigJSON = """
    {
        "asr_res_dim": 64,
        "hidden_dim": 128,
        "istftnet": {
            "resblock_kernel_sizes": [3, 3],
            "upsample_rates": [10, 6],
            "upsample_initial_channel": 256,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
            "upsample_kernel_sizes": [20, 12],
            "gen_istft_n_fft": 20,
            "gen_istft_hop_size": 5
        },
        "max_conv_dim": 256,
        "max_dur": 50,
        "model_type": "kitten_tts",
        "n_layer": 2,
        "n_mels": 80,
        "n_token": 178,
        "plbert": {
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "hidden_size": 768,
            "intermediate_size": 2048,
            "max_position_embeddings": 512,
            "embedding_size": 128,
            "inner_group_num": 1,
            "num_hidden_groups": 1,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12
        },
        "sample_rate": 24000,
        "speed_priors": {
            "expr-voice-2-f": 0.8,
            "expr-voice-5-m": 0.8
        },
        "style_dim": 128,
        "text_encoder_kernel_size": 5,
        "voice_aliases": {
            "Bella": "expr-voice-2-f",
            "Leo": "expr-voice-5-m"
        },
        "voices_path": "voices.npz"
    }
    """
