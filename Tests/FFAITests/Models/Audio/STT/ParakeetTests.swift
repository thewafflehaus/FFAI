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
// ParakeetTests — unit tests for Parakeet config parsing, front-end, and
// registry detection. No checkpoint load; all tests run offline.
//
// Validates:
//   * ParakeetConfig.from(_:) parses V2 and V3 config shapes correctly.
//   * ParakeetModel.handles(_:) fires on real config signatures.
//   * AudioModelRegistry.handles(_:) routes Parakeet configs.
//   * ParakeetFrontEnd.logMelFeatures produces finite, correctly-shaped output.
//   * ParakeetTokeniser.decode handles BPE boundary markers.

import Foundation
import Testing

@testable import FFAI

@Suite("Parakeet")
struct ParakeetTests {

    // ─── Helper: build a minimal raw config dict ──────────────────────

    /// Build the raw JSON-dict equivalent of a Parakeet config.
    /// `vocabSize` / `numExtraOutputs` differ between V2 and V3.
    private func rawParakeetConfig(
        vocabSize: Int = 1024,
        numExtraOutputs: Int = 5
    ) -> [String: Any] {
        return [
            "model_defaults": [
                "enc_hidden": 1024,
                "pred_hidden": 640,
                "joint_hidden": 640,
                "tdt_durations": [0, 1, 2, 3, 4],
                "num_tdt_durations": 5,
            ] as [String: Any],
            "preprocessor": [
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.025,
                "window_stride": 0.01,
                "window": "hann",
                "features": 128,
                "n_fft": 512,
                "dither": 1e-5,
                "pad_to": 0,
                "pad_value": 0.0,
            ] as [String: Any],
            "encoder": [
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 128,
                "n_layers": 24,
                "d_model": 1024,
                "n_heads": 8,
                "ff_expansion_factor": 4,
                "subsampling_factor": 8,
                "subsampling_conv_channels": 256,
                "causal_downsampling": false,
                "self_attention_model": "rel_pos",
                "pos_emb_max_len": 5000,
                "conv_kernel_size": 9,
                "use_bias": false,
            ] as [String: Any],
            "decoder": [
                "_target_": "nemo.collections.asr.modules.RNNTDecoder",
                "blank_as_pad": true,
                "prednet": [
                    "pred_hidden": 640,
                    "pred_rnn_layers": 2,
                ] as [String: Any],
                "vocab_size": vocabSize,
            ] as [String: Any],
            "joint": [
                "_target_": "nemo.collections.asr.modules.RNNTJoint",
                "num_classes": vocabSize,
                "num_extra_outputs": numExtraOutputs,
                "vocabulary": (0 ..< vocabSize).map { "<tok\($0)>" },
                "jointnet": [
                    "joint_hidden": 640,
                    "activation": "relu",
                    "encoder_hidden": 1024,
                    "pred_hidden": 640,
                ] as [String: Any],
            ] as [String: Any],
            "decoding": [
                "model_type": "tdt",
                "durations": [0, 1, 2, 3, 4],
                "greedy": ["max_symbols": 10] as [String: Any],
            ] as [String: Any],
        ]
    }

    private func makeConfig(_ raw: [String: Any]) -> ModelConfig {
        ModelConfig(architecture: nil, modelType: nil, raw: raw)
    }

    // ─── Config parsing ───────────────────────────────────────────────

    @Test("ParakeetConfig.from — parses V2 (vocab 1024) correctly")
    func parseConfigV2() throws {
        let raw = rawParakeetConfig(vocabSize: 1024, numExtraOutputs: 5)
        let cfg = try ParakeetConfig.from(makeConfig(raw))

        // Preprocessor
        #expect(cfg.preprocessor.sampleRate == 16_000)
        #expect(cfg.preprocessor.nMels == 128)
        #expect(cfg.preprocessor.nFFT == 512)
        #expect(cfg.preprocessor.winLength == 400)  // 0.025 * 16000
        #expect(cfg.preprocessor.hopLength == 160)  // 0.010 * 16000
        #expect(cfg.preprocessor.normalise == "per_feature")

        // Encoder
        #expect(cfg.encoder.featIn == 128)
        #expect(cfg.encoder.nLayers == 24)
        #expect(cfg.encoder.dModel == 1024)
        #expect(cfg.encoder.nHeads == 8)
        #expect(cfg.encoder.subsamplingFactor == 8)
        #expect(cfg.encoder.convKernelSize == 9)

        // Prediction network
        #expect(cfg.predNet.predHidden == 640)
        #expect(cfg.predNet.predRnnLayers == 2)

        // Joint
        #expect(cfg.joint.jointHidden == 640)
        #expect(cfg.joint.numClasses == 1025)  // vocab + blank
        #expect(cfg.joint.numExtraOutputs == 5)

        // TDT
        #expect(cfg.tdtDurations == [0, 1, 2, 3, 4])
        #expect(cfg.maxSymbolsPerStep == 10)

        // Blank token
        #expect(cfg.blankTokenId == 1024)  // vocabulary.count
    }

    @Test("ParakeetConfig.from — parses V3 (vocab 8192) correctly")
    func parseConfigV3() throws {
        let raw = rawParakeetConfig(vocabSize: 8192, numExtraOutputs: 5)
        let cfg = try ParakeetConfig.from(makeConfig(raw))

        #expect(cfg.vocabulary.count == 8192)
        #expect(cfg.blankTokenId == 8192)
        #expect(cfg.joint.numClasses == 8193)  // 8192 + blank
    }

    @Test("ParakeetConfig.from — throws on missing encoder")
    func parseConfig_missingEncoder() throws {
        var raw = rawParakeetConfig()
        raw.removeValue(forKey: "encoder")
        #expect(throws: ParakeetConfigError.missingEncoder) {
            try ParakeetConfig.from(makeConfig(raw))
        }
    }

    @Test("ParakeetConfig.from — throws on missing model_defaults")
    func parseConfig_missingModelDefaults() throws {
        var raw = rawParakeetConfig()
        raw.removeValue(forKey: "model_defaults")
        #expect(throws: ParakeetConfigError.missingModelDefaults) {
            try ParakeetConfig.from(makeConfig(raw))
        }
    }

    @Test("ParakeetConfig.from — throws on missing tdt_durations")
    func parseConfig_missingTDTDurations() throws {
        var raw = rawParakeetConfig()
        var defaults = raw["model_defaults"] as! [String: Any]
        defaults.removeValue(forKey: "tdt_durations")
        raw["model_defaults"] = defaults
        #expect(throws: ParakeetConfigError.missingTDTDurations) {
            try ParakeetConfig.from(makeConfig(raw))
        }
    }

    // ─── Detection ────────────────────────────────────────────────────

    @Test("ParakeetModel.handles — true for TDT config")
    func handles_trueForParakeet() {
        let raw = rawParakeetConfig()
        #expect(ParakeetModel.handles(makeConfig(raw)))
    }

    @Test("ParakeetModel.handles — false for missing tdt_durations")
    func handles_falseWithoutTDT() {
        var raw = rawParakeetConfig()
        var defaults = raw["model_defaults"] as! [String: Any]
        defaults.removeValue(forKey: "tdt_durations")
        raw["model_defaults"] = defaults
        #expect(!ParakeetModel.handles(makeConfig(raw)))
    }

    @Test("ParakeetModel.handles — false for text model config")
    func handles_falseForTextModel() {
        let raw: [String: Any] = [
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        ]
        #expect(!ParakeetModel.handles(makeConfig(raw)))
    }

    // ─── AudioModelRegistry detection ────────────────────────────────

    @Test("AudioModelRegistry.handles — true for Parakeet config")
    func registry_handlesParakeet() {
        let raw = rawParakeetConfig()
        #expect(AudioModelRegistry.handles(makeConfig(raw)))
    }

    @Test("AudioModelRegistry.handles — false for text model")
    func registry_handlesTextModel_false() {
        let raw: [String: Any] = ["model_type": "qwen3", "hidden_size": 2048]
        #expect(!AudioModelRegistry.handles(makeConfig(raw)))
    }

    // ─── Front-end ────────────────────────────────────────────────────

    @Test("ParakeetFrontEnd.logMelFeatures — finite, correct shape")
    func frontEnd_finiteShape() {
        let cfg = ParakeetPreprocessorConfig(
            sampleRate: 16_000,
            nMels: 128,
            nFFT: 512,
            winLength: 400,
            hopLength: 160,
            preemph: 0.97,
            logZeroGuardValue: pow(2, -24),
            normalise: "per_feature"
        )
        // 1 s of a 440 Hz tone
        let sr = 16_000
        var wave = [Float](repeating: 0, count: sr)
        for i in 0 ..< sr {
            wave[i] = 0.3 * sin(2 * Float.pi * 440 * Float(i) / Float(sr))
        }
        let mel = ParakeetFrontEnd.logMelFeatures(waveform: wave, cfg: cfg)
        // Expect [nFrames, 128]
        let nFrames = mel.count / cfg.nMels
        #expect(nFrames > 0)
        #expect(mel.count == nFrames * cfg.nMels)
        #expect(
            mel.allSatisfy { $0.isFinite },
            "log-Mel front-end produced non-finite values")
    }

    @Test("ParakeetFrontEnd.logMelFeatures — empty waveform returns empty")
    func frontEnd_empty() {
        let cfg = ParakeetPreprocessorConfig(
            sampleRate: 16_000, nMels: 128, nFFT: 512,
            winLength: 400, hopLength: 160, preemph: 0.97,
            logZeroGuardValue: pow(2, -24), normalise: "per_feature"
        )
        let mel = ParakeetFrontEnd.logMelFeatures(waveform: [], cfg: cfg)
        #expect(mel.isEmpty)
    }

    @Test("ParakeetFrontEnd — global normalise produces finite features")
    func frontEnd_globalNormalise() {
        let cfg = ParakeetPreprocessorConfig(
            sampleRate: 16_000, nMels: 128, nFFT: 512,
            winLength: 400, hopLength: 160, preemph: 0.97,
            logZeroGuardValue: pow(2, -24), normalise: "global"
        )
        let wave = (0 ..< 8_000).map { Float(0.1) * sin(Float($0)) }
        let mel = ParakeetFrontEnd.logMelFeatures(waveform: wave, cfg: cfg)
        #expect(!mel.isEmpty)
        #expect(mel.allSatisfy { $0.isFinite })
    }

    // ─── Tokeniser ────────────────────────────────────────────────────

    @Test("ParakeetTokeniser.decode — BPE boundary markers replaced with spaces")
    func tokeniser_decodesWordBoundaries() {
        let vocab = ["▁Hello", "▁world", "!"]
        let text = ParakeetTokeniser.decode(tokens: [0, 1, 2], vocabulary: vocab)
        // Leading space stripped by trimming
        #expect(text == "Hello world!")
    }

    @Test("ParakeetTokeniser.decode — out-of-range tokens skipped")
    func tokeniser_outOfRange() {
        let vocab = ["a", "b"]
        let text = ParakeetTokeniser.decode(tokens: [0, 99, 1], vocabulary: vocab)
        #expect(text == "ab")
    }

    @Test("ParakeetTokeniser.isSpecial — identifies special tokens")
    func tokeniser_isSpecial() {
        let vocab = ["<unk>", "hello", "<|endoftext|>", "<pad>", "world"]
        #expect(ParakeetTokeniser.isSpecial(0, vocabulary: vocab))  // <unk>
        #expect(!ParakeetTokeniser.isSpecial(1, vocabulary: vocab))  // hello
        #expect(ParakeetTokeniser.isSpecial(2, vocabulary: vocab))  // <|endoftext|>
        #expect(ParakeetTokeniser.isSpecial(3, vocabulary: vocab))  // <pad>
        #expect(!ParakeetTokeniser.isSpecial(4, vocabulary: vocab))  // world
    }

    // ─── Mel filterbank ───────────────────────────────────────────────

    @Test("ParakeetFrontEnd.melFilterbank — non-negative, shape correct")
    func melFilterbank_shape() {
        let nMels = 128
        let nFFT = 512
        let bank = ParakeetFrontEnd.melFilterbank(
            sampleRate: 16_000, nFFT: nFFT, nMels: nMels)
        let nFreq = nFFT / 2 + 1
        #expect(bank.count == nMels * nFreq)
        #expect(bank.allSatisfy { $0 >= 0 })
        // Each row's energy across FFT bins. With (sr=16k, nFFT=512,
        // nMels=128) the FFT bin spacing is 31.25 Hz, while the lowest
        // few Mel-spaced triangles span only ~14-30 Hz — so the
        // lowest band(s) contain no FFT bin and produce an all-zero
        // row. This matches librosa's documented "Empty filters
        // detected in mel frequency basis" behavior and is expected
        // for any narrow-band Slaney filterbank.
        let rowSums = (0 ..< nMels).map { m in
            (0 ..< nFreq).map { k in bank[m * nFreq + k] }.reduce(0, +)
        }
        // Allow up to a handful of empty rows at the low end (matches
        // librosa). The vast majority must be strictly positive.
        let emptyRows = rowSums.filter { $0 <= 0 }.count
        #expect(
            emptyRows <= 4,
            Comment(
                rawValue: "Expected at most 4 empty mel bands "
                    + "at the low end (librosa-equivalent Slaney); got "
                    + "\(emptyRows)."))
        // All non-empty rows must be bounded (Slaney height = 2/(hi-lo)
        // in Hz, so the peak is well under the Nyquist bandwidth).
        #expect(rowSums.allSatisfy { $0 < 10.0 })
    }
}
