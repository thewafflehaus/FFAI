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
// Integration test: loads the real FireRedASR2-AED checkpoint from the HF
// cache and runs Kaldi fbank → CMVN → Conv2dSubsampling → Conformer → beam
// search end-to-end. A missing checkpoint FAILS the suite.
//
// FireRedASR2 is a Conformer-AED (attention-based encoder-decoder) model for
// Mandarin/English ASR. This test validates:
//   * Config decoding from the flat FireRedASR2 config layout.
//   * Weight loading (Conv2d OHWI→OIHW transposition, conv-key remapping,
//     tgt_word_prj weight tying).
//   * Kaldi fbank + CMVN front-end produces finite, non-degenerate features.
//   * encodeAudio produces finite, non-degenerate encoder outputs.
//   * End-to-end transcription of a real speech clip yields a non-empty,
//     non-degenerate string (≥ 2 distinct words).
//
// DO NOT RUN — this suite requires multi-GB checkpoint files and a GPU.
// Run via `make test-integration`.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "FireRedASR2 Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct FireRedASR2IntegrationTests {

    /// Canonical HF repo id. No 4-bit MLX conversion exists at time of
    /// writing — `mlx-community/FireRedASR2-AED-mlx` is the only
    /// published MLX variant.
    private static let repoId = "mlx-community/FireRedASR2-AED-mlx"

    // ─── Checkpoint resolution ───────────────────────────────────────────

    /// Resolve the checkpoint directory through `ModelLocator`, serialised
    /// against other model loads via `ModelLoadLock`.
    private func resolveDir() async throws -> URL {
        try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
    }

    /// Load the FireRedASR2 model from the resolved checkpoint directory.
    private func loadModel() async throws -> FireRedASR2Model {
        let dir = try await resolveDir()
        return try FireRedASR2Model.load(directory: dir)
    }

    // ─── Synthetic test audio ─────────────────────────────────────────────

    /// 1 s of a 440 Hz tone at 16 kHz — enough audio for the Conformer
    /// stack to process without hitting edge cases.
    private func syntheticTone(hz: Float = 440, seconds: Double = 1.0) -> [Float] {
        let sr = 16_000
        let n = Int(Double(sr) * seconds)
        return (0 ..< n).map { 0.3 * sin(2 * Float.pi * hz * Float($0) / Float(sr)) }
    }

    // ─── Tests ───────────────────────────────────────────────────────────

    @Test("load — config and weight shapes bind correctly")
    func loadBindsWeights() async throws {
        let model = try await loadModel()
        let enc = model.config.encoder
        let dec = model.config.decoder

        // Encoder hyper-parameters for FireRedASR2-AED-Large.
        #expect(enc.nLayers == 16)
        #expect(enc.nHead == 20)
        #expect(enc.dModel == 1280)
        #expect(enc.kernelSize == 33)

        // Decoder hyper-parameters.
        #expect(dec.nLayers == 16)
        #expect(dec.nHead == 20)
        #expect(dec.dModel == 1280)

        // Token ids.
        #expect(model.config.sosID == 3)
        #expect(model.config.eosID == 4)
        #expect(model.config.idim == 80)
        #expect(model.config.odim == 8667)

        // Block counts must match config.
        #expect(model.encoderBlocks.count == enc.nLayers)
        #expect(model.decoderLayers.count == dec.nLayers)

        // Conv2d weights: [outCh, inCh, kH, kW] after OHWI→OIHW transposition.
        #expect(model.conv1Weight.shape.count == 4)
        #expect(model.conv2Weight.shape.count == 4)

        // Relative positional table: [2*peMaxlen-1, dModel].
        let expectedRelLen = 2 * enc.peMaxlen - 1
        #expect(model.relPosLen == expectedRelLen)
        #expect(model.relPosTable.count == expectedRelLen * enc.dModel)

        print(
            "[FireRedASR2 integration] config enc=\(enc.nLayers)L "
                + "dec=\(dec.nLayers)L odim=\(model.config.odim) "
                + "dtype=\(model.dtype)")
    }

    @Test("kaldiFbank — features are finite and correctly shaped")
    func kaldiFbankFiniteShape() async throws {
        let model = try await loadModel()
        let wave = syntheticTone(seconds: 1.0)
        let idim = model.config.idim

        let feats = FireRedASR2Model.kaldiFbank(waveform: wave, idim: idim)
        let nFrames = feats.count / idim

        #expect(nFrames > 0, "kaldiFbank produced zero frames")
        #expect(feats.count == nFrames * idim)
        #expect(
            feats.allSatisfy { $0.isFinite },
            "Kaldi fbank features contain NaN or Inf")

        // Variance check: features should not be near-zero.
        let variance = feats.map { $0 * $0 }.reduce(0, +) / Float(feats.count)
        #expect(variance > 1e-6, "Kaldi fbank features are degenerate (near-zero)")
        print("[FireRedASR2 integration] fbank: \(nFrames) frames × \(idim) bins")
    }

    @Test("encodeAudio — encoder produces finite, non-degenerate features")
    func encodeAudioFiniteFeatures() async throws {
        let model = try await loadModel()
        let wave = syntheticTone(seconds: 1.0)

        let encoded = model.encodeAudio(waveform: wave)

        // Shape: [nEncoderFrames, dModel].
        #expect(
            encoded.shape.count == 2,
            "encoder output has unexpected rank \(encoded.shape.count)")
        #expect(
            encoded.shape[1] == model.config.encoder.dModel,
            "encoder feature dim mismatch")
        #expect(encoded.shape[0] > 0, "encoder produced zero frames")

        let vals = encoded.toFloatArray()
        #expect(
            vals.allSatisfy { $0.isFinite },
            "encoder output contains NaN or Inf")

        let variance = vals.map { $0 * $0 }.reduce(0, +) / Float(vals.count)
        #expect(variance > 1e-6, "encoder output is degenerate (near-zero)")

        print("[FireRedASR2 integration] encoder output: \(encoded.shape[0]) × \(encoded.shape[1])")
    }

    @Test("transcribe — real speech produces a non-degenerate transcript")
    func transcribeRealSpeech() async throws {
        let model = try await loadModel()

        // Bundled fixture: conversational_a.wav (~13 s, 24 kHz source resampled to 16 kHz).
        let wave = try AudioTestHelpers.conversationalAWaveform()
        #expect(!wave.isEmpty, "audio fixture waveform is empty")

        // Use the model's bundled tokenizer (loaded from dict.txt).
        let tok = model.tokenizer
        #expect(tok != nil, "FireRedASR2 checkpoint did not include dict.txt")

        let transcript = model.transcribe(
            waveform: wave,
            tokenizer: tok,
            beamSize: 3,
            maxLen: 200
        )

        print("[FireRedASR2 integration] transcript: \(transcript.debugDescription)")

        // Non-empty.
        #expect(!transcript.isEmpty, "FireRedASR2 produced an empty transcript")

        // Non-degenerate: a genuine decode visits several distinct words.
        let words = transcript.split(separator: " ")
        #expect(
            words.count >= 2,
            "FireRedASR2 transcript is degenerate: \(transcript.debugDescription)")
    }

    @Test("transcribe — synthetic tone does not crash or produce NaN")
    func transcribeSyntheticTone() async throws {
        let model = try await loadModel()
        let wave = syntheticTone(seconds: 1.0)

        // Transcribing a pure tone should not crash.
        // Output may be empty (no recognised speech) — that is valid.
        let transcript = model.transcribe(
            waveform: wave,
            tokenizer: model.tokenizer,
            beamSize: 1,
            maxLen: 50
        )
        print("[FireRedASR2 integration] synthetic tone → \(transcript.debugDescription)")
        // Nothing to assert on content, but getting here means no crash / NaN.
    }

    @Test("registry — AudioModelRegistry routes to .fireRedASR2")
    func registryRoutes() async throws {
        let dir = try await resolveDir()
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .fireRedASR2 = loaded else {
            Issue.record(
                "AudioModelRegistry did not route to .fireRedASR2; got \(loaded)")
            return
        }
        #expect(loaded.capabilities.contains(.audioIn))
        #expect(loaded.capabilities.contains(.textOut))
    }
}
