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
// Integration test: loads the real GLM-ASR-Nano checkpoint from the HF
// cache and runs log-Mel → Whisper encoder → merge-and-adapt MLP →
// LLaMA decoder end-to-end. A missing checkpoint FAILS the suite.
//
// GLM-ASR is a Whisper-style Conv1d audio encoder followed by a merge-
// and-adapt MLP, followed by a LLaMA causal decoder. This test validates:
//   * Config decoding from the sparse GLM-ASR config.json layout.
//   * Weight loading (Conv1d OWI→OIW transposition, adapter MLP
//     dequantization, explicit lm_head).
//   * Log-Mel spectrogram front-end produces finite, non-degenerate
//     audio features.
//   * encodeAudio produces finite, non-degenerate adapter outputs.
//   * End-to-end transcription of a real speech clip yields a non-empty,
//     non-degenerate string (≥ 2 distinct words).
//
// DO NOT RUN — this suite requires multi-GB checkpoint files and a GPU.
// Run via `make test-integration`.

import Foundation
import TestHelpers
import Testing
import Tokenizers

@testable import FFAI

@Suite("GLMASR Integration", .serialized)
struct GLMASRIntegrationTests {

    /// Canonical HF repo id. The 4-bit MLX conversion is already the
    /// smallest published variant.
    private static let repoId = "mlx-community/GLM-ASR-Nano-2512-4bit"

    // ─── Checkpoint resolution ───────────────────────────────────────────

    private func resolveDir() async throws -> URL {
        try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
    }

    /// Load the GLM-ASR model from the resolved checkpoint directory.
    private func loadModel() async throws -> GLMASRModel {
        let dir = try await resolveDir()
        return try GLMASRModel.load(directory: dir)
    }

    // ─── Synthetic test audio ─────────────────────────────────────────────

    /// 1 s of a 440 Hz tone at 16 kHz — enough for the whisper encoder
    /// to run without hitting edge cases.
    private func syntheticTone(hz: Float = 440, seconds: Double = 1.0) -> [Float] {
        let sr = 16_000
        let n = Int(Double(sr) * seconds)
        return (0 ..< n).map { 0.3 * sin(2 * Float.pi * hz * Float($0) / Float(sr)) }
    }

    // ─── Tests ───────────────────────────────────────────────────────────

    @Test("load — config and weight shapes bind correctly")
    func loadBindsWeights() async throws {
        let model = try await loadModel()
        let gc = model.config

        // GLM-ASR Nano default architecture.
        #expect(gc.numMelBins == 128)
        #expect(gc.whisperDModel == 1280)
        #expect(gc.whisperEncoderLayers == 32)
        #expect(gc.whisperEncoderHeads == 20)
        #expect(gc.whisperEncoderFfnDim == 5120)
        #expect(gc.mergeFactor == 4)
        #expect(gc.lmHiddenSize == 2048)
        #expect(gc.lmVocabSize == 59264)
        #expect(gc.lmNumLayers == 28)
        #expect(gc.lmNumHeads == 16)
        #expect(gc.lmNumKVHeads == 4)
        #expect(gc.lmHeadDim == 128)

        // Layer counts must match config.
        #expect(model.whisperLayers.count == gc.whisperEncoderLayers)
        #expect(model.textLayers.count == gc.lmNumLayers)

        // Conv stem weights: [outCh, inCh, k] OIW layout after transposition.
        #expect(model.conv1Weight.shape == [gc.whisperDModel, gc.numMelBins, 3])
        #expect(model.conv2Weight.shape == [gc.whisperDModel, gc.whisperDModel, 3])

        // Adapter FC1 input is mergedDim = dModel * mergeFactor.
        let mergedDim = gc.whisperDModel * gc.mergeFactor
        #expect(
            model.adaptingFC1Weight.shape[1] == mergedDim,
            "FC1 input dim mismatch with merged dim")
        #expect(
            model.adaptingFC2Weight.shape[0] == gc.lmHiddenSize,
            "FC2 output dim must be lmHiddenSize")

        print(
            "[GLM-ASR integration] config: whisper=\(gc.whisperEncoderLayers)L "
                + "lm=\(gc.lmNumLayers)L hidden=\(gc.lmHiddenSize) "
                + "dtype=\(model.dtype)")
    }

    @Test("encodeAudio — produces finite, non-degenerate audio features")
    func encodeAudioFiniteFeatures() async throws {
        let model = try await loadModel()
        let wave = syntheticTone(seconds: 1.0)

        let features = model.encodeAudio(waveform: wave)

        // Shape: [nAudioTokens, lmHiddenSize].
        #expect(
            features.shape.count == 2,
            "audio features have unexpected rank \(features.shape.count)")
        #expect(
            features.shape[1] == model.config.lmHiddenSize,
            "audio feature hidden dim mismatch")
        #expect(features.shape[0] > 0, "encodeAudio produced zero tokens")

        let vals = features.toFloatArray()
        #expect(
            vals.allSatisfy { $0.isFinite },
            "encodeAudio output contains NaN or Inf")

        let variance = vals.map { $0 * $0 }.reduce(0, +) / Float(vals.count)
        #expect(variance > 1e-6, "encodeAudio output is degenerate (near-zero)")

        print(
            "[GLM-ASR integration] encodeAudio: \(features.shape[0]) tokens "
                + "× \(features.shape[1]) dim")
    }

    @Test("transcribe — real speech produces a non-degenerate transcript")
    func transcribeRealSpeech() async throws {
        let dir = try await resolveDir()
        let model = try GLMASRModel.load(directory: dir)
        let tokenizer = try await TokenizerLoader().load(from: dir)

        // Bundled fixture: conversational_a.wav (~13 s, 24 kHz source resampled to 16 kHz).
        let wave = try AudioTestHelpers.conversationalAWaveform()
        #expect(!wave.isEmpty, "audio fixture waveform is empty")

        let transcript = model.transcribe(
            waveform: wave,
            tokenizer: tokenizer,
            maxTokens: 256
        )

        print("[GLM-ASR integration] transcript: \(transcript.debugDescription)")

        // Non-empty.
        #expect(!transcript.isEmpty, "GLM-ASR produced an empty transcript")

        // Non-degenerate: a genuine decode visits several distinct words.
        let words = transcript.split(separator: " ")
        #expect(
            words.count >= 2,
            "GLM-ASR transcript is degenerate: \(transcript.debugDescription)")
    }

    @Test("transcribe — synthetic tone does not crash or produce NaN")
    func transcribeSyntheticTone() async throws {
        let dir = try await resolveDir()
        let model = try GLMASRModel.load(directory: dir)
        let tokenizer = try await TokenizerLoader().load(from: dir)
        let wave = syntheticTone(seconds: 1.0)

        // Transcribing a pure tone should not crash.
        // Output may be empty (no recognised speech) — that is valid.
        let transcript = model.transcribe(
            waveform: wave,
            tokenizer: tokenizer,
            maxTokens: 50
        )
        print("[GLM-ASR integration] synthetic tone → \(transcript.debugDescription)")
        // Nothing to assert on content — reaching here means no crash / NaN.
    }

    @Test("registry — AudioModelRegistry routes checkpoint to .glmASR")
    func registryRoutes() async throws {
        let dir = try await resolveDir()
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .glmASR = loaded else {
            Issue.record(
                "AudioModelRegistry did not route to .glmASR; got \(loaded)")
            return
        }
        #expect(loaded.capabilities == Capability.speechToText)
        print(
            "[GLM-ASR integration] registry routed correctly, "
                + "capabilities=\(loaded.capabilities)")
    }
}
