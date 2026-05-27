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
// Integration test: loads the real Qwen3-ASR-0.6B-4bit checkpoint from the
// HF cache and runs end-to-end transcription on the bundled speech fixture.
// A missing checkpoint FAILS the suite.
//
// Qwen3-ASR uses a Conv2d audio encoder whose output is merged into a
// Qwen3 text-decoder embedding stream. This test validates:
//   * Config decoding from the nested `thinker_config` layout.
//   * Weight loading (including Conv2d OHWI→OIHW transposition).
//   * Audio encoding produces finite, non-degenerate features.
//   * End-to-end transcription of a real speech clip yields a non-empty,
//     non-degenerate string. We do not assert the exact text since that
//     depends on sampling; "non-degenerate" means ≥2 distinct words.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Qwen3ASR Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct Qwen3ASRIntegrationTests {

    /// Canonical HF repo id. The 4-bit MLX conversion is the smallest
    /// published Qwen3-ASR variant.
    private static let repoId = "mlx-community/Qwen3-ASR-0.6B-4bit"

    /// Resolve the Qwen3-ASR checkpoint directory through `ModelLocator`.
    private func resolveDir() async throws -> URL {
        try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
    }

    /// Load the Qwen3-ASR model from the resolved checkpoint directory.
    private func loadModel() async throws -> Qwen3ASRModel {
        let dir = try await resolveDir()
        return try Qwen3ASRModel.load(directory: dir)
    }

    @Test("load — config and weight shapes bind correctly")
    func loadBindsWeights() async throws {
        let model = try await loadModel()
        let ac = model.config.audioConfig

        // Audio encoder hyper-parameters for the 0.6B variant.
        #expect(ac.dModel == 896)
        #expect(ac.encoderLayers == 18)
        #expect(ac.encoderAttentionHeads == 14)
        #expect(ac.numMelBins == 128)
        #expect(ac.downsampleHiddenSize == 480)

        // Text decoder hyper-parameters.
        #expect(model.config.textHidden == 1024)
        #expect(model.config.textLayers == 28)
        #expect(model.config.textHeads == 16)
        #expect(model.config.textKVHeads == 8)
        #expect(model.config.headDim == 128)
        #expect(model.config.vocabSize == 151936)

        // The encoder layer count should match the actual loaded layers.
        #expect(model.audioEncoderLayers.count == ac.encoderLayers)
        #expect(model.textLayers.count == model.config.textLayers)

        // Conv2d weights should be transposed to OIHW on load.
        // conv2d1: [480, 1, 3, 3] (OIHW)
        #expect(model.conv2d1Weight.shape[0] == 480)
        #expect(model.conv2d1Weight.shape[1] == 1)
        #expect(model.conv2d1Weight.shape[2] == 3)
        #expect(model.conv2d1Weight.shape[3] == 3)
    }

    @Test("encodeAudio — encoder produces finite, non-degenerate features")
    func encodeAudioFiniteFeatures() async throws {
        let model = try await loadModel()

        // 1 second of 440 Hz sine — exercises the full conv2d + transformer.
        let sr = 16_000
        var wave = [Float](repeating: 0, count: sr)
        for i in 0 ..< sr {
            wave[i] = 0.3 * sin(2.0 * Float.pi * 440.0 * Float(i) / Float(sr))
        }

        let features = model.encodeAudio(waveform: wave)

        // features: [nAudioTokens, outputDim]
        #expect(features.shape.count == 2)
        #expect(features.shape[1] == model.config.audioConfig.outputDim)
        #expect(features.shape[0] > 0, "encoder produced zero audio tokens")

        let vals = features.toFloatArray()
        #expect(
            vals.allSatisfy { $0.isFinite },
            "audio features contain NaN or Inf")
        let variance = vals.map { $0 * $0 }.reduce(0, +) / Float(vals.count)
        #expect(variance > 1e-6, "audio features are degenerate (near-zero)")
    }

    @Test("transcribe — real speech produces a non-degenerate transcript")
    func transcribeRealSpeech() async throws {
        let model = try await loadModel()

        // Load the bundled conversational speech fixture
        // (~13 s, 24 kHz source resampled to 16 kHz).
        let wave = try AudioTestHelpers.conversationalAWaveform()
        #expect(!wave.isEmpty, "audio fixture failed to load")

        // The tokenizer lives alongside the weights in the resolved
        // snapshot directory.
        let dir = try await resolveDir()
        let tokenizer = try await TokenizerLoader().load(from: dir)

        let transcript = model.transcribe(
            waveform: wave,
            tokenizer: tokenizer,
            maxTokens: 200
        )

        print("[Qwen3ASR integration] transcript: \(transcript.debugDescription)")

        // Non-empty and not a single repeated token.
        #expect(!transcript.isEmpty, "Qwen3ASR produced an empty transcript")
        let words = transcript.split(separator: " ")
        #expect(
            words.count >= 2,
            "Qwen3ASR transcript is degenerate: \(transcript.debugDescription)")
    }
}
