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
// FishSpeech integration test — load mlx-community/fish-audio-s2-pro-8bit
// from the HuggingFace cache (resolved through the standard
// ModelLocator + ModelLoadLock pattern) and run a Stage-1 + Stage-2
// synthesis pass.
//
// Coverage:
//   - Config parsed from real config.json
//   - Weights loaded (8-bit quantized, sharded via model.safetensors.index.json)
//   - FishSpeechModel constructed (slow backbone + fast decoder)
//   - generateCodes produces code frames (Stage-1)
//   - synthesize(...) + FishS1DAC decodes to a waveform (Stage-2) when codec
//       weights are available in the snapshot directory or a codec/ sub-folder.
//       If codec weights are absent the test asserts codecNotAvailable.
//
// DO NOT RUN this suite via `make test-unit`. Run serialised with
//   make test-integration
// Heavy GPU usage; do NOT run in parallel with other ModelTests.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "FishSpeech Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct FishSpeechIntegrationTests {

    /// Canonical HF repo id — the 8-bit MLX conversion of fish-audio-s2-pro.
    private static let repoId = "mlx-community/fish-audio-s2-pro-8bit"

    /// Resolve the cached snapshot directory through the standard
    /// ModelLocator path, serialised by ModelLoadLock so concurrent
    /// integration tests don't double-resolve the same checkpoint.
    private func resolveDirectory() async throws -> URL {
        try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
    }

    /// Build a FishSpeechModel from the resolved snapshot — the helper
    /// keeps the codec-path-aware FishSpeechModel.load surface intact
    /// (the codec lives alongside the model in `codec/` or `vocoder/`
    /// sub-folders, which `load(config:weights:directory:device:)`
    /// discovers automatically).
    private func loadModel() async throws -> FishSpeechModel {
        let dir = try await resolveDirectory()
        let config = try ModelConfig.load(from: dir)
        let weights = try SafeTensorsBundle(directory: dir)
        return try FishSpeechModel.load(
            config: config, weights: weights, directory: dir, device: .shared
        )
    }

    @Test("load config + weights from cached checkpoint")
    func loadCheckpoint() async throws {
        let dir = try await resolveDirectory()

        // Load config
        let config = try ModelConfig.load(from: dir)
        #expect(config.modelType == "fish_qwen3_omni")

        // Load weights
        let weights = try SafeTensorsBundle(directory: dir)
        #expect(!weights.allKeys.isEmpty)

        // Verify key weight shapes are present.
        #expect(weights.has("model.norm.weight"))
        #expect(weights.has("model.fast_norm.weight"))
        #expect(weights.has("model.fast_output.weight"))

        // Build the model.
        let model = try FishSpeechModel.load(
            config: config,
            weights: weights,
            directory: dir,
            device: .shared
        )

        // Structural invariants.
        #expect(model.fishConfig.sampleRate == 44_100)
        #expect(model.fishConfig.numCodebooks == 10)
        #expect(model.fishConfig.textConfig.nLayer == 36)
        #expect(model.fishConfig.audioDecoderConfig.nLayer == 4)
        #expect(model.slowLayers.count == 36)
        #expect(model.fastLayers.count == 4)
        #expect(model.sampleRate == 44_100)

        // AudioModel conformance: the model is registered correctly.
        #expect(FishSpeech.modelTypes.contains(model.fishConfig.modelType))
    }

    @Test("synthesize produces a waveform (Stage-2) or throws codecNotAvailable")
    func synthesizeStage2() async throws {
        let model = try await loadModel()

        // Stage-2: when FishS1DAC codec weights are present in the snapshot
        // directory (or a codec/ sub-folder), synthesize returns a non-empty
        // waveform. When codec weights are absent it throws codecNotAvailable.
        // Both outcomes are valid; any other error is a regression.
        do {
            let pcm = try model.synthesize(
                text: "Hello world.",
                parameters: AudioGenerationParameters(maxTokens: 32)
            )

            // If we get here, codec was loaded and decoding succeeded.
            #expect(!pcm.isEmpty, "synthesize returned an empty waveform")
            #expect(pcm.allSatisfy { $0.isFinite }, "waveform contains non-finite samples")

            // Sane amplitude: RMS should be well above silence and below clipping.
            let rms = sqrt(pcm.map { $0 * $0 }.reduce(0, +) / Float(pcm.count))
            #expect(rms > 0.0, "RMS is zero — codec returned silence")
            #expect(rms < 2.0, "RMS is implausibly large — codec may have exploded")

        } catch AudioGenerationError.codecNotAvailable {
            // Codec weights absent from this snapshot — acceptable.
            print(
                "FishSpeech Stage-2: codec weights not found, synthesize threw codecNotAvailable (expected)."
            )
        } catch {
            Issue.record("synthesize threw unexpected error: \(error)")
        }
    }

    @Test("generateCodes produces non-empty code frames for short text (Stage-1)")
    func generateCodesSmoke() async throws {
        let model = try await loadModel()

        // generateCodes is the pre-codec stage; it should complete without
        // throwing and return at least one frame.
        let codes = try model.generateCodes(
            text: "Hi.",
            maxTokens: 16,
            temperature: 0.0,  // greedy for determinism
            topP: 1.0,
            topK: 1,
            device: .shared
        )

        // codes is [numCodebooks][numFrames]; both dimensions > 0.
        #expect(codes.count == model.fishConfig.numCodebooks)
        #expect(codes.first?.isEmpty == false)
    }
}
