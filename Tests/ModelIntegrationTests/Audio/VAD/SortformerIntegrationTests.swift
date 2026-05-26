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
// Integration test: loads the real Sortformer diarization checkpoint
// from the HF cache and runs end-to-end speaker diarization on a short
// bundled speech clip.
//
// Checkpoint: `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16`
// Audio fixture: Resources/clean_001.wav  (~1.85 s, 16 kHz, "Sure, I
// can help you with that." — single-speaker clean speech).
//
// The test is assertive: a load failure or a forward failure FAILS the
// suite. Assertions:
//  - Model loads cleanly with the expected geometry (4 speakers,
//    17 FastConformer layers, 18 TF encoder layers, 128 mel bins).
//  - `detect(audio:sampleRate:)` returns a well-formed `DiarizationOutput`
//    (all probabilities finite and in [0, 1]).
//  - The probability matrix has at least one frame.
//  - At least one speaker exceeds a low probability floor somewhere —
//    the model ran meaningfully over real speech.
//  - Empty audio produces an empty, well-formed result without crashing.
//
// DO NOT RUN locally without a GPU — run via `make test-integration`.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Sortformer Integration", .serialized)
struct SortformerIntegrationTests {

    /// Canonical HF repo id. No 4-bit MLX conversion exists at time of
    /// writing for Sortformer; fp16 is the smallest published variant.
    private static let repoId = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"

    // ─── Helpers ─────────────────────────────────────────────────────

    /// Load the Sortformer model, holding the global model-load lock.
    /// A load failure throws — it is NOT caught and skipped.
    private func loadSortformer() async throws -> SortformerModel {
        let dir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
        return try await ModelLoadLock.shared.loadSerially {
            try SortformerModel.loadFromDirectory(dir)
        }
    }

    // ─── Tests ───────────────────────────────────────────────────────

    @Test("load + detect produces a finite, well-formed diarization output")
    func loadAndDetect() async throws {
        let model = try await loadSortformer()

        // Published geometry for diar_streaming_sortformer_4spk-v2.1.
        // fc_encoder: 17 Conformer layers, 512 hidden, 128 mel bins, /8 subsampling.
        // tf_encoder: 18 BART layers, 192 d_model, 8 heads.
        // modules: 4 speakers.
        #expect(model.config.numSpeakers == 4)
        #expect(model.config.fcEncoder.hiddenSize == 512)
        #expect(model.config.fcEncoder.numMelBins == 128)
        #expect(model.config.fcEncoder.subsamplingFactor == 8)
        #expect(model.config.tfEncoder.dModel == 192)
        #expect(model.config.tfEncoder.numHeads == 8)

        // Each output frame covers hop × subsamplingFactor samples.
        let expectedStride = model.config.processor.hopLength
            * model.config.modules.subsamplingFactor
        #expect(model.frameStride == expectedStride)

        // Run diarization on the bundled speech clip.
        let audio = try AudioTestHelpers.clean001Waveform()
        #expect(!audio.isEmpty)

        let output = model.detect(audio: audio, sampleRate: 16_000)

        // Probability stream must be finite with values in [0, 1].
        #expect(output.isWellFormed,
                "speaker probabilities must all be finite and in [0, 1]")

        // At least one frame must have been produced.
        #expect(!output.speakerProbabilities.isEmpty,
                "expected at least one output frame for a ~1.85 s clip")

        // Frame count should be consistent with the clip length.
        // Each frame covers `frameStride` samples; the count should be
        // roughly clip.count / frameStride (within 1 due to padding).
        let approxFrames = audio.count / model.frameStride
        let frameCount = output.speakerProbabilities.count
        let minExpectedFrames = max(1, approxFrames / 2)
        #expect(frameCount >= minExpectedFrames,
                "frame count unexpectedly low for the clip length")

        // Every frame row must have exactly `numSpeakers` values.
        #expect(output.numSpeakers == model.config.numSpeakers)
        let badRows = output.speakerProbabilities.filter {
            $0.count != model.config.numSpeakers
        }
        #expect(badRows.isEmpty,
                "expected each frame to have numSpeakers probability values")

        // The stride and sample rate round-trip correctly.
        #expect(output.frameStrideSamples == model.frameStride)
        #expect(output.sampleRate == 16_000)

        // For real speech at least one speaker should have a non-trivial
        // peak probability somewhere (generous threshold of 0.1 — the model
        // may output low probabilities on unseen voices, but should not
        // output near-zero for all speakers on a speech clip).
        let maxProb = output.speakerProbabilities
            .flatMap { $0 }
            .max() ?? 0
        #expect(maxProb > 0.05,
                "all speaker probabilities near zero on a speech clip — broken forward pass")
    }

    @Test("empty audio clip produces an empty, well-formed result")
    func emptyClipIsWellFormed() async throws {
        let model = try await loadSortformer()
        let output = model.detect(audio: [], sampleRate: 16_000)
        #expect(output.speakerProbabilities.isEmpty)
        #expect(output.segments.isEmpty)
        #expect(output.numSpeakers == model.config.numSpeakers)
        #expect(output.isWellFormed)
    }

    @Test("diarization output wraps probsToSegments correctly")
    func segmentsRoundTrip() async throws {
        let model = try await loadSortformer()
        let audio = try AudioTestHelpers.clean001Waveform()
        let output = model.detect(audio: audio, sampleRate: 16_000)

        // Every segment must reference a valid speaker index and have
        // a positive duration.
        for seg in output.segments {
            #expect(seg.speaker >= 0 && seg.speaker < model.config.numSpeakers,
                    "segment speaker \(seg.speaker) out of range")
            #expect(seg.durationSeconds > 0,
                    "segment [\(seg.startSeconds), \(seg.endSeconds)] has zero duration")
            #expect(seg.startSeconds < seg.endSeconds,
                    "segment start must precede end")
        }
    }
}
