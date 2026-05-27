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
// DeepFilterNetIntegrationTests — end-to-end speech enhancement test.
//
// Downloads (or hits cache) the `mlx-community/DeepFilterNet-mlx` checkpoint,
// loads the V3 model, and enhances a short synthetic waveform.
//
// A load failure FAILS the suite — no "skip if missing" logic. The model
// load is serialised through `ModelLoadLock.shared` so concurrent suites
// don't race the GPU.
//
// DO NOT RUN with `swift test` directly — use `make test-integration` to
// keep model tests serialised and within the memory budget.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

/// Synthetic clean audio fixture for offline 48 kHz path — separate
/// from the shared 16 kHz `clean_001.wav` fixture (which targets STT).
/// DeepFilterNet operates at 48 kHz, so this synthetic sine stands in
/// as "clean speech" for the offline path.
enum DeepFilterNetFixtures {
    /// ~0.1 s of 440 Hz sine at 48 kHz — serves as a "clean speech"
    /// proxy for offline testing. Level is ~-12 dBFS.
    static func clean001Waveform() -> [Float] {
        let sampleRate: Float = 48_000
        let duration: Float = 0.1
        let freq: Float = 440.0
        let amplitude: Float = 0.25
        let n = Int(sampleRate * duration)
        return (0 ..< n).map { i in
            amplitude * sinf(2.0 * Float.pi * freq * Float(i) / sampleRate)
        }
    }
}

@Suite(
    "DeepFilterNet Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct DeepFilterNetIntegrationTests {

    /// Canonical HF repo id. DeepFilterNet does not ship a 4-bit
    /// MLX conversion; the upstream `-mlx` repo is the only published
    /// variant and is small (~1 MB) so quantisation is not motivated.
    private static let repoId = "mlx-community/DeepFilterNet-mlx"

    @Test("enhance returns non-empty waveform with same length as input")
    func loadAndEnhance() async throws {
        // mlx-community/DeepFilterNet-mlx ships v1/v2/v3 subfolders.
        // Default: load v3 (recommended).
        let model = try await ModelLoadLock.shared.loadSerially {
            try await DeepFilterNetModel.fromPretrained(
                Self.repoId,
                subfolder: "v3"
            )
        }

        let waveform = DeepFilterNetFixtures.clean001Waveform()
        #expect(!waveform.isEmpty)

        let enhanced = try model.enhance(waveform: waveform)

        // Core contract: output is non-empty and has the same length as input.
        #expect(!enhanced.isEmpty, "enhanced waveform must not be empty")
        #expect(
            enhanced.count == waveform.count,
            "enhanced length \(enhanced.count) must equal input length \(waveform.count)"
        )

        // Samples should be in [-1, 1] (iSTFT applies a hard clip).
        #expect(
            enhanced.allSatisfy { abs($0) <= 1.0 + 1e-5 },
            "all enhanced samples must be in [-1, 1]")

        // At least some samples should be non-zero (the model processed the audio).
        let hasNonZero = enhanced.contains { abs($0) > 1e-6 }
        #expect(hasNonZero, "enhanced waveform must contain non-zero samples")
    }
}
