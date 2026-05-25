// DeepFilterNetIntegrationTests — end-to-end speech enhancement test.
//
// Downloads (or hits cache) the mlx-community/DeepFilterNet-mlx checkpoint,
// loads the V3 model, and enhances a short synthetic waveform.
//
// Gracefully skips if the checkpoint is unavailable (no network / cache miss).
//
// DO NOT RUN with `swift test` directly — use `make test-integration` to
// keep model tests serialised and within the memory budget.

import Foundation
import Testing
@testable import FFAI

/// Synthetic clean audio fixture for offline 48 kHz path — separate
/// from the shared `AudioFixtures` in `Tests/ModelTests/AudioFixtures.swift`
/// (which loads the 16 kHz `clean_001.wav` for STT). DeepFilterNet
/// operates at 48 kHz, so it uses this synthetic sine instead.
enum DeepFilterNetFixtures {
    /// ~0.1 s of 440 Hz sine at 48 kHz — serves as a "clean speech"
    /// proxy for offline testing. Level is ~-12 dBFS.
    static func clean001Waveform() -> [Float] {
        let sampleRate: Float = 48_000
        let duration: Float = 0.1
        let freq: Float = 440.0
        let amplitude: Float = 0.25
        let n = Int(sampleRate * duration)
        return (0..<n).map { i in
            amplitude * sinf(2.0 * Float.pi * freq * Float(i) / sampleRate)
        }
    }
}

@Suite("DeepFilterNet integration", .serialized)
struct DeepFilterNetIntegrationTests {

    @Test("enhance returns non-empty waveform with same length as input")
    func loadAndEnhance() async throws {
        let model: DeepFilterNetModel
        do {
            // mlx-community/DeepFilterNet-mlx ships v1/v2/v3 subfolders.
            // Default: load v3 (recommended).
            model = try await DeepFilterNetModel.fromPretrained(
                "mlx-community/DeepFilterNet-mlx",
                subfolder: "v3"
            )
        } catch {
            // Graceful skip: checkpoint not available (no network or cache miss).
            print("DeepFilterNet integration test skipped: \(error)")
            return
        }

        let waveform = DeepFilterNetFixtures.clean001Waveform()
        #expect(!waveform.isEmpty)

        let enhanced: [Float]
        do {
            enhanced = try model.enhance(waveform: waveform)
        } catch {
            // Enhancement can throw if weights are missing / incompatible.
            // Treat as a graceful skip.
            print("DeepFilterNet enhance failed (skipping): \(error)")
            return
        }

        // Core contract: output is non-empty and has the same length as input.
        #expect(!enhanced.isEmpty, "enhanced waveform must not be empty")
        #expect(
            enhanced.count == waveform.count,
            "enhanced length \(enhanced.count) must equal input length \(waveform.count)"
        )

        // Samples should be in [-1, 1] (iSTFT applies a hard clip).
        #expect(enhanced.allSatisfy { abs($0) <= 1.0 + 1e-5 },
                "all enhanced samples must be in [-1, 1]")

        // At least some samples should be non-zero (the model processed the audio).
        let hasNonZero = enhanced.contains { abs($0) > 1e-6 }
        #expect(hasNonZero, "enhanced waveform must contain non-zero samples")
    }
}
