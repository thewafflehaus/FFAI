// Slow integration test: downloads (or hits cache) the FireRedVAD
// checkpoint and runs voice-activity detection on a real speech clip.
//
// Checkpoint: FireRedTeam/FireRedVAD
// Sub-directory: VAD/ (non-streaming variant)
// Audio fixture: AudioFixtures.clean001Waveform()
//   (~1.85 s, 16 kHz, "Sure, I can help you with that." — single-speaker clean speech).
//
// The test is assertive: a load failure or a forward failure FAILS the
// suite. Assertions:
//   - Model loads cleanly with the expected architecture geometry.
//   - `detect(audio:sampleRate:)` returns a well-formed `VADOutput`
//     (all probabilities finite and in [0, 1]).
//   - The probability stream has at least one frame.
//   - The max speech probability exceeds 0.3 on real speech — the model
//     ran meaningfully over the "Sure, I can help you with that." clip.
//   - At least one speech segment is detected.
//   - Empty audio produces an empty, well-formed result without crashing.
//
// DISABLED REASON:
//   No mlx-community safetensors conversion of FireRedTeam/FireRedVAD
//   exists as of 2026-05-22. The loader reads the PyTorch `.pth.tar`
//   directly from the `FireRedTeam/FireRedVAD` HuggingFace snapshot, but
//   the HF repository ships the checkpoint in a sub-directory (`VAD/`)
//   rather than at the snapshot root. `ModelLocator` resolves the top-
//   level snapshot directory (`FireRedTeam/FireRedVAD`), not the `VAD/`
//   sub-directory, and the sub-directory layout is non-standard for
//   HuggingFace model cards.
//
//   To enable this suite:
//   1. Download the snapshot manually:
//      python3 -c "from huggingface_hub import snapshot_download; snapshot_download('FireRedTeam/FireRedVAD')"
//   2. Locate the VAD sub-directory:
//      ~/.cache/huggingface/hub/models--FireRedTeam--FireRedVAD/snapshots/<hash>/VAD/
//   3. Update `vadDirectory` below to the full resolved path.
//   4. Remove the `.disabled` trait from the @Suite.
//   Alternatively, when an mlx-community safetensors snapshot appears at
//   the repository root, update the repoId constant and re-enable.
//
// DO NOT RUN locally without downloading the checkpoint first.

import Foundation
import Testing
@testable import FFAI

// DISABLED: no mlx-community FireRedVAD safetensors conversion exists upstream;
// re-enable when a checkpoint lands at the HF repo root or a conversion appears.
@Suite("FireRedVAD integration",
       .disabled("no mlx-community FireRedVAD conversion exists upstream — re-enable when a root-level safetensors checkpoint lands at FireRedTeam/FireRedVAD"),
       .serialized)
struct FireRedVADIntegrationTests {

    /// Top-level HuggingFace repo id.
    static let repoId = "FireRedTeam/FireRedVAD"

    /// The `VAD/` sub-directory within the snapshot holds the non-streaming model.
    /// Set this to the resolved absolute path of the VAD sub-directory after
    /// downloading the snapshot.
    static let vadSubdir = "VAD"

    // ─── Helpers ─────────────────────────────────────────────────────

    /// Resolve and load the FireRedVAD model, holding the global model-load lock.
    /// A load failure throws — it is NOT caught and skipped.
    private func loadFireRedVAD() async throws -> FireRedVADModel {
        // Resolve the top-level snapshot and append the VAD sub-directory.
        let snapshotDir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
        let vadDir = snapshotDir.appendingPathComponent(Self.vadSubdir)
        return try await ModelLoadLock.shared.loadSerially {
            try FireRedVADModel.loadFromDirectory(vadDir)
        }
    }

    // ─── Tests ───────────────────────────────────────────────────────

    @Test("load + detect — produces a finite probability stream on a real speech clip")
    func loadAndDetect() async throws {
        let model = try await loadFireRedVAD()

        // Verify published architecture geometry (from model.pth.tar args).
        #expect(model.config.numBlocks == 8)
        #expect(model.config.hiddenSize == 256)
        #expect(model.config.projSize == 128)
        #expect(model.config.lookbackOrder == 20)
        #expect(model.config.lookaheadOrder == 20)
        #expect(model.config.idim == 80)
        #expect(model.config.odim == 1)
        #expect(model.config.frameShiftSamples == 160)   // 10 ms at 16 kHz
        #expect(model.config.frameLengthSamples == 400)  // 25 ms at 16 kHz

        let waveform = try AudioFixtures.clean001Waveform()
        let output = try model.detect(audio: waveform, sampleRate: 16000)

        // Probability stream must be finite and in [0, 1].
        #expect(output.isWellFormed)
        #expect(!output.probabilities.isEmpty)
        #expect(output.frameStrideSamples == 160)
        #expect(output.sampleRate == 16000)

        // Frame count: (waveformLen - frameLengthSamples) / frameShiftSamples + 1
        // for snip_edges=true. The clean_001.wav clip is ~1.85 s = ~29600 samples.
        let expectedFrames = (waveform.count - 400) / 160 + 1
        #expect(output.probabilities.count == expectedFrames,
                "expected \(expectedFrames) frames, got \(output.probabilities.count)")

        // The "Sure, I can help you with that." clip must produce at least one
        // speech segment and a max probability above a generous floor.
        let maxProb = output.probabilities.max() ?? 0
        #expect(maxProb > 0.3,
                "max speech probability \(maxProb) — FireRedVAD should detect the speech clip")
        #expect(!output.segments.isEmpty)

        // Total speech must be positive and not span the full clip duration.
        let clipDuration = Double(waveform.count) / 16000.0
        #expect(output.totalSpeechSeconds > 0)
        #expect(output.totalSpeechSeconds < clipDuration)

        print("FireRedVAD detect: \(output.probabilities.count) frames, "
              + "maxProb=\(maxProb), segments=\(output.segments.count), "
              + "totalSpeech=\(output.totalSpeechSeconds)s")
    }

    @Test("detect — empty clip yields an empty, well-formed result")
    func emptyClip() async throws {
        let model = try await loadFireRedVAD()
        let output = try model.detect(audio: [], sampleRate: 16000)
        #expect(output.probabilities.isEmpty)
        #expect(output.segments.isEmpty)
        #expect(output.isWellFormed)
    }

    @Test("detect — unsupported sample rate is rejected")
    func unsupportedSampleRate() async throws {
        let model = try await loadFireRedVAD()
        #expect(throws: FireRedVADError.self) {
            _ = try model.detect(audio: [0, 0, 0], sampleRate: 44100)
        }
    }

    @Test("registry dispatch — resolves FireRedTeam/FireRedVAD/VAD to .fireRedVAD kind")
    func registryDispatch() async throws {
        let snapshotDir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
        let vadDir = snapshotDir.appendingPathComponent(Self.vadSubdir)
        let loaded = try await ModelLoadLock.shared.loadSerially {
            try VADModelRegistry.loadFromDirectory(vadDir)
        }
        #expect(loaded.kind == .fireRedVAD)
    }
}
