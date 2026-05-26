// Slow integration test: downloads (or hits cache) the SileroVAD
// checkpoint and runs voice-activity detection on a synthetic clip.
//
// SileroVAD is tiny (~1.5M params, a few MB on disk), so the
// integration suite stays fast. The test clip is generated in-process —
// a 200 Hz tone burst surrounded by silence — so no audio fixture is
// needed; the burst should register as at least one speech segment.
//
// This suite is assertive: a load failure FAILS the test rather than
// silently skipping. The contract is "the SileroVAD checkpoint loads
// and actually detects the tone burst as speech".

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("SileroVAD Integration", .serialized)
struct SileroVADIntegrationTests {

    /// The published SileroVAD checkpoint.
    static let repoId = "mlx-community/silero-vad"

    /// Build a mono 16 kHz clip: `silenceMs` of silence, then a tone
    /// burst of `burstMs`, then `silenceMs` of silence again. The tone
    /// is a 200 Hz sine at moderate amplitude — energetic enough to
    /// trip the speech detector, while the silence regions stay near
    /// zero.
    private func syntheticClip(sampleRate: Int = 16000,
                               silenceMs: Int = 600,
                               burstMs: Int = 1200) -> [Float] {
        let silenceN = sampleRate * silenceMs / 1000
        let burstN = sampleRate * burstMs / 1000
        var clip = [Float](repeating: 0, count: silenceN)
        let twoPiF = 2 * Float.pi * 200
        for i in 0..<burstN {
            let t = Float(i) / Float(sampleRate)
            // Mix two tones so the spectrum is broadband-ish, like voice.
            clip.append(0.4 * sinf(twoPiF * t) + 0.2 * sinf(2 * twoPiF * t))
        }
        clip.append(contentsOf: [Float](repeating: 0, count: silenceN))
        return clip
    }

    /// Load the SileroVAD model, holding the global model-load lock. A
    /// load failure throws — it is NOT caught and skipped, so a broken
    /// loader fails the suite.
    private func loadSileroVAD() async throws -> SileroVADModel {
        try await ModelLoadLock.shared.loadSerially {
            try await SileroVADModel.fromPretrained(Self.repoId)
        }
    }

    @Test("load + detect produces a finite probability stream with speech")
    func loadAndDetect() async throws {
        let model = try await loadSileroVAD()

        // Branch geometry from the published config.
        #expect(model.config.branch16k.chunkSize == 512)
        #expect(model.config.branch16k.contextSize == 64)
        #expect(model.config.branch8k.chunkSize == 256)

        let clip = syntheticClip()
        let output = try model.detect(audio: clip, sampleRate: 16000)

        // The probability stream must be finite and in [0, 1].
        #expect(output.isWellFormed)
        #expect(!output.probabilities.isEmpty)
        #expect(output.frameStrideSamples == 512)
        #expect(output.sampleRate == 16000)

        // Frame count should match: ceil(clipLen / chunkSize).
        let expectedFrames = (clip.count + 511) / 512
        #expect(output.probabilities.count == expectedFrames)

        // The tone burst should produce at least one speech segment, and
        // some frame should cross a generous probability floor.
        let maxProb = output.probabilities.max() ?? 0
        #expect(maxProb > 0.3,
                "max speech probability \(maxProb) — the forward pass should detect the tone burst")
        #expect(!output.segments.isEmpty)

        // Detected speech should not span the whole clip — the leading
        // and trailing silence should keep total speech below the full
        // duration.
        if let first = output.segments.first {
            #expect(first.durationSeconds > 0)
        }
        #expect(output.totalSpeechSeconds < Double(clip.count) / 16000.0)
    }

    @Test("empty clip yields an empty, well-formed result")
    func emptyClip() async throws {
        let model = try await loadSileroVAD()
        let output = try model.detect(audio: [], sampleRate: 16000)
        #expect(output.probabilities.isEmpty)
        #expect(output.segments.isEmpty)
        #expect(output.isWellFormed)
    }

    @Test("unsupported sample rate is rejected")
    func unsupportedSampleRate() async throws {
        let model = try await loadSileroVAD()
        #expect(throws: SileroVADError.self) {
            _ = try model.detect(audio: [0, 0, 0], sampleRate: 44100)
        }
    }
}
