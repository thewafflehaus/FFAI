// Slow integration test: downloads (or hits cache) the SileroVAD
// checkpoint and runs voice-activity detection on a synthetic clip.
// Skipped automatically if the network or checkpoint isn't available.
//
// SileroVAD is tiny (~1.5M params, a few MB on disk), so the
// integration suite stays fast. The test clip is generated in-process —
// a 200 Hz tone burst surrounded by silence — so no audio fixture is
// needed; the burst should register as at least one speech segment.

import Foundation
import Testing
@testable import FFAI

@Suite("SileroVAD integration", .serialized)
struct SileroVADIntegrationTests {

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

    @Test("load + detect produces a finite probability stream with speech")
    func loadAndDetect() async throws {
        let model: SileroVADModel
        do {
            model = try await SileroVADModel.fromPretrained("mlx-community/silero-vad")
        } catch {
            print("SileroVAD integration test skipped: \(error)")
            return
        }

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
        #expect(maxProb > 0.3)
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
        let model: SileroVADModel
        do {
            model = try await SileroVADModel.fromPretrained("mlx-community/silero-vad")
        } catch {
            print("SileroVAD integration test skipped: \(error)")
            return
        }
        let output = try model.detect(audio: [], sampleRate: 16000)
        #expect(output.probabilities.isEmpty)
        #expect(output.segments.isEmpty)
        #expect(output.isWellFormed)
    }

    @Test("unsupported sample rate is rejected")
    func unsupportedSampleRate() async throws {
        let model: SileroVADModel
        do {
            model = try await SileroVADModel.fromPretrained("mlx-community/silero-vad")
        } catch {
            print("SileroVAD integration test skipped: \(error)")
            return
        }
        #expect(throws: SileroVADError.self) {
            _ = try model.detect(audio: [0, 0, 0], sampleRate: 44100)
        }
    }
}
