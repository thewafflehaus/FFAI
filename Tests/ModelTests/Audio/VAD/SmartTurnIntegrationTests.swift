// Slow integration test: downloads (or hits cache) the SmartTurn
// checkpoint and runs conversational endpoint detection on a synthetic
// clip.
//
// SmartTurn-v3 is tiny (~8M params, a few MB on disk) so the
// integration suite stays fast. The model is loaded via
// `VADModelRegistry` to exercise the registry's architecture-detection
// + dispatch path, then `predictEndpoint` is run on an in-process clip
// (no audio fixture needed) and the utterance-level "turn complete"
// probability is asserted to be a real, finite, in-range result.
//
// This suite is assertive: a load failure FAILS the test rather than
// silently skipping. The contract is "the SmartTurn checkpoint loads
// and produces a coherent endpoint probability".

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("SmartTurn Integration", .serialized)
struct SmartTurnIntegrationTests {

    /// The published SmartTurn-v3 checkpoint (matches the
    /// `MLXAUDIO_SMARTTURN_REPO` default in mlx-audio-swift).
    static let repoId = "mlx-community/smart-turn-v3"

    /// Build a mono 16 kHz clip: `silenceMs` of silence framing a tone
    /// burst of `burstMs`. SmartTurn consumes a whole utterance and
    /// emits one probability, so the exact content matters less than it
    /// being a finite, non-degenerate waveform of realistic length.
    private func syntheticClip(sampleRate: Int = 16000,
                               silenceMs: Int = 400,
                               burstMs: Int = 800) -> [Float] {
        let silenceN = sampleRate * silenceMs / 1000
        let burstN = sampleRate * burstMs / 1000
        var clip = [Float](repeating: 0, count: silenceN)
        let twoPiF = 2 * Float.pi * 180
        for i in 0..<burstN {
            let t = Float(i) / Float(sampleRate)
            // Two tones so the spectrum is broadband-ish, like voice.
            clip.append(0.35 * sinf(twoPiF * t) + 0.18 * sinf(2 * twoPiF * t))
        }
        clip.append(contentsOf: [Float](repeating: 0, count: silenceN))
        return clip
    }

    /// Load the SmartTurn model through `VADModelRegistry`, holding the
    /// global model-load lock. A load failure throws — it is NOT caught
    /// and skipped, so a broken loader fails the suite.
    private func loadSmartTurn() async throws -> SmartTurnModel {
        let loaded = try await ModelLoadLock.shared.loadSerially {
            try await VADModelRegistry.fromPretrained(Self.repoId)
        }
        guard case .smartTurn(let model) = loaded else {
            Issue.record("VADModelRegistry resolved \(Self.repoId) to \(loaded.kind), expected .smartTurn")
            throw SmartTurnError.invalidAudio("registry dispatched to the wrong family")
        }
        return model
    }

    @Test("load — SmartTurn checkpoint binds a coherent encoder config")
    func loadBindsConfig() async throws {
        let model = try await loadSmartTurn()
        let c = model.config
        // Encoder geometry must be self-consistent: headDim divides
        // evenly and every published dimension is positive.
        #expect(c.dModel > 0)
        #expect(c.encoderLayers > 0)
        #expect(c.encoderAttentionHeads > 0)
        #expect(c.dModel % c.encoderAttentionHeads == 0)
        #expect(c.numMelBins > 0)
        #expect(c.nFft > 0)
        #expect(c.hopLength > 0)
        #expect(c.samplingRate == 16000)
        // The encoder stack must have loaded `encoderLayers` layers.
        #expect(model.layers.count == c.encoderLayers)
    }

    @Test("predictEndpoint — produces a finite in-range turn probability")
    func predictEndpointFiniteProbability() async throws {
        let model = try await loadSmartTurn()
        let clip = syntheticClip()
        let output = model.predictEndpoint(audio: clip)

        // The endpoint probability must be a real number in [0, 1].
        #expect(output.probability.isFinite)
        #expect(output.probability >= 0)
        #expect(output.probability <= 1)
        // `prediction` must be the thresholded probability.
        let expected = output.probability > model.config.threshold ? 1 : 0
        #expect(output.prediction == expected)
        print("SmartTurn endpoint probability=\(output.probability) "
              + "prediction=\(output.prediction)")
    }

    @Test("predictEndpoint — an explicit threshold overrides the config")
    func predictEndpointThresholdOverride() async throws {
        let model = try await loadSmartTurn()
        let clip = syntheticClip()
        // Threshold 0 forces a positive prediction; threshold 1 forces
        // a negative one — for any probability in [0, 1].
        let positive = model.predictEndpoint(audio: clip, threshold: 0.0)
        let negative = model.predictEndpoint(audio: clip, threshold: 1.0)
        #expect(positive.prediction == 1)
        #expect(negative.prediction == 0)
        // The underlying probability is threshold-independent.
        #expect(positive.probability == negative.probability)
    }

    @Test("predictEndpoint — silence yields a finite, well-formed result")
    func predictEndpointSilence() async throws {
        let model = try await loadSmartTurn()
        // A short silent clip: melFeatures left-pads it to the fixed
        // encoder length, so the forward must still produce a real
        // probability rather than NaN / Inf.
        let silence = [Float](repeating: 0, count: 16000)
        let output = model.predictEndpoint(audio: silence)
        #expect(output.probability.isFinite)
        #expect(output.probability >= 0 && output.probability <= 1)
    }
}
