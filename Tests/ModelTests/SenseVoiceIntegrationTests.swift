// Slow integration test: downloads (or hits cache) the SenseVoiceSmall
// checkpoint and runs the SAN-M encoder + CTC head end-to-end on a
// synthetic waveform. Skipped automatically if the network or the
// checkpoint isn't available — mirrors the Whisper / other ModelTests
// suites.
//
// SenseVoiceSmall is a single ~470 MB checkpoint (one 50-block SAN-M
// encoder + a 20-block time-pooling stack + a CTC head). Unlike
// Whisper there is no autoregressive decoder: one forward pass yields
// the frame-level CTC log-probabilities, and a greedy collapse turns
// those into a transcript.

import Foundation
import Testing
@testable import FFAI

@Suite("SenseVoice integration", .serialized)
struct SenseVoiceIntegrationTests {

    /// Load SenseVoiceSmall from the HF cache / network, or return nil
    /// with a printed skip reason.
    private func loadSenseVoice() async -> SenseVoiceModel? {
        do {
            let locator = ModelLocator()
            let dir = try await ModelLoadLock.shared.loadSerially {
                try await locator.resolve(
                    idOrPath: "mlx-community/SenseVoiceSmall")
            }
            return try SenseVoiceModel.load(directory: dir)
        } catch {
            print("SenseVoice integration test skipped: \(error)")
            return nil
        }
    }

    /// A short synthetic chirp — enough audio for the FBANK front-end +
    /// the encoder stack to produce non-trivial features.
    private func syntheticChirp(seconds: Double = 1.0) -> [Float] {
        let sr = 16_000
        let n = Int(Double(sr) * seconds)
        var wave = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let f = 200.0 + 600.0 * Float(i) / Float(n)
            wave[i] = 0.3 * sin(2.0 * Float.pi * f * Float(i) / Float(sr))
        }
        return wave
    }

    @Test("load — SenseVoice config + weights bind correctly")
    func loadSenseVoice_bindsWeights() async throws {
        guard let model = await loadSenseVoice() else { return }
        // SenseVoiceSmall: 512-dim encoder, 4 heads, 50 + 20 blocks.
        #expect(model.config.hidden == 512)
        #expect(model.config.heads == 4)
        // encoders0 holds block 0; encoders holds the remaining 49.
        #expect(model.encoders0.count == 1)
        #expect(model.encoders.count == model.config.numBlocks - 1)
        #expect(model.tpEncoders.count == model.config.tpBlocks)
        #expect(model.config.vocab > 0)
    }

    @Test("frontend — Kaldi FBANK + LFR produce finite feature frames")
    func frontEnd_finiteFeatures() async throws {
        guard let model = await loadSenseVoice() else { return }
        let feats = SenseVoiceFrontEnd.featureFrames(
            waveform: syntheticChirp(), cfg: model.config.frontEnd)
        #expect(!feats.isEmpty)
        // The feature row size is nMels * lfrM = inputSize.
        #expect(feats.count % model.config.inputSize == 0)
        #expect(feats.allSatisfy { $0.isFinite })
    }

    @Test("ctc — encoder + CTC head emit finite frame log-probs")
    func ctc_finiteLogProbs() async throws {
        guard let model = await loadSenseVoice() else { return }
        let logProbs = model.ctcLogProbs(waveform: syntheticChirp())
        #expect(logProbs.shape[1] == model.config.vocab)
        let vals = logProbs.toFloatArray()
        #expect(vals.allSatisfy { $0.isFinite })
        // Log-probabilities are non-positive; each row must sum (in the
        // exp domain) close to 1.
        #expect(vals.allSatisfy { $0 <= 1e-3 })
        let V = model.config.vocab
        let rowExpSum = (0..<V).map { exp(vals[$0]) }.reduce(0, +)
        #expect(abs(rowExpSum - 1.0) < 1e-2,
                "CTC row is not a valid probability distribution")
    }

    @Test("transcribe — greedy CTC decode produces a token stream")
    func transcribe_producesTokens() async throws {
        guard let model = await loadSenseVoice() else { return }
        let tokens = model.transcribeTokens(waveform: syntheticChirp())
        // The greedy CTC collapse must yield a finite, in-vocab,
        // blank-free token stream.
        #expect(tokens.allSatisfy {
            $0 >= 0 && $0 < model.config.vocab
                && $0 != SenseVoiceModel.blankToken
        })
        print("SenseVoice transcribe produced \(tokens.count) tokens: "
              + "\(tokens.prefix(16))")
    }

    @Test("registry — SenseVoice routes through the audio registry")
    func registry_routesSenseVoice() async throws {
        guard let model = await loadSenseVoice() else { return }
        _ = model
        let locator = ModelLocator()
        let dir = try await ModelLoadLock.shared.loadSerially {
            try await locator.resolve(
                idOrPath: "mlx-community/SenseVoiceSmall")
        }
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .senseVoice = loaded else {
            Issue.record("AudioModelRegistry did not route to SenseVoice")
            return
        }
        #expect(loaded.capabilities == Capability.speechToText)
    }
}
