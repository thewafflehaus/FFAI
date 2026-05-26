// Integration test: loads the real parakeet-tdt-0.6b-v2 checkpoint
// from the HF cache and runs the audio front-end + Conformer encoder +
// greedy TDT decoder end-to-end. A load failure FAILS the suite —
// `loadParakeet()` is `throws` and the checkpoint is a hard requirement.
//
// parakeet-tdt-0.6b-v2 uses a 1024-token BPE vocabulary. The Conformer
// encoder + dual-LSTM prediction network are shared with the v3
// variant; v2 is the canonical published smallest variant.
//
// DO NOT RUN — this suite requires multi-GB checkpoint files and a GPU.
// Run via `make test-integration`.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Parakeet Integration", .serialized)
struct ParakeetIntegrationTests {

    /// Canonical HF repo id. No 4-bit MLX conversion exists at time of
    /// writing for the parakeet-tdt family.
    private static let repoId = "mlx-community/parakeet-tdt-0.6b-v2"

    // ─── Checkpoint resolution ───────────────────────────────────────

    /// Resolve the Parakeet checkpoint directory through `ModelLocator`,
    /// serialised against other model loads via `ModelLoadLock`.
    private func resolveParakeetDir() async throws -> URL {
        try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
    }

    /// Load a Parakeet model, resolving the checkpoint directory first.
    private func loadParakeet() async throws -> ParakeetModel {
        let dir = try await resolveParakeetDir()
        return try ParakeetModel.load(directory: dir)
    }

    // ─── Synthetic test audio ─────────────────────────────────────────

    /// 1 s of a 440 Hz tone at 16 kHz — enough audio for the Conformer
    /// stack to process without hitting edge cases.
    private func syntheticTone(hz: Float = 440, seconds: Double = 1.0) -> [Float] {
        let sr = 16_000
        let n = Int(Double(sr) * seconds)
        return (0..<n).map { 0.3 * sin(2 * Float.pi * hz * Float($0) / Float(sr)) }
    }

    /// Load the bundled conversational speech fixture, if present.
    /// Falls back to a synthetic tone when the fixture file is absent
    /// (the worktree does not include the Resources/ subtree). Routes
    /// through `AudioPreprocessing.loadWaveform` so the 24 kHz source
    /// is resampled to 16 kHz for Parakeet's front-end.
    private func speechFixtureWaveform() -> [Float] {
        if let wave = try? AudioTestHelpers.conversationalAWaveform(),
           !wave.isEmpty {
            return wave
        }
        return syntheticTone()
    }

    // ─── Tests ───────────────────────────────────────────────────────

    @Test("load — Parakeet config + weights bind correctly")
    func load_bindsWeights() async throws {
        let model = try await loadParakeet()

        // Conformer encoder geometry
        #expect(model.config.encoder.nLayers == 24)
        #expect(model.config.encoder.dModel == 1024)
        #expect(model.config.encoder.nHeads == 8)
        #expect(model.config.encoder.subsamplingFactor == 8)
        #expect(model.config.encoder.convKernelSize == 9)

        // Preprocessor
        #expect(model.config.preprocessor.sampleRate == 16_000)
        #expect(model.config.preprocessor.nMels == 128)

        // Prediction network
        #expect(model.config.predNet.predRnnLayers == 2)
        #expect(model.config.predNet.predHidden == 640)

        // TDT durations
        #expect(model.config.tdtDurations == [0, 1, 2, 3, 4])

        // Vocabulary present
        #expect(model.config.vocabulary.count > 0)

        // Block count matches config
        #expect(model.blocks.count == model.config.encoder.nLayers)

        // LSTM layers match config
        #expect(model.lstmLayers.count == model.config.predNet.predRnnLayers)

        print("Parakeet loaded: vocab=\(model.config.vocabulary.count) "
              + "blankId=\(model.config.blankTokenId)")
    }

    @Test("frontend — NeMo Mel features are finite and correctly shaped")
    func frontEnd_finiteFeatures() async throws {
        let model = try await loadParakeet()
        let wave = syntheticTone(seconds: 1.0)
        let cfg = model.config.preprocessor
        let mel = ParakeetFrontEnd.logMelFeatures(waveform: wave, cfg: cfg)

        let nFrames = mel.count / cfg.nMels
        #expect(nFrames > 0, "front-end produced zero frames")
        #expect(mel.count == nFrames * cfg.nMels)
        #expect(mel.allSatisfy { $0.isFinite },
                "Parakeet front-end produced non-finite mel values")

        // Per-feature normalisation: each Mel bin should have near-zero mean.
        let nMels = cfg.nMels
        var maxAbsMean: Float = 0
        for m in 0..<nMels {
            var sum: Float = 0
            for t in 0..<nFrames { sum += mel[t * nMels + m] }
            let mean = abs(sum / Float(nFrames))
            if mean > maxAbsMean { maxAbsMean = mean }
        }
        #expect(maxAbsMean < 0.5,
                "Per-feature normalisation left large bias: maxAbsMean=\(maxAbsMean)")
        print("Parakeet front-end: \(nFrames) frames × \(nMels) Mel bins")
    }

    @Test("transcribe — synthetic tone produces a finite, non-crashing decode")
    func transcribe_syntheticTone() async throws {
        // Transcribing a pure tone should not crash or produce NaN.
        // The output may be empty (no speech detected) — that is valid.
        let model = try await loadParakeet()
        let wave = syntheticTone(seconds: 1.0)
        let tokens = model.transcribeTokens(waveform: wave)
        #expect(tokens.allSatisfy { $0 >= 0 && $0 < model.config.blankTokenId },
                "Transcribed token out of vocabulary range")
        print("Parakeet: synthetic tone decoded to \(tokens.count) tokens")
    }

    @Test("transcribe — real speech decodes to a non-degenerate token stream")
    func transcribe_realSpeech() async throws {
        let model = try await loadParakeet()
        let wave = speechFixtureWaveform()
        #expect(!wave.isEmpty, "speech fixture waveform is empty")

        let tokens = model.transcribeTokens(waveform: wave)
        // A real utterance must produce a non-empty, in-vocab token stream.
        #expect(!tokens.isEmpty,
                "Parakeet produced no tokens for real speech")
        #expect(tokens.allSatisfy { $0 >= 0 && $0 < model.config.blankTokenId },
                "Parakeet token out of vocabulary range")

        // Non-degenerate: a genuine decode visits several distinct ids.
        let distinct = Set(tokens).count
        #expect(distinct > 1,
                "Parakeet transcript is a single repeated token (degenerate decode)")

        let text = model.transcribe(waveform: wave)
        print("Parakeet transcribed \(tokens.count) tokens "
              + "(\(distinct) distinct) → \"\(text)\"")
    }

    @Test("registry — Parakeet routes through AudioModelRegistry")
    func registry_routesParakeet() async throws {
        let dir = try await resolveParakeetDir()
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .parakeet = loaded else {
            Issue.record("AudioModelRegistry did not route to Parakeet")
            return
        }
        #expect(loaded.capabilities.contains(.audioIn))
        #expect(loaded.capabilities.contains(.textOut))
    }
}
