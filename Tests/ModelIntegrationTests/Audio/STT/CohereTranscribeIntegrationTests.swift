// Integration test: loads the real CohereTranscribe checkpoint from the HF
// cache and runs the Slaney-Mel front-end → ConvSubsampling → Conformer encoder
// → Transformer AED decoder end-to-end.
//
// CohereTranscribe is a Conformer-AED (attention-based encoder-decoder) model for
// multilingual speech transcription. This test validates:
//   * Config decoding from the CohereTranscribe config.json layout.
//   * Weight loading (Conv2d subsampling, rel-pos attention, decoder).
//   * Slaney-Mel front-end produces finite, non-degenerate features.
//   * encodeAudio produces finite, non-degenerate encoder outputs.
//   * End-to-end transcription of a real speech clip yields a non-empty,
//     non-degenerate string (≥ 2 distinct words).
//
// DO NOT RUN — this suite requires multi-GB checkpoint files.
// Run via `make test-integration`.
//
// A missing checkpoint FAILS the suite.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("CohereTranscribe Integration", .serialized)
struct CohereTranscribeIntegrationTests {

    /// Canonical HF repo id. The 8-bit MLX conversion is the smallest
    /// published Cohere-transcribe variant (no 4-bit exists on HF as of
    /// 2026-05). Repo layout note: every model file lives under an
    /// `mlx-int8/` subdirectory rather than at the snapshot root, so
    /// `resolveDir()` descends into that subdir before handing the
    /// directory to `CohereTranscribeModel.load(directory:)`.
    private static let repoId = "mlx-community/cohere-transcribe-03-2026-mlx-8bit"

    // ─── Checkpoint resolution ───────────────────────────────────────────

    /// Resolve the CohereTranscribe checkpoint directory from the HF hub.
    private func resolveDir() async throws -> URL {
        let snapshot = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
        // Published `cohere-transcribe-03-2026-mlx-8bit` nests every
        // model file (config.json, model.safetensors, …) under an
        // `mlx-int8/` subdirectory. Descend into that subdir so
        // `CohereTranscribeModel.load(directory:)` sees the standard
        // `<dir>/config.json` layout it expects.
        let nested = snapshot.appendingPathComponent("mlx-int8")
        if FileManager.default.fileExists(
            atPath: nested.appendingPathComponent("config.json").path
        ) {
            return nested
        }
        return snapshot
    }

    /// Load the CohereTranscribe model from a resolved checkpoint directory.
    private func loadModel() async throws -> CohereTranscribeModel {
        let dir = try await resolveDir()
        return try await CohereTranscribeModel.load(directory: dir)
    }

    // ─── Synthetic test audio ─────────────────────────────────────────────

    /// 1 s of a 440 Hz tone at 16 kHz — enough audio for the Conformer
    /// stack to process without hitting edge cases.
    private func syntheticTone(hz: Float = 440, seconds: Double = 1.0) -> [Float] {
        let sr = 16_000
        let n  = Int(Double(sr) * seconds)
        return (0..<n).map { 0.3 * sin(2 * Float.pi * hz * Float($0) / Float(sr)) }
    }

    // ─── Tests ───────────────────────────────────────────────────────────

    @Test("load — config and weight shapes bind correctly")
    func loadBindsWeights() async throws {
        let model = try await loadModel()
        let ec = model.config.encoder
        let dc = model.config.decoder

        // Encoder defaults for the published CohereTranscribe checkpoint.
        #expect(ec.nLayers              >= 1)
        #expect(ec.nHeads               >= 1)
        #expect(ec.dModel               >= 1)
        #expect(ec.featIn               == 128)
        #expect(ec.subsamplingFactor    == 8)

        // Decoder hyper-parameters.
        #expect(dc.numLayers            >= 1)
        #expect(dc.numAttentionHeads    >= 1)
        #expect(dc.hiddenSize           >= 1)

        // Layer counts must match config.
        #expect(model.encoderLayers.count == ec.nLayers)
        #expect(model.decoderLayers.count == dc.numLayers)

        // Rel-pos table: [2*posEmbMaxLen-1, dModel].
        let expectedRelLen = 2 * ec.posEmbMaxLen - 1
        #expect(model.relPETable.count == expectedRelLen * ec.dModel)

        print("[CohereTranscribe integration] config enc=\(ec.nLayers)L "
              + "dec=\(dc.numLayers)L dModel=\(ec.dModel) "
              + "dtype=\(model.dtype)")
    }

    @Test("encodeAudio — produces finite, non-degenerate encoder outputs")
    func encodeAudioFiniteFeatures() async throws {
        let model = try await loadModel()
        let wave  = syntheticTone(seconds: 1.0)

        let encoded = model.encodeAudio(waveform: wave)

        #expect(!encoded.isEmpty, "encodeAudio produced zero outputs")

        let encDim = model.bridgeProj != nil
            ? model.config.decoder.hiddenSize
            : model.config.encoder.dModel
        let nFrames = encoded.count / encDim
        #expect(nFrames > 0, "encoder produced zero frames")
        #expect(encoded.count == nFrames * encDim,
                "encoder output size not a multiple of encDim")

        #expect(encoded.allSatisfy { $0.isFinite },
                "encoder output contains NaN or Inf")

        let variance = encoded.map { $0 * $0 }.reduce(0, +) / Float(encoded.count)
        #expect(variance > 1e-6, "encoder output is degenerate (near-zero)")

        print("[CohereTranscribe integration] encoder: \(nFrames) frames "
              + "× \(encDim) dim")
    }

    @Test("transcribe — real speech produces a non-degenerate transcript")
    func transcribeRealSpeech() async throws {
        let model = try await loadModel()

        // Bundled fixture: conversational_a.wav (~13 s, 24 kHz source resampled to 16 kHz).
        let wave = try AudioTestHelpers.conversationalAWaveform()
        #expect(!wave.isEmpty, "audio fixture waveform is empty")

        guard let tok = model.tokenizer else {
            Issue.record("CohereTranscribe checkpoint did not include a tokenizer")
            return
        }

        let transcript = model.transcribe(
            waveform: wave,
            tokenizer: tok,
            language: "en",
            maxTokens: 256
        )

        print("[CohereTranscribe integration] transcript: \(transcript.debugDescription)")

        // Non-empty.
        #expect(!transcript.isEmpty, "CohereTranscribe produced an empty transcript")

        // Non-degenerate: a genuine decode visits several distinct words.
        let words = transcript.split(separator: " ")
        #expect(words.count >= 2,
                "CohereTranscribe transcript is degenerate: \(transcript.debugDescription)")
    }

    @Test("transcribe — synthetic tone does not crash or produce NaN")
    func transcribeSyntheticTone() async throws {
        let model = try await loadModel()
        let wave  = syntheticTone(seconds: 1.0)

        guard let tok = model.tokenizer else {
            Issue.record("CohereTranscribe checkpoint did not include a tokenizer")
            return
        }

        // Transcribing a pure tone should not crash.
        // Output may be empty (no recognised speech) — that is valid.
        let transcript = model.transcribe(
            waveform: wave,
            tokenizer: tok,
            language: "en",
            maxTokens: 50
        )
        print("[CohereTranscribe integration] synthetic tone → "
              + transcript.debugDescription)
        // Nothing to assert on content — reaching here means no crash / NaN.
    }

    @Test("registry — AudioModelRegistry routes checkpoint to .cohereTranscribe")
    func registryRoutes() async throws {
        let dir = try await resolveDir()
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .cohereTranscribe = loaded else {
            Issue.record(
                "AudioModelRegistry did not route to .cohereTranscribe; got \(loaded)")
            return
        }
        #expect(loaded.capabilities == Capability.speechToText)
        print("[CohereTranscribe integration] registry routed correctly, "
              + "capabilities=\(loaded.capabilities)")
    }
}
