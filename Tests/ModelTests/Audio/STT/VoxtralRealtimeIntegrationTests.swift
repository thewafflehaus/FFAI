// Integration test: loads the real Voxtral-Mini-4B-Realtime checkpoint from
// the HF cache and runs end-to-end transcription on the bundled speech fixture.
// A missing checkpoint FAILS the suite.
//
// VoxtralRealtime uses:
//   * A Slaney mel front-end (fMax = 8 kHz, not Nyquist).
//   * A 32-layer sliding-window causal audio encoder with interleaved RoPE.
//   * A Mistral-style decoder with GQA and AdaRMSNorm time conditioning.
//   * A Tekken byte-level tokenizer (tekken.json).
//
// This test validates:
//   * Config parsing from the real checkpoint config.json layout.
//   * Weight loading including Conv1d NLC→NCL transposition.
//   * Audio encoding produces finite, non-degenerate features.
//   * End-to-end transcription of a speech clip yields a non-empty,
//     coherent string. Exact text is not asserted (greedy decode variance).

// !! DO NOT RUN THIS FILE — integration tests require downloaded model
// !! weights (multi-GB) and a serialized GPU. Run via:
// !!   make test-integration

import Foundation
import Testing
@testable import FFAI

@Suite("VoxtralRealtime Integration", .serialized)
struct VoxtralRealtimeIntegrationTests {

    /// Resolve the Voxtral-Mini-4B checkpoint. Prefers the 4-bit variant
    /// for speed; falls back to 6-bit and fp16.
    private func loadModel() async throws -> VoxtralRealtimeModel {
        let dir = try await AudioFixtures.resolveCheckpoint(
            mlxAudioSlugs: [
                "mlx-community_Voxtral-Mini-4B-Realtime-2602-4bit",
                "mlx-community_Voxtral-Mini-4B-Realtime-6bit",
                "mlx-community_Voxtral-Mini-4B-Realtime-2602-fp16",
            ],
            repoIds: [
                "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
                "mlx-community/Voxtral-Mini-4B-Realtime-6bit",
                "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16",
            ]
        )
        return try VoxtralRealtimeModel.load(directory: dir)
    }

    @Test("load — config and weight shapes bind correctly")
    func loadBindsWeights() async throws {
        let model = try await loadModel()
        let ec = model.config.encoderConfig
        let dc = model.config.decoderConfig
        let ac = model.config.audioConfig

        // Published Mini-4B encoder hyper-parameters.
        #expect(ec.dim == 1280)
        #expect(ec.nLayers == 32)
        #expect(ec.nHeads == 32)
        #expect(ec.headDim == 64)
        #expect(ec.slidingWindow == 750)
        #expect(ec.downsampleFactor == 4)

        // Published Mini-4B decoder hyper-parameters.
        #expect(dc.dim == 3072)
        #expect(dc.nLayers == 26)
        #expect(dc.nHeads == 32)
        #expect(dc.nKVHeads == 8)
        #expect(dc.headDim == 128)
        #expect(dc.vocabSize == 131072)
        #expect(dc.tiedEmbeddings == true)
        #expect(dc.adaRmsNormTCond == true)

        // Audio front-end.
        #expect(ac.samplingRate == 16_000)
        #expect(ac.numMelBins == 128)
        #expect(ac.hopLength == 160)
        #expect(ac.windowSize == 400)

        // Verify layer counts match the loaded arrays.
        #expect(model.encoderLayers.count == ec.nLayers)
        #expect(model.decoderLayers.count == dc.nLayers)

        // Conv weights must be transposed to NCL [outCh, inCh, k] on load.
        // conv0: [1280, 128, 3] — out=1280, in=128, k=3.
        #expect(model.conv0Weight.shape == [1280, 128, 3])
        // conv1: [1280, 1280, 3] — out=1280, in=1280, k=3.
        #expect(model.conv1Weight.shape == [1280, 1280, 3])
    }

    @Test("encodeAudio — encoder produces finite, non-degenerate features")
    func encodeAudioFiniteFeatures() async throws {
        let model = try await loadModel()
        let dc = model.config.decoderConfig

        // 1 second of 440 Hz sine — exercises the full conv stem + transformer.
        let sr = 16_000
        var wave = [Float](repeating: 0, count: sr)
        for i in 0..<sr {
            wave[i] = 0.3 * sin(2.0 * Float.pi * 440.0 * Float(i) / Float(sr))
        }

        let features = model.encodeAudio(waveform: wave)

        // Features: [nAudioTokens, decoderDim].
        #expect(features.shape.count == 2)
        #expect(features.shape[1] == dc.dim)
        #expect(features.shape[0] > 0, "encoder produced zero audio tokens")

        let vals = features.toFloatArray()
        #expect(vals.allSatisfy { $0.isFinite },
                "audio features contain NaN or Inf")
        let variance = vals.map { $0 * $0 }.reduce(0, +) / Float(vals.count)
        #expect(variance > 1e-8, "audio features are degenerate (near-zero)")
    }

    @Test("transcribe — real speech produces a non-degenerate transcript")
    func transcribeRealSpeech() async throws {
        let model = try await loadModel()

        // Load the bundled conversational speech fixture (~13 s, 24 kHz resampled to 16 kHz).
        let wave = try AudioFixtures.conversationalAWaveform()
        #expect(!wave.isEmpty, "audio fixture failed to load")

        let transcript = model.transcribe(waveform: wave, maxTokens: 256)

        print("[VoxtralRealtime integration] transcript: \(transcript.debugDescription)")

        // Non-empty and at least two words (non-degenerate).
        #expect(!transcript.isEmpty,
                "VoxtralRealtime produced an empty transcript")
        let words = transcript.split(separator: " ")
        #expect(words.count >= 2,
                "VoxtralRealtime transcript is degenerate: \(transcript.debugDescription)")
    }
}
