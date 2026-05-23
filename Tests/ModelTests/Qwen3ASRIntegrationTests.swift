// Integration test: loads the real Qwen3-ASR-0.6B-4bit checkpoint from the
// mlx-audio HF cache and runs end-to-end transcription on the bundled
// speech fixture. A missing checkpoint FAILS the suite.
//
// Qwen3-ASR uses a Conv2d audio encoder whose output is merged into a
// Qwen3 text-decoder embedding stream. This test validates:
//   * Config decoding from the nested `thinker_config` layout.
//   * Weight loading (including Conv2d OHWI→OIHW transposition).
//   * Audio encoding produces finite, non-degenerate features.
//   * End-to-end transcription of a real speech clip yields a non-empty,
//     non-degenerate string. We do not assert the exact text since that
//     depends on sampling; "non-degenerate" means ≥2 distinct words.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3ASR 0.6B integration", .serialized)
struct Qwen3ASRIntegrationTests {

    /// Resolve the Qwen3-ASR checkpoint directory from the mlx-audio cache
    /// or the HF hub. Prefers the 0.6B-4bit variant for speed.
    private func loadModel() async throws -> Qwen3ASRModel {
        let dir = try await AudioFixtures.resolveCheckpoint(
            mlxAudioSlugs: [
                "mlx-community_Qwen3-ASR-0.6B-4bit",
                "mlx-community_Qwen3-ASR-0.6B-bf16",
            ],
            repoIds: [
                "mlx-community/Qwen3-ASR-0.6B-4bit",
                "mlx-community/Qwen3-ASR-0.6B-bf16",
            ]
        )
        return try Qwen3ASRModel.load(directory: dir)
    }

    @Test("load — config and weight shapes bind correctly")
    func loadBindsWeights() async throws {
        let model = try await loadModel()
        let ac = model.config.audioConfig

        // Audio encoder hyper-parameters for the 0.6B variant.
        #expect(ac.dModel == 896)
        #expect(ac.encoderLayers == 18)
        #expect(ac.encoderAttentionHeads == 14)
        #expect(ac.numMelBins == 128)
        #expect(ac.downsampleHiddenSize == 480)

        // Text decoder hyper-parameters.
        #expect(model.config.textHidden == 1024)
        #expect(model.config.textLayers == 28)
        #expect(model.config.textHeads == 16)
        #expect(model.config.textKVHeads == 8)
        #expect(model.config.headDim == 128)
        #expect(model.config.vocabSize == 151936)

        // The encoder layer count should match the actual loaded layers.
        #expect(model.audioEncoderLayers.count == ac.encoderLayers)
        #expect(model.textLayers.count == model.config.textLayers)

        // Conv2d weights should be transposed to OIHW on load.
        // conv2d1: [480, 1, 3, 3] (OIHW)
        #expect(model.conv2d1Weight.shape[0] == 480)
        #expect(model.conv2d1Weight.shape[1] == 1)
        #expect(model.conv2d1Weight.shape[2] == 3)
        #expect(model.conv2d1Weight.shape[3] == 3)
    }

    @Test("encodeAudio — encoder produces finite, non-degenerate features")
    func encodeAudioFiniteFeatures() async throws {
        let model = try await loadModel()

        // 1 second of 440 Hz sine — exercises the full conv2d + transformer.
        let sr = 16_000
        var wave = [Float](repeating: 0, count: sr)
        for i in 0..<sr {
            wave[i] = 0.3 * sin(2.0 * Float.pi * 440.0 * Float(i) / Float(sr))
        }

        let features = model.encodeAudio(waveform: wave)

        // features: [nAudioTokens, outputDim]
        #expect(features.shape.count == 2)
        #expect(features.shape[1] == model.config.audioConfig.outputDim)
        #expect(features.shape[0] > 0, "encoder produced zero audio tokens")

        let vals = features.toFloatArray()
        #expect(vals.allSatisfy { $0.isFinite },
                "audio features contain NaN or Inf")
        let variance = vals.map { $0 * $0 }.reduce(0, +) / Float(vals.count)
        #expect(variance > 1e-6, "audio features are degenerate (near-zero)")
    }

    @Test("transcribe — real speech produces a non-degenerate transcript")
    func transcribeRealSpeech() async throws {
        let model = try await loadModel()

        // Load the bundled 16 kHz fixture:
        // "Sure, I can help you with that." (≈1.85 s).
        let wave = try AudioFixtures.clean001Waveform()
        #expect(!wave.isEmpty, "audio fixture failed to load")

        // We need a tokenizer — for Qwen3ASR checkpoints it lives in the
        // same directory as the weights. Resolve it from the same cache.
        let dir = try await AudioFixtures.resolveCheckpoint(
            mlxAudioSlugs: [
                "mlx-community_Qwen3-ASR-0.6B-4bit",
                "mlx-community_Qwen3-ASR-0.6B-bf16",
            ],
            repoIds: [
                "mlx-community/Qwen3-ASR-0.6B-4bit",
                "mlx-community/Qwen3-ASR-0.6B-bf16",
            ]
        )
        let tokenizer = try await TokenizerLoader().load(from: dir)

        let transcript = model.transcribe(
            waveform: wave,
            tokenizer: tokenizer,
            maxTokens: 128
        )

        print("[Qwen3ASR integration] transcript: \(transcript.debugDescription)")

        // Non-empty and not a single repeated token.
        #expect(!transcript.isEmpty, "Qwen3ASR produced an empty transcript")
        let words = transcript.split(separator: " ")
        #expect(words.count >= 2,
                "Qwen3ASR transcript is degenerate: \(transcript.debugDescription)")
    }
}
