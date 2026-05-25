// Integration tests for LFMAudio — LiquidAI's LFM2.5-Audio speech-to-speech
// / omni family. Loads the real checkpoint from the HF cache and exercises
// the audio-encoding path that produces feature tokens for the LFM2 backbone.
//
// Checkpoint resolution order (first complete snapshot wins):
//   1. mlx-audio flat cache:  mlx-community_LFM2.5-Audio-1.5B-6bit
//   2. mlx-audio flat cache:  mlx-community_LFM2.5-Audio-1.5B-bf16
//   3. HF hub (download):     mlx-community/LFM2.5-Audio-1.5B-4bit
//
// A missing checkpoint FAILS the test — there is no "skip if absent" logic.
// The integration suite is serialized to avoid pinning the GPU with multiple
// concurrent model loads.
//
// DO NOT run this file with `swift test`; use `make test-integration`.

import Foundation
import Testing
@testable import FFAI

@Suite("LFMAudio integration", .serialized)
struct LFMAudioIntegrationTests {

    // ── Checkpoint resolution ────────────────────────────────────────────

    /// Load the LFMAudio model from the HF cache / network. Throws on failure
    /// so a missing checkpoint surfaces as a test failure, not a silent pass.
    private func loadLFMAudio() async throws -> LFMAudioModel {
        let dir = try await AudioFixtures.resolveCheckpoint(
            mlxAudioSlugs: [
                "mlx-community_LFM2.5-Audio-1.5B-6bit",
                "mlx-community_LFM2.5-Audio-1.5B-bf16"
            ],
            repoIds: [
                "mlx-community/LFM2.5-Audio-1.5B-4bit"
            ])
        return try await ModelLoadLock.shared.loadSerially {
            try LFMAudioModel.load(directory: dir)
        }
    }

    // ── Load path ────────────────────────────────────────────────────────

    @Test("load — conformer + adapter + backbone weights bind correctly")
    func load_weightsBindCorrectly() async throws {
        let model = try await loadLFMAudio()

        // ConformerEncoder block count must match config.
        #expect(model.blocks.count == model.config.encoder.nLayers)

        // Subsampling out-projection shape: [dModel, convCh * (featIn/8)]
        let dModel = model.config.encoder.dModel
        let convCh = model.config.encoder.subsamplingConvChannels
        let featIn = model.config.encoder.featIn
        let expectedOutW = dModel * convCh * (featIn / 8)
        #expect(model.subsampling.outW.count == expectedOutW)

        // AdapterMLP must have at least one linear layer.
        #expect(!model.adapter.linearWs.isEmpty)

        // LFM2 backbone must have layers.
        #expect(model.lfm.layers.count > 0)

        // Config reports the expected omni-audio capabilities via the registry.
        let cfgHandle = LFMAudioModel.handles(
            ModelConfig(architecture: "Lfm2AudioForConditionalGeneration",
                        modelType: "lfm_audio",
                        raw: [:]))
        #expect(cfgHandle)
    }

    // ── Audio encoding path ──────────────────────────────────────────────

    @Test("encodeAudio — synthetic 350 Hz tone produces finite feature tokens")
    func encodeAudio_syntheticTone() async throws {
        let model = try await loadLFMAudio()
        let sr = model.config.preprocessor.sampleRate   // 16 000 Hz

        // 1 s 350 Hz pure tone at 16 kHz.
        var wave = [Float](repeating: 0, count: sr)
        for i in 0..<sr {
            wave[i] = 0.3 * sin(2.0 * Float.pi * 350.0 * Float(i) / Float(sr))
        }

        let features = model.encodeAudio(waveform: wave)

        // Shape: [nTokens, lfmHidden]
        #expect(features.shape.count == 2)
        #expect(features.shape[1] == model.config.lfmHidden,
                "second dim must equal lfmHidden \(model.config.lfmHidden)")
        #expect(features.shape[0] > 0,
                "LFMAudio produced zero tokens for a 1-s tone")

        // All values finite.
        let vals = features.toFloatArray()
        #expect(vals.allSatisfy { $0.isFinite },
                "LFMAudio audio features contain NaN / Inf")

        // Non-degenerate (non-zero variance).
        let mean = vals.reduce(0, +) / Float(vals.count)
        let variance = vals.map { ($0 - mean) * ($0 - mean) }
            .reduce(0, +) / Float(vals.count)
        #expect(variance > 1e-8,
                "LFMAudio audio features are degenerate (variance=\(variance))")

        print("LFMAudio encoded 1-s tone → \(features.shape[0]) tokens "
              + "× \(features.shape[1]) dims, variance=\(variance)")
    }

    @Test("encodeAudio — real speech produces non-degenerate feature tokens")
    func encodeAudio_realSpeech() async throws {
        let model = try await loadLFMAudio()
        // Bundled conversational speech fixture (~13 s, 24 kHz → 16 kHz).
        let wave = try AudioFixtures.conversationalAWaveform()
        #expect(!wave.isEmpty, "AudioFixtures.conversationalAWaveform() returned an empty array")

        let features = model.encodeAudio(waveform: wave)

        #expect(features.shape.count == 2)
        #expect(features.shape[1] == model.config.lfmHidden)
        #expect(features.shape[0] > 0,
                "LFMAudio produced no tokens for real speech")

        let vals = features.toFloatArray()
        #expect(vals.allSatisfy { $0.isFinite })

        let mean = vals.reduce(0, +) / Float(vals.count)
        let variance = vals.map { ($0 - mean) * ($0 - mean) }
            .reduce(0, +) / Float(vals.count)
        #expect(variance > 1e-6,
                "LFMAudio real-speech features are degenerate (variance=\(variance))")

        print("LFMAudio encoded real speech → \(features.shape[0]) tokens "
              + "× \(features.shape[1]) dims, variance=\(variance)")
    }

    @Test("encodeAudio — short clip does not crash or return wrong shape")
    func encodeAudio_shortClip() async throws {
        let model = try await loadLFMAudio()
        let sr = model.config.preprocessor.sampleRate

        // 100 ms — just 1600 samples; the subsampling may produce very few
        // tokens, but must not crash or return a malformed tensor.
        let count = sr / 10
        var wave = [Float](repeating: 0, count: count)
        for i in 0..<count {
            wave[i] = 0.1 * sin(2.0 * Float.pi * 200.0 * Float(i) / Float(sr))
        }

        let features = model.encodeAudio(waveform: wave)
        // Shape must be valid (may have 0 tokens for very short input).
        #expect(features.shape.count == 2)
        #expect(features.shape[1] == model.config.lfmHidden)

        if features.shape[0] > 0 {
            let vals = features.toFloatArray()
            #expect(vals.allSatisfy { $0.isFinite })
        }

        print("LFMAudio encoded 100-ms clip → \(features.shape[0]) tokens")
    }

    // ── Registry load path ───────────────────────────────────────────────

    @Test("AudioModelRegistry.load — returns .lfmAudio case for LFMAudio checkpoint")
    func registryLoad_returnsLFMAudioCase() async throws {
        let dir = try await AudioFixtures.resolveCheckpoint(
            mlxAudioSlugs: [
                "mlx-community_LFM2.5-Audio-1.5B-6bit",
                "mlx-community_LFM2.5-Audio-1.5B-bf16"
            ],
            repoIds: [
                "mlx-community/LFM2.5-Audio-1.5B-4bit"
            ])
        let loaded = try await ModelLoadLock.shared.loadSerially {
            try await AudioModelRegistry.load(directory: dir)
        }
        guard case .lfmAudio(let model) = loaded else {
            Issue.record("Expected .lfmAudio but got \(loaded)")
            return
        }
        #expect(loaded.capabilities == Capability.omniAudio)
        #expect(model.blocks.count == model.config.encoder.nLayers)
        print("AudioModelRegistry correctly loaded LFMAudio (\(model.config.modelType))")
    }
}
