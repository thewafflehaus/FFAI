// Integration test: loads the real Qwen2.5-Omni-3B checkpoint from the
// HF cache and exercises the audio-in path — the Whisper-style audio
// encoder that produces feature tokens in the text backbone hidden dim.
// A load failure FAILS the suite — `loadQwenOmni()` is `throws` and the
// checkpoint is a hard requirement, not a "skip if missing".
//
// Qwen2.5-Omni-3B is the smallest published Omni checkpoint. FFAI's
// Phase 7 contribution is the audio-in path; the vision path is the
// separate Qwen-VL port. This suite verifies the audio tower loads and
// encodes a waveform (synthetic + real speech) into finite,
// correctly-shaped feature tokens.

import Foundation
import Testing
@testable import FFAI

@Suite("QwenOmni Integration", .serialized)
struct QwenOmniIntegrationTests {

    /// Load Qwen-Omni from the HF cache / network. Throws on failure so
    /// a missing checkpoint fails the test instead of skipping it.
    private func loadQwenOmni() async throws -> QwenOmniModel {
        let dir = try await AudioFixtures.resolveCheckpoint(
            repoIds: ["Qwen/Qwen2.5-Omni-3B"])
        return try await ModelLoadLock.shared.loadSerially {
            try QwenOmniModel.load(directory: dir)
        }
    }

    @Test("load — Qwen-Omni audio tower binds correctly")
    func loadQwenOmni_bindsAudioTower() async throws {
        let model = try await loadQwenOmni()
        #expect(model.config.encoderLayers > 0)
        #expect(model.config.encoderHidden > 0)
        #expect(model.config.textHidden > 0)
        // The audio tower's encoder block count must match the config.
        #expect(model.audioEncoder.layers.count
                == model.config.encoderLayers)
    }

    @Test("encodeAudio — produces finite feature tokens in text hidden dim")
    func encodeAudio_finiteFeatures() async throws {
        let model = try await loadQwenOmni()
        // 1 s of a 16 kHz tone — enough frames through the conv stem +
        // transformer stack to exercise the full encoder.
        let sr = 16_000
        var wave = [Float](repeating: 0, count: sr)
        for i in 0..<sr {
            wave[i] = 0.3 * sin(2.0 * Float.pi * 350.0 * Float(i) / Float(sr))
        }
        let features = model.encodeAudio(waveform: wave)
        // Audio features are projected into the text backbone hidden dim
        // so they can be spliced into a Qwen3 prompt stream.
        #expect(features.shape[1] == model.config.textHidden)
        let vals = features.toFloatArray()
        #expect(vals.allSatisfy { $0.isFinite })
        let variance = vals.map { $0 * $0 }.reduce(0, +) / Float(vals.count)
        #expect(variance > 1e-6, "QwenOmni audio features are degenerate")
        print("QwenOmni encoded audio into \(features.shape[0]) tokens "
              + "of dim \(features.shape[1])")
    }

    @Test("encodeAudio — real speech yields non-degenerate feature tokens")
    func encodeAudio_realSpeech() async throws {
        let model = try await loadQwenOmni()
        // The bundled conversational speech fixture (~13 s, 24 kHz resampled to 16 kHz).
        let wave = try AudioFixtures.conversationalAWaveform()
        #expect(!wave.isEmpty, "fixture waveform failed to load")

        let features = model.encodeAudio(waveform: wave)
        #expect(features.shape[1] == model.config.textHidden)
        #expect(features.shape[0] > 0,
                "QwenOmni produced no audio feature tokens for real speech")
        let vals = features.toFloatArray()
        #expect(vals.allSatisfy { $0.isFinite })
        // Real speech must produce features with genuine variance — a
        // flat / constant feature map is a degenerate encode.
        let mean = vals.reduce(0, +) / Float(vals.count)
        let variance = vals.map { ($0 - mean) * ($0 - mean) }
            .reduce(0, +) / Float(vals.count)
        #expect(variance > 1e-6, "QwenOmni real-speech features are degenerate")
        print("QwenOmni encoded real speech into \(features.shape[0]) "
              + "tokens, feature variance=\(variance)")
    }
}
