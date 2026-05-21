// Slow integration test: downloads (or hits cache) a Qwen-Omni
// checkpoint and exercises the audio-in path — the Whisper-style audio
// encoder that produces feature tokens in the text backbone hidden
// dim. Skipped automatically if the network or the checkpoint isn't
// available — mirrors the other ModelTests suites.
//
// Qwen2.5-Omni-3B is the smallest published Omni checkpoint. FFAI's
// Phase 7 contribution is the audio-in path; the vision path is the
// separate Qwen-VL port. This suite verifies the audio tower loads and
// encodes a waveform into finite, correctly-shaped feature tokens.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen-Omni audio-in integration", .serialized)
struct QwenOmniIntegrationTests {

    /// Load Qwen-Omni from the HF cache / network, or return nil with a
    /// printed skip reason.
    private func loadQwenOmni() async -> QwenOmniModel? {
        for repoId in ["Qwen/Qwen2.5-Omni-3B", "mlx-community/Qwen2.5-Omni-3B"] {
            do {
                let locator = ModelLocator()
                let dir = try await ModelLoadLock.shared.loadSerially {
                    try await locator.resolve(idOrPath: repoId)
                }
                return try QwenOmniModel.load(directory: dir)
            } catch {
                print("QwenOmni load from \(repoId) skipped: \(error)")
            }
        }
        return nil
    }

    @Test("load — Qwen-Omni audio tower binds correctly")
    func loadQwenOmni_bindsAudioTower() async throws {
        guard let model = await loadQwenOmni() else {
            print("QwenOmni integration test skipped: checkpoint unavailable")
            return
        }
        #expect(model.audioEncoder.config.nLayers > 0)
        #expect(model.audioEncoder.config.hidden > 0)
        #expect(model.config.textHidden > 0)
        // The audio tower's encoder block count must match the config.
        #expect(model.audioEncoder.layers.count
                == model.config.encoderLayers)
    }

    @Test("encodeAudio — produces finite feature tokens in text hidden dim")
    func encodeAudio_finiteFeatures() async throws {
        guard let model = await loadQwenOmni() else {
            print("QwenOmni integration test skipped: checkpoint unavailable")
            return
        }
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
}
