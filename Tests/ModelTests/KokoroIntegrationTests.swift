// Slow integration test: downloads (or hits cache) a Kokoro TTS
// checkpoint and exercises the GPU iSTFTNet vocoder tail. Skipped
// automatically if the network or the checkpoint isn't available —
// mirrors the other ModelTests suites.
//
// Kokoro-82M is the published checkpoint; FFAI's Phase 7 contribution
// is the iSTFTNet vocoder (Ops.vocoderISTFT). This suite verifies the
// model loads, the vocoder constructs from the checkpoint config, and
// a synthesized waveform is non-degenerate (finite, non-silent).

import Foundation
import Testing
@testable import FFAI

@Suite("Kokoro TTS integration", .serialized)
struct KokoroIntegrationTests {

    /// Load Kokoro from the HF cache / network, or return nil with a
    /// printed skip reason.
    private func loadKokoro() async -> KokoroModel? {
        // Try the common published Kokoro repo ids in order.
        for repoId in ["hexgrad/Kokoro-82M", "prince-canuma/Kokoro-82M"] {
            do {
                let locator = ModelLocator()
                let dir = try await ModelLoadLock.shared.loadSerially {
                    try await locator.resolve(idOrPath: repoId)
                }
                return try KokoroModel.load(directory: dir)
            } catch {
                print("Kokoro load from \(repoId) skipped: \(error)")
            }
        }
        return nil
    }

    @Test("load — Kokoro config binds the iSTFTNet vocoder")
    func loadKokoro_bindsVocoder() async throws {
        guard let model = await loadKokoro() else {
            print("Kokoro integration test skipped: checkpoint unavailable")
            return
        }
        // Kokoro's iSTFTNet head uses a tiny FFT (20) with hop 5.
        #expect(model.vocoder.nFFT > 0)
        #expect(model.vocoder.hopLength > 0)
        #expect(model.config.sampleRate == 24_000)
        // The phoneme vocabulary should have loaded from config.json.
        #expect(!model.phonemeVocab.isEmpty)
    }

    @Test("synthesize — vocoder produces a non-degenerate waveform")
    func synthesize_nonDegenerateWaveform() async throws {
        guard let model = await loadKokoro() else {
            print("Kokoro integration test skipped: checkpoint unavailable")
            return
        }
        // A predicted complex spectrogram (the acoustic decoder's
        // output) — frequency-sweep content so the reconstruction is
        // a real, non-constant utterance-length waveform.
        let nFrames = 200   // ~ a short utterance at hop 5
        let nFreq = model.vocoder.nFFT / 2 + 1
        var re = [Float](repeating: 0, count: nFrames * nFreq)
        var im = [Float](repeating: 0, count: nFrames * nFreq)
        for f in 0..<nFrames {
            for k in 0..<nFreq {
                let phase = 2.0 * Float.pi * Float(k) * Float(f) / Float(nFrames)
                re[f * nFreq + k] = 0.4 * cos(phase)
                im[f * nFreq + k] = 0.4 * sin(phase)
            }
        }
        let reT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
        reT.copyIn(from: re)
        let imT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
        imT.copyIn(from: im)

        let waveform = model.synthesizeFromSpectrogram(specRe: reT, specIm: imT)
        let expectedLen = (nFrames - 1) * model.vocoder.hopLength
            + model.vocoder.nFFT
        #expect(waveform.shape == [expectedLen])

        let samples = waveform.toFloatArray()
        // Non-degenerate: finite, non-silent, not a constant.
        #expect(samples.allSatisfy { $0.isFinite })
        let energy = samples.map { $0 * $0 }.reduce(0, +)
        #expect(energy > 1e-4, "Kokoro vocoder produced a silent waveform")
        let distinct = Set(samples.map { ($0 * 1000).rounded() }).count
        #expect(distinct > 10, "Kokoro vocoder produced a constant waveform")
        print("Kokoro synthesized \(expectedLen) samples, "
              + "energy=\(energy), distinct=\(distinct)")
    }
}
