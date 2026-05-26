// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Integration test: loads a real Kokoro-82M checkpoint from the HF
// cache and exercises the GPU iSTFTNet vocoder tail. A load failure
// FAILS the suite — `loadKokoro()` is `throws` and the checkpoint is a
// hard requirement, not a "skip if missing".
//
// Kokoro-82M is the published checkpoint; FFAI's Phase 7 contribution
// is the iSTFTNet vocoder (Ops.vocoderISTFT). This suite verifies the
// model loads, the vocoder constructs from the checkpoint config, and a
// synthesized waveform is non-degenerate (finite, non-silent).
//
// KokoroModel.load is config-driven — the vocoder geometry comes from
// `config.json`'s `istftnet` block and no safetensors weights are read
// (the StyleTTS2 acoustic stack is a separate port). The 4-bit MLX
// conversion is the smallest published Kokoro variant and carries the
// canonical config.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Kokoro Integration", .serialized)
struct KokoroIntegrationTests {

    /// Canonical HF repo id. The 4-bit MLX conversion is the smallest
    /// published Kokoro variant.
    private static let repoId = "mlx-community/Kokoro-82M-4bit"

    /// Load Kokoro from the HF cache. Throws on failure so a missing
    /// checkpoint fails the test instead of skipping it.
    private func loadKokoro() async throws -> KokoroModel {
        let dir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
        return try KokoroModel.load(directory: dir)
    }

    @Test("load — Kokoro config binds the iSTFTNet vocoder")
    func loadKokoro_bindsVocoder() async throws {
        let model = try await loadKokoro()
        // Kokoro's iSTFTNet head uses a tiny FFT (20) with hop 5.
        #expect(model.vocoder.nFFT > 0)
        #expect(model.vocoder.hopLength > 0)
        #expect(model.config.sampleRate == 24_000)
        // The phoneme vocabulary should have loaded from config.json.
        #expect(!model.phonemeVocab.isEmpty)
    }

    @Test("synthesize — vocoder produces a non-degenerate waveform")
    func synthesize_nonDegenerateWaveform() async throws {
        let model = try await loadKokoro()
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
