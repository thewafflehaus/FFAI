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
// Integration test: loads the real SenseVoiceSmall checkpoint from the
// HF cache and runs the SAN-M encoder + CTC head end-to-end. A load
// failure FAILS the suite — `loadSenseVoice()` is `throws` and the
// checkpoint is a hard requirement, not a "skip if missing".
//
// SenseVoiceSmall is a single ~470 MB checkpoint (one 50-block SAN-M
// encoder + a 20-block time-pooling stack + a CTC head). Unlike Whisper
// there is no autoregressive decoder: one forward pass yields the
// frame-level CTC log-probabilities, and a greedy collapse turns those
// into a transcript.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("SenseVoice Integration", .serialized)
struct SenseVoiceIntegrationTests {

    /// Canonical HF repo id. No 4-bit MLX conversion exists at time of
    /// writing for the SenseVoice family.
    private static let repoId = "mlx-community/SenseVoiceSmall"

    private func resolveDir() async throws -> URL {
        try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
    }

    /// Load SenseVoiceSmall from the HF cache / network. Throws on
    /// failure so a missing checkpoint fails the test.
    private func loadSenseVoice() async throws -> SenseVoiceModel {
        let dir = try await resolveDir()
        return try SenseVoiceModel.load(directory: dir)
    }

    /// A short synthetic chirp — enough audio for the FBANK front-end +
    /// the encoder stack to produce non-trivial features.
    private func syntheticChirp(seconds: Double = 1.0) -> [Float] {
        let sr = 16_000
        let n = Int(Double(sr) * seconds)
        var wave = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            let f = 200.0 + 600.0 * Float(i) / Float(n)
            wave[i] = 0.3 * sin(2.0 * Float.pi * f * Float(i) / Float(sr))
        }
        return wave
    }

    @Test("load — SenseVoice config + weights bind correctly")
    func loadSenseVoice_bindsWeights() async throws {
        let model = try await loadSenseVoice()
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
        let model = try await loadSenseVoice()
        let feats = SenseVoiceFrontEnd.featureFrames(
            waveform: syntheticChirp(), cfg: model.config.frontEnd)
        #expect(!feats.isEmpty)
        // The feature row size is nMels * lfrM = inputSize.
        #expect(feats.count % model.config.inputSize == 0)
        #expect(feats.allSatisfy { $0.isFinite })
    }

    @Test("ctc — encoder + CTC head emit finite frame log-probs")
    func ctc_finiteLogProbs() async throws {
        let model = try await loadSenseVoice()
        let logProbs = model.ctcLogProbs(waveform: syntheticChirp())
        #expect(logProbs.shape[1] == model.config.vocab)
        let vals = logProbs.toFloatArray()
        #expect(vals.allSatisfy { $0.isFinite })
        // Log-probabilities are non-positive; each row must sum (in the
        // exp domain) close to 1.
        #expect(vals.allSatisfy { $0 <= 1e-3 })
        let V = model.config.vocab
        let rowExpSum = (0 ..< V).map { exp(vals[$0]) }.reduce(0, +)
        #expect(
            abs(rowExpSum - 1.0) < 1e-2,
            "CTC row is not a valid probability distribution")
    }

    @Test("transcribe — real speech decodes to a non-degenerate token stream")
    func transcribe_realSpeech() async throws {
        let model = try await loadSenseVoice()
        // The bundled conversational speech fixture (~13 s, 24 kHz resampled to 16 kHz).
        let wave = try AudioTestHelpers.conversationalAWaveform()
        #expect(!wave.isEmpty, "fixture waveform failed to load")

        let tokens = model.transcribeTokens(waveform: wave)
        // The greedy CTC collapse must yield a non-empty, in-vocab,
        // blank-free token stream for a real utterance.
        #expect(
            !tokens.isEmpty,
            "SenseVoice produced no tokens for real speech")
        #expect(
            tokens.allSatisfy {
                $0 >= 0 && $0 < model.config.vocab
                    && $0 != SenseVoiceModel.blankToken
            })
        // Non-degenerate: a genuine CTC decode visits several distinct
        // ids, not one repeated token.
        let distinct = Set(tokens).count
        #expect(
            distinct > 1,
            "SenseVoice transcript is a single repeated token (degenerate CTC decode)")
        print(
            "SenseVoice transcribed real speech into \(tokens.count) "
                + "tokens (\(distinct) distinct): \(tokens.prefix(16))")
    }

    @Test("registry — SenseVoice routes through the audio registry")
    func registry_routesSenseVoice() async throws {
        let dir = try await resolveDir()
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .senseVoice = loaded else {
            Issue.record("AudioModelRegistry did not route to SenseVoice")
            return
        }
        #expect(loaded.capabilities == Capability.speechToText)
    }
}
