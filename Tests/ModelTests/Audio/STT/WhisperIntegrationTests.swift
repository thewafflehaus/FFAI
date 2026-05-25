// Integration test: loads the real whisper-tiny checkpoint from the HF
// cache and runs the audio encoder + decoder end-to-end. A load failure
// FAILS the suite — `loadWhisper()` is `throws` and the checkpoint is a
// hard requirement, not a "skip if missing".
//
// whisper-tiny is the smallest Whisper variant (~150 MB f32), so the
// integration suite stays fast. The architecture (AudioEncoder + a
// cross-attending text decoder) is identical for base → large-v3.
//
// WhisperModel works at the token-id level (it has no bundled
// tokenizer), so the transcription assertions check that the decoder
// emits a non-empty, in-vocab, *non-degenerate* token stream — a real
// utterance through a pre-trained decoder yields several distinct
// tokens, not one repeated id.

import Foundation
import Testing
@testable import FFAI

@Suite("Whisper Integration", .serialized)
struct WhisperIntegrationTests {

    /// Load whisper-tiny from the HF cache / network. Throws on failure
    /// so a missing checkpoint fails the test instead of skipping it.
    private func loadWhisper() async throws -> WhisperModel {
        let dir = try await AudioFixtures.resolveCheckpoint(
            repoIds: ["openai/whisper-tiny", "openai/whisper-base"])
        return try WhisperModel.load(directory: dir)
    }

    @Test("load — Whisper config + weights bind correctly")
    func loadWhisper_bindsWeights() async throws {
        let model = try await loadWhisper()
        // whisper-tiny: d_model 384, 4 enc / 4 dec layers, 6 heads.
        // whisper-base (the fallback): d_model 512, 6 / 6 layers.
        #expect(model.config.hidden > 0)
        #expect(model.config.encoderLayers > 0)
        #expect(model.config.decoderLayers > 0)
        #expect(model.config.nMels == 80)
        #expect(model.encoder.layers.count == model.config.encoderLayers)
        #expect(model.decoderLayers.count == model.config.decoderLayers)
    }

    @Test("encode — audio encoder produces finite features")
    func encodeAudio_finiteFeatures() async throws {
        let model = try await loadWhisper()
        // 1 s of a 16 kHz tone — enough frames to exercise the conv
        // stem + the transformer stack.
        let sr = 16_000
        var wave = [Float](repeating: 0, count: sr)
        for i in 0..<sr {
            wave[i] = 0.3 * sin(2.0 * Float.pi * 440.0 * Float(i) / Float(sr))
        }
        let features = model.encodeAudio(waveform: wave)
        #expect(features.shape[1] == model.config.hidden)
        let vals = features.toFloatArray()
        #expect(vals.allSatisfy { $0.isFinite })
        // Post-LayerNorm features must not be degenerate.
        let variance = vals.map { $0 * $0 }.reduce(0, +) / Float(vals.count)
        #expect(variance > 1e-6, "Whisper audio features are degenerate")
    }

    @Test("decode — decoder emits finite logits cross-attending to audio")
    func decode_finiteLogits() async throws {
        let model = try await loadWhisper()
        let sr = 16_000
        var wave = [Float](repeating: 0, count: sr / 2)
        for i in 0..<wave.count {
            wave[i] = 0.25 * sin(2.0 * Float.pi * 330.0 * Float(i) / Float(sr))
        }
        let features = model.encodeAudio(waveform: wave)

        // Whisper's decoder prompt prefix is the special tokens
        // <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>.
        // Their exact ids vary by tokenizer build; for a forward-shape
        // smoke test any in-vocab prefix exercises the same code path.
        let prefix = [50258, 50259, 50359, 50363]
            .filter { $0 < model.config.vocab }
        let tokens = prefix.isEmpty ? [0] : prefix
        let logits = model.decoderLogits(tokenIds: tokens,
                                         audioFeatures: features)
        #expect(logits.count == model.config.vocab)
        #expect(logits.allSatisfy { $0.isFinite })
        // The decoder is pre-trained — the top logit should pull ahead
        // of the noise floor (not a flat distribution).
        let sorted = logits.sorted(by: >)
        #expect(sorted[0] > sorted[min(99, sorted.count - 1)],
                "Whisper decoder produced a degenerate logit distribution")
    }

    @Test("transcribe — real speech decodes to a non-degenerate token stream")
    func transcribe_realSpeech() async throws {
        let model = try await loadWhisper()
        // The bundled conversational speech fixture (~13 s, 24 kHz resampled to 16 kHz).
        let wave = try AudioFixtures.conversationalAWaveform()
        #expect(!wave.isEmpty, "fixture waveform failed to load")

        let features = model.encodeAudio(waveform: wave)
        #expect(features.shape[1] == model.config.hidden)

        let prefix = [50258, 50259, 50359, 50363]
            .filter { $0 < model.config.vocab }
        let eos = 50257 < model.config.vocab ? 50257 : model.config.vocab
        let generated = model.generateTranscript(
            audioFeatures: features,
            initialTokens: prefix.isEmpty ? [0] : prefix,
            eosToken: eos, maxTokens: 200)

        // A real utterance must produce a non-empty, in-vocab stream.
        #expect(!generated.isEmpty,
                "Whisper produced no transcript tokens for real speech")
        #expect(generated.allSatisfy { $0 >= 0 && $0 < model.config.vocab })
        // Non-degenerate: a genuine decode visits several distinct ids,
        // not one token repeated (the classic stuck-decoder failure).
        let distinct = Set(generated).count
        #expect(distinct > 1,
                "Whisper transcript is a single repeated token (degenerate decode)")
        print("Whisper transcribed real speech into \(generated.count) "
              + "tokens (\(distinct) distinct): \(generated.prefix(16))")
    }
}
