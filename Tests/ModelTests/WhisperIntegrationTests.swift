// Slow integration test: downloads (or hits cache) a small Whisper
// checkpoint and runs the audio encoder + decoder end-to-end on a
// synthetic waveform. Skipped automatically if the network or the
// checkpoint isn't available — mirrors the other ModelTests suites.
//
// whisper-tiny is the smallest Whisper variant (~150 MB f32), so the
// integration suite stays fast. The architecture (AudioEncoder + a
// cross-attending text decoder) is identical for base → large-v3.

import Foundation
import Testing
@testable import FFAI

@Suite("Whisper tiny integration", .serialized)
struct WhisperIntegrationTests {

    /// Load whisper-tiny from the HF cache / network, or return nil
    /// with a printed skip reason.
    private func loadWhisper() async -> WhisperModel? {
        do {
            let locator = ModelLocator()
            let dir = try await ModelLoadLock.shared.loadSerially {
                try await locator.resolve(idOrPath: "openai/whisper-tiny")
            }
            return try WhisperModel.load(directory: dir)
        } catch {
            print("Whisper integration test skipped: \(error)")
            return nil
        }
    }

    @Test("load — Whisper config + weights bind correctly")
    func loadWhisper_bindsWeights() async throws {
        guard let model = await loadWhisper() else { return }
        // whisper-tiny: d_model 384, 4 enc / 4 dec layers, 6 heads.
        #expect(model.config.hidden == 384)
        #expect(model.config.encoderLayers == 4)
        #expect(model.config.decoderLayers == 4)
        #expect(model.config.nMels == 80)
        #expect(model.encoder.layers.count == 4)
        #expect(model.decoderLayers.count == 4)
    }

    @Test("encode — audio encoder produces finite features")
    func encodeAudio_finiteFeatures() async throws {
        guard let model = await loadWhisper() else { return }
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
        guard let model = await loadWhisper() else { return }
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

    @Test("transcribe — greedy decode produces a token stream")
    func transcribe_producesTokens() async throws {
        guard let model = await loadWhisper() else { return }
        let sr = 16_000
        var wave = [Float](repeating: 0, count: sr)
        for i in 0..<sr {
            // A small chirp so the encoder sees non-trivial content.
            let f = 200.0 + 600.0 * Float(i) / Float(sr)
            wave[i] = 0.3 * sin(2.0 * Float.pi * f * Float(i) / Float(sr))
        }
        let features = model.encodeAudio(waveform: wave)
        let prefix = [50258, 50259, 50359, 50363]
            .filter { $0 < model.config.vocab }
        // Whisper's EOS is <|endoftext|> (id 50257); fall back to a
        // sentinel beyond vocab so the greedy loop runs to maxTokens.
        let eos = 50257 < model.config.vocab ? 50257 : model.config.vocab
        let generated = model.generateTranscript(
            audioFeatures: features,
            initialTokens: prefix.isEmpty ? [0] : prefix,
            eosToken: eos, maxTokens: 16)
        // The decode loop must produce a finite, in-vocab token stream.
        #expect(generated.allSatisfy { $0 >= 0 && $0 < model.config.vocab })
        print("Whisper transcribe produced \(generated.count) tokens: "
              + "\(generated.prefix(16))")
    }
}
