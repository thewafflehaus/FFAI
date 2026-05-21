// Slow integration test: downloads (or hits cache) a Marvis / CSM TTS
// checkpoint and exercises the dual-transformer acoustic model. Skipped
// automatically if the network or the checkpoint isn't available —
// mirrors the other ModelTests suites.
//
// Marvis-TTS is built on Sesame's CSM architecture; FFAI's contribution
// is the CSM acoustic model — the backbone + depth-decoder transformers
// (built on FFAI's `LlamaLayer` blocks), the embedding tables and audio
// heads, and the frame-generation loop. The Mimi neural codec (the
// waveform tail) is a separate codec port; this suite verifies the
// model loads and `generateFrames` emits a finite Mimi code matrix —
// the contract the codec consumes.

import Foundation
import Testing
@testable import FFAI

@Suite("Marvis (CSM) TTS integration", .serialized)
struct MarvisIntegrationTests {

    /// Load Marvis from the HF cache / network, or return nil with a
    /// printed skip reason.
    private func loadMarvis() async -> MarvisModel? {
        for repoId in [
            "Marvis-AI/marvis-tts-250m-v0.2-MLX-fp16",
            "Marvis-AI/marvis-tts-250m-v0.1-MLX",
        ] {
            do {
                let locator = ModelLocator()
                let dir = try await ModelLoadLock.shared.loadSerially {
                    try await locator.resolve(idOrPath: repoId)
                }
                return try await MarvisModel.load(directory: dir)
            } catch {
                print("Marvis load from \(repoId) skipped: \(error)")
            }
        }
        return nil
    }

    @Test("load — CSM checkpoint binds both transformers")
    func loadMarvis_bindsTransformers() async throws {
        guard let model = await loadMarvis() else {
            print("Marvis integration test skipped: checkpoint unavailable")
            return
        }
        // The dual transformer: a backbone and a depth decoder.
        #expect(model.backbone.layers.count > 0)
        #expect(model.decoder.layers.count > 0)
        #expect(model.config.audioNumCodebooks > 0)
        #expect(model.sampleRate == 24_000)
    }

    @Test("generateFrames — decode emits a finite Mimi code matrix")
    func generateFrames_emitsCodes() async throws {
        guard let model = await loadMarvis() else {
            print("Marvis integration test skipped: checkpoint unavailable")
            return
        }
        // Greedy decode, capped short for test runtime.
        let codes = try model.generateFrames(
            text: "Hello.", speaker: 0, maxFrames: 8, temperature: 0)
        // One row per Mimi codebook.
        #expect(codes.count == model.config.audioNumCodebooks)
        #expect(!codes[0].isEmpty, "Marvis produced no audio frames")
        // Every codebook row has the same frame count.
        let nFrames = codes[0].count
        #expect(codes.allSatisfy { $0.count == nFrames })
        // Codes are valid Mimi indices — within the audio vocabulary.
        let audioVocab = model.config.audioVocabSize
        #expect(codes.allSatisfy { row in
            row.allSatisfy { $0 >= 0 && $0 < audioVocab }
        })
        print("Marvis generated \(nFrames) Mimi frames "
              + "× \(codes.count) codebooks")
    }
}
