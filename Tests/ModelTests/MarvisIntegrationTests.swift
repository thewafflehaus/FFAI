// Integration test: loads a real Marvis / CSM TTS checkpoint and
// exercises the dual-transformer acoustic model. A load failure FAILS
// the suite — `loadMarvis()` is `throws` and the checkpoint is a hard
// requirement, not a "skip if missing".
//
// Marvis-TTS is built on Sesame's CSM architecture; FFAI's contribution
// is the CSM acoustic model — the backbone + depth-decoder transformers
// (built on FFAI's `LlamaLayer` blocks), the embedding tables and audio
// heads, and the frame-generation loop. The Mimi neural codec (the
// waveform tail) is a separate codec port; this suite verifies the
// model loads and `generateFrames` emits a finite Mimi code matrix —
// the contract the codec consumes.
//
// ── DISABLED ──────────────────────────────────────────────────────────
// `MarvisModel.buildTransformer` loads every projection with
// `loadLinear(..., quantization: nil)` — i.e. the CSM loader only
// supports an unquantized (fp16 / bf16) checkpoint. The only Marvis
// checkpoints published on HuggingFace (and the only ones in the local
// HF cache) are mlx affine-quantized: `Marvis-AI/marvis-tts-250m-v0.2-
// MLX-4bit` and `-8bit`, whose `*_proj.weight` tensors are U32-packed
// with separate `scales` / `biases`. Loading those through the
// nil-quantization path binds a U32 packed tensor as a dense `Linear`
// weight and produces a broken model. The non-quantized
// `Marvis-AI/marvis-tts-250m-v0.2-MLX` snapshot is not present on disk.
// Until `MarvisModel.load` plumbs `config.quantization` into
// `buildTransformer` (so `loadLinear` takes the quantized branch), no
// loadable Marvis checkpoint exists — the suite is disabled rather than
// silently skipped.

import Foundation
import Testing
@testable import FFAI

@Suite("Marvis (CSM) TTS integration", .serialized,
       .disabled("MarvisModel.buildTransformer hard-codes quantization nil; every cached Marvis checkpoint is mlx affine-quantized (U32-packed weights). Re-enable once MarvisModel.load plumbs config.quantization into buildTransformer, or an unquantized checkpoint is cached."))
struct MarvisIntegrationTests {

    /// Load Marvis from the HF cache. Throws on failure so a missing
    /// checkpoint fails the test instead of skipping it.
    private func loadMarvis() async throws -> MarvisModel {
        let dir = try await AudioFixtures.resolveCheckpoint(
            mlxAudioSlugs: ["Marvis-AI_marvis-tts-250m-v0.2-MLX",
                            "Marvis-AI_marvis-tts-250m-v0.2-MLX-8bit"],
            repoIds: ["Marvis-AI/marvis-tts-250m-v0.2-MLX"])
        return try await ModelLoadLock.shared.loadSerially {
            try await MarvisModel.load(directory: dir)
        }
    }

    @Test("load — CSM checkpoint binds both transformers")
    func loadMarvis_bindsTransformers() async throws {
        let model = try await loadMarvis()
        // The dual transformer: a backbone and a depth decoder.
        #expect(model.backbone.layers.count > 0)
        #expect(model.decoder.layers.count > 0)
        #expect(model.config.audioNumCodebooks > 0)
        #expect(model.sampleRate == 24_000)
    }

    @Test("generateFrames — decode emits a finite Mimi code matrix")
    func generateFrames_emitsCodes() async throws {
        let model = try await loadMarvis()
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
