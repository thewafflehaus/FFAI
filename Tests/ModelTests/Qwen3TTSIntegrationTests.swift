// Slow integration test: downloads (or hits cache) a Qwen3TTS
// checkpoint and exercises the stage-1 config decode + family
// detection. Skipped automatically if the network or the checkpoint
// isn't available — mirrors the other ModelTests suites.
//
// Qwen3TTS is a staged port; this build ships stage 1 (config decoding
// + `AudioModelRegistry` detection). This suite verifies a real
// Qwen3TTS checkpoint loads into a `Qwen3TTSModel`, the nested
// `talker_config` decodes, and the staged `synthesize` reports itself
// as not-yet-wired rather than producing garbage. The talker / code-
// predictor / codec stages will extend this suite with synthesis
// assertions when they land.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3TTS integration", .serialized)
struct Qwen3TTSIntegrationTests {

    /// Load Qwen3TTS from the HF cache / network, or return nil with a
    /// printed skip reason.
    private func loadQwen3TTS() async -> Qwen3TTSModel? {
        for repoId in [
            "mlx-community/Qwen3-TTS-Flash-bf16",
            "Qwen/Qwen3-TTS-Flash",
        ] {
            do {
                let locator = ModelLocator()
                let dir = try await ModelLoadLock.shared.loadSerially {
                    try await locator.resolve(idOrPath: repoId)
                }
                return try Qwen3TTSModel.load(directory: dir)
            } catch {
                print("Qwen3TTS load from \(repoId) skipped: \(error)")
            }
        }
        return nil
    }

    @Test("load — Qwen3TTS checkpoint decodes the talker config")
    func loadQwen3TTS_decodesTalker() async throws {
        guard let model = await loadQwen3TTS() else {
            print("Qwen3TTS integration test skipped: checkpoint unavailable")
            return
        }
        // The talker is a Qwen3-style transformer.
        #expect(model.config.talker.nLayers > 0)
        #expect(model.config.talker.hidden > 0)
        #expect(model.config.talker.vocabSize > 0)
        #expect(model.sampleRate == 24_000)
    }

    @Test("synthesize — staged port reports synthesis as not wired")
    func synthesize_reportsStaged() async throws {
        guard let model = await loadQwen3TTS() else {
            print("Qwen3TTS integration test skipped: checkpoint unavailable")
            return
        }
        // Stage 1 ships config + detection only; synthesis throws a
        // typed, descriptive error rather than producing garbage audio.
        #expect(throws: Qwen3TTSError.self) {
            _ = try model.synthesize(text: "Hello.")
        }
    }
}
