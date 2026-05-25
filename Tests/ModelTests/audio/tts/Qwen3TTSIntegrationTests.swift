// Integration test: loads a real Qwen3TTS checkpoint from the HF cache
// and exercises the stage-1 config decode + family detection. A load
// failure FAILS the suite — `loadQwen3TTS()` is `throws` and the
// checkpoint is a hard requirement, not a "skip if missing".
//
// Qwen3TTS is a staged port; this build ships stage 1 (config decoding
// + `AudioModelRegistry` detection + the retained weight bundle). The
// `synthesize` path is intentionally not wired — it throws a typed
// `Qwen3TTSError`. This suite asserts the real checkpoint loads, the
// nested `talker_config` decodes, the audio registry routes it, and the
// staged `synthesize` throws *by design* rather than producing garbage.
// The talker / code-predictor / codec stages will extend this suite
// with synthesis assertions when they land.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3TTS integration", .serialized)
struct Qwen3TTSIntegrationTests {

    /// The bf16 mlx-audio snapshot is a complete checkpoint on disk
    /// (config + model.safetensors); the HF-hub repo id is the network
    /// fallback.
    private let mlxAudioSlugs = ["mlx-community_Qwen3-TTS-12Hz-0.6B-Base-bf16"]
    private let repoIds = ["mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"]

    /// Load Qwen3TTS from the HF cache. Throws on failure so a missing
    /// checkpoint fails the test instead of skipping it.
    private func loadQwen3TTS() async throws -> Qwen3TTSModel {
        let dir = try await AudioFixtures.resolveCheckpoint(
            mlxAudioSlugs: mlxAudioSlugs, repoIds: repoIds)
        return try Qwen3TTSModel.load(directory: dir)
    }

    @Test("load — Qwen3TTS checkpoint decodes the talker config")
    func loadQwen3TTS_decodesTalker() async throws {
        let model = try await loadQwen3TTS()
        // The talker is a Qwen3-style transformer.
        #expect(model.config.talker.nLayers > 0)
        #expect(model.config.talker.hidden > 0)
        #expect(model.config.talker.vocabSize > 0)
        #expect(model.sampleRate == 24_000)
        // The weight bundle is retained for the follow-on talker stage.
        #expect(model.weights.allKeys.isEmpty == false)
    }

    @Test("registry — Qwen3TTS routes through the audio registry")
    func registry_routesQwen3TTS() async throws {
        let dir = try await AudioFixtures.resolveCheckpoint(
            mlxAudioSlugs: mlxAudioSlugs, repoIds: repoIds)
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .qwen3TTS = loaded else {
            Issue.record("AudioModelRegistry did not route to Qwen3TTS")
            return
        }
        #expect(loaded.capabilities == Capability.textToSpeech)
    }

    @Test("synthesize — staged port reports synthesis as not wired")
    func synthesize_reportsStaged() async throws {
        let model = try await loadQwen3TTS()
        // Stage 1 ships config + detection only; synthesis throws a
        // typed, descriptive error rather than producing garbage audio.
        // The model MUST load first — only the `synthesize` call throws.
        #expect(throws: Qwen3TTSError.self) {
            _ = try model.synthesize(text: "Hello.")
        }
    }
}
