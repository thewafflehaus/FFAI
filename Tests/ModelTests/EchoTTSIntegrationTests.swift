// Integration test: loads a real mlx-community/echo-tts-base checkpoint
// from the HF cache and exercises the config + load path. A missing
// checkpoint FAILS the suite — the checkpoint is a hard requirement, not
// a "skip if missing".
//
// EchoTTS is a diffusion-based TTS model; the GPU forward pass (DiT
// diffusion + Fish S1 DAC decode) is not yet wired in this FFAI build.
// This suite verifies:
//   1. The checkpoint loads without error (`EchoTTSModel.load`).
//   2. The decoded config matches the published base-model geometry.
//   3. The PCA state (`pca_state.safetensors`) is parsed correctly.
//   4. The weight file is present and non-empty.
//   5. `generatePlaceholder` returns a structurally valid waveform tensor.
//   6. `synthesize` correctly reports the forward pass as not wired.
//   7. The AudioModelRegistry detects the checkpoint as a TTS model.
//
// DO NOT RUN individual tests from the ModelTests suite without `make
// test-integration` — the suite serializes loads to avoid OOM from
// parallel GPU allocations across suites.

import Foundation
import Testing
@testable import FFAI

@Suite("EchoTTS integration", .serialized)
struct EchoTTSIntegrationTests {

    /// Resolve the echo-tts-base checkpoint from the local HF cache.
    /// Prefers the standard HF blob layout (`models--org--repo/snapshots/`).
    /// Throws if the checkpoint is absent so the test fails with a clear
    /// message rather than silently skipping.
    private func resolveCheckpoint() throws -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let hfHub = home.appendingPathComponent(".cache/huggingface/hub")

        // Standard HF blob layout.
        let hfDir = hfHub
            .appendingPathComponent("models--mlx-community--echo-tts-base")
        if let snapshotDir = latestSnapshot(in: hfDir) {
            if isCompleteSnapshot(snapshotDir) { return snapshotDir }
        }

        // Flat mlx-audio layout (alternate local cache).
        let mlxDir = hfHub.appendingPathComponent(
            "mlx-audio/mlx-community_echo-tts-base")
        if isCompleteSnapshot(mlxDir) { return mlxDir }

        throw EchoTTSIntegrationError.checkpointNotFound
    }

    /// Pick the newest (lexicographically last) snapshot under
    /// `models--*/snapshots/`. Returns nil when no snapshots exist.
    private func latestSnapshot(in hfDir: URL) -> URL? {
        let snapshotsDir = hfDir.appendingPathComponent("snapshots")
        guard let entries = try? FileManager.default.contentsOfDirectory(
            atPath: snapshotsDir.path)
        else { return nil }
        let sorted = entries.sorted().reversed()
        return sorted.first.map { snapshotsDir.appendingPathComponent($0) }
    }

    /// True when `dir` has a `config.json` and at least one `.safetensors`
    /// file — the minimum a complete snapshot needs.
    private func isCompleteSnapshot(_ dir: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        else { return false }
        guard let entries = try? fm.contentsOfDirectory(atPath: dir.path)
        else { return false }
        return entries.contains { $0.hasSuffix(".safetensors") }
    }

    // ─── Tests ────────────────────────────────────────────────────────────

    @Test("load — resolves the checkpoint and decodes the config")
    func load_decodesConfig() throws {
        let dir = try resolveCheckpoint()
        let model = try EchoTTSModel.load(directory: dir)

        // Published base-model geometry.
        #expect(model.config.sampleRate == 44100)
        #expect(model.config.dit.modelSize == 2048)
        #expect(model.config.dit.numLayers == 24)
        #expect(model.config.dit.numHeads == 16)
        #expect(model.config.dit.latentSize == 80)
        #expect(model.config.dit.speakerPatchSize == 4)
        #expect(model.config.dit.textVocabSize == 256)
        #expect(model.config.sampler.sequenceLength == 640)
        #expect(model.config.maxTextLength == 768)
        print("EchoTTS config: sampleRate=\(model.config.sampleRate), "
              + "ditLayers=\(model.config.dit.numLayers), "
              + "seqLen=\(model.config.sampler.sequenceLength)")
    }

    @Test("load — PCA state tensors are present and correctly shaped")
    func load_pcaStatePresent() throws {
        let dir = try resolveCheckpoint()
        let model = try EchoTTSModel.load(directory: dir)

        // PCA components: [latentSize, codecDim]. latentSize=80 for base.
        #expect(model.pcaComponents != nil, "pcaComponents should be loaded")
        #expect(model.pcaMean != nil, "pcaMean should be loaded")
        if let comp = model.pcaComponents {
            #expect(comp.shape.count == 2)
            #expect(comp.shape[0] == model.config.dit.latentSize,
                    "pca_components first dim should match latentSize")
        }
        // latentScale is a positive scalar (defaults to 1.0 if absent).
        #expect(model.latentScale > 0)
        print("EchoTTS PCA: components=\(model.pcaComponents?.shape ?? []), "
              + "mean=\(model.pcaMean?.shape ?? []), scale=\(model.latentScale)")
    }

    @Test("load — model weights are present and non-empty")
    func load_weightsPresent() throws {
        let dir = try resolveCheckpoint()
        let model = try EchoTTSModel.load(directory: dir)

        // A non-trivial checkpoint should carry hundreds of weight tensors.
        #expect(model.weightCount > 0, "model.safetensors should have weights")
        print("EchoTTS weight count: \(model.weightCount)")
    }

    @Test("load — AudioModelRegistry detects the checkpoint as TTS")
    func load_registryDetectsAsTTS() throws {
        let dir = try resolveCheckpoint()
        let config = try ModelConfig.load(from: dir)
        #expect(EchoTTSModel.handles(config))
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config)
                == Capability.textToSpeech)
    }

    @Test("generatePlaceholder — returns a structurally valid waveform")
    func generatePlaceholder_isValid() throws {
        let dir = try resolveCheckpoint()
        let model = try EchoTTSModel.load(directory: dir)

        // A 0.1 s placeholder at 44100 Hz → 4410 samples.
        let wav = model.generatePlaceholder(durationSeconds: 0.1)
        let expectedSamples = max(1, Int(0.1 * Double(model.sampleRate)))
        #expect(wav.shape == [expectedSamples])
        #expect(wav.dtype == .f32)
        let samples = wav.toArray(as: Float.self)
        #expect(samples.allSatisfy { $0.isFinite })
        print("EchoTTS placeholder: \(expectedSamples) samples at \(model.sampleRate) Hz")
    }

    @Test("synthesize — correctly reports the forward pass as not wired")
    func synthesize_reportsNotWired() throws {
        let dir = try resolveCheckpoint()
        let model = try EchoTTSModel.load(directory: dir)
        // The diffusion forward pass requires GPU operators (batched GEMM +
        // SDPA) that are not yet in the FFAI Ops set.
        #expect(throws: EchoTTSError.self) {
            _ = try model.synthesize(text: "Hello, world.")
        }
    }
}

// ─── Fixtures error ───────────────────────────────────────────────────────────

private enum EchoTTSIntegrationError: Error, CustomStringConvertible {
    case checkpointNotFound

    var description: String {
        "EchoTTS integration: checkpoint mlx-community/echo-tts-base not found "
            + "in HF cache (~/.cache/huggingface/hub). "
            + "Run: huggingface-cli download mlx-community/echo-tts-base"
    }
}
