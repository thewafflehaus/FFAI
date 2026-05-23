// Integration test: loads a real MOSS-TTS-Nano checkpoint from the HF cache
// and exercises config decoding + registry detection + staged synthesize error.
//
// A load failure FAILS the suite — `loadMossTTSNano()` is `throws` and the
// checkpoint is a hard requirement, not a "skip if missing".
//
// MOSS-TTS-Nano is a staged port; stage 1 ships config decoding + registry
// detection + the retained weight bundle. This suite verifies:
//   1. A real checkpoint loads without error.
//   2. The config decodes the n_vq, gpt2_config, and audio tokenizer fields.
//   3. The audio registry routes the directory to .mossTTSNano.
//   4. `synthesize` throws `MossTTSNanoError.synthesisNotWired` by design
//      — this is the correct staged behavior, not a bug.
//
// DO NOT RUN this suite via `make test-integration` during CI until the
// full synthesis pipeline lands in a follow-on stage.

import Foundation
import Testing
@testable import FFAI

@Suite("MOSS-TTS-Nano integration", .serialized)
struct MossTTSNanoIntegrationTests {

    // ─── Checkpoint resolution ────────────────────────────────────────

    private static var hfCacheRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    private static func isCompleteSnapshot(_ dir: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        else { return false }
        guard let entries = try? fm.contentsOfDirectory(atPath: dir.path)
        else { return false }
        // Nano uses .safetensors; accept any weights file.
        return entries.contains { $0.hasSuffix(".safetensors") || $0.hasSuffix(".npz") }
    }

    /// Resolve a MOSS-TTS-Nano checkpoint directory from the HF cache.
    private func resolveMossTTSNanoCheckpoint() async throws -> URL {
        let root = Self.hfCacheRoot
        let candidates = [
            "models--mlx-community--MOSS-TTS-Nano-100M",
        ]
        let fm = FileManager.default
        for slug in candidates {
            let snapshots = root.appendingPathComponent(slug)
                .appendingPathComponent("snapshots")
            guard let subs = try? fm.contentsOfDirectory(
                at: snapshots, includingPropertiesForKeys: nil)
            else { continue }
            if let dir = subs.first(where: { Self.isCompleteSnapshot($0) }) {
                return dir
            }
        }
        // HF network fallback.
        let locator = ModelLocator(downloader: ModelDownloader())
        return try await locator.resolve(idOrPath: "mlx-community/MOSS-TTS-Nano-100M")
    }

    // ─── Load helper ──────────────────────────────────────────────────

    private func loadMossTTSNano() async throws -> MossTTSNanoModel {
        let dir = try await resolveMossTTSNanoCheckpoint()
        return try MossTTSNanoModel.load(directory: dir)
    }

    // ─── Tests ───────────────────────────────────────────────────────

    @Test("load — MOSS-TTS-Nano config decodes n_vq and gpt2_config from real checkpoint")
    func load_decodesConfig() async throws {
        let model = try await loadMossTTSNano()
        // The Nano checkpoint uses 16 VQ codebooks and 48 kHz sample rate.
        #expect(model.config.nVQ > 0)
        #expect(model.config.audioTokenizerSampleRate > 0)
        // The gpt2_config must decode with sensible transformer dimensions.
        #expect(model.config.gpt2Config.nEmbd > 0)
        #expect(model.config.gpt2Config.nLayer > 0)
        #expect(model.config.gpt2Config.nHead > 0)
        #expect(model.config.gpt2Config.headDim > 0)
        // The local transformer must have nPositions = nVQ + 1.
        let localCfg = model.config.localGPT2Config()
        #expect(localCfg.nPositions == model.config.nVQ + 1)
        #expect(!model.weights.allKeys.isEmpty)
        print("[MOSS-TTS-Nano] Loaded: modelType=\(model.config.modelType), "
              + "nVQ=\(model.config.nVQ), "
              + "sampleRate=\(model.config.audioTokenizerSampleRate), "
              + "gpt2.nEmbd=\(model.config.gpt2Config.nEmbd), "
              + "gpt2.nLayer=\(model.config.gpt2Config.nLayer), "
              + "localPositions=\(localCfg.nPositions), "
              + "weightKeys=\(model.weights.allKeys.count)")
    }

    @Test("load — audioCodebookSizes matches n_vq")
    func load_audioCodebookSizesMatchNVQ() async throws {
        let model = try await loadMossTTSNano()
        #expect(model.config.audioCodebookSizes.count == model.config.nVQ)
        #expect(model.config.audioCodebookSizes.allSatisfy { $0 > 0 })
        print("[MOSS-TTS-Nano] audioCodebookSizes: \(model.config.audioCodebookSizes)")
    }

    @Test("registry — AudioModelRegistry routes checkpoint to .mossTTSNano")
    func registry_routesMossTTSNano() async throws {
        let dir = try await resolveMossTTSNanoCheckpoint()
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .mossTTSNano = loaded else {
            Issue.record("AudioModelRegistry did not route to .mossTTSNano; got \(loaded)")
            return
        }
        #expect(loaded.capabilities == Capability.textToSpeech)
        print("[MOSS-TTS-Nano] Registry routed correctly, capabilities=\(loaded.capabilities)")
    }

    @Test("synthesize — staged port reports synthesis as not wired")
    func synthesize_reportsStaged() async throws {
        let model = try await loadMossTTSNano()
        #expect(throws: MossTTSNanoError.self) {
            _ = try model.synthesize(text: "Hello, MOSS-TTS-Nano.")
        }
    }

    @Test("load — weight bundle contains expected top-level key prefixes")
    func load_weightKeyPrefixes() async throws {
        let model = try await loadMossTTSNano()
        let keys = model.weights.allKeys
        // MOSS-TTS-Nano checkpoints use prefixes like transformer.*, local_transformer.*,
        // audio_embeddings.*, text_lm_head.*, audio_lm_heads.*, etc.
        let knownPrefixes = ["transformer", "local_transformer", "audio_embeddings"]
        let found = knownPrefixes.filter { prefix in
            keys.contains { $0.hasPrefix(prefix) }
        }
        if found.isEmpty {
            let topPrefixes = Set(keys.map { String($0.prefix(20)) }).sorted()
            Issue.record("Expected at least one of \(knownPrefixes); found: \(topPrefixes)")
        }
        print("[MOSS-TTS-Nano] Weight prefixes found: \(found)")
    }

    @Test("load — audio tokenizer type is moss-audio-tokenizer-nano")
    func load_audioTokenizerType() async throws {
        let model = try await loadMossTTSNano()
        #expect(model.config.audioTokenizerType.contains("nano"))
        print("[MOSS-TTS-Nano] audioTokenizerType=\(model.config.audioTokenizerType)")
    }
}
