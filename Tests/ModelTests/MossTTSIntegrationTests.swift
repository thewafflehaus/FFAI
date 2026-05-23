// Integration test: loads a real MOSS-TTS checkpoint from the HF cache
// and exercises config decoding + registry detection + staged synthesize error.
//
// A load failure FAILS the suite — `loadMossTTS()` is `throws` and the
// checkpoint is a hard requirement, not a "skip if missing".
//
// MOSS-TTS is a staged port; stage 1 ships config decoding + registry
// detection + the retained weight bundle. This suite verifies:
//   1. A real checkpoint loads without error.
//   2. The config decodes the n_vq, sampling_rate, and language_config fields.
//   3. The audio registry routes the directory to .mossTTS.
//   4. `synthesize` throws `MossTTSError.synthesisNotWired` by design
//      — this is the correct staged behavior, not a bug.
//
// DO NOT RUN this suite via `make test-integration` during CI until the
// full synthesis pipeline lands in a follow-on stage.

import Foundation
import Testing
@testable import FFAI

@Suite("MOSS-TTS integration", .serialized)
struct MossTTSIntegrationTests {

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
        return entries.contains { $0.hasSuffix(".safetensors") }
    }

    /// Resolve a MOSS-TTS-8B checkpoint directory from the HF cache.
    private func resolveMossTTSCheckpoint() async throws -> URL {
        let root = Self.hfCacheRoot
        let candidates = [
            "models--mlx-community--MOSS-TTS-8B-8bit",
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
        return try await locator.resolve(idOrPath: "mlx-community/MOSS-TTS-8B-8bit")
    }

    // ─── Load helper ──────────────────────────────────────────────────

    private func loadMossTTS() async throws -> MossTTSModel {
        let dir = try await resolveMossTTSCheckpoint()
        return try MossTTSModel.load(directory: dir)
    }

    // ─── Tests ───────────────────────────────────────────────────────

    @Test("load — MOSS-TTS config decodes n_vq and language_config from real checkpoint")
    func load_decodesConfig() async throws {
        let model = try await loadMossTTS()
        // The 8B checkpoint uses 32 VQ codebooks and 24 kHz sample rate.
        #expect(model.config.nVQ > 0)
        #expect(model.config.samplingRate > 0)
        // The language_config must decode as Qwen3.
        #expect(model.config.languageConfig.modelType == "qwen3")
        #expect(model.config.languageConfig.hiddenSize > 0)
        #expect(model.config.languageConfig.numHiddenLayers > 0)
        #expect(!model.weights.allKeys.isEmpty)
        print("[MOSS-TTS] Loaded: modelType=\(model.config.modelType), "
              + "nVQ=\(model.config.nVQ), "
              + "samplingRate=\(model.config.samplingRate), "
              + "languageModel=\(model.config.languageConfig.modelType), "
              + "hiddenSize=\(model.config.languageConfig.hiddenSize), "
              + "weightKeys=\(model.weights.allKeys.count)")
    }

    @Test("registry — AudioModelRegistry routes checkpoint to .mossTTS")
    func registry_routesMossTTS() async throws {
        let dir = try await resolveMossTTSCheckpoint()
        let loaded = try AudioModelRegistry.load(directory: dir)
        guard case .mossTTS = loaded else {
            Issue.record("AudioModelRegistry did not route to .mossTTS; got \(loaded)")
            return
        }
        #expect(loaded.capabilities == Capability.textToSpeech)
        print("[MOSS-TTS] Registry routed correctly, capabilities=\(loaded.capabilities)")
    }

    @Test("synthesize — staged port reports synthesis as not wired")
    func synthesize_reportsStaged() async throws {
        let model = try await loadMossTTS()
        #expect(throws: MossTTSError.self) {
            _ = try model.synthesize(text: "Hello, MOSS-TTS.")
        }
    }

    @Test("load — weight bundle contains expected top-level key prefixes")
    func load_weightKeyPrefixes() async throws {
        let model = try await loadMossTTS()
        let keys = model.weights.allKeys
        // MOSS-TTS checkpoints use prefixes like embedding_list.*, language_model.*,
        // lm_heads.*, emb_ext.* etc.
        let knownPrefixes = ["embedding_list", "language_model", "lm_heads", "emb_ext"]
        let found = knownPrefixes.filter { prefix in
            keys.contains { $0.hasPrefix(prefix) }
        }
        if found.isEmpty {
            let topPrefixes = Set(keys.map { String($0.prefix(16)) }).sorted()
            Issue.record("Expected at least one of \(knownPrefixes); found: \(topPrefixes)")
        }
        print("[MOSS-TTS] Weight prefixes found: \(found)")
    }
}
