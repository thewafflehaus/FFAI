// Integration test: loads a real PocketTTS checkpoint from the HF cache
// and exercises config decoding + registry detection + staged synthesize error.
//
// A load failure FAILS the suite — `loadPocketTTS()` is `throws` and the
// checkpoint is a hard requirement, not a "skip if missing".
//
// PocketTTS is a staged port; stage 1 ships config decoding + registry
// detection + the retained weight bundle. This suite verifies:
//   1. A real checkpoint loads without error.
//   2. The config decodes the flow_lm and mimi blocks correctly.
//   3. The audio registry routes the directory to .pocketTTS.
//   4. `synthesize("Hello world.")` throws `PocketTTSError.synthesisNotWired`
//      by design — this is the correct staged behavior, not a bug.
//
// When the full synthesis pipeline lands in a follow-on stage, test 4
// should be replaced with an assertive waveform test:
//   let audio = try model.synthesize(text: "Hello world.")
//   #expect(!audio.isEmpty)
//   let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
//   #expect(rms > 1e-4 && rms < 1.0)
//
// DO NOT RUN this suite via `make test-integration` during CI until the
// full synthesis pipeline lands in a follow-on stage.

import Foundation
import Testing
@testable import FFAI

@Suite("PocketTTS integration", .serialized)
struct PocketTTSIntegrationTests {

    // ─── Checkpoint resolution ────────────────────────────────────────

    /// Root of the HF hub cache.
    private static var hfCacheRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    /// True when `dir` is a usable snapshot: has `config.json` plus at
    /// least one `.safetensors` file.
    private static func isCompleteSnapshot(_ dir: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        else { return false }
        guard let entries = try? fm.contentsOfDirectory(atPath: dir.path)
        else { return false }
        return entries.contains { $0.hasSuffix(".safetensors") }
    }

    /// Resolve the PocketTTS checkpoint directory from the HF cache.
    ///
    /// Tries the mlx-community/pocket-tts snapshot first (present on dev
    /// machines after a `mlx_audio` or HF pull), then falls back to a
    /// `ModelLocator` download so the test is self-healing on CI.
    ///
    /// Throws if no candidate resolves — the caller lets that fail the test.
    private func resolvePocketTTSCheckpoint() async throws -> URL {
        let root = Self.hfCacheRoot
        let candidates = [
            "models--mlx-community--pocket-tts",
        ]
        let fm = FileManager.default
        for slug in candidates {
            let snapshots = root
                .appendingPathComponent(slug)
                .appendingPathComponent("snapshots")
            guard let subs = try? fm.contentsOfDirectory(
                at: snapshots, includingPropertiesForKeys: nil)
            else { continue }
            // Take the first snapshot dir that looks complete.
            if let dir = subs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
                .first(where: { Self.isCompleteSnapshot($0) }) {
                return dir
            }
        }
        // HF network fallback.
        let locator = ModelLocator(downloader: ModelDownloader())
        return try await locator.resolve(idOrPath: "mlx-community/pocket-tts")
    }

    // ─── Load helper ──────────────────────────────────────────────────

    /// Load a PocketTTS model from the HF cache.
    /// Throws on failure so a missing checkpoint FAILS the test.
    private func loadPocketTTS() async throws -> PocketTTSModel {
        let dir = try await resolvePocketTTSCheckpoint()
        return try PocketTTSModel.load(directory: dir)
    }

    // ─── Tests ───────────────────────────────────────────────────────

    @Test("load — PocketTTS config decodes flow_lm and mimi blocks from real checkpoint")
    func load_decodesConfig() async throws {
        let model = try await loadPocketTTS()

        // Flow LM transformer block.
        #expect(model.config.flowLM.transformer.dModel > 0)
        #expect(model.config.flowLM.transformer.numHeads > 0)
        #expect(model.config.flowLM.transformer.numLayers > 0)
        #expect(model.config.flowLM.transformer.dimFeedforward > 0)

        // Flow net block.
        #expect(model.config.flowLM.flow.dim > 0)
        #expect(model.config.flowLM.flow.depth > 0)

        // Lookup table.
        #expect(model.config.flowLM.lookupTable.nBins > 0)

        // Mimi codec.
        #expect(model.config.mimi.sampleRate == 24_000)
        #expect(model.config.mimi.frameRate > 0)
        #expect(model.config.mimi.seanet.hopLength > 0)
        #expect(model.config.mimi.quantizer.dimension > 0)
        #expect(model.config.mimi.quantizer.outputDimension > 0)

        // Overall.
        #expect(model.sampleRate == 24_000)
        #expect(!model.weights.allKeys.isEmpty)

        print("[PocketTTS] Loaded: modelType=\(model.config.modelType), "
              + "dModel=\(model.config.flowLM.transformer.dModel), "
              + "sampleRate=\(model.sampleRate), "
              + "frameRate=\(model.config.mimi.frameRate), "
              + "weightKeys=\(model.weights.allKeys.count)")
    }

    @Test("load — flow_lm transformer matches expected pocket-tts architecture")
    func load_transformerMatchesArchitecture() async throws {
        let model = try await loadPocketTTS()
        let xf = model.config.flowLM.transformer
        // Canonical pocket-tts checkpoint: d_model=1024, num_heads=16, num_layers=6.
        #expect(xf.dModel == 1_024)
        #expect(xf.numHeads == 16)
        #expect(xf.numLayers == 6)
        #expect(xf.hiddenScale == 4)
        #expect(xf.dimFeedforward == 4_096)
    }

    @Test("load — mimi seanet ratios match expected pocket-tts architecture")
    func load_seanetRatiosMatch() async throws {
        let model = try await loadPocketTTS()
        let seanet = model.config.mimi.seanet
        // Canonical: ratios=[6,5,4], hopLength=120.
        #expect(seanet.ratios == [6, 5, 4])
        #expect(seanet.hopLength == 120)
    }

    @Test("registry — AudioModelRegistry routes checkpoint to .pocketTTS")
    func registry_routesPocketTTS() async throws {
        let dir = try await resolvePocketTTSCheckpoint()
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .pocketTTS = loaded else {
            Issue.record("AudioModelRegistry did not route to .pocketTTS; got \(loaded)")
            return
        }
        #expect(loaded.capabilities == Capability.textToSpeech)
        print("[PocketTTS] Registry routed correctly, capabilities=\(loaded.capabilities)")
    }

    @Test("synthesize — staged port reports synthesis as not wired")
    func synthesize_reportsStaged() async throws {
        let model = try await loadPocketTTS()
        // Stage 1 ships config + detection only; `synthesize` is staged.
        // The model MUST load first — only the `synthesize` call throws.
        //
        // TODO(follow-on): When the full synthesis pipeline lands, replace this
        // with an assertive waveform test:
        //   let audio = try model.synthesize(text: "Hello world.")
        //   #expect(!audio.isEmpty)
        //   let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        //   #expect(rms > 1e-4 && rms < 1.0)
        #expect(throws: PocketTTSError.self) {
            _ = try model.synthesize(text: "Hello world.")
        }
    }

    @Test("load — weight bundle contains expected top-level key prefixes")
    func load_weightKeyPrefixes() async throws {
        let model = try await loadPocketTTS()
        let keys = model.weights.allKeys
        // PocketTTS checkpoints use prefixes: flow_lm.*, mimi.*.
        let knownPrefixes = ["flow_lm.", "mimi."]
        let found = knownPrefixes.filter { prefix in
            keys.contains { $0.hasPrefix(prefix) }
        }
        if found.isEmpty {
            let topPrefixes = Set(keys.map { String($0.prefix(12)) }).sorted()
            Issue.record("Expected at least one of \(knownPrefixes); found: \(topPrefixes)")
        }
        #expect(!found.isEmpty)
        print("[PocketTTS] Weight prefixes found: \(found)")
    }
}
