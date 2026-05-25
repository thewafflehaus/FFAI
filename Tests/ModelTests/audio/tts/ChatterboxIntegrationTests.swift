// Integration test: loads a real Chatterbox checkpoint from the HF
// cache and exercises config decoding + registry detection.
//
// A load failure FAILS the suite — `loadChatterbox()` is `throws` and
// the checkpoint is a hard requirement, not a "skip if missing".
//
// Chatterbox is a staged port; stage 1 ships config decoding + registry
// detection + the retained weight bundle. The T3 backbone, S3Gen flow
// decoder, and HiFi-GAN vocoder are follow-on stages. This suite
// verifies:
//   1. A real checkpoint loads without error.
//   2. The config decodes the T3 and (for Turbo) GPT-2 blocks.
//   3. The audio registry routes the directory to .chatterbox.
//   4. `synthesize` throws `ChatterboxError.synthesisNotWired` by design
//      — this is the correct staged behavior, not a bug.
//
// DO NOT RUN this suite via `make test-integration` during CI until the
// full synthesis pipeline lands in a follow-on stage.

import Foundation
import Testing
@testable import FFAI

@Suite("Chatterbox TTS integration", .serialized)
struct ChatterboxIntegrationTests {

    // ─── Checkpoint resolution ────────────────────────────────────────

    /// Root of the HF cache.
    private static var hfCacheRoot: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/huggingface/hub")
    }

    /// True when `dir` looks like a usable snapshot: has `config.json`
    /// plus at least one `.safetensors` file.
    private static func isCompleteSnapshot(_ dir: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        else { return false }
        guard let entries = try? fm.contentsOfDirectory(atPath: dir.path)
        else { return false }
        return entries.contains { $0.hasSuffix(".safetensors") }
    }

    /// Resolve a Chatterbox checkpoint directory from the HF cache.
    ///
    /// Tries the Turbo fp16 snapshot first (fastest, complete on most
    /// dev machines), then the 4-bit and 8-bit quantized variants, then
    /// the Regular fp16 snapshot, and finally the ResembleAI Turbo origin.
    /// Falls back to HF download via `ModelLocator` if none are cached.
    ///
    /// Throws if no candidate resolves — the caller lets that fail the test.
    private func resolveChatterboxCheckpoint() async throws -> URL {
        let root = Self.hfCacheRoot
        let blobRoot = "snapshots"

        // Ordered candidate directories — (model-slug, snapshot-hash).
        // The snapshot hashes are pinned to known-good cached versions.
        let localCandidates: [(slug: String, snap: String)] = [
            ("models--mlx-community--chatterbox-turbo-fp16",
             "b2d0a13aa7cfff0a06d9acb247ae91c8f19a6d75"),
            ("models--mlx-community--chatterbox-turbo-4bit", ""),
            ("models--mlx-community--chatterbox-turbo-8bit", ""),
            ("models--mlx-community--Chatterbox-TTS-fp16",
             "77c7c8f9307beb5dc6c03cebc7942c4de9d617c9"),
            ("models--ResembleAI--chatterbox-turbo", ""),
        ]

        let fm = FileManager.default
        for candidate in localCandidates {
            let snapshotsDir = root
                .appendingPathComponent(candidate.slug)
                .appendingPathComponent(blobRoot)
            // Try the pinned snapshot first; fall back to scanning for any snapshot.
            var dirs: [URL] = []
            if !candidate.snap.isEmpty {
                dirs.append(snapshotsDir.appendingPathComponent(candidate.snap))
            }
            if let subs = try? fm.contentsOfDirectory(at: snapshotsDir,
                                                       includingPropertiesForKeys: nil) {
                dirs.append(contentsOf: subs.sorted {
                    $0.lastPathComponent < $1.lastPathComponent
                })
            }
            if let dir = dirs.first(where: { Self.isCompleteSnapshot($0) }) {
                return dir
            }
        }

        // HF network fallback — try the Turbo fp16 hub repo.
        let locator = ModelLocator()
        return try await locator.resolve(idOrPath: "mlx-community/chatterbox-turbo-fp16")
    }

    // ─── Load helpers ─────────────────────────────────────────────────

    /// Load a Chatterbox model from the HF cache.
    /// Throws on failure so a missing checkpoint FAILS the test.
    private func loadChatterbox() async throws -> ChatterboxModel {
        let dir = try await resolveChatterboxCheckpoint()
        return try ChatterboxModel.load(directory: dir)
    }

    // ─── Tests ───────────────────────────────────────────────────────

    @Test("load — Chatterbox config decodes T3 block from real checkpoint")
    func load_decodesT3Config() async throws {
        let model = try await loadChatterbox()
        // The T3 config must always decode — it carries the speech vocab size
        // and the backbone identifier for both Regular and Turbo.
        #expect(model.config.t3.speechTokensDictSize > 0)
        #expect(model.config.t3.startSpeechToken > 0)
        #expect(model.config.t3.stopSpeechToken >= 0)
        #expect(model.config.sampleRate == 24_000)
        // The weight bundle must be non-empty.
        #expect(model.weights.allKeys.isEmpty == false)
        print("[Chatterbox] Loaded: modelType=\(model.config.modelType), "
              + "isTurbo=\(model.config.isTurbo), "
              + "sampleRate=\(model.config.sampleRate), "
              + "speechVocab=\(model.config.t3.speechTokensDictSize), "
              + "weightKeys=\(model.weights.allKeys.count)")
    }

    @Test("load — Turbo checkpoint decodes the GPT-2 backbone config")
    func load_turboDecodesGPT2Config() async throws {
        let model = try await loadChatterbox()
        guard model.config.isTurbo else {
            print("[Chatterbox] Skipping GPT-2 check — loaded Regular variant")
            return
        }
        // Turbo must expose the GPT-2 Medium backbone config.
        let gpt2 = model.config.gpt2
        #expect(gpt2 != nil)
        #expect((gpt2?.nLayer ?? 0) > 0)
        #expect((gpt2?.hiddenSize ?? 0) > 0)
        #expect((gpt2?.headDim ?? 0) > 0)
        print("[Chatterbox] GPT-2 config: nLayer=\(gpt2?.nLayer ?? -1), "
              + "hiddenSize=\(gpt2?.hiddenSize ?? -1), "
              + "headDim=\(gpt2?.headDim ?? -1)")
    }

    @Test("registry — AudioModelRegistry routes checkpoint to .chatterbox")
    func registry_routesChatterbox() async throws {
        let dir = try await resolveChatterboxCheckpoint()
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .chatterbox = loaded else {
            Issue.record("AudioModelRegistry did not route to .chatterbox")
            return
        }
        #expect(loaded.capabilities == Capability.textToSpeech)
        print("[Chatterbox] Registry routed correctly, capabilities=\(loaded.capabilities)")
    }

    @Test("synthesize — staged port reports synthesis as not wired")
    func synthesize_reportsStaged() async throws {
        let model = try await loadChatterbox()
        // Stage 1 ships config + detection only; synthesis throws a
        // typed, descriptive error rather than producing garbage audio.
        // The model MUST load first — only the `synthesize` call throws.
        #expect(throws: ChatterboxError.self) {
            _ = try model.synthesize(text: "Hello, my name is Chatterbox.")
        }
    }

    @Test("load — weight bundle contains expected top-level prefixes")
    func load_weightKeyPrefixes() async throws {
        let model = try await loadChatterbox()
        let keys = model.weights.allKeys
        // Chatterbox checkpoints use top-level prefixes: ve.*, t3.*, s3gen.*
        // (Regular) or similar (Turbo). At least one of these must be present
        // to confirm the weights are the right checkpoint.
        let knownPrefixes = ["ve.", "t3.", "s3gen."]
        let found = knownPrefixes.filter { prefix in
            keys.contains { $0.hasPrefix(prefix) }
        }
        if found.isEmpty {
            let topPrefixes = Set(keys.map { String($0.prefix(6)) }).sorted()
            let msg = "Expected at least one of \(knownPrefixes) in weight keys; found: \(topPrefixes)"
            Issue.record("\(msg)")
        }
        #expect(!found.isEmpty)
        print("[Chatterbox] Weight prefixes found: \(found)")
    }
}
