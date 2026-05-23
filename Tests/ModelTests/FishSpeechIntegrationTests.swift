// FishSpeech integration test — load mlx-community/fish-audio-s2-pro-8bit
// from the HuggingFace cache (pre-downloaded at ~/.cache/huggingface/hub/)
// and run a short Stage-1 synthesis pass.
//
// Stage-1 coverage:
//   ✅  Config parsed from real config.json
//   ✅  Weights loaded (8-bit quantized, sharded via model.safetensors.index.json)
//   ✅  FishSpeechModel constructed (slow backbone + fast decoder)
//   ✅  synthesize(...) reaches the codec stub and throws codecNotAvailable
//
// Stage-2 (FishS1DAC waveform decode) is NOT tested here; see planning/
// issues/features/F-DAC-001 for the follow-up.
//
// DO NOT RUN this suite via `make test-unit`. Run serialised with
//   make test-integration
// Heavy GPU usage; do NOT run in parallel with other ModelTests.

import Foundation
import Testing
@testable import FFAI

@Suite("FishSpeech S2 Pro 8-bit integration", .serialized)
struct FishSpeechIntegrationTests {

    // Resolved HuggingFace snapshot directory (pre-cached).
    // Uses the standard HF hub layout: blobs are symlinked from snapshots/.
    private static let repoID = "mlx-community/fish-audio-s2-pro-8bit"

    @Test("load config + weights from cached checkpoint")
    func loadModel() async throws {
        let dir: URL
        do {
            dir = try await resolveSnapshotDirectory(repoID: Self.repoID)
        } catch {
            print("FishSpeech integration test skipped (checkpoint not cached): \(error)")
            return
        }

        // Load config
        let config = try ModelConfig.load(from: dir)
        #expect(config.modelType == "fish_qwen3_omni")

        // Load weights
        let weights = try SafeTensorsBundle(directory: dir)
        #expect(!weights.allKeys.isEmpty)

        // Verify key weight shapes are present.
        #expect(weights.has("model.norm.weight"))
        #expect(weights.has("model.fast_norm.weight"))
        #expect(weights.has("model.fast_output.weight"))

        // Build the model.
        let model = try FishSpeechModel.load(
            config: config,
            weights: weights,
            directory: dir,
            device: .shared
        )

        // Structural invariants.
        #expect(model.fishConfig.sampleRate == 44_100)
        #expect(model.fishConfig.numCodebooks == 10)
        #expect(model.fishConfig.textConfig.nLayer == 36)
        #expect(model.fishConfig.audioDecoderConfig.nLayer == 4)
        #expect(model.slowLayers.count == 36)
        #expect(model.fastLayers.count == 4)
        #expect(model.sampleRate == 44_100)

        // AudioModel conformance: the model is registered correctly.
        #expect(FishSpeech.modelTypes.contains(model.fishConfig.modelType))
    }

    @Test("synthesize throws codecNotAvailable (Stage-1 documented stub)")
    func synthesizeStage1() async throws {
        let dir: URL
        do {
            dir = try await resolveSnapshotDirectory(repoID: Self.repoID)
        } catch {
            print("FishSpeech Stage-1 test skipped (checkpoint not cached): \(error)")
            return
        }

        let config = try ModelConfig.load(from: dir)
        let weights = try SafeTensorsBundle(directory: dir)
        let model = try FishSpeechModel.load(
            config: config, weights: weights, directory: dir, device: .shared
        )

        // Stage-1: synthesize must throw codecNotAvailable, not any other error.
        // If it throws something unexpected the test fails, giving visibility
        // into regressions in the generate-codes path.
        do {
            _ = try model.synthesize(
                text: "Hello world.",
                parameters: AudioGenerationParameters(maxTokens: 32)
            )
            Issue.record("Expected synthesize to throw; it returned without error")
        } catch AudioGenerationError.codecNotAvailable {
            // Expected — FishS1DAC not yet ported.
        } catch {
            Issue.record("synthesize threw unexpected error: \(error)")
        }
    }

    @Test("generateCodes produces non-empty code frames for short text (Stage-1)")
    func generateCodesSmoke() async throws {
        let dir: URL
        do {
            dir = try await resolveSnapshotDirectory(repoID: Self.repoID)
        } catch {
            print("FishSpeech generateCodes test skipped (checkpoint not cached): \(error)")
            return
        }

        let config = try ModelConfig.load(from: dir)
        let weights = try SafeTensorsBundle(directory: dir)
        let model = try FishSpeechModel.load(
            config: config, weights: weights, directory: dir, device: .shared
        )

        // generateCodes is the pre-codec stage; it should complete without
        // throwing and return at least one frame.
        let codes = try model.generateCodes(
            text: "Hi.",
            maxTokens: 16,
            temperature: 0.0,  // greedy for determinism
            topP: 1.0,
            topK: 1,
            device: .shared
        )

        // codes is [numCodebooks][numFrames]; both dimensions > 0.
        #expect(codes.count == model.fishConfig.numCodebooks)
        #expect(codes.first?.isEmpty == false)
    }

    // ─── Helper ─────────────────────────────────────────────────────────

    /// Find the snapshot directory for a HuggingFace repo in the local cache.
    /// Searches `$HF_HOME/hub` → `~/.cache/huggingface/hub` for the
    /// standard `models--<org>--<repo>/snapshots/<hash>/` layout.
    private func resolveSnapshotDirectory(repoID: String) async throws -> URL {
        let cacheRoot: URL
        if let hfHome = ProcessInfo.processInfo.environment["HF_HOME"] {
            cacheRoot = URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        } else {
            cacheRoot = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/huggingface/hub")
        }

        // Convert "org/repo" → "models--org--repo"
        let folderName = "models--" + repoID.replacingOccurrences(of: "/", with: "--")
        let snapshotsDir = cacheRoot
            .appendingPathComponent(folderName)
            .appendingPathComponent("snapshots")

        guard FileManager.default.fileExists(atPath: snapshotsDir.path) else {
            throw CocoaError(.fileNoSuchFile,
                             userInfo: [NSFilePathErrorKey: snapshotsDir.path])
        }

        let hashes = try FileManager.default.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.isDirectoryKey]
        ).filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true }

        guard let snap = hashes.first else {
            throw CocoaError(.fileNoSuchFile,
                             userInfo: [NSFilePathErrorKey: snapshotsDir.path])
        }
        return snap
    }
}
