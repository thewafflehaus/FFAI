// Integration test for MossFormer2SE speech enhancement.
//
// Requires the `starkdmi/MossFormer2-SE-fp16` checkpoint to be present in the
// HF cache (~/.cache/huggingface/hub/models--starkdmi--MossFormer2-SE-fp16)
// with at least one .safetensors weight file. The config-only blob already
// present in the cache is enough for registry + config tests; the full
// enhance() test requires the weights.
//
// Tests verified here:
//   1. Config loads and decodes correct values from the cached checkpoint.
//   2. AudioModelRegistry routes the checkpoint directory to .mossFormer2SE.
//   3. enhance() on a 1-second synthetic 48 kHz waveform returns a non-empty
//      output with the same length as the input (within ±hopLength samples).
//
// DO NOT RUN this suite via `make test-integration` until weights are cached.

import Foundation
import Testing
@testable import FFAI

@Suite("MossFormer2SE Integration", .serialized)
struct MossFormer2SEIntegrationTests {

    // ─── Checkpoint resolution ────────────────────────────────────────

    private static var hfCacheRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    /// Resolve the cached MossFormer2-SE checkpoint snapshot directory.
    /// Returns nil when no snapshot is found (skips tests gracefully).
    private static func resolveCheckpoint() -> URL? {
        let root = hfCacheRoot
        let slug = "models--starkdmi--MossFormer2-SE-fp16"
        let snapshotsDir = root.appendingPathComponent(slug)
            .appendingPathComponent("snapshots")
        guard let subs = try? FileManager.default.contentsOfDirectory(
            at: snapshotsDir, includingPropertiesForKeys: nil)
        else { return nil }
        // Return the first snapshot directory that contains config.json.
        return subs.first {
            FileManager.default.fileExists(
                atPath: $0.appendingPathComponent("config.json").path)
        }
    }

    /// True when `dir` contains at least one .safetensors weight file.
    private static func hasWeights(_ dir: URL) -> Bool {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            atPath: dir.path)
        else { return false }
        return entries.contains { $0.hasSuffix(".safetensors") }
    }

    // ─── Synthetic waveform fixture ───────────────────────────────────

    /// A 1-second 48 kHz mono sine-wave waveform at 440 Hz.
    /// Used as a stand-in for clean speech when no WAV fixture is available.
    private static func syntheticWaveform(
        sampleRate: Int = 48_000,
        durationSeconds: Float = 1.0,
        frequency: Float = 440.0
    ) -> [Float] {
        let numSamples = Int(Float(sampleRate) * durationSeconds)
        return (0..<numSamples).map { i -> Float in
            0.5 * sin(2.0 * Float.pi * frequency * Float(i) / Float(sampleRate))
        }
    }

    // ─── Tests ───────────────────────────────────────────────────────

    @Test("config — checkpoint config.json decodes expected fields")
    func configDecodesFromCheckpoint() throws {
        let snap = try #require(Self.resolveCheckpoint(),
                                "MossFormer2-SE checkpoint not cached locally")
        let modelConfig = try ModelConfig.load(from: snap)
        let se = MossFormer2SEConfig.from(modelConfig)
        #expect(se.sampleRate == 48_000)
        #expect(se.inChannels == 180)
        #expect(se.outChannels == 512)
        #expect(se.outChannelsFinal == 961)
        #expect(se.numMels == 60)
        #expect(3 * se.numMels == se.inChannels)
        print("[MossFormer2SE] Config decoded: sr=\(se.sampleRate), "
              + "inC=\(se.inChannels), outC=\(se.outChannels), "
              + "outCF=\(se.outChannelsFinal), nBlocks=\(se.numBlocks)")
    }

    @Test("registry — AudioModelRegistry routes checkpoint to .mossFormer2SE (config only)")
    func registryRoutesFromConfig() throws {
        let snap = try #require(Self.resolveCheckpoint(),
                                "MossFormer2-SE checkpoint not cached locally")
        let modelConfig = try ModelConfig.load(from: snap)
        // Detection is config-only; no weights needed.
        #expect(MossFormer2SEModel.handles(modelConfig))
        #expect(AudioModelRegistry.handles(modelConfig))
        #expect(AudioModelRegistry.capabilities(for: modelConfig) == Capability.speechToSpeech)
        print("[MossFormer2SE] Registry detection: OK, capabilities=\(Capability.speechToSpeech)")
    }

    @Test("enhance — produces non-empty output matching input length")
    func enhanceProducesMatchingLengthOutput() throws {
        let snap = try #require(Self.resolveCheckpoint(),
                                "MossFormer2-SE checkpoint not cached locally")
        try #require(Self.hasWeights(snap),
                     "MossFormer2-SE snapshot is missing .safetensors weights")

        let model = try MossFormer2SEModel.load(directory: snap)
        let waveform = Self.syntheticWaveform(sampleRate: model.sampleRate)

        let enhanced = try model.enhance(waveform: waveform)

        // Output must be non-empty.
        #expect(!enhanced.isEmpty)

        // Output length must be within ±hopLength samples of input length.
        let hopLength = model.config.winInc
        let lengthDelta = abs(enhanced.count - waveform.count)
        #expect(lengthDelta <= hopLength)

        // Output values must be finite (no NaN or Inf).
        let allFinite = enhanced.allSatisfy { $0.isFinite }
        #expect(allFinite)

        print("[MossFormer2SE] enhance: in=\(waveform.count) out=\(enhanced.count) "
              + "finite=\(allFinite) "
              + "maxAbs=\(enhanced.map { abs($0) }.max() ?? 0)")
    }
}
