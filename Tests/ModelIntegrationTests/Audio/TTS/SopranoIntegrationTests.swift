// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Integration test: loads a real Soprano checkpoint from the HF cache
// and exercises config decoding + registry detection + end-to-end synthesis.
//
// A load failure FAILS the suite — `loadSoprano80M()` is `throws` and the
// checkpoint is a hard requirement, not a "skip if missing".
//
// This suite verifies:
//   1. A real Soprano-80M checkpoint loads without error.
//   2. The config decodes transformer and decoder hyper-parameters correctly.
//   3. The audio registry routes the directory to .soprano.
//   4. `synthesize("Hello world.")` returns a non-empty Float waveform with
//      a sane RMS level (proves end-to-end codepath: LLM + Vocos decoder).
//   5. A Soprano-1.1 checkpoint (LLM-only) loads and throws
//      SopranoError.decoderNotAvailable on synthesize.
//
// DO NOT RUN this suite via `make test-integration` during CI — the
// checkpoints are multi-GB and require an ML-capable Mac.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Soprano Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct SopranoIntegrationTests {

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

    /// Resolve a Soprano-80M checkpoint from the HF cache.
    ///
    /// Tries `mlx-community/Soprano-80M-4bit` from the local cache first,
    /// then falls back to a HF download.
    private func resolveSoprano80MCheckpoint() async throws -> URL {
        let root = Self.hfCacheRoot
        let candidates = [
            "models--mlx-community--Soprano-80M-bf16"
        ]
        let fm = FileManager.default
        for slug in candidates {
            let snapshots = root.appendingPathComponent(slug)
                .appendingPathComponent("snapshots")
            guard
                let subs = try? fm.contentsOfDirectory(
                    at: snapshots, includingPropertiesForKeys: nil)
            else { continue }
            if let dir = subs.first(where: { Self.isCompleteSnapshot($0) }) {
                return dir
            }
        }
        let locator = ModelLocator(downloader: ModelDownloader())
        return try await locator.resolve(idOrPath: "mlx-community/Soprano-80M-4bit")
    }

    /// Resolve a Soprano-1.1 (LLM-only) checkpoint from the HF cache.
    private func resolveSoprano11Checkpoint() async throws -> URL {
        let root = Self.hfCacheRoot
        let candidates = [
            "models--mlx-community--Soprano-1.1-80M-bf16"
        ]
        let fm = FileManager.default
        for slug in candidates {
            let snapshots = root.appendingPathComponent(slug)
                .appendingPathComponent("snapshots")
            guard
                let subs = try? fm.contentsOfDirectory(
                    at: snapshots, includingPropertiesForKeys: nil)
            else { continue }
            if let dir = subs.first(where: { Self.isCompleteSnapshot($0) }) {
                return dir
            }
        }
        let locator = ModelLocator(downloader: ModelDownloader())
        return try await locator.resolve(idOrPath: "ekryski/Soprano-1.1-80M-4bit")
    }

    // ─── Load helpers ─────────────────────────────────────────────────

    private func loadSoprano80M() async throws -> SopranoModel {
        let dir = try await resolveSoprano80MCheckpoint()
        return try await SopranoModel.load(directory: dir)
    }

    private func loadSoprano11() async throws -> SopranoModel {
        let dir = try await resolveSoprano11Checkpoint()
        return try await SopranoModel.load(directory: dir)
    }

    // ─── Tests ────────────────────────────────────────────────────────

    @Test("load — Soprano-80M config decodes transformer fields from real checkpoint")
    func load_decodesTransformerConfig() async throws {
        let model = try await loadSoprano80M()
        #expect(model.config.hiddenSize > 0)
        #expect(model.config.numHiddenLayers > 0)
        #expect(model.config.numAttentionHeads > 0)
        #expect(model.config.headDim > 0)
        #expect(model.config.vocabSize > 0)
        #expect(model.config.intermediateSize > 0)
        print(
            "[Soprano-80M] Loaded: hiddenSize=\(model.config.hiddenSize), "
                + "numLayers=\(model.config.numHiddenLayers), "
                + "heads=\(model.config.numAttentionHeads), "
                + "kvHeads=\(model.config.numKeyValueHeads), "
                + "headDim=\(model.config.headDim), "
                + "vocab=\(model.config.vocabSize)")
    }

    @Test("load — Soprano-80M config decodes decoder hyper-parameters")
    func load_decodesDecoderConfig() async throws {
        let model = try await loadSoprano80M()
        #expect(model.config.hasDecoderConfig)
        #expect((model.config.decoderDim ?? 0) > 0)
        #expect((model.config.decoderNumLayers ?? 0) > 0)
        #expect((model.config.hopLength ?? 0) > 0)
        #expect((model.config.nFft ?? 0) > 0)
        #expect((model.config.upscale ?? 0) > 0)
        #expect(model.decoder != nil)
        print(
            "[Soprano-80M] Decoder: dim=\(model.config.decoderDim ?? -1), "
                + "layers=\(model.config.decoderNumLayers ?? -1), "
                + "hopLength=\(model.config.hopLength ?? -1), "
                + "nFFT=\(model.config.nFft ?? -1), "
                + "upscale=\(model.config.upscale ?? -1)")
    }

    @Test("load — Soprano-80M sample rate is 32 000 Hz")
    func load_sampleRate32kHz() async throws {
        let model = try await loadSoprano80M()
        #expect(model.sampleRate == 32_000)
    }

    @Test("registry — AudioModelRegistry routes Soprano-80M checkpoint to .soprano")
    func registry_routesSoprano80M() async throws {
        let dir = try await resolveSoprano80MCheckpoint()
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .soprano = loaded else {
            Issue.record("AudioModelRegistry did not route to .soprano; got \(loaded)")
            return
        }
        #expect(loaded.capabilities == Capability.textToSpeech)
        print("[Soprano-80M] Registry routed correctly, capabilities=\(loaded.capabilities)")
    }

    @Test("synthesize — Soprano-80M produces non-empty waveform with sane RMS for 'Hello world.'")
    func synthesize_producesWaveform() async throws {
        let model = try await loadSoprano80M()
        let samples = try model.synthesize(
            text: "Hello world.",
            parameters: AudioGenerationParameters(
                maxTokens: 256,
                temperature: 0.7,
                topP: 0.7))

        // Basic sanity — non-empty output.
        #expect(
            !samples.isEmpty,
            "synthesize returned an empty waveform for 'Hello world.'")

        // Minimum duration: expect at least ~0.1 s of audio at 32 kHz.
        let minSamples = model.sampleRate / 10
        #expect(
            samples.count >= minSamples,
            "waveform too short: \(samples.count) samples (expected ≥ \(minSamples))")

        // RMS sanity — audio should not be silence (RMS > 1e-4) and not clipped (RMS < 0.9).
        let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))
        #expect(rms > 1e-4, "waveform RMS (\(rms)) too low — possible silence")
        #expect(rms < 0.9, "waveform RMS (\(rms)) too high — possible clipping")

        print(
            "[Soprano-80M] synthesize: samples=\(samples.count), "
                + "duration=\(String(format: "%.2f", Double(samples.count) / Double(model.sampleRate)))s, "
                + "rms=\(String(format: "%.4f", rms))")
    }

    @Test("load — Soprano-1.1 loads without error (LLM-only checkpoint)")
    func load_soprano11LoadsWithoutError() async throws {
        let model = try await loadSoprano11()
        // Soprano-1.1 has no decoder in the checkpoint.
        #expect(model.config.hasDecoderConfig == false)
        #expect(model.decoder == nil)
        // Transformer fields must still decode.
        #expect(model.config.hiddenSize > 0)
        #expect(model.config.numHiddenLayers > 0)
        print(
            "[Soprano-1.1] Loaded: hiddenSize=\(model.config.hiddenSize), "
                + "numLayers=\(model.config.numHiddenLayers), "
                + "hasDecoderConfig=\(model.config.hasDecoderConfig)")
    }

    @Test("registry — AudioModelRegistry routes Soprano-1.1 checkpoint to .soprano")
    func registry_routesSoprano11() async throws {
        let dir = try await resolveSoprano11Checkpoint()
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .soprano = loaded else {
            Issue.record("AudioModelRegistry did not route to .soprano; got \(loaded)")
            return
        }
        #expect(loaded.capabilities == Capability.textToSpeech)
    }

    @Test("synthesize — Soprano-1.1 throws decoderNotAvailable (LLM-only checkpoint)")
    func synthesize_soprano11ThrowsDecoderNotAvailable() async throws {
        let model = try await loadSoprano11()
        #expect(throws: SopranoError.self) {
            _ = try model.synthesize(text: "Hello world.")
        }
        // Verify it's specifically decoderNotAvailable.
        do {
            _ = try model.synthesize(text: "Hello world.")
        } catch SopranoError.decoderNotAvailable {
            // Expected path — pass.
        } catch {
            Issue.record("Expected SopranoError.decoderNotAvailable but got: \(error)")
        }
    }
}
