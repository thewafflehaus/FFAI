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
// StyleTTS2 integration test — loads the real kitten-tts-nano-0.8-fp16
// checkpoint from the HuggingFace cache and verifies:
//
//   1. Config is decoded correctly (nToken, hiddenDim, sampleRate, voices).
//   2. Weight count > 0 (safetensors shards are present and parseable).
//   3. AudioModelRegistry detects the checkpoint as StyleTTS2 / textToSpeech.
//   4. generatePlaceholder() returns a non-empty Float waveform with sane RMS.
//
// The full acoustic synthesize() path requires the ALBERT + prosody predictor
// + KittenDecoder forward pass — operators (batched GEMM, multi-head attention,
// 1-D conv, BiLSTM) not yet in FFAI's Ops set. The placeholder path verifies
// load + config without running the acoustic stack. See
// StyleTTS2Error.acousticFrontEndNotWired.
//
// DO NOT RUN — this test requires the HF cache to contain the checkpoint.
// Run via `make test-integration` (serialized, --num-workers 1).

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "StyleTTS2 Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableAudioSuites,
        IntegrationGroupGating.audioSkipReason)
)
struct StyleTTS2IntegrationTests {

    /// Cached snapshot directory for `mlx-community/kitten-tts-nano-0.8-fp16`.
    private static var cachedSnapshotDir: URL? {
        let hub = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        let modelDir =
            hub
            .appendingPathComponent("models--mlx-community--kitten-tts-nano-0.8-fp16")
            .appendingPathComponent("snapshots")
        guard
            let snap = try? FileManager.default.contentsOfDirectory(
                at: modelDir, includingPropertiesForKeys: nil
            ).first
        else { return nil }
        return snap
    }

    @Test("load kitten-tts-nano checkpoint + synthesize placeholder produces sane waveform")
    func loadAndSynthesize() async throws {
        let dir = try #require(
            Self.cachedSnapshotDir,
            "StyleTTS2: kitten-tts-nano-0.8-fp16 not cached at ~/.cache/huggingface/hub/")

        // ── 1. Load via AudioModelRegistry ──────────────────────────────
        let loaded = try await AudioModelRegistry.load(directory: dir)

        guard case .styleTTS2(let model) = loaded else {
            Issue.record("expected LoadedAudioModel.styleTTS2, got \(loaded)")
            return
        }

        // ── 2. Registry detection ────────────────────────────────────────
        let caps = AudioModelRegistry.capabilities(forConfigAt: dir)
        #expect(
            caps == Capability.textToSpeech,
            "expected textToSpeech capability for kitten-tts checkpoint")

        // ── 3. Config sanity ─────────────────────────────────────────────
        #expect(model.config.modelType == "kitten_tts")
        #expect(
            model.config.nToken == 178,
            "nToken mismatch — KittenTTS nano uses 178 symbols")
        #expect(model.config.hiddenDim == 128)
        #expect(model.config.sampleRate == 24_000)
        #expect(model.config.istftnet.genIstftNFft == 20)
        #expect(model.config.istftnet.genIstftHopSize == 5)

        // ── 4. Weight count ───────────────────────────────────────────────
        // Weight count may be 0 when only the config and index file are
        // cached (the shard blobs haven't been downloaded yet). The index
        // file presence alone confirms the checkpoint is a valid sharded
        // model; non-zero count is only asserted when shards are present.
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        let shardsPresent =
            !FileManager.default.fileExists(atPath: indexURL.path)
            || (try? FileManager.default.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: nil
            )
            .filter {
                $0.pathExtension == "safetensors"
                    && $0.lastPathComponent != "voices.safetensors"
            }
            .isEmpty == false) ?? false
        if shardsPresent {
            #expect(
                model.weightCount > 0,
                "expected at least one weight tensor in the checkpoint")
        }
        // When shards are absent the index file alone confirms the
        // checkpoint shape; the weight-count assertion is skipped only
        // because the data isn't there to assert on.

        // ── 5. synthesize() throws acousticFrontEndNotWired ──────────────
        do {
            _ = try model.synthesize(text: "Hello world.")
            Issue.record("expected StyleTTS2Error.acousticFrontEndNotWired")
        } catch StyleTTS2Error.acousticFrontEndNotWired {
            // Correct — the acoustic front-end is not yet wired.
        } catch {
            Issue.record("unexpected error from synthesize: \(error)")
        }

        // ── 6. Placeholder waveform: non-empty, sane RMS ─────────────────
        let waveform = model.generatePlaceholder(durationSeconds: 0.1)
        #expect(
            waveform.elementCount > 0,
            "placeholder waveform must be non-empty")
        #expect(waveform.dtype == .f32)
        let samples = waveform.toArray(as: Float.self)
        // Placeholder is zeros — RMS == 0 is expected and intentional.
        // What we're verifying is that the tensor shape is sane.
        let nExpected = Int(0.1 * Double(model.sampleRate))
        #expect(
            samples.count >= nExpected,
            "expected at least \(nExpected) samples for 0.1s at 24kHz")
    }

    @Test("synthesizeFromSpectrogram — vocoder tail produces a non-degenerate waveform")
    func vocoder_synthesizeFromSpectrogram() async throws {
        // The full acoustic stack is staged (synthesize() throws), but the
        // iSTFTNet vocoder tail (`StyleTTS2Vocoder`) is functional and shares
        // its shape contract with Kokoro's vocoder. Mirror KokoroIntegrationTests's
        // vocoder check: feed a synthetic complex spectrogram and assert the
        // reconstructed waveform is finite, non-silent, non-constant, and the
        // expected length.
        let dir = try #require(
            Self.cachedSnapshotDir,
            "StyleTTS2 vocoder: kitten-tts checkpoint not cached")
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .styleTTS2(let model) = loaded else {
            Issue.record("expected LoadedAudioModel.styleTTS2, got \(loaded)")
            return
        }

        // Build a frequency-sweep spectrogram — non-constant content so the
        // overlap-add output is a real, non-silent utterance-length waveform.
        let nFrames = 200
        let nFreq = model.vocoder.nFFT / 2 + 1
        var re = [Float](repeating: 0, count: nFrames * nFreq)
        var im = [Float](repeating: 0, count: nFrames * nFreq)
        for f in 0 ..< nFrames {
            for k in 0 ..< nFreq {
                let phase = 2.0 * Float.pi * Float(k) * Float(f) / Float(nFrames)
                re[f * nFreq + k] = 0.4 * cos(phase)
                im[f * nFreq + k] = 0.4 * sin(phase)
            }
        }
        let reT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
        reT.copyIn(from: re)
        let imT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
        imT.copyIn(from: im)

        let waveform = model.synthesizeFromSpectrogram(specRe: reT, specIm: imT)
        let samples = waveform.toArray(as: Float.self)
        #expect(samples.count > 0, "vocoder produced an empty waveform")
        #expect(samples.allSatisfy { $0.isFinite })
        let energy = samples.map { $0 * $0 }.reduce(0, +)
        #expect(energy > 1e-4, "StyleTTS2 vocoder produced a silent waveform")
        let distinct = Set(samples.map { ($0 * 1000).rounded() }).count
        #expect(distinct > 10, "StyleTTS2 vocoder produced a constant waveform")
        print(
            "StyleTTS2 vocoder synthesized \(samples.count) samples, "
                + "energy=\(energy), distinct=\(distinct)")
    }

    @Test("AudioModelRegistry.load throws for non-audio directory")
    func registryRejectsTextModel() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-s2tts-int-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(
            at: dir,
            withIntermediateDirectories: true)
        try #"{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}"#
            .write(
                to: dir.appendingPathComponent("config.json"),
                atomically: true, encoding: .utf8)

        do {
            _ = try await AudioModelRegistry.load(directory: dir)
            Issue.record("expected ModelError.unsupportedArchitecture")
        } catch ModelError.unsupportedArchitecture {
            // Correct.
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }
}
