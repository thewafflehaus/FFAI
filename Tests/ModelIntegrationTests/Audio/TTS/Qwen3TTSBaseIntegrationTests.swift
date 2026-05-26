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
// Integration test: loads a real Qwen3TTSBase checkpoint from the HF
// cache and exercises the config decode, family detection, audio-code
// generation, and SNAC codec-unavailable staged-port behaviour.
//
// DO NOT RUN this suite directly — `ModelTests` loads multi-GB HuggingFace
// snapshots and must be serialized. Use:
//
//   make test-integration
//
// A load failure FAILS the suite. A missing checkpoint surfaces as a
// thrown error so the test fails rather than silently passing.
//
// The SNAC decoder is a separate port; this build verifies the model
// loads and `generateCodes` runs, while `synthesize` throws the typed
// `Qwen3TTSBaseError.codecUnavailable` by design.
//
// ## Checkpoint
//
// Primary: `mlx-community/VyvoTTS-EN-Beta-4bit`
// A 4-bit quantized Qwen3 LLM backbone trained with the VyvoTTS protocol.
// The Qwen3Dense loader handles the quantized weight layout transparently.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3TTSBase Integration", .serialized)
struct Qwen3TTSBaseIntegrationTests {

    /// Resolve and load the VyvoTTS checkpoint. Throws on failure so a
    /// missing checkpoint fails the test instead of skipping it.
    private func loadModel() async throws -> Qwen3TTSBaseModel {
        let locator = ModelLocator()
        let dir = try await locator.resolve(
            idOrPath: "mlx-community/VyvoTTS-EN-Beta-4bit")
        return try await Qwen3TTSBaseModel.load(directory: dir)
    }

    @Test("load — VyvoTTS checkpoint binds the Qwen3 backbone")
    func load_bindsQwen3Backbone() async throws {
        let model = try await loadModel()
        // The backbone is a Qwen3 dense transformer.
        #expect(model.backbone.nLayers > 0)
        // Extended vocabulary must exceed the base Qwen3 vocab.
        #expect(model.backbone.vocab > 151_936)
        // 24 kHz audio output.
        #expect(model.sampleRate == 24_000)
    }

    @Test("promptTokens — VyvoTTS framing is well-formed")
    func promptTokens_framing() async throws {
        let model = try await loadModel()
        let ids = model.promptTokens(text: "Hello there.", voice: "en-us-1")
        // Frame: [SOH] (voice: ) text [EOT][EOH]
        #expect(ids.first == Qwen3TTSBaseTokens.startOfHuman)
        #expect(ids[ids.count - 2] == Qwen3TTSBaseTokens.endOfText)
        #expect(ids.last == Qwen3TTSBaseTokens.endOfHuman)
        #expect(ids.count > 4)
    }

    @Test("promptTokens — voice prefix is incorporated into the token sequence")
    func promptTokens_voicePrefix() async throws {
        let model = try await loadModel()
        let withVoice    = model.promptTokens(text: "Hi.", voice: "en-us-1")
        let withoutVoice = model.promptTokens(text: "Hi.")
        // The voiced sequence should be longer (prefix "en-us-1: " is extra).
        #expect(withVoice.count > withoutVoice.count)
    }

    @Test("registry — VyvoTTS routes through AudioModelRegistry")
    func registry_routesVyvoTTS() async throws {
        let locator = ModelLocator()
        let dir = try await locator.resolve(
            idOrPath: "mlx-community/VyvoTTS-EN-Beta-4bit")
        let loaded = try await AudioModelRegistry.load(directory: dir)
        guard case .qwen3TTSBase = loaded else {
            Issue.record("AudioModelRegistry did not route to Qwen3TTSBase")
            return
        }
        #expect(loaded.capabilities == Capability.textToSpeech)
    }

    @Test("generateCodes — greedy decode emits finite SNAC code planes")
    func generateCodes_emitsCodes() async throws {
        let model = try await loadModel()
        // Greedy decode, capped short for test runtime (16 SNAC frames).
        let planes = try model.generateCodes(
            text: "Hi.", voice: "en-us-1",
            maxFrames: 16, temperature: 0)
        #expect(planes.count == 3)
        // At least one complete SNAC frame was emitted.
        #expect(!planes[0].isEmpty, "Qwen3TTSBase produced no SNAC codes")
        // SNAC up-sampling: layer2 = 2×, layer3 = 4× layer1's length.
        #expect(planes[1].count == 2 * planes[0].count)
        #expect(planes[2].count == 4 * planes[0].count)
        print("Qwen3TTSBase generated \(planes[0].count) SNAC frames")
    }

    @Test("synthesize — staged port reports codec as unavailable")
    func synthesize_reportsCodecUnavailable() async throws {
        let model = try await loadModel()
        // The SNAC decoder is not wired — synthesize must throw the typed error.
        #expect(throws: Qwen3TTSBaseError.self) {
            _ = try model.synthesize(text: "Hello.")
        }
    }
}
