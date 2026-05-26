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
// LlamaTTSTests — unit tests for the LlamaTTS / Orpheus TTS family:
// OrpheusTokens constants, config parsing, family detection, registry
// routing, the SNAC code de-interleaver, and the staged-synthesis error
// surface.
//
// Validates:
//   * OrpheusTokens constants match the published values.
//   * LlamaTTSModel.handles(_:) accepts canonical model_types,
//     architectures, and structural fallback (Llama + audio vocab +
//     sample_rate).
//   * AudioModelRegistry routes LlamaTTS configs to Capability.textToSpeech.
//   * LlamaTTSModel.deinterleaveSNACCodes turns 7-token frames into 3
//     SNAC code planes with the expected per-position offsets.
//   * LlamaTTSError descriptions render the family prefix.

import Foundation
import Testing
@testable import FFAI

@Suite("LlamaTTS")
struct LlamaTTSTests {

    // ─── OrpheusTokens constants ─────────────────────────────────────────

    @Test("OrpheusTokens — published special-token ids match Orpheus reference")
    func orpheusTokens() {
        #expect(OrpheusTokens.startOfHuman == 128_259)
        #expect(OrpheusTokens.endOfHuman == 128_260)
        #expect(OrpheusTokens.endOfText == 128_009)
        #expect(OrpheusTokens.startOfSpeech == 128_257)
        #expect(OrpheusTokens.endOfSpeech == 128_258)
        #expect(OrpheusTokens.padToken == 128_263)
        #expect(OrpheusTokens.audioStart == 128_261)
        #expect(OrpheusTokens.audioEnd == 128_262)
        #expect(OrpheusTokens.audioTokenOffset == 128_266)
        #expect(OrpheusTokens.snacCodebookStride == 4096)
    }

    // ─── Config decoding ─────────────────────────────────────────────────

    @Test("LlamaTTSConfig — default sample rate is 24 kHz")
    func configDefaultSampleRate() {
        let cfg = LlamaTTSConfig()
        #expect(cfg.sampleRate == 24_000)
    }

    @Test("LlamaTTSConfig.from — uses config sample_rate when present")
    func configReadsSampleRate() {
        let raw: [String: Any] = ["sample_rate": 22_050]
        let config = ModelConfig(architecture: nil, modelType: "orpheus", raw: raw)
        let cfg = LlamaTTSConfig.from(config)
        #expect(cfg.sampleRate == 22_050)
    }

    // ─── Registry detection ──────────────────────────────────────────────

    @Test("LlamaTTSModel.modelTypes — contains llama_tts and orpheus")
    func modelTypesContents() {
        let types = LlamaTTSModel.modelTypes
        #expect(types.contains("llama_tts"))
        #expect(types.contains("orpheus"))
    }

    @Test("LlamaTTSModel.handles — true for orpheus model_type")
    func handlesByModelType() {
        let config = ModelConfig(architecture: nil, modelType: "orpheus",
                                 raw: ["model_type": "orpheus"])
        #expect(LlamaTTSModel.handles(config))
    }

    @Test("LlamaTTSModel.handles — true for OrpheusForConditionalGeneration arch")
    func handlesByArchitecture() {
        let config = ModelConfig(
            architecture: "OrpheusForConditionalGeneration",
            modelType: nil, raw: [:])
        #expect(LlamaTTSModel.handles(config))
    }

    @Test("LlamaTTSModel.handles — Llama base with audio vocab + sample_rate is accepted")
    func handlesStructuralLlamaAudio() {
        let raw: [String: Any] = [
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 156_940,  // > audioTokenOffset (128_266)
            "sample_rate": 24_000,
        ]
        let config = ModelConfig(architecture: "LlamaForCausalLM",
                                 modelType: "llama", raw: raw)
        #expect(LlamaTTSModel.handles(config))
    }

    @Test("LlamaTTSModel.handles — plain text Llama (no audio vocab) is rejected")
    func handlesFalseForPlainLlama() {
        let raw: [String: Any] = [
            "model_type": "llama",
            "vocab_size": 32_000,
        ]
        let config = ModelConfig(architecture: "LlamaForCausalLM",
                                 modelType: "llama", raw: raw)
        #expect(!LlamaTTSModel.handles(config))
    }

    // ─── AudioModelRegistry routing ──────────────────────────────────────

    @Test("AudioModelRegistry.capabilities — LlamaTTS maps to textToSpeech")
    func registryCapability() {
        let config = ModelConfig(architecture: nil, modelType: "orpheus",
                                 raw: ["model_type": "orpheus"])
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.textToSpeech)
    }

    // ─── SNAC code de-interleave ────────────────────────────────────────

    @Test("deinterleaveSNACCodes — empty input yields three empty planes")
    func deinterleaveEmpty() {
        let planes = LlamaTTSModel.deinterleaveSNACCodes([])
        #expect(planes.count == 3)
        #expect(planes[0].isEmpty)
        #expect(planes[1].isEmpty)
        #expect(planes[2].isEmpty)
    }

    @Test("deinterleaveSNACCodes — one full frame produces layer counts (1, 2, 4)")
    func deinterleaveOneFrame() {
        let stride = OrpheusTokens.snacCodebookStride
        // Token at position k is `k * stride` so each layer subtracts its
        // offset and ends up at zero (per-position offset round-trip).
        let frame = (0..<7).map { $0 * stride }
        let planes = LlamaTTSModel.deinterleaveSNACCodes(frame)
        #expect(planes[0].count == 1)
        #expect(planes[1].count == 2)
        #expect(planes[2].count == 4)
        // Each entry decoded to zero after subtracting its per-position offset.
        for layer in planes {
            for v in layer { #expect(v == 0) }
        }
    }

    @Test("deinterleaveSNACCodes — partial trailing frame is dropped")
    func deinterleavePartialFrameDropped() {
        // 9 tokens = 1 full frame (7) + 2 leftover.
        let tokens = Array(0..<9)
        let planes = LlamaTTSModel.deinterleaveSNACCodes(tokens)
        #expect(planes[0].count == 1)
        #expect(planes[1].count == 2)
        #expect(planes[2].count == 4)
    }

    // ─── Error stringification ───────────────────────────────────────────

    @Test("LlamaTTSError.description — codecUnavailable mentions LlamaTTS")
    func errorDescriptionCodec() {
        let err = LlamaTTSError.codecUnavailable
        let desc = err.description
        #expect(desc.contains("LlamaTTS") || desc.contains("SNAC"))
    }

    @Test("LlamaTTSError.description — noAudioCodes mentions audio")
    func errorDescriptionNoAudio() {
        let err = LlamaTTSError.noAudioCodes
        let desc = err.description
        #expect(desc.contains("LlamaTTS") || desc.contains("audio"))
    }

    @Test("LlamaTTSError.description — missingConfig mentions config")
    func errorDescriptionMissingConfig() {
        let err = LlamaTTSError.missingConfig
        let desc = err.description
        #expect(desc.contains("LlamaTTS") || desc.contains("config"))
    }
}
