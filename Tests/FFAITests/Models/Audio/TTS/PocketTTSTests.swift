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
// Unit tests for the PocketTTS family (Kyutai flow-matching TTS).
//
// Covers:
//   • Config decoding — from canonical and minimal config.json payloads.
//   • Sub-config decoding: FlowLM, Transformer, LookupTable, Seanet, Mimi.
//   • Registry detection — model_type, structural paths, negative cases.
//   • AudioModelRegistry integration — handles + capabilities.
//   • Staged synthesis error — synthesize() throws PocketTTSError.synthesisNotWired.
//
// These tests are fast and offline — no checkpoint weights are loaded.

import Foundation
import Testing
@testable import FFAI

@Suite("PocketTTS")
struct PocketTTSTests {

    // ─── Helper: build a complete canonical config ────────────────────

    /// Mirrors the structure of `mlx-community/pocket-tts/config.json`.
    private static func canonicalRaw() -> [String: Any] {
        [
            "model_type": "pocket_tts",
            "flow_lm": [
                "dtype": "float32",
                "flow": ["dim": 512, "depth": 6],
                "transformer": [
                    "hidden_scale": 4,
                    "max_period": 10000.0,
                    "d_model": 1024,
                    "num_heads": 16,
                    "num_layers": 6,
                ],
                "lookup_table": [
                    "dim": 1024,
                    "n_bins": 4000,
                    "tokenizer": "sentencepiece",
                    "tokenizer_path": "hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model",
                ],
                "weights_path": nil as Any?,
            ] as [String: Any],
            "mimi": [
                "dtype": "float32",
                "sample_rate": 24000,
                "channels": 1,
                "frame_rate": 12.5,
                "seanet": [
                    "dimension": 512,
                    "channels": 1,
                    "n_filters": 64,
                    "n_residual_layers": 1,
                    "ratios": [6, 5, 4],
                    "kernel_size": 7,
                    "residual_kernel_size": 3,
                    "last_kernel_size": 3,
                    "dilation_base": 2,
                    "pad_mode": "constant",
                    "compress": 2,
                ] as [String: Any],
                "transformer": [
                    "d_model": 512,
                    "input_dimension": 512,
                    "output_dimensions": [512],
                    "num_heads": 8,
                    "num_layers": 2,
                    "layer_scale": 0.01,
                    "context": 250,
                    "dim_feedforward": 2048,
                    "max_period": 10000.0,
                ] as [String: Any],
                "quantizer": [
                    "dimension": 32,
                    "output_dimension": 512,
                ] as [String: Any],
                "weights_path": nil as Any?,
            ] as [String: Any],
            "weights_path": "hf://kyutai/pocket-tts/tts_b6369a24.safetensors",
            "weights_path_without_voice_cloning": "hf://kyutai/pocket-tts-without-voice-cloning/tts_b6369a24.safetensors",
            "model_path": nil as Any?,
        ]
    }

    // ─── Config decoding: top level ──────────────────────────────────

    @Test("PocketTTSConfig — decodes from canonical pocket_tts model_type")
    func configDecodesFromModelType() {
        let raw = Self.canonicalRaw()
        let config = ModelConfig(architecture: nil, modelType: "pocket_tts", raw: raw)
        let ptts = PocketTTSConfig.from(config)
        #expect(ptts != nil)
        #expect(ptts?.modelType == "pocket_tts")
        #expect(ptts?.sampleRate == 24_000)
    }

    @Test("PocketTTSConfig — decodes from minimal config with flow_lm + mimi only")
    func configDecodesFromMinimal() {
        let config = ModelConfig(
            architecture: nil, modelType: nil,
            raw: ["flow_lm": [:] as [String: Any], "mimi": [:] as [String: Any]])
        let ptts = PocketTTSConfig.from(config)
        // Structural detection: both blocks present but no model_type.
        #expect(ptts != nil)
        // Should fill in defaults.
        #expect(ptts?.mimi.sampleRate == 24_000)
        #expect(ptts?.mimi.frameRate == 12.5)
    }

    @Test("PocketTTSConfig — returns nil for unrelated configs")
    func configRejectsUnrelated() {
        let config = ModelConfig(
            architecture: nil, modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 4096])
        #expect(PocketTTSConfig.from(config) == nil)
    }

    @Test("PocketTTSConfig — weightsPath and weightsPathWithoutVoiceCloning decode")
    func configDecodesWeightsPaths() {
        let config = ModelConfig(architecture: nil, modelType: "pocket_tts", raw: Self.canonicalRaw())
        let ptts = PocketTTSConfig.from(config)!
        #expect(ptts.weightsPath?.contains("pocket-tts") == true)
        #expect(ptts.weightsPathWithoutVoiceCloning?.contains("without-voice-cloning") == true)
    }

    // ─── Flow LM sub-config ───────────────────────────────────────────

    @Test("PocketTTSFlowConfig — decodes dim and depth from canonical config")
    func flowConfigDecodes() {
        let raw: [String: Any] = ["dim": 512, "depth": 6]
        let flow = PocketTTSFlowConfig.from(raw)
        #expect(flow.dim == 512)
        #expect(flow.depth == 6)
    }

    @Test("PocketTTSFlowConfig — falls back to defaults for empty dict")
    func flowConfigDefaults() {
        let flow = PocketTTSFlowConfig.from([:])
        #expect(flow.dim == 512)
        #expect(flow.depth == 6)
    }

    @Test("PocketTTSTransformerConfig — decodes all fields from canonical config")
    func transformerConfigDecodes() {
        let raw: [String: Any] = [
            "hidden_scale": 4,
            "max_period": 10000.0,
            "d_model": 1024,
            "num_heads": 16,
            "num_layers": 6,
        ]
        let xf = PocketTTSTransformerConfig.from(raw)
        #expect(xf.hiddenScale == 4)
        #expect(xf.dModel == 1_024)
        #expect(xf.numHeads == 16)
        #expect(xf.numLayers == 6)
        #expect(xf.dimFeedforward == 4_096)   // 4 × 1024
        #expect(abs(xf.maxPeriod - 10_000) < 1)
    }

    @Test("PocketTTSTransformerConfig — dimFeedforward is hiddenScale × dModel")
    func transformerDimFeedforward() {
        let raw: [String: Any] = ["hidden_scale": 6, "d_model": 512]
        let xf = PocketTTSTransformerConfig.from(raw)
        #expect(xf.dimFeedforward == 3_072)
    }

    @Test("PocketTTSLookupTableConfig — decodes tokenizer fields")
    func lookupTableConfigDecodes() {
        let raw: [String: Any] = [
            "dim": 1024,
            "n_bins": 4000,
            "tokenizer": "sentencepiece",
            "tokenizer_path": "hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model",
        ]
        let lut = PocketTTSLookupTableConfig.from(raw)
        #expect(lut.dim == 1_024)
        #expect(lut.nBins == 4_000)
        #expect(lut.tokenizer == "sentencepiece")
        #expect(lut.tokenizerPath.contains("pocket-tts"))
    }

    // ─── Mimi sub-config ──────────────────────────────────────────────

    @Test("PocketTTSSeanetConfig — decodes ratios and strides from canonical config")
    func seanetConfigDecodes() {
        let raw: [String: Any] = [
            "dimension": 512,
            "channels": 1,
            "n_filters": 64,
            "n_residual_layers": 1,
            "ratios": [6, 5, 4],
            "kernel_size": 7,
            "residual_kernel_size": 3,
            "last_kernel_size": 3,
            "dilation_base": 2,
            "pad_mode": "constant",
            "compress": 2,
        ]
        let s = PocketTTSSeanetConfig.from(raw)
        #expect(s.dimension == 512)
        #expect(s.nFilters == 64)
        #expect(s.ratios == [6, 5, 4])
        // hopLength = 6 × 5 × 4 = 120.
        #expect(s.hopLength == 120)
        #expect(s.padMode == "constant")
    }

    @Test("PocketTTSQuantizerConfig — decodes dimension and outputDimension")
    func quantizerConfigDecodes() {
        let raw: [String: Any] = ["dimension": 32, "output_dimension": 512]
        let q = PocketTTSQuantizerConfig.from(raw)
        #expect(q.dimension == 32)
        #expect(q.outputDimension == 512)
    }

    @Test("PocketTTSMimiConfig — encoderFrameRate is sampleRate / hopLength")
    func mimiEncoderFrameRate() {
        let raw = (Self.canonicalRaw()["mimi"] as! [String: Any])
        let mimi = PocketTTSMimiConfig.from(raw)
        // 24000 / (6×5×4) = 24000 / 120 = 200.
        #expect(abs(mimi.encoderFrameRate - 200.0) < 0.01)
    }

    @Test("PocketTTSMimiConfig — decodes sampleRate and frameRate")
    func mimiConfigDecodes() {
        let raw = (Self.canonicalRaw()["mimi"] as! [String: Any])
        let mimi = PocketTTSMimiConfig.from(raw)
        #expect(mimi.sampleRate == 24_000)
        #expect(abs(mimi.frameRate - 12.5) < 0.001)
        #expect(mimi.transformer.numLayers == 2)
        #expect(mimi.transformer.numHeads == 8)
        #expect(mimi.quantizer.dimension == 32)
        #expect(mimi.quantizer.outputDimension == 512)
    }

    // ─── Registry detection ───────────────────────────────────────────

    @Test("PocketTTSModel.handles — detects from model_type pocket_tts")
    func handlesModelType() {
        let config = ModelConfig(
            architecture: nil, modelType: "pocket_tts",
            raw: ["model_type": "pocket_tts"])
        #expect(PocketTTSModel.handles(config))
    }

    @Test("PocketTTSModel.handles — detects structurally from flow_lm + mimi co-presence")
    func handlesStructural() {
        let config = ModelConfig(
            architecture: nil, modelType: nil,
            raw: [
                "flow_lm": ["transformer": [:] as [String: Any]] as [String: Any],
                "mimi": ["frame_rate": 12.5] as [String: Any],
            ])
        #expect(PocketTTSModel.handles(config))
    }

    @Test("PocketTTSModel.handles — does not detect LLM models")
    func doesNotDetectLlama() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama", "hidden_size": 4_096])
        #expect(!PocketTTSModel.handles(config))
    }

    @Test("PocketTTSModel.handles — does not detect MOSS-TTS")
    func doesNotDetectMossTTS() {
        let config = ModelConfig(
            architecture: "MossTTSDelayModel", modelType: "moss_tts",
            raw: ["model_type": "moss_tts", "n_vq": 32])
        #expect(!PocketTTSModel.handles(config))
    }

    @Test("PocketTTSModel.handles — does not detect models with only one sub-block")
    func doesNotDetectPartialStructural() {
        // Only flow_lm without mimi — not enough.
        let config = ModelConfig(
            architecture: nil, modelType: nil,
            raw: ["flow_lm": [:] as [String: Any]])
        #expect(!PocketTTSModel.handles(config))
    }

    // ─── AudioModelRegistry integration ──────────────────────────────

    @Test("AudioModelRegistry — handles PocketTTS from model_type")
    func registryHandlesPocketTTS() {
        let config = ModelConfig(
            architecture: nil, modelType: "pocket_tts",
            raw: ["model_type": "pocket_tts"])
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.textToSpeech)
    }

    @Test("AudioModelRegistry — does not handle unrelated models")
    func registryDoesNotHandleLlama() {
        let config = ModelConfig(
            architecture: "LlamaForCausalLM", modelType: "llama",
            raw: ["model_type": "llama"])
        #expect(!AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == nil)
    }

    @Test("AudioModelRegistry — capabilities for PocketTTS is textToSpeech")
    func registryCapabilities() {
        let config = ModelConfig(
            architecture: nil, modelType: "pocket_tts",
            raw: ["model_type": "pocket_tts"])
        let caps = AudioModelRegistry.capabilities(for: config)
        #expect(caps == Capability.textToSpeech)
        #expect(caps?.contains(.audioOut) == true)
        #expect(caps?.contains(.textIn) == true)
    }

    // ─── Staged synthesis error ───────────────────────────────────────

    @Test("PocketTTSError — synthesisNotWired description mentions flow LM and Mimi")
    func errorDescriptionContainsKeyTerms() {
        let err = PocketTTSError.synthesisNotWired
        let desc = err.description
        #expect(desc.contains("flow LM") || desc.contains("flow"))
        #expect(desc.contains("Mimi") || desc.contains("codec"))
        #expect(desc.contains("stage") || desc.contains("Stage"))
    }

    @Test("PocketTTSError — missingConfig carries field name")
    func missingConfigError() {
        let err = PocketTTSError.missingConfig("flow_lm or mimi")
        #expect(err.description.contains("flow_lm"))
    }

    @Test("PocketTTSError — missingVoice carries voice name")
    func missingVoiceError() {
        let err = PocketTTSError.missingVoice("alba")
        #expect(err.description.contains("alba"))
    }

    @Test("PocketTTSModel — synthesize throws synthesisNotWired (offline)")
    func synthesizeThrowsSynthesisNotWired() {
        // Build a minimal in-memory config — no weights needed to test the
        // staged error path.
        let raw = Self.canonicalRaw()
        let modelConfig = ModelConfig(architecture: nil, modelType: "pocket_tts", raw: raw)
        guard let config = PocketTTSConfig.from(modelConfig) else {
            Issue.record("PocketTTSConfig.from returned nil for canonical config")
            return
        }

        // Build a stub SafeTensorsBundle from a temp dir with an empty .safetensors.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-ptts-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            // Minimal valid safetensors: 8-byte header length + empty JSON.
            let header = "{}"
            let headerBytes = Array(header.utf8)
            var headerLen = UInt64(headerBytes.count)
            var data = Data()
            withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
            data.append(contentsOf: headerBytes)
            try data.write(to: dir.appendingPathComponent("model.safetensors"))
        } catch {
            Issue.record("Failed to write stub safetensors: \(error)")
            return
        }

        do {
            let bundle = try SafeTensorsBundle(directory: dir)
            let model = PocketTTSModel(config: config, weights: bundle)
            #expect(throws: PocketTTSError.self) {
                _ = try model.synthesize(text: "Hello, PocketTTS.")
            }
        } catch {
            Issue.record("Unexpected error in bundle/model construction: \(error)")
        }
    }

    // ─── sampleRate convenience ───────────────────────────────────────

    @Test("PocketTTSModel — sampleRate reflects mimi.sampleRate")
    func sampleRate() {
        let raw = Self.canonicalRaw()
        let modelConfig = ModelConfig(architecture: nil, modelType: "pocket_tts", raw: raw)
        let config = PocketTTSConfig.from(modelConfig)!
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-ptts-sr-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let header = "{}"
            let headerBytes = Array(header.utf8)
            var headerLen = UInt64(headerBytes.count)
            var data = Data()
            withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
            data.append(contentsOf: headerBytes)
            try data.write(to: dir.appendingPathComponent("model.safetensors"))
            let bundle = try SafeTensorsBundle(directory: dir)
            let model = PocketTTSModel(config: config, weights: bundle)
            #expect(model.sampleRate == 24_000)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}
