// Unit tests for the SAMAudio family:
//   • Config decode (defaults + custom fields)
//   • Registry detection (architecture + model_type routing)
//   • Capability declaration
//
// No checkpoint required. All tests run without network access.

import Foundation
import Testing
@testable import FFAI

@Suite("SAMAudio")
struct SAMAudioTests {

    // ─── Config decode ────────────────────────────────────────────────────

    @Test("SAMAudioConfig defaults round-trip via JSON")
    func configDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(SAMAudioConfig.self, from: data)

        // T5 text encoder defaults
        #expect(config.textEncoder.name == "t5-base")
        #expect(config.textEncoder.dim == 768)

        // Codec defaults
        #expect(config.audioCodec.hopLength == 512)
        #expect(config.audioCodec.sampleRate == 44100)
        #expect(config.audioCodec.codebookDim == 128)

        // Transformer defaults (large variant)
        #expect(config.transformer.dim == 2816)
        #expect(config.transformer.nHeads == 22)
        #expect(config.transformer.nLayers == 22)
        #expect(config.transformer.contextDim == 2816)
        #expect(config.transformer.outChannels == 256)
    }

    @Test("SAMAudioConfig decodes custom transformer dimensions")
    func configCustomTransformer() throws {
        let json = """
        {
          "transformer": {
            "dim": 1024,
            "n_heads": 8,
            "n_layers": 12,
            "context_dim": 1024,
            "out_channels": 256
          }
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(SAMAudioConfig.self, from: data)

        #expect(config.transformer.dim == 1024)
        #expect(config.transformer.nHeads == 8)
        #expect(config.transformer.nLayers == 12)
        #expect(config.transformer.contextDim == 1024)
    }

    @Test("SAMAudioConfig decodes inChannels from JSON")
    func configInChannels() throws {
        let json = """
        {
          "in_channels": 768,
          "num_anchors": 3,
          "anchor_embedding_dim": 128
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(SAMAudioConfig.self, from: data)
        #expect(config.inChannels == 768)
        #expect(config.numAnchors == 3)
        #expect(config.anchorEmbeddingDim == 128)
    }

    @Test("SAMAudioConfig static presets have expected shapes")
    func configPresets() {
        let small = SAMAudioConfig.small
        #expect(small.transformer.dim == 1024)
        #expect(small.transformer.nLayers == 12)

        let base = SAMAudioConfig.base
        #expect(base.transformer.dim == 1536)
        #expect(base.transformer.nLayers == 16)

        let large = SAMAudioConfig.large
        #expect(large.transformer.dim == 2816)
        #expect(large.transformer.nLayers == 22)
    }

    // ─── ODE options ──────────────────────────────────────────────────────

    @Test("SAMAudioODEOptions default is midpoint, stepSize=2/32")
    func odeDefaults() {
        let opts = SAMAudioODEOptions.default
        #expect(opts.method == .midpoint)
        #expect(opts.stepSize == Float(2.0 / 32.0))
    }

    @Test("SAMAudioODEOptions euler method round-trips JSON")
    func odeJsonRoundTrip() throws {
        let opts = SAMAudioODEOptions(method: .euler, stepSize: 0.1)
        let data = try JSONEncoder().encode(opts)
        let decoded = try JSONDecoder().decode(SAMAudioODEOptions.self, from: data)
        #expect(decoded.method == .euler)
        #expect(decoded.stepSize == 0.1)
    }

    // ─── Registry detection ───────────────────────────────────────────────

    @Test("SAMAudio.architectures contains SAMAudioForSeparation")
    func registryArchitectureSet() {
        #expect(SAMAudio.architectures.contains("SAMAudioForSeparation"))
    }

    @Test("SAMAudio.modelTypes contains sam_audio")
    func registryModelTypeSet() {
        #expect(SAMAudio.modelTypes.contains("sam_audio"))
    }

    @Test("AudioModelRegistry.handles returns true for sam_audio model_type")
    func registryRoutesByModelType() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-samaudio-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let configJSON = """
        {"model_type": "sam_audio"}
        """
        try configJSON.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8
        )

        let config = try ModelConfig.load(from: dir)
        #expect(SAMAudio.handles(config))
        #expect(AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == Capability.speechToSpeech)
    }

    @Test("AudioModelRegistry.handles returns false for unknown model_type")
    func registryRejectsUnknown() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-samaudio-unk-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let configJSON = """
        {"architectures": ["NotAudioModel"], "model_type": "unknown_audio_xyz_not_real"}
        """
        try configJSON.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8
        )

        let config = try ModelConfig.load(from: dir)
        #expect(!SAMAudio.handles(config))
        #expect(!AudioModelRegistry.handles(config))
        #expect(AudioModelRegistry.capabilities(for: config) == nil)
    }

    // ─── Capability ───────────────────────────────────────────────────────

    @Test("SAMAudio.capability contains audioIn and audioOut (speechToSpeech set)")
    func capabilitySet() {
        #expect(SAMAudio.capability.contains(.audioIn))
        #expect(SAMAudio.capability.contains(.audioOut))
        // speechToSpeech is a static Set alias for [.audioIn, .audioOut]
        #expect(SAMAudio.capability == Capability.speechToSpeech)
    }

    @Test("Capability.speechToSpeech static set contains audioIn and audioOut")
    func speechToSpeechStaticSet() {
        #expect(Capability.speechToSpeech.contains(.audioIn))
        #expect(Capability.speechToSpeech.contains(.audioOut))
    }

    @Test("SAMAudioError descriptions are non-empty")
    func errorDescriptions() {
        let errors: [SAMAudioError] = [
            .invalidAudioShape([1, 2, 3]),
            .mismatchedBatchCounts,
            .invalidStepSize(1.5),
            .missingTextMask,
            .noCompatibleWeights,
            .missingModelWeights(5),
            .invalidChunkConfiguration(chunkSeconds: 2, overlapSeconds: 3),
            .unsupportedBatchSize(2),
            .chunkedAnchorsNotSupported,
            .modelFilesNotFound("/tmp/foo"),
        ]
        for e in errors {
            #expect(!e.description.isEmpty)
        }
    }

    // ─── SAMAudioModel construction ───────────────────────────────────────

    @Test("SAMAudioModel initialises with default config without crashing")
    func modelInit() {
        let model = SAMAudioModel(config: SAMAudioConfig())
        #expect(model.config.transformer.dim == 2816)
        // No weights loaded yet.
        #expect(model.loadedParameterCount == 0)
    }

    @Test("SAMAudioModel.loadWeights throws noCompatibleWeights for empty bundle")
    func modelLoadWeightsEmptyBundle() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-samaudio-wt-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let header = "{}"
        let headerBytes = Array(header.utf8)
        var headerLen = UInt64(headerBytes.count)
        var data = Data()
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(contentsOf: headerBytes)
        try data.write(to: dir.appendingPathComponent("model.safetensors"))

        let bundle = try SafeTensorsBundle(directory: dir)
        let model = SAMAudioModel(config: SAMAudioConfig())
        do {
            try model.loadWeights(from: bundle)
            Issue.record("Expected SAMAudioError.noCompatibleWeights")
        } catch let e as SAMAudioError {
            if case .noCompatibleWeights = e { /* expected */ }
            else { Issue.record("Unexpected: \(e)") }
        }
    }
}
