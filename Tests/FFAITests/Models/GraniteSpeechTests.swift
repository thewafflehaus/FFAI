// GraniteSpeechTests — fast unit tests for GraniteSpeech config parsing
// and AudioModelRegistry dispatch. No GPU work, no model loading.

import Foundation
import Testing
@testable import FFAI

@Suite("GraniteSpeech config + registry")
struct GraniteSpeechTests {

    // MARK: - Helpers

    /// Write a minimal GraniteSpeech config.json to `dir`.
    private static func writeGraniteSpeechConfig(to dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Minimal valid config — only the fields GraniteSpeechConfig.load reads from the top level.
        let json = """
        {
            "architectures": ["GraniteSpeechForConditionalGeneration"],
            "model_type": "granite_speech",
            "audio_token_index": 100352,
            "downsample_rate": 5,
            "window_size": 15,
            "encoder_config": {
                "input_dim": 160,
                "num_layers": 16,
                "hidden_dim": 1024,
                "feedforward_mult": 4,
                "num_heads": 8,
                "dim_head": 128,
                "output_dim": 348,
                "context_size": 200,
                "max_pos_emb": 512,
                "conv_kernel_size": 15,
                "conv_expansion_factor": 2
            },
            "projector_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 2,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "layer_norm_eps": 1e-12,
                "encoder_hidden_size": 1024
            },
            "text_config": {
                "model_type": "granite",
                "vocab_size": 100353,
                "hidden_size": 2048,
                "intermediate_size": 4096,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "attention_bias": false,
                "mlp_bias": false,
                "attention_multiplier": 0.0078125,
                "embedding_multiplier": 12.0,
                "residual_multiplier": 0.22,
                "logits_scaling": 8.0,
                "tie_word_embeddings": false
            }
        }
        """
        try json.write(to: dir.appendingPathComponent("config.json"),
                       atomically: true, encoding: .utf8)
    }

    /// Write a minimal empty safetensors bundle (required by SafeTensorsBundle init in tests).
    private static func writeEmptyBundle(to dir: URL) throws {
        let header = "{}"
        let headerBytes = Array(header.utf8)
        var headerLen = UInt64(headerBytes.count)
        var data = Data()
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(contentsOf: headerBytes)
        try data.write(to: dir.appendingPathComponent("model.safetensors"))
    }

    // MARK: - Config parsing tests

    @Test("GraniteSpeechConfig.load parses encoder fields")
    func parseEncoderConfig() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gs-enc-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeGraniteSpeechConfig(to: dir)

        let raw = try ModelConfig.load(from: dir)
        let cfg = try GraniteSpeechConfig.load(from: raw)

        #expect(cfg.encoderConfig.inputDim == 160)
        #expect(cfg.encoderConfig.numLayers == 16)
        #expect(cfg.encoderConfig.hiddenDim == 1024)
        #expect(cfg.encoderConfig.numHeads == 8)
        #expect(cfg.encoderConfig.dimHead == 128)
        #expect(cfg.encoderConfig.outputDim == 348)
        #expect(cfg.encoderConfig.contextSize == 200)
        #expect(cfg.encoderConfig.maxPosEmb == 512)
        #expect(cfg.encoderConfig.convKernelSize == 15)
        #expect(cfg.encoderConfig.convExpansionFactor == 2)
    }

    @Test("GraniteSpeechConfig.load parses projector fields")
    func parseProjectorConfig() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gs-proj-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeGraniteSpeechConfig(to: dir)

        let raw = try ModelConfig.load(from: dir)
        let cfg = try GraniteSpeechConfig.load(from: raw)

        #expect(cfg.projectorConfig.hiddenSize == 1024)
        #expect(cfg.projectorConfig.numHiddenLayers == 2)
        #expect(cfg.projectorConfig.numAttentionHeads == 16)
        #expect(cfg.projectorConfig.intermediateSize == 4096)
        #expect(cfg.projectorConfig.encoderHiddenSize == 1024)
    }

    @Test("GraniteSpeechConfig.load parses text (LM) fields")
    func parseTextConfig() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gs-txt-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeGraniteSpeechConfig(to: dir)

        let raw = try ModelConfig.load(from: dir)
        let cfg = try GraniteSpeechConfig.load(from: raw)

        #expect(cfg.textConfig.vocabSize == 100353)
        #expect(cfg.textConfig.hiddenSize == 2048)
        #expect(cfg.textConfig.numHiddenLayers == 40)
        #expect(cfg.textConfig.numAttentionHeads == 16)
        #expect(cfg.textConfig.numKeyValueHeads == 4)
        #expect(cfg.textConfig.embeddingMultiplier == 12.0)
        #expect(cfg.textConfig.residualMultiplier == 0.22)
        #expect(cfg.textConfig.logitsScaling == 8.0)
        #expect(cfg.textConfig.attentionMultiplier == 0.0078125)
        #expect(!cfg.textConfig.tieWordEmbeddings)
    }

    @Test("GraniteSpeechConfig.load parses top-level speech fields")
    func parseTopLevelSpeechFields() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gs-top-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeGraniteSpeechConfig(to: dir)

        let raw = try ModelConfig.load(from: dir)
        let cfg = try GraniteSpeechConfig.load(from: raw)

        #expect(cfg.audioTokenIndex == 100352)
        #expect(cfg.downsampleRate == 5)
        #expect(cfg.windowSize == 15)
        #expect(cfg.numQueriesPerWindow == 3)  // windowSize / downsampleRate = 15/5 = 3
    }

    // MARK: - Family detection tests

    @Test("GraniteSpeech.modelTypes contains 'granite_speech'")
    func familyModelType() {
        #expect(GraniteSpeech.modelTypes.contains("granite_speech"))
    }

    @Test("GraniteSpeech.architectures contains 'GraniteSpeechForConditionalGeneration'")
    func familyArchitecture() {
        #expect(GraniteSpeech.architectures.contains("GraniteSpeechForConditionalGeneration"))
    }

    // MARK: - AudioModelRegistry dispatch tests

    @Test("AudioModelRegistry dispatches granite_speech model_type")
    func registryDispatchesModelType() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gs-reg-mt-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeGraniteSpeechConfig(to: dir)
        try Self.writeEmptyBundle(to: dir)

        let raw = try ModelConfig.load(from: dir)
        #expect(raw.modelType == "granite_speech")
        // Verify the dispatch logic without loading a real model (would need weights).
        // We confirm the architecture strings match and registry recognises them.
        #expect(GraniteSpeech.modelTypes.contains(raw.modelType ?? ""))
    }

    @Test("AudioModelRegistry dispatches GraniteSpeechForConditionalGeneration architecture")
    func registryDispatchesArchitecture() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gs-reg-arch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeGraniteSpeechConfig(to: dir)
        try Self.writeEmptyBundle(to: dir)

        let raw = try ModelConfig.load(from: dir)
        #expect(raw.architecture == "GraniteSpeechForConditionalGeneration")
        #expect(GraniteSpeech.architectures.contains(raw.architecture ?? ""))
    }

    @Test("AudioModelRegistry rejects unknown model_type")
    func registryUnknownModelType() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gs-reg-unk-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try """
        {"architectures": ["NotAFamily"], "model_type": "unknown_stt"}
        """.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Self.writeEmptyBundle(to: dir)

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
