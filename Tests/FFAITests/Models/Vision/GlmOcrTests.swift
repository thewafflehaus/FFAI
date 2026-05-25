// GlmOcrTests — unit tests for GlmOcr family registry and config parsing.
// These tests run against a small inline config and do not require network
// access or a GPU.

import Foundation
import Testing
@testable import FFAI

@Suite("GlmOcr Vision Tests")
struct GlmOcrTests {

    // ── Helpers ──────────────────────────────────────────────────────

    static func makeGlmOcrDir(json: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-glmocr-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try json.write(to: dir.appendingPathComponent("config.json"),
                       atomically: true, encoding: .utf8)
        return dir
    }

    // Minimal GLM-OCR config.json that mirrors the real checkpoint.
    static let minimalConfigJSON = """
    {
      "architectures": ["GlmOcrForConditionalGeneration"],
      "model_type": "glm_ocr",
      "image_token_id": 59280,
      "eos_token_id": [59246, 59253],
      "quantization": {"bits": 4, "group_size": 64},
      "text_config": {
        "model_type": "glm_ocr_text",
        "hidden_size": 1536,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 4608,
        "vocab_size": 59392,
        "rms_norm_eps": 1e-05,
        "max_position_embeddings": 131072,
        "num_nextn_predict_layers": 1,
        "tie_word_embeddings": false,
        "rope_parameters": {"rope_theta": 10000}
      },
      "vision_config": {
        "model_type": "glm_ocr_vision",
        "depth": 24,
        "hidden_size": 1024,
        "num_heads": 16,
        "intermediate_size": 4096,
        "patch_size": 14,
        "out_hidden_size": 1536,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "rms_norm_eps": 1e-05
      }
    }
    """

    // ── Architecture / model_type registry ───────────────────────────

    @Test("GlmOcr.architectures contains GlmOcrForConditionalGeneration")
    func architecturesContainsExpected() {
        #expect(GlmOcr.architectures.contains("GlmOcrForConditionalGeneration"))
    }

    @Test("GlmOcr.modelTypes contains glm_ocr")
    func modelTypesContainsExpected() {
        #expect(GlmOcr.modelTypes.contains("glm_ocr"))
    }

    @Test("GlmOcr.defaultImageTokenId matches checkpoint")
    func defaultImageTokenId() {
        #expect(GlmOcr.defaultImageTokenId == 59280)
    }

    @Test("GlmOcr.defaultEosTokenId matches checkpoint")
    func defaultEosTokenId() {
        #expect(GlmOcr.defaultEosTokenId == 59246)
    }

    // ── Config parsing ────────────────────────────────────────────────

    @Test("config.architecture is GlmOcrForConditionalGeneration")
    func configArchitecture() throws {
        let dir = try Self.makeGlmOcrDir(json: Self.minimalConfigJSON)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cfg = try ModelConfig.load(from: dir)
        #expect(cfg.architecture == "GlmOcrForConditionalGeneration")
        #expect(cfg.modelType == "glm_ocr")
    }

    @Test("config image_token_id parses to 59280")
    func configImageTokenId() throws {
        let dir = try Self.makeGlmOcrDir(json: Self.minimalConfigJSON)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cfg = try ModelConfig.load(from: dir)
        #expect(cfg.int("image_token_id") == 59280)
    }

    @Test("config text_config nested fields parse correctly")
    func configTextConfig() throws {
        let dir = try Self.makeGlmOcrDir(json: Self.minimalConfigJSON)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cfg = try ModelConfig.load(from: dir)
        let raw = cfg.nested("text_config")
        #expect(raw != nil)
        let tc = ModelConfig(architecture: nil, modelType: nil, raw: raw!)
        #expect(tc.hiddenSize == 1536)
        #expect(tc.numLayers == 16)
        #expect(tc.numAttentionHeads == 16)
        #expect(tc.numKeyValueHeads == 8)
        #expect(tc.headDim == 128)
        #expect(tc.intermediateSize == 4608)
        #expect(tc.vocabSize == 59392)
        #expect(tc.int("max_position_embeddings") == 131072)
        #expect(tc.int("num_nextn_predict_layers") == 1)
        #expect(tc.tieWordEmbeddings == false)
    }

    @Test("config vision_config nested fields parse correctly")
    func configVisionConfig() throws {
        let dir = try Self.makeGlmOcrDir(json: Self.minimalConfigJSON)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cfg = try ModelConfig.load(from: dir)
        let raw = cfg.nested("vision_config")
        #expect(raw != nil)
        let vc = ModelConfig(architecture: nil, modelType: nil, raw: raw!)
        #expect(vc.int("depth") == 24)
        #expect(vc.hiddenSize == 1024)
        #expect(vc.int("num_heads") == 16)
        #expect(vc.intermediateSize == 4096)
        #expect(vc.int("patch_size") == 14)
        #expect(vc.int("out_hidden_size") == 1536)
        #expect(vc.int("spatial_merge_size") == 2)
        #expect(vc.int("temporal_patch_size") == 2)
    }

    @Test("config quantization parses to bits=4 group_size=64")
    func configQuantization() throws {
        let dir = try Self.makeGlmOcrDir(json: Self.minimalConfigJSON)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cfg = try ModelConfig.load(from: dir)
        #expect(cfg.quantization?.bits == 4)
        #expect(cfg.quantization?.groupSize == 64)
    }

    // ── ModelRegistry detection ────────────────────────────────────────

    @Test("ModelRegistry rejects unknown architecture with unsupportedArchitecture")
    func registryRejectsUnknownArch() throws {
        let dir = try Self.makeGlmOcrDir(json: """
        {"architectures": ["UnknownVLM"], "model_type": "unknown_type"}
        """)
        defer { try? FileManager.default.removeItem(at: dir) }
        // Write a minimal safetensors stub so the bundle loads.
        let header = "{}"
        let headerBytes = Array(header.utf8)
        var headerLen = UInt64(headerBytes.count)
        var data = Data()
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(contentsOf: headerBytes)
        try data.write(to: dir.appendingPathComponent("model.safetensors"))

        let cfg = try ModelConfig.load(from: dir)
        let bundle = try SafeTensorsBundle(directory: dir)
        do {
            _ = try ModelRegistry.dispatchAndLoad(
                config: cfg, weights: bundle,
                options: LoadOptions(), device: .shared)
            Issue.record("expected throw for unknown architecture")
        } catch let e as ModelError {
            if case .unsupportedArchitecture(_) = e {
                // expected
            } else {
                Issue.record("unexpected ModelError: \(e)")
            }
        }
    }

    @Test("GlmOcr.architectures is recognised by ModelRegistry dispatch path")
    func registryRecognisesGlmOcrArchitecture() {
        // This is a pure registry-table check — no GPU or file I/O needed.
        // Just confirm the arch is in the set that dispatchAndLoad would match.
        #expect(GlmOcr.architectures.contains("GlmOcrForConditionalGeneration"))
        #expect(!Llama.architectures.contains("GlmOcrForConditionalGeneration"))
        #expect(!Qwen3.architectures.contains("GlmOcrForConditionalGeneration"))
    }

    @Test("GlmOcr.modelTypes is recognised by ModelRegistry dispatch path")
    func registryRecognisesGlmOcrModelType() {
        #expect(GlmOcr.modelTypes.contains("glm_ocr"))
        #expect(!Llama.modelTypes.contains("glm_ocr"))
        #expect(!Qwen3.modelTypes.contains("glm_ocr"))
    }

    // ── GlmOcrRGBImage ────────────────────────────────────────────────

    @Test("GlmOcrRGBImage.solid produces correct shape and values")
    func solidImage() {
        let img = GlmOcrRGBImage.solid(width: 4, height: 4, r: 0.5, g: 0.25, b: 0.1)
        #expect(img.width == 4)
        #expect(img.height == 4)
        #expect(img.data.count == 4 * 4 * 3)
        #expect(img.data[0] == 0.5)
        #expect(img.data[1] == 0.25)
        #expect(img.data[2] == 0.1)
        // All pixels identical.
        for i in stride(from: 0, to: img.data.count, by: 3) {
            #expect(img.data[i]   == 0.5)
            #expect(img.data[i+1] == 0.25)
            #expect(img.data[i+2] == 0.1)
        }
    }

    @Test("GlmOcrRGBImage initializer validates data count")
    func imageDataCountPrecondition() {
        // A valid image should not crash.
        let validData = [Float](repeating: 0, count: 2 * 2 * 3)
        let img = GlmOcrRGBImage(data: validData, height: 2, width: 2)
        #expect(img.height == 2)
        #expect(img.width == 2)
    }

    // ── GlmOcrError descriptions ──────────────────────────────────────

    @Test("GlmOcrError.missingConfig contains 'missing'")
    func errorMissingConfigDescription() {
        let e = GlmOcrError.missingConfig
        #expect(String(describing: e).contains("missing"))
    }

    @Test("GlmOcrError.missingTensor contains the tensor name")
    func errorMissingTensorDescription() {
        let e = GlmOcrError.missingTensor("model.visual.blocks.0.attn.qkv.weight")
        let desc = String(describing: e)
        #expect(desc.contains("model.visual.blocks.0.attn.qkv.weight"))
    }
}
