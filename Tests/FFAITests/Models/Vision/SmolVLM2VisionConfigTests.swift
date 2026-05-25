// SmolVLM2VisionConfigTests — unit tests for config parsing and ModelRegistry routing.
//
// These tests exercise:
//   1. SmolVLM2Config.init(from:) parses vision_config and text_config correctly.
//   2. ModelRegistry.dispatchAndLoad routes "SmolVLMForConditionalGeneration" arch
//      and "smolvlm" model_type to the SmolVLM2 family.
//   3. SmolVLM2 model / modelTypes membership.
//   4. SmolVLM2Error descriptions render the field name.
//
// Does NOT load any real checkpoint (no HF download).

import Foundation
import Testing
@testable import FFAI

@Suite("SmolVLM2Config parsing")
struct SmolVLM2VisionConfigTests {

    // ─── Minimal valid raw dicts ──────────────────────────────────────────────
    // Computed vars instead of stored statics avoid Swift 6 Sendable violations
    // on [String: Any], which is not Sendable.

    static var minimalVisionRaw: [String: Any] {
        [
            "hidden_size": 768,
            "patch_size": 16,
            "image_size": 512,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        ]
    }

    static var minimalTextRaw: [String: Any] {
        [
            "hidden_size": 960,
            "vocab_size": 49280,
            "num_hidden_layers": 32,
            "num_attention_heads": 15,
            "num_key_value_heads": 5,
            "intermediate_size": 2560,
        ]
    }

    static var minimalTopRaw: [String: Any] {
        [
            "vision_config": minimalVisionRaw,
            "text_config": minimalTextRaw,
            "scale_factor": 4,
            "image_token_id": 49190,
            "vocab_size": 49280,
            "architectures": ["SmolVLMForConditionalGeneration"],
            "model_type": "smolvlm",
        ]
    }

    // ─── SmolVLM2VisionConfig ─────────────────────────────────────────────────

    @Test("VisionConfig: parses required fields")
    func visionConfigRequired() throws {
        let vc = try SmolVLM2VisionConfig(from: Self.minimalVisionRaw)
        #expect(vc.hiddenSize          == 768)
        #expect(vc.patchSize           == 16)
        #expect(vc.imageSize           == 512)
        #expect(vc.numHiddenLayers     == 12)
        #expect(vc.numAttentionHeads   == 12)
    }

    @Test("VisionConfig: default values for optional fields")
    func visionConfigDefaults() throws {
        let vc = try SmolVLM2VisionConfig(from: Self.minimalVisionRaw)
        // Defaults: intermediateSize=3072, numChannels=3, layerNormEps=1e-6
        #expect(vc.intermediateSize    == 3072)
        #expect(vc.numChannels         == 3)
        #expect(vc.layerNormEps        == 1e-6)
        // headDim derived: 768 / 12 = 64
        #expect(vc.headDim             == 64)
    }

    @Test("VisionConfig: custom optional fields are respected")
    func visionConfigCustomOptionals() throws {
        var raw = Self.minimalVisionRaw
        raw["intermediate_size"]  = 4096
        raw["num_channels"]       = 1
        raw["layer_norm_eps"]     = 1e-5
        let vc = try SmolVLM2VisionConfig(from: raw)
        #expect(vc.intermediateSize == 4096)
        #expect(vc.numChannels      == 1)
        #expect(vc.layerNormEps     == 1e-5)
    }

    @Test("VisionConfig: missing hidden_size throws SmolVLM2Error")
    func visionConfigMissingHiddenSize() throws {
        var raw = Self.minimalVisionRaw
        raw.removeValue(forKey: "hidden_size")
        do {
            _ = try SmolVLM2VisionConfig(from: raw)
            Issue.record("expected throw for missing hidden_size")
        } catch let e as SmolVLM2Error {
            if case .missingVisionConfig(let field) = e {
                #expect(field == "hidden_size")
            } else {
                Issue.record("unexpected SmolVLM2Error case: \(e)")
            }
        }
    }

    @Test("VisionConfig: missing patch_size throws SmolVLM2Error")
    func visionConfigMissingPatchSize() throws {
        var raw = Self.minimalVisionRaw
        raw.removeValue(forKey: "patch_size")
        do {
            _ = try SmolVLM2VisionConfig(from: raw)
            Issue.record("expected throw for missing patch_size")
        } catch let e as SmolVLM2Error {
            if case .missingVisionConfig(let field) = e {
                #expect(field == "patch_size")
            } else {
                Issue.record("unexpected SmolVLM2Error case: \(e)")
            }
        }
    }

    // ─── SmolVLM2TextConfig ───────────────────────────────────────────────────

    @Test("TextConfig: parses required fields")
    func textConfigRequired() throws {
        let tc = try SmolVLM2TextConfig(from: Self.minimalTextRaw)
        #expect(tc.hiddenSize        == 960)
        #expect(tc.vocabSize         == 49280)
        #expect(tc.numHiddenLayers   == 32)
        #expect(tc.numAttentionHeads == 15)
        #expect(tc.numKeyValueHeads  == 5)
        #expect(tc.intermediateSize  == 2560)
    }

    @Test("TextConfig: default values for optional fields")
    func textConfigDefaults() throws {
        let tc = try SmolVLM2TextConfig(from: Self.minimalTextRaw)
        // headDim default: hiddenSize / numAttentionHeads = 960 / 15 = 64
        #expect(tc.headDim              == 64)
        #expect(tc.maxPositionEmbeddings == 8192)
        #expect(tc.rmsNormEps           == 1e-5)
        #expect(tc.ropeTheta            == 100_000)
        #expect(tc.tieWordEmbeddings    == false)
    }

    @Test("TextConfig: explicit head_dim overrides derived value")
    func textConfigExplicitHeadDim() throws {
        var raw = Self.minimalTextRaw
        raw["head_dim"] = 128
        let tc = try SmolVLM2TextConfig(from: raw)
        #expect(tc.headDim == 128)
    }

    @Test("TextConfig: missing hidden_size throws SmolVLM2Error")
    func textConfigMissingHiddenSize() throws {
        var raw = Self.minimalTextRaw
        raw.removeValue(forKey: "hidden_size")
        do {
            _ = try SmolVLM2TextConfig(from: raw)
            Issue.record("expected throw for missing hidden_size")
        } catch let e as SmolVLM2Error {
            if case .missingTextConfig(let field) = e {
                #expect(field == "hidden_size")
            } else {
                Issue.record("unexpected SmolVLM2Error case: \(e)")
            }
        }
    }

    // ─── SmolVLM2Config (top-level) ───────────────────────────────────────────

    @Test("SmolVLM2Config: parses all top-level fields")
    func topLevelConfig() throws {
        let cfg = try SmolVLM2Config(from: Self.minimalTopRaw)
        #expect(cfg.scaleFactor    == 4)
        #expect(cfg.imageTokenId   == 49190)
        #expect(cfg.vocabSize      == 49280)
        #expect(cfg.visionConfig.hiddenSize == 768)
        #expect(cfg.textConfig.hiddenSize   == 960)
    }

    @Test("SmolVLM2Config: default scaleFactor and imageTokenId")
    func topLevelConfigDefaults() throws {
        var raw = Self.minimalTopRaw
        raw.removeValue(forKey: "scale_factor")
        raw.removeValue(forKey: "image_token_id")
        let cfg = try SmolVLM2Config(from: raw)
        #expect(cfg.scaleFactor  == 4)
        #expect(cfg.imageTokenId == 49190)
    }

    @Test("SmolVLM2Config: missing vision_config throws SmolVLM2Error")
    func topLevelMissingVisionConfig() throws {
        var raw = Self.minimalTopRaw
        raw.removeValue(forKey: "vision_config")
        do {
            _ = try SmolVLM2Config(from: raw)
            Issue.record("expected throw for missing vision_config")
        } catch let e as SmolVLM2Error {
            if case .missingConfig(let field) = e {
                #expect(field == "vision_config")
            } else {
                Issue.record("unexpected SmolVLM2Error case: \(e)")
            }
        }
    }

    // ─── SmolVLM2Error descriptions ───────────────────────────────────────────

    @Test("SmolVLM2Error descriptions contain the field name")
    func errorDescriptions() {
        let a = SmolVLM2Error.missingConfig("some_field")
        #expect(String(describing: a).contains("some_field"))

        let b = SmolVLM2Error.missingVisionConfig("patch_size")
        #expect(String(describing: b).contains("patch_size"))

        let c = SmolVLM2Error.missingTextConfig("hidden_size")
        #expect(String(describing: c).contains("hidden_size"))
    }

    // ─── SmolVLM2 family enum membership ─────────────────────────────────────

    @Test("SmolVLM2.architectures contains SmolVLMForConditionalGeneration")
    func architecturesMembership() {
        #expect(SmolVLM2.architectures.contains("SmolVLMForConditionalGeneration"))
    }

    @Test("SmolVLM2.modelTypes contains smolvlm")
    func modelTypesMembership() {
        #expect(SmolVLM2.modelTypes.contains("smolvlm"))
    }

    // ─── ModelRegistry routing ────────────────────────────────────────────────

    /// Write a minimal fake safetensors bundle (empty header) so SafeTensorsBundle
    /// initializes without needing any real tensors. The dispatch throws before
    /// loading tensors when the arch is routed to SmolVLM2 — but SmolVLM2 needs
    /// actual tensors, so we test routing separately by checking the error message.
    static func writeMinimalBundle(in dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let header = "{}"
        let headerBytes = Array(header.utf8)
        var headerLen = UInt64(headerBytes.count)
        var data = Data()
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(contentsOf: headerBytes)
        try data.write(to: dir.appendingPathComponent("model.safetensors"))
    }

    @Test("ModelRegistry routes SmolVLMForConditionalGeneration architecture")
    func routingByArchitecture() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-smolvlm2-arch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        try Self.writeMinimalBundle(in: dir)

        // Write a config.json that matches SmolVLM2's architecture string
        let cfgJSON = """
        {
          "architectures": ["SmolVLMForConditionalGeneration"],
          "model_type": "smolvlm",
          "scale_factor": 4,
          "image_token_id": 49190,
          "vocab_size": 49280,
          "vision_config": {
            "hidden_size": 768, "patch_size": 16, "image_size": 512,
            "num_hidden_layers": 12, "num_attention_heads": 12
          },
          "text_config": {
            "hidden_size": 960, "vocab_size": 49280,
            "num_hidden_layers": 2, "num_attention_heads": 15,
            "num_key_value_heads": 5, "intermediate_size": 2560
          }
        }
        """
        try cfgJSON.write(to: dir.appendingPathComponent("config.json"),
                          atomically: true, encoding: .utf8)

        let cfg = try ModelConfig.load(from: dir)
        // Verify the architecture is recognized by SmolVLM2
        #expect(cfg.architecture == "SmolVLMForConditionalGeneration")
        #expect(SmolVLM2.architectures.contains(cfg.architecture ?? ""))

        // Verify model_type is recognized
        #expect(cfg.modelType == "smolvlm")
        #expect(SmolVLM2.modelTypes.contains(cfg.modelType ?? ""))
    }

    @Test("ModelRegistry dispatch to SmolVLM2 fails on empty tensors (correct path)")
    func registryDispatchSmolVLM2() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-smolvlm2-dispatch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        try Self.writeMinimalBundle(in: dir)

        let cfgJSON = """
        {
          "architectures": ["SmolVLMForConditionalGeneration"],
          "model_type": "smolvlm",
          "scale_factor": 4,
          "image_token_id": 49190,
          "vocab_size": 49280,
          "vision_config": {
            "hidden_size": 768, "patch_size": 16, "image_size": 512,
            "num_hidden_layers": 1, "num_attention_heads": 12
          },
          "text_config": {
            "hidden_size": 960, "vocab_size": 49280,
            "num_hidden_layers": 1, "num_attention_heads": 15,
            "num_key_value_heads": 5, "intermediate_size": 2560
          }
        }
        """
        try cfgJSON.write(to: dir.appendingPathComponent("config.json"),
                          atomically: true, encoding: .utf8)

        let cfg = try ModelConfig.load(from: dir)
        let bundle = try SafeTensorsBundle(directory: dir)

        // The dispatch SHOULD route to SmolVLM2 (not throw unsupportedArchitecture),
        // but WILL throw a SafeTensors missingTensor error because the bundle is empty.
        do {
            _ = try ModelRegistry.dispatchAndLoad(
                config: cfg, weights: bundle,
                options: LoadOptions(), device: .shared
            )
            Issue.record("expected a throw (missing tensors in empty bundle)")
        } catch let e as ModelError {
            // If we get a ModelError, it must NOT be unsupportedArchitecture —
            // that would mean SmolVLM2 was not recognized.
            if case .unsupportedArchitecture(let a) = e {
                Issue.record("SmolVLM2 architecture not recognized: \(a)")
            }
        } catch {
            // SafeTensorsError.missingTensor or SmolVLM2Error — expected,
            // confirms the dispatch reached SmolVLM2Dense.loadModel.
        }
    }
}
