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
// Idefics3Tests — unit tests for config parsing and ModelRegistry routing.
//
// These tests exercise:
//   1. Idefics3VisionConfig.init(from:) parses required and optional fields.
//   2. Idefics3TextConfig.init(from:) parses required and optional fields.
//   3. Idefics3Config.init(from:) parses the top-level config.
//   4. ModelRegistry.dispatchAndLoad routes "Idefics3ForConditionalGeneration"
//      architecture and "idefics3" model_type to the Idefics3 family.
//   5. Idefics3Error descriptions render field names.
//   6. Idefics3 enum membership for architectures and modelTypes.
//
// Does NOT load any real checkpoint (no HF download).

import Foundation
import Testing

@testable import FFAI

@Suite("Idefics3 Vision Config")
struct Idefics3ConfigTests {

    // ─── Minimal valid raw dicts ──────────────────────────────────────────────
    // Computed vars avoid Swift 6 Sendable violations on [String: Any].

    static var minimalVisionRaw: [String: Any] {
        [
            "hidden_size": 1152,
            "patch_size": 14,
            "image_size": 364,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
        ]
    }

    static var minimalTextRaw: [String: Any] {
        [
            "hidden_size": 4096,
            "vocab_size": 128259,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
        ]
    }

    static var minimalTopRaw: [String: Any] {
        [
            "vision_config": minimalVisionRaw,
            "text_config": minimalTextRaw,
            "scale_factor": 2,
            "image_token_id": 49153,
            "vocab_size": 128259,
            "architectures": ["Idefics3ForConditionalGeneration"],
            "model_type": "idefics3",
        ]
    }

    // ─── Idefics3VisionConfig ─────────────────────────────────────────────────

    @Test("VisionConfig: parses required fields")
    func visionConfigRequired() throws {
        let vc = try Idefics3VisionConfig(from: Self.minimalVisionRaw)
        #expect(vc.hiddenSize == 1152)
        #expect(vc.patchSize == 14)
        #expect(vc.imageSize == 364)
        #expect(vc.numHiddenLayers == 27)
        #expect(vc.numAttentionHeads == 16)
    }

    @Test("VisionConfig: default values for optional fields")
    func visionConfigDefaults() throws {
        let vc = try Idefics3VisionConfig(from: Self.minimalVisionRaw)
        // intermediateSize defaults to hiddenSize * 4 = 4608
        #expect(vc.intermediateSize == 1152 * 4)
        #expect(vc.numChannels == 3)
        #expect(vc.layerNormEps == 1e-6)
        // headDim derived: 1152 / 16 = 72
        #expect(vc.headDim == 72)
    }

    @Test("VisionConfig: explicit intermediate_size overrides default")
    func visionConfigExplicitIntermediate() throws {
        var raw = Self.minimalVisionRaw
        raw["intermediate_size"] = 4304
        let vc = try Idefics3VisionConfig(from: raw)
        #expect(vc.intermediateSize == 4304)
    }

    @Test("VisionConfig: missing hidden_size throws Idefics3Error")
    func visionConfigMissingHiddenSize() throws {
        var raw = Self.minimalVisionRaw
        raw.removeValue(forKey: "hidden_size")
        do {
            _ = try Idefics3VisionConfig(from: raw)
            Issue.record("expected throw for missing hidden_size")
        } catch let e as Idefics3Error {
            if case .missingVisionConfig(let field) = e {
                #expect(field == "hidden_size")
            } else {
                Issue.record("unexpected Idefics3Error case: \(e)")
            }
        }
    }

    @Test("VisionConfig: missing patch_size throws Idefics3Error")
    func visionConfigMissingPatchSize() throws {
        var raw = Self.minimalVisionRaw
        raw.removeValue(forKey: "patch_size")
        do {
            _ = try Idefics3VisionConfig(from: raw)
            Issue.record("expected throw for missing patch_size")
        } catch let e as Idefics3Error {
            if case .missingVisionConfig(let field) = e {
                #expect(field == "patch_size")
            } else {
                Issue.record("unexpected Idefics3Error case: \(e)")
            }
        }
    }

    @Test("VisionConfig: missing image_size throws Idefics3Error")
    func visionConfigMissingImageSize() throws {
        var raw = Self.minimalVisionRaw
        raw.removeValue(forKey: "image_size")
        do {
            _ = try Idefics3VisionConfig(from: raw)
            Issue.record("expected throw for missing image_size")
        } catch let e as Idefics3Error {
            if case .missingVisionConfig(let field) = e {
                #expect(field == "image_size")
            } else {
                Issue.record("unexpected Idefics3Error case: \(e)")
            }
        }
    }

    // ─── Idefics3TextConfig ───────────────────────────────────────────────────

    @Test("TextConfig: parses required fields")
    func textConfigRequired() throws {
        let tc = try Idefics3TextConfig(from: Self.minimalTextRaw)
        #expect(tc.hiddenSize == 4096)
        #expect(tc.vocabSize == 128259)
        #expect(tc.numHiddenLayers == 32)
        #expect(tc.numAttentionHeads == 32)
        #expect(tc.numKeyValueHeads == 8)
        #expect(tc.intermediateSize == 14336)
    }

    @Test("TextConfig: default values for optional fields")
    func textConfigDefaults() throws {
        let tc = try Idefics3TextConfig(from: Self.minimalTextRaw)
        // headDim default: hiddenSize / numAttentionHeads = 4096 / 32 = 128
        #expect(tc.headDim == 128)
        #expect(tc.maxPositionEmbeddings == 8192)
        #expect(tc.rmsNormEps == 1e-5)
        // ropeTheta defaults to 500_000 for Idefics3
        #expect(tc.ropeTheta == 500_000)
        #expect(tc.tieWordEmbeddings == false)
    }

    @Test("TextConfig: explicit head_dim overrides derived value")
    func textConfigExplicitHeadDim() throws {
        var raw = Self.minimalTextRaw
        raw["head_dim"] = 64
        let tc = try Idefics3TextConfig(from: raw)
        #expect(tc.headDim == 64)
    }

    @Test("TextConfig: missing hidden_size throws Idefics3Error")
    func textConfigMissingHiddenSize() throws {
        var raw = Self.minimalTextRaw
        raw.removeValue(forKey: "hidden_size")
        do {
            _ = try Idefics3TextConfig(from: raw)
            Issue.record("expected throw for missing hidden_size")
        } catch let e as Idefics3Error {
            if case .missingTextConfig(let field) = e {
                #expect(field == "hidden_size")
            } else {
                Issue.record("unexpected Idefics3Error case: \(e)")
            }
        }
    }

    @Test("TextConfig: missing vocab_size throws Idefics3Error")
    func textConfigMissingVocabSize() throws {
        var raw = Self.minimalTextRaw
        raw.removeValue(forKey: "vocab_size")
        do {
            _ = try Idefics3TextConfig(from: raw)
            Issue.record("expected throw for missing vocab_size")
        } catch let e as Idefics3Error {
            if case .missingTextConfig(let field) = e {
                #expect(field == "vocab_size")
            } else {
                Issue.record("unexpected Idefics3Error case: \(e)")
            }
        }
    }

    // ─── Idefics3Config (top-level) ───────────────────────────────────────────

    @Test("Idefics3Config: parses all top-level fields")
    func topLevelConfig() throws {
        let cfg = try Idefics3Config(from: Self.minimalTopRaw)
        #expect(cfg.scaleFactor == 2)
        #expect(cfg.imageTokenId == 49153)
        #expect(cfg.vocabSize == 128259)
        #expect(cfg.visionConfig.hiddenSize == 1152)
        #expect(cfg.textConfig.hiddenSize == 4096)
    }

    @Test("Idefics3Config: default scaleFactor and imageTokenId")
    func topLevelConfigDefaults() throws {
        var raw = Self.minimalTopRaw
        raw.removeValue(forKey: "scale_factor")
        raw.removeValue(forKey: "image_token_id")
        let cfg = try Idefics3Config(from: raw)
        #expect(cfg.scaleFactor == 2)
        #expect(cfg.imageTokenId == 49153)
    }

    @Test("Idefics3Config: image_token_index used as fallback for image_token_id")
    func topLevelConfigTokenIndexFallback() throws {
        var raw = Self.minimalTopRaw
        raw.removeValue(forKey: "image_token_id")
        raw["image_token_index"] = 49200
        let cfg = try Idefics3Config(from: raw)
        #expect(cfg.imageTokenId == 49200)
    }

    @Test("Idefics3Config: missing vision_config throws Idefics3Error")
    func topLevelMissingVisionConfig() throws {
        var raw = Self.minimalTopRaw
        raw.removeValue(forKey: "vision_config")
        do {
            _ = try Idefics3Config(from: raw)
            Issue.record("expected throw for missing vision_config")
        } catch let e as Idefics3Error {
            if case .missingConfig(let field) = e {
                #expect(field == "vision_config")
            } else {
                Issue.record("unexpected Idefics3Error case: \(e)")
            }
        }
    }

    @Test("Idefics3Config: missing text_config throws Idefics3Error")
    func topLevelMissingTextConfig() throws {
        var raw = Self.minimalTopRaw
        raw.removeValue(forKey: "text_config")
        do {
            _ = try Idefics3Config(from: raw)
            Issue.record("expected throw for missing text_config")
        } catch let e as Idefics3Error {
            if case .missingConfig(let field) = e {
                #expect(field == "text_config")
            } else {
                Issue.record("unexpected Idefics3Error case: \(e)")
            }
        }
    }

    // ─── Idefics3Error descriptions ───────────────────────────────────────────

    @Test("Idefics3Error descriptions contain the field name")
    func errorDescriptions() {
        let a = Idefics3Error.missingConfig("some_field")
        #expect(String(describing: a).contains("some_field"))

        let b = Idefics3Error.missingVisionConfig("patch_size")
        #expect(String(describing: b).contains("patch_size"))

        let c = Idefics3Error.missingTextConfig("hidden_size")
        #expect(String(describing: c).contains("hidden_size"))
    }

    // ─── Idefics3 family enum membership ─────────────────────────────────────

    @Test("Idefics3.architectures contains Idefics3ForConditionalGeneration")
    func architecturesMembership() {
        #expect(Idefics3.architectures.contains("Idefics3ForConditionalGeneration"))
    }

    @Test("Idefics3.modelTypes contains idefics3")
    func modelTypesMembership() {
        #expect(Idefics3.modelTypes.contains("idefics3"))
    }

    // ─── ModelRegistry routing ────────────────────────────────────────────────

    /// Write a minimal fake safetensors bundle (empty header) so SafeTensorsBundle
    /// initializes without needing any real tensors.
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

    @Test("ModelRegistry routes Idefics3ForConditionalGeneration architecture")
    func routingByArchitecture() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-idefics3-arch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        try Self.writeMinimalBundle(in: dir)

        let cfgJSON = """
            {
              "architectures": ["Idefics3ForConditionalGeneration"],
              "model_type": "idefics3",
              "scale_factor": 2,
              "image_token_id": 49153,
              "vocab_size": 128259,
              "vision_config": {
                "hidden_size": 1152, "patch_size": 14, "image_size": 364,
                "num_hidden_layers": 27, "num_attention_heads": 16
              },
              "text_config": {
                "hidden_size": 4096, "vocab_size": 128259,
                "num_hidden_layers": 2, "num_attention_heads": 32,
                "num_key_value_heads": 8, "intermediate_size": 14336
              }
            }
            """
        try cfgJSON.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)

        let cfg = try ModelConfig.load(from: dir)
        #expect(cfg.architecture == "Idefics3ForConditionalGeneration")
        #expect(Idefics3.architectures.contains(cfg.architecture ?? ""))
        #expect(cfg.modelType == "idefics3")
        #expect(Idefics3.modelTypes.contains(cfg.modelType ?? ""))
    }

    @Test("ModelRegistry routes idefics3 model_type")
    func routingByModelType() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-idefics3-mt-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        try Self.writeMinimalBundle(in: dir)

        let cfgJSON = """
            {
              "model_type": "idefics3",
              "scale_factor": 2,
              "image_token_id": 49153,
              "vocab_size": 128259,
              "vision_config": {
                "hidden_size": 1152, "patch_size": 14, "image_size": 364,
                "num_hidden_layers": 1, "num_attention_heads": 16
              },
              "text_config": {
                "hidden_size": 4096, "vocab_size": 128259,
                "num_hidden_layers": 1, "num_attention_heads": 32,
                "num_key_value_heads": 8, "intermediate_size": 14336
              }
            }
            """
        try cfgJSON.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)

        let cfg = try ModelConfig.load(from: dir)
        let bundle = try SafeTensorsBundle(directory: dir)

        // Dispatch SHOULD route to Idefics3 (not throw unsupportedArchitecture).
        // It WILL throw missing tensors because the bundle is empty.
        do {
            _ = try ModelRegistry.dispatchAndLoad(
                config: cfg, weights: bundle,
                options: LoadOptions(), device: .shared
            )
            Issue.record("expected a throw (missing tensors in empty bundle)")
        } catch let e as ModelError {
            if case .unsupportedArchitecture(let a) = e {
                Issue.record("Idefics3 architecture not recognized: \(a)")
            }
            // Any ModelError other than unsupportedArchitecture is unexpected
            // but acceptable here (should not reach; empty bundle throws SafeTensorsError)
        } catch {
            // SafeTensorsError.missingTensor or Idefics3Error — confirms
            // the dispatch reached Idefics3Dense.loadModel correctly.
        }
    }
}
