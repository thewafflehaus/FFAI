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
import Foundation
import Testing

@testable import FFAI

// ─── Config parse ─────────────────────────────────────────────────────────────

@Suite("Paligemma Vision Config")
struct PaligemmaTests {

    // A minimal PaliGemma-like config.json payload to exercise ModelConfig
    // parsing without touching any weight files.
    private static let minimalConfig: String = #"""
        {
            "architectures": ["PaliGemmaForConditionalGeneration"],
            "model_type": "paligemma",
            "hidden_size": 2048,
            "image_token_index": 257152,
            "pad_token_id": 0,
            "vocab_size": 257216,
            "projection_dim": 2048,
            "text_config": {
                "hidden_size": 2048,
                "intermediate_size": 16384,
                "model_type": "gemma",
                "num_attention_heads": 8,
                "num_hidden_layers": 18,
                "num_key_value_heads": 1,
                "vocab_size": 257216,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0
            },
            "vision_config": {
                "hidden_size": 1152,
                "image_size": 448,
                "intermediate_size": 4352,
                "model_type": "siglip_vision_model",
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "patch_size": 14
            }
        }
        """#

    private static func writeTempConfig() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-paligemma-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try minimalConfig.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)
        return dir
    }

    @Test("ModelConfig.load parses paligemma model_type")
    func configParseModelType() throws {
        let dir = try Self.writeTempConfig()
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg = try ModelConfig.load(from: dir)
        #expect(cfg.modelType == "paligemma")
    }

    @Test("ModelConfig.load parses PaliGemmaForConditionalGeneration architecture")
    func configParseArchitecture() throws {
        let dir = try Self.writeTempConfig()
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg = try ModelConfig.load(from: dir)
        #expect(cfg.architecture == "PaliGemmaForConditionalGeneration")
    }

    @Test("ModelConfig.load reads hidden_size from top level")
    func configHiddenSize() throws {
        let dir = try Self.writeTempConfig()
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg = try ModelConfig.load(from: dir)
        #expect(cfg.hiddenSize == 2048)
    }

    @Test("Paligemma.architectures contains PaliGemmaForConditionalGeneration")
    func architecturesContains() {
        #expect(Paligemma.architectures.contains("PaliGemmaForConditionalGeneration"))
    }

    @Test("Paligemma.modelTypes contains paligemma")
    func modelTypesContains() {
        #expect(Paligemma.modelTypes.contains("paligemma"))
    }

    @Test("Paligemma.variant returns PaligemmaStandard")
    func variantIsPaligemmaStandard() throws {
        let dir = try Self.writeTempConfig()
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg = try ModelConfig.load(from: dir)
        let v = try Paligemma.variant(for: cfg)
        #expect(v is PaligemmaStandard.Type)
    }

    @Test("VisionLanguageArchitectures.architectures contains PaliGemmaForConditionalGeneration")
    func vlmArchitectureRegistered() {
        #expect(
            VisionLanguageArchitectures.architectures.contains(
                "PaliGemmaForConditionalGeneration"))
    }

    @Test("ModelRegistry dispatches PaliGemmaForConditionalGeneration to loadPaligemma path")
    func registryDispatchByArchitecture() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-pg-dispatch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Write a config that only has the architecture string — no weights needed
        // because we expect a throw before any weight access.
        try #"""
        {"architectures": ["PaliGemmaForConditionalGeneration"], "model_type": "paligemma"}
        """#.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)

        // Empty safetensors bundle
        let header = "{}"
        let headerBytes = Array(header.utf8)
        var headerLen = UInt64(headerBytes.count)
        var data = Data()
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(contentsOf: headerBytes)
        try data.write(to: dir.appendingPathComponent("model.safetensors"))

        let cfg = try ModelConfig.load(from: dir)
        let bundle = try SafeTensorsBundle(directory: dir)

        // Should dispatch to paligemma loader and throw a PaligemmaError
        // (missing config fields), NOT ModelError.unsupportedArchitecture.
        do {
            _ = try ModelRegistry.dispatchAndLoad(
                config: cfg, weights: bundle,
                options: LoadOptions(), device: .shared
            )
            Issue.record("expected throw")
        } catch let e as PaligemmaError {
            // Correct — we hit the paligemma loader and failed on missing config.
            #expect(
                String(describing: e).contains("text_config")
                    || String(describing: e).contains("missing"))
        } catch let e as ModelError {
            // If it throws unsupportedArchitecture the dispatch didn't route correctly.
            Issue.record("expected PaligemmaError, got ModelError: \(e)")
        } catch {
            // Any other error (SafeTensorsError, etc.) is fine — we got past dispatch.
            // The important thing is that it did NOT throw .unsupportedArchitecture.
        }
    }

    @Test("ModelRegistry dispatches paligemma model_type to loadPaligemma path")
    func registryDispatchByModelType() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-pg-mt-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        try #"""
        {"model_type": "paligemma"}
        """#.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)

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
                options: LoadOptions(), device: .shared
            )
            Issue.record("expected throw")
        } catch is ModelError {
            Issue.record("dispatched to wrong loader (ModelError.unsupportedArchitecture)")
        } catch {
            // Anything else — we reached the paligemma loader.
        }
    }

    @Test("PaligemmaError.missingConfig description is human-readable")
    func errorDescription() {
        let e = PaligemmaError.missingConfig("text_config")
        #expect(String(describing: e).contains("text_config"))
        let e2 = PaligemmaError.imageNotSet
        #expect(String(describing: e2).contains("setImagePixels"))
    }
}
