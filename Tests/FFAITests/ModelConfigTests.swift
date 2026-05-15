import Foundation
import Testing
@testable import FFAI

@Suite("ModelConfig")
struct ModelConfigTests {
    static func writeConfig(_ json: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-cfg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("config.json")
        try json.write(to: url, atomically: true, encoding: .utf8)
        return dir
    }

    @Test("load + standard accessors")
    func loadStandard() throws {
        let dir = try Self.writeConfig("""
        {
          "architectures": ["LlamaForCausalLM"],
          "model_type": "llama",
          "vocab_size": 128256,
          "hidden_size": 2048,
          "intermediate_size": 8192,
          "num_hidden_layers": 16,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 64,
          "rms_norm_eps": 1e-5,
          "rope_theta": 500000.0,
          "tie_word_embeddings": true,
          "eos_token_id": 128001,
          "bos_token_id": 128000
        }
        """)
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg = try ModelConfig.load(from: dir)
        #expect(cfg.architecture == "LlamaForCausalLM")
        #expect(cfg.modelType == "llama")
        #expect(cfg.vocabSize == 128256)
        #expect(cfg.hiddenSize == 2048)
        #expect(cfg.intermediateSize == 8192)
        #expect(cfg.numLayers == 16)
        #expect(cfg.numAttentionHeads == 32)
        #expect(cfg.numKeyValueHeads == 8)
        #expect(cfg.headDim == 64)
        #expect(cfg.rmsNormEps == 1e-5)
        #expect(cfg.ropeTheta == 500000.0)
        #expect(cfg.tieWordEmbeddings == true)
        #expect(cfg.eosTokenId == 128001)
        #expect(cfg.bosTokenId == 128000)
    }

    @Test("headDim derived from hidden_size / num_attention_heads when absent")
    func headDimDerived() throws {
        let dir = try Self.writeConfig("""
        {"hidden_size": 4096, "num_attention_heads": 32}
        """)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cfg = try ModelConfig.load(from: dir)
        #expect(cfg.headDim == 128)
    }

    @Test("numKeyValueHeads falls back to numAttentionHeads")
    func kvHeadsFallback() throws {
        let dir = try Self.writeConfig("""
        {"num_attention_heads": 16}
        """)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cfg = try ModelConfig.load(from: dir)
        #expect(cfg.numKeyValueHeads == 16)
    }

    @Test("eosTokenId accepts both single int and list")
    func eosVariants() throws {
        let single = try Self.writeConfig("""
        {"eos_token_id": 7}
        """)
        defer { try? FileManager.default.removeItem(at: single) }
        #expect(try ModelConfig.load(from: single).eosTokenId == 7)

        let list = try Self.writeConfig("""
        {"eos_token_id": [11, 12, 13]}
        """)
        defer { try? FileManager.default.removeItem(at: list) }
        #expect(try ModelConfig.load(from: list).eosTokenId == 11)
    }

    @Test("nested + has + intArray + string + bool accessors")
    func accessorMix() throws {
        let dir = try Self.writeConfig("""
        {
          "rope_scaling": {"factor": 32.0, "rope_type": "llama3"},
          "some_array": [1, 2, 3],
          "some_string": "hello",
          "some_bool": false
        }
        """)
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg = try ModelConfig.load(from: dir)
        #expect(cfg.has("rope_scaling"))
        #expect(!cfg.has("definitely_missing"))
        #expect(cfg.intArray("some_array") == [1, 2, 3])
        #expect(cfg.string("some_string") == "hello")
        #expect(cfg.bool("some_bool") == false)
        let scaling = cfg.nested("rope_scaling")
        #expect(scaling?["rope_type"] as? String == "llama3")
    }

    @Test("missing config.json throws")
    func missingFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-cfg-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try ModelConfig.load(from: dir)
            Issue.record("expected throw")
        } catch {
            // any error fine — file genuinely doesn't exist
        }
    }

    @Test("malformed json throws")
    func malformed() throws {
        let dir = try Self.writeConfig("not json at all")
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try ModelConfig.load(from: dir)
            Issue.record("expected throw")
        } catch {
            // ok
        }
    }
}
