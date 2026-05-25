import Foundation
import Testing
@testable import FFAI

@Suite("ModelRegistry + ModelError")
struct ModelRegistryTests {
    static func writeBundle(in dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Empty safetensors file (header == "{}"). Bundle.dispatchAndLoad
        // will throw before touching tensor data when arch is unknown.
        let header = "{}"
        let headerBytes = Array(header.utf8)
        var headerLen = UInt64(headerBytes.count)
        var data = Data()
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(contentsOf: headerBytes)
        try data.write(to: dir.appendingPathComponent("model.safetensors"))
    }

    @Test("unknown architecture is rejected with ModelError.unsupportedArchitecture")
    func unknownArchitecture() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-reg-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeBundle(in: dir)

        let cfgURL = dir.appendingPathComponent("config.json")
        try #"""
        {"architectures": ["NotAFamily"], "model_type": "alien"}
        """#.write(to: cfgURL, atomically: true, encoding: .utf8)

        let cfg = try ModelConfig.load(from: dir)
        let bundle = try SafeTensorsBundle(directory: dir)
        do {
            _ = try ModelRegistry.dispatchAndLoad(
                config: cfg, weights: bundle,
                options: LoadOptions(), device: .shared
            )
            Issue.record("expected throw")
        } catch let e as ModelError {
            if case .unsupportedArchitecture(let a) = e {
                #expect(a == "NotAFamily")
            } else {
                Issue.record("expected .unsupportedArchitecture, got \(e)")
            }
        }
    }

    @Test("ModelError.description renders each case")
    func errorDescriptions() {
        let a = ModelError.unsupportedArchitecture("foo")
        #expect(String(describing: a).contains("foo"))
        let b = ModelError.unsupportedModelType("bar")
        #expect(String(describing: b).contains("bar"))
        let c = ModelError.capabilityNotAvailable(.audioOut)
        #expect(String(describing: c).contains("audioOut"))
    }

    @Test("LlamaError.missingConfig description")
    func llamaMissingConfig() {
        let e = LlamaError.missingConfig
        #expect(String(describing: e).contains("missing"))
    }

    @Test("Llama.variant returns LlamaDense for any llama config")
    func variantDispatch() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-variant-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try #"{"model_type": "llama"}"#.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8
        )
        let cfg = try ModelConfig.load(from: dir)
        let v = try Llama.variant(for: cfg)
        #expect(v is LlamaDense.Type)
    }
}
