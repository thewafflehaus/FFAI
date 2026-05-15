import Foundation
import Testing
@testable import FFAI

@Suite("ModelLocator")
struct ModelLocatorTests {
    @Test("isLocalPath classifies paths vs repo ids")
    func classify() {
        #expect(ModelLocator.isLocalPath("/abs/path"))
        #expect(ModelLocator.isLocalPath("./relative"))
        #expect(ModelLocator.isLocalPath("../sibling"))
        #expect(ModelLocator.isLocalPath("~/home/path"))
        #expect(!ModelLocator.isLocalPath("meta-llama/Llama-3.2-1B"))
        #expect(!ModelLocator.isLocalPath("unsloth/llama"))
        #expect(!ModelLocator.isLocalPath("Qwen/Qwen3-4B"))
    }

    @Test("default download patterns include weights + tokenizer + config")
    func defaultPatterns() {
        let p = ModelLocator.defaultDownloadPatterns
        #expect(p.contains("*.safetensors"))
        #expect(p.contains("*.json"))
        #expect(p.contains("*.jinja"))
    }

    @Test("local-path resolution returns the directory if it exists")
    func resolveLocalExisting() async throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let url = try await ModelLocator().resolve(idOrPath: tmp.path)
        #expect(url.path == tmp.path)
    }

    @Test("local-path resolution throws when directory missing")
    func resolveLocalMissing() async {
        do {
            _ = try await ModelLocator().resolve(idOrPath: "/__definitely_not_present__/ffai")
            Issue.record("expected error")
        } catch let e as ModelLocatorError {
            if case .localPathNotFound = e { /* ok */ } else {
                Issue.record("expected .localPathNotFound, got \(e)")
            }
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }
}
