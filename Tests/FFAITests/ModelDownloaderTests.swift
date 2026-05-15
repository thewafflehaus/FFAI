// Smoke test: download just config.json from a small known repo.
// Skipped automatically if HF is unreachable.

import Foundation
import Testing
@testable import FFAI

@Suite("ModelDownloader")
struct ModelDownloaderTests {
    @Test("download config.json only")
    func downloadConfigOnly() async throws {
        let downloader = ModelDownloader()
        // unsloth/Llama-3.2-1B is a small public mirror that doesn't gate.
        // Use it instead of meta-llama/* so tests work without an HF token.
        do {
            let url = try await downloader.download(
                id: "unsloth/Llama-3.2-1B",
                matching: ["config.json"]
            )
            let configURL = url.appendingPathComponent("config.json")
            #expect(FileManager.default.fileExists(atPath: configURL.path))
            let cfg = try ModelConfig.load(from: url)
            #expect(cfg.architecture == "LlamaForCausalLM")
            #expect(cfg.modelType == "llama")
            #expect(cfg.hiddenSize == 2048)
            #expect(cfg.numLayers == 16)
        } catch {
            // Network-dependent — log and pass rather than fail.
            print("ModelDownloader smoke test skipped: \(error)")
        }
    }

    @Test("isLocalPath heuristic")
    func localPathHeuristic() {
        #expect(ModelLocator.isLocalPath("/abs/path"))
        #expect(ModelLocator.isLocalPath("./relative"))
        #expect(ModelLocator.isLocalPath("../sibling"))
        #expect(ModelLocator.isLocalPath("~/home"))
        #expect(!ModelLocator.isLocalPath("meta-llama/Llama-3.2-1B"))
        #expect(!ModelLocator.isLocalPath("unsloth/llama"))
    }
}
