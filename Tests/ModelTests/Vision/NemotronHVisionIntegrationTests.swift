// Slow integration test for Nemotron-VLM (NVIDIA's Nemotron Nano VL —
// a ViT tower + multi-modal projector + the NemotronH stack-interleaved
// hybrid text backbone).
//
// Verifies the vision path end-to-end on a real checkpoint: the ViT
// vision tower loads into the shared `VisionEncoder`, runs its
// bidirectional-attention forward, the multi-modal projector maps the
// patch tokens into the NemotronH text hidden dim, the cross-modal
// splice injects them, and the fused stream decodes coherent text
// through the NemotronH backbone (which now supports embedding-input
// forward for the splice).
//
// Conditional on a cached MLX conversion of Nemotron-VL existing in
// the local HF cache. As of writing no `mlx-community` conversion has
// shipped — `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` is the only
// public Nemotron-VL and it ships as a raw PyTorch remote-code
// checkpoint that does not match FFAI's `text_config.model_type ==
// nemotron_h` layout. When an mlx-style conversion is cached, the
// suite auto-enables; until then each test exits early. No code
// change needed when a checkpoint lands — drop the snapshot into the
// HF cache and the suite goes green on the next run.

import Foundation
import Testing
@testable import FFAI

@Suite("NemotronH Vision Integration", .serialized)
struct NemotronHVisionIntegrationTests {

    static let modelId = "mlx-community/Llama-3.1-Nemotron-Nano-VL-8B-V1-4bit"

    /// True when an mlx-style Nemotron-VL conversion is already cached
    /// locally — checks the standard HF cache layout `models--<org>--<name>`
    /// plus the alternate `mlx-audio/` mirror. Returns `false` when no
    /// snapshot is present so the tests self-skip without attempting a
    /// network download for a model that doesn't exist on HF yet.
    private func nemotronVLIsCached() -> Bool {
        let candidateIds = [
            "mlx-community/Llama-3.1-Nemotron-Nano-VL-8B-V1-4bit",
            "mlx-community/Llama-3.1-Nemotron-Nano-VL-8B-V1-8bit",
            "mlx-community/Llama-3.1-Nemotron-Nano-VL-8B-V1-bf16",
            "mlx-community/Nemotron-12B-v2-VL-bf16",
            "mlx-community/Nemotron-12B-v2-VL-4bit",
        ]
        let home = FileManager.default.homeDirectoryForCurrentUser
        let hfRoot = home.appendingPathComponent(".cache/huggingface/hub")
        let fm = FileManager.default
        for id in candidateIds {
            let cacheName = "models--" + id.replacingOccurrences(of: "/", with: "--")
            let dir = hfRoot.appendingPathComponent(cacheName)
            if fm.fileExists(atPath: dir.path) { return true }
        }
        return false
    }

    @Test("load — Nemotron-VLM checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        try #require(nemotronVLIsCached(),
                     "Nemotron-VL checkpoint not cached locally — no mlx-style conversion published on HF yet")
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        // The checkpoint is a VLM — vlModel is present, .visionIn is
        // available, and the text backbone supports the splice.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.visionIn))
        #expect(m.engine.supportsEmbeddingInput)  // VLM splice prerequisite

        let vlm = try #require(m.vlModel)
        // The vision tower contributes a positive run of projected tokens.
        #expect(vlm.imageTokenCount > 0)
    }

    @Test("enable / disable .visionIn — runtime capability flip")
    func capabilityFlip() async throws {
        try #require(nemotronVLIsCached(),
                     "Nemotron-VL checkpoint not cached locally — no mlx-style conversion published on HF yet")
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        #expect(m.availableCapabilities.contains(.visionIn))
        m.disable(.visionIn)
        #expect(!m.isEnabled(.visionIn))
        m.enable(.visionIn)
        #expect(m.isEnabled(.visionIn))
    }

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        try #require(nemotronVLIsCached(),
                     "Nemotron-VL checkpoint not cached locally — no mlx-style conversion published on HF yet")
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "Nemotron-VLM checkpoint is not a VLM")

        // Build an image+text prompt: a run of `imageTokenCount`
        // image-placeholder tokens followed by a text question.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(text: "Describe this image.")
        let promptTokens = Array(repeating: imageTokenId,
                                 count: vlm.imageTokenCount) + questionTokens

        // A real photograph — the golden-retriever fixture.
        let image = try VLMTestSupport.dogImage()

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence first, then the content check: the caption should
        // mention a dog.
        expectCoherentOutput(generated, minTokens: 8,
                             label: "Nemotron-VLM image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Nemotron-VLM generated: \(text)")
        VLMTestSupport.expectMentionsDog(text, label: "Nemotron-VLM")
    }
}
