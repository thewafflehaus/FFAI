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
// Slow integration test for Qwen 3-VL-MoE (the Qwen3-VL ViT tower + the
// Qwen 3.5 mixture-of-experts hybrid text backbone, the
// `Qwen3VLMoeForConditionalGeneration` checkpoint).
//
// Verifies the vision path end-to-end on a real checkpoint: the
// Qwen3-VL vision tower loads, runs its full-attention + M-RoPE forward
// and patch-merger, the cross-modal splice injects the merged image
// tokens, and the fused stream decodes coherent text through the
// Qwen 3.5-MoE backbone (which now supports embedding-input forward for
// the splice).
//
// Uses the mlx-community Qwen3-VL-30B-A3B 4-bit conversion — the only
// published Qwen3-VL-MoE. The checkpoint MUST load — a load failure
// fails the test.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Qwen35 Vision Integration", .serialized)
struct Qwen35VisionIntegrationTests {

    static let modelId = "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"

    @Test("load — Qwen 3-VL-MoE checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        // The checkpoint is a VLM — vlModel is present, .imageIn is
        // available, and the text backbone supports the splice.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.engine.supportsEmbeddingInput)  // VLM splice prerequisite

        let vlm = try #require(m.vlModel)
        // The vision tower contributes a positive run of merged tokens.
        #expect(vlm.imageTokenCount > 0)
    }

    @Test("enable / disable .imageIn — runtime capability flip")
    func capabilityFlip() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        #expect(m.availableCapabilities.contains(.imageIn))
        m.disable(.imageIn)
        #expect(!m.isEnabled(.imageIn))
        m.enable(.imageIn)
        #expect(m.isEnabled(.imageIn))
    }

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "Qwen 3-VL-MoE checkpoint is not a VLM")

        // Build an image+text prompt: a run of `imageTokenCount`
        // <|image_pad|> placeholders followed by a text question.
        let imageTokenId = vlm.imageTokenId
        let questionTokens = m.tokenizer.encode(
            text: "<|im_start|>user\nDescribe this image.<|im_end|>\n"
                + "<|im_start|>assistant\n")
        let promptTokens = Array(repeating: imageTokenId,
                                 count: vlm.imageTokenCount) + questionTokens

        // A real photograph — the golden-retriever fixture.
        let image = try VisionTestHelpers.dogImage()

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence first, then the content check: the caption should
        // mention a dog.
        expectCoherentOutput(generated, minTokens: 8,
                             label: "Qwen 3-VL-MoE image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 3-VL-MoE generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "Qwen 3-VL-MoE")
    }
}
