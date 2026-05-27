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
// Slow integration test for FastVLM (Apple's FastViTHD + mlp2x_gelu
// projector + Qwen2 text backbone, the `LlavaQwen2ForCausalLM` /
// `llava_qwen2` checkpoint).
//
// Verifies the FastVLM vision path end-to-end on a real checkpoint:
// the FastViTHD tower loads its reparameterized conv weights (including
// BN folding), the mlp2x_gelu projector maps 3072-dim feature tokens to
// the Qwen2 text hidden dim (896), the cross-modal splice injects the
// image tokens, and the fused stream decodes coherent text through the
// Qwen2-0.5B backbone.
//
// Uses the mlx-community FastVLM-0.5B-bf16 conversion. The checkpoint
// MUST be available at the standard HuggingFace cache path.
//
// DO NOT run this test directly with `swift test` — use `make test-integration`
// to serialize model loads and avoid GPU memory pressure.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "FastVLM Vision Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct FastVLMIntegrationTests {

    // mlx-community ships the bf16 conversion as the canonical FastVLM
    // VL checkpoint — it matches the test's docstring above and exercises
    // the FastViTHD vision tower's `loadPW2D` path on its native bf16
    // layout. The 4-bit variants (`ekryski/FastVLM-0.5B-4bit`,
    // `InsightKeeper/FastVLM-0.5B-MLX-4bit`) crash the vision-tower
    // loader at an out-of-range shape index because the pointwise-conv
    // helpers don't yet route quantized weights through `loadLinear` —
    // tracked separately. Until that lands, the bf16 conversion is the
    // contracted vision-correctness signal for FastVLM.
    static let modelId = "mlx-community/FastVLM-0.5B-bf16"

    @Test("load — FastVLM checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        // The checkpoint is a VLM — vlModel is present, .imageIn is
        // available, and the text backbone is a Qwen2-0.5B engine.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.engine.hidden == 896)  // Qwen2-0.5B text hidden
        #expect(m.engine.supportsEmbeddingInput)  // VLM splice prerequisite

        let vlm = try #require(m.vlModel)
        // 1024px input: 4× stem + 4 stride-2 PEs → 16×16 = 256 tokens.
        #expect(vlm.imageTokenCount == 256)
        // Image placeholder token id is -200 (FastVLM processor convention).
        #expect(vlm.imageTokenId == -200)
    }

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "FastVLM checkpoint is not a VLM")

        // Build an image+text prompt.
        // FastVLM's chat template puts <image> at the start of the user
        // message; the processor replaces <image> with token id -200.
        // Here we assemble the token stream directly: 256 image placeholders
        // followed by a tokenized question.
        let imageTokenId = vlm.imageTokenId  // -200
        let questionTokens = m.tokenizer.encode(
            text: "What animal is in the image?\nAssistant:")
        let promptTokens =
            Array(
                repeating: imageTokenId,
                count: vlm.imageTokenCount) + questionTokens

        // A real photograph — the golden-retriever fixture.
        let image = try VisionTestHelpers.dogImage()

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence first, then the content check.
        expectCoherentOutput(
            generated, minTokens: 4,
            label: "FastVLM image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("FastVLM generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "FastVLM")
    }
}
