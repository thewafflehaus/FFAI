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
// Slow integration test for Gemma 3 VL (the SigLIP + Gemma 3 text
// `Gemma3ForConditionalGeneration` checkpoint).
//
// Verifies the Phase 6.5 vision path end-to-end on a real checkpoint:
// the SigLIP vision tower loads into the shared `VisionEncoder`, the
// multi-modal projector pools + projects the patch grid, the cross-modal
// splice injects the image tokens, and the fused stream decodes coherent
// text through the Gemma 3 backbone.
//
// Uses the mlx-community 4B-it conversion (smallest published Gemma 3
// VLM). The checkpoint MUST load — a load failure fails the test.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Gemma3 Vision Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableVisionSuites,
        IntegrationGroupGating.visionSkipReason)
)
struct Gemma3VisionIntegrationTests {

    static let modelId = "mlx-community/gemma-3-4b-it-4bit"

    @Test("load — Gemma 3 VL checkpoint loads with vision capability")
    func loadVLCheckpoint() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }

        // The checkpoint is a VLM — vlModel is present, .imageIn is
        // available, and the text backbone is a Gemma 3 4B engine.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.engine.hidden == 2560)  // 4B text hidden
        #expect(m.engine.supportsEmbeddingInput)  // VLM splice prerequisite

        let vlm = try #require(m.vlModel)
        // SigLIP-896 / patch-14 → 64×64 patches, pooled 4×4 → 256 tokens.
        #expect(vlm.imageTokenCount == 256)
    }

    @Test("enable / disable .imageIn — runtime capability flip")
    func capabilityFlip() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        // A VL checkpoint can toggle .imageIn at runtime.
        #expect(m.availableCapabilities.contains(.imageIn))
        m.disable(.imageIn)
        #expect(!m.isEnabled(.imageIn))
        m.enable(.imageIn)
        #expect(m.isEnabled(.imageIn))
        // textIn / textOut are universal — disable is a no-op.
        m.disable(.textIn)
        #expect(m.isEnabled(.textIn))
    }

    @Test("image + text prompt — describes the dog photo")
    func imageTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.modelId)
        }
        let vlm = try #require(m.vlModel, "Gemma 3 VL checkpoint is not a VLM")

        // Build the prompt with the Gemma 3 chat template. HF's
        // `processing_gemma3.py` expands every `<image>` content item
        // into:
        //
        //     <start_of_image>(image_soft_token×N)<end_of_image>
        //
        // and the chat template wraps it INSIDE the user turn — not
        // before it. The previous version of this test passed a raw run
        // of `image_soft_token`s BEFORE `<start_of_turn>user` and without
        // any start_of_image / end_of_image markers, giving the model
        // no structural signal that those tokens were a single image
        // block. Result: a coherent but blind response — "Okay, I'll do
        // my best to describe the image. I'm sorry if my description is
        // difficult to assess!" — i.e. the model decoded a polite
        // disclaimer because it never saw the image.
        //
        // Token map (from the checkpoint's `config.json`):
        //   • `<start_of_image>` = `boi_token_index` → id 255999
        //   • `<end_of_image>`   = `eoi_token_index` → id 256000
        //   • `<image_soft_token>` = `image_token_index` → id 262144
        let imageTokenId = vlm.imageTokenId
        let header = m.tokenizer.encode(
            text: "<bos><start_of_turn>user\n<start_of_image>")
        let trailer = m.tokenizer.encode(
            text: "<end_of_image>Describe this image.<end_of_turn>\n"
                + "<start_of_turn>model\n")
        let promptTokens =
            header
            + Array(repeating: imageTokenId, count: vlm.imageTokenCount)
            + trailer

        // A real photograph — the golden-retriever fixture.
        let image = try VisionTestHelpers.dogImage()

        let generated = try vlm.generate(
            promptTokens: promptTokens, image: image,
            // 200-token default across VLM integration tests — chat
            // models often emit a polite preamble ("Here's a description
            // of the image:") before the actual caption, and 64-token
            // tests cut off mid-preamble.
            maxTokens: 200, eosTokenId: m.config.eosTokenId, eosTokenIds: m.config.eosTokenIds)

        // Coherence first, then the content check: the caption should
        // mention a dog.
        expectCoherentOutput(
            generated, minTokens: 8,
            label: "Gemma 3 VL image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Gemma 3 VL generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "Gemma 3 VL")
    }
}
