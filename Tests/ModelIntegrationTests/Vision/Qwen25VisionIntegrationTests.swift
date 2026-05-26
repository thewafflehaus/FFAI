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
// Slow integration test for Qwen 2.5-VL (the dynamic-resolution
// windowed-attention ViT + Qwen 2.x text backbone
// `Qwen2_5_VLForConditionalGeneration` checkpoint).
//
// Verifies the vision path end-to-end on a real checkpoint: the
// Qwen 2.5-VL vision tower loads, runs its windowed-attention + M-RoPE
// forward and patch-merger, the cross-modal splice injects the merged
// image tokens, and the fused stream decodes coherent text through the
// Qwen 2.x backbone (routed through the Llama dense engine, which now
// supports embedding-input forward for the splice).
//
// Uses the mlx-community 3B-Instruct conversion (smallest published
// Qwen 2.5-VL). The checkpoint MUST load — a load failure fails the test.
// Now also covers the video path (merged from the prior
// Qwen25VisionVideoIntegrationTests.swift); the video flow requires the
// 7B-Instruct-4bit conversion because it ships the video tokenizer.

import Foundation
import Testing
@testable import FFAI
import TestHelpers

@Suite("Qwen2.5 Vision Integration (image + video)", .serialized)
struct Qwen25VisionIntegrationTests {

    static let modelId = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

    /// The smallest published Qwen 2.5-VL conversion that ships the
    /// video tokenizer (`<|video_pad|>` / `<|vision_start|>` /
    /// `<|vision_end|>`) — the 7B-Instruct-4bit mlx-community release.
    static let videoModelId = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

    /// Number of evenly-spaced frames to pull out of `cat.mp4`. Must be
    /// a multiple of `temporal_patch_size` (2) so the vision tower
    /// doesn't have to pad with the last frame — keeps the placeholder
    /// arithmetic exact.
    static let frameCount = 8

    @Test("load — Qwen 2.5-VL checkpoint loads with vision capability")
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
        let vlm = try #require(m.vlModel, "Qwen 2.5-VL checkpoint is not a VLM")

        // Build an image+text prompt: a run of `imageTokenCount`
        // <|image_pad|> placeholders wrapped in vision-start / vision-end
        // markers, followed by a text question. Qwen's chat template
        // normally expands the placeholder; here we assemble it directly.
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
                             label: "Qwen 2.5-VL image+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 2.5-VL generated: \(text)")
        VisionTestHelpers.expectMentionsDog(text, label: "Qwen 2.5-VL")
    }

    // ─── Video ───────────────────────────────────────────────────────────

    @Test("load — checkpoint reports .videoIn capability")
    func loadVLCheckpointVideo() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.videoModelId)
        }
        // The Qwen 2.5-VL family declares text + vision + video.
        #expect(m.vlModel != nil)
        #expect(m.availableCapabilities.contains(.imageIn))
        #expect(m.availableCapabilities.contains(.videoIn))

        let vlm = try #require(m.vlModel)
        // The video splice needs the family loader to thread
        // `video_token_id` through to VisionModel.init.
        #expect(vlm.videoTokenId != nil)
    }

    @Test("video + text prompt — describes the cat clip")
    func videoTextGeneration() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(Self.videoModelId)
        }
        let vlm = try #require(m.vlModel, "Qwen 2.5-VL checkpoint is not a VLM")
        let videoTokenId = try #require(vlm.videoTokenId)

        // Pull `frameCount` evenly-spaced frames from cat.mp4.
        let frames = try VisionTestHelpers.catVideoFrames(maxFrames: Self.frameCount)

        // The merged-token-per-temporal-patch count is the same as one
        // image's merged token count — the vision tower repeats the
        // spatial grid per temporal patch. With `temporal_patch_size`
        // = 2 and 8 frames, the splice substitutes 4 × `imageTokenCount`
        // rows.
        let temporalPatchSize = 2
        let videoTokenCount = (frames.count / temporalPatchSize) * vlm.imageTokenCount

        // Build the standard Qwen 2.5-VL video prompt:
        //   <|im_start|>user\n<|vision_start|><|video_pad|>...<|vision_end|>What's in this video?<|im_end|>\n
        //   <|im_start|>assistant\n
        // We assemble the placeholder run directly (Qwen's chat
        // template normally expands `<|video_pad|>` to the right count
        // from frame metadata).
        let preTokens = m.tokenizer.encode(
            text: "<|im_start|>user\n<|vision_start|>")
        let postTokens = m.tokenizer.encode(
            text: "<|vision_end|>What's in this video?<|im_end|>\n"
                + "<|im_start|>assistant\n")
        let promptTokens = preTokens
            + Array(repeating: videoTokenId, count: videoTokenCount)
            + postTokens

        let generated = try vlm.generate(
            promptTokens: promptTokens, videoFrames: frames,
            maxTokens: 200,
            eosTokenId: m.config.eosTokenId,
            eosTokenIds: m.config.eosTokenIds)

        // Coherence first, then the content check: the caption should
        // mention a cat (or kitten — model verbosity varies).
        expectCoherentOutput(generated, minTokens: 8,
                             label: "Qwen 2.5-VL video+text")
        let text = m.tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        print("Qwen 2.5-VL video generated: \(text)")
        VisionTestHelpers.expectMentionsCat(text, label: "Qwen 2.5-VL video")
    }
}
