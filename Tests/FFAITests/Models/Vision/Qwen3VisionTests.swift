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

// Unit tests for the Qwen 3-VL family (the full-attention
// ViT + Qwen 3 dense text backbone `Qwen3VLForConditionalGeneration`
// checkpoint).
//
// Offline — covers VL routing and `Qwen3VLVisionConfig.decode`, which
// turns the nested `vision_config` into the full-attention ViT geometry
// (depth, head dim, spatial-merge unit, learned position-table size).
@Suite("Qwen3 Vision")
struct Qwen3VisionConfigTests {

    /// A representative `Qwen3VLForConditionalGeneration` config.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "depth": 24,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "out_hidden_size": 2048,
            "num_heads": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 32 * 32,
            "in_channels": 3,
        ]
        let textConfig: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": 2048,
            "num_hidden_layers": 36,
        ]
        let raw: [String: Any] = [
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "image_token_id": 151_655,
            "vision_config": visionConfig,
            "text_config": textConfig,
        ]
        return ModelConfig(architecture: "Qwen3VLForConditionalGeneration",
                           modelType: "qwen3_vl", raw: raw)
    }

    @Test("routes as a vision-language checkpoint")
    func routesAsVisionLanguage() {
        let cfg = makeConfig()
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        #expect(VisionLanguageArchitectures.architectures
            .contains("Qwen3VLForConditionalGeneration"))
        #expect(Qwen3VL.defaultImageTokenId == 151_655)
    }

    @Test("vision_config decodes into full-attention ViT geometry")
    func visionConfigDecode() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        let parsed = try Qwen3VLVisionConfig.decode(vc)
        #expect(parsed.depth == 24)
        #expect(parsed.hidden == 1152)
        #expect(parsed.intermediate == 4304)
        #expect(parsed.outHidden == 2048)
        #expect(parsed.numHeads == 16)
        #expect(parsed.patchSize == 16)
        #expect(parsed.spatialMergeSize == 2)
        #expect(parsed.numPositionEmbeddings == 1024)
        // Derived geometry.
        #expect(parsed.headDim == 72)           // 1152 / 16
        #expect(parsed.mergeUnit == 4)          // 2 × 2 patches per token
    }

    @Test("vision_config decode falls back to documented defaults")
    func visionConfigDefaults() throws {
        let minimal = ModelConfig(architecture: nil, modelType: nil, raw: [
            "depth": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "patch_size": 16,
            "spatial_merge_size": 2,
        ])
        let parsed = try Qwen3VLVisionConfig.decode(minimal)
        #expect(parsed.intermediate == 768 * 4)         // hidden * 4 fallback
        #expect(parsed.outHidden == 768)                // hidden fallback
        #expect(parsed.numPositionEmbeddings == 32 * 32) // default 32×32 grid
        #expect(parsed.temporalPatchSize == 2)
        #expect(parsed.inChannels == 3)
    }
}
