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

// Unit tests for the Qwen 2.5-VL family (the
// windowed-attention ViT + Qwen 2.x text backbone
// `Qwen2_5_VLForConditionalGeneration` checkpoint).
//
// Offline — covers VL routing and `Qwen25VLVisionConfig.decode`, which
// turns the nested `vision_config` into the windowed-attention ViT
// geometry (depth, head dim, spatial-merge unit, full-attention block
// set) the vision tower is built from.
@Suite("Qwen25 Vision")
struct Qwen25VisionConfigTests {

    /// A representative `Qwen2_5_VLForConditionalGeneration` config.
    private func makeConfig() -> ModelConfig {
        let visionConfig: [String: Any] = [
            "depth": 32,
            "hidden_size": 1280,
            "intermediate_size": 3420,
            "out_hidden_size": 2048,
            "num_heads": 16,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "window_size": 112,
            "fullatt_block_indexes": [7, 15, 23, 31],
            "in_chans": 3,
        ]
        let textConfig: [String: Any] = [
            "model_type": "qwen2",
            "hidden_size": 2048,
            "num_hidden_layers": 36,
        ]
        let raw: [String: Any] = [
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "image_token_id": 151_655,
            "vision_config": visionConfig,
            "text_config": textConfig,
        ]
        return ModelConfig(architecture: "Qwen2_5_VLForConditionalGeneration",
                           modelType: "qwen2_5_vl", raw: raw)
    }

    @Test("routes as a vision-language checkpoint")
    func routesAsVisionLanguage() {
        let cfg = makeConfig()
        #expect(VisionLanguageArchitectures.isVisionLanguage(cfg))
        #expect(VisionLanguageArchitectures.architectures
            .contains("Qwen2_5_VLForConditionalGeneration"))
        #expect(Qwen25VL.defaultImageTokenId == 151_655)
        #expect(cfg.int("image_token_id") == 151_655)
    }

    @Test("vision_config decodes into windowed-attention ViT geometry")
    func visionConfigDecode() throws {
        let vc = try #require(makeConfig().subConfig("vision_config"))
        let parsed = try Qwen25VLVisionConfig.decode(vc)
        #expect(parsed.depth == 32)
        #expect(parsed.hidden == 1280)
        #expect(parsed.intermediate == 3420)
        #expect(parsed.outHidden == 2048)
        #expect(parsed.numHeads == 16)
        #expect(parsed.patchSize == 14)
        #expect(parsed.spatialMergeSize == 2)
        #expect(parsed.windowSize == 112)
        #expect(parsed.fullattBlockIndexes == Set([7, 15, 23, 31]))
        // Derived geometry.
        #expect(parsed.headDim == 80)           // 1280 / 16
        #expect(parsed.mergeUnit == 4)          // 2 × 2 patches per token
    }

    @Test("vision_config decode falls back to documented defaults")
    func visionConfigDefaults() throws {
        // Minimal config — only the required fields. `decode` fills
        // intermediate / out_hidden / window_size / temporal_patch_size.
        let minimal = ModelConfig(architecture: nil, modelType: nil, raw: [
            "depth": 16,
            "hidden_size": 1024,
            "num_heads": 8,
            "patch_size": 14,
            "spatial_merge_size": 2,
        ])
        let parsed = try Qwen25VLVisionConfig.decode(minimal)
        #expect(parsed.intermediate == 1024 * 4) // hidden * 4 fallback
        #expect(parsed.outHidden == 1024)        // hidden fallback
        #expect(parsed.windowSize == 112)        // documented default
        #expect(parsed.temporalPatchSize == 2)
        #expect(parsed.inChannels == 3)
        #expect(parsed.fullattBlockIndexes.isEmpty)
    }
}
