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
// FastVLMTests — root-file unit tests for `Sources/FFAI/Models/FastVLM.swift`.
//
// Offline. The vision config + dispatch routing surface is exhaustively
// covered by `Tests/FFAITests/Models/Vision/FastVLMTests.swift`. This
// file focuses on the family-root constants + the `FastVLMError`
// description shape + the standalone `fastVLMTextConfigWithDefaults`
// helper, none of which are exercised by the Vision test.

import Foundation
import Testing

@testable import FFAI

@Suite("FastVLM Family Root — error + helper")
struct FastVLMRootTests {

    @Test("FastVLM advertises the canonical model_type + architecture")
    func registration() {
        #expect(FastVLM.modelTypes.contains("llava_qwen2"))
        #expect(FastVLM.architectures.contains("LlavaQwen2ForCausalLM"))
    }

    @Test("default image_token_id is the documented -200 sentinel")
    func imageTokenId() {
        #expect(FastVLM.defaultImageTokenId == -200)
    }

    @Test("FastVLMError stringifies every case with its payload")
    func errorDescriptions() {
        #expect(FastVLMError.missingConfig.description.contains("FastVLM"))
        #expect(FastVLMError.missingTensor("foo").description.contains("foo"))
    }

    @Test("fastVLMTextConfigWithDefaults supplies Qwen 2-0.5B fallbacks")
    func textDefaultsHelper() {
        // An empty raw dict should fall back to every documented Qwen2
        // default so the LlamaDense loader sees a complete text config.
        let merged = fastVLMTextConfigWithDefaults([:], vocabFallback: nil)
        #expect(merged["num_attention_heads"] as? Int == 14)
        #expect(merged["num_key_value_heads"] as? Int == 2)
        #expect(merged["head_dim"] as? Int == 64)
        #expect(merged["vocab_size"] as? Int == 151_936)
        #expect(merged["tie_word_embeddings"] as? Bool == true)
    }

    @Test("fastVLMTextConfigWithDefaults honours checkpoint overrides")
    func textDefaultsOverride() {
        let raw: [String: Any] = [
            "num_attention_heads": 24,
            "vocab_size": 200_000,
        ]
        let merged = fastVLMTextConfigWithDefaults(raw, vocabFallback: 999)
        // Checkpoint-declared fields win over both the defaults AND the
        // explicit vocab fallback.
        #expect(merged["num_attention_heads"] as? Int == 24)
        #expect(merged["vocab_size"] as? Int == 200_000)
    }

    @Test("fastVLMTextConfigWithDefaults strips nested VLM-only keys")
    func textDefaultsStripsVLMKeys() {
        let raw: [String: Any] = [
            "vision_config": ["foo": "bar"],
            "mm_projector_type": "mlp2x_gelu",
            "mm_hidden_size": 3072,
            "image_token_index": -200,
            "image_token_id": -200,
            "model_type": "llava_qwen2",
            "architectures": ["LlavaQwen2ForCausalLM"],
        ]
        let merged = fastVLMTextConfigWithDefaults(raw, vocabFallback: nil)
        #expect(merged["vision_config"] == nil)
        #expect(merged["mm_projector_type"] == nil)
        #expect(merged["mm_hidden_size"] == nil)
        #expect(merged["image_token_id"] == nil)
        #expect(merged["image_token_index"] == nil)
        #expect(merged["architectures"] == nil)
    }

    @Test("vocabFallback only fires when no checkpoint vocab_size is present")
    func textDefaultsVocabFallback() {
        let merged = fastVLMTextConfigWithDefaults([:], vocabFallback: 123_456)
        #expect(merged["vocab_size"] as? Int == 123_456)
    }
}
