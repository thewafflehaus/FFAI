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
// ModelInfoTests — unit-test the ModelInfo struct itself.
//
// End-to-end coverage of `Model.info` lives in the integration tests
// (Gemma2 / Llama / Qwen3 etc. all assert at least their probed
// `hidden` / `nLayers` shape against the canonical config). This
// suite covers the struct's invariants in isolation:
//
//   • gqaFanOut computes correctly across MHA / GQA / MQA
//   • Sendable + every field round-trips
//   • Default-generation-parameters carries family-specific defaults

import Foundation
import Testing

@testable import FFAI

@Suite("ModelInfo struct")
struct ModelInfoTests {

    /// Canonical Llama 3.2 1B-style shape — GQA fan-out 32/8 = 4.
    @Test("gqaFanOut — vanilla GQA (32 heads / 8 KV heads = 4)")
    func gqaFanOutGqa() {
        let info = sampleModelInfo(nHeads: 32, nKVHeads: 8)
        #expect(info.gqaFanOut == 4)
    }

    /// MHA model: nHeads == nKVHeads → fan-out 1.
    @Test("gqaFanOut — MHA (no GQA)")
    func gqaFanOutMha() {
        let info = sampleModelInfo(nHeads: 16, nKVHeads: 16)
        #expect(info.gqaFanOut == 1)
    }

    /// MQA model: nKVHeads = 1 → fan-out = nHeads.
    @Test("gqaFanOut — MQA (single KV head)")
    func gqaFanOutMqa() {
        let info = sampleModelInfo(nHeads: 12, nKVHeads: 1)
        #expect(info.gqaFanOut == 12)
    }

    /// Guard against divide-by-zero when nKVHeads is somehow 0
    /// (shouldn't happen but the implementation clamps via `max(_, 1)`).
    @Test("gqaFanOut — defensive clamp on 0 KV heads")
    func gqaFanOutDefensiveClamp() {
        let info = sampleModelInfo(nHeads: 8, nKVHeads: 0)
        #expect(info.gqaFanOut == 8)
    }

    /// All fields propagate through the initializer with no munging.
    /// Smoke test that the struct stays a plain value carrier.
    @Test("field round-trip — every property reads back what was set")
    func fieldRoundTrip() {
        let params = GenerationParameters(
            maxTokens: 200, temperature: 0.7, topP: 0.95, topK: 40)
        let info = ModelInfo(
            modelId: "test/test-1b",
            architecture: "TestForCausalLM",
            modelType: "test",
            family: "TestModel",
            dtype: .bf16,
            hidden: 2048, nLayers: 24, nHeads: 16, nKVHeads: 4,
            headDim: 128, vocab: 32000, maxSeq: 8192,
            quantization: ModelConfig.QuantizationConfig(bits: 4, groupSize: 64),
            defaultGenerationParameters: params,
            availableCapabilities: [.textIn, .textOut],
            enabledCapabilities: [.textIn, .textOut],
            parameterCount: 1_234_567_890,
            parameterBytes: 800_000_000,
            bosTokenId: 1,
            eosTokenIds: [2, 107],
            tieWordEmbeddings: true,
            supportsEmbeddingInput: true,
            isVLM: false,
            imageTokenCount: nil
        )
        #expect(info.modelId == "test/test-1b")
        #expect(info.architecture == "TestForCausalLM")
        #expect(info.modelType == "test")
        #expect(info.family == "TestModel")
        #expect(info.dtype == .bf16)
        #expect(info.hidden == 2048)
        #expect(info.nLayers == 24)
        #expect(info.nHeads == 16)
        #expect(info.nKVHeads == 4)
        #expect(info.headDim == 128)
        #expect(info.vocab == 32000)
        #expect(info.maxSeq == 8192)
        #expect(info.quantization?.bits == 4)
        #expect(info.quantization?.groupSize == 64)
        #expect(info.defaultGenerationParameters.maxTokens == 200)
        #expect(info.defaultGenerationParameters.temperature == 0.7)
        #expect(info.defaultGenerationParameters.topP == 0.95)
        #expect(info.defaultGenerationParameters.topK == 40)
        #expect(info.availableCapabilities == [.textIn, .textOut])
        #expect(info.enabledCapabilities == [.textIn, .textOut])
        #expect(info.parameterCount == 1_234_567_890)
        #expect(info.parameterBytes == 800_000_000)
        #expect(info.bosTokenId == 1)
        #expect(info.eosTokenIds == [2, 107])
        #expect(info.tieWordEmbeddings == true)
        #expect(info.supportsEmbeddingInput == true)
        #expect(info.isVLM == false)
        #expect(info.imageTokenCount == nil)
        #expect(info.gqaFanOut == 4)
    }

    /// VLM variant: imageTokenCount carries through; isVLM flips on.
    @Test("VLM info — isVLM + imageTokenCount populated")
    func vlmInfo() {
        let info = sampleModelInfo(isVLM: true, imageTokenCount: 256)
        #expect(info.isVLM == true)
        #expect(info.imageTokenCount == 256)
    }

    /// Quantization absence is rendered as nil, not silent default.
    @Test("full-precision — quantization nil")
    func fullPrecision() {
        let info = sampleModelInfo(quantization: nil)
        #expect(info.quantization == nil)
    }

    // ── Helpers ────────────────────────────────────────────────────────

    /// Build a ModelInfo with overridable fields. Defaults match a
    /// generic small-Llama shape so individual tests only need to
    /// pass the field they're stressing.
    private func sampleModelInfo(
        nHeads: Int = 16,
        nKVHeads: Int = 4,
        quantization: ModelConfig.QuantizationConfig? =
            ModelConfig.QuantizationConfig(bits: 4, groupSize: 64),
        isVLM: Bool = false,
        imageTokenCount: Int? = nil
    ) -> ModelInfo {
        ModelInfo(
            modelId: "test/test",
            architecture: "TestForCausalLM",
            modelType: "test",
            family: "TestModel",
            dtype: .bf16,
            hidden: 2048, nLayers: 16, nHeads: nHeads, nKVHeads: nKVHeads,
            headDim: 128, vocab: 32000, maxSeq: 4096,
            quantization: quantization,
            defaultGenerationParameters: GenerationParameters(),
            availableCapabilities: [.textIn, .textOut],
            enabledCapabilities: [.textIn, .textOut],
            parameterCount: 1_000_000,
            parameterBytes: 4_000_000,
            bosTokenId: 1,
            eosTokenIds: [2],
            tieWordEmbeddings: false,
            supportsEmbeddingInput: true,
            isVLM: isVLM,
            imageTokenCount: imageTokenCount
        )
    }
}
