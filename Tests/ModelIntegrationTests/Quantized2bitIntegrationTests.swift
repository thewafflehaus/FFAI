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
// End-to-end test: download FFAI's own 2-bit Qwen3.5-0.8B conversion
// (`ekryski/Qwen3.5-0.8B-2bit`, produced via `ffai convert --bits 2
// --quantize-embeddings`) and exercise the dequant_gemv_int2 +
// dequant_gather_int2 + mt_qmm_mma_int2 kernels end-to-end through the
// Qwen3.5-VL loader and centered-RMSNorm fold path.
//
// ── No coherence assertion ──
// Pure 2-bit quantization at 0.8B parameters is below the threshold
// where this architecture retains coherent decode. `mlx-community` does
// not publish a pure 2-bit Qwen3.5-0.8B for the same reason (their
// 0.8B offering is `Qwen3.5-0.8B-mixed_2_6`). Other quantization
// integration tests (3/4/5/6/8-bit, in this same directory) keep their
// `expectCoherentOutput` checks because those bit-widths *do* produce
// coherent text at this size. This 2-bit test deliberately omits that
// assertion — it's a kernel-path gate, not a quality gate.
//
// What this test does assert:
//   * the loader picks the Qwen3.5-VL dispatch path (vision_config
//     present + `model.visual.*` tensors present)
//   * `bits == 2` round-trips through the config parser
//   * generation produces the requested token count with finite logits
//     (no NaN/Inf — would surface a dispatch-shape or dequant-arithmetic
//     bug in the int2 kernels) at production-realistic shapes
//     (hidden=1024, nLayers=24, headDim=256, vocab=248320).

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("Qwen3.5-VL 2-bit Integration", .serialized)
struct Quantized2bitIntegrationTests {

    @Test("load + generate runs int2 kernels end-to-end (no coherence check)")
    func loadAndGenerate() async throws {
        let modelId = "ekryski/Qwen3.5-0.8B-2bit"
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 60

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        #expect(m.config.quantization?.bits == 2)
        #expect(m.config.quantization?.groupSize == 64)

        // Qwen3.5-0.8B (text_config): hidden=1024, nLayers=24, nHeads=8,
        // nKVHeads=2, headDim=256. Confirms the Qwen3.5-VL loader bound
        // weights through the language_model.* sub-tree correctly.
        #expect(m.engine.hidden == 1024)
        #expect(m.engine.nLayers == 24)
        #expect(m.engine.nHeads == 8)
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.headDim == 256)

        // Forward at position 0 — exercises dequant_gemv_int2 (per-token
        // linear projections) and dequant_gather_int2 (embedding lookup).
        let caches = m.engine.makeLayerCaches()
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        for (_, score) in top { #expect(score.isFinite, "int2 logit non-finite: \(score)") }

        // Greedy generate — exercises the int2 dispatch in the steady-
        // state decode loop. No coherence assertion (see file header);
        // we only verify the dispatch completes and returns the
        // requested token count without NaN/Inf in throughput stats.
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)
        #expect(result.generatedTokens.count == maxTokens)
    }
}
