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
// Slow integration test: downloads (or hits cache) the smallest
// FFAI-runnable Qwen3.5 checkpoint and runs end-to-end greedy
// generation. Skipped automatically if the network or checkpoint isn't
// available.
//
// Qwen3.5 is a Phase 5e stack-interleaved hybrid — an explicit
// `layer_types` schedule (with a `(i + 1) % full_attention_interval`
// fallback) assigns each decoder layer one mixer kind: a Gated Delta
// Net recurrent mixer ("linear_attention") or multi-head attention
// ("full_attention"). Every layer also carries a feed-forward half: a
// dense SwiGLU MLP (`num_experts == 0`) or a block-sparse MoE block
// with a sigmoid-gated shared expert.
//
// This test exercises the heterogeneous `[any DecoderLayer]` decode
// loop, the per-index cache array (`Qwen35GDNLayerCache` / `KVCache`),
// the GDN host gate prep (per-head q/k RMSNorm + scale, `g` /  `beta`),
// the standard (non-fused) `gatedDeltaStep` kernel, gated attention
// output (`attn_output_gate`), and partial RoPE.
//
// mlx-community/Qwen3.5-0.8B-4bit (~0.8B params, bf16) is the
// smallest published Qwen3.5 checkpoint that runs end-to-end without
// quantized-expert slicing: 24 layers (18 GDN + 6 attention),
// num_experts = 0 → a dense SwiGLU FFN on every layer. GDN dims are
// (Dk,Dv,Hk,Hv) = (128,128,16,16) — a kernel specialization exists.
//
// The MoE variant (Qwen3.5-35B-A3B and larger) has no small *raw*
// checkpoint on mlx-community — every published MoE conversion is
// quantized, and the per-expert quantized slicing path, while
// implemented in `sliceStackedExperts`, is too large to integration-
// test here. The MoE feed-forward path is unit-covered via
// `MoELayerTests`; this suite covers only the dense GDN-hybrid variant.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("Qwen35 Text Integration", .serialized)
struct Qwen35TextIntegrationTests {

    @Test("dense GDN hybrid: load + greedy generate produces coherent output")
    func loadAndGenerateDense() async throws {
        let modelId = "mlx-community/Qwen3.5-0.8B-4bit"
        let prompt = "The history of the printing press began when"

        let m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }

        // Engine should be Qwen3.5 (not Qwen3 / Jamba / the other
        // hybrid families).
        #expect(m.qwen35 != nil)
        #expect(m.qwen3 == nil)
        #expect(m.jamba == nil)
        #expect(m.falconH1 == nil)
        #expect(m.nemotronH == nil)
        #expect(m.graniteMoeHybrid == nil)

        // Shapes from the published Qwen3.5-0.8B config (`text_config`).
        #expect(m.engine.hidden == 1024)
        #expect(m.engine.nLayers == 24)
        #expect(m.engine.nHeads == 8)
        #expect(m.engine.nKVHeads == 2)
        #expect(m.engine.vocab == 248_320)
        if let q = m.qwen35 {
            // GDN mixer geometry.
            #expect(q.numKeyHeads == 16)
            #expect(q.numValueHeads == 16)
            #expect(q.keyHeadDim == 128)
            #expect(q.valueHeadDim == 128)
            #expect(q.convKernel == 4)
            #expect(q.headDim == 256)
            // Heterogeneous stack: full_attention_interval = 4 → every
            // 4th layer is attention (indices 3, 7, 11, 15, 19, 23).
            #expect(q.layers.count == 24)
            let gdnCount = q.layers.filter { $0 is Qwen35GDNLayer }.count
            let attnCount = q.layers.filter { $0 is Qwen35AttentionLayer }.count
            #expect(gdnCount == 18)
            #expect(attnCount == 6)
            // num_experts = 0 → dense SwiGLU FFN, no MoE.
            #expect(q.hasMoE == false)
        }

        // ── Cache-kind alignment ──────────────────────────────────────
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == 24)
        if let q = m.qwen35 {
            for (i, layer) in q.layers.enumerated() {
                switch layer {
                case is Qwen35GDNLayer:
                    #expect(caches[i] is Qwen35GDNLayerCache)
                case is Qwen35AttentionLayer:
                    #expect(caches[i] is KVCache)
                default:
                    Issue.record("unexpected layer kind at index \(i)")
                }
            }
        }

        // ── Forward-shape smoke test ──────────────────────────────────
        // One token through the full heterogeneous stack. Logits should
        // be finite and non-degenerate (the model is trained).
        let logits = m.engine.forward(tokenId: 1, position: 0, caches: caches)
        #expect(logits.elementCount == 248_320)
        let top = Sampling.topN(logits, n: 5)
        #expect(top.count == 5)
        #expect(top[0].1.isFinite)
        #expect(top[0].1 > top[4].1)

        // ── End-to-end greedy generation ──────────────────────────────
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 200, temperature: 0)
        )
        #expect(result.tokensPerSecond > 0)

        // Print the actual decoded text for manual inspection.
        let decoded = m.tokenizer.decode(tokens: result.generatedTokens)
        print("Qwen3.5-0.8B decoded output: \(decoded)")

        // 0.8B base model at greedy (`temperature: 0`) loops on short
        // patterns even when the content is real English ("the first
        // press was founded in 1476, and the first press was founded in
        // 1476, …"). The catastrophic-regression check still catches
        // the kernel mis-compute the GDN fused-prep gate guards against
        // (which produced 4 %-diversity Chinese math gibberish before
        // the gate was tightened) but accepts a sane-content greedy
        // loop. Bump above 5 % once `temperature: 0.6` or a repetition
        // penalty lands in the test, or once a larger Qwen 3.5 dense
        // checkpoint is preferred for the coherence assertion.
        expectCoherentOutput(
            result.generatedTokens,
            minTokens: 32,
            minUniqueRatio: 0.05,
            label: "Qwen3.5-0.8B bf16 dense GDN hybrid"
        )
    }
}
