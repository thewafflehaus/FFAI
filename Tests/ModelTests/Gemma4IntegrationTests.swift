// Slow integration test for Gemma 4. First-light verification that the
// family's architectural quirks compose into coherent generated text on
// a real checkpoint:
//
//   • two attention geometries (sliding head_dim 256 + global 512)
//   • ProportionalRoPE on the global layers
//   • value RMSNorm, per-head q/k norms, plain RMSNorm (no Gemma
//     `(1 + weight)` fold — Gemma 4 dropped that convention)
//   • leading `<bos>` prefix (Gemma 4's tokenizer post-processor
//     does not add one — `Gemma4Model.requiresLeadingBOS`)
//   • Per-Layer Embeddings (PLE) — E2B / E4B are PLE variants
//   • per-layer learned scalar
//   • sqrt(hidden) embed scale, GELU MLP, tied embeddings
//   • final logit soft-capping
//
// The smallest Gemma 4 checkpoint on mlx-community is the E2B size,
// which is itself a PLE (Gemma4E) variant — there is no PLE-free dense
// checkpoint below 31B and no MoE checkpoint below 26B, so the dense
// and MoE variants are guard-skipped here (no small/raw checkpoint).
// Tests are skipped entirely if the checkpoint is not available
// locally.

import Foundation
import Testing
@testable import FFAI

@Suite("Gemma 4 integration", .serialized)
struct Gemma4IntegrationTests {

    @Test("Gemma4E (E2B): load + greedy generate produces coherent output")
    func loadAndGenerateE2B() async throws {
        let modelId = "mlx-community/gemma-4-e2b-it-bf16"
        // Open-ended prompt: a short factual question ("The capital of
        // France is") elicits a 2-token answer ("Paris.") and the model
        // correctly stops, which can't exercise the coherence checker's
        // `minTokens` budget. A story opener keeps the model generating
        // sustained prose across every architectural path (sliding +
        // global attention, PLE, KV sharing, soft-capping).
        //
        // A modest token budget keeps the run bounded: the 512-wide
        // global-attention layers run a host-side SDPA whose cost grows
        // with the KV length, so a long generation is expensive.
        let prompt = "Once upon a time, in a quiet village"
        let maxTokens = 24

        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Gemma 4 E2B integration test skipped: \(error)")
            return
        }

        // E2B canonical shapes (from HF text_config):
        //   hidden 1536, 35 layers, 8 heads, sliding head_dim 256,
        //   1 sliding KV head.
        #expect(m.engine.hidden == 1536)
        #expect(m.engine.nLayers == 35)
        #expect(m.engine.nHeads == 8)
        #expect(m.engine.headDim == 256)

        // Resolved as the PLE (Gemma4E) variant.
        let gemma4 = m.engine as? Gemma4Model
        #expect(gemma4 != nil, "expected Gemma4Model engine")
        #expect(gemma4?.ple != nil, "E2B is a PLE variant — Gemma4PLE must be present")

        // Per-layer cache geometry: sliding layers windowed, global
        // layers unbounded. E2B layer_types: every 5th layer (idx 4, 9,
        // 14, 19, 24, 29, 34) is full_attention ⇒ 7 global, 28 sliding.
        let caches = m.engine.makeLayerCaches()
        var slidingCount = 0
        var globalCount = 0
        for (i, c) in caches.enumerated() {
            guard let kv = c as? any KVCacheProtocol else { continue }
            let isGlobal = (i + 1) % 5 == 0
            switch kv.eviction {
            case .window:
                #expect(!isGlobal, "layer \(i): expected global (.unbounded)")
                slidingCount += 1
            case .unbounded:
                #expect(isGlobal, "layer \(i): expected sliding (.window)")
                globalCount += 1
            }
        }
        #expect(globalCount == 7)
        #expect(slidingCount == 28)

        // Greedy decode is verified coherent: the incoherent-output bug
        // was a missing leading `<bos>` token. Gemma 4 is BOS-critical,
        // but unlike Gemma 3 its `tokenizer.json` post-processor's
        // `single` template is bare (no `<bos>` special token), so
        // `Tokenizer.encode` returned no BOS and the residual stream was
        // subtly wrong from token 0. `Gemma4Model.requiresLeadingBOS`
        // now drives `Generate.encodePrompt` to prepend it. See
        // `Sources/FFAI/Generate.swift` and `Models/Gemma4.swift`.
        //
        // NOTE — slow: the 512-wide global-attention layers run a
        // host-side SDPA (~60 s/token in a debug build), so this single
        // 24-token generation takes ~30 min. A 512-wide `Ops.sdpaDecode`
        // specialization is the follow-up that brings it back to a
        // routine-cost integration test.

        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: maxTokens, temperature: 0)
        )
        print("Gemma 4 E2B decoded: \"\(prompt)\(result.text)\"")
        #expect(result.tokensPerSecond > 0)
        expectCoherentOutput(result.generatedTokens, minTokens: maxTokens,
                             label: "Gemma 4 E2B-it bf16")
    }

    @Test("Gemma4Dense (31B): load + greedy generate produces coherent output")
    func loadAndGenerateDense() async throws {
        // The smallest non-PLE dense Gemma 4 is 31B — only available as
        // a 4-/8-bit quantization. Guard-skip if not present locally.
        let modelId = "mlx-community/gemma-4-31b-it-4bit"
        let prompt = "Once upon a time, in a quiet village"

        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Gemma 4 31B integration test skipped (no small/raw checkpoint): \(error)")
            return
        }
        let gemma4 = m.engine as? Gemma4Model
        #expect(gemma4 != nil)
        #expect(gemma4?.ple == nil, "31B is a dense (non-PLE) variant")

        // Generation stays gated behind `GEMMA4_RUN_GENERATION` for the
        // dense 31B variant — *not* a coherence issue (the BOS fix that
        // un-gated the E2B test applies here too), but a separate
        // dense-path limitation: the 31B `finalNorm` is an `RMSNorm`
        // over hidden=5376, and `Ops.rmsNorm`'s single-row kernel caps
        // at n=4096 (1024-thread × 4 elements). It fatal-errors before
        // emitting a token. The fix is a chunked / row-wise final-norm
        // path for >4096-wide hidden states; until then this generation
        // check is opt-in so it cannot crash the CI suite process.
        guard ProcessInfo.processInfo.environment["GEMMA4_RUN_GENERATION"] == "1" else {
            print("Gemma 4 31B generation check skipped " +
                  "(set GEMMA4_RUN_GENERATION=1 to run — dense-path " +
                  "rmsNorm n=5376 cap, see comment).")
            return
        }
        let result = try await m.generate(
            prompt: prompt,
            parameters: GenerationParameters(maxTokens: 48, temperature: 0)
        )
        expectCoherentOutput(result.generatedTokens, minTokens: 48,
                             label: "Gemma 4 31B-it 4bit")
    }
}
