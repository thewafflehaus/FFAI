// Smoke test for FFAI greedy-decode determinism.  Loads Qwen3 1.7B bf16
// (the same model the GoldenFixture tests use) and runs the SAME
// single-token forward pass three times back-to-back.  Logs the top-5
// logits and the argmax for each run.  If the logits — or even just the
// argmax — vary across runs, FFAI's greedy decode is nondeterministic at
// temperature=0, which is a correctness bug independent of any MLX
// parity gap.
//
// Skipped automatically if the network/checkpoint isn't available.

import Foundation
import Testing
@testable import FFAI

@Suite("FFAI determinism smoke", .serialized)
struct DeterminismSmokeTests {

    @Test("forwardSample(BOS) returns the same token on three back-to-back calls")
    func forwardSampleIsDeterministic() async throws {
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load("mlx-community/Qwen3-1.7B-bf16") }
        } catch {
            print("determinism smoke skipped: \(error)")
            return
        }
        guard let qwen3 = m.qwen3 else {
            print("determinism smoke: expected Qwen3 engine")
            return
        }

        // Three back-to-back forwards from a fresh KV cache each time.
        // Each call should produce identical logits → identical argmax.
        var sampled: [Int] = []
        var topFivePerRun: [[(Int, Float)]] = []
        for _ in 0..<3 {
            let caches = m.engine.makeLayerCaches()
            let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
            topFivePerRun.append(Sampling.topN(logits, n: 5))
            let token = qwen3.forwardSample(tokenId: 0, position: 0,
                                            caches: m.engine.makeLayerCaches(),
                                            device: .shared)
            sampled.append(token)
        }
        print("DETERMINISM sampled tokens: \(sampled)")
        for (i, top) in topFivePerRun.enumerated() {
            print("DETERMINISM run \(i) top-5: \(top)")
        }

        #expect(sampled[0] == sampled[1],
                "forwardSample drifted between run 0 and run 1")
        #expect(sampled[1] == sampled[2],
                "forwardSample drifted between run 1 and run 2")
    }

    /// Full multi-token generate, three back-to-back from a freshly-loaded
    /// model. If single-forward is deterministic but multi-token decode
    /// drifts, the nondeterminism lives in the cache-update / SDPA-attend
    /// pipeline (or the generate loop's bookkeeping).
    @Test("generate(prompt) returns the same token stream on three back-to-back calls")
    func multiTokenGenerateIsDeterministic() async throws {
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load("mlx-community/Qwen3-1.7B-bf16") }
        } catch {
            print("multi-token determinism smoke skipped: \(error)")
            return
        }

        var streams: [[Int]] = []
        for run in 0..<3 {
            let result = try await m.generate(
                prompt: "The capital of France is",
                parameters: GenerationParameters(maxTokens: 8, temperature: 0)
            )
            print("MULTI-DETERMINISM run \(run): \(result.generatedTokens)")
            streams.append(result.generatedTokens)
        }
        #expect(streams[0] == streams[1],
                "multi-token generate drifted between run 0 and run 1")
        #expect(streams[1] == streams[2],
                "multi-token generate drifted between run 1 and run 2")
    }

    /// After two prefill forwards (positions 0 and 1) on the SAME model,
    /// dump a hash of the layer-0 K cache contents.  If the hash is
    /// deterministic across three back-to-back attempts, the KV update
    /// kernel is reliable and we should look further down the pipeline
    /// (SDPA, post-attention).  If the hash varies, the KV write itself
    /// is nondeterministic.
    @Test("layer-0 K cache contents are deterministic after 2 prefill forwards")
    func kvCacheIsDeterministic() async throws {
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load("mlx-community/Qwen3-1.7B-bf16") }
        } catch {
            print("KV-cache determinism smoke skipped: \(error)")
            return
        }
        guard let qwen3 = m.qwen3 else {
            print("KV-cache determinism smoke: expected Qwen3 engine"); return
        }

        // First 5 prompt tokens (no BOS for Qwen3): "The capital of France is".
        let promptTokens = m.tokenizer.encode(text: "The capital of France is")

        var hashes: [[UInt64]] = []
        for run in 0..<3 {
            let caches = m.engine.makeLayerCaches()
            for (pos, t) in promptTokens.enumerated() {
                _ = qwen3.forwardSample(tokenId: t, position: pos,
                                         caches: caches, device: .shared)
            }
            // Hash each layer's K cache so we can see at which layer the
            // drift (if any) starts.
            var perLayer: [UInt64] = []
            for cache in caches {
                guard let kv = cache as? KVCache else {
                    perLayer.append(0); continue
                }
                let raw = kv.kBuffer.buffer.contents().bindMemory(
                    to: UInt8.self, capacity: kv.kBuffer.byteCount)
                var h: UInt64 = 1469598103934665603  // FNV-1a 64-bit offset
                for i in 0..<kv.kBuffer.byteCount {
                    h ^= UInt64(raw[i])
                    h &*= 1099511628211
                }
                perLayer.append(h)
            }
            print("KV-HASH run \(run): \(perLayer.prefix(3))…\(perLayer.suffix(2))")
            hashes.append(perLayer)
        }
        #expect(hashes[0] == hashes[1], "KV cache drifted between run 0 and 1")
        #expect(hashes[1] == hashes[2], "KV cache drifted between run 1 and 2")
    }
}
