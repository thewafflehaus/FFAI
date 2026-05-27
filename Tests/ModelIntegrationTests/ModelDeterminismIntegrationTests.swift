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
// Integration tests covering FFAI greedy-decode determinism.  Loads
// Qwen3 1.7B bf16 (the same model the GoldenFixture tests use) and
// runs the SAME single-token forward pass three times back-to-back.
// Logs the top-5 logits and the argmax for each run.  If the logits —
// or even just the argmax — vary across runs, FFAI's greedy decode is
// nondeterministic at temperature=0, which is a correctness bug
// independent of any MLX parity gap.
//
// Skipped automatically if the network/checkpoint isn't available.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite(
    "Model Determinism Integration", .serialized,
    .enabled(
        if: IntegrationGroupGating.enableQuantizedSuites,
        IntegrationGroupGating.quantizedSkipReason)
)
struct ModelDeterminismIntegrationTests {

    @Test("forwardSample(BOS) returns the same token on three back-to-back calls")
    func forwardSampleIsDeterministic() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/Qwen3-1.7B-bf16")
        }
        let qwen3 = try #require(m.qwen3, "determinism smoke: expected Qwen3 engine")

        // Three back-to-back forwards from a fresh KV cache each time.
        // Each call should produce identical logits → identical argmax.
        var sampled: [Int] = []
        var topFivePerRun: [[(Int, Float)]] = []
        for _ in 0 ..< 3 {
            let caches = m.engine.makeLayerCaches()
            let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
            topFivePerRun.append(Sampling.topN(logits, n: 5))
            let token = qwen3.forwardSample(
                tokenId: 0, position: 0,
                caches: m.engine.makeLayerCaches(),
                device: .shared)
            sampled.append(token)
        }
        print("DETERMINISM sampled tokens: \(sampled)")
        for (i, top) in topFivePerRun.enumerated() {
            print("DETERMINISM run \(i) top-5: \(top)")
        }

        #expect(
            sampled[0] == sampled[1],
            "forwardSample drifted between run 0 and run 1")
        #expect(
            sampled[1] == sampled[2],
            "forwardSample drifted between run 1 and run 2")
    }

    /// Full multi-token generate, three back-to-back from a freshly-loaded
    /// model. If single-forward is deterministic but multi-token decode
    /// drifts, the nondeterminism lives in the cache-update / SDPA-attend
    /// pipeline (or the generate loop's bookkeeping).
    @Test("generate(prompt) returns the same token stream on three back-to-back calls")
    func multiTokenGenerateIsDeterministic() async throws {
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/Qwen3-1.7B-bf16")
        }

        var streams: [[Int]] = []
        for run in 0 ..< 3 {
            let result = try await m.generate(
                prompt: "The capital of France is",
                parameters: GenerationParameters(maxTokens: 8, temperature: 0)
            )
            print("MULTI-DETERMINISM run \(run): \(result.generatedTokens)")
            streams.append(result.generatedTokens)
        }
        #expect(
            streams[0] == streams[1],
            "multi-token generate drifted between run 0 and run 1")
        #expect(
            streams[1] == streams[2],
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
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load("mlx-community/Qwen3-1.7B-bf16")
        }
        let qwen3 = try #require(m.qwen3, "KV-cache determinism smoke: expected Qwen3 engine")

        // First 5 prompt tokens (no BOS for Qwen3): "The capital of France is".
        let promptTokens = m.tokenizer.encode(text: "The capital of France is")

        var hashes: [[UInt64]] = []
        for run in 0 ..< 3 {
            let caches = m.engine.makeLayerCaches()
            for (pos, t) in promptTokens.enumerated() {
                _ = qwen3.forwardSample(
                    tokenId: t, position: pos,
                    caches: caches, device: .shared)
            }
            // Hash each layer's K cache so we can see at which layer the
            // drift (if any) starts.
            //
            // KEY: hash only the WRITTEN portion of the cache, not the
            // full pre-allocated capacity. KVCache.kBuffer is shape
            // [nKVHeads, maxSeq, headDim] but `length` positions are
            // filled. For Qwen3-1.7B with maxSeq=32K and 5 prefill
            // tokens, hashing the whole 64 MB-per-layer buffer takes
            // gigabytes of byte iteration in Swift debug mode — the
            // test appeared to "hang" for tens of minutes. Hashing only
            // `length` rows of each head reduces it to ~10 KB per layer
            // and the assertion is unchanged (the unwritten suffix is
            // pre-zeroed at init, identical across runs anyway).
            var perLayer: [UInt64] = []
            for cache in caches {
                guard let kv = cache as? KVCache else {
                    perLayer.append(0)
                    continue
                }
                let len = kv.length
                let bytesPerRow = kv.headDim * kv.dtype.byteSize
                let bytesPerHeadCapacity = kv.maxSeq * bytesPerRow
                let raw = kv.kBuffer.buffer.contents()
                    .advanced(by: kv.kBuffer.offset)
                    .bindMemory(to: UInt8.self, capacity: kv.kBuffer.byteCount)
                var h: UInt64 = 1_469_598_103_934_665_603  // FNV-1a 64-bit offset
                // For each KV head, hash its filled prefix
                // `[h * maxSeq + 0 ..< h * maxSeq + length]` rows.
                for headIdx in 0 ..< kv.nKVHeads {
                    let headStart = headIdx * bytesPerHeadCapacity
                    let filledBytes = len * bytesPerRow
                    for i in 0 ..< filledBytes {
                        h ^= UInt64(raw[headStart + i])
                        h &*= 1_099_511_628_211
                    }
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
