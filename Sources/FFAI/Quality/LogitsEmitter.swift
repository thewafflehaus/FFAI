// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// LogitsEmitter — drives a per-token forward loop over a corpus and
// returns the per-position full-vocab logit traces needed by
// `KLDivergence.positionMetrics`. The pair (baseline trace, codec
// trace) is the input to the regression gate for every TQ+ / AURA
// quality port.

import Foundation
import Metal

public enum LogitsEmitter {

    /// Run `tokenIds` through `model` one token at a time, returning
    /// the post-`lmHead` logits at every position as `[T, vocab]`
    /// `Float`s. The caller owns the model + decides which KV cache
    /// type to back it with (fp16 baseline vs AURA codec).
    ///
    /// **Cost:** one CPU-side `commit` + `waitUntilCompleted` per
    /// token. Use small corpora (a few hundred tokens) for KLD
    /// validation — at 1.7B model + 256-tok corpus this completes in
    /// ~10s on M5 Max.
    ///
    /// `maxSeq` is passed through to the cache allocator. Keep it
    /// only as large as the input so cache allocation stays bounded —
    /// the default Qwen3 maxSeq is 256K and allocating that much KV
    /// for a 64-token corpus wastes seconds + several GB.
    public static func emit(
        model: any LanguageModel, tokenIds: [Int],
        maxSeq: Int? = nil,
        device: Device = .shared
    ) -> [[Float]] {
        precondition(!tokenIds.isEmpty, "LogitsEmitter.emit: empty token list")
        let cap = maxSeq ?? max(tokenIds.count, 256)
        let caches = model.makeLayerCaches(maxSeq: cap, device: device)
        var trace: [[Float]] = []
        trace.reserveCapacity(tokenIds.count)
        for (i, tok) in tokenIds.enumerated() {
            // Use the no-cmd default extension so each token gets a
            // fresh cmd-buffer that is committed + waited internally —
            // this matches the pattern in existing FFAI bench tests
            // (Qwen35MoEBenchIntegrationTests.decodeBenchT1) and avoids
            // any subtle issue with our caller-supplied cmd buffer
            // interacting with families that allocate their own work
            // buffers inside forward(...).
            let logits = model.forward(
                tokenId: tok, position: i,
                caches: caches, device: device)
            // Use toFloatArray so we handle f32 / f16 / bf16 logits dtypes
            // correctly — `toArray(as: Float.self)` reinterprets raw bytes
            // and segfaults on the half-precision logits Qwen3 emits.
            trace.append(logits.toFloatArray())
        }
        return trace
    }

    /// Compare two logit traces produced by `emit(...)` (same corpus,
    /// different KV cache types) into the aggregated KLD metrics
    /// reported by the canonical `llama-perplexity --kl-divergence`
    /// harness. `tokenIds[t + 1]` is the ground-truth "next token" at
    /// position `t`; the last position is dropped since there's no
    /// next-token reference for it.
    public static func compare(
        baseline: [[Float]], codec: [[Float]], tokenIds: [Int]
    ) -> KLDivergence.AggregateMetrics {
        precondition(
            baseline.count == codec.count
                && baseline.count == tokenIds.count,
            "LogitsEmitter.compare: trace and tokenIds length mismatch "
                + "(baseline=\(baseline.count), codec=\(codec.count), "
                + "tokens=\(tokenIds.count))")
        precondition(
            tokenIds.count >= 2,
            "LogitsEmitter.compare: need at least 2 positions to score "
                + "(position t predicts tokenIds[t + 1])")
        var positions: [KLDivergence.PositionMetrics] = []
        positions.reserveCapacity(tokenIds.count - 1)
        for t in 0 ..< (tokenIds.count - 1) {
            positions.append(
                KLDivergence.positionMetrics(
                    baselineLogits: baseline[t],
                    codecLogits: codec[t],
                    nextTokenId: tokenIds[t + 1]))
        }
        return KLDivergence.aggregate(positions)
    }
}
