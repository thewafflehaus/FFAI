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

import Metal
import Testing

@testable import FFAI

/// AURA cache rollback correctness. The AURA encode kernel
/// `atomic_or`-accumulates into the packed buffer, so a slot that is
/// *re-written* (the spec-decode `truncate` + re-append path: draft a
/// block, reject it, re-decode the accepted continuation into the same
/// physical slots) MUST have its stale bits cleared first or they OR
/// through and corrupt the dequant. This pins the high-water-mark
/// zero-on-rewrite fix.
@Suite("AURAQuantizedKVCache rollback")
struct AURAQuantizedKVCacheTests {

    private static let nKVHeads = 2
    private static let headDim = 128  // power of two (SRHT requirement)
    private static let ctx = 16

    /// Build a fresh AURA cache (aura4) via the production factory so the
    /// codebooks + per-layer SRHT rotation match a real load. Same
    /// `layerIndex` → same rotation, so two caches are directly
    /// comparable.
    private func makeCache() -> AURAQuantizedKVCache {
        let device = Device.shared
        let kind = KVCacheKind.auraQuantized(scheme: .default)
        let scratch = makeAttentionScratch(
            kind: kind, nKVHeads: Self.nKVHeads, headDim: Self.headDim,
            contextLength: Self.ctx, dtype: .f16, device: device)!
        return makeAttentionCache(
            kind: kind, scratch: scratch,
            nKVHeads: Self.nKVHeads, headDim: Self.headDim,
            contextLength: Self.ctx, dtype: .f16, eviction: .unbounded,
            layerIndex: 0, device: device) as! AURAQuantizedKVCache
    }

    /// Deterministic, distinct K/V vectors for a token `seed`.
    private func token(_ seed: Float) -> (Tensor, Tensor) {
        let device = Device.shared
        let n = Self.nKVHeads * Self.headDim
        var kd = [Float](repeating: 0, count: n)
        var vd = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            kd[i] = Foundation.sin(seed + Float(i) * 0.01)
            vd[i] = Foundation.cos(seed + Float(i) * 0.013)
        }
        let k = Tensor.empty(shape: [Self.nKVHeads, Self.headDim], dtype: .f16, device: device)
        let v = Tensor.empty(shape: [Self.nKVHeads, Self.headDim], dtype: .f16, device: device)
        k.copyIn(from: kd)
        v.copyIn(from: vd)
        return (k, v)
    }

    /// Append every `seed` on a SINGLE command buffer. Batching keeps the
    /// in-flight command-buffer count low — the parallel unit suite churns
    /// the shared residency set, and a high per-test cmdbuf count widens
    /// the window for a contention-induced flake.
    private func appendAll(_ cache: AURAQuantizedKVCache, _ seeds: [Float]) {
        let device = Device.shared
        let cmd = device.makeCommandBuffer()
        for s in seeds {
            let (k, v) = token(s)
            cache.appendOnGPU(kFlat: k, vFlat: v, on: cmd)
        }
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    private func dequantK(_ cache: AURAQuantizedKVCache) -> [Float] {
        let device = Device.shared
        let cmd = device.makeCommandBuffer()
        let (k, _) = cache.prepareForAttention(on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return k.toArray(as: Float.self)
    }

    /// One round: dequant a clean `prefix+real` reference against a
    /// `prefix → draft → truncate → real` rollback, and return the worst
    /// element-wise divergence over the live region. The codec is
    /// deterministic (same inputs + same per-layer rotation), so a
    /// correct rollback yields ≈0; without the zero-on-rewrite fix the
    /// re-decoded slots carry OR'd draft+real bits and diverge wildly.
    private func rollbackMaxErr(
        prefix: [Float], draft: [Float], real: [Float]
    ) -> Float {
        let ref = makeCache()
        appendAll(ref, prefix + real)

        let test = makeCache()
        appendAll(test, prefix + draft)
        test.truncate(toLength: prefix.count)
        appendAll(test, real)
        precondition(test.length == ref.length)

        let refK = dequantK(ref)
        let testK = dequantK(test)
        let length = test.length
        var maxErr: Float = 0
        for h in 0 ..< Self.nKVHeads {
            for pos in 0 ..< length {
                for d in 0 ..< Self.headDim {
                    let idx = h * Self.ctx * Self.headDim + pos * Self.headDim + d
                    maxErr = Swift.max(maxErr, abs(refK[idx] - testK[idx]))
                }
            }
        }
        return maxErr
    }

    @Test("re-appending after truncate clears stale atomic_or bits (spec-decode rollback)")
    func truncateReappendMatchesClean() {
        let prefix: [Float] = [1.0, 2.0]
        let draft: [Float] = [9.0, 9.5, 9.9]  // rejected speculative block
        let real: [Float] = [3.0, 4.0]  // accepted continuation

        // The fix under test is DETERMINISTIC: if it regresses, the
        // re-decoded slots carry OR'd garbage and `maxErr` is huge on
        // EVERY attempt. The parallel unit suite, however, has a known
        // systemic GPU-driver flakiness under heavy concurrent
        // command-buffer submission (plan.md Phase 9 — the same class
        // that intermittently hits `ssm_step` / `sdpaDecode`) that can
        // corrupt an exact GPU-output comparison once in a while. So
        // retry: pass as soon as any attempt is clean (a real regression
        // fails all attempts; a transient contention flake clears).
        var lastErr = Float.greatestFiniteMagnitude
        for _ in 0 ..< 6 {
            lastErr = rollbackMaxErr(prefix: prefix, draft: draft, real: real)
            if lastErr < 1e-3 { break }
        }
        #expect(
            lastErr < 1e-3,
            "re-appended slots diverge from clean reference (maxErr=\(lastErr)) — stale atomic_or bits leaked through truncate + re-append")
    }
}
