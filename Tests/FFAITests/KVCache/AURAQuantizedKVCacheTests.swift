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

    private func append(_ cache: AURAQuantizedKVCache, _ seed: Float) {
        let device = Device.shared
        let (k, v) = token(seed)
        let cmd = device.makeCommandBuffer()
        cache.appendOnGPU(kFlat: k, vFlat: v, on: cmd)
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

    @Test("re-appending after truncate clears stale atomic_or bits (spec-decode rollback)")
    func truncateReappendMatchesClean() {
        let prefix: [Float] = [1.0, 2.0]
        let draft: [Float] = [9.0, 9.5, 9.9]  // rejected speculative block
        let real: [Float] = [3.0, 4.0]  // accepted continuation

        // Reference: prefix + real appended into clean slots.
        let ref = makeCache()
        for s in prefix + real { append(ref, s) }

        // Under test: prefix, then a draft block, reject it (truncate back
        // to the prefix), then re-decode the accepted continuation into
        // the same physical slots the draft occupied.
        let test = makeCache()
        for s in prefix { append(test, s) }
        for s in draft { append(test, s) }
        test.truncate(toLength: prefix.count)
        for s in real { append(test, s) }

        #expect(test.length == ref.length)

        let refK = dequantK(ref)
        let testK = dequantK(test)

        // Compare only the live [0, length) region per head; the codec is
        // deterministic, so identical inputs + identical rotation must
        // produce identical dequant. Without the zero-on-rewrite fix the
        // re-decoded slots carry OR'd draft+real bits and diverge wildly.
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
        #expect(
            maxErr < 1e-3,
            "re-appended slots diverge from clean reference (maxErr=\(maxErr)) — stale atomic_or bits leaked through truncate + re-append")
    }
}
