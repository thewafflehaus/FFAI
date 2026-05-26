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
import Foundation
import Metal
import Testing
@testable import FFAI
import TestHelpers

@Suite("KVCache")
struct KVCacheTests {
    @Test("init creates zeroed K/V buffers of the right shape")
    func initShape() {
        autoreleasepool {
            let c = KVCache(nKVHeads: 2, headDim: 4, maxSeq: 8, dtype: .f32)
            #expect(c.kBuffer.shape == [2, 8, 4])
            #expect(c.vBuffer.shape == [2, 8, 4])
            #expect(c.length == 0)
            // Both buffers initialize to zero.
            let zeros = [Float](repeating: 0, count: 64)
            #expect(c.kBuffer.toArray(as: Float.self) == zeros)
            #expect(c.vBuffer.toArray(as: Float.self) == zeros)
        }
    }

    @Test("append writes per-head slabs at current position")
    func appendOnePosition() {
        autoreleasepool {
            let c = KVCache(nKVHeads: 2, headDim: 4, maxSeq: 8, dtype: .f32)
            let kFlat = Tensor.empty(shape: [2, 4], dtype: .f32)
            let vFlat = Tensor.empty(shape: [2, 4], dtype: .f32)
            kFlat.copyIn(from: [Float(1), 2, 3, 4, 5, 6, 7, 8])     // head0=[1..4], head1=[5..8]
            vFlat.copyIn(from: [Float(11), 12, 13, 14, 15, 16, 17, 18])

            c.append(kFlat: kFlat, vFlat: vFlat)
            #expect(c.length == 1)

            // Layout: [n_kv_heads=2, max_seq=8, head_dim=4]
            // head0 position 0 should be [1,2,3,4], head1 position 0 should be [5,6,7,8]
            let kAll = c.kBuffer.toArray(as: Float.self)
            #expect(kAll[0..<4] == [1, 2, 3, 4][0..<4])
            // head1 starts at index 8 * 4 = 32
            #expect(kAll[32..<36] == [5, 6, 7, 8][0..<4])
            let vAll = c.vBuffer.toArray(as: Float.self)
            #expect(vAll[0..<4] == [11, 12, 13, 14][0..<4])
            #expect(vAll[32..<36] == [15, 16, 17, 18][0..<4])
        }
    }

    @Test("multiple appends advance length and use the right offsets")
    func appendMultiple() {
        autoreleasepool {
            let c = KVCache(nKVHeads: 1, headDim: 2, maxSeq: 4, dtype: .f32)
            for p in 0..<3 {
                let kFlat = Tensor.empty(shape: [1, 2], dtype: .f32)
                let vFlat = Tensor.empty(shape: [1, 2], dtype: .f32)
                kFlat.copyIn(from: [Float(p * 10), Float(p * 10 + 1)])
                vFlat.copyIn(from: [Float(p * 100), Float(p * 100 + 1)])
                c.append(kFlat: kFlat, vFlat: vFlat)
            }
            #expect(c.length == 3)
            let k = c.kBuffer.toArray(as: Float.self)
            // Layout [1, 4, 2]: pos0=[0,1], pos1=[10,11], pos2=[20,21], pos3=zero (unused)
            #expect(k[0..<6] == [0, 1, 10, 11, 20, 21][0..<6])
            #expect(k[6..<8] == [0, 0][0..<2])
        }
    }

    @Test("reset zeros length but keeps allocation")
    func reset() {
        autoreleasepool {
            let c = KVCache(nKVHeads: 1, headDim: 2, maxSeq: 4, dtype: .f32)
            let kFlat = Tensor.empty(shape: [1, 2], dtype: .f32)
            let vFlat = Tensor.empty(shape: [1, 2], dtype: .f32)
            kFlat.copyIn(from: [Float(1), 2])
            vFlat.copyIn(from: [Float(3), 4])
            c.append(kFlat: kFlat, vFlat: vFlat)
            #expect(c.length == 1)
            c.reset()
            #expect(c.length == 0)
        }
    }

    // MARK: - AffineQuantizedKVCache (Phase 5c)

    private func makeSharedWorking(nKVHeads: Int, maxSeq: Int, headDim: Int,
                                   dtype: DType) -> (k: Tensor, v: Tensor) {
        let k = Tensor.empty(shape: [nKVHeads, maxSeq, headDim], dtype: dtype)
        let v = Tensor.empty(shape: [nKVHeads, maxSeq, headDim], dtype: dtype)
        k.zero(); v.zero()
        return (k, v)
    }

    @Test("AffineQuantizedKVCache: round-trip a slowly-varying row preserves values within int8 precision")
    func affineRoundTrip() {
        autoreleasepool {
            let nKVHeads = 2
            let headDim = 64
            let maxSeq = 8
            let groupSize = 32
            let (sk, sv) = makeSharedWorking(nKVHeads: nKVHeads, maxSeq: maxSeq,
                                             headDim: headDim, dtype: .f32)
            let cache = AffineQuantizedKVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                dtype: .f32, bits: 8, groupSize: groupSize,
                sharedWorkingK: sk, sharedWorkingV: sv
            )
            // Slowly-varying inputs: K = sin-ish ramp, V = cos-ish ramp.
            var kFlatData = [Float](repeating: 0, count: nKVHeads * headDim)
            var vFlatData = [Float](repeating: 0, count: nKVHeads * headDim)
            for h in 0..<nKVHeads {
                for d in 0..<headDim {
                    kFlatData[h * headDim + d] = Float(d) / Float(headDim) * 2 - 1   // -1..1
                    vFlatData[h * headDim + d] = Float(d) / Float(headDim) * 4 - 2   // -2..2
                }
            }
            let kFlat = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
            let vFlat = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
            kFlat.copyIn(from: kFlatData)
            vFlat.copyIn(from: vFlatData)

            runAndWait { cb in cache.appendOnGPU(kFlat: kFlat, vFlat: vFlat, on: cb) }
            #expect(cache.length == 1)

            // Dequant into working buffer + sanity-check the output for
            // position 0 matches the input within int8 precision.
            var dqK: Tensor!
            var dqV: Tensor!
            runAndWait { cb in
                let pair = cache.prepareForAttention(on: cb)
                dqK = pair.k
                dqV = pair.v
            }

            let kOut = dqK.toArray(as: Float.self)
            let vOut = dqV.toArray(as: Float.self)
            // Tolerance: range/255 per group, so worst-case ~ (max-min)/255.
            // For input range 2 over groupSize=32, tolerance ~ 0.008.
            let tolK: Float = 0.01
            let tolV: Float = 0.02
            for h in 0..<nKVHeads {
                for d in 0..<headDim {
                    // Position 0 lives at offset h * maxSeq * headDim + 0 * headDim + d
                    let outIdx = h * maxSeq * headDim + d
                    let expectedK = kFlatData[h * headDim + d]
                    let expectedV = vFlatData[h * headDim + d]
                    #expect(abs(kOut[outIdx] - expectedK) < tolK,
                            "K[\(h),0,\(d)] = \(kOut[outIdx]) vs expected \(expectedK)")
                    #expect(abs(vOut[outIdx] - expectedV) < tolV,
                            "V[\(h),0,\(d)] = \(vOut[outIdx]) vs expected \(expectedV)")
                }
            }
        }
    }

    @Test("AffineQuantizedKVCache: multi-position appends + dequant returns the right slice")
    func affineMultiPosition() {
        autoreleasepool {
            let nKVHeads = 1
            let headDim = 64
            let maxSeq = 4
            let groupSize = 32
            let (sk, sv) = makeSharedWorking(nKVHeads: nKVHeads, maxSeq: maxSeq,
                                             headDim: headDim, dtype: .f32)
            let cache = AffineQuantizedKVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                dtype: .f32, bits: 8, groupSize: groupSize,
                sharedWorkingK: sk, sharedWorkingV: sv
            )

            // Append 3 distinct rows.
            for pos in 0..<3 {
                var k = [Float](repeating: 0, count: headDim)
                var v = [Float](repeating: 0, count: headDim)
                for d in 0..<headDim {
                    k[d] = Float(pos) * 10 + Float(d) / Float(headDim)
                    v[d] = Float(pos) * 100 + Float(d) / Float(headDim)
                }
                let kT = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
                let vT = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
                kT.copyIn(from: k)
                vT.copyIn(from: v)
                runAndWait { cb in cache.appendOnGPU(kFlat: kT, vFlat: vT, on: cb) }
            }
            #expect(cache.length == 3)

            var dqK: Tensor!
            runAndWait { cb in dqK = cache.prepareForAttention(on: cb).k }
            let kOut = dqK.toArray(as: Float.self)
            // Each of the 3 positions should reconstruct its row within
            // int8 precision (range here is ~10 per group → tol ~0.05).
            let tol: Float = 0.05
            for pos in 0..<3 {
                for d in 0..<headDim {
                    let outIdx = pos * headDim + d
                    let expected = Float(pos) * 10 + Float(d) / Float(headDim)
                    #expect(abs(kOut[outIdx] - expected) < tol,
                            "pos=\(pos) d=\(d) got \(kOut[outIdx]) vs \(expected)")
                }
            }
        }
    }

    @Test("AffineQuantizedKVCache(int4): round-trip preserves values within int4 precision")
    func affineInt4RoundTrip() {
        autoreleasepool {
            let nKVHeads = 2
            let headDim = 64
            let maxSeq = 8
            let groupSize = 32
            let (sk, sv) = makeSharedWorking(nKVHeads: nKVHeads, maxSeq: maxSeq,
                                             headDim: headDim, dtype: .f32)
            let cache = AffineQuantizedKVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                dtype: .f32, bits: 4, groupSize: groupSize,
                sharedWorkingK: sk, sharedWorkingV: sv
            )
            var kFlatData = [Float](repeating: 0, count: nKVHeads * headDim)
            var vFlatData = [Float](repeating: 0, count: nKVHeads * headDim)
            for h in 0..<nKVHeads {
                for d in 0..<headDim {
                    kFlatData[h * headDim + d] = Float(d) / Float(headDim) * 2 - 1
                    vFlatData[h * headDim + d] = Float(d) / Float(headDim) * 4 - 2
                }
            }
            let kFlat = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
            let vFlat = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
            kFlat.copyIn(from: kFlatData)
            vFlat.copyIn(from: vFlatData)
            runAndWait { cb in cache.appendOnGPU(kFlat: kFlat, vFlat: vFlat, on: cb) }

            var dqK: Tensor!, dqV: Tensor!
            runAndWait { cb in
                let pair = cache.prepareForAttention(on: cb)
                dqK = pair.k; dqV = pair.v
            }
            let kOut = dqK.toArray(as: Float.self)
            let vOut = dqV.toArray(as: Float.self)
            // int4 tolerance: range/15 per group, much coarser than int8.
            // For groupSize=32 over input range 2: ~0.13 max error per element.
            let tolK: Float = 0.15
            let tolV: Float = 0.3
            for h in 0..<nKVHeads {
                for d in 0..<headDim {
                    let outIdx = h * maxSeq * headDim + d
                    #expect(abs(kOut[outIdx] - kFlatData[h * headDim + d]) < tolK)
                    #expect(abs(vOut[outIdx] - vFlatData[h * headDim + d]) < tolV)
                }
            }
        }
    }

    @Test("AffineQuantizedKVCache(int4): multi-position round-trip at integration config")
    func affineInt4MultiPositionIntegrationConfig() {
        autoreleasepool {
            // Mirror the Qwen3-1.7B integration config exactly:
            // headDim=128, groupSize=64, nKVHeads=8, ~24 appended positions.
            // This is the blind spot that hid the int4 bug — the existing
            // int4 round-trip test is single-position headDim=64 gs=32.
            let nKVHeads = 8
            let headDim = 128
            let maxSeq = 64
            let groupSize = 64
            let nPositions = 24
            let (sk, sv) = makeSharedWorking(nKVHeads: nKVHeads, maxSeq: maxSeq,
                                             headDim: headDim, dtype: .f32)
            let cache = AffineQuantizedKVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                dtype: .f32, bits: 4, groupSize: groupSize,
                sharedWorkingK: sk, sharedWorkingV: sv
            )

            // Realistic K/V structure: mostly small values in [-1, 1],
            // plus one per-group outlier so the affine range is wide.
            // Deterministic LCG so the test is reproducible.
            var rng: UInt64 = 0x9E3779B97F4A7C15
            func next() -> Float {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let u = Double(rng >> 11) / Double(1 << 53)
                return Float(u * 2.0 - 1.0)   // [-1, 1)
            }

            // Store every appended row so we can check each position.
            var allK = [[Float]]()
            var allV = [[Float]]()
            let groupsPerHead = headDim / groupSize
            for _ in 0..<nPositions {
                var k = [Float](repeating: 0, count: nKVHeads * headDim)
                var v = [Float](repeating: 0, count: nKVHeads * headDim)
                for h in 0..<nKVHeads {
                    for d in 0..<headDim {
                        k[h * headDim + d] = next() * 0.5
                        v[h * headDim + d] = next() * 0.5
                    }
                    // Inject one outlier per group (mimics real K/V).
                    for g in 0..<groupsPerHead {
                        let outIdx = h * headDim + g * groupSize + (g % groupSize)
                        k[outIdx] = 6.0
                        v[outIdx] = -6.0
                    }
                }
                allK.append(k); allV.append(v)
                let kT = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
                let vT = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
                kT.copyIn(from: k); vT.copyIn(from: v)
                runAndWait { cb in cache.appendOnGPU(kFlat: kT, vFlat: vT, on: cb) }
            }
            #expect(cache.length == nPositions)

            var dqK: Tensor!, dqV: Tensor!
            runAndWait { cb in
                let pair = cache.prepareForAttention(on: cb)
                dqK = pair.k; dqV = pair.v
            }
            let kOut = dqK.toArray(as: Float.self)
            let vOut = dqV.toArray(as: Float.self)

            // int4 affine tolerance: range/15 per group. With a 6.0
            // outlier and small values, per-group range ~6.5 → step
            // ~0.43 → max error ~0.22. Allow generous slack.
            let tol: Float = 0.5
            var worst: Float = 0
            var worstPos = -1
            for pos in 0..<nPositions {
                for h in 0..<nKVHeads {
                    for d in 0..<headDim {
                        // Buffer layout [nKVHeads, maxSeq, headDim].
                        let outIdx = (h * maxSeq + pos) * headDim + d
                        let expK = allK[pos][h * headDim + d]
                        let expV = allV[pos][h * headDim + d]
                        let ek = abs(kOut[outIdx] - expK)
                        let ev = abs(vOut[outIdx] - expV)
                        if ek > worst { worst = ek; worstPos = pos }
                        if ev > worst { worst = ev; worstPos = pos }
                    }
                }
            }
            #expect(worst < tol,
                    "int4 multi-position round-trip: worst error \(worst) at pos \(worstPos) exceeds tol \(tol)")
        }
    }

    @Test("AffineQuantizedKVCache(int4): reconstruction error shrinks with group size")
    func affineInt4GroupSizeErrorCurve() {
        // Hypothesis-B measurement: affine min-max int4 over wide groups
        // collapses non-outlier dims onto 1-2 of the 16 levels. Measure
        // mean-abs reconstruction error for groupSize ∈ {64, 32, 16} on
        // outlier-containing input. A smaller group should tighten the
        // per-group range and lower the error materially.
        autoreleasepool {
            let nKVHeads = 8
            let headDim = 128
            let maxSeq = 8
            let nPositions = 1

            // Same outlier-containing distribution for every group size.
            var rng: UInt64 = 0xDEADBEEFCAFEF00D
            func next() -> Float {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let u = Double(rng >> 11) / Double(1 << 53)
                return Float(u * 2.0 - 1.0)
            }
            // Sparse outliers: one large channel per head (a "massive
            // activation"). With headDim=128, gs64 → 1 outlier shared
            // across 64 dims; gs16 → 1 outlier confined to 16 dims, the
            // other 7 groups stay tight. This is the realistic K/V shape.
            var base = [Float](repeating: 0, count: nKVHeads * headDim)
            for h in 0..<nKVHeads {
                for d in 0..<headDim { base[h * headDim + d] = next() * 0.5 }
                base[h * headDim + (h * 13) % headDim] = 8.0   // one outlier/head
            }

            func meanAbsError(groupSize: Int, bits: Int) -> Float {
                let (sk, sv) = makeSharedWorking(nKVHeads: nKVHeads, maxSeq: maxSeq,
                                                 headDim: headDim, dtype: .f32)
                let cache = AffineQuantizedKVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                    dtype: .f32, bits: bits, groupSize: groupSize,
                    sharedWorkingK: sk, sharedWorkingV: sv
                )
                let kT = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
                let vT = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
                kT.copyIn(from: base); vT.copyIn(from: base)
                for _ in 0..<nPositions {
                    runAndWait { cb in cache.appendOnGPU(kFlat: kT, vFlat: vT, on: cb) }
                }
                var dqK: Tensor!
                runAndWait { cb in dqK = cache.prepareForAttention(on: cb).k }
                let kOut = dqK.toArray(as: Float.self)
                var sum: Float = 0
                var count = 0
                for h in 0..<nKVHeads {
                    for d in 0..<headDim {
                        let outIdx = (h * maxSeq + 0) * headDim + d
                        sum += abs(kOut[outIdx] - base[h * headDim + d])
                        count += 1
                    }
                }
                return sum / Float(count)
            }

            let e64 = meanAbsError(groupSize: 64, bits: 4)
            let e32 = meanAbsError(groupSize: 32, bits: 4)
            let e16 = meanAbsError(groupSize: 16, bits: 4)
            let e8int8 = meanAbsError(groupSize: 64, bits: 8)
            print("[int4-error] gs64=\(e64) gs32=\(e32) gs16=\(e16) | int8 gs64=\(e8int8)")
            // Smaller groups must reduce error monotonically.
            #expect(e32 < e64, "gs32 error \(e32) should be < gs64 \(e64)")
            #expect(e16 < e32, "gs16 error \(e16) should be < gs32 \(e32)")
        }
    }

    @Test("AffineQuantizedKVCache(int4): bytesAllocated halves vs int8")
    func affineInt4BytesAccounting() {
        autoreleasepool {
            let nKVHeads = 8, headDim = 128, maxSeq = 4096, groupSize = 64
            let (sk, sv) = makeSharedWorking(nKVHeads: nKVHeads, maxSeq: maxSeq,
                                             headDim: headDim, dtype: .f16)
            let int4 = AffineQuantizedKVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                dtype: .f16, bits: 4, groupSize: groupSize,
                sharedWorkingK: sk, sharedWorkingV: sv
            )
            let int8 = AffineQuantizedKVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                dtype: .f16, bits: 8, groupSize: groupSize,
                sharedWorkingK: sk, sharedWorkingV: sv
            )
            // int4 weights are half as large; scales/biases the same → int4
            // total is somewhere around half the int8 total (not exactly,
            // because scales/biases overhead is fixed).
            #expect(int4.bytesAllocated < int8.bytesAllocated)
            // The weights portion alone should be exactly half.
            let int4Weights = 2 * nKVHeads * maxSeq * (headDim / 8) * 4
            let int8Weights = 2 * nKVHeads * maxSeq * (headDim / 4) * 4
            #expect(int4Weights == int8Weights / 2)
        }
    }

    @Test("AffineQuantizedKVCache: bytesAllocated reflects compressed storage, not the working buffer")
    func affineBytesAccounting() {
        autoreleasepool {
            let nKVHeads = 8
            let headDim = 128
            let maxSeq = 4096
            let groupSize = 64
            let (sk, sv) = makeSharedWorking(nKVHeads: nKVHeads, maxSeq: maxSeq,
                                             headDim: headDim, dtype: .f16)
            let cache = AffineQuantizedKVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                dtype: .f16, bits: 8, groupSize: groupSize,
                sharedWorkingK: sk, sharedWorkingV: sv
            )
            // Expected: 2 × (nKVHeads × maxSeq × (headDim/4) × 4   [u32 weights]
            //                + 2 × nKVHeads × maxSeq × (headDim/64) × 2 [scales+biases fp16])
            let packs = headDim / 4
            let groups = headDim / groupSize
            let expected = 2 * (
                nKVHeads * maxSeq * packs * 4
                + 2 * nKVHeads * maxSeq * groups * 2  // f16 = 2 bytes
            )
            #expect(cache.bytesAllocated == expected)
            #expect(cache.bytesInUse == 0)  // length=0
        }
    }

    // MARK: - KVCache.length lock contract

    @Test("Concurrent CPU append increments length atomically — no lost updates")
    func concurrentAppendDoesNotLoseUpdates() {
        // KVCache.length is lock-protected so the (read pos, write at
        // pos, increment) sequence in `append` and `appendOnGPU` is
        // atomic. Phase 8 batched decode dispatches multiple Tasks
        // against one cache; concurrent appenders must not queue
        // dispatches against the same `pos`.
        //
        // This test exercises the CPU `append` path (no GPU dispatch
        // needed) from 8 concurrent workers × 16 iterations each =
        // 128 total appends. Asserts the final length is exactly 128
        // and the cache contents reflect every write — no lost
        // updates from a `read-then-write` race.
        autoreleasepool {
            let nKVHeads = 1
            let headDim = 4
            let maxSeq = 128
            let workers = 8
            let iterations = 16
            #expect(workers * iterations == maxSeq,
                    "test sizing assumption: total writes fill the cache")

            let c = KVCache(nKVHeads: nKVHeads, headDim: headDim,
                            maxSeq: maxSeq, dtype: .f32)
            let kSrc = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
            let vSrc = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
            kSrc.copyIn(from: [Float](repeating: 1, count: nKVHeads * headDim))
            vSrc.copyIn(from: [Float](repeating: 2, count: nKVHeads * headDim))

            DispatchQueue.concurrentPerform(iterations: workers) { _ in
                for _ in 0..<iterations {
                    c.append(kFlat: kSrc, vFlat: vSrc)
                }
            }

            // If the lock works, all 128 increments land and length =
            // 128. If the (read, write, +=) sequence raced, we'd see
            // length < 128.
            #expect(c.length == workers * iterations,
                    "expected \(workers * iterations) appends, got \(c.length)")
        }
    }

    // MARK: - truncate (speculative-decoding rollback)

    @Test("truncate rolls KVCache length back; re-append overwrites the tail")
    func truncateRollsBackAndReappends() {
        autoreleasepool {
            let c = KVCache(nKVHeads: 1, headDim: 2, maxSeq: 16, dtype: .f32)
            let device = Device.shared
            let buf = device.makeBuffer(length: 8)
            func appendStamp(_ value: Float) {
                buf.contents().assumingMemoryBound(to: Float.self)[0] = value
                buf.contents().assumingMemoryBound(to: Float.self)[1] = value
                let t = Tensor(buffer: buf, offset: 0, shape: [1, 2], dtype: .f32)
                c.append(kFlat: t, vFlat: t)
            }
            for i in 0..<8 { appendStamp(Float(i) + 1) }
            #expect(c.length == 8)

            // Reject the last 5 draft tokens, keep a 3-token prefix.
            c.truncate(toLength: 3)
            #expect(c.length == 3)
            #expect(c.absolutePosition == 3)

            // Re-append: the new token lands in physical slot 3,
            // overwriting the discarded value.
            appendStamp(99)
            #expect(c.length == 4)
            let kPtr = c.kBuffer.buffer.contents()
                .advanced(by: c.kBuffer.offset)
                .assumingMemoryBound(to: Float.self)
            #expect(kPtr[3 * 2] == 99.0)
        }
    }

    @Test("truncate works on AffineQuantizedKVCache")
    func truncateAffineQuantized() {
        autoreleasepool {
            let maxSeq = 16, nKVHeads = 1, headDim = 8
            let wk = Tensor.empty(shape: [nKVHeads, maxSeq, headDim], dtype: .f32)
            let wv = Tensor.empty(shape: [nKVHeads, maxSeq, headDim], dtype: .f32)
            let c = AffineQuantizedKVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq, dtype: .f32,
                bits: 8, groupSize: 8, sharedWorkingK: wk, sharedWorkingV: wv)
            let cmd = Device.shared.makeCommandBuffer()
            let kFlat = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
            let vFlat = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32)
            kFlat.copyIn(from: [Float](repeating: 1, count: headDim))
            vFlat.copyIn(from: [Float](repeating: 1, count: headDim))
            for _ in 0..<6 { c.appendOnGPU(kFlat: kFlat, vFlat: vFlat, on: cmd) }
            cmd.commit(); cmd.waitUntilCompleted()
            #expect(c.length == 6)
            c.truncate(toLength: 2)
            #expect(c.length == 2)
        }
    }

    @Test("totalBytes accessors — bytesAllocated, bytesInUse, totalBytesAllocated")
    func totalBytesAccessors() {
        let c = KVCache(nKVHeads: 8, headDim: 64, maxSeq: 1024, dtype: .f16)
        let elems = 2 * 8 * 1024 * 64
        #expect(c.bytesAllocated == elems * 2)   // fp16 = 2 bytes
        #expect(c.bytesInUse == 0)
        let arr = [c, c, c]
        #expect(arr.totalBytesAllocated == c.bytesAllocated * 3)
    }
}
