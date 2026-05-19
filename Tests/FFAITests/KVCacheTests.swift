import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("KVCache", .serialized)
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
}
