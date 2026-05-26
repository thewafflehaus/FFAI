import Foundation
import Metal
import Testing
@testable import FFAI
import TestHelpers

@Suite("Ops")
struct OpsTests {
    @Test("add f32 — c[i] = a[i] + b[i]")
    func addF32() {
        autoreleasepool {
            let a = Tensor.empty(shape: [5], dtype: .f32)
            let b = Tensor.empty(shape: [5], dtype: .f32)
            a.copyIn(from: [Float(1), 2, 3, 4, 5])
            b.copyIn(from: [Float(10), 20, 30, 40, 50])
            var out: Tensor!
            runAndWait { cb in out = Ops.add(a, b, on: cb) }
            #expect(out.toArray(as: Float.self) == [11, 22, 33, 44, 55])
        }
    }

    @Test("mul f32 — c[i] = a[i] * b[i]")
    func mulF32() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f32)
            let b = Tensor.empty(shape: [4], dtype: .f32)
            a.copyIn(from: [Float(1), 2, 3, 4])
            b.copyIn(from: [Float(5), 6, 7, 8])
            var out: Tensor!
            runAndWait { cb in out = Ops.mul(a, b, on: cb) }
            #expect(out.toArray(as: Float.self) == [5, 12, 21, 32])
        }
    }

    @Test("silu f32 — out[i] = x / (1 + exp(-x))")
    func siluF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(0), 1, -1, 2])
            var out: Tensor!
            runAndWait { cb in out = Ops.silu(x, on: cb) }
            let result = out.toArray(as: Float.self)
            // silu(0) = 0
            #expect(abs(result[0]) < 1e-5)
            // silu(1) ≈ 0.7311
            #expect(abs(result[1] - Float(1.0 / (1.0 + exp(-1.0)))) < 1e-3)
            // silu(-1) ≈ -0.2689
            #expect(abs(result[2] - Float(-1.0 / (1.0 + exp(1.0)))) < 1e-3)
            // silu(2) ≈ 1.7616
            #expect(abs(result[3] - Float(2.0 / (1.0 + exp(-2.0)))) < 1e-3)
        }
    }

    @Test("sigmoid f32 — out[i] = 1 / (1 + exp(-x))")
    func sigmoidF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(0), 1, -1, 2])
            var out: Tensor!
            runAndWait { cb in out = Ops.sigmoid(x, on: cb) }
            let r = out.toArray(as: Float.self)
            // sigmoid(0) = 0.5
            #expect(abs(r[0] - 0.5) < 1e-5)
            // sigmoid(1) ≈ 0.7311
            #expect(abs(r[1] - Float(1.0 / (1.0 + exp(-1.0)))) < 1e-3)
            // sigmoid(-1) ≈ 0.2689
            #expect(abs(r[2] - Float(1.0 / (1.0 + exp(1.0)))) < 1e-3)
            // sigmoid(2) ≈ 0.8808
            #expect(abs(r[3] - Float(1.0 / (1.0 + exp(-2.0)))) < 1e-3)
        }
    }

    @Test("ropePartial f32 — rotates only the first rotaryDim of each head")
    func ropePartialRotatesSubset() {
        autoreleasepool {
            // headDim=4, rotaryDim=2: a single head, rotate dims [0,1),
            // pass through dims [2,4). Position 1, theta_base=10000.
            let qk = Tensor.empty(shape: [4], dtype: .f32)
            qk.copyIn(from: [Float(1), 0, 7, 9])
            runAndWait { cb in
                Ops.ropePartial(qk, position: 1, headDim: 4, rotaryDim: 2,
                                thetaBase: 10000, on: cb)
            }
            let r = qk.toArray(as: Float.self)
            // rotaryDim=2 → one rotate-half pair (0, 1). inv_freq=1,
            // theta=1, cos≈0.5403, sin≈0.8415.
            //   r[0] = x[0]*cos - x[1]*sin = 1*0.5403 - 0*0.8415 = 0.5403
            //   r[1] = x[0]*sin + x[1]*cos = 1*0.8415 + 0*0.5403 = 0.8415
            #expect(abs(r[0] - 0.5403) < 1e-3)
            #expect(abs(r[1] - 0.8415) < 1e-3)
            // Dims [2,4) are outside rotaryDim → untouched pass-through.
            #expect(abs(r[2] - 7) < 1e-5)
            #expect(abs(r[3] - 9) < 1e-5)
        }
    }

    @Test("ropePartial f32 — rotaryDim == headDim matches full rope")
    func ropePartialFullEqualsRope() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f32)
            a.copyIn(from: [Float(1), 0, 0, 1])
            let b = Tensor.empty(shape: [4], dtype: .f32)
            b.copyIn(from: [Float(1), 0, 0, 1])
            var full: Tensor!
            runAndWait { cb in
                full = Ops.rope(a, position: 1, headDim: 4,
                                thetaBase: 10000, on: cb)
            }
            runAndWait { cb in
                Ops.ropePartial(b, position: 1, headDim: 4, rotaryDim: 4,
                                thetaBase: 10000, on: cb)
            }
            let rf = full.toArray(as: Float.self)
            let rp = b.toArray(as: Float.self)
            for i in 0..<4 {
                #expect(abs(rf[i] - rp[i]) < 1e-4, "i=\(i): \(rf[i]) vs \(rp[i])")
            }
        }
    }

    @Test("relu f32 — out[i] = max(x[i], 0)")
    func reluF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [5], dtype: .f32)
            x.copyIn(from: [Float(0), 1, -1, 2, -3.5])
            var out: Tensor!
            runAndWait { cb in out = Ops.relu(x, on: cb) }
            let result = out.toArray(as: Float.self)
            #expect(result == [0, 1, 0, 2, 0])
        }
    }

    @Test("squared-relu via relu + mul — NemotronH MLP activation")
    func squaredReluF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(-2), 0, 1.5, 3])
            var out: Tensor!
            runAndWait { cb in
                let r = Ops.relu(x, on: cb)
                out = Ops.mul(r, r, on: cb)
            }
            let result = out.toArray(as: Float.self)
            // relu(x)^2: negatives clamp to 0, positives square.
            #expect(result == [0, 0, 2.25, 9])
        }
    }

    @Test("gelu f32 — out[i] = 0.5 * x * (1 + tanh(...))")
    func geluF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(0), 1, -1, 2])
            var out: Tensor!
            runAndWait { cb in out = Ops.gelu(x, on: cb) }
            let result = out.toArray(as: Float.self)
            // tanh-approx GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
            func ref(_ v: Float) -> Float {
                let c: Float = Float((2.0 / Double.pi).squareRoot())
                let inner = c * (v + 0.044715 * v * v * v)
                return 0.5 * v * (1 + tanh(inner))
            }
            #expect(abs(result[0] - ref(0)) < 1e-3)
            #expect(abs(result[1] - ref(1)) < 1e-3)
            #expect(abs(result[2] - ref(-1)) < 1e-3)
            #expect(abs(result[3] - ref(2)) < 1e-3)
        }
    }

    @Test("gelu bf16 — extreme inputs stay finite (Gemma 3 regression)")
    func geluBf16ExtremeInputs() {
        // Gemma 3 1B layer-1 gate values span [-10.25, 10.81] in bf16.
        // The tanh argument hits k*(x + 0.044715*x^3) ≈ ±54 there,
        // beyond Metal's native bf16 tanh range and beyond what an
        // (exp(2x)-1)/(exp(2x)+1) evaluation can survive in bf16's
        // 8-bit exponent. Test that gelu(bf16) over the failing range
        // produces only finite values — no NaN, no inf.
        autoreleasepool {
            let xs: [Float] = stride(from: Float(-15.0), through: 15.0, by: 0.25).map { $0 }
            let n = xs.count
            let x = Tensor.empty(shape: [n], dtype: .bf16)
            // Pack fp32 → bf16 via FFAI's existing test helper.
            let xPtr = x.buffer.contents().bindMemory(to: UInt16.self, capacity: n)
            for (i, v) in xs.enumerated() {
                xPtr[i] = floatToBf16BitsForTest(v)
            }
            var out: Tensor!
            runAndWait { cb in out = Ops.gelu(x, on: cb) }
            let outPtr = out.buffer.contents().bindMemory(to: UInt16.self, capacity: n)
            for i in 0..<n {
                let v = bf16BitsToFloatForTest(outPtr[i])
                #expect(v.isFinite,
                        "gelu(\(xs[i])) in bf16 → \(v); must be finite")
            }
        }
    }

    @Test("gather f32 — picks the right rows")
    func gatherF32() {
        autoreleasepool {
            // table[3, 2] = [[10,11], [20,21], [30,31]]
            let table = Tensor.empty(shape: [3, 2], dtype: .f32)
            table.copyIn(from: [Float(10), 11, 20, 21, 30, 31])
            let ids = Tensor.empty(shape: [2], dtype: .u32)
            ids.copyIn(from: [UInt32(2), 0])
            var out: Tensor!
            runAndWait { cb in out = Ops.gather(table: table, tokenIds: ids, on: cb) }
            #expect(out.shape == [2, 2])
            #expect(out.toArray(as: Float.self) == [30, 31, 10, 11])
        }
    }

    @Test("gemv f32 — out[i] = sum_j W[i,j] * x[j]")
    func gemvF32() {
        autoreleasepool {
            // W [3, 2] = [[1,2], [3,4], [5,6]]
            let w = Tensor.empty(shape: [3, 2], dtype: .f32)
            w.copyIn(from: [Float(1), 2, 3, 4, 5, 6])
            let x = Tensor.empty(shape: [2], dtype: .f32)
            x.copyIn(from: [Float(7), 8])
            // expected: [1*7+2*8, 3*7+4*8, 5*7+6*8] = [23, 53, 83]
            var out: Tensor!
            runAndWait { cb in out = Ops.gemv(weight: w, input: x, on: cb) }
            #expect(out.toArray(as: Float.self) == [23, 53, 83])
        }
    }

    @Test("auraRotatePerHead f32 — permutation matrix applied per head")
    func auraRotatePerHeadPermutation() {
        autoreleasepool {
            // headDim = 4, nHeads = 3. Rotation is the cyclic permutation
            // P[i, j] = 1 iff j == (i + 1) % 4, else 0; row-major.
            // (P · v)[i] = v[(i+1) % 4]  — shifts each head's slice left
            // by one, with wraparound. Use distinct values per head so a
            // bug that crosses head boundaries surfaces immediately.
            let headDim = 4
            let nHeads = 3
            var rot = [Float](repeating: 0, count: headDim * headDim)
            for i in 0..<headDim {
                rot[i * headDim + ((i + 1) % headDim)] = 1
            }
            let rotation = Tensor.empty(shape: [headDim, headDim], dtype: .f32)
            rotation.copyIn(from: rot)

            // Per-head input: [10,11,12,13, 20,21,22,23, 30,31,32,33].
            var inVals = [Float]()
            for h in 0..<nHeads {
                for i in 0..<headDim {
                    inVals.append(Float((h + 1) * 10 + i))
                }
            }
            let x = Tensor.empty(shape: [nHeads * headDim], dtype: .f32)
            x.copyIn(from: inVals)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.auraRotatePerHead(x, rotation: rotation,
                                            nHeads: nHeads, headDim: headDim, on: cb)
            }
            let got = out.toArray(as: Float.self)

            // Expected: each head's slice cyclically shifted left by one.
            //   head 0: [10,11,12,13] → [11,12,13,10]
            //   head 1: [20,21,22,23] → [21,22,23,20]
            //   head 2: [30,31,32,33] → [31,32,33,30]
            var expected = [Float]()
            for h in 0..<nHeads {
                for i in 0..<headDim {
                    expected.append(Float((h + 1) * 10 + ((i + 1) % headDim)))
                }
            }
            #expect(got == expected)
        }
    }

    @Test("auraRotatePerHead f32 — identity rotation is a no-op")
    func auraRotatePerHeadIdentity() {
        autoreleasepool {
            // Identity rotation must round-trip the input exactly; this
            // pins the "no rotation" baseline path the AURA cache used
            // pre-SRHT and matches what the AURARotation.identityMatrix
            // helper emits.
            let headDim = 8
            let nHeads = 2
            var rot = [Float](repeating: 0, count: headDim * headDim)
            for i in 0..<headDim { rot[i * headDim + i] = 1 }
            let rotation = Tensor.empty(shape: [headDim, headDim], dtype: .f32)
            rotation.copyIn(from: rot)

            let inVals: [Float] = (0..<(nHeads * headDim)).map { Float($0) * 0.5 - 1.25 }
            let x = Tensor.empty(shape: [nHeads * headDim], dtype: .f32)
            x.copyIn(from: inVals)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.auraRotatePerHead(x, rotation: rotation,
                                            nHeads: nHeads, headDim: headDim, on: cb)
            }
            let got = out.toArray(as: Float.self)
            for i in 0..<inVals.count {
                #expect(abs(got[i] - inVals[i]) < 1e-5,
                        "i=\(i) got \(got[i]) expected \(inVals[i])")
            }
        }
    }

    @Test("rmsNorm f32 — y = x / rms(x) * weight")
    func rmsNormF32() {
        autoreleasepool {
            // Use the smallest valid size for this kernel: n must be a
            // multiple of 128 (32-lane simdgroup × 4 elements/thread).
            // See Ops.rmsNorm preconditions for the full constraint set.
            let n = 128
            let xs: [Float] = (0..<n).map { Float($0 + 1) }   // [1, 2, …, 128]
            let ws: [Float] = Array(repeating: Float(1), count: n)

            let x = Tensor.empty(shape: [n], dtype: .f32)
            x.copyIn(from: xs)
            let weight = Tensor.empty(shape: [n], dtype: .f32)
            weight.copyIn(from: ws)

            var out: Tensor!
            runAndWait { cb in out = Ops.rmsNorm(x, weight: weight, eps: 1e-6, on: cb) }
            let result = out.toArray(as: Float.self)

            // CPU reference: rms = sqrt(mean(x^2)); y = x / rms * weight.
            let ssq = xs.reduce(Float(0)) { $0 + $1 * $1 }
            let expectedRms = (ssq / Float(n)).squareRoot()
            for i in 0..<n {
                let expected = xs[i] / expectedRms
                #expect(abs(result[i] - expected) < 1e-2,
                        "i=\(i) got \(result[i]) expected \(expected)")
            }
        }
    }

    @Test("rmsNorm f32 — wide row (n=5376) routes to mt_rms_norm_wide")
    func rmsNormWideF32() {
        autoreleasepool {
            // n = 5376 (Gemma 4 31B hidden) is past the 4096 cap of the
            // 4-elements-per-thread kernel, so Ops.rmsNorm routes to the
            // strided mt_rms_norm_wide kernel. eps eps=1e-6 like Gemma 4.
            let n = 5376
            let eps: Float = 1e-6
            let xs: [Float] = (0..<n).map { Float(($0 % 37) - 18) * 0.21 }
            let ws: [Float] = (0..<n).map { 1.0 + Float($0 % 11) * 0.03 }

            let x = Tensor.empty(shape: [n], dtype: .f32)
            x.copyIn(from: xs)
            let weight = Tensor.empty(shape: [n], dtype: .f32)
            weight.copyIn(from: ws)

            var out: Tensor!
            runAndWait { cb in out = Ops.rmsNorm(x, weight: weight, eps: eps, on: cb) }
            let result = out.toArray(as: Float.self)

            // CPU reference: rms = sqrt(mean(x^2) + eps); y = x/rms*weight.
            let ssq = xs.reduce(Float(0)) { $0 + $1 * $1 }
            let expectedRms = (ssq / Float(n) + eps).squareRoot()
            for i in 0..<n {
                let expected = xs[i] / expectedRms * ws[i]
                #expect(abs(result[i] - expected) < 1e-3,
                        "i=\(i) got \(result[i]) expected \(expected)")
            }
        }
    }

    @Test("rope f32 at position 0 is identity (cos=1, sin=0)")
    func ropePos0Identity() {
        autoreleasepool {
            let qk = Tensor.empty(shape: [1, 4], dtype: .f32)
            qk.copyIn(from: [Float(1), 2, 3, 4])
            var out: Tensor!
            runAndWait { cb in
                out = Ops.rope(qk, position: 0, headDim: 4, thetaBase: 10000, on: cb)
            }
            let r = out.toArray(as: Float.self)
            // theta = 0 → cos = 1, sin = 0 → identity
            for i in 0..<4 {
                #expect(abs(r[i] - Float(i + 1)) < 1e-4, "i=\(i) got \(r[i])")
            }
        }
    }

    @Test("rope f32 at position 1, head_dim=4, theta_base=10000")
    func ropePos1() {
        autoreleasepool {
            let qk = Tensor.empty(shape: [1, 4], dtype: .f32)
            qk.copyIn(from: [Float(1), 0, 0, 1])
            var out: Tensor!
            runAndWait { cb in
                out = Ops.rope(qk, position: 1, headDim: 4, thetaBase: 10000, on: cb)
            }
            // i_pair=0: inv_freq = 1, theta = 1, cos≈0.5403, sin≈0.8415
            //   x[0] * cos - x[2] * sin = 1*0.5403 - 0*0.8415 = 0.5403
            //   x[0] * sin + x[2] * cos = 1*0.8415 + 0*0.5403 = 0.8415
            // i_pair=1: inv_freq = 1/sqrt(10000) = 0.01, theta = 0.01, cos≈1, sin≈0.01
            //   x[1]*cos - x[3]*sin = 0*1 - 1*0.01 = -0.01
            //   x[1]*sin + x[3]*cos = 0*0.01 + 1*1 = 1
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0] - 0.5403) < 1e-3)   // index 0
            #expect(abs(r[1] - (-0.01)) < 5e-3)  // index 1 (looser tol for f32 trig)
            #expect(abs(r[2] - 0.8415) < 1e-3)   // index 0 + half_dim
            #expect(abs(r[3] - 1.0) < 1e-3)      // index 1 + half_dim
        }
    }

    @Test("sdpaDecode f32 — single position attends to itself")
    func sdpaSinglePosition() {
        autoreleasepool {
            // Kernel invariant: head_dim must be 128 (32 simdgroups × 32 lanes ×
            // 4 elements/lane). Below 128 the wrapper preconditions catch it;
            // before the preconditions existed, the test ran with head_dim=4
            // and pinned the GPU.
            let D = 128
            let kvStride = 4   // pre-allocated capacity
            let nKV = 1        // only the first position is filled
            let nQHeads = 1
            let nKVHeads = 1

            let q = Tensor.empty(shape: [nQHeads, D], dtype: .f32)
            let k = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)
            let v = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)

            // Q is the unit vector e_0; K[0] is the same so dot(Q, K[0]) = 1.
            var qData = [Float](repeating: 0, count: nQHeads * D)
            qData[0] = 1
            q.copyIn(from: qData)

            var kData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            kData[0] = 1                              // K[head=0, pos=0, d=0]
            k.copyIn(from: kData)

            // V[0] is an arbitrary recognizable vector; positions [1..3]
            // are zero so even if the kernel read past `n_kv` we'd notice.
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            for d in 0..<D { vData[d] = Float(d + 1) }   // [1, 2, …, 128]
            v.copyIn(from: vData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.sdpaDecode(q: q, k: k, v: v,
                                     nQHeads: nQHeads, nKVHeads: nKVHeads,
                                     headDim: D,
                                     nKV: nKV, kvStride: kvStride,
                                     scale: 1.0, on: cb)
            }
            // n_kv = 1 → softmax([single_score]) = 1 → output == V[0].
            let r = out.toArray(as: Float.self)
            for d in 0..<D {
                #expect(abs(r[d] - Float(d + 1)) < 1e-4,
                        "out[\(d)] = \(r[d]), expected \(d + 1)")
            }
        }
    }

    @Test("sdpaDecode f32 — sinkEnd + windowStart skip the masked range")
    func sdpaSlidingWindowAndSinks() {
        autoreleasepool {
            // head_dim=128 is the only variant carrying the sink/window
            // constexprs. Four KV positions; the kernel attends
            // [0, sinkEnd) ∪ [windowStart, nKV) and skips the gap.
            let D = 128
            let kvStride = 4
            let nKV = 4
            let nQHeads = 1
            let nKVHeads = 1

            let q = Tensor.empty(shape: [nQHeads, D], dtype: .f32)
            let k = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)
            let v = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)

            // Q = e_0. Every K row is also e_0, so all four positions
            // share the same score → softmax weights them equally over
            // whatever subset the kernel actually attends.
            var qData = [Float](repeating: 0, count: nQHeads * D)
            qData[0] = 1
            q.copyIn(from: qData)

            var kData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            for pos in 0..<kvStride { kData[pos * D + 0] = 1 }
            k.copyIn(from: kData)

            // V[pos] = constant vector of value `pos`. The attention
            // output's first element is the mean of the attended `pos`
            // values, which makes the attended set directly checkable.
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            for pos in 0..<kvStride {
                for d in 0..<D { vData[pos * D + d] = Float(pos) }
            }
            v.copyIn(from: vData)

            // windowStart=2, sinkEnd=0 → attends positions {2, 3} only.
            // mean(2, 3) = 2.5.
            var windowed: Tensor!
            runAndWait { cb in
                windowed = Ops.sdpaDecode(q: q, k: k, v: v,
                                          nQHeads: nQHeads, nKVHeads: nKVHeads,
                                          headDim: D, nKV: nKV, kvStride: kvStride,
                                          scale: 1.0, on: cb,
                                          sinkEnd: 0, windowStart: 2)
            }
            let wr = windowed.toArray(as: Float.self)
            #expect(abs(wr[0] - 2.5) < 1e-4,
                    "windowStart=2 should attend {2,3}: got \(wr[0]), expected 2.5")

            // sinkEnd=1, windowStart=3 → attends {0} ∪ {3}.
            // mean(0, 3) = 1.5.
            var sinked: Tensor!
            runAndWait { cb in
                sinked = Ops.sdpaDecode(q: q, k: k, v: v,
                                        nQHeads: nQHeads, nKVHeads: nKVHeads,
                                        headDim: D, nKV: nKV, kvStride: kvStride,
                                        scale: 1.0, on: cb,
                                        sinkEnd: 1, windowStart: 3)
            }
            let sr = sinked.toArray(as: Float.self)
            #expect(abs(sr[0] - 1.5) < 1e-4,
                    "sinkEnd=1, windowStart=3 should attend {0,3}: got \(sr[0]), expected 1.5")

            // sinkEnd=0, windowStart=0 → dense full attention {0,1,2,3}.
            // mean(0,1,2,3) = 1.5 — same mean as above by coincidence,
            // so cross-check that an explicit dense call equals the
            // default-argument call.
            var dense: Tensor!
            runAndWait { cb in
                dense = Ops.sdpaDecode(q: q, k: k, v: v,
                                       nQHeads: nQHeads, nKVHeads: nKVHeads,
                                       headDim: D, nKV: nKV, kvStride: kvStride,
                                       scale: 1.0, on: cb,
                                       sinkEnd: 0, windowStart: 0)
            }
            let dr = dense.toArray(as: Float.self)
            #expect(abs(dr[0] - 1.5) < 1e-4,
                    "dense attention over {0,1,2,3}: got \(dr[0]), expected 1.5")
        }
    }

    // MARK: - softmax_categorical_sample

    /// Helper: run the GPU sample once with a given uniform draw, return token id.
    private func runGPUSample(logits: Tensor, temperature: Float, uniform: Float) -> Int {
        let tBuf = Tensor.empty(shape: [1], dtype: .f32)
        tBuf.copyIn(from: [temperature])
        let uBuf = Tensor.empty(shape: [1], dtype: .f32)
        uBuf.copyIn(from: [uniform])
        let out = Tensor.empty(shape: [1], dtype: .u32)
        runAndWait { cb in
            Ops.softmaxCategoricalSample(logits, into: out, temperature: tBuf,
                                         uniform: uBuf, on: cb)
        }
        return Int(out.toArray(as: UInt32.self)[0])
    }

    @Test("GPU sample: peaked distribution always picks the peak", .disabled("bisect"))
    func gpuSamplePeaked() {
        autoreleasepool {
            let l = Tensor.empty(shape: [5], dtype: .f32)
            l.copyIn(from: [Float(0), 0, 10, 0, 0])
            for u in [Float(0.001), 0.5, 0.999] {
                #expect(runGPUSample(logits: l, temperature: 1.0, uniform: u) == 2)
            }
        }
    }

    @Test("GPU sample: uniform logits + uniform=0.0 picks index 0", .disabled("bisect"))
    func gpuSampleUniformLow() {
        autoreleasepool {
            let l = Tensor.empty(shape: [4], dtype: .f32)
            l.copyIn(from: [Float(1), 1, 1, 1])
            #expect(runGPUSample(logits: l, temperature: 1.0, uniform: 0.0) == 0)
        }
    }

    @Test("GPU sample: uniform logits + uniform near 1 picks last index", .disabled("bisect"))
    func gpuSampleUniformHigh() {
        autoreleasepool {
            let l = Tensor.empty(shape: [4], dtype: .f32)
            l.copyIn(from: [Float(1), 1, 1, 1])
            #expect(runGPUSample(logits: l, temperature: 1.0, uniform: 0.99) == 3)
        }
    }

    @Test("GPU sample: matches CPU CDF walk over a 32-vocab sweep", .disabled("bisect"))
    func gpuSampleMatchesCPU() {
        autoreleasepool {
            let n = 32
            var vals = [Float](repeating: 0, count: n)
            for i in 0..<n { vals[i] = Float(i) / 5.0 }
            let l = Tensor.empty(shape: [n], dtype: .f32)
            l.copyIn(from: vals)

            let T: Float = 1.5
            let invT = 1.0 / T
            let scaled = vals.map { $0 * invT }
            let maxL = scaled.max() ?? 0
            let expSums = scaled.map { Foundation.exp(Double($0 - maxL)) }
            let total = expSums.reduce(0, +)
            func cpuExpected(uniform: Float) -> Int {
                let target = Double(uniform) * total
                var cum = 0.0
                for i in 0..<n {
                    cum += expSums[i]
                    if cum >= target { return i }
                }
                return n - 1
            }
            for u in stride(from: Float(0.05), to: 1.0, by: 0.1) {
                let gpu = runGPUSample(logits: l, temperature: T, uniform: u)
                let cpu = cpuExpected(uniform: u)
                #expect(gpu == cpu, "uniform=\(u): gpu=\(gpu) cpu=\(cpu)")
            }
        }
    }

    @Test("GPU sample: large vocab + sparse peak (model-shaped)", .disabled("bisect"))
    func gpuSampleLargeVocab() {
        autoreleasepool {
            // Real models have vocab ~128-152K with a sparse peak.
            // Reproduce that shape to flush out any precision/overflow bugs.
            let n = 152_000
            var vals = [Float](repeating: -10.0, count: n)
            // Three tokens have most of the mass.
            vals[42] = 8.0      // dominant
            vals[1000] = 6.0    // secondary
            vals[50_000] = 4.0  // tertiary
            let l = Tensor.empty(shape: [n], dtype: .f32)
            l.copyIn(from: vals)

            // CPU reference (same algorithm as the kernel).
            let T: Float = 1.0
            let invT = 1.0 / T
            let scaled = vals.map { $0 * invT }
            let maxL = scaled.max() ?? 0
            let expSums = scaled.map { Foundation.exp(Double($0 - maxL)) }
            let total = expSums.reduce(0, +)

            // Sweep uniform draws — each must match the CPU CDF walk.
            for u in stride(from: Float(0.001), to: 1.0, by: 0.07) {
                let gpu = runGPUSample(logits: l, temperature: T, uniform: u)

                let target = Double(u) * total
                var cum = 0.0
                var expected = n - 1
                for i in 0..<n {
                    cum += expSums[i]
                    if cum >= target { expected = i; break }
                }
                #expect(gpu == expected, "uniform=\(u): gpu=\(gpu) cpu=\(expected)")
            }
        }
    }

    @Test("GPU sample: works on bf16 + f16 logits", .disabled("bisect"))
    func gpuSampleDtypes() {
        autoreleasepool {
            let f16 = Tensor.empty(shape: [4], dtype: .f16)
            f16.copyIn(from: [Float16(0), 0, 8, 0])   // peak at index 2
            #expect(runGPUSample(logits: f16, temperature: 1.0, uniform: 0.5) == 2)

            let bf16 = Tensor.empty(shape: [4], dtype: .bf16)
            // bf16 representations of 0, 0, 8, 0:
            bf16.copyIn(from: [UInt16(0), 0, 0x4100, 0])   // 0x4100 = bf16(8.0)
            #expect(runGPUSample(logits: bf16, temperature: 1.0, uniform: 0.5) == 2)
        }
    }

    // MARK: - argmax (GPU reduction kernel)

    /// Direct test of `Ops.argmax`. The GPU reduction kernel must return
    /// the index of the largest logit. Uses a vocab-realistic length so
    /// the 256-thread reduction spans many elements per thread.
    @Test("argmax f32 — GPU reduction returns index of the largest logit")
    func argmaxF32() {
        autoreleasepool {
            // Production-realistic logits length (covers the per-thread
            // strided scan in the 256-thread reduction).
            let n = 4096
            var logits = [Float](repeating: 0, count: n)
            for i in 0..<n { logits[i] = Float((i * 31) % 997) * 0.001 }
            let peak = 2718
            logits[peak] = 99.0   // unambiguous maximum
            let cpuArgmax = logits.indices.max(by: { logits[$0] < logits[$1] })!

            let logitsT = Tensor.empty(shape: [n], dtype: .f32)
            logitsT.copyIn(from: logits)
            let out = Tensor.empty(shape: [1], dtype: .u32)
            runAndWait { cb in Ops.argmax(logitsT, into: out, on: cb) }
            #expect(Int(out.toArray(as: UInt32.self)[0]) == cpuArgmax)
            #expect(Int(out.toArray(as: UInt32.self)[0]) == peak)
        }
    }

    /// argmax over f16/bf16 logits — exercises the half-precision kernel
    /// variants. The peak is large enough to survive bf16's 8-bit mantissa.
    @Test("argmax f16/bf16 — half-precision reduction returns the peak index")
    func argmaxHalfPrecision() {
        autoreleasepool {
            let n = 512

            let f16 = Tensor.empty(shape: [n], dtype: .f16)
            // Ramp of small values, one clear peak at index 300.
            var f16Data = (0..<n).map { Float16(Float($0 % 7) * 0.5) }
            f16Data[300] = 64.0
            f16.copyIn(from: f16Data)
            let outF16 = Tensor.empty(shape: [1], dtype: .u32)
            runAndWait { cb in Ops.argmax(f16, into: outF16, on: cb) }
            #expect(Int(outF16.toArray(as: UInt32.self)[0]) == 300)

            let bf16 = Tensor.empty(shape: [n], dtype: .bf16)
            // bf16 bits: ramp of small values, one peak (0x4280 = bf16(64.0)).
            var bf16Bits = [UInt16](repeating: 0x3F00, count: n)   // bf16(0.5)
            bf16Bits[123] = 0x4280                                  // bf16(64.0)
            bf16.copyIn(from: bf16Bits)
            let outBf16 = Tensor.empty(shape: [1], dtype: .u32)
            runAndWait { cb in Ops.argmax(bf16, into: outBf16, on: cb) }
            #expect(Int(outBf16.toArray(as: UInt32.self)[0]) == 123)
        }
    }

    // MARK: - softplus

    /// Direct test of `Ops.softplus`: out[i] = log(1 + exp(x[i])).
    @Test("softplus f32 — out[i] = log(1 + exp(x[i]))")
    func softplusF32() {
        autoreleasepool {
            let xs: [Float] = [0, 1, -1, 2, -5, 8, -20]
            let x = Tensor.empty(shape: [xs.count], dtype: .f32)
            x.copyIn(from: xs)
            var out: Tensor!
            runAndWait { cb in out = Ops.softplus(x, on: cb) }
            let r = out.toArray(as: Float.self)
            for i in 0..<xs.count {
                // Numerically-stable CPU reference: softplus(x) =
                // max(x,0) + log1p(exp(-|x|)).
                let v = xs[i]
                let expected = max(v, 0) + Float(log1p(Double(exp(-abs(v)))))
                #expect(abs(r[i] - expected) < 1e-3,
                        "i=\(i) x=\(v) got \(r[i]) expected \(expected)")
            }
        }
    }

    // MARK: - rmsNormRows (multi-row RMSNorm reduction kernel)

    /// Direct test of `Ops.rmsNormRows`. Each of `nRows` rows is
    /// independently normalized: y = x / rms(x) * weight. The kernel is
    /// reduction-mode — rowSize must satisfy the rms_norm dispatch
    /// invariant (multiple of 128; one threadgroup per row).
    @Test("rmsNormRows f32 — each row independently normalized")
    func rmsNormRowsF32() {
        autoreleasepool {
            // rowSize=128 is the smallest legal size (32-lane simdgroup ×
            // 4 elements/thread → TPG = rowSize/4 = 32). See Ops.rmsNorm /
            // mlx/rms_norm.rs DISPATCH INVARIANTS.
            let nRows = 3
            let rowSize = 128
            let eps: Float = 1e-6

            // Distinct data per row so a row-offset bug would surface.
            var xData = [Float](repeating: 0, count: nRows * rowSize)
            for row in 0..<nRows {
                for d in 0..<rowSize {
                    xData[row * rowSize + d] = Float(d + 1) * Float(row + 1) * 0.1
                }
            }
            // Non-uniform weight so the weight multiply is exercised.
            let wData: [Float] = (0..<rowSize).map { 1.0 + Float($0 % 5) * 0.1 }

            let x = Tensor.empty(shape: [nRows, rowSize], dtype: .f32)
            x.copyIn(from: xData)
            let w = Tensor.empty(shape: [rowSize], dtype: .f32)
            w.copyIn(from: wData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.rmsNormRows(x, weight: w, eps: eps,
                                      nRows: nRows, rowSize: rowSize, on: cb)
            }
            let r = out.toArray(as: Float.self)

            for row in 0..<nRows {
                var ssq: Float = 0
                for d in 0..<rowSize {
                    let v = xData[row * rowSize + d]
                    ssq += v * v
                }
                let rms = (ssq / Float(rowSize) + eps).squareRoot()
                for d in 0..<rowSize {
                    let expected = xData[row * rowSize + d] / rms * wData[d]
                    let got = r[row * rowSize + d]
                    #expect(abs(got - expected) < 1e-2,
                            "row=\(row) d=\(d) got \(got) expected \(expected)")
                }
            }
        }
    }

    @Test("addAndRmsNorm — residual = a+b, normed = rmsNorm(a+b, weight)")
    func addAndRmsNormCorrectness() {
        autoreleasepool {
            // Two-row case so we exercise both the nRows=1 and
            // nRows>1 dispatch paths in one test. n=128 is the
            // smallest legal width (TPG = n/4 = 32 = simdgroup).
            let nRows = 2, n = 128
            let eps: Float = 1e-6
            let aData: [Float] = (0..<nRows * n).map { Float(($0 % 19) - 9) * 0.31 }
            let bData: [Float] = (0..<nRows * n).map { Float(($0 % 13) - 6) * 0.17 }
            let wData: [Float] = (0..<n).map { 1.0 + Float($0 % 7) * 0.05 }

            let a = Tensor.empty(shape: [nRows, n], dtype: .f32)
            a.copyIn(from: aData)
            let b = Tensor.empty(shape: [nRows, n], dtype: .f32)
            b.copyIn(from: bData)
            let weight = Tensor.empty(shape: [n], dtype: .f32)
            weight.copyIn(from: wData)

            let cmd = Device.shared.makeCommandBuffer()
            let (residual, normed) = Ops.addAndRmsNorm(
                a, b, weight: weight, eps: eps,
                nRows: nRows, rowSize: n, on: cmd)
            cmd.commit(); cmd.waitUntilCompleted()

            let residArr = residual.toArray(as: Float.self)
            let normedArr = normed.toArray(as: Float.self)

            // CPU reference: residual is a+b; normed is RMSNorm(a+b)·w.
            for row in 0..<nRows {
                var ssq: Float = 0
                var sums = [Float](repeating: 0, count: n)
                for d in 0..<n {
                    let s = aData[row * n + d] + bData[row * n + d]
                    sums[d] = s
                    ssq += s * s
                }
                let rms = (ssq / Float(n) + eps).squareRoot()
                for d in 0..<n {
                    let expectedResid = sums[d]
                    let expectedNormed = sums[d] / rms * wData[d]
                    let gotResid = residArr[row * n + d]
                    let gotNormed = normedArr[row * n + d]
                    #expect(abs(gotResid - expectedResid) < 1e-3,
                            "residual row=\(row) d=\(d): got \(gotResid) expected \(expectedResid)")
                    #expect(abs(gotNormed - expectedNormed) < 1e-2,
                            "normed row=\(row) d=\(d): got \(gotNormed) expected \(expectedNormed)")
                }
            }
        }
    }

    @Test("sdpaMulti — uniform K gives uniform attention (output = mean V)")
    func sdpaMultiUniformKMeansV() {
        autoreleasepool {
            // With every K row identical, all scores tie → softmax is
            // uniform → each query's output is the plain mean of the
            // attended V rows. A reference that needs no SDPA oracle.
            let headDim = 128, nQHeads = 2, nKVHeads = 1
            let baseKV = 0, nQuery = 4
            let kvStride = baseKV + nQuery
            let scale = 1.0 / Float(Double(headDim).squareRoot())

            let q = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32)
            q.copyIn(from: (0..<nQuery * nQHeads * headDim).map { Float($0 % 7) * 0.1 })

            // K: every row the same constant vector.
            let k = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            k.copyIn(from: [Float](repeating: 0.5, count: nKVHeads * kvStride * headDim))

            // V: row t holds the constant value `t` so the mean is easy.
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * headDim)
            for t in 0..<kvStride {
                for d in 0..<headDim { vData[t * headDim + d] = Float(t) }
            }
            let v = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            v.copyIn(from: vData)

            let cmd = Device.shared.makeCommandBuffer()
            // Full (non-causal) mode → every query attends all 4 V rows,
            // so each output element is mean(0,1,2,3) = 1.5.
            let out = Ops.sdpaMulti(q: q, k: k, v: v,
                                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                                    baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                                    causal: false, scale: scale, on: cmd)
            cmd.commit(); cmd.waitUntilCompleted()

            let result = out.toArray(as: Float.self)
            #expect(result.count == nQuery * nQHeads * headDim)
            for value in result {
                #expect(abs(value - 1.5) < 1e-3, "expected mean V = 1.5, got \(value)")
            }
        }
    }

    @Test("sdpaMulti — causal mode: query r attends V rows 0...r")
    func sdpaMultiCausalPrefixMeans() {
        autoreleasepool {
            let headDim = 128, nQHeads = 1, nKVHeads = 1
            let baseKV = 0, nQuery = 4
            let kvStride = baseKV + nQuery
            let scale = 1.0 / Float(Double(headDim).squareRoot())

            let q = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32)
            q.copyIn(from: [Float](repeating: 0.3, count: nQuery * nQHeads * headDim))
            let k = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            k.copyIn(from: [Float](repeating: 0.5, count: nKVHeads * kvStride * headDim))
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * headDim)
            for t in 0..<kvStride {
                for d in 0..<headDim { vData[t * headDim + d] = Float(t) }
            }
            let v = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            v.copyIn(from: vData)

            let cmd = Device.shared.makeCommandBuffer()
            let out = Ops.sdpaMulti(q: q, k: k, v: v,
                                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                                    baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                                    causal: true, scale: scale, on: cmd)
            cmd.commit(); cmd.waitUntilCompleted()

            // Causal: query r attends rows 0...r → output = mean(0...r).
            let result = out.toArray(as: Float.self)
            for r in 0..<nQuery {
                let expected = Float(r) / 2.0   // mean(0,1,...,r)
                for d in 0..<headDim {
                    let got = result[r * headDim + d]
                    #expect(abs(got - expected) < 1e-3,
                            "query \(r) d=\(d): expected \(expected), got \(got)")
                }
            }
        }
    }

    @Test("sdpaBidirectional — d32/d64/d72 uniform K give mean V (no-causal contract)")
    func sdpaBidirectionalUniformKMeansV() {
        autoreleasepool {
            // Vision-tower contract: every query attends every key.
            // With identical K rows, softmax is uniform → each output
            // equals the plain mean of V across the attended block.
            // Cover every supported head_dim so the routing in
            // Ops.sdpaBidirectional is exercised end-to-end.
            for headDim in [32, 64, 72] {
                let nQHeads = 2, nKVHeads = 1
                let baseKV = 0, nQuery = 4
                let kvStride = baseKV + nQuery
                let scale = 1.0 / Float(Double(headDim).squareRoot())

                let q = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32)
                q.copyIn(from: (0..<nQuery * nQHeads * headDim).map { Float($0 % 7) * 0.1 })

                // K: every row the same constant vector → uniform scores.
                let k = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
                k.copyIn(from: [Float](repeating: 0.5,
                                       count: nKVHeads * kvStride * headDim))

                // V: row t holds the constant value `t` so mean is easy.
                var vData = [Float](repeating: 0, count: nKVHeads * kvStride * headDim)
                for t in 0..<kvStride {
                    for d in 0..<headDim { vData[t * headDim + d] = Float(t) }
                }
                let v = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
                v.copyIn(from: vData)

                let cmd = Device.shared.makeCommandBuffer()
                let out = Ops.sdpaBidirectional(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                    baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                    scale: scale, on: cmd)
                cmd.commit(); cmd.waitUntilCompleted()

                // Every query attends V rows {0,1,2,3} → mean = 1.5.
                let result = out.toArray(as: Float.self)
                #expect(result.count == nQuery * nQHeads * headDim)
                for value in result {
                    #expect(abs(value - 1.5) < 1e-3,
                            "headDim=\(headDim): expected mean V = 1.5, got \(value)")
                }
            }
        }
    }

    @Test("sdpaDecode f32 — head_dim 512 (Gemma 4 global layer)")
    func sdpaDecodeD512() {
        autoreleasepool {
            // Gemma 4 global-attention layout: head_dim 512, 8 q-heads,
            // 1 KV head (GQA fan-out 8). Routes to the d512 kernel,
            // which dispatches at 512 threads/threadgroup (the 16-wide
            // per-lane footprint caps the pipeline below 1024).
            let D = 512
            let kvStride = 4
            let nKV = 1
            let nQHeads = 8
            let nKVHeads = 1

            let q = Tensor.empty(shape: [nQHeads, D], dtype: .f32)
            let k = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)
            let v = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)

            // Every q-head = e_0; K[0] = e_0 → dot(Q, K[0]) = 1.
            var qData = [Float](repeating: 0, count: nQHeads * D)
            for h in 0..<nQHeads { qData[h * D + 0] = 1 }
            q.copyIn(from: qData)

            var kData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            kData[0] = 1                                 // K[head=0, pos=0, d=0]
            k.copyIn(from: kData)

            // V[0] = ramp; positions [1..3] are zero so an over-read
            // past `n_kv` would show up in the output.
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            for d in 0..<D { vData[d] = Float(d + 1) }   // [1, 2, …, 512]
            v.copyIn(from: vData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.sdpaDecode(q: q, k: k, v: v,
                                     nQHeads: nQHeads, nKVHeads: nKVHeads,
                                     headDim: D, nKV: nKV, kvStride: kvStride,
                                     scale: 1.0, on: cb)
            }
            // n_kv = 1 → softmax([single_score]) = 1 → output == V[0]
            // for every q-head.
            let r = out.toArray(as: Float.self)
            for h in 0..<nQHeads {
                for d in 0..<D {
                    #expect(abs(r[h * D + d] - Float(d + 1)) < 1e-3,
                            "head \(h) out[\(d)] = \(r[h * D + d]), expected \(d + 1)")
                }
            }
        }
    }

    @Test("ropeYaRN — factor=1 collapses to plain RoPE")
    func ropeYaRNFactorOneIsPlainRope() {
        autoreleasepool {
            // factor=1 → interpolation == extrapolation → the YaRN ramp
            // is a no-op and the kernel reduces to plain RoPE. `.plain`
            // also has attn_factor 1, so it must match Ops.rope exactly.
            let headDim = 128, nHeads = 2
            let qk = Tensor.empty(shape: [nHeads, headDim], dtype: .f32)
            qk.copyIn(from: (0..<nHeads * headDim).map { Float($0 % 13) * 0.1 - 0.5 })

            let cmd = Device.shared.makeCommandBuffer()
            let plain = Ops.rope(qk, position: 64, headDim: headDim,
                                 thetaBase: 1_000_000, scaling: .none, on: cmd)
            let yarn = Ops.ropeYaRN(qk, position: 64, headDim: headDim,
                                    thetaBase: 1_000_000, yarn: .plain, on: cmd)
            cmd.commit(); cmd.waitUntilCompleted()

            let p = plain.toArray(as: Float.self)
            let y = yarn.toArray(as: Float.self)
            for i in 0..<p.count {
                #expect(abs(p[i] - y[i]) < 1e-5, "i=\(i): plain \(p[i]) vs yarn \(y[i])")
            }
        }
    }

    @Test("ropeYaRN — position 0 is identity")
    func ropeYaRNIdentityAtPositionZero() {
        autoreleasepool {
            // position 0 → theta 0 → cos 1 / sin 0; attn_factor 1 → the
            // rotation is the identity regardless of the YaRN band.
            let headDim = 128, nHeads = 2
            let input = (0..<nHeads * headDim).map { Float($0 % 7) * 0.1 }
            let qk = Tensor.empty(shape: [nHeads, headDim], dtype: .f32)
            qk.copyIn(from: input)

            let yarn = Ops.RoPEYaRN.from(headDim: headDim, thetaBase: 1_000_000,
                                         factor: 16, betaFast: 32, betaSlow: 1,
                                         originalMaxPosition: 16384)
            let cmd = Device.shared.makeCommandBuffer()
            let out = Ops.ropeYaRN(qk, position: 0, headDim: headDim,
                                   thetaBase: 1_000_000, yarn: yarn, on: cmd)
            cmd.commit(); cmd.waitUntilCompleted()

            let o = out.toArray(as: Float.self)
            for i in 0..<o.count {
                #expect(abs(o[i] - input[i]) < 1e-5, "i=\(i): expected \(input[i]), got \(o[i])")
            }
        }
    }

    @Test("RoPEYaRN.from derives a sane correction band")
    func ropeYaRNFromCorrectionBand() {
        // Nemotron-Labs-Diffusion params — the band must land inside
        // [0, headDim) with high > low.
        let y = Ops.RoPEYaRN.from(headDim: 128, thetaBase: 1_000_000,
                                  factor: 16, betaFast: 32, betaSlow: 1,
                                  originalMaxPosition: 16384)
        #expect(y.factor == 16)
        #expect(y.low >= 0 && y.low < y.high)
        #expect(y.high <= 127)
        #expect(y.attnFactor == 1)   // mscale == mscale_all_dim default
    }

    @Test("gemm — multi-row matmul matches a CPU reference")
    func gemmMatchesCPU() {
        autoreleasepool {
            // out[r,o] = Σ_k weight[o,k]·input[r,k]. nRows / outDim are
            // not multiples of the 32×32 tile — exercises the edge path.
            let nRows = 5, inDim = 48, outDim = 7
            let wData = (0..<outDim * inDim).map { Float($0 % 11) * 0.1 - 0.4 }
            let xData = (0..<nRows * inDim).map { Float($0 % 9) * 0.1 - 0.2 }
            let weight = Tensor.empty(shape: [outDim, inDim], dtype: .f32)
            let input = Tensor.empty(shape: [nRows, inDim], dtype: .f32)
            weight.copyIn(from: wData)
            input.copyIn(from: xData)

            let cmd = Device.shared.makeCommandBuffer()
            let out = Ops.gemm(weight: weight, input: input, nRows: nRows, on: cmd)
            cmd.commit(); cmd.waitUntilCompleted()

            let got = out.toArray(as: Float.self)
            #expect(got.count == nRows * outDim)
            for r in 0..<nRows {
                for o in 0..<outDim {
                    var acc: Float = 0
                    for k in 0..<inDim { acc += wData[o * inDim + k] * xData[r * inDim + k] }
                    #expect(abs(got[r * outDim + o] - acc) < 1e-4,
                            "r=\(r) o=\(o): expected \(acc), got \(got[r * outDim + o])")
                }
            }
        }
    }

    @Test("ropeProportional f32 — matches a CPU ProportionalRoPE oracle")
    func ropeProportionalMatchesOracle() {
        autoreleasepool {
            // Gemma 4 global-layer ProportionalRoPE: dims=512,
            // rotatedDim=128. Rotates pairs (i, i + 256) for i ∈ [0, 64)
            // at frequency theta^(-2i/512); dims [64, 256) and their
            // partners pass through untouched. Drives the shared
            // ffai_rope_llama kernel — this pins the pairing offset
            // (headDim/2, NOT rotatedDim/2) and the frequency
            // denominator (the full headDim).
            let headDim = 512
            let rotatedDim = 128
            let theta: Float = 1_000_000
            let position = 7
            let nHeads = 3

            // Deterministic, non-trivial input.
            var data = [Float](repeating: 0, count: nHeads * headDim)
            for i in 0..<data.count {
                data[i] = Float((i * 7) % 19 - 9) * 0.13
            }
            let qk = Tensor.empty(shape: [nHeads, headDim], dtype: .f32)
            qk.copyIn(from: data)

            runAndWait { cb in
                Gemma4Ops.ropeProportional(
                    qk, position: position, headDim: headDim,
                    rotatedDim: rotatedDim, thetaBase: theta, on: cb)
            }
            let got = qk.toArray(as: Float.self)

            // CPU oracle of the reference's ProportionalRoPE.
            let half = headDim / 2            // 256 — pairing offset
            let rotatedPairs = rotatedDim / 2 // 64 — rotated pair count
            var expected = data
            for h in 0..<nHeads {
                let base = h * headDim
                for i in 0..<rotatedPairs {
                    // inv_freq = theta^(-2i/headDim) = theta^(-i/half)
                    let invFreq = Float(
                        pow(Double(theta), -Double(i) / Double(half)))
                    let angle = Double(position) * Double(invFreq)
                    let c = Float(cos(angle))
                    let s = Float(sin(angle))
                    let x1 = data[base + i]
                    let x2 = data[base + i + half]
                    expected[base + i] = x1 * c - x2 * s
                    expected[base + i + half] = x1 * s + x2 * c
                }
                // i ∈ [rotatedPairs, half): pass-through (unchanged).
            }
            for idx in 0..<got.count {
                #expect(abs(got[idx] - expected[idx]) < 1e-3,
                        "idx=\(idx): got \(got[idx]) expected \(expected[idx])")
            }
        }
    }
}
