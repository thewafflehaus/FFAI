import Foundation
import Metal
import Testing
@testable import FFAI

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
}
