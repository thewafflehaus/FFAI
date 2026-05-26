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
// GPU correctness / smoke tests for `Ops.*` wrappers that don't fit
// neatly into the elementwise / SDPA / dequant suites: blit/cast/fused
// activations, KV cache append + round-trip, GDN/Mamba prep+chunk +
// fused mixer norms, SDPA prefill-MMA, MoE BM=8 / scalar M=1 variants,
// unpermute, dynamic-M dequant GEMM.
//
// Each test follows the canonical OpsTests.swift pattern:
//   autoreleasepool { … runAndWait { cb in Ops.foo(…, on: cb) } … }
//
// For ops with an obvious cheap CPU reference we assert numerical
// correctness; for ops with deep multi-tensor shape contracts we
// allocate production-realistic inputs and assert the output buffer
// has the right element count plus all-finite (no NaN / Inf) values,
// which still exercises the kernel dispatch path end-to-end.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import FFAI

@Suite("Ops — special-path wrappers")
struct OpsSpecialPathTests {

    // MARK: - Element-wise blit / cast / fused

    @Test("copy f32 — blit duplicates src element-for-element")
    func copyF32() {
        autoreleasepool {
            let src = Tensor.empty(shape: [6], dtype: .f32)
            src.copyIn(from: [Float(1), -2, 3, -4, 5, -6])
            let dst = Tensor.empty(shape: [6], dtype: .f32)
            dst.zero()
            runAndWait { cb in Ops.copy(src, into: dst, on: cb) }
            #expect(dst.toArray(as: Float.self) == [1, -2, 3, -4, 5, -6])
        }
    }

    @Test("castToF32 f16 — promotes half-precision values to fp32")
    func castToF32FromF16() {
        autoreleasepool {
            let src = Tensor.empty(shape: [4], dtype: .f16)
            src.copyIn(from: [Float16(0.5), -1.25, 2, 0])
            let dst = Tensor.empty(shape: [4], dtype: .f32)
            dst.zero()
            runAndWait { cb in Ops.castToF32(src, into: dst, on: cb) }
            let got = dst.toArray(as: Float.self)
            // f16 representable exactly for these values.
            #expect(abs(got[0] - 0.5) < 1e-6)
            #expect(abs(got[1] - -1.25) < 1e-6)
            #expect(abs(got[2] - 2.0) < 1e-6)
            #expect(abs(got[3]) < 1e-6)
        }
    }

    @Test("siluCastToF32 f16 — fused silu + promotion matches silu(x)")
    func siluCastToF32FromF16() {
        autoreleasepool {
            let src = Tensor.empty(shape: [4], dtype: .f16)
            src.copyIn(from: [Float16(0), 1, -1, 2])
            let dst = Tensor.empty(shape: [4], dtype: .f32)
            dst.zero()
            runAndWait { cb in Ops.siluCastToF32(src, into: dst, on: cb) }
            let r = dst.toArray(as: Float.self)
            // silu(x) = x * sigmoid(x). bf16/f16 input narrows accuracy.
            #expect(abs(r[0]) < 1e-3)
            #expect(abs(r[1] - Float(1.0 / (1.0 + exp(-1.0)))) < 5e-3)
            #expect(abs(r[2] - Float(-1.0 / (1.0 + exp(1.0)))) < 5e-3)
            #expect(abs(r[3] - Float(2.0 / (1.0 + exp(-2.0)))) < 5e-3)
        }
    }

    @Test("castToF32Two f16 — same as two sequential castToF32 calls")
    func castToF32TwoMatchesSequential() {
        autoreleasepool {
            let n = 4
            let a = Tensor.empty(shape: [n], dtype: .f16)
            a.copyIn(from: [Float16(0.5), 2, -2, 4])
            let b = Tensor.empty(shape: [n], dtype: .f16)
            b.copyIn(from: [Float16(-1), 0, 1, 0.25])
            let refA = Tensor.empty(shape: [n], dtype: .f32)
            let refB = Tensor.empty(shape: [n], dtype: .f32)
            runAndWait { cb in
                Ops.castToF32(a, into: refA, on: cb)
                Ops.castToF32(b, into: refB, on: cb)
            }
            let outA = Tensor.empty(shape: [n], dtype: .f32)
            let outB = Tensor.empty(shape: [n], dtype: .f32)
            runAndWait { cb in
                Ops.castToF32Two(a, into: outA, b, into: outB, on: cb)
            }
            let rA = refA.toArray(as: Float.self)
            let gA = outA.toArray(as: Float.self)
            let rB = refB.toArray(as: Float.self)
            let gB = outB.toArray(as: Float.self)
            for i in 0 ..< n {
                #expect(abs(gA[i] - rA[i]) < 1e-5, "a[\(i)]")
                #expect(abs(gB[i] - rB[i]) < 1e-5, "b[\(i)]")
            }
        }
    }

    @Test("castToF32Three bf16 — same as three sequential castToF32 calls")
    func castToF32ThreeMatchesSequential() {
        autoreleasepool {
            // bf16 round-trip — fewer mantissa bits than f16 but still
            // exact when source values fit the type.
            let n = 4
            let aSrc: [Float] = [0.5, 2, -2, 4]
            let bSrc: [Float] = [-1, 0, 1, 0.25]
            let cSrc: [Float] = [3.5, -0.125, 1.0, 0]
            // bf16 store: pick the top 16 bits of each Float bit pattern.
            func toBF16Bits(_ xs: [Float]) -> [UInt16] {
                xs.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            }
            let a = Tensor.empty(shape: [n], dtype: .bf16)
            a.copyIn(from: toBF16Bits(aSrc))
            let b = Tensor.empty(shape: [n], dtype: .bf16)
            b.copyIn(from: toBF16Bits(bSrc))
            let c = Tensor.empty(shape: [n], dtype: .bf16)
            c.copyIn(from: toBF16Bits(cSrc))
            let refA = Tensor.empty(shape: [n], dtype: .f32)
            let refB = Tensor.empty(shape: [n], dtype: .f32)
            let refC = Tensor.empty(shape: [n], dtype: .f32)
            runAndWait { cb in
                Ops.castToF32(a, into: refA, on: cb)
                Ops.castToF32(b, into: refB, on: cb)
                Ops.castToF32(c, into: refC, on: cb)
            }
            let outA = Tensor.empty(shape: [n], dtype: .f32)
            let outB = Tensor.empty(shape: [n], dtype: .f32)
            let outC = Tensor.empty(shape: [n], dtype: .f32)
            runAndWait { cb in
                Ops.castToF32Three(
                    a, into: outA, b, into: outB, c, into: outC, on: cb)
            }
            let rA = refA.toArray(as: Float.self)
            let gA = outA.toArray(as: Float.self)
            let rB = refB.toArray(as: Float.self)
            let gB = outB.toArray(as: Float.self)
            let rC = refC.toArray(as: Float.self)
            let gC = outC.toArray(as: Float.self)
            for i in 0 ..< n {
                #expect(abs(gA[i] - rA[i]) < 1e-5, "a[\(i)]")
                #expect(abs(gB[i] - rB[i]) < 1e-5, "b[\(i)]")
                #expect(abs(gC[i] - rC[i]) < 1e-5, "c[\(i)]")
            }
        }
    }

    @Test("swigluMany f32 — N=3 separate dispatches share one encoder")
    func swigluManyMatchesSequential() {
        autoreleasepool {
            let n = 8
            // Build three (gate, up) pairs of different sizes — the
            // wrapper sets PSO once and dispatches each independently.
            let g0 = Tensor.empty(shape: [n], dtype: .f32)
            g0.copyIn(from: (0 ..< n).map { Float($0) * 0.1 })
            let u0 = Tensor.empty(shape: [n], dtype: .f32)
            u0.copyIn(from: (0 ..< n).map { Float($0 + 1) })
            let g1 = Tensor.empty(shape: [n], dtype: .f32)
            g1.copyIn(from: (0 ..< n).map { Float($0) * -0.2 })
            let u1 = Tensor.empty(shape: [n], dtype: .f32)
            u1.copyIn(from: (0 ..< n).map { Float($0) * 0.5 + 1 })
            let g2 = Tensor.empty(shape: [n], dtype: .f32)
            g2.copyIn(from: (0 ..< n).map { Float($0) * 0.3 - 0.1 })
            let u2 = Tensor.empty(shape: [n], dtype: .f32)
            u2.copyIn(from: (0 ..< n).map { Float(n - $0) })
            var ref0: Tensor!
            var ref1: Tensor!
            var ref2: Tensor!
            runAndWait { cb in
                ref0 = Ops.swiglu(gate: g0, up: u0, on: cb)
                ref1 = Ops.swiglu(gate: g1, up: u1, on: cb)
                ref2 = Ops.swiglu(gate: g2, up: u2, on: cb)
            }
            let out0 = Tensor.empty(shape: [n], dtype: .f32)
            let out1 = Tensor.empty(shape: [n], dtype: .f32)
            let out2 = Tensor.empty(shape: [n], dtype: .f32)
            runAndWait { cb in
                Ops.swigluMany(
                    gates: [g0, g1, g2],
                    ups: [u0, u1, u2],
                    outs: [out0, out1, out2],
                    on: cb)
            }
            let pairs: [(Tensor, Tensor)] = [(ref0, out0), (ref1, out1), (ref2, out2)]
            for (idx, pair) in pairs.enumerated() {
                let r = pair.0.toArray(as: Float.self)
                let g = pair.1.toArray(as: Float.self)
                for i in 0 ..< n {
                    #expect(abs(r[i] - g[i]) < 1e-5, "pair \(idx) [\(i)]")
                }
            }
        }
    }

    @Test("gatedMixerNormMany f32 — T=2 matches two sequential gatedMixerNorm calls")
    func gatedMixerNormManyMatchesSequential() {
        autoreleasepool {
            let t = 2
            let hv = 2
            let dv = 8
            let total = t * hv * dv
            let perToken = hv * dv
            let yData = (0 ..< total).map { Float($0 + 1) * 0.1 }
            let zData = (0 ..< total).map { Float16(Float($0 % 5) * 0.2 - 0.4) }
            let weightData = (0 ..< dv).map { Float16(0.5 + Float($0) * 0.05) }
            let weight = Tensor.empty(shape: [dv], dtype: .f16)
            weight.copyIn(from: weightData)
            let epsBuf = Tensor.empty(shape: [1], dtype: .f32)
            epsBuf.copyIn(from: [Float(1e-6)])
            // Reference: two single-token gatedMixerNorm dispatches with
            // pre-sliced inputs. All allocations + slicing happen up
            // front; runAndWait only carries kernel work.
            var refSlices: [Tensor] = []
            for tt in 0 ..< t {
                let ySlice = Tensor.empty(shape: [hv, dv], dtype: .f32)
                ySlice.copyIn(from: Array(yData[tt * perToken ..< (tt + 1) * perToken]))
                let zSlice = Tensor.empty(shape: [hv, dv], dtype: .f16)
                zSlice.copyIn(from: Array(zData[tt * perToken ..< (tt + 1) * perToken]))
                let outSlice = Tensor.empty(shape: [hv, dv], dtype: .f16)
                runAndWait { cb in
                    Ops.gatedMixerNorm(
                        y: ySlice, z: zSlice, weight: weight, epsBuf: epsBuf,
                        into: outSlice,
                        numValueHeads: hv, valueHeadDim: dv, on: cb)
                }
                refSlices.append(outSlice)
            }
            // Batched dispatch over the full T·Hv·Dv span.
            let y = Tensor.empty(shape: [t, hv, dv], dtype: .f32)
            y.copyIn(from: yData)
            let z = Tensor.empty(shape: [t, hv, dv], dtype: .f16)
            z.copyIn(from: zData)
            let outMany = Tensor.empty(shape: [t, hv, dv], dtype: .f16)
            runAndWait { cb in
                Ops.gatedMixerNormMany(
                    y: y, z: z, weight: weight, epsBuf: epsBuf,
                    into: outMany,
                    t: t, numValueHeads: hv, valueHeadDim: dv, on: cb)
            }
            let g = outMany.toArray(as: Float16.self)
            for tt in 0 ..< t {
                let r = refSlices[tt].toArray(as: Float16.self)
                for i in 0 ..< perToken {
                    let rf = Float(r[i])
                    let gf = Float(g[tt * perToken + i])
                    #expect(abs(rf - gf) < 5e-3, "t=\(tt) i=\(i): ref \(rf) vs got \(gf)")
                }
            }
        }
    }

    @Test("siluCastF32PlusCastF32Two f16 — same as sequential silu+two casts")
    func siluCastF32PlusCastF32TwoFromF16() {
        autoreleasepool {
            let n = 4
            // Reference: three separate dispatches.
            let s = Tensor.empty(shape: [n], dtype: .f16)
            s.copyIn(from: [Float16(0), 1, -1, 2])
            let a = Tensor.empty(shape: [n], dtype: .f16)
            a.copyIn(from: [Float16(0.5), 2, -2, 4])
            let b = Tensor.empty(shape: [n], dtype: .f16)
            b.copyIn(from: [Float16(-1), 0, 1, 0.25])
            let refSilu = Tensor.empty(shape: [n], dtype: .f32)
            let refA = Tensor.empty(shape: [n], dtype: .f32)
            let refB = Tensor.empty(shape: [n], dtype: .f32)
            runAndWait { cb in
                Ops.siluCastToF32(s, into: refSilu, on: cb)
                Ops.castToF32(a, into: refA, on: cb)
                Ops.castToF32(b, into: refB, on: cb)
            }
            // Batched: one encoder, switches PSO between silu+cast and plain cast.
            let outSilu = Tensor.empty(shape: [n], dtype: .f32)
            let outA = Tensor.empty(shape: [n], dtype: .f32)
            let outB = Tensor.empty(shape: [n], dtype: .f32)
            runAndWait { cb in
                Ops.siluCastF32PlusCastF32Two(
                    siluIn: s, into: outSilu,
                    a, into: outA,
                    b, into: outB,
                    on: cb)
            }
            let rRef = refSilu.toArray(as: Float.self)
            let rGot = outSilu.toArray(as: Float.self)
            let aRef = refA.toArray(as: Float.self)
            let aGot = outA.toArray(as: Float.self)
            let bRef = refB.toArray(as: Float.self)
            let bGot = outB.toArray(as: Float.self)
            for i in 0 ..< n {
                #expect(abs(rGot[i] - rRef[i]) < 1e-5, "silu[\(i)]")
                #expect(abs(aGot[i] - aRef[i]) < 1e-5, "a[\(i)]")
                #expect(abs(bGot[i] - bRef[i]) < 1e-5, "b[\(i)]")
            }
        }
    }

    @Test("swiglu f32 — out[i] = silu(gate[i]) * up[i]")
    func swigluF32() {
        autoreleasepool {
            let gate = Tensor.empty(shape: [4], dtype: .f32)
            let up = Tensor.empty(shape: [4], dtype: .f32)
            gate.copyIn(from: [Float(0), 1, -1, 2])
            up.copyIn(from: [Float(3), 5, 7, 11])
            var out: Tensor!
            runAndWait { cb in out = Ops.swiglu(gate: gate, up: up, on: cb) }
            let r = out.toArray(as: Float.self)
            // silu(0)=0 → 0 * 3 = 0; silu(1) ≈ 0.7311 → * 5 ≈ 3.656; etc.
            #expect(abs(r[0]) < 1e-5)
            #expect(abs(r[1] - 5 * Float(1.0 / (1.0 + exp(-1.0)))) < 1e-3)
            #expect(abs(r[2] - 7 * Float(-1.0 / (1.0 + exp(1.0)))) < 1e-3)
            #expect(abs(r[3] - 11 * Float(2.0 / (1.0 + exp(-2.0)))) < 1e-3)
        }
    }

    @Test("sigmoidScalarFMA f32 — out = base + sigmoid(gate) * value")
    func sigmoidScalarFMAF32() {
        autoreleasepool {
            let gate = Tensor.empty(shape: [1], dtype: .f32)
            gate.copyIn(from: [Float(0)])  // sigmoid(0) = 0.5
            let value = Tensor.empty(shape: [4], dtype: .f32)
            value.copyIn(from: [Float(2), 4, 6, 8])
            let base = Tensor.empty(shape: [4], dtype: .f32)
            base.copyIn(from: [Float(10), 20, 30, 40])
            let out = Tensor.empty(shape: [4], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.sigmoidScalarFMA(
                    gate: gate, value: value, base: base,
                    into: out, on: cb)
            }
            // sigmoid(0) = 0.5 → out[i] = base[i] + 0.5 * value[i]
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0] - 11) < 1e-4)
            #expect(abs(r[1] - 22) < 1e-4)
            #expect(abs(r[2] - 33) < 1e-4)
            #expect(abs(r[3] - 44) < 1e-4)
        }
    }

    @Test("sigmoidScalarFMAResidual f32 — out = residual + base + sigmoid(gate) * value")
    func sigmoidScalarFMAResidualF32() {
        autoreleasepool {
            let gate = Tensor.empty(shape: [1], dtype: .f32)
            gate.copyIn(from: [Float(0)])  // sigmoid(0) = 0.5
            let value = Tensor.empty(shape: [4], dtype: .f32)
            value.copyIn(from: [Float(2), 4, 6, 8])
            let base = Tensor.empty(shape: [4], dtype: .f32)
            base.copyIn(from: [Float(10), 20, 30, 40])
            let residual = Tensor.empty(shape: [4], dtype: .f32)
            residual.copyIn(from: [Float(100), 200, 300, 400])
            let out = Tensor.empty(shape: [4], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.sigmoidScalarFMAResidual(
                    gate: gate, value: value, base: base, residual: residual,
                    into: out, on: cb)
            }
            // out[i] = residual[i] + base[i] + 0.5 * value[i]
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0] - 111) < 1e-4)
            #expect(abs(r[1] - 222) < 1e-4)
            #expect(abs(r[2] - 333) < 1e-4)
            #expect(abs(r[3] - 444) < 1e-4)
        }
    }

    @Test("scalarFMA f32 — out = base + scalar * value")
    func scalarFMAF32() {
        autoreleasepool {
            let scalar = Tensor.empty(shape: [1], dtype: .f32)
            scalar.copyIn(from: [Float(0.25)])
            let value = Tensor.empty(shape: [4], dtype: .f32)
            value.copyIn(from: [Float(4), 8, 12, 16])
            let base = Tensor.empty(shape: [4], dtype: .f32)
            base.copyIn(from: [Float(10), 20, 30, 40])
            let out = Tensor.empty(shape: [4], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.scalarFMA(
                    scalar: scalar, value: value, base: base,
                    into: out, on: cb)
            }
            // 0.25 * [4,8,12,16] = [1,2,3,4]; + base = [11,22,33,44]
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0] - 11) < 1e-4)
            #expect(abs(r[1] - 22) < 1e-4)
            #expect(abs(r[2] - 33) < 1e-4)
            #expect(abs(r[3] - 44) < 1e-4)
        }
    }

    @Test("scalarFMAMany f32 — N=3 serial accumulate into acc on one encoder")
    func scalarFMAManyF32() {
        autoreleasepool {
            let n = 4
            // Three (scalar, value) pairs accumulating into acc starting
            // from initial values. Final acc[i] = acc0[i] + sum_k s_k * v_k[i].
            let scalars: [Tensor] = (0 ..< 3).map { _ in
                Tensor.empty(shape: [1], dtype: .f32)
            }
            scalars[0].copyIn(from: [Float(1.0)])
            scalars[1].copyIn(from: [Float(2.0)])
            scalars[2].copyIn(from: [Float(0.5)])
            let values: [Tensor] = (0 ..< 3).map { _ in
                Tensor.empty(shape: [n], dtype: .f32)
            }
            values[0].copyIn(from: [Float(1), 2, 3, 4])
            values[1].copyIn(from: [Float(10), 20, 30, 40])
            values[2].copyIn(from: [Float(100), 200, 300, 400])
            let acc = Tensor.empty(shape: [n], dtype: .f32)
            acc.copyIn(from: [Float(1000), 1000, 1000, 1000])
            runAndWait { cb in
                Ops.scalarFMAMany(scalars: scalars, values: values, acc: acc, on: cb)
            }
            // acc[i] = 1000 + 1·v0[i] + 2·v1[i] + 0.5·v2[i]
            // i=0: 1000 + 1 + 20 + 50 = 1071
            // i=1: 1000 + 2 + 40 + 100 = 1142
            // i=2: 1000 + 3 + 60 + 150 = 1213
            // i=3: 1000 + 4 + 80 + 200 = 1284
            let r = acc.toArray(as: Float.self)
            #expect(abs(r[0] - 1071) < 1e-3)
            #expect(abs(r[1] - 1142) < 1e-3)
            #expect(abs(r[2] - 1213) < 1e-3)
            #expect(abs(r[3] - 1284) < 1e-3)
        }
    }

    @Test("scalarFMAChain8 f32 — out = sum_{k=0..8} scalar_k * value_k")
    func scalarFMAChain8F32() {
        autoreleasepool {
            let n = 4
            let scalars: [Tensor] = (0 ..< 8).map { i in
                let t = Tensor.empty(shape: [1], dtype: .f32)
                t.copyIn(from: [Float(i + 1)])  // 1, 2, 3, ..., 8
                return t
            }
            let values: [Tensor] = (0 ..< 8).map { i in
                let t = Tensor.empty(shape: [n], dtype: .f32)
                t.copyIn(from: (0 ..< n).map { Float((i + 1) * ($0 + 1)) })
                return t
            }
            let out = Tensor.empty(shape: [n], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.scalarFMAChain8(scalars: scalars, values: values, out: out, on: cb)
            }
            // value_k[j] = (k+1)·(j+1); scalar_k = k+1
            // out[j] = sum_{k=0..7} (k+1)^2 · (j+1) = (j+1) · sum k^2 (k=1..8) = (j+1) · 204
            let r = out.toArray(as: Float.self)
            for j in 0 ..< n {
                let expected = Float((j + 1) * 204)
                #expect(abs(r[j] - expected) < 1e-3)
            }
        }
    }

    // MARK: - KV cache append

    @Test("kvCacheUpdateKVMany f32 — writes T rows at the right positions")
    func kvCacheUpdateKVManyF32() {
        autoreleasepool {
            let t = 2
            let nKV = 2
            let headDim = 4
            let maxSeq = 4
            let kSrc = Tensor.empty(shape: [t, nKV, headDim], dtype: .f32)
            // Token 0: heads [1..4, 5..8]; token 1: heads [9..12, 13..16]
            kSrc.copyIn(from: (0 ..< t * nKV * headDim).map { Float($0 + 1) })
            let vSrc = Tensor.empty(shape: [t, nKV, headDim], dtype: .f32)
            vSrc.copyIn(from: (0 ..< t * nKV * headDim).map { Float($0 + 1) * 10 })
            let kCache = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            kCache.zero()
            let vCache = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            vCache.zero()
            let positions = Tensor.empty(shape: [t], dtype: .u32)
            positions.copyIn(from: [UInt32(1), UInt32(2)])
            runAndWait { cb in
                Ops.kvCacheUpdateKVMany(
                    kSrc: kSrc, kCache: kCache,
                    vSrc: vSrc, vCache: vCache,
                    positions: positions, t: t,
                    nKVHeads: nKV, headDim: headDim, maxSeq: maxSeq, on: cb)
            }
            let kGot = kCache.toArray(as: Float.self)
            // head 0 row 1 ← token0 head0 = [1,2,3,4]
            #expect(Array(kGot[4 ..< 8]) == [1, 2, 3, 4])
            // head 0 row 2 ← token1 head0 = [9,10,11,12]
            #expect(Array(kGot[8 ..< 12]) == [9, 10, 11, 12])
            // head 1 row 1 ← token0 head1 = [5,6,7,8]
            #expect(Array(kGot[20 ..< 24]) == [5, 6, 7, 8])
            // head 1 row 2 ← token1 head1 = [13,14,15,16]
            #expect(Array(kGot[24 ..< 28]) == [13, 14, 15, 16])
            // Untouched slots stay zero.
            #expect(Array(kGot[0 ..< 4]) == [0, 0, 0, 0])
            #expect(Array(kGot[12 ..< 16]) == [0, 0, 0, 0])
            let vGot = vCache.toArray(as: Float.self)
            #expect(Array(vGot[4 ..< 8]) == [10, 20, 30, 40])
            #expect(Array(vGot[8 ..< 12]) == [90, 100, 110, 120])
        }
    }

    @Test("kvCacheUpdateKV f32 — appends K and V rows on one encoder")
    func kvCacheUpdateKVF32() {
        autoreleasepool {
            let nKV = 2
            let headDim = 4
            let maxSeq = 3
            let kSrc = Tensor.empty(shape: [nKV, headDim], dtype: .f32)
            kSrc.copyIn(from: [Float(1), 2, 3, 4, 5, 6, 7, 8])
            let vSrc = Tensor.empty(shape: [nKV, headDim], dtype: .f32)
            vSrc.copyIn(from: [Float(10), 20, 30, 40, 50, 60, 70, 80])
            let kCache = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            kCache.zero()
            let vCache = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            vCache.zero()
            runAndWait { cb in
                Ops.kvCacheUpdateKV(
                    kSrc: kSrc, kCache: kCache,
                    vSrc: vSrc, vCache: vCache,
                    nKVHeads: nKV, headDim: headDim,
                    maxSeq: maxSeq, position: 1, on: cb)
            }
            let kGot = kCache.toArray(as: Float.self)
            #expect(Array(kGot[4 ..< 8]) == [1, 2, 3, 4])
            #expect(Array(kGot[16 ..< 20]) == [5, 6, 7, 8])
            let vGot = vCache.toArray(as: Float.self)
            #expect(Array(vGot[4 ..< 8]) == [10, 20, 30, 40])
            #expect(Array(vGot[16 ..< 20]) == [50, 60, 70, 80])
        }
    }

    @Test("kvCacheUpdate f32 — writes one row into [nKV, maxSeq, headDim]")
    func kvCacheUpdateF32() {
        autoreleasepool {
            let nKV = 2
            let headDim = 4
            let maxSeq = 3
            let src = Tensor.empty(shape: [nKV, headDim], dtype: .f32)
            src.copyIn(from: [
                Float(1), 2, 3, 4,  // head 0
                5, 6, 7, 8,
            ])  // head 1
            let cache = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            cache.zero()
            runAndWait { cb in
                Ops.kvCacheUpdate(
                    src: src, into: cache,
                    nKVHeads: nKV, headDim: headDim,
                    maxSeq: maxSeq, position: 1, on: cb)
            }
            // Expect head 0 row 1 = [1,2,3,4], head 1 row 1 = [5,6,7,8],
            // all other slots still zero.
            let got = cache.toArray(as: Float.self)
            // head 0
            #expect(got[0 ..< 4] == [0, 0, 0, 0])  // row 0
            #expect(Array(got[4 ..< 8]) == [1, 2, 3, 4])  // row 1
            #expect(got[8 ..< 12] == [0, 0, 0, 0])  // row 2
            // head 1
            #expect(got[12 ..< 16] == [0, 0, 0, 0])  // row 0
            #expect(Array(got[16 ..< 20]) == [5, 6, 7, 8])  // row 1
            #expect(got[20 ..< 24] == [0, 0, 0, 0])  // row 2
        }
    }

    // MARK: - KV quantization round-trips
    //
    // For each precision we run quantize → bulk-dequant on the same row
    // and expect the recovered value to be close to the original
    // (affine quantization is approximate, so we use coarse tolerances).
    // Each round-trip exercises both the encode and decode kernel.

    @Test("quantizeKVInt8 + bulkDequantKVInt8 — round-trip recovers input")
    func quantizeBulkDequantKVInt8RoundTrip() {
        autoreleasepool {
            let nKV = 1
            let headDim = 32
            let maxSeq = 4
            let groupSize = 32
            // src has known per-element values; one row only is written.
            let srcVals: [Float] = (0 ..< headDim).map { Float($0) * 0.05 - 0.7 }
            let src = Tensor.empty(shape: [nKV, headDim], dtype: .f32)
            src.copyIn(from: srcVals)
            // int8 packs 4 values per uint32.
            let packs = headDim / 4
            let groups = headDim / groupSize
            let w = Tensor.empty(shape: [nKV, maxSeq, packs], dtype: .u32)
            let s = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            let b = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            w.zero()
            s.zero()
            b.zero()

            let pos = 2
            runAndWait { cb in
                Ops.quantizeKVInt8(
                    src: src,
                    weights: w, scales: s, biases: b,
                    nKVHeads: nKV, headDim: headDim,
                    maxSeq: maxSeq, groupSize: groupSize,
                    position: pos, on: cb)
            }
            // Bulk-dequant into a working buffer of the same layout as
            // the cache (`[nKV, maxSeq, headDim]`) for `pos+1` positions
            // so the written slot is included.
            let working = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            working.zero()
            runAndWait { cb in
                Ops.bulkDequantKVInt8(
                    weights: w, scales: s, biases: b,
                    into: working,
                    nKVHeads: nKV, headDim: headDim,
                    maxSeq: maxSeq, groupSize: groupSize,
                    nPositions: pos + 1, on: cb)
            }
            let got = working.toArray(as: Float.self)
            // Recovered slice for (head 0, pos 2) lives at offset
            // `pos * headDim`.
            for i in 0 ..< headDim {
                let want = srcVals[i]
                let recovered = got[pos * headDim + i]
                // int8 affine quant ≈ src/255 ≈ 0.005 abs tolerance.
                #expect(
                    abs(recovered - want) < 0.01,
                    "i=\(i): got \(recovered) vs \(want)")
            }
        }
    }

    @Test("quantizeKVInt4 + bulkDequantKVInt4 — round-trip stays in tolerance")
    func quantizeBulkDequantKVInt4RoundTrip() {
        autoreleasepool {
            let nKV = 1
            let headDim = 32
            let maxSeq = 4
            let groupSize = 32
            let srcVals: [Float] = (0 ..< headDim).map { Float($0) * 0.05 - 0.7 }
            let src = Tensor.empty(shape: [nKV, headDim], dtype: .f32)
            src.copyIn(from: srcVals)
            // int4 packs 8 values per uint32.
            let packs = headDim / 8
            let groups = headDim / groupSize
            let w = Tensor.empty(shape: [nKV, maxSeq, packs], dtype: .u32)
            let s = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            let b = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            w.zero()
            s.zero()
            b.zero()

            let pos = 1
            runAndWait { cb in
                Ops.quantizeKVInt4(
                    src: src,
                    weights: w, scales: s, biases: b,
                    nKVHeads: nKV, headDim: headDim,
                    maxSeq: maxSeq, groupSize: groupSize,
                    position: pos, on: cb)
            }
            let working = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            working.zero()
            runAndWait { cb in
                Ops.bulkDequantKVInt4(
                    weights: w, scales: s, biases: b,
                    into: working,
                    nKVHeads: nKV, headDim: headDim,
                    maxSeq: maxSeq, groupSize: groupSize,
                    nPositions: pos + 1, on: cb)
            }
            let got = working.toArray(as: Float.self)
            // int4 affine quant: range/15 ≈ 0.11 step → 0.06 abs tolerance.
            for i in 0 ..< headDim {
                let want = srcVals[i]
                let recovered = got[pos * headDim + i]
                #expect(
                    abs(recovered - want) < 0.07,
                    "i=\(i): got \(recovered) vs \(want)")
            }
        }
    }

    @Test("quantizeKVAffine + bulkDequantKVAffine — bits=8 dispatch round-trips")
    func quantizeBulkDequantKVAffineBits8() {
        autoreleasepool {
            let nKV = 1
            let headDim = 32
            let maxSeq = 2
            let groupSize = 32
            let srcVals: [Float] = (0 ..< headDim).map { Float($0) * 0.04 - 0.5 }
            let src = Tensor.empty(shape: [nKV, headDim], dtype: .f32)
            src.copyIn(from: srcVals)
            let packs = headDim / 4
            let groups = headDim / groupSize
            let w = Tensor.empty(shape: [nKV, maxSeq, packs], dtype: .u32)
            let s = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            let b = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            w.zero()
            s.zero()
            b.zero()

            runAndWait { cb in
                Ops.quantizeKVAffine(
                    src: src,
                    weights: w, scales: s, biases: b,
                    nKVHeads: nKV, headDim: headDim,
                    maxSeq: maxSeq, groupSize: groupSize,
                    position: 0, bits: 8, on: cb)
            }
            let working = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            working.zero()
            runAndWait { cb in
                Ops.bulkDequantKVAffine(
                    weights: w, scales: s, biases: b,
                    into: working,
                    nKVHeads: nKV, headDim: headDim,
                    maxSeq: maxSeq, groupSize: groupSize,
                    nPositions: 1, bits: 8, on: cb)
            }
            let got = working.toArray(as: Float.self)
            for i in 0 ..< headDim {
                #expect(
                    abs(got[i] - srcVals[i]) < 0.01,
                    "i=\(i): got \(got[i]) vs \(srcVals[i])")
            }
        }
    }

    // MARK: - Fused mixer / GDN prep / chunk recurrence

    @Test("gatedMixerNorm f32 — exercises kernel without NaN / shape drift")
    func gatedMixerNormSmoke() {
        autoreleasepool {
            // Shapes pinned to the kernel's contract:
            //   y, z, out : [Hv, Dv] (Dv % 4 == 0)
            //   w         : [Dv]
            //   epsBuf    : [1] f32
            let hv = 2
            let dv = 8
            let y = Tensor.empty(shape: [hv, dv], dtype: .f32)
            let yVals: [Float] = (0 ..< (hv * dv)).map { Float($0) * 0.05 + 0.1 }
            y.copyIn(from: yVals)
            let z = Tensor.empty(shape: [hv, dv], dtype: .f32)
            z.copyIn(from: (0 ..< (hv * dv)).map { Float($0) * 0.03 - 0.2 })
            let weight = Tensor.empty(shape: [dv], dtype: .f32)
            weight.copyIn(from: (0 ..< dv).map { _ in Float(1) })
            let epsBuf = Tensor.empty(shape: [1], dtype: .f32)
            epsBuf.copyIn(from: [Float(1e-5)])
            let out = Tensor.empty(shape: [hv, dv], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.gatedMixerNorm(
                    y: y, z: z, weight: weight, epsBuf: epsBuf,
                    into: out,
                    numValueHeads: hv, valueHeadDim: dv, on: cb)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == hv * dv)
            for v in got { #expect(v.isFinite, "non-finite output: \(v)") }
        }
    }

    @Test("gatedDeltaPrepStep f32 — exercises fused prep + recurrence dispatch")
    func gatedDeltaPrepStepSmoke() {
        autoreleasepool {
            // dk and dv must be multiples of 32; hv % hk == 0.
            let b = 1
            let dk = 32
            let dv = 32
            let hv = 2
            let hk = 1
            // convOut layout: [B, 2·Hk·Dk + Hv·Dv].
            let convOutLen = 2 * hk * dk + hv * dv
            let convOut = Tensor.empty(shape: [b, convOutLen], dtype: .f32)
            convOut.copyIn(from: (0 ..< convOutLen).map { Float($0) * 0.01 })
            let aLog = Tensor.empty(shape: [hv], dtype: .f32)
            aLog.copyIn(from: (0 ..< hv).map { _ in Float(-0.5) })
            let dtBias = Tensor.empty(shape: [hv], dtype: .f32)
            dtBias.copyIn(from: (0 ..< hv).map { _ in Float(0.0) })
            // aRaw / bRaw are [B, Hv] (or [Hv] for B=1, same buffer).
            let aRaw = Tensor.empty(shape: [hv], dtype: .f32)
            aRaw.copyIn(from: (0 ..< hv).map { _ in Float(0.1) })
            let bRaw = Tensor.empty(shape: [hv], dtype: .f32)
            bRaw.copyIn(from: (0 ..< hv).map { _ in Float(0.2) })
            let qNorm = Tensor.empty(shape: [hk * dk], dtype: .f32)
            qNorm.copyIn(from: (0 ..< (hk * dk)).map { _ in Float(1) })
            let kNorm = Tensor.empty(shape: [hk * dk], dtype: .f32)
            kNorm.copyIn(from: (0 ..< (hk * dk)).map { _ in Float(1) })
            // GDN state shape per GDNStateCache: [Hv, Dv, Dk].
            let stateIn = Tensor.empty(shape: [hv, dv, dk], dtype: .f32)
            stateIn.zero()
            let stateOut = Tensor.empty(shape: [hv, dv, dk], dtype: .f32)
            stateOut.zero()
            // y for B=1 is [Hv, Dv].
            let y = Tensor.empty(shape: [hv, dv], dtype: .f32)
            y.zero()
            runAndWait { cb in
                Ops.gatedDeltaPrepStep(
                    convOut: convOut, aLog: aLog, dtBias: dtBias,
                    aRaw: aRaw, bRaw: bRaw,
                    qNormWeight: qNorm, kNormWeight: kNorm,
                    stateIn: stateIn, stateOut: stateOut, y: y,
                    batchSize: b, dk: dk, dv: dv, hv: hv, hk: hk, on: cb)
            }
            let yVals = y.toArray(as: Float.self)
            #expect(yVals.count == hv * dv)
            for v in yVals { #expect(v.isFinite, "y has non-finite: \(v)") }
            let stateVals = stateOut.toArray(as: Float.self)
            for v in stateVals { #expect(v.isFinite, "state has non-finite: \(v)") }
        }
    }

    @Test("gatedDeltaChunk f32 — multi-token recurrence sweep stays finite")
    func gatedDeltaChunkSmoke() {
        // TODO: needs production-shape correctness reference — math
        // matches `mt_gated_delta_step` over `T` tokens but the
        // CPU oracle requires the full Gated Delta Net recurrence to be
        // reimplemented in Swift. Smoke-test: assert dispatch finishes
        // and outputs are finite / correctly sized.
        autoreleasepool {
            let tSteps = 2
            let hk = 1
            let hv = 1
            let dk = 32
            let dv = 32
            let q = Tensor.empty(shape: [tSteps, hk, dk], dtype: .f32)
            q.copyIn(from: (0 ..< (tSteps * hk * dk)).map { Float($0) * 0.01 })
            let k = Tensor.empty(shape: [tSteps, hk, dk], dtype: .f32)
            k.copyIn(from: (0 ..< (tSteps * hk * dk)).map { Float($0) * 0.02 })
            let v = Tensor.empty(shape: [tSteps, hv, dv], dtype: .f32)
            v.copyIn(from: (0 ..< (tSteps * hv * dv)).map { Float($0) * 0.03 })
            let g = Tensor.empty(shape: [tSteps, hv], dtype: .f32)
            g.copyIn(from: (0 ..< (tSteps * hv)).map { _ in Float(0.9) })
            let beta = Tensor.empty(shape: [tSteps, hv], dtype: .f32)
            beta.copyIn(from: (0 ..< (tSteps * hv)).map { _ in Float(0.3) })
            let stateIn = Tensor.empty(shape: [hv, dv, dk], dtype: .f32)
            stateIn.zero()
            let stateOut = Tensor.empty(shape: [hv, dv, dk], dtype: .f32)
            stateOut.zero()
            let y = Tensor.empty(shape: [tSteps, hv, dv], dtype: .f32)
            y.zero()
            let tLen = Tensor.empty(shape: [1], dtype: .u32)
            tLen.copyIn(from: [UInt32(tSteps)])
            runAndWait { cb in
                Ops.gatedDeltaChunk(
                    q: q, k: k, v: v, g: g, beta: beta,
                    stateIn: stateIn, into: y, stateOut: stateOut,
                    tLen: tLen,
                    numKeyHeads: hk, numValueHeads: hv,
                    keyHeadDim: dk, valueHeadDim: dv, on: cb)
            }
            let yVals = y.toArray(as: Float.self)
            #expect(yVals.count == tSteps * hv * dv)
            for value in yVals { #expect(value.isFinite, "y non-finite: \(value)") }
            let stateVals = stateOut.toArray(as: Float.self)
            for value in stateVals { #expect(value.isFinite, "state non-finite: \(value)") }
        }
    }

    // MARK: - SDPA prefill MMA

    @Test("sdpaPrefillMma f32 — dispatch over 32-aligned qLen produces finite out")
    func sdpaPrefillMmaSmoke() {
        // TODO: needs production-shape correctness reference — a CPU
        // softmax-attention oracle exists for sdpaDecode; we could
        // extend it to T queries but at the small shapes the metaltile
        // GPU correctness test already validates against. Smoke-test:
        // dispatch a 32-aligned qLen and assert the output is finite.
        autoreleasepool {
            let nQHeads = 4
            let nKVHeads = 2
            let headDim = 64
            let qLen = 32
            let kLen = 32
            let q = Tensor.empty(shape: [nQHeads, qLen, headDim], dtype: .f32)
            q.copyIn(
                from: (0 ..< (nQHeads * qLen * headDim))
                    .map { Float($0 % 11) * 0.01 })
            let k = Tensor.empty(shape: [nKVHeads, kLen, headDim], dtype: .f32)
            k.copyIn(
                from: (0 ..< (nKVHeads * kLen * headDim))
                    .map { Float($0 % 13) * 0.01 })
            let v = Tensor.empty(shape: [nKVHeads, kLen, headDim], dtype: .f32)
            v.copyIn(
                from: (0 ..< (nKVHeads * kLen * headDim))
                    .map { Float($0 % 17) * 0.01 })
            let scale = 1.0 / Float(headDim).squareRoot()
            var out: Tensor!
            runAndWait { cb in
                out = Ops.sdpaPrefillMma(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                    qLen: qLen, kLen: kLen, scale: scale, on: cb)
            }
            #expect(out.shape == [nQHeads, qLen, headDim])
            let got = out.toArray(as: Float.self)
            #expect(got.count == nQHeads * qLen * headDim)
            for v in got { #expect(v.isFinite, "non-finite: \(v)") }
        }
    }

    // MARK: - MoE batched-prefill variants

    @Test("moeGatherDequantGemmInt4Bm8 f32 — dispatch on BM=8 tile shape")
    func moeBm8Smoke() {
        // TODO: needs production-shape correctness reference — the
        // kernel's BM=8 tile is exercised at decode top-K; the BM=16
        // canonical path (moeGatherDequantGemmInt4) has a full
        // correctness test in MoEBgemmBm64MppTests.swift. Smoke-test
        // asserts the dispatch produces correctly-sized finite output.
        autoreleasepool {
            let nExperts = 2
            let mTotal = 8
            let nOut = 32
            let kIn = 32
            let groupSize = 32
            // weight packed: [nExperts, nOut, kIn/8] u32; one expert per
            // row is selected via `indices` (CSR offsets aren't used by
            // the BM=8 dispatcher — see Ops.moeGatherDequantGemmInt4Bm8).
            let packs = kIn / 8
            let weight = Tensor.empty(shape: [nExperts, nOut, packs], dtype: .u32)
            weight.zero()
            let groups = kIn / groupSize
            let scales = Tensor.empty(shape: [nExperts, nOut, groups], dtype: .f32)
            scales.copyIn(
                from: (0 ..< (nExperts * nOut * groups))
                    .map { Float($0) * 0.01 + 0.1 })
            let biases = Tensor.empty(shape: [nExperts, nOut, groups], dtype: .f32)
            biases.copyIn(
                from: (0 ..< (nExperts * nOut * groups))
                    .map { Float($0) * -0.005 })
            let indices = Tensor.empty(shape: [mTotal], dtype: .u32)
            indices.copyIn(from: (0 ..< mTotal).map { UInt32($0 % nExperts) })
            let input = Tensor.empty(shape: [mTotal, kIn], dtype: .f32)
            input.copyIn(from: (0 ..< (mTotal * kIn)).map { Float($0) * 0.001 })
            let out = Tensor.empty(shape: [mTotal, nOut], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.moeGatherDequantGemmInt4Bm8(
                    input: input, weight: weight, scales: scales, biases: biases,
                    indices: indices,
                    mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                    on: cb, into: out)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == mTotal * nOut)
            for v in got { #expect(v.isFinite, "non-finite: \(v)") }
        }
    }

    @Test("moeGatherDequantGemmInt4M1 f32 — scalar T=1 dispatch finishes finite")
    func moeM1Smoke() {
        // TODO: needs production-shape correctness reference — the
        // canonical `moeGatherDequantGemmInt4` test in
        // MoEBgemmBm64MppTests.swift covers the cooperative path; this
        // scalar `m1` variant has the same math but a per-element
        // simd_sum reduction. Smoke-test asserts dispatch produces
        // correctly-sized finite output.
        autoreleasepool {
            let nExperts = 2
            let tRows = 1
            let mOut = 32
            let kIn = 32
            let groupSize = 32
            let packs = kIn / 8
            let weight = Tensor.empty(shape: [nExperts, mOut, packs], dtype: .u32)
            weight.zero()
            let groups = kIn / groupSize
            let scales = Tensor.empty(shape: [nExperts, mOut, groups], dtype: .f32)
            scales.copyIn(
                from: (0 ..< (nExperts * mOut * groups))
                    .map { Float($0) * 0.01 + 0.1 })
            let biases = Tensor.empty(shape: [nExperts, mOut, groups], dtype: .f32)
            biases.copyIn(
                from: (0 ..< (nExperts * mOut * groups))
                    .map { Float($0) * -0.005 })
            // CSR expertOffsets: tRows rows total mapped expert 0..0
            // (all rows route to expert 0). Layout: [n_experts + 1].
            let expertOffsets = Tensor.empty(shape: [nExperts + 1], dtype: .u32)
            expertOffsets.copyIn(from: [UInt32(0), UInt32(tRows), UInt32(tRows)])
            let x = Tensor.empty(shape: [tRows, kIn], dtype: .f32)
            x.copyIn(from: (0 ..< (tRows * kIn)).map { Float($0) * 0.01 })
            let out = Tensor.empty(shape: [tRows, mOut], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.moeGatherDequantGemmInt4M1(
                    x, weight, scales, biases, expertOffsets,
                    tRows, mOut, kIn, nExperts, groupSize,
                    cb, out)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == tRows * mOut)
            for v in got { #expect(v.isFinite, "non-finite: \(v)") }
        }
    }

    @Test("moeUnpermute f32 — weighted scatter-sum produces correct shape")
    func moeUnpermuteSmoke() {
        // TODO: needs production-shape correctness reference — the math
        // is a per-token gather + weighted sum across top-K expert
        // outputs, validated in production MoE forward tests. Smoke
        // here asserts dispatch produces correctly-sized output that
        // matches a hand-computed two-token case.
        autoreleasepool {
            let nRows = 2
            let hidden = 4
            let k = 2
            // expertOutputs: [nRows·k, hidden] — easy values per slot.
            let expertOutputs = Tensor.empty(shape: [nRows * k, hidden], dtype: .f32)
            expertOutputs.copyIn(from: [
                Float(1), 1, 1, 1,  // row 0 slot 0 → at pos 0
                Float(2), 2, 2, 2,  // row 0 slot 1 → at pos 1
                Float(3), 3, 3, 3,  // row 1 slot 0 → at pos 2
                Float(4), 4, 4, 4,  // row 1 slot 1 → at pos 3
            ])
            // Identity permutation: slot (row, k) lives at position
            // row*k + k.
            let invPerm = Tensor.empty(shape: [nRows, k], dtype: .u32)
            invPerm.copyIn(from: [UInt32(0), 1, 2, 3])
            // Equal-weight combine.
            let weights = Tensor.empty(shape: [nRows, k], dtype: .f32)
            weights.copyIn(from: [Float(0.5), 0.5, 0.5, 0.5])
            let out = Tensor.empty(shape: [nRows, hidden], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.moeUnpermute(
                    expertOutputs: expertOutputs,
                    invPerm: invPerm, topKWeights: weights,
                    into: out,
                    nRows: nRows, hidden: hidden, k: k, on: cb)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == nRows * hidden)
            // Expected: row 0 = 0.5*1 + 0.5*2 = 1.5; row 1 = 0.5*3 + 0.5*4 = 3.5
            for c in 0 ..< hidden {
                #expect(abs(got[c] - 1.5) < 1e-4, "row0 col\(c): \(got[c])")
                #expect(abs(got[hidden + c] - 3.5) < 1e-4, "row1 col\(c): \(got[hidden + c])")
            }
        }
    }

    // MARK: - Dynamic-M dequant GEMM

    @Test("dequantGemmDynamicM f32 — T=32 aligned fast path runs finite")
    func dequantGemmDynamicMSmoke() {
        // TODO: needs production-shape correctness reference — the
        // canonical 4-bit dequant + matmul oracle is exercised at the
        // GEMV scale in QuantizedOpsTests.swift; the dynamic-M kernel
        // shares the same dequant math but tiles across M. Smoke-test
        // dispatches the 32-aligned fast path and asserts output is
        // correctly sized and finite.
        autoreleasepool {
            let t = 32
            let nOut = 32
            let kIn = 32
            let groupSize = 32
            let packs = kIn / 8
            let weight = Tensor.empty(shape: [nOut, packs], dtype: .u32)
            // Non-zero quantized payload so the dequant path produces
            // varying outputs.
            weight.copyIn(from: (0 ..< (nOut * packs)).map { UInt32($0 + 1) })
            let groups = kIn / groupSize
            let scales = Tensor.empty(shape: [nOut, groups], dtype: .f32)
            scales.copyIn(
                from: (0 ..< (nOut * groups))
                    .map { Float($0) * 0.01 + 0.05 })
            let biases = Tensor.empty(shape: [nOut, groups], dtype: .f32)
            biases.copyIn(
                from: (0 ..< (nOut * groups))
                    .map { Float($0) * -0.005 })
            let input = Tensor.empty(shape: [t, kIn], dtype: .f32)
            input.copyIn(from: (0 ..< (t * kIn)).map { Float($0) * 0.001 })
            let out = Tensor.empty(shape: [t, nOut], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.dequantGemmDynamicM(
                    input: input,
                    weight: weight, scales: scales, biases: biases,
                    t: t, nOut: nOut, kIn: kIn, groupSize: groupSize,
                    on: cb, device: .shared, into: out)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == t * nOut)
            for v in got { #expect(v.isFinite, "non-finite: \(v)") }
        }
    }
}
