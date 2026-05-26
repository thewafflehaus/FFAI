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
// Cover the f16 and bf16 dtype branches of every Ops method.
// (OpsTests.swift covers f32; this file mirrors the same checks for the
// half-precision dtypes so we exercise every kernel variant.)

import Foundation
import Metal
import TestHelpers
import Testing

@testable import FFAI

@Suite("Ops dtypes (f16 + bf16)")
struct OpsDtypeTests {
    // ─── add ─────────────────────────────────────────────────────────

    @Test("add f16")
    func addF16() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f16)
            let b = Tensor.empty(shape: [4], dtype: .f16)
            a.copyIn(from: [Float16(1), 2, 3, 4])
            b.copyIn(from: [Float16(10), 20, 30, 40])
            var out: Tensor!
            runAndWait { cb in out = Ops.add(a, b, on: cb) }
            let r = out.toArray(as: Float16.self).map { Float($0) }
            #expect(r == [11, 22, 33, 44])
        }
    }

    @Test("add bf16")
    func addBf16() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .bf16)
            let b = Tensor.empty(shape: [4], dtype: .bf16)
            // bf16 = top 16 bits of f32. 1, 2, 3, 4
            a.copyIn(from: [UInt16(0x3F80), 0x4000, 0x4040, 0x4080])
            // 10, 20, 30, 40
            b.copyIn(from: [UInt16(0x4120), 0x41A0, 0x41F0, 0x4220])
            var out: Tensor!
            runAndWait { cb in out = Ops.add(a, b, on: cb) }
            let bits = out.toArray(as: UInt16.self)
            let r = bits.map { Float(bitPattern: UInt32($0) << 16) }
            #expect(r == [11, 22, 33, 44])
        }
    }

    // ─── mul ─────────────────────────────────────────────────────────

    @Test("mul f16")
    func mulF16() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f16)
            let b = Tensor.empty(shape: [4], dtype: .f16)
            a.copyIn(from: [Float16(1), 2, 3, 4])
            b.copyIn(from: [Float16(2), 2, 2, 2])
            var out: Tensor!
            runAndWait { cb in out = Ops.mul(a, b, on: cb) }
            #expect(out.toArray(as: Float16.self).map { Float($0) } == [2, 4, 6, 8])
        }
    }

    @Test("mul bf16")
    func mulBf16() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .bf16)
            let b = Tensor.empty(shape: [4], dtype: .bf16)
            a.copyIn(from: [UInt16(0x3F80), 0x4000, 0x4040, 0x4080])
            b.copyIn(from: [UInt16(0x4000), 0x4000, 0x4000, 0x4000])
            var out: Tensor!
            runAndWait { cb in out = Ops.mul(a, b, on: cb) }
            let r = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            #expect(r == [2, 4, 6, 8])
        }
    }

    // ─── silu ────────────────────────────────────────────────────────

    @Test("silu f16 — out[i] = x / (1 + exp(-x))")
    func siluF16() {
        autoreleasepool {
            let x = Tensor.empty(shape: [3], dtype: .f16)
            x.copyIn(from: [Float16(0), 1, -1])
            var out: Tensor!
            runAndWait { cb in out = Ops.silu(x, on: cb) }
            let r = out.toArray(as: Float16.self).map { Float($0) }
            #expect(abs(r[0]) < 1e-2)
            #expect(abs(r[1] - Float(1.0 / (1.0 + exp(-1.0)))) < 5e-3)
            #expect(abs(r[2] - Float(-1.0 / (1.0 + exp(1.0)))) < 5e-3)
        }
    }

    @Test("silu bf16")
    func siluBf16() {
        autoreleasepool {
            let x = Tensor.empty(shape: [2], dtype: .bf16)
            // 0, 1
            x.copyIn(from: [UInt16(0x0000), 0x3F80])
            var out: Tensor!
            runAndWait { cb in out = Ops.silu(x, on: cb) }
            let r = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            #expect(abs(r[0]) < 1e-2)
            #expect(abs(r[1] - Float(1.0 / (1.0 + exp(-1.0)))) < 5e-2)  // bf16 is coarse
        }
    }

    // ─── gather ──────────────────────────────────────────────────────

    @Test("gather f16")
    func gatherF16() {
        autoreleasepool {
            let table = Tensor.empty(shape: [3, 2], dtype: .f16)
            table.copyIn(from: [Float16(10), 11, 20, 21, 30, 31])
            let ids = Tensor.empty(shape: [2], dtype: .u32)
            ids.copyIn(from: [UInt32(2), 0])
            var out: Tensor!
            runAndWait { cb in out = Ops.gather(table: table, tokenIds: ids, on: cb) }
            #expect(out.toArray(as: Float16.self).map { Float($0) } == [30, 31, 10, 11])
        }
    }

    @Test("gather bf16")
    func gatherBf16() {
        autoreleasepool {
            let table = Tensor.empty(shape: [3, 2], dtype: .bf16)
            // Rows [10, 11], [20, 21], [30, 31]
            table.copyIn(from: [UInt16(0x4120), 0x4130, 0x41A0, 0x41A8, 0x41F0, 0x41F8])
            let ids = Tensor.empty(shape: [1], dtype: .u32)
            ids.copyIn(from: [UInt32(1)])
            var out: Tensor!
            runAndWait { cb in out = Ops.gather(table: table, tokenIds: ids, on: cb) }
            let r = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            #expect(r == [20, 21])
        }
    }

    // ─── gemv ────────────────────────────────────────────────────────

    @Test("gemv f16")
    func gemvF16() {
        autoreleasepool {
            let w = Tensor.empty(shape: [3, 2], dtype: .f16)
            w.copyIn(from: [Float16(1), 2, 3, 4, 5, 6])
            let x = Tensor.empty(shape: [2], dtype: .f16)
            x.copyIn(from: [Float16(7), 8])
            var out: Tensor!
            runAndWait { cb in out = Ops.gemv(weight: w, input: x, on: cb) }
            let r = out.toArray(as: Float16.self).map { Float($0) }
            #expect(r == [23, 53, 83])
        }
    }

    @Test("gemv bf16")
    func gemvBf16() {
        autoreleasepool {
            let w = Tensor.empty(shape: [3, 2], dtype: .bf16)
            w.copyIn(from: [UInt16(0x3F80), 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0])  // 1..6
            let x = Tensor.empty(shape: [2], dtype: .bf16)
            x.copyIn(from: [UInt16(0x40E0), 0x4100])  // 7, 8
            var out: Tensor!
            runAndWait { cb in out = Ops.gemv(weight: w, input: x, on: cb) }
            let r = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            // bf16 of 23, 53, 83 — coarse precision so allow a bit of slop
            #expect(abs(r[0] - 23) < 1)
            #expect(abs(r[1] - 53) < 1)
            #expect(abs(r[2] - 83) < 1)
        }
    }

    // ─── rmsNorm ─────────────────────────────────────────────────────

    @Test("rmsNorm f16")
    func rmsNormF16() {
        autoreleasepool {
            // Kernel invariant: n must be a multiple of 128 (32-lane simdgroup ×
            // 4 elements/thread). Mirrors the f32 fix in OpsTests.rmsNormF32.
            let n = 128
            let xs: [Float16] = (0 ..< n).map { Float16(Float($0 + 1)) }
            let ws: [Float16] = Array(repeating: Float16(1), count: n)
            let x = Tensor.empty(shape: [n], dtype: .f16)
            x.copyIn(from: xs)
            let w = Tensor.empty(shape: [n], dtype: .f16)
            w.copyIn(from: ws)
            var out: Tensor!
            runAndWait { cb in out = Ops.rmsNorm(x, weight: w, eps: 1e-6, on: cb) }
            let r = out.toArray(as: Float16.self).map { Float($0) }
            // CPU reference in fp32 (xs is bounded enough to round-trip clean).
            let xsF: [Float] = xs.map { Float($0) }
            let ssq = xsF.reduce(Float(0)) { $0 + $1 * $1 }
            let expectedRms = (ssq / Float(n)).squareRoot()
            // f16 has 10-bit mantissa; tolerance accounts for both the kernel's
            // fp32 → fp16 storage cast and the magnitude of values (up to 128).
            for i in 0 ..< n {
                let expected = xsF[i] / expectedRms
                #expect(
                    abs(r[i] - expected) < 0.1,
                    "i=\(i) got \(r[i]) expected \(expected)")
            }
        }
    }

    @Test("rmsNorm bf16")
    func rmsNormBf16() {
        autoreleasepool {
            // Same n=128 multiple-of-128 invariant as f16 above. bf16 has 7-bit
            // mantissa → looser tolerance still required.
            let n = 128
            // bf16 input encoded as UInt16: value = 1.0 / Float(i+1).
            // For simplicity, build f32 values then narrow via bitcast.
            func f32ToBf16(_ v: Float) -> UInt16 {
                let bits = v.bitPattern
                // Round to nearest, ties to even — bf16 = high 16 bits of fp32
                // with a round-up if the low 16 bits are > 0x8000 (or == 0x8000
                // and the low bit of the high half is 1).
                let lo = bits & 0xFFFF
                let hi = UInt16(bits >> 16)
                let roundUp =
                    lo > 0x8000 || (lo == 0x8000 && (hi & 1) == 1)
                return roundUp ? hi &+ 1 : hi
            }
            let xs: [UInt16] = (0 ..< n).map { f32ToBf16(Float($0 + 1)) }
            let ws: [UInt16] = Array(repeating: f32ToBf16(1.0), count: n)
            let x = Tensor.empty(shape: [n], dtype: .bf16)
            x.copyIn(from: xs)
            let w = Tensor.empty(shape: [n], dtype: .bf16)
            w.copyIn(from: ws)
            var out: Tensor!
            runAndWait { cb in out = Ops.rmsNorm(x, weight: w, eps: 1e-6, on: cb) }
            let r = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            let xsF: [Float] = (0 ..< n).map { Float($0 + 1) }
            let ssq = xsF.reduce(Float(0)) { $0 + $1 * $1 }
            let expectedRms = (ssq / Float(n)).squareRoot()
            // bf16 mantissa is 7 bits → ~1% relative error at unit magnitude;
            // tolerance scaled for values up to 128.
            for i in 0 ..< n {
                let expected = xsF[i] / expectedRms
                #expect(
                    abs(r[i] - expected) < 0.5,
                    "i=\(i) got \(r[i]) expected \(expected)")
            }
        }
    }

    // ─── rope ────────────────────────────────────────────────────────

    @Test("rope f16 at position 0 is identity")
    func ropeF16Identity() {
        autoreleasepool {
            let qk = Tensor.empty(shape: [1, 4], dtype: .f16)
            qk.copyIn(from: [Float16(1), 2, 3, 4])
            var out: Tensor!
            runAndWait { cb in
                out = Ops.rope(qk, position: 0, headDim: 4, thetaBase: 10000, on: cb)
            }
            let r = out.toArray(as: Float16.self).map { Float($0) }
            for i in 0 ..< 4 {
                #expect(abs(r[i] - Float(i + 1)) < 1e-2)
            }
        }
    }

    @Test("rope bf16 at position 0 is identity")
    func ropeBf16Identity() {
        autoreleasepool {
            let qk = Tensor.empty(shape: [1, 4], dtype: .bf16)
            qk.copyIn(from: [UInt16(0x3F80), 0x4000, 0x4040, 0x4080])
            var out: Tensor!
            runAndWait { cb in
                out = Ops.rope(qk, position: 0, headDim: 4, thetaBase: 10000, on: cb)
            }
            let r = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            for i in 0 ..< 4 {
                #expect(abs(r[i] - Float(i + 1)) < 1e-2)
            }
        }
    }

    // ─── sdpaDecode ──────────────────────────────────────────────────

    @Test("sdpaDecode f16 — single position")
    func sdpaF16() {
        autoreleasepool {
            // Kernel invariant: head_dim must be 128. Mirrors the f32 fix in
            // OpsTests.sdpaSinglePosition.
            let D = 128
            let kvStride = 4
            let nKV = 1
            let nQHeads = 1
            let nKVHeads = 1

            let q = Tensor.empty(shape: [nQHeads, D], dtype: .f16)
            let k = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f16)
            let v = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f16)

            var qData = [Float16](repeating: 0, count: nQHeads * D)
            qData[0] = 1
            q.copyIn(from: qData)

            var kData = [Float16](repeating: 0, count: nKVHeads * kvStride * D)
            kData[0] = 1
            k.copyIn(from: kData)

            var vData = [Float16](repeating: 0, count: nKVHeads * kvStride * D)
            for d in 0 ..< D { vData[d] = Float16(Float(d + 1)) }
            v.copyIn(from: vData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.sdpaDecode(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: D,
                    nKV: nKV, kvStride: kvStride, scale: 1.0, on: cb)
            }
            let r = out.toArray(as: Float16.self).map { Float($0) }
            // f16 round-trip on integers 1..128 is exact (well within mantissa).
            for d in 0 ..< D {
                #expect(
                    abs(r[d] - Float(d + 1)) < 0.5,
                    "out[\(d)] = \(r[d]), expected \(d + 1)")
            }
        }
    }

    @Test("sdpaDecode bf16 — single position")
    func sdpaBf16() {
        autoreleasepool {
            // Kernel invariant: head_dim must be 128. Mirrors the f32 fix.
            // bf16 has 7-bit mantissa → looser tolerance.
            let D = 128
            let kvStride = 4
            let nKV = 1
            let nQHeads = 1
            let nKVHeads = 1

            func f32ToBf16(_ v: Float) -> UInt16 {
                let bits = v.bitPattern
                let lo = bits & 0xFFFF
                let hi = UInt16(bits >> 16)
                let roundUp =
                    lo > 0x8000 || (lo == 0x8000 && (hi & 1) == 1)
                return roundUp ? hi &+ 1 : hi
            }

            let q = Tensor.empty(shape: [nQHeads, D], dtype: .bf16)
            let k = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .bf16)
            let v = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .bf16)

            var qBits = [UInt16](repeating: 0, count: nQHeads * D)
            qBits[0] = f32ToBf16(1.0)
            q.copyIn(from: qBits)

            var kBits = [UInt16](repeating: 0, count: nKVHeads * kvStride * D)
            kBits[0] = f32ToBf16(1.0)
            k.copyIn(from: kBits)

            var vBits = [UInt16](repeating: 0, count: nKVHeads * kvStride * D)
            for d in 0 ..< D { vBits[d] = f32ToBf16(Float(d + 1)) }
            v.copyIn(from: vBits)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.sdpaDecode(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: D,
                    nKV: nKV, kvStride: kvStride, scale: 1.0, on: cb)
            }
            let r = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            // bf16 round-error at magnitudes up to 128 is ~0.5 (1/256 × 128 = 0.5).
            for d in 0 ..< D {
                #expect(
                    abs(r[d] - Float(d + 1)) < 1.0,
                    "out[\(d)] = \(r[d]), expected \(d + 1)")
            }
        }
    }
}
