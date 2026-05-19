// Cover the f16 and bf16 dtype branches of every Ops method.
// (OpsTests.swift covers f32; this file mirrors the same checks for the
// half-precision dtypes so we exercise every kernel variant.)

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops dtypes (f16 + bf16)", .serialized)
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
            let x = Tensor.empty(shape: [4], dtype: .f16)
            x.copyIn(from: [Float16(1), 2, 3, 4])
            let w = Tensor.empty(shape: [4], dtype: .f16)
            w.copyIn(from: [Float16(1), 1, 1, 1])
            var out: Tensor!
            runAndWait { cb in out = Ops.rmsNorm(x, weight: w, eps: 1e-6, on: cb) }
            let r = out.toArray(as: Float16.self).map { Float($0) }
            let expectedRms = Float((30.0 / 4.0).squareRoot())
            for i in 0..<4 {
                let expected = Float(i + 1) / expectedRms
                #expect(abs(r[i] - expected) < 1e-2)
            }
        }
    }

    @Test("rmsNorm bf16")
    func rmsNormBf16() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .bf16)
            x.copyIn(from: [UInt16(0x3F80), 0x4000, 0x4040, 0x4080])  // 1..4
            let w = Tensor.empty(shape: [4], dtype: .bf16)
            w.copyIn(from: [UInt16(0x3F80), 0x3F80, 0x3F80, 0x3F80])  // 1, 1, 1, 1
            var out: Tensor!
            runAndWait { cb in out = Ops.rmsNorm(x, weight: w, eps: 1e-6, on: cb) }
            let r = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            let expectedRms = Float((30.0 / 4.0).squareRoot())
            for i in 0..<4 {
                let expected = Float(i + 1) / expectedRms
                #expect(abs(r[i] - expected) < 5e-2)  // bf16 is coarse
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
            for i in 0..<4 {
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
            for i in 0..<4 {
                #expect(abs(r[i] - Float(i + 1)) < 1e-2)
            }
        }
    }

    // ─── sdpaDecode ──────────────────────────────────────────────────

    @Test("sdpaDecode f16 — single position")
    func sdpaF16() {
        autoreleasepool {
            let q = Tensor.empty(shape: [1, 4], dtype: .f16)
            let k = Tensor.empty(shape: [1, 4, 4], dtype: .f16)
            let v = Tensor.empty(shape: [1, 4, 4], dtype: .f16)
            q.copyIn(from: [Float16(1), 0, 0, 0])
            var kData = [Float16](repeating: 0, count: 16)
            kData[0] = 1
            k.copyIn(from: kData)
            var vData = [Float16](repeating: 0, count: 16)
            vData[0] = 7; vData[1] = 8; vData[2] = 9; vData[3] = 10
            v.copyIn(from: vData)
            var out: Tensor!
            runAndWait { cb in
                out = Ops.sdpaDecode(q: q, k: k, v: v,
                                     nQHeads: 1, nKVHeads: 1, headDim: 4,
                                     nKV: 1, kvStride: 4, scale: 1.0, on: cb)
            }
            let r = out.toArray(as: Float16.self).map { Float($0) }
            #expect(r == [7, 8, 9, 10])
        }
    }

    @Test("sdpaDecode bf16 — single position")
    func sdpaBf16() {
        autoreleasepool {
            let q = Tensor.empty(shape: [1, 4], dtype: .bf16)
            let k = Tensor.empty(shape: [1, 4, 4], dtype: .bf16)
            let v = Tensor.empty(shape: [1, 4, 4], dtype: .bf16)
            var qBits = [UInt16](repeating: 0, count: 4); qBits[0] = 0x3F80   // 1
            q.copyIn(from: qBits)
            var kBits = [UInt16](repeating: 0, count: 16); kBits[0] = 0x3F80
            k.copyIn(from: kBits)
            var vBits = [UInt16](repeating: 0, count: 16)
            vBits[0] = 0x40E0  // 7
            vBits[1] = 0x4100  // 8
            vBits[2] = 0x4110  // 9
            vBits[3] = 0x4120  // 10
            v.copyIn(from: vBits)
            var out: Tensor!
            runAndWait { cb in
                out = Ops.sdpaDecode(q: q, k: k, v: v,
                                     nQHeads: 1, nKVHeads: 1, headDim: 4,
                                     nKV: 1, kvStride: 4, scale: 1.0, on: cb)
            }
            let r = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            #expect(r == [7, 8, 9, 10])
        }
    }
}
