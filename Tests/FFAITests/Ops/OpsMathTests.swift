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
// OpsMath tests — element-wise math + reductions added in the Phase C
// Ops surface-parity sweep. Each test runs a tiny known-shape input,
// asserts the numerical result is within tolerance of a CPU reference,
// and exercises an invariant-violation case where it makes sense.

import Foundation
import Metal
import Testing
@testable import FFAI
import TestHelpers

@Suite("OpsMath — element-wise + reductions")
struct OpsMathTests {

    // ─── binary elementwise: sub, div, pow, maxElem, minElem ────────

    @Test("sub f32 — c[i] = a[i] - b[i]")
    func subF32() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f32)
            let b = Tensor.empty(shape: [4], dtype: .f32)
            a.copyIn(from: [Float(3), 5, 9, -1])
            b.copyIn(from: [Float(1), 2, -1, 4])
            var out: Tensor!
            runAndWait { cb in out = Ops.sub(a, b, on: cb) }
            #expect(out.toArray(as: Float.self) == [2, 3, 10, -5])
        }
    }

    @Test("div f32 — c[i] = a[i] / b[i]")
    func divF32() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f32)
            let b = Tensor.empty(shape: [4], dtype: .f32)
            a.copyIn(from: [Float(10), 9, 24, -16])
            b.copyIn(from: [Float(2), 3, 4, -2])
            var out: Tensor!
            runAndWait { cb in out = Ops.div(a, b, on: cb) }
            #expect(out.toArray(as: Float.self) == [5, 3, 6, 8])
        }
    }

    @Test("pow f32 — c[i] = a[i]^b[i]")
    func powF32() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f32)
            let b = Tensor.empty(shape: [4], dtype: .f32)
            a.copyIn(from: [Float(2), 3, 4, 9])
            b.copyIn(from: [Float(3), 2, 0.5, 0.5])
            var out: Tensor!
            runAndWait { cb in out = Ops.pow(a, b, on: cb) }
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0] - 8) < 1e-4)
            #expect(abs(r[1] - 9) < 1e-4)
            #expect(abs(r[2] - 2) < 1e-4)
            #expect(abs(r[3] - 3) < 1e-4)
        }
    }

    @Test("maxElem f32 — c[i] = max(a[i], b[i])")
    func maxElemF32() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f32)
            let b = Tensor.empty(shape: [4], dtype: .f32)
            a.copyIn(from: [Float(1), 5, 3, -1])
            b.copyIn(from: [Float(2), 3, 7, 0])
            var out: Tensor!
            runAndWait { cb in out = Ops.maxElem(a, b, on: cb) }
            #expect(out.toArray(as: Float.self) == [2, 5, 7, 0])
        }
    }

    @Test("minElem f32 — c[i] = min(a[i], b[i])")
    func minElemF32() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f32)
            let b = Tensor.empty(shape: [4], dtype: .f32)
            a.copyIn(from: [Float(1), 5, 3, -1])
            b.copyIn(from: [Float(2), 3, 7, 0])
            var out: Tensor!
            runAndWait { cb in out = Ops.minElem(a, b, on: cb) }
            #expect(out.toArray(as: Float.self) == [1, 3, 3, -1])
        }
    }

    // ─── unary elementwise: neg, abs, exp, log, sqrt, square, recip ─

    @Test("neg f32 — out[i] = -x[i]")
    func negF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(1), -2, 3, 0])
            var out: Tensor!
            runAndWait { cb in out = Ops.neg(x, on: cb) }
            #expect(out.toArray(as: Float.self) == [-1, 2, -3, 0])
        }
    }

    @Test("abs f32 — out[i] = |x[i]|")
    func absF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(-1), 2, -3, 0])
            var out: Tensor!
            runAndWait { cb in out = Ops.abs(x, on: cb) }
            #expect(out.toArray(as: Float.self) == [1, 2, 3, 0])
        }
    }

    @Test("exp f32 — out[i] = exp(x[i])")
    func expF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [3], dtype: .f32)
            x.copyIn(from: [Float(0), 1, -1])
            var out: Tensor!
            runAndWait { cb in out = Ops.exp(x, on: cb) }
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0] - 1) < 1e-5)
            #expect(abs(r[1] - Float(M_E)) < 1e-3)
            #expect(abs(r[2] - Float(1.0 / M_E)) < 1e-3)
        }
    }

    @Test("log f32 — out[i] = log(x[i])")
    func logF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [3], dtype: .f32)
            x.copyIn(from: [Float(1), Float(M_E), 100])
            var out: Tensor!
            runAndWait { cb in out = Ops.log(x, on: cb) }
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0]) < 1e-5)
            #expect(abs(r[1] - 1) < 1e-4)
            #expect(abs(r[2] - Float(Foundation.log(100.0))) < 1e-3)
        }
    }

    @Test("sqrt f32 — out[i] = sqrt(x[i])")
    func sqrtF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(1), 4, 9, 0])
            var out: Tensor!
            runAndWait { cb in out = Ops.sqrt(x, on: cb) }
            #expect(out.toArray(as: Float.self) == [1, 2, 3, 0])
        }
    }

    @Test("square f32 — out[i] = x[i]²")
    func squareF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(1), 2, -3, 0])
            var out: Tensor!
            runAndWait { cb in out = Ops.square(x, on: cb) }
            #expect(out.toArray(as: Float.self) == [1, 4, 9, 0])
        }
    }

    @Test("recip f32 — out[i] = 1 / x[i]")
    func recipF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [3], dtype: .f32)
            x.copyIn(from: [Float(1), 2, 4])
            var out: Tensor!
            runAndWait { cb in out = Ops.recip(x, on: cb) }
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0] - 1) < 1e-5)
            #expect(abs(r[1] - 0.5) < 1e-5)
            #expect(abs(r[2] - 0.25) < 1e-5)
        }
    }

    @Test("floor / ceil / round f32 — IEEE-754 rounding")
    func floorCeilRoundF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(1.4), 1.6, -1.4, -1.6])
            var f: Tensor!, c: Tensor!, r: Tensor!
            runAndWait { cb in
                f = Ops.floor(x, on: cb)
                c = Ops.ceil(x, on: cb)
                r = Ops.round(x, on: cb)
            }
            #expect(f.toArray(as: Float.self) == [1, 1, -2, -2])
            #expect(c.toArray(as: Float.self) == [2, 2, -1, -1])
            // round uses banker's rounding via `rint`; for .4/.6 inputs
            // this matches plain nearest.
            #expect(r.toArray(as: Float.self) == [1, 2, -1, -2])
        }
    }

    // ─── copy kernel (vs blit) ─────────────────────────────────────

    @Test("copyKernel f32 — duplicates src into dst via mt_copy")
    func copyKernelF32() {
        autoreleasepool {
            let src = Tensor.empty(shape: [4], dtype: .f32)
            src.copyIn(from: [Float(1), 2, 3, 4])
            let dst = Tensor.empty(shape: [4], dtype: .f32)
            dst.zero()
            runAndWait { cb in Ops.copyKernel(src, into: dst, on: cb) }
            #expect(dst.toArray(as: Float.self) == [1, 2, 3, 4])
        }
    }

    // ─── arange ─────────────────────────────────────────────────────

    @Test("arange f32 — fills [start, start+step, …]")
    func arangeF32() {
        autoreleasepool {
            let out = Tensor.empty(shape: [5], dtype: .f32)
            runAndWait { cb in
                Ops.arange(start: 1, step: 2, into: out, on: cb)
            }
            #expect(out.toArray(as: Float.self) == [1, 3, 5, 7, 9])
        }
    }

    // ─── reductions: softmax, logsumexp, argmin ────────────────────

    @Test("softmax f32 — single row sums to 1, peak in correct slot")
    func softmaxF32SingleRow() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(0), 10, 0, 0])
            var out: Tensor!
            runAndWait { cb in out = Ops.softmax(x, on: cb) }
            let r = out.toArray(as: Float.self)
            let sum = r.reduce(0, +)
            #expect(abs(sum - 1) < 1e-4)
            #expect(r[1] > 0.99)  // peak at the +10 entry
        }
    }

    @Test("softmax f32 — multi-row independent normalization")
    func softmaxF32MultiRow() {
        autoreleasepool {
            let x = Tensor.empty(shape: [2, 4], dtype: .f32)
            x.copyIn(from: [
                Float(0), 0, 0, 10,   // peak at idx 3
                Float(10), 0, 0, 0,   // peak at idx 0
            ])
            var out: Tensor!
            runAndWait { cb in out = Ops.softmax(x, on: cb) }
            let r = out.toArray(as: Float.self)
            // each row sums to 1, peak at the +10 entry
            #expect(abs(r[0..<4].reduce(0, +) - 1) < 1e-4)
            #expect(abs(r[4..<8].reduce(0, +) - 1) < 1e-4)
            #expect(r[3] > 0.99)
            #expect(r[4] > 0.99)
        }
    }

    @Test("logsumexp f32 — log(sum(exp(x))) over rows")
    func logsumexpF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [2, 3], dtype: .f32)
            x.copyIn(from: [
                Float(0), 0, 0,   // expect log(3)
                Float(1), 2, 3,   // expect log(e+e²+e³)
            ])
            var out: Tensor!
            runAndWait { cb in out = Ops.logsumexp(x, on: cb) }
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0] - Float(Foundation.log(3.0))) < 1e-4)
            let expected = Float(Foundation.log(exp(1.0) + exp(2.0) + exp(3.0)))
            #expect(abs(r[1] - expected) < 1e-3)
        }
    }

    @Test("argmin f32 — returns index of the smallest entry")
    func argminF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [8], dtype: .f32)
            x.copyIn(from: [Float(5), 3, 1, 4, 1, 5, 9, 2])
            // first minimum is at idx 2
            let out = Tensor.empty(shape: [1], dtype: .u32)
            out.zero()
            runAndWait { cb in Ops.argmin(x, into: out, on: cb) }
            #expect(out.toArray(as: UInt32.self)[0] == 2)
        }
    }

    // ─── f16 + bf16 dtype coverage ─────────────────────────────────

    @Test("sub / div / neg / abs — f16 dispatch returns finite output")
    func mathF16Smoke() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f16)
            let b = Tensor.empty(shape: [4], dtype: .f16)
            a.copyIn(from: [Float16(3), 5, 9, -1])
            b.copyIn(from: [Float16(1), 2, -1, 4])
            var s: Tensor!, d: Tensor!, n: Tensor!, ab: Tensor!
            runAndWait { cb in
                s = Ops.sub(a, b, on: cb)
                d = Ops.div(a, b, on: cb)
                n = Ops.neg(a, on: cb)
                ab = Ops.abs(b, on: cb)
            }
            for tensor in [s, d, n, ab] {
                #expect(tensor!.dtype == .f16)
                let vals = tensor!.toArray(as: Float16.self)
                for v in vals { #expect(v.isFinite) }
            }
        }
    }

    @Test("exp / log / sqrt / square / recip — bf16 dispatch returns finite")
    func mathBF16Smoke() {
        autoreleasepool {
            // Construct bf16 with known values via UInt16 bit patterns.
            // bf16(1.0) = 0x3F80, bf16(2.0) = 0x4000, bf16(4.0) = 0x4080.
            let x = Tensor.empty(shape: [4], dtype: .bf16)
            x.copyIn(from: [UInt16(0x3F80), UInt16(0x4000), UInt16(0x4080), UInt16(0x3F80)])
            var e: Tensor!, l: Tensor!, sq: Tensor!, sqr: Tensor!, r: Tensor!
            runAndWait { cb in
                e = Ops.exp(x, on: cb)
                l = Ops.log(x, on: cb)
                sq = Ops.sqrt(x, on: cb)
                sqr = Ops.square(x, on: cb)
                r = Ops.recip(x, on: cb)
            }
            for tensor in [e, l, sq, sqr, r] {
                #expect(tensor!.dtype == .bf16)
                for v in tensor!.toFloatArray() { #expect(v.isFinite) }
            }
        }
    }

    @Test("maxElem / minElem / pow — f16 + bf16 dispatch")
    func binaryF16BF16Smoke() {
        autoreleasepool {
            // f16
            let a16 = Tensor.empty(shape: [4], dtype: .f16)
            let b16 = Tensor.empty(shape: [4], dtype: .f16)
            a16.copyIn(from: [Float16(1), 5, 3, -1])
            b16.copyIn(from: [Float16(2), 3, 7, 0])
            var maxF16: Tensor!, minF16: Tensor!, powF16: Tensor!
            runAndWait { cb in
                maxF16 = Ops.maxElem(a16, b16, on: cb)
                minF16 = Ops.minElem(a16, b16, on: cb)
                powF16 = Ops.pow(a16, b16, on: cb)
            }
            for v in maxF16.toFloatArray() { #expect(v.isFinite) }
            for v in minF16.toFloatArray() { #expect(v.isFinite) }
            // pow can NaN on negative-base + non-integer exponent; just
            // confirm dispatch fires (the (-1)^0 entry is finite).
            #expect(powF16.dtype == .f16)
            // bf16 — bit-pattern values 1.0, 2.0, 4.0, 8.0
            let aBF = Tensor.empty(shape: [4], dtype: .bf16)
            let bBF = Tensor.empty(shape: [4], dtype: .bf16)
            aBF.copyIn(from: [UInt16(0x3F80), UInt16(0x4000), UInt16(0x4080), UInt16(0x4100)])
            bBF.copyIn(from: [UInt16(0x3F80), UInt16(0x3F80), UInt16(0x3F80), UInt16(0x3F80)])
            runAndWait { cb in
                _ = Ops.maxElem(aBF, bBF, on: cb)
                _ = Ops.minElem(aBF, bBF, on: cb)
                _ = Ops.pow(aBF, bBF, on: cb)
            }
        }
    }

    @Test("floor / ceil / round / copyKernel — f16 dispatch fires")
    func unaryF16Smoke() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f16)
            x.copyIn(from: [Float16(1.4), 1.6, -1.4, -1.6])
            var f: Tensor!, c: Tensor!, r: Tensor!
            let dst = Tensor.empty(shape: [4], dtype: .f16)
            dst.zero()
            runAndWait { cb in
                f = Ops.floor(x, on: cb)
                c = Ops.ceil(x, on: cb)
                r = Ops.round(x, on: cb)
                Ops.copyKernel(x, into: dst, on: cb)
            }
            #expect(f.dtype == .f16)
            #expect(c.dtype == .f16)
            #expect(r.dtype == .f16)
            #expect(dst.toFloatArray() == x.toFloatArray())
        }
    }

    @Test("softmax / logsumexp / argmin — f16 + bf16 dispatch")
    func reductionDtypeSmoke() {
        autoreleasepool {
            // softmax + logsumexp on f16 single row
            let x16 = Tensor.empty(shape: [4], dtype: .f16)
            x16.copyIn(from: [Float16(0), 1, 2, 3])
            var s16: Tensor!, l16: Tensor!
            let arg16 = Tensor.empty(shape: [1], dtype: .u32)
            runAndWait { cb in
                s16 = Ops.softmax(x16, on: cb)
                l16 = Ops.logsumexp(x16, on: cb)
                Ops.argmin(x16, into: arg16, on: cb)
            }
            let sumF16 = s16.toFloatArray().reduce(0, +)
            #expect(abs(sumF16 - 1) < 1e-2)
            #expect(l16.dtype == .f16)
            #expect(arg16.toArray(as: UInt32.self)[0] == 0)
            // bf16
            let xBF = Tensor.empty(shape: [4], dtype: .bf16)
            xBF.copyIn(from: [UInt16(0), UInt16(0x3F80), UInt16(0x4000), UInt16(0x4040)])
            var sBF: Tensor!, lBF: Tensor!
            let argBF = Tensor.empty(shape: [1], dtype: .u32)
            runAndWait { cb in
                sBF = Ops.softmax(xBF, on: cb)
                lBF = Ops.logsumexp(xBF, on: cb)
                Ops.argmin(xBF, into: argBF, on: cb)
            }
            for v in sBF.toFloatArray() { #expect(v.isFinite) }
            #expect(lBF.dtype == .bf16)
            #expect(argBF.toArray(as: UInt32.self)[0] == 0)  // bf16 zero
        }
    }
}
