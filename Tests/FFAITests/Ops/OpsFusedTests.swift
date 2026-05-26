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
// OpsFused tests — fused activations + norms added in the Phase C
// Ops surface-parity sweep.

import Foundation
import Metal
import Testing
@testable import FFAI
import TestHelpers

@Suite("OpsFused — fused gate / norm wrappers")
struct OpsFusedTests {

    // ─── fused gate activations ────────────────────────────────────

    @Test("fusedGateGelu f32 — out[i] = gelu(gate[i]) * up[i]")
    func fusedGateGeluF32() {
        autoreleasepool {
            let gate = Tensor.empty(shape: [4], dtype: .f32)
            let up = Tensor.empty(shape: [4], dtype: .f32)
            gate.copyIn(from: [Float(0), 1, -1, 2])
            up.copyIn(from: [Float(3), 5, 7, 11])
            var out: Tensor!
            runAndWait { cb in out = Ops.fusedGateGelu(gate: gate, up: up, on: cb) }
            let r = out.toArray(as: Float.self)
            // gelu(0)=0 → 0*3=0
            #expect(abs(r[0]) < 1e-4)
            // gelu(1) ≈ 0.8413 → * 5 ≈ 4.21
            #expect(abs(r[1] - 5 * 0.8413) < 5e-2)
            // gelu(2) ≈ 1.9546 → * 11 ≈ 21.5
            #expect(abs(r[3] - 11 * 1.9546) < 5e-1)
        }
    }

    @Test("fusedGateClippedSwiglu f32 — dispatch returns finite output")
    func fusedGateClippedSwigluSmoke() {
        autoreleasepool {
            let gate = Tensor.empty(shape: [4], dtype: .f32)
            let up = Tensor.empty(shape: [4], dtype: .f32)
            gate.copyIn(from: [Float(1), -1, 2, 0.5])
            up.copyIn(from: [Float(3), 5, 7, 11])
            var out: Tensor!
            runAndWait { cb in out = Ops.fusedGateClippedSwiglu(gate: gate, up: up, on: cb) }
            #expect(out.shape == [4])
            for v in out.toArray(as: Float.self) {
                #expect(v.isFinite, "non-finite output: \(v)")
            }
        }
    }

    @Test("sigmoidMul f32 — out[i] = a[i] * sigmoid(b[i])")
    func sigmoidMulF32() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f32)
            let b = Tensor.empty(shape: [4], dtype: .f32)
            a.copyIn(from: [Float(2), 4, 6, 8])
            b.copyIn(from: [Float(0), 0, 0, 0])  // sigmoid(0) = 0.5
            var out: Tensor!
            runAndWait { cb in out = Ops.sigmoidMul(a, b, on: cb) }
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0] - 1) < 1e-4)
            #expect(abs(r[1] - 2) < 1e-4)
            #expect(abs(r[2] - 3) < 1e-4)
            #expect(abs(r[3] - 4) < 1e-4)
        }
    }

    // ─── fused norms ───────────────────────────────────────────────

    @Test("rmsNormResidual f32 — y = residual + w * x / rms(x)")
    func rmsNormResidualF32() {
        autoreleasepool {
            // n=128 (min for the fused kernel: TPG = n/4 = 32 ≥ simdgroup).
            let n = 128
            let x = Tensor.empty(shape: [n], dtype: .f32)
            let residual = Tensor.empty(shape: [n], dtype: .f32)
            let weight = Tensor.empty(shape: [n], dtype: .f32)
            // All x = 2 → rms(x) = 2; weight = 1, residual = 0 → out = 1
            x.copyIn(from: Array(repeating: Float(2), count: n))
            residual.copyIn(from: Array(repeating: Float(0), count: n))
            weight.copyIn(from: Array(repeating: Float(1), count: n))
            let epsBuf = Tensor.empty(shape: [1], dtype: .f32)
            epsBuf.copyIn(from: [Float(1e-12)])
            var out: Tensor!
            runAndWait { cb in
                out = Ops.rmsNormResidual(x: x, residual: residual,
                                          weight: weight, epsBuf: epsBuf, on: cb)
            }
            let r = out.toArray(as: Float.self)
            for v in r { #expect(abs(v - 1) < 1e-3) }
        }
    }

    @Test("gatedRmsNorm f32 — w * y * rsqrt(mean(y²)+eps) * silu(z)")
    func gatedRmsNormF32() {
        autoreleasepool {
            let n = 128
            // y all 2 → rms(y) = 2 → y * rsqrt = 1. weight = 1.
            // z all 0 → silu(0) = 0 → output should be 0.
            let y = Tensor.empty(shape: [n], dtype: .f32)
            let z = Tensor.empty(shape: [n], dtype: .f32)
            let weight = Tensor.empty(shape: [n], dtype: .f32)
            y.copyIn(from: Array(repeating: Float(2), count: n))
            z.copyIn(from: Array(repeating: Float(0), count: n))
            weight.copyIn(from: Array(repeating: Float(1), count: n))
            let epsBuf = Tensor.empty(shape: [1], dtype: .f32)
            epsBuf.copyIn(from: [Float(1e-12)])
            var out: Tensor!
            runAndWait { cb in
                out = Ops.gatedRmsNorm(y: y, z: z, weight: weight, epsBuf: epsBuf, on: cb)
            }
            let r = out.toArray(as: Float.self)
            for v in r { #expect(abs(v) < 1e-3, "expected ~0, got \(v)") }
        }
    }

    @Test("rmsNormSmall f32 — n=64 row, all-2 input → all-1 output")
    func rmsNormSmallF32() {
        autoreleasepool {
            // n=64 is the small-rms floor (TPG = n/2 = 32 = simdgroup).
            let n = 64
            let x = Tensor.empty(shape: [n], dtype: .f32)
            let weight = Tensor.empty(shape: [n], dtype: .f32)
            x.copyIn(from: Array(repeating: Float(2), count: n))
            weight.copyIn(from: Array(repeating: Float(1), count: n))
            let epsBuf = Tensor.empty(shape: [1], dtype: .f32)
            epsBuf.copyIn(from: [Float(1e-12)])
            var out: Tensor!
            runAndWait { cb in
                out = Ops.rmsNormSmall(x, weight: weight, epsBuf: epsBuf, on: cb)
            }
            let r = out.toArray(as: Float.self)
            for v in r { #expect(abs(v - 1) < 1e-3) }
        }
    }

    // ─── validators (pure, no trap) ────────────────────────────────

    @Test("validateRmsNormResidual rejects bad widths")
    func validateRmsNormResidualWidth() {
        #expect(OpsValidation.validateRmsNormResidual(n: 0) != nil)
        #expect(OpsValidation.validateRmsNormResidual(n: 100) != nil)   // not multiple of 128
        #expect(OpsValidation.validateRmsNormResidual(n: 8192) != nil)  // > 4096
        #expect(OpsValidation.validateRmsNormResidual(n: 128) == nil)
        #expect(OpsValidation.validateRmsNormResidual(n: 4096) == nil)
    }

    @Test("validateGatedRmsNorm rejects bad widths")
    func validateGatedRmsNormWidth() {
        #expect(OpsValidation.validateGatedRmsNorm(n: 64) != nil)
        #expect(OpsValidation.validateGatedRmsNorm(n: 5120) != nil)
        #expect(OpsValidation.validateGatedRmsNorm(n: 256) == nil)
    }

    @Test("validateRmsNormSmall enforces n ∈ [64, 2048] and even")
    func validateRmsNormSmallWidth() {
        #expect(OpsValidation.validateRmsNormSmall(n: 0) != nil)
        #expect(OpsValidation.validateRmsNormSmall(n: 63) != nil)    // odd
        #expect(OpsValidation.validateRmsNormSmall(n: 32) != nil)    // < 64
        #expect(OpsValidation.validateRmsNormSmall(n: 4096) != nil)  // > 2048
        #expect(OpsValidation.validateRmsNormSmall(n: 64) == nil)
        #expect(OpsValidation.validateRmsNormSmall(n: 96) == nil)
        #expect(OpsValidation.validateRmsNormSmall(n: 2048) == nil)
    }

    // ─── f16 + bf16 dtype coverage ─────────────────────────────────

    @Test("fused gate / sigmoidMul / rmsNormSmall — f16 + bf16 dispatch")
    func fusedDtypeSmoke() {
        autoreleasepool {
            // f16 gates
            let g16 = Tensor.empty(shape: [4], dtype: .f16)
            let u16 = Tensor.empty(shape: [4], dtype: .f16)
            g16.copyIn(from: [Float16(0.5), -1, 2, 0])
            u16.copyIn(from: [Float16(2), 4, 6, 8])
            var t1: Tensor!, t2: Tensor!, t3: Tensor!
            runAndWait { cb in
                t1 = Ops.fusedGateGelu(gate: g16, up: u16, on: cb)
                t2 = Ops.fusedGateClippedSwiglu(gate: g16, up: u16, on: cb)
                t3 = Ops.sigmoidMul(g16, u16, on: cb)
            }
            for tensor in [t1, t2, t3] {
                #expect(tensor!.dtype == .f16)
                for v in tensor!.toFloatArray() { #expect(v.isFinite) }
            }

            // bf16 rmsNormSmall + rmsNormResidual + gatedRmsNorm
            let n = 128
            let xBF = Tensor.empty(shape: [n], dtype: .bf16)
            let wBF = Tensor.empty(shape: [n], dtype: .bf16)
            let residBF = Tensor.empty(shape: [n], dtype: .bf16)
            // 2.0 as bf16 = 0x4000
            xBF.copyIn(from: Array(repeating: UInt16(0x4000), count: n))
            wBF.copyIn(from: Array(repeating: UInt16(0x3F80), count: n))  // 1.0
            residBF.copyIn(from: Array(repeating: UInt16(0), count: n))
            let epsBuf = Tensor.empty(shape: [1], dtype: .f32)
            epsBuf.copyIn(from: [Float(1e-12)])
            var rBF: Tensor!, smBF: Tensor!
            runAndWait { cb in
                rBF = Ops.rmsNormResidual(x: xBF, residual: residBF,
                                          weight: wBF, epsBuf: epsBuf, on: cb)
                smBF = Ops.rmsNormSmall(xBF.reshaped(to: [n]), weight: wBF,
                                        epsBuf: epsBuf, on: cb)
            }
            #expect(rBF.dtype == .bf16)
            #expect(smBF.dtype == .bf16)
            for v in rBF.toFloatArray() { #expect(v.isFinite) }
            for v in smBF.toFloatArray() { #expect(v.isFinite) }

            // gatedRmsNorm: y is fp32, z/w/out are bf16
            let yF32 = Tensor.empty(shape: [n], dtype: .f32)
            yF32.copyIn(from: Array(repeating: Float(2), count: n))
            let zBF = Tensor.empty(shape: [n], dtype: .bf16)
            zBF.copyIn(from: Array(repeating: UInt16(0x3F80), count: n))
            var gBF: Tensor!
            runAndWait { cb in
                gBF = Ops.gatedRmsNorm(y: yF32, z: zBF, weight: wBF,
                                       epsBuf: epsBuf, on: cb)
            }
            #expect(gBF.dtype == .bf16)
            for v in gBF.toFloatArray() { #expect(v.isFinite) }
        }
    }
}
