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
// SSMStateCacheTests — the SSMStateCache class + the ssm_step kernel
// against a CPU reference implementation of Mamba 2's selective
// scan. Multi-step runs assert state recurrence behaves identically
// to the CPU reference within fp32 precision.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import FFAI

@Suite("SSMStateCache + ssm_step kernel")
struct SSMStateCacheTests {

    // MARK: - State cache plumbing

    @Test("init creates zeroed fp32 state of the right shape")
    func initShape() {
        autoreleasepool {
            let cache = SSMStateCache(nHeads: 2, stateDim: 4, headDim: 8)
            #expect(cache.h.shape == [2, 4, 8])
            #expect(cache.h.dtype == .f32)
            #expect(cache.h.toArray(as: Float.self).allSatisfy { $0 == 0 })
            // Size: nHeads * stateDim * headDim * 4 bytes
            #expect(cache.bytesAllocated == 2 * 4 * 8 * 4)
        }
    }

    @Test("reset zeroes the state")
    func resetClears() {
        autoreleasepool {
            let cache = SSMStateCache(nHeads: 1, stateDim: 2, headDim: 2)
            // Poke a non-zero value in via raw memory write.
            let ptr = cache.h.buffer.contents().assumingMemoryBound(to: Float.self)
            ptr[0] = 1.0
            ptr[1] = 2.0
            ptr[2] = 3.0
            ptr[3] = 4.0
            #expect(cache.h.toArray(as: Float.self) == [1, 2, 3, 4])
            cache.reset()
            #expect(cache.h.toArray(as: Float.self) == [0, 0, 0, 0])
        }
    }

    @Test("totalBytesAllocated across an array")
    func totalBytes() {
        autoreleasepool {
            let c1 = SSMStateCache(nHeads: 2, stateDim: 4, headDim: 8)
            let c2 = SSMStateCache(nHeads: 2, stateDim: 4, headDim: 8)
            #expect([c1, c2].totalBytesAllocated == 2 * c1.bytesAllocated)
        }
    }

    // MARK: - ssm_step kernel correctness

    /// CPU reference for one Mamba 2 selective-scan decode step.
    /// Same math as the kernel:
    ///   h[head, n, d]_new = exp(A[head] * dt) * h[head, n, d]_old
    ///                       + dt * B[n] * x[head, d]
    ///   y[head, d]         = Σ_n  C[n] * h[head, n, d]_new
    private func cpuRefStep(
        x: [Float], a: [Float], b: [Float], c: [Float], dt: [Float],
        h: inout [Float],
        nHeads: Int, stateDim: Int, headDim: Int
    ) -> [Float] {
        var y = [Float](repeating: 0, count: nHeads * headDim)
        for head in 0 ..< nHeads {
            let dtH = dt[head]
            let decay = Foundation.exp(a[head] * dtH)
            for d in 0 ..< headDim {
                let xVal = x[head * headDim + d]
                var yVal: Float = 0
                for n in 0 ..< stateDim {
                    let hIdx = head * stateDim * headDim + n * headDim + d
                    let bVal = b[n]
                    let cVal = c[n]
                    let newH = decay * h[hIdx] + dtH * bVal * xVal
                    h[hIdx] = newH
                    yVal += cVal * newH
                }
                y[head * headDim + d] = yVal
            }
        }
        return y
    }

    /// Run one GPU step. Returns (newState, y).
    private func gpuStep(
        x: [Float], a: [Float], b: [Float], c: [Float], dt: [Float],
        initialState: [Float],
        nHeads: Int, stateDim: Int, headDim: Int
    ) -> (state: [Float], y: [Float]) {
        let xT = Tensor.empty(shape: [nHeads, headDim], dtype: .f32)
        xT.copyIn(from: x)
        let aT = Tensor.empty(shape: [nHeads], dtype: .f32)
        aT.copyIn(from: a)
        let bT = Tensor.empty(shape: [stateDim], dtype: .f32)
        bT.copyIn(from: b)
        let cT = Tensor.empty(shape: [stateDim], dtype: .f32)
        cT.copyIn(from: c)
        let dtT = Tensor.empty(shape: [nHeads], dtype: .f32)
        dtT.copyIn(from: dt)

        let cache = SSMStateCache(nHeads: nHeads, stateDim: stateDim, headDim: headDim)
        cache.h.copyIn(from: initialState)

        let yT = Tensor.empty(shape: [nHeads, headDim], dtype: .f32)
        yT.zero()

        runAndWait { cb in
            Ops.ssmStep(
                x: xT, a: aT, b: bT, c: cT, dt: dtT,
                state: cache.h, into: yT,
                nHeads: nHeads, headDim: headDim, stateDim: stateDim,
                on: cb)
        }
        return (cache.h.toArray(as: Float.self), yT.toArray(as: Float.self))
    }

    @Test("ssm_step matches CPU reference on a single step")
    func singleStepMatchesCPU() {
        autoreleasepool {
            let nHeads = 2
            let stateDim = 4
            let headDim = 8
            var initialH = [Float](repeating: 0, count: nHeads * stateDim * headDim)
            // Seed the state with non-zero values so the decay branch is exercised.
            for i in 0 ..< initialH.count { initialH[i] = Float(i) * 0.01 }

            // Realistic inputs: x in [-1, 1], a negative (decay), B/C small,
            // dt small positive.
            var x = [Float](repeating: 0, count: nHeads * headDim)
            for i in 0 ..< x.count { x[i] = Float((i % 7) - 3) * 0.1 }
            let a: [Float] = [-1.5, -0.7]  // per-head
            var b = [Float](repeating: 0, count: stateDim)
            for i in 0 ..< stateDim { b[i] = Float(i) * 0.5 + 0.3 }
            var c = [Float](repeating: 0, count: stateDim)
            for i in 0 ..< stateDim { c[i] = Float(i + 1) * 0.2 }
            // Per-head dt (Mamba 2 spec) — vary across heads to exercise the lookup.
            let dt: [Float] = [0.05, 0.08]

            var cpuH = initialH
            let cpuY = cpuRefStep(
                x: x, a: a, b: b, c: c, dt: dt,
                h: &cpuH,
                nHeads: nHeads, stateDim: stateDim, headDim: headDim)
            let (gpuH, gpuY) = gpuStep(
                x: x, a: a, b: b, c: c, dt: dt,
                initialState: initialH,
                nHeads: nHeads, stateDim: stateDim, headDim: headDim)

            // Compare state buffers element-wise — fp32 math, kernel and
            // CPU should be identical to ~1e-6.
            let tol: Float = 1e-5
            for i in 0 ..< cpuH.count {
                #expect(
                    abs(gpuH[i] - cpuH[i]) < tol,
                    "h[\(i)] gpu=\(gpuH[i]) cpu=\(cpuH[i])")
            }
            for i in 0 ..< cpuY.count {
                #expect(
                    abs(gpuY[i] - cpuY[i]) < tol,
                    "y[\(i)] gpu=\(gpuY[i]) cpu=\(cpuY[i])")
            }
        }
    }

    @Test("ssm_step matches CPU reference across 12 sequential decode steps")
    func multiStepMatchesCPU() {
        autoreleasepool {
            let nHeads = 4
            let stateDim = 8
            let headDim = 16
            var cpuH = [Float](repeating: 0, count: nHeads * stateDim * headDim)
            let cache = SSMStateCache(nHeads: nHeads, stateDim: stateDim, headDim: headDim)

            // Fixed per-step inputs.
            let a = (0 ..< nHeads).map { Float(-$0 - 1) * 0.3 }  // per-head, all negative
            let b = (0 ..< stateDim).map { Float($0) * 0.1 + 0.05 }
            let c = (0 ..< stateDim).map { Float($0 + 1) * 0.07 }
            // Per-head dt — same for all heads here, just exercises the array path.
            let dt: [Float] = Array(repeating: 0.02, count: nHeads)

            for step in 0 ..< 12 {
                // Different x per step.
                var x = [Float](repeating: 0, count: nHeads * headDim)
                for i in 0 ..< x.count {
                    x[i] = Float((step + i) % 11 - 5) * 0.08
                }
                let cpuY = cpuRefStep(
                    x: x, a: a, b: b, c: c, dt: dt,
                    h: &cpuH,
                    nHeads: nHeads, stateDim: stateDim, headDim: headDim)
                // Drive the GPU cache.
                let xT = Tensor.empty(shape: [nHeads, headDim], dtype: .f32)
                xT.copyIn(from: x)
                let aT = Tensor.empty(shape: [nHeads], dtype: .f32)
                aT.copyIn(from: a)
                let bT = Tensor.empty(shape: [stateDim], dtype: .f32)
                bT.copyIn(from: b)
                let cT = Tensor.empty(shape: [stateDim], dtype: .f32)
                cT.copyIn(from: c)
                let dtT = Tensor.empty(shape: [nHeads], dtype: .f32)
                dtT.copyIn(from: dt)
                let yT = Tensor.empty(shape: [nHeads, headDim], dtype: .f32)
                yT.zero()
                runAndWait { cb in
                    Ops.ssmStep(
                        x: xT, a: aT, b: bT, c: cT, dt: dtT,
                        state: cache.h, into: yT,
                        nHeads: nHeads, headDim: headDim, stateDim: stateDim,
                        on: cb)
                }
                let gpuH = cache.h.toArray(as: Float.self)
                let gpuY = yT.toArray(as: Float.self)
                // Drift would accumulate across steps if the kernel had any
                // bug; assert at every step.
                let tol: Float = 1e-4  // slightly looser to absorb step accumulation
                for i in 0 ..< cpuH.count {
                    #expect(
                        abs(gpuH[i] - cpuH[i]) < tol,
                        "step=\(step) h[\(i)] gpu=\(gpuH[i]) cpu=\(cpuH[i])")
                }
                for i in 0 ..< cpuY.count {
                    #expect(
                        abs(gpuY[i] - cpuY[i]) < tol,
                        "step=\(step) y[\(i)] gpu=\(gpuY[i]) cpu=\(cpuY[i])")
                }
            }
        }
    }

    @Test("ssm_step on bf16 inputs still matches CPU reference within bf16 tolerance")
    func bf16InputsMatchCPU() {
        autoreleasepool {
            let nHeads = 2
            let stateDim = 4
            let headDim = 8
            var cpuH = [Float](repeating: 0, count: nHeads * stateDim * headDim)
            let cache = SSMStateCache(nHeads: nHeads, stateDim: stateDim, headDim: headDim)

            let x = (0 ..< (nHeads * headDim)).map { Float($0 % 5 - 2) * 0.1 }
            let a: [Float] = [-1.0, -0.5]
            let b = (0 ..< stateDim).map { Float($0) * 0.2 + 0.1 }
            let c = (0 ..< stateDim).map { Float($0 + 1) * 0.15 }
            let dt: [Float] = Array(repeating: 0.03, count: nHeads)

            _ = cpuRefStep(
                x: x, a: a, b: b, c: c, dt: dt,
                h: &cpuH,
                nHeads: nHeads, stateDim: stateDim, headDim: headDim)

            // GPU: drive with bf16-precision inputs.
            let xT = Tensor.empty(shape: [nHeads, headDim], dtype: .bf16)
            xT.copyIn(
                from: x.map { f -> UInt16 in
                    UInt16(truncatingIfNeeded: f.bitPattern >> 16)
                })
            let aT = Tensor.empty(shape: [nHeads], dtype: .bf16)
            aT.copyIn(
                from: a.map { f -> UInt16 in
                    UInt16(truncatingIfNeeded: f.bitPattern >> 16)
                })
            let bT = Tensor.empty(shape: [stateDim], dtype: .bf16)
            bT.copyIn(
                from: b.map { f -> UInt16 in
                    UInt16(truncatingIfNeeded: f.bitPattern >> 16)
                })
            let cT = Tensor.empty(shape: [stateDim], dtype: .bf16)
            cT.copyIn(
                from: c.map { f -> UInt16 in
                    UInt16(truncatingIfNeeded: f.bitPattern >> 16)
                })
            let dtT = Tensor.empty(shape: [nHeads], dtype: .bf16)
            dtT.copyIn(
                from: dt.map { f -> UInt16 in
                    UInt16(truncatingIfNeeded: f.bitPattern >> 16)
                })
            let yT = Tensor.empty(shape: [nHeads, headDim], dtype: .bf16)
            yT.zero()

            runAndWait { cb in
                Ops.ssmStep(
                    x: xT, a: aT, b: bT, c: cT, dt: dtT,
                    state: cache.h, into: yT,
                    nHeads: nHeads, headDim: headDim, stateDim: stateDim,
                    on: cb)
            }

            // State (fp32) should still match the CPU reference closely;
            // outputs (bf16) only to bf16 precision (~0.02 abs at our scale).
            let stateTol: Float = 1e-2
            let outTol: Float = 0.05
            let gpuH = cache.h.toArray(as: Float.self)
            for i in 0 ..< cpuH.count {
                #expect(
                    abs(gpuH[i] - cpuH[i]) < stateTol,
                    "h[\(i)] gpu=\(gpuH[i]) cpu=\(cpuH[i])")
            }
            let gpuYBits = yT.toArray(as: UInt16.self)
            for i in 0 ..< (nHeads * headDim) {
                // Decode bf16 back to fp32 for comparison.
                let bits = UInt32(gpuYBits[i]) << 16
                let gpuYf = Float(bitPattern: bits)
                // Recompute CPU expected y for index i.
                // (We didn't store the CPU y from above — recompute trivially
                // from cpuH: y[head, d] = Σ_n C[n] * h[head, n, d].)
                let head = i / headDim
                let d = i - head * headDim
                var expected: Float = 0
                for n in 0 ..< stateDim {
                    expected += c[n] * cpuH[head * stateDim * headDim + n * headDim + d]
                }
                #expect(
                    abs(gpuYf - expected) < outTol,
                    "y[\(i)] gpu=\(gpuYf) cpu=\(expected)")
            }
        }
    }
}
