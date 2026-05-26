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
// GDNStateCacheTests — the GDNStateCache class + the gated_delta_step
// kernel against a CPU reference of the Gated Delta Net recurrence.
// Multi-step runs assert the state recurrence behaves identically to
// the CPU reference within fp32 precision.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import FFAI

@Suite("GDNStateCache + gated_delta_step kernel")
struct GDNStateCacheTests {

    // MARK: - State cache plumbing

    @Test("init creates two zeroed fp32 state buffers of the right shape")
    func initShape() {
        autoreleasepool {
            let cache = GDNStateCache(
                numValueHeads: 8, valueHeadDim: 64,
                keyHeadDim: 64)
            #expect(cache.current.shape == [8, 64, 64])
            #expect(cache.next.shape == [8, 64, 64])
            #expect(cache.current.dtype == .f32)
            #expect(cache.next.dtype == .f32)
            #expect(cache.current.toArray(as: Float.self).allSatisfy { $0 == 0 })
            #expect(cache.next.toArray(as: Float.self).allSatisfy { $0 == 0 })
            #expect(cache.length == 0)
            #expect(cache.maxSeq == .max)
            // Two buffers: 2 * Hv * Dv * Dk * 4 bytes.
            #expect(cache.bytesAllocated == 2 * 8 * 64 * 64 * 4)
            #expect(cache.bytesInUse == 0)
        }
    }

    @Test("swap exchanges current/next and advances length")
    func swapPingPongs() {
        autoreleasepool {
            let cache = GDNStateCache(
                numValueHeads: 1, valueHeadDim: 2,
                keyHeadDim: 32)
            // Mark the two buffers distinctly via raw writes.
            let curPtr = cache.current.buffer.contents()
                .assumingMemoryBound(to: Float.self)
            let nextPtr = cache.next.buffer.contents()
                .assumingMemoryBound(to: Float.self)
            curPtr[0] = 11.0
            nextPtr[0] = 22.0

            cache.swap()
            #expect(cache.length == 1)
            // After the swap, `current` should be the buffer that held 22.
            let nowCur = cache.current.buffer.contents()
                .assumingMemoryBound(to: Float.self)
            #expect(nowCur[0] == 22.0)
            #expect(cache.bytesInUse == cache.bytesAllocated)

            cache.swap()
            #expect(cache.length == 2)
        }
    }

    @Test("reset zeroes both buffers and the step counter")
    func resetClears() {
        autoreleasepool {
            let cache = GDNStateCache(
                numValueHeads: 1, valueHeadDim: 1,
                keyHeadDim: 32)
            cache.current.buffer.contents()
                .assumingMemoryBound(to: Float.self)[0] = 5.0
            cache.next.buffer.contents()
                .assumingMemoryBound(to: Float.self)[0] = 7.0
            cache.swap()
            #expect(cache.length == 1)

            cache.reset()
            #expect(cache.length == 0)
            #expect(cache.current.toArray(as: Float.self).allSatisfy { $0 == 0 })
            #expect(cache.next.toArray(as: Float.self).allSatisfy { $0 == 0 })
            #expect(cache.bytesInUse == 0)
        }
    }

    @Test("totalBytesAllocated across an array")
    func totalBytes() {
        autoreleasepool {
            let c1 = GDNStateCache(
                numValueHeads: 8, valueHeadDim: 64,
                keyHeadDim: 64)
            let c2 = GDNStateCache(
                numValueHeads: 8, valueHeadDim: 64,
                keyHeadDim: 64)
            #expect([c1, c2].totalBytesAllocated == 2 * c1.bytesAllocated)
        }
    }

    @Test("conforms to LayerCacheProtocol")
    func protocolConformance() {
        autoreleasepool {
            let cache: LayerCacheProtocol = GDNStateCache(
                numValueHeads: 1, valueHeadDim: 1, keyHeadDim: 32)
            #expect(cache.length == 0)
            #expect(cache.maxSeq == .max)
            cache.reset()
            #expect(cache.length == 0)
        }
    }

    // MARK: - gated_delta_step kernel correctness

    // Test config — the smallest emitted kernel: Dk=64, Dv=64, Hk=8, Hv=8.
    private static let dk = 64
    private static let dv = 64
    private static let hk = 8
    private static let hv = 8

    /// CPU reference for the GDN recurrence over `tSteps` sequential
    /// steps. Single-batch. Mirrors the kernel's per-head fp32 state
    /// update + output projection exactly.
    ///
    /// Layouts (T = tSteps):
    ///   q, k    [T, Hk, Dk]
    ///   v       [T, Hv, Dv]
    ///   g, beta [T, Hv]
    ///   state   [Hv, Dv, Dk]  (mutated in place)
    ///   returns y [T, Hv, Dv]
    private func cpuRef(
        q: [Float], k: [Float], v: [Float], g: [Float], beta: [Float],
        state: inout [Float], tSteps: Int
    ) -> [Float] {
        let dk = Self.dk
        let dv = Self.dv
        let hk = Self.hk
        let hv = Self.hv
        let headsPerKV = hv / hk
        var y = [Float](repeating: 0, count: tSteps * hv * dv)
        for hvIdx in 0 ..< hv {
            let hkIdx = hvIdx / headsPerKV
            let n = hvIdx  // single-batch → n = hv
            for t in 0 ..< tSteps {
                let qkBase = (t * hk + hkIdx) * dk
                let vBase = (t * hv + hvIdx) * dv
                let gbBase = t * hv + hvIdx
                let gVal = g[gbBase]
                let betaVal = beta[gbBase]
                for d in 0 ..< dv {
                    let sRow = (n * dv + d) * dk
                    var kvMem: Float = 0
                    for c in 0 ..< dk {
                        state[sRow + c] *= gVal
                        kvMem += state[sRow + c] * k[qkBase + c]
                    }
                    let delta = (v[vBase + d] - kvMem) * betaVal
                    var out: Float = 0
                    for c in 0 ..< dk {
                        state[sRow + c] += k[qkBase + c] * delta
                        out += state[sRow + c] * q[qkBase + c]
                    }
                    y[vBase + d] = out
                }
            }
        }
        return y
    }

    /// Run `tSteps` of the GDN recurrence on the GPU by looping the
    /// single-step `mt_gated_delta_step` kernel once per token —
    /// `mt_gated_delta_step` performs exactly one recurrence step, so
    /// multi-step prefill is a per-token loop, double-buffering state
    /// via `GDNStateCache.swap()`. Returns (finalState, y).
    private func gpuRun(
        q: [Float], k: [Float], v: [Float], g: [Float], beta: [Float],
        initialState: [Float], tSteps: Int
    ) -> (state: [Float], y: [Float]) {
        let dk = Self.dk
        let dv = Self.dv
        let hk = Self.hk
        let hv = Self.hv

        let cache = GDNStateCache(
            numValueHeads: hv, valueHeadDim: dv,
            keyHeadDim: dk)
        cache.current.copyIn(from: initialState)

        var y = [Float](repeating: 0, count: tSteps * hv * dv)

        for t in 0 ..< tSteps {
            // Per-token slices — q/k are [Hk, Dk], v is [Hv, Dv],
            // g/beta are [Hv].
            let qkLen = hk * dk
            let vLen = hv * dv
            let qT = Tensor.empty(shape: [hk, dk], dtype: .f32)
            qT.copyIn(from: Array(q[(t * qkLen) ..< ((t + 1) * qkLen)]))
            let kT = Tensor.empty(shape: [hk, dk], dtype: .f32)
            kT.copyIn(from: Array(k[(t * qkLen) ..< ((t + 1) * qkLen)]))
            let vT = Tensor.empty(shape: [hv, dv], dtype: .f32)
            vT.copyIn(from: Array(v[(t * vLen) ..< ((t + 1) * vLen)]))
            let gT = Tensor.empty(shape: [hv], dtype: .f32)
            gT.copyIn(from: Array(g[(t * hv) ..< ((t + 1) * hv)]))
            let betaT = Tensor.empty(shape: [hv], dtype: .f32)
            betaT.copyIn(from: Array(beta[(t * hv) ..< ((t + 1) * hv)]))

            let yT = Tensor.empty(shape: [hv, dv], dtype: .f32)
            yT.zero()

            runAndWait { cb in
                Ops.gatedDeltaStep(
                    q: qT, k: kT, v: vT, g: gT, beta: betaT,
                    stateIn: cache.current, into: yT, stateOut: cache.next,
                    numKeyHeads: hk, numValueHeads: hv,
                    keyHeadDim: dk, valueHeadDim: dv, on: cb)
            }
            // The kernel wrote the updated state into `next`; swap so
            // the next token reads it via `current`.
            cache.swap()

            let yStep = yT.toArray(as: Float.self)
            for i in 0 ..< vLen { y[t * vLen + i] = yStep[i] }
        }
        // After the final swap, the freshly-written state is `current`.
        return (cache.current.toArray(as: Float.self), y)
    }

    @Test("gated_delta_step matches CPU reference on a single step")
    func singleStepMatchesCPU() {
        autoreleasepool {
            let dk = Self.dk
            let dv = Self.dv
            let hk = Self.hk
            let hv = Self.hv
            let tSteps = 1

            var q = [Float](repeating: 0, count: tSteps * hk * dk)
            for i in 0 ..< q.count { q[i] = Float((i % 17) - 8) * 0.01 }
            var k = [Float](repeating: 0, count: tSteps * hk * dk)
            for i in 0 ..< k.count { k[i] = Float((i % 13) - 6) * 0.01 }
            var v = [Float](repeating: 0, count: tSteps * hv * dv)
            for i in 0 ..< v.count { v[i] = Float((i % 11) - 5) * 0.02 }
            // g ∈ (0, 1): contractive decay gate.
            var g = [Float](repeating: 0, count: tSteps * hv)
            for i in 0 ..< g.count { g[i] = 0.85 + 0.01 * Float(i % 10) }
            var beta = [Float](repeating: 0, count: tSteps * hv)
            for i in 0 ..< beta.count { beta[i] = 0.3 + 0.04 * Float(i % 8) }
            // Non-zero initial state so the decay path is exercised.
            var initialState = [Float](repeating: 0, count: hv * dv * dk)
            for i in 0 ..< initialState.count {
                initialState[i] = Float((i % 19) - 9) * 0.005
            }

            var cpuState = initialState
            let cpuY = cpuRef(
                q: q, k: k, v: v, g: g, beta: beta,
                state: &cpuState, tSteps: tSteps)
            let (gpuState, gpuY) = gpuRun(
                q: q, k: k, v: v, g: g, beta: beta,
                initialState: initialState,
                tSteps: tSteps)

            let tol: Float = 1e-4
            for i in 0 ..< cpuState.count {
                #expect(
                    abs(gpuState[i] - cpuState[i]) < tol,
                    "state[\(i)] gpu=\(gpuState[i]) cpu=\(cpuState[i])")
            }
            for i in 0 ..< cpuY.count {
                #expect(
                    abs(gpuY[i] - cpuY[i]) < tol,
                    "y[\(i)] gpu=\(gpuY[i]) cpu=\(cpuY[i])")
            }
        }
    }

    @Test("gated_delta_step matches CPU reference across 12 steps")
    func multiStepMatchesCPU() {
        autoreleasepool {
            let dk = Self.dk
            let dv = Self.dv
            let hk = Self.hk
            let hv = Self.hv
            let tSteps = 12

            var q = [Float](repeating: 0, count: tSteps * hk * dk)
            for i in 0 ..< q.count { q[i] = Float((i % 23) - 11) * 0.008 }
            var k = [Float](repeating: 0, count: tSteps * hk * dk)
            for i in 0 ..< k.count { k[i] = Float((i % 29) - 14) * 0.008 }
            var v = [Float](repeating: 0, count: tSteps * hv * dv)
            for i in 0 ..< v.count { v[i] = Float((i % 31) - 15) * 0.015 }
            var g = [Float](repeating: 0, count: tSteps * hv)
            for i in 0 ..< g.count { g[i] = 0.88 + 0.008 * Float(i % 12) }
            var beta = [Float](repeating: 0, count: tSteps * hv)
            for i in 0 ..< beta.count { beta[i] = 0.25 + 0.03 * Float(i % 9) }
            let initialState = [Float](repeating: 0, count: hv * dv * dk)

            var cpuState = initialState
            let cpuY = cpuRef(
                q: q, k: k, v: v, g: g, beta: beta,
                state: &cpuState, tSteps: tSteps)
            let (gpuState, gpuY) = gpuRun(
                q: q, k: k, v: v, g: g, beta: beta,
                initialState: initialState,
                tSteps: tSteps)

            // 12 sequential steps compound fp reassociation noise from
            // the simd_sum reductions; 1e-4 is comfortably tight.
            let tol: Float = 1e-4
            for i in 0 ..< cpuState.count {
                #expect(
                    abs(gpuState[i] - cpuState[i]) < tol,
                    "state[\(i)] gpu=\(gpuState[i]) cpu=\(cpuState[i])")
            }
            for i in 0 ..< cpuY.count {
                #expect(
                    abs(gpuY[i] - cpuY[i]) < tol,
                    "y[\(i)] gpu=\(gpuY[i]) cpu=\(cpuY[i])")
            }
        }
    }
}
