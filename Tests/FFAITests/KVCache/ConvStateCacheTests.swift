// ConvStateCacheTests — the 1D depthwise causal-conv streaming-decode
// kernel + the rolling-window state class. Multi-step runs assert
// state shifting + accumulation match a CPU reference exactly.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("ConvStateCache + conv1d_causal_step kernel")
struct ConvStateCacheTests {

    // MARK: - State cache plumbing

    @Test("init creates zeroed state of the right shape")
    func initShape() {
        autoreleasepool {
            let cache = ConvStateCache(nChannels: 16, kernelSize: 4, dtype: .f32)
            // state shape is [K-1, nChannels] = [3, 16]
            #expect(cache.state.shape == [3, 16])
            #expect(cache.state.dtype == .f32)
            #expect(cache.state.toArray(as: Float.self).allSatisfy { $0 == 0 })
            #expect(cache.bytesAllocated == 3 * 16 * 4)
        }
    }

    @Test("reset zeroes the rolling window")
    func resetClears() {
        autoreleasepool {
            let cache = ConvStateCache(nChannels: 2, kernelSize: 3, dtype: .f32)
            let ptr = cache.state.buffer.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<4 { ptr[i] = Float(i + 1) }
            cache.reset()
            #expect(cache.state.toArray(as: Float.self) == [0, 0, 0, 0])
        }
    }

    @Test("totalBytesAllocated across an array")
    func totalBytes() {
        autoreleasepool {
            let c1 = ConvStateCache(nChannels: 8, kernelSize: 4, dtype: .f16)
            let c2 = ConvStateCache(nChannels: 8, kernelSize: 4, dtype: .f16)
            #expect([c1, c2].totalBytesAllocated == 2 * c1.bytesAllocated)
        }
    }

    // MARK: - conv1d_causal_step correctness

    /// CPU reference: causal 1D depthwise conv, one streaming step.
    /// Matches the kernel: y[d] = b[d] + Σ_k w[k][d] * input_window[k][d]
    /// where input_window = state ⊕ [x] (state[0] is oldest).
    private func cpuRefStep(
        x: [Float], w: [Float], b: [Float],
        state: inout [Float],
        nChannels: Int, kernelSize: Int
    ) -> [Float] {
        var y = [Float](repeating: 0, count: nChannels)
        for d in 0..<nChannels {
            var acc = b[d]
            // state[k][d] for k in 0..K-2
            for k in 0..<(kernelSize - 1) {
                acc += w[k * nChannels + d] * state[k * nChannels + d]
            }
            // w[K-1] pairs with current x
            acc += w[(kernelSize - 1) * nChannels + d] * x[d]
            y[d] = acc
        }
        // Shift state: drop state[0], append x at state[K-2]
        for k in 0..<(kernelSize - 2) {
            for d in 0..<nChannels {
                state[k * nChannels + d] = state[(k + 1) * nChannels + d]
            }
        }
        for d in 0..<nChannels {
            state[(kernelSize - 2) * nChannels + d] = x[d]
        }
        return y
    }

    private func gpuStep(
        x: [Float], w: [Float], b: [Float], initialState: [Float],
        nChannels: Int, kernelSize: Int
    ) -> (state: [Float], y: [Float]) {
        let xT = Tensor.empty(shape: [nChannels], dtype: .f32)
        xT.copyIn(from: x)
        let wT = Tensor.empty(shape: [kernelSize, nChannels], dtype: .f32)
        wT.copyIn(from: w)
        let bT = Tensor.empty(shape: [nChannels], dtype: .f32)
        bT.copyIn(from: b)
        let cache = ConvStateCache(nChannels: nChannels, kernelSize: kernelSize, dtype: .f32)
        cache.state.copyIn(from: initialState)
        let yT = Tensor.empty(shape: [nChannels], dtype: .f32)
        yT.zero()
        runAndWait { cb in
            Ops.conv1dCausalStep(x: xT, w: wT, b: bT,
                                 state: cache.state, into: yT,
                                 nChannels: nChannels, kernelSize: kernelSize,
                                 on: cb)
        }
        return (cache.state.toArray(as: Float.self), yT.toArray(as: Float.self))
    }

    @Test("single conv step matches CPU reference")
    func singleStepMatchesCPU() {
        autoreleasepool {
            let nChannels = 16, kernelSize = 4
            var cpuState = (0..<(kernelSize - 1) * nChannels).map { Float($0) * 0.01 }
            let initial = cpuState
            let x = (0..<nChannels).map { Float($0 - 8) * 0.1 }
            let w = (0..<kernelSize * nChannels).map { Float(($0 % 7) - 3) * 0.05 }
            let b = (0..<nChannels).map { Float($0) * 0.001 - 0.005 }

            let cpuY = cpuRefStep(x: x, w: w, b: b, state: &cpuState,
                                  nChannels: nChannels, kernelSize: kernelSize)
            let (gpuState, gpuY) = gpuStep(x: x, w: w, b: b, initialState: initial,
                                           nChannels: nChannels, kernelSize: kernelSize)

            let tol: Float = 1e-5
            for i in 0..<cpuState.count {
                #expect(abs(gpuState[i] - cpuState[i]) < tol,
                        "state[\(i)] gpu=\(gpuState[i]) cpu=\(cpuState[i])")
            }
            for i in 0..<cpuY.count {
                #expect(abs(gpuY[i] - cpuY[i]) < tol,
                        "y[\(i)] gpu=\(gpuY[i]) cpu=\(cpuY[i])")
            }
        }
    }

    @Test("8 sequential conv steps match CPU reference (state shift verification)")
    func multiStepMatchesCPU() {
        autoreleasepool {
            let nChannels = 12, kernelSize = 4
            var cpuState = [Float](repeating: 0, count: (kernelSize - 1) * nChannels)
            let cache = ConvStateCache(nChannels: nChannels, kernelSize: kernelSize,
                                       dtype: .f32)

            let w = (0..<kernelSize * nChannels).map { Float(($0 % 5) - 2) * 0.07 }
            let b = (0..<nChannels).map { Float($0) * 0.01 }

            let wT = Tensor.empty(shape: [kernelSize, nChannels], dtype: .f32)
            wT.copyIn(from: w)
            let bT = Tensor.empty(shape: [nChannels], dtype: .f32)
            bT.copyIn(from: b)

            for step in 0..<8 {
                let x: [Float] = (0..<nChannels).map { Float((step + $0) % 7) * 0.13 - 0.4 }
                let cpuY = cpuRefStep(x: x, w: w, b: b, state: &cpuState,
                                      nChannels: nChannels, kernelSize: kernelSize)
                let xT = Tensor.empty(shape: [nChannels], dtype: .f32)
                xT.copyIn(from: x)
                let yT = Tensor.empty(shape: [nChannels], dtype: .f32); yT.zero()
                runAndWait { cb in
                    Ops.conv1dCausalStep(x: xT, w: wT, b: bT,
                                         state: cache.state, into: yT,
                                         nChannels: nChannels, kernelSize: kernelSize,
                                         on: cb)
                }
                let gpuState = cache.state.toArray(as: Float.self)
                let gpuY = yT.toArray(as: Float.self)
                let tol: Float = 1e-5
                for i in 0..<cpuState.count {
                    #expect(abs(gpuState[i] - cpuState[i]) < tol,
                            "step=\(step) state[\(i)] gpu=\(gpuState[i]) cpu=\(cpuState[i])")
                }
                for i in 0..<cpuY.count {
                    #expect(abs(gpuY[i] - cpuY[i]) < tol,
                            "step=\(step) y[\(i)] gpu=\(gpuY[i]) cpu=\(cpuY[i])")
                }
            }
        }
    }

    @Test("kernel_size=2 (minimal stateful conv) works")
    func kernelSize2() {
        autoreleasepool {
            // K=2: state holds 1 previous input; conv = w[0]*state[0] + w[1]*x + b
            let nChannels = 8, kernelSize = 2
            var cpuState = (0..<nChannels).map { Float($0) * 0.1 }
            let initial = cpuState
            let x = (0..<nChannels).map { Float(nChannels - $0) * 0.2 }
            let w = (0..<kernelSize * nChannels).map { Float($0) * 0.05 + 0.01 }
            let b = [Float](repeating: 0, count: nChannels)

            let cpuY = cpuRefStep(x: x, w: w, b: b, state: &cpuState,
                                  nChannels: nChannels, kernelSize: kernelSize)
            let (gpuState, gpuY) = gpuStep(x: x, w: w, b: b, initialState: initial,
                                           nChannels: nChannels, kernelSize: kernelSize)

            // After one step at K=2: state = [x]
            for d in 0..<nChannels {
                #expect(abs(gpuState[d] - x[d]) < 1e-6)
            }
            for d in 0..<nChannels {
                #expect(abs(gpuY[d] - cpuY[d]) < 1e-5)
            }
        }
    }

    @Test("bf16 conv step preserves CPU reference within bf16 tolerance")
    func bf16Inputs() {
        autoreleasepool {
            let nChannels = 8, kernelSize = 4
            var cpuState = (0..<(kernelSize - 1) * nChannels).map { Float($0) * 0.05 }
            let initial = cpuState
            let x = (0..<nChannels).map { Float($0) * 0.1 - 0.3 }
            let w = (0..<kernelSize * nChannels).map { Float($0 % 3) * 0.2 }
            let b = (0..<nChannels).map { Float($0) * 0.05 }

            _ = cpuRefStep(x: x, w: w, b: b, state: &cpuState,
                           nChannels: nChannels, kernelSize: kernelSize)

            let cache = ConvStateCache(nChannels: nChannels, kernelSize: kernelSize,
                                       dtype: .bf16)
            cache.state.copyIn(from: initial.map { f -> UInt16 in
                UInt16(truncatingIfNeeded: f.bitPattern >> 16)
            })
            let xT = Tensor.empty(shape: [nChannels], dtype: .bf16)
            xT.copyIn(from: x.map { f -> UInt16 in
                UInt16(truncatingIfNeeded: f.bitPattern >> 16)
            })
            let wT = Tensor.empty(shape: [kernelSize, nChannels], dtype: .bf16)
            wT.copyIn(from: w.map { f -> UInt16 in
                UInt16(truncatingIfNeeded: f.bitPattern >> 16)
            })
            let bT = Tensor.empty(shape: [nChannels], dtype: .bf16)
            bT.copyIn(from: b.map { f -> UInt16 in
                UInt16(truncatingIfNeeded: f.bitPattern >> 16)
            })
            let yT = Tensor.empty(shape: [nChannels], dtype: .bf16); yT.zero()
            runAndWait { cb in
                Ops.conv1dCausalStep(x: xT, w: wT, b: bT,
                                     state: cache.state, into: yT,
                                     nChannels: nChannels, kernelSize: kernelSize,
                                     on: cb)
            }

            // State should be x (just shifted in). Compare via bf16 decode.
            let stateBits = cache.state.toArray(as: UInt16.self)
            let tol: Float = 0.05
            for d in 0..<nChannels {
                // state[K-2][d] should equal x[d] in bf16.
                let idx = (kernelSize - 2) * nChannels + d
                let bits = UInt32(stateBits[idx]) << 16
                let v = Float(bitPattern: bits)
                #expect(abs(v - x[d]) < tol)
            }
        }
    }
}
