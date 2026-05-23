// MTLIndirectCommandBuffer smoke test — verify FFAI's PSOs can execute
// via Apple's ICB API. Doesn't yet test perf, just plumbing: record a
// trivial compute command into an ICB, execute it through a compute
// encoder, verify the output matches a direct dispatch.
//
// Foundational test for the per-token graph-replay work that targets
// decode launch overhead (~48 ms of 60 ms/token is CPU dispatch
// latency at T=1). ICB recording requires
// `supportIndirectCommandBuffers = true` on every PSO — flipped in
// PSOCache.swift (commit 650e65d). This test validates the flag took.

import Foundation
import Metal
import Testing
@testable import MetalTileSwift

@Suite("ICB plumbing smoke")
struct IndirectCommandBufferSmokeTests {

    @Test("ICB executes a recorded compute command and produces the same result as direct dispatch")
    func icbExecutesRecordedSigmoidF32() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("no Metal device")
            return
        }
        guard let queue = device.makeCommandQueue() else {
            Issue.record("no command queue")
            return
        }

        // Input vector for sigmoid: 256 floats covering a range that
        // makes correctness easy to verify.
        let n = 256
        let inputHost: [Float] = (0..<n).map { Float($0) * 0.05 - 6.0 }  // -6 .. ~6.8
        let inBytes = n * MemoryLayout<Float>.stride
        guard
            let inBuf = device.makeBuffer(bytes: inputHost, length: inBytes, options: .storageModeShared),
            let outDirect = device.makeBuffer(length: inBytes, options: .storageModeShared),
            let outIcb = device.makeBuffer(length: inBytes, options: .storageModeShared)
        else {
            Issue.record("buffer alloc")
            return
        }

        // PSO via the cache (uses descriptor + supportIndirectCommandBuffers).
        let pso = PSOCache.shared.pipelineState(for: "mt_sigmoid_f32")
        #expect(pso.maxTotalThreadsPerThreadgroup > 0,
                "PSO instantiated for ICB-supporting compile")

        let tgWidth = min(n, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)

        // ── Direct dispatch (reference path) ──────────────────────────
        do {
            guard let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                Issue.record("direct encoder")
                return
            }
            enc.setComputePipelineState(pso)
            enc.setBuffer(inBuf, offset: 0, index: 0)
            enc.setBuffer(outDirect, offset: 0, index: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // ── ICB-recorded dispatch ─────────────────────────────────────
        let descriptor = MTLIndirectCommandBufferDescriptor()
        descriptor.commandTypes = .concurrentDispatch
        descriptor.inheritBuffers = false
        descriptor.maxKernelBufferBindCount = 4  // we bind 2; cushion
        descriptor.inheritPipelineState = false
        guard let icb = device.makeIndirectCommandBuffer(
            descriptor: descriptor, maxCommandCount: 1,
            options: .storageModeShared) else {
            Issue.record("ICB alloc")
            return
        }
        let icbCmd = icb.indirectComputeCommandAt(0)
        icbCmd.setComputePipelineState(pso)
        icbCmd.setKernelBuffer(inBuf, offset: 0, at: 0)
        icbCmd.setKernelBuffer(outIcb, offset: 0, at: 1)
        icbCmd.concurrentDispatchThreads(grid, threadsPerThreadgroup: tg)

        do {
            guard let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                Issue.record("ICB encoder")
                return
            }
            // Resources referenced by the ICB must be marked usable —
            // Metal can't see through ICB bindings to track residency.
            enc.useResource(inBuf, usage: .read)
            enc.useResource(outIcb, usage: .write)
            enc.executeCommandsInBuffer(icb, range: 0..<1)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // ── Compare ───────────────────────────────────────────────────
        let directPtr = outDirect.contents().bindMemory(to: Float.self, capacity: n)
        let icbPtr = outIcb.contents().bindMemory(to: Float.self, capacity: n)
        var maxAbsDiff: Float = 0
        for i in 0..<n {
            maxAbsDiff = max(maxAbsDiff, abs(directPtr[i] - icbPtr[i]))
            // Spot-check correctness against `1 / (1 + exp(-x))`.
            if i == 0 || i == n - 1 || i == n / 2 {
                let ref = 1.0 / (1.0 + Foundation.exp(-Double(inputHost[i])))
                #expect(abs(Double(icbPtr[i]) - ref) < 1e-5,
                        "ICB sigmoid[\(i)] = \(icbPtr[i]); reference = \(ref)")
            }
        }
        #expect(maxAbsDiff == 0, "ICB output exactly matches direct dispatch (max |Δ| = \(maxAbsDiff))")
        print("ICB smoke: n=\(n), max |Δ| ICB vs direct = \(maxAbsDiff) (zero is expected)")
    }

    @Test("ICB-replay perf — N=600 chained sigmoid dispatches via ICB vs direct encoder")
    func icbReplayPerfVsDirectEncoder() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("no Metal device")
            return
        }
        guard let queue = device.makeCommandQueue() else {
            Issue.record("no command queue")
            return
        }

        // 600 dispatches matches Qwen3.6-A3B decode T=1 (40 layers × ~15
        // kernels each). Each dispatch is on a tiny vector so GPU work
        // is in microseconds — the per-dispatch overhead dominates,
        // mirroring the decode bottleneck.
        let nCommands = 600
        let vecSize = 256
        let bytes = vecSize * MemoryLayout<Float>.stride
        let inputHost: [Float] = (0..<vecSize).map { Float($0) * 0.01 }
        guard
            let inBuf = device.makeBuffer(bytes: inputHost, length: bytes,
                                          options: .storageModeShared),
            let outBuf = device.makeBuffer(length: bytes, options: .storageModeShared)
        else {
            Issue.record("buffer alloc")
            return
        }

        let pso = PSOCache.shared.pipelineState(for: "mt_sigmoid_f32")
        let tgWidth = min(vecSize, 256)
        let grid = MTLSize(width: vecSize, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)

        // ── Warm-up both paths so PSO JIT + queue priming don't skew timing.
        for _ in 0..<3 {
            guard let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pso)
            enc.setBuffer(inBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            for _ in 0..<10 {
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            }
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // ── Direct encoder path — N dispatches on one encoder, one commit.
        var directTimes: [Double] = []
        for _ in 0..<5 {
            guard let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pso)
            enc.setBuffer(inBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            let t0 = Date()
            for _ in 0..<nCommands {
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            }
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            directTimes.append(Date().timeIntervalSince(t0))
        }
        directTimes.sort()
        let directMedian = directTimes[directTimes.count / 2]

        // ── ICB path — record N commands once, execute via single
        // encoder.executeCommandsInBuffer.
        let icbDescriptor = MTLIndirectCommandBufferDescriptor()
        icbDescriptor.commandTypes = .concurrentDispatch
        icbDescriptor.inheritBuffers = false
        icbDescriptor.maxKernelBufferBindCount = 4
        icbDescriptor.inheritPipelineState = false
        guard let icb = device.makeIndirectCommandBuffer(
            descriptor: icbDescriptor, maxCommandCount: nCommands,
            options: .storageModeShared) else {
            Issue.record("ICB alloc")
            return
        }
        for i in 0..<nCommands {
            let c = icb.indirectComputeCommandAt(i)
            c.setComputePipelineState(pso)
            c.setKernelBuffer(inBuf, offset: 0, at: 0)
            c.setKernelBuffer(outBuf, offset: 0, at: 1)
            c.concurrentDispatchThreads(grid, threadsPerThreadgroup: tg)
        }

        var icbTimes: [Double] = []
        for _ in 0..<5 {
            guard let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else { return }
            enc.useResource(inBuf, usage: .read)
            enc.useResource(outBuf, usage: .write)
            let t0 = Date()
            enc.executeCommandsInBuffer(icb, range: 0..<nCommands)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            icbTimes.append(Date().timeIntervalSince(t0))
        }
        icbTimes.sort()
        let icbMedian = icbTimes[icbTimes.count / 2]

        let directMs = directMedian * 1000
        let icbMs = icbMedian * 1000
        let speedup = directMedian / icbMedian
        print("ICB perf N=\(nCommands): direct=\(String(format: "%.3f", directMs))ms icb=\(String(format: "%.3f", icbMs))ms speedup=\(String(format: "%.2f", speedup))x")
        print("ICB perf N=\(nCommands): direct runs ms=\(directTimes.map { String(format: "%.3f", $0 * 1000) })")
        print("ICB perf N=\(nCommands): icb    runs ms=\(icbTimes.map { String(format: "%.3f", $0 * 1000) })")
        // Don't assert a specific speedup — Metal's behaviour at this
        // dispatch density depends on scheduler / thermals. The print
        // is the evidence.
    }
}
