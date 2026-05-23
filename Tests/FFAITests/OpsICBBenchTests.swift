// OpsICB bench — measures Ops-layer ICB recording vs direct dispatch.
//
// Builds a synthetic chain mimicking decode hot-path shape (alternating
// silu / sigmoid / mul / silu / ...) at hidden=2048 (~Qwen3.6-A3B
// scale) and runs it both ways:
//   * Direct: 1× Ops dispatch per element per iteration → N iters.
//   * ICB:    record once, execute N times via ICBRecorder.
//
// Reports per-iter wall + ratio. Also bit-identical check on iter 1
// output between paths.

import Foundation
import XCTest
import Metal
import MetalTileSwift
@testable import FFAI

final class OpsICBBenchTests: XCTestCase {

    func testOpsICBChainMatchesDirect() {
        let device = Device.shared
        let hidden = 2048
        let nDispatches = 600   // matches Qwen3.6-A3B decode dispatch count
        let iters = 50

        // Pre-allocate ping-pong buffers a / b. Chain alternates which
        // is read vs written. f32 throughout (mul kernel only emits f32).
        let a = Tensor.empty(shape: [hidden], dtype: .f32, device: device)
        let b = Tensor.empty(shape: [hidden], dtype: .f32, device: device)
        let c = Tensor.empty(shape: [hidden], dtype: .f32, device: device)

        // Init `a` with a fixed seed (deterministic compare).
        var hostInit = [Float](repeating: 0, count: hidden)
        for i in 0..<hidden { hostInit[i] = Float(i % 17) * 0.01 - 0.05 }
        a.buffer.contents()
            .advanced(by: a.offset)
            .copyMemory(from: &hostInit, byteCount: hidden * 4)

        // ──────────── Direct dispatch reference ────────────
        // Build one canonical iteration that mutates a→b→c→a→b→...
        // Pattern per "step":  c = silu(a) * sigmoid(b) ; rotate → b=c.
        // Each step is 3 dispatches. 600 / 3 = 200 steps per iter.
        let stepsPerIter = nDispatches / 3
        precondition(stepsPerIter * 3 == nDispatches,
                     "test invariant: nDispatches must be divisible by 3")

        @inline(__always)
        func directIter() {
            let cmd = device.makeCommandBuffer()
            for _ in 0..<stepsPerIter {
                _ = Ops.silu(a, on: cmd, into: c)             // c = silu(a)
                _ = Ops.sigmoid(b, on: cmd, into: a)          // a = sigmoid(b)
                // mul: b = c * a
                MetalTileKernels.mt_mul_f32(
                    a: c.buffer, aOffset: c.offset,
                    b: a.buffer, bOffset: a.offset,
                    out: b.buffer, outOffset: b.offset,
                    gridSize: MTLSize(width: hidden, height: 1, depth: 1),
                    threadgroupSize: MTLSize(width: min(1024, hidden), height: 1, depth: 1),
                    on: cmd)
            }
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // Warm + reference output.
        // Reseed a (since iters mutate it; we want bit-identical reference).
        a.buffer.contents().advanced(by: a.offset)
            .copyMemory(from: &hostInit, byteCount: hidden * 4)
        // Zero b (start state).
        var zero = [Float](repeating: 0, count: hidden)
        b.buffer.contents().advanced(by: b.offset)
            .copyMemory(from: &zero, byteCount: hidden * 4)
        directIter()  // warm
        // Capture reference after warm.
        let refA = a.toFloatArray()
        let refB = b.toFloatArray()

        // Direct timing.
        // Reset state.
        a.buffer.contents().advanced(by: a.offset)
            .copyMemory(from: &hostInit, byteCount: hidden * 4)
        b.buffer.contents().advanced(by: b.offset)
            .copyMemory(from: &zero, byteCount: hidden * 4)
        let dT0 = Date()
        for _ in 0..<iters { directIter() }
        let directS = Date().timeIntervalSince(dT0)

        // ──────────── ICB record + replay ────────────
        let recorder = ICBRecorder(
            device: device.mtlDevice,
            maxCommands: nDispatches + 16,
            paramsBytes: 4096,
            maxKernelBufferBindCount: 8)
        // Record one canonical iteration.
        for _ in 0..<stepsPerIter {
            OpsICB.silu(a, into: c, recorder: recorder)
            OpsICB.sigmoid(b, into: a, recorder: recorder)
            OpsICB.mul(c, a, into: b, recorder: recorder)
        }

        // Warm + correctness compare. Reset state.
        a.buffer.contents().advanced(by: a.offset)
            .copyMemory(from: &hostInit, byteCount: hidden * 4)
        b.buffer.contents().advanced(by: b.offset)
            .copyMemory(from: &zero, byteCount: hidden * 4)
        let warmCmd = device.makeCommandBuffer()
        recorder.execute(on: warmCmd)
        warmCmd.commit()
        warmCmd.waitUntilCompleted()
        let icbA = a.toFloatArray()
        let icbB = b.toFloatArray()
        // Bit-identical comparison.
        var maxDelta: Float = 0
        for i in 0..<hidden {
            maxDelta = max(maxDelta, abs(refA[i] - icbA[i]))
            maxDelta = max(maxDelta, abs(refB[i] - icbB[i]))
        }
        print("OpsICB chain max|Δ| vs direct: \(maxDelta)")
        XCTAssertLessThanOrEqual(maxDelta, 1e-5,
                                  "OpsICB chain diverged from direct path")

        // ICB timing.
        a.buffer.contents().advanced(by: a.offset)
            .copyMemory(from: &hostInit, byteCount: hidden * 4)
        b.buffer.contents().advanced(by: b.offset)
            .copyMemory(from: &zero, byteCount: hidden * 4)
        let iT0 = Date()
        for _ in 0..<iters {
            let cmd = device.makeCommandBuffer()
            recorder.execute(on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let icbS = Date().timeIntervalSince(iT0)

        let directPerIterMs = directS / Double(iters) * 1000
        let icbPerIterMs = icbS / Double(iters) * 1000
        let speedup = directS / icbS
        print("OpsICB bench (\(nDispatches) dispatches × \(iters) iters @ hidden=\(hidden)):")
        print("  direct:  \(String(format: "%.3f", directS))s total,  \(String(format: "%.3f", directPerIterMs)) ms/iter")
        print("  icb:     \(String(format: "%.3f", icbS))s total,  \(String(format: "%.3f", icbPerIterMs)) ms/iter")
        print("  speedup: \(String(format: "%.2fx", speedup))")
        print("  → estimated decode overhead saved per token: \(String(format: "%.2f", directPerIterMs - icbPerIterMs)) ms")
    }
}
