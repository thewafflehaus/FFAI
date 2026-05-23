// MoERouter GPU kernel (`mt_moe_router_topk`) — correctness tests vs
// the CPU `MoERouter.route` reference. The GPU kernel is the entry
// point of ITER 56's GPU MoE router path (`FFAI_MOE_GPU_ROUTER=1`);
// these tests pin down its mode-1 (Qwen3-MoE: softmax-over-chosen-k,
// sum-to-1) and mode-0 (Qwen3-Next: raw softmax probs) semantics
// independently of any model load.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops.moeRouterTopK — GPU vs CPU correctness")
struct MoERouterGPUTests {

    /// Submit + wait a no-op buffer to flush the shared queue before
    /// host readback. Matches `MoELayerTests.flushQueue` convention.
    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    /// Run `Ops.moeRouterTopK` with `dtype` activation, return the
    /// (indices, weights) the GPU wrote.
    private static func runGPU(
        logitsArr: [Float], nExperts: Int, k: Int,
        normTopkProb: Bool, dtype: DType
    ) -> (indices: [UInt32], weights: [Float]) {
        let logits = Tensor.empty(shape: [nExperts], dtype: dtype)
        switch dtype {
        case .f32: logits.copyIn(from: logitsArr)
        case .f16: logits.copyIn(from: logitsArr.map { Float16($0) })
        case .bf16:
            // bf16: take high 16 bits of the f32 bit pattern.
            logits.copyIn(from: logitsArr.map { UInt16($0.bitPattern >> 16) })
        default: preconditionFailure("unsupported logits dtype for test")
        }
        let indicesOut = Tensor.empty(shape: [k], dtype: .u32)
        let weightsOut = Tensor.empty(shape: [k], dtype: dtype)
        let cmd = Device.shared.makeCommandBuffer()
        Ops.moeRouterTopK(
            logits: logits, indicesOut: indicesOut, weightsOut: weightsOut,
            nExperts: nExperts, k: k, normTopkProb: normTopkProb, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        flushQueue()
        let idx = indicesOut.toArray(as: UInt32.self)
        let w = weightsOut.toFloatArray()
        return (idx, w)
    }

    /// Hand-computed Qwen3-MoE-style routing (mode 1) for tiny shape.
    /// nExperts=4, k=2, logits=[1, 3, 2, 0]; chosen={1,2}; softmax
    /// over chosen→ sum-to-1.
    @Test("mode 1: 4-expert top-2 indices + weights match CPU")
    func mode1TinyMatchesCpu() {
        let logits: [Float] = [1, 3, 2, 0]
        let cpu = MoERouter(nExperts: 4, topK: 2,
                            gatingMode: .softmaxThenTopK, normTopKProb: true)
            .route(logits: logits)
        let gpu = Self.runGPU(logitsArr: logits, nExperts: 4, k: 2,
                              normTopkProb: true, dtype: .f32)
        #expect(gpu.indices.map { Int($0) } == cpu.indices)
        for i in 0..<cpu.weights.count {
            #expect(abs(gpu.weights[i] - cpu.weights[i]) < 1e-4)
        }
        // Mode 1 invariant: chosen-k weights sum to 1.
        let sum = gpu.weights.reduce(0, +)
        #expect(abs(sum - 1) < 1e-4)
    }

    /// Mode 0 (Qwen3-Next): same inputs, weights are raw softmax probs
    /// of the picked experts, which sum to LESS than 1.
    @Test("mode 0: weights are raw softmax probs, sum < 1")
    func mode0TinyMatchesCpu() {
        let logits: [Float] = [1, 3, 2, 0]
        let cpu = MoERouter(nExperts: 4, topK: 2,
                            gatingMode: .softmaxThenTopK, normTopKProb: false)
            .route(logits: logits)
        let gpu = Self.runGPU(logitsArr: logits, nExperts: 4, k: 2,
                              normTopkProb: false, dtype: .f32)
        #expect(gpu.indices.map { Int($0) } == cpu.indices)
        for i in 0..<cpu.weights.count {
            #expect(abs(gpu.weights[i] - cpu.weights[i]) < 1e-4)
        }
        let sum = gpu.weights.reduce(0, +)
        #expect(sum < 0.95, "mode 0 weights must sum to <1, got \(sum)")
    }

    /// Qwen3.6-A3B production shape: nExperts=128, k=8, mode 1.
    /// Random logits — argmax-pick must match CPU; weight values
    /// within softmax-precision tolerance.
    @Test("Qwen3.6 shape (nExperts=128, k=8) mode 1 matches CPU on random logits")
    func qwen36ShapeRandomMatchesCpu() {
        let nExperts = 128, k = 8
        var rng = SystemRandomNumberGenerator()
        var logits = [Float]()
        logits.reserveCapacity(nExperts)
        for _ in 0..<nExperts {
            // Logits roughly in MoE gate range; scaled so winners separate
            // cleanly + softmax doesn't underflow.
            let u = Float(rng.next()) / Float(UInt64.max)
            logits.append(u * 6 - 3)  // [-3, 3]
        }
        let cpu = MoERouter(nExperts: nExperts, topK: k,
                            gatingMode: .softmaxThenTopK, normTopKProb: true)
            .route(logits: logits)
        let gpu = Self.runGPU(logitsArr: logits, nExperts: nExperts, k: k,
                              normTopkProb: true, dtype: .f32)
        #expect(gpu.indices.map { Int($0) } == cpu.indices,
                "GPU/CPU index mismatch")
        for i in 0..<k {
            #expect(abs(gpu.weights[i] - cpu.weights[i]) < 5e-4,
                    "weight[\(i)] gpu=\(gpu.weights[i]) cpu=\(cpu.weights[i])")
        }
        // sum-to-1 invariant
        let sum = gpu.weights.reduce(0, +)
        #expect(abs(sum - 1) < 5e-4)
    }

    /// f16 dtype path: same kernel template, smaller activation
    /// precision. Indices must still match; weights within 1e-3 (f16
    /// has ~3 decimal digits).
    @Test("f16 dtype indices match CPU, weights within f16 tolerance")
    func f16IndicesMatchWeightsApproximate() {
        let nExperts = 128, k = 8
        var logits = [Float]()
        for i in 0..<nExperts {
            // Deterministic pattern — winners at known positions
            logits.append(Float((i * 37) % 17) - 8)
        }
        let cpu = MoERouter(nExperts: nExperts, topK: k,
                            gatingMode: .softmaxThenTopK, normTopKProb: true)
            .route(logits: logits)
        let gpu = Self.runGPU(logitsArr: logits, nExperts: nExperts, k: k,
                              normTopkProb: true, dtype: .f16)
        #expect(gpu.indices.map { Int($0) } == cpu.indices)
        for i in 0..<k {
            #expect(abs(gpu.weights[i] - cpu.weights[i]) < 1e-3)
        }
    }
}
