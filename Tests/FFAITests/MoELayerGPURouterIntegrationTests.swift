// ITER 72 (Bagel 2): end-to-end correctness test for the GPU MoE router
// branch (ITER 56) in `MoELayer.decode`. Constructs a tiny synthetic
// MoELayer with random int4 weights replicated across per-expert
// linears AND a matching `StackedInt4Experts`. Runs decode with the
// env flag OFF (CPU-sync path, the existing battle-tested
// implementation) and ON (GPU router branch). Compares outputs
// element-wise.
//
// This is the validation gate that proves the GPU router wiring (router
// topK + 16 indexed gate+up qmms + swigluMany + 8 indexed down qmms +
// chain8) produces mathematically equivalent output to the legacy CPU
// route + per-expert qmm chain. Without it, ITER 56 is wire-correct at
// the kernel level (kernels pass their own tests) but the orchestration
// in `MoELayer.decode` could still have a wiring bug — e.g. wrong
// `slot * h.dtype.byteSize` arithmetic on `routerWeightsScratch`, or
// mis-ordering of gate/up/down across the Many calls.
//
// Memory: ~50KB total (hidden=32, moeIntermediate=32, nExperts=8,
// groupSize=32, bf16 dtype) — safe to run alongside other tests.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("MoELayer.decode — GPU router branch matches CPU-sync output")
struct MoELayerGPURouterIntegrationTests {

    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    @Test("bf16 hidden=32 moeIntermediate=32: GPU router output matches CPU-sync within tol")
    func gpuRouterMatchesCpuSyncBf16() {
        runCase(dtype: .bf16, tolerance: 5e-2)
    }

    @Test("f16 hidden=32 moeIntermediate=32: GPU router output matches CPU-sync within tol")
    func gpuRouterMatchesCpuSyncF16() {
        runCase(dtype: .f16, tolerance: 1e-2)
    }

    @Test("f32 hidden=32 moeIntermediate=32: GPU router output matches CPU-sync within tol")
    func gpuRouterMatchesCpuSyncF32() {
        runCase(dtype: .f32, tolerance: 1e-3)
    }

    private func runCase(dtype: DType, tolerance: Float) {
        let nExperts = 8, topK = 8
        let hidden = 32, moeIntermediate = 32, groupSize = 32
        let packedPerRow = moeIntermediate / 8  // int4 → 8 vals/u32; here 4
        let packedPerRowDown = hidden / 8       // = 4
        let nGroupsGate = hidden / groupSize    // = 1
        let nGroupsDown = moeIntermediate / groupSize  // = 1

        // ─── 1. Generate random data for ALL expert weights ──────────
        // Stacked-shape data; per-expert linears slice into the same.
        // gate/up weights: [nExperts, moeIntermediate, hidden/8] u32
        // down weights:    [nExperts, hidden, moeIntermediate/8] u32
        var seed: UInt64 = 0xDEAD_BEEF_C0FFEE
        @inline(__always)
        func xorshift() -> UInt32 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return UInt32(truncatingIfNeeded: seed)
        }

        // Per-expert × per-proj packed u32 weights.
        // Layout in our raw buffers: linear array, expert-major.
        let gateWeightTotal = nExperts * moeIntermediate * packedPerRow
        let downWeightTotal = nExperts * hidden * packedPerRowDown
        var gateWBytes = [UInt32](); gateWBytes.reserveCapacity(gateWeightTotal)
        var upWBytes = [UInt32](); upWBytes.reserveCapacity(gateWeightTotal)
        var downWBytes = [UInt32](); downWBytes.reserveCapacity(downWeightTotal)
        for _ in 0..<gateWeightTotal { gateWBytes.append(xorshift()) }
        for _ in 0..<gateWeightTotal { upWBytes.append(xorshift()) }
        for _ in 0..<downWeightTotal { downWBytes.append(xorshift()) }

        // Scales/biases: small random in [-0.1, 0.1] to keep activations bounded.
        @inline(__always)
        func smallRandom() -> Float {
            Float(Int32(truncatingIfNeeded: xorshift())) / Float(Int32.max) * 0.1
        }
        let gateScaleBiasTotal = nExperts * moeIntermediate * nGroupsGate
        let downScaleBiasTotal = nExperts * hidden * nGroupsDown
        let gateScalesF32 = (0..<gateScaleBiasTotal).map { _ in smallRandom() + 0.05 }
        let gateBiasesF32 = (0..<gateScaleBiasTotal).map { _ in smallRandom() }
        let upScalesF32 = (0..<gateScaleBiasTotal).map { _ in smallRandom() + 0.05 }
        let upBiasesF32 = (0..<gateScaleBiasTotal).map { _ in smallRandom() }
        let downScalesF32 = (0..<downScaleBiasTotal).map { _ in smallRandom() + 0.05 }
        let downBiasesF32 = (0..<downScaleBiasTotal).map { _ in smallRandom() }

        // ─── 2. Build per-expert QuantizedLinear arrays ──────────────
        func makeQLinear(weightSlice: [UInt32], scalesSlice: [Float],
                         biasesSlice: [Float], outDim: Int, packed: Int,
                         nGroups: Int) -> QuantizedLinear {
            let w = Tensor.empty(shape: [outDim, packed], dtype: .u32)
            w.copyIn(from: weightSlice)
            let s = Tensor.empty(shape: [outDim, nGroups], dtype: dtype)
            Self.writeF32(s, scalesSlice, dtype: dtype)
            let b = Tensor.empty(shape: [outDim, nGroups], dtype: dtype)
            Self.writeF32(b, biasesSlice, dtype: dtype)
            return QuantizedLinear(weight: w, scales: s, biases: b,
                                    bits: 4, groupSize: groupSize)
        }

        var gateProj: [AnyLinear] = []
        var upProj: [AnyLinear] = []
        var downProj: [AnyLinear] = []
        for e in 0..<nExperts {
            let gWStart = e * moeIntermediate * packedPerRow
            let gWEnd = gWStart + moeIntermediate * packedPerRow
            let gSStart = e * moeIntermediate * nGroupsGate
            let gSEnd = gSStart + moeIntermediate * nGroupsGate
            let dWStart = e * hidden * packedPerRowDown
            let dWEnd = dWStart + hidden * packedPerRowDown
            let dSStart = e * hidden * nGroupsDown
            let dSEnd = dSStart + hidden * nGroupsDown
            gateProj.append(AnyLinear(makeQLinear(
                weightSlice: Array(gateWBytes[gWStart..<gWEnd]),
                scalesSlice: Array(gateScalesF32[gSStart..<gSEnd]),
                biasesSlice: Array(gateBiasesF32[gSStart..<gSEnd]),
                outDim: moeIntermediate, packed: packedPerRow,
                nGroups: nGroupsGate)))
            upProj.append(AnyLinear(makeQLinear(
                weightSlice: Array(upWBytes[gWStart..<gWEnd]),
                scalesSlice: Array(upScalesF32[gSStart..<gSEnd]),
                biasesSlice: Array(upBiasesF32[gSStart..<gSEnd]),
                outDim: moeIntermediate, packed: packedPerRow,
                nGroups: nGroupsGate)))
            downProj.append(AnyLinear(makeQLinear(
                weightSlice: Array(downWBytes[dWStart..<dWEnd]),
                scalesSlice: Array(downScalesF32[dSStart..<dSEnd]),
                biasesSlice: Array(downBiasesF32[dSStart..<dSEnd]),
                outDim: hidden, packed: packedPerRowDown,
                nGroups: nGroupsDown)))
        }

        // ─── 3. Build StackedInt4Experts with identical bytes ─────────
        let gateW = Tensor.empty(shape: [nExperts, moeIntermediate, packedPerRow], dtype: .u32)
        gateW.copyIn(from: gateWBytes)
        let upW = Tensor.empty(shape: [nExperts, moeIntermediate, packedPerRow], dtype: .u32)
        upW.copyIn(from: upWBytes)
        let downW = Tensor.empty(shape: [nExperts, hidden, packedPerRowDown], dtype: .u32)
        downW.copyIn(from: downWBytes)
        let gateS = Tensor.empty(shape: [nExperts, moeIntermediate, nGroupsGate], dtype: dtype)
        Self.writeF32(gateS, gateScalesF32, dtype: dtype)
        let gateB = Tensor.empty(shape: [nExperts, moeIntermediate, nGroupsGate], dtype: dtype)
        Self.writeF32(gateB, gateBiasesF32, dtype: dtype)
        let upS = Tensor.empty(shape: [nExperts, moeIntermediate, nGroupsGate], dtype: dtype)
        Self.writeF32(upS, upScalesF32, dtype: dtype)
        let upB = Tensor.empty(shape: [nExperts, moeIntermediate, nGroupsGate], dtype: dtype)
        Self.writeF32(upB, upBiasesF32, dtype: dtype)
        let downS = Tensor.empty(shape: [nExperts, hidden, nGroupsDown], dtype: dtype)
        Self.writeF32(downS, downScalesF32, dtype: dtype)
        let downB = Tensor.empty(shape: [nExperts, hidden, nGroupsDown], dtype: dtype)
        Self.writeF32(downB, downBiasesF32, dtype: dtype)
        let stacked = MoELayer.StackedInt4Experts(
            gateWeight: gateW, gateScales: gateS, gateBiases: gateB,
            upWeight: upW, upScales: upS, upBiases: upB,
            downWeight: downW, downScales: downS, downBiases: downB,
            numExperts: nExperts, moeIntermediate: moeIntermediate,
            hidden: hidden, groupSize: groupSize, dtype: dtype)

        // ─── 4. Gate router projection (hidden → nExperts) ───────────
        // Use a small dense Linear for the gate (not int4) — keeps the
        // test focused on the per-expert path. The MoELayer happily
        // wraps either kind in its `gate: AnyLinear`.
        let gateRouterW = Tensor.empty(shape: [nExperts, hidden], dtype: dtype)
        let gateRouterF32 = (0..<(nExperts * hidden)).map { _ in smallRandom() }
        Self.writeF32(gateRouterW, gateRouterF32, dtype: dtype)
        let routerGate = AnyLinear(Linear(weight: gateRouterW))

        // ─── 5. Two MoELayer instances differing only by env flag ────
        let router = MoERouter(nExperts: nExperts, topK: topK,
                                gatingMode: .softmaxThenTopK,
                                normTopKProb: true)
        // CPU-sync path baseline.
        unsetenv("FFAI_MOE_GPU_ROUTER")
        let layerCpu = MoELayer(gate: routerGate,
                                 gateProj: gateProj, upProj: upProj, downProj: downProj,
                                 router: router, hidden: hidden,
                                 stackedInt4Experts: stacked)
        // GPU router path.
        setenv("FFAI_MOE_GPU_ROUTER", "1", 1)
        let layerGpu = MoELayer(gate: routerGate,
                                 gateProj: gateProj, upProj: upProj, downProj: downProj,
                                 router: router, hidden: hidden,
                                 stackedInt4Experts: stacked)
        unsetenv("FFAI_MOE_GPU_ROUTER")

        // ─── 6. Forward both with same input, compare ─────────────────
        let h = Tensor.empty(shape: [hidden], dtype: dtype)
        let hF32 = (0..<hidden).map { _ in smallRandom() }
        Self.writeF32(h, hF32, dtype: dtype)

        let device = Device.shared
        let cmdCpu = device.makeCommandBuffer()
        let outCpu = layerCpu.decode(h, position: 0,
                                       cache: StatelessLayerCache(),
                                       cmd: cmdCpu, device: device)
        // `layerCpu.decode` commits its own cmd via internal `work` cmd
        // chain; we just need to flush before host read.
        Self.flushQueue()
        let cpuArr = outCpu.toFloatArray()

        let cmdGpu = device.makeCommandBuffer()
        let outGpu = layerGpu.decode(h, position: 0,
                                       cache: StatelessLayerCache(),
                                       cmd: cmdGpu, device: device)
        // `MoELayer.decode` commits `cmdGpu` internally (both CPU-sync
        // and GPU-router branches). A separate no-op flush is enough to
        // wait for everything-in-flight to complete before host read.
        Self.flushQueue()
        let gpuArr = outGpu.toFloatArray()

        // Compare.
        var maxDiff: Float = 0
        var firstFailIdx = -1
        for i in 0..<hidden {
            let d = abs(cpuArr[i] - gpuArr[i])
            if d > maxDiff { maxDiff = d }
            if d >= tolerance && firstFailIdx < 0 { firstFailIdx = i }
        }
        #expect(maxDiff < tolerance)
        if maxDiff >= tolerance {
            print("[\(dtype)] maxDiff=\(maxDiff) firstFail=\(firstFailIdx) " +
                  "cpu=\(firstFailIdx >= 0 ? cpuArr[firstFailIdx] : 0) " +
                  "gpu=\(firstFailIdx >= 0 ? gpuArr[firstFailIdx] : 0)")
        }
    }

    private static func writeF32(_ t: Tensor, _ src: [Float], dtype: DType) {
        switch dtype {
        case .f32: t.copyIn(from: src)
        case .f16: t.copyIn(from: src.map { Float16($0) })
        case .bf16:
            t.copyIn(from: src.map { UInt16($0.bitPattern >> 16) })
        default: preconditionFailure("unsupported dtype")
        }
    }
}
