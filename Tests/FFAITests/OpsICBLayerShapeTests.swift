// OpsICB layer-shape soak — record a realistic decode-layer's worth of
// dispatches (RMSNorm + multiple int4 qgemvs + sigmoid·mul) into an
// ICB, replay it, and check output + speedup against a direct-encoder
// path.
//
// Mimics the dispatch shape of `Qwen35DenseMLP.forward` /
// `qwen35FinalNormLmHead` etc. — validates the ICB Day 2 plumbing
// works for non-trivial decode-shape workloads. Real
// `Qwen35Model.forwardICB` wiring is the next iter.

import Foundation
import Metal
import Testing
@testable import FFAI
@testable import MetalTileSwift

@Suite("OpsICB layer-shape ICB soak")
struct OpsICBLayerShapeTests {

    @Test("RMSNorm + 3× int4 qgemv records into ICB + matches direct path")
    func mlpShapeEquivalence() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("no Metal device"); return
        }
        guard let queue = device.makeCommandQueue() else {
            Issue.record("no command queue"); return
        }
        // MLP-block shape: hidden=2048, intermediate=4096, group=64.
        let hidden = 2048, inter = 4096, groupSize = 64
        let dtype: DType = .bf16

        let fixture = makeMLPFixture(hidden: hidden, inter: inter,
                                     groupSize: groupSize, dtype: dtype)

        // ── Direct path: build the dispatch chain into a normal cmd buf ──
        let directOutNorm = Tensor.empty(shape: [hidden], dtype: dtype)
        let directGate = Tensor.empty(shape: [inter], dtype: dtype)
        let directUp = Tensor.empty(shape: [inter], dtype: dtype)
        let directDown = Tensor.empty(shape: [hidden], dtype: dtype)
        let cmdDirect = Device.shared.makeCommandBuffer()
        let normedDirect = Ops.rmsNorm(fixture.input,
                                        weight: fixture.normWeight,
                                        eps: 1e-6, on: cmdDirect,
                                        into: directOutNorm)
        _ = Ops.dequantGemvInt4(weight: fixture.gateW, scales: fixture.gateS,
                                 biases: fixture.gateB,
                                 input: normedDirect, groupSize: groupSize,
                                 on: cmdDirect, into: directGate)
        _ = Ops.dequantGemvInt4(weight: fixture.upW, scales: fixture.upS,
                                 biases: fixture.upB,
                                 input: normedDirect, groupSize: groupSize,
                                 on: cmdDirect, into: directUp)
        _ = Ops.dequantGemvInt4(weight: fixture.downW, scales: fixture.downS,
                                 biases: fixture.downB,
                                 input: directGate, groupSize: groupSize,
                                 on: cmdDirect, into: directDown)
        cmdDirect.commit(); cmdDirect.waitUntilCompleted()

        // ── ICB path: same chain, recorded into ICB, executed once ──
        let icbOutNorm = Tensor.empty(shape: [hidden], dtype: dtype)
        let icbGate = Tensor.empty(shape: [inter], dtype: dtype)
        let icbUp = Tensor.empty(shape: [inter], dtype: dtype)
        let icbDown = Tensor.empty(shape: [hidden], dtype: dtype)
        let paramsBudget = 4 * 64  // 4 dispatches, ≤64 bytes each
        let rec = ICBRecorder(device: device,
                              maxCommands: 4,
                              paramsBytes: paramsBudget)
        OpsICB.rmsNorm(fixture.input, weight: fixture.normWeight,
                       epsBuf: fixture.epsBuf, into: icbOutNorm,
                       recorder: rec)
        rec.groupBoundary()  // gate/up depend on norm output
        OpsICB.dequantGemvInt4(weight: fixture.gateW, scales: fixture.gateS,
                                biases: fixture.gateB, input: icbOutNorm,
                                into: icbGate, groupSize: groupSize,
                                recorder: rec)
        OpsICB.dequantGemvInt4(weight: fixture.upW, scales: fixture.upS,
                                biases: fixture.upB, input: icbOutNorm,
                                into: icbUp, groupSize: groupSize,
                                recorder: rec)
        rec.groupBoundary()  // down depends on gate output
        OpsICB.dequantGemvInt4(weight: fixture.downW, scales: fixture.downS,
                                biases: fixture.downB, input: icbGate,
                                into: icbDown, groupSize: groupSize,
                                recorder: rec)
        let cmdICB = queue.makeCommandBuffer()!
        rec.execute(on: cmdICB)
        cmdICB.commit(); cmdICB.waitUntilCompleted()

        // ── Equivalence ──
        func absEq(_ a: Tensor, _ b: Tensor, _ name: String) {
            let aArr = a.toFloatArray(); let bArr = b.toFloatArray()
            var maxDiff: Float = 0
            for i in 0..<aArr.count {
                let d = abs(aArr[i] - bArr[i]); if d > maxDiff { maxDiff = d }
            }
            print("[\(name)] max |Δ| = \(maxDiff)")
            #expect(maxDiff == 0, "\(name): ICB diverged from direct")
        }
        absEq(directOutNorm, icbOutNorm, "rmsNorm")
        absEq(directGate, icbGate, "gate qgemv")
        absEq(directUp, icbUp, "up qgemv")
        absEq(directDown, icbDown, "down qgemv")
    }

    @Test("ICB-replay perf: 40 layers' worth of MLP chains vs direct rebound")
    func layerStackSpeedup() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("no Metal device"); return
        }
        guard let queue = device.makeCommandQueue() else {
            Issue.record("no command queue"); return
        }
        let hidden = 2048, inter = 4096, groupSize = 64
        let dtype: DType = .bf16
        let nLayers = 40  // Qwen3.6-A3B layer count
        let fixture = makeMLPFixture(hidden: hidden, inter: inter,
                                     groupSize: groupSize, dtype: dtype)
        let outNorm = Tensor.empty(shape: [hidden], dtype: dtype)
        let outGate = Tensor.empty(shape: [inter], dtype: dtype)
        let outUp = Tensor.empty(shape: [inter], dtype: dtype)
        let outDown = Tensor.empty(shape: [hidden], dtype: dtype)

        // Record 40 layers × 4 dispatches = 160 commands.
        let nCmds = nLayers * 4
        let paramsBudget = nCmds * 32
        let rec = ICBRecorder(device: device,
                              maxCommands: nCmds, paramsBytes: paramsBudget)
        for _ in 0..<nLayers {
            OpsICB.rmsNorm(fixture.input, weight: fixture.normWeight,
                            epsBuf: fixture.epsBuf, into: outNorm,
                            recorder: rec)
            rec.groupBoundary()
            OpsICB.dequantGemvInt4(weight: fixture.gateW, scales: fixture.gateS,
                                    biases: fixture.gateB, input: outNorm,
                                    into: outGate, groupSize: groupSize,
                                    recorder: rec)
            OpsICB.dequantGemvInt4(weight: fixture.upW, scales: fixture.upS,
                                    biases: fixture.upB, input: outNorm,
                                    into: outUp, groupSize: groupSize,
                                    recorder: rec)
            rec.groupBoundary()
            OpsICB.dequantGemvInt4(weight: fixture.downW, scales: fixture.downS,
                                    biases: fixture.downB, input: outGate,
                                    into: outDown, groupSize: groupSize,
                                    recorder: rec)
            rec.groupBoundary()
        }
        // Warm up.
        do {
            let cmd = queue.makeCommandBuffer()!
            rec.execute(on: cmd)
            cmd.commit(); cmd.waitUntilCompleted()
        }
        // Time 5 ICB replays.
        var icbTimes: [Double] = []
        for _ in 0..<5 {
            let cmd = queue.makeCommandBuffer()!
            let t0 = Date()
            rec.execute(on: cmd)
            cmd.commit(); cmd.waitUntilCompleted()
            icbTimes.append(Date().timeIntervalSince(t0))
        }
        icbTimes.sort()
        let icbMedian = icbTimes[2]

        // Time 5 direct-encoder rebound runs (one fresh encoder per dispatch).
        var directTimes: [Double] = []
        for _ in 0..<5 {
            let cmd = Device.shared.makeCommandBuffer()
            let t0 = Date()
            for _ in 0..<nLayers {
                let normed = Ops.rmsNorm(fixture.input,
                                          weight: fixture.normWeight,
                                          eps: 1e-6, on: cmd, into: outNorm)
                _ = Ops.dequantGemvInt4(weight: fixture.gateW, scales: fixture.gateS,
                                         biases: fixture.gateB,
                                         input: normed, groupSize: groupSize,
                                         on: cmd, into: outGate)
                _ = Ops.dequantGemvInt4(weight: fixture.upW, scales: fixture.upS,
                                         biases: fixture.upB,
                                         input: normed, groupSize: groupSize,
                                         on: cmd, into: outUp)
                _ = Ops.dequantGemvInt4(weight: fixture.downW, scales: fixture.downS,
                                         biases: fixture.downB,
                                         input: outGate, groupSize: groupSize,
                                         on: cmd, into: outDown)
            }
            cmd.commit(); cmd.waitUntilCompleted()
            directTimes.append(Date().timeIntervalSince(t0))
        }
        directTimes.sort()
        let directMedian = directTimes[2]
        let speedup = directMedian / icbMedian
        print("Layer-stack N=\(nCmds) cmds: " +
              "ICB median=\(String(format: "%.3f", icbMedian * 1000))ms, " +
              "direct median=\(String(format: "%.3f", directMedian * 1000))ms, " +
              "speedup=\(String(format: "%.2fx", speedup))")
        // Informational only — no speedup assertion. At qgemv-heavy
        // workloads the kernel time dominates and barrier-induced
        // serialization makes ICB slower than direct rebound. The win
        // pattern is many small dispatches with low GPU work per
        // dispatch (silu / sigmoid chains, indexed mul, etc.) where
        // host encoder overhead dominates. Keep the test to monitor
        // the ratio across iterations.
    }

    // ── Helpers ─────────────────────────────────────────────────────

    struct MLPFixture {
        let input: Tensor, normWeight: Tensor, epsBuf: Tensor
        let gateW: Tensor, gateS: Tensor, gateB: Tensor
        let upW: Tensor, upS: Tensor, upB: Tensor
        let downW: Tensor, downS: Tensor, downB: Tensor
    }

    private func makeMLPFixture(hidden: Int, inter: Int,
                                 groupSize: Int, dtype: DType) -> MLPFixture {
        var seed: UInt64 = 0xC0DE_F00D
        @inline(__always) func xs() -> UInt32 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return UInt32(truncatingIfNeeded: seed)
        }
        @inline(__always) func rsmall() -> Float {
            Float(Int32(truncatingIfNeeded: xs())) / Float(Int32.max) * 0.2
        }
        let input = Tensor.empty(shape: [hidden], dtype: dtype)
        Self.writeF32(input, (0..<hidden).map { _ in rsmall() }, dtype: dtype)
        let normWeight = Tensor.empty(shape: [hidden], dtype: dtype)
        Self.writeF32(normWeight, (0..<hidden).map { _ in 1.0 + rsmall() * 0.1 },
                      dtype: dtype)
        let epsBuf = Tensor.empty(shape: [1], dtype: .f32)
        epsBuf.copyIn(from: [Float(1e-6)])

        func makeWeights(_ outDim: Int, _ inDim: Int)
            -> (w: Tensor, s: Tensor, b: Tensor)
        {
            let packedPerRow = inDim / 8
            let nGroups = inDim / groupSize
            let w = Tensor.empty(shape: [outDim, packedPerRow], dtype: .u32)
            var wBytes = [UInt32](); wBytes.reserveCapacity(outDim * packedPerRow)
            for _ in 0..<(outDim * packedPerRow) { wBytes.append(xs()) }
            w.copyIn(from: wBytes)
            let s = Tensor.empty(shape: [outDim, nGroups], dtype: dtype)
            Self.writeF32(s,
                          (0..<(outDim * nGroups)).map { _ in rsmall() + 0.05 },
                          dtype: dtype)
            let b = Tensor.empty(shape: [outDim, nGroups], dtype: dtype)
            Self.writeF32(b,
                          (0..<(outDim * nGroups)).map { _ in rsmall() },
                          dtype: dtype)
            return (w, s, b)
        }
        let g = makeWeights(inter, hidden)
        let u = makeWeights(inter, hidden)
        let d = makeWeights(hidden, inter)
        return MLPFixture(input: input, normWeight: normWeight, epsBuf: epsBuf,
                          gateW: g.w, gateS: g.s, gateB: g.b,
                          upW: u.w, upS: u.s, upB: u.b,
                          downW: d.w, downS: d.s, downB: d.b)
    }

    private static func writeF32(_ t: Tensor, _ src: [Float], dtype: DType) {
        switch dtype {
        case .f32: t.copyIn(from: src)
        case .f16: t.copyIn(from: src.map { Float16($0) })
        case .bf16: t.copyIn(from: src.map { UInt16($0.bitPattern >> 16) })
        default: preconditionFailure("unsupported dtype")
        }
    }
}
