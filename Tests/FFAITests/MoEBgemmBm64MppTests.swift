// FFAI wrapper correctness tests for `Ops.moeGatherDequantGemmInt4Bm64Mpp`.
//
// Background: the upstream metaltile kernel
// `mt_moe_gather_qmm_mma_int4_bm64_mpp_bf16` passes its own GPU correctness
// tests (incl. multi-n-tile / multi-K-block / group_size=64 cells we added
// upstream), but FFAI's `forwardManyEquivalence` Qwen3.6-A3B canary trips
// when the wrapper is engaged via `FFAI_MOE_BGEMM_BM64=1` (argmax 279 vs
// ref 52290). These tests recreate the upstream clean_tile + multi-tile
// shapes through the FFAI wrapper. If they pass, the drift in
// `forwardManyEquivalence` is a production-only issue (call-site shape,
// tensor offset). If they fail, the wrapper has a bug.

import Foundation
import Metal
import Testing
@testable import FFAI
import MetalTileSwift

@Suite("MoE bm64_mpp wrapper correctness")
struct MoEBgemmBm64MppTests {
    static func pack8(_ nibbles: [UInt32]) -> UInt32 {
        precondition(nibbles.count == 8)
        var word: UInt32 = 0
        for i in 0..<8 {
            word |= (nibbles[i] & 0xF) << (UInt32(i) * 4)
        }
        return word
    }

    static func packRow(_ row: [UInt32]) -> [UInt32] {
        precondition(row.count % 8 == 0)
        var out: [UInt32] = []
        out.reserveCapacity(row.count / 8)
        for i in stride(from: 0, to: row.count, by: 8) {
            out.append(pack8(Array(row[i..<i+8])))
        }
        return out
    }

    static func f32ToBf16(_ f: Float) -> UInt16 {
        let bits = f.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    static func bf16ToF32(_ b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }

    /// CPU scalar m1 oracle: per-row dequantized matmul against per-row
    /// expert. Returns Float row-major `[tRows, nOut]`. Runs in f32 — the
    /// most accurate reference available to compare both kernels against.
    /// Args mirror the kernel's buffer layout.
    static func m1Oracle(
        x: [Float], scales: [Float], biases: [Float], weight: [UInt32],
        indices: [UInt32], expertOffsets: [UInt32],
        tRows: Int, nOut: Int, kIn: Int, nExperts: Int, groupSize: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: tRows * nOut)
        let packsPerRow = kIn / 8
        let groupsPerRow = kIn / groupSize
        for r in 0..<tRows {
            let e = Int(indices[r])
            precondition(e < nExperts, "m1Oracle: indices[\(r)]=\(e) >= nExperts=\(nExperts)")
            let wExpertBase = e * nOut * packsPerRow
            let sbExpertBase = e * nOut * groupsPerRow
            for n in 0..<nOut {
                var acc: Float = 0
                for k in stride(from: 0, to: kIn, by: 8) {
                    let packDev = wExpertBase + n * packsPerRow + (k / 8)
                    let packed = weight[packDev]
                    let g = k / groupSize
                    let sbOff = sbExpertBase + n * groupsPerRow + g
                    let s = scales[sbOff]
                    let b = biases[sbOff]
                    for nibble in 0..<8 {
                        let q = Float((packed >> (UInt32(nibble) * 4)) & 0xF)
                        let w = s * q + b
                        acc += w * x[r * kIn + k + nibble]
                    }
                }
                out[r * nOut + n] = acc
            }
        }
        return out
    }

    /// Smooth sin/cos inputs matching the upstream Rust test pattern.
    /// Computes via Double then narrows — avoids the Swift type-inference
    /// crash on chained Float arithmetic inside closures.
    static func sinInputs(
        scaleAmp: Double, scaleOff: Double, scaleFreq: Double,
        biasAmp: Double, biasOff: Double, biasFreq: Double,
        xAmp: Double, xFreq: Double,
        groupsTotal: Int, xCount: Int
    ) -> (scales: [Float], biases: [Float], x: [Float]) {
        var scales = [Float](); scales.reserveCapacity(groupsTotal)
        var biases = [Float](); biases.reserveCapacity(groupsTotal)
        for i in 0..<groupsTotal {
            scales.append(Float(scaleOff + scaleAmp * Foundation.sin(Double(i) * scaleFreq)))
            biases.append(Float(biasOff + biasAmp * Foundation.cos(Double(i) * biasFreq)))
        }
        var x = [Float](); x.reserveCapacity(xCount)
        for i in 0..<xCount {
            x.append(Float(xAmp * Foundation.sin(Double(i) * xFreq)))
        }
        return (scales, biases, x)
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        precondition(a.count == b.count)
        var dot: Double = 0
        var na: Double = 0
        var nb: Double = 0
        for i in 0..<a.count {
            let x = Double(a[i])
            let y = Double(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        }
        return dot / (Foundation.sqrt(na) * Foundation.sqrt(nb) + 1e-12)
    }

    // Shared test body — driven by config. Keeping a single function body
    // sidesteps a Swift type-inference crash we hit on two near-identical
    // autoreleasepool blocks in this same file.
    static func runWrapperCompare(
        nExperts: Int, tRows: Int, nOut: Int, kIn: Int, groupSize: Int,
        seedW: Int, seedS: Int, seedB: Int, seedX: Int,
        sparseIndices: Bool = false,
        label: String
    ) -> Double {
        var indicesHost = [UInt32]()
        indicesHost.reserveCapacity(tRows)
        if sparseIndices {
            // Production-like: tRows < nExperts. Each row picks a different
            // expert id, sorted ascending. Skips most experts.
            for r in 0..<tRows {
                indicesHost.append(UInt32((r * (nExperts / max(tRows, 1))) % nExperts))
            }
            indicesHost.sort()
        } else {
            for r in 0..<tRows {
                indicesHost.append(UInt32(r / (tRows / nExperts)))
            }
        }

        let totalWeights = nExperts * nOut * kIn
        var weightUnpacked = [UInt32]()
        weightUnpacked.reserveCapacity(totalWeights)
        for i in 0..<totalWeights {
            weightUnpacked.append(UInt32((i * seedW + 3) & 0xF))
        }
        var weightPacked: [UInt32] = []
        weightPacked.reserveCapacity(totalWeights / 8)
        for r in stride(from: 0, to: totalWeights, by: kIn) {
            weightPacked.append(contentsOf: packRow(Array(weightUnpacked[r..<r+kIn])))
        }

        let groupsTotal = nExperts * nOut * (kIn / groupSize)
        var scales = [Float](); scales.reserveCapacity(groupsTotal)
        var biases = [Float](); biases.reserveCapacity(groupsTotal)
        for i in 0..<groupsTotal {
            let s: Float = 0.005 + 0.001 * Float((i * seedS) % 137) / 137.0
            let b: Float = -0.02 + 0.005 * Float((i * seedB) % 149) / 149.0
            scales.append(s)
            biases.append(b)
        }
        var x = [Float](); x.reserveCapacity(tRows * kIn)
        for i in 0..<(tRows * kIn) {
            let v: Float = 0.05 * Float((i * seedX) % 223) / 223.0 - 0.025
            x.append(v)
        }

        let xBits = x.map { f32ToBf16($0) }
        let scalesBits = scales.map { f32ToBf16($0) }
        let biasesBits = biases.map { f32ToBf16($0) }

        let xT = Tensor.empty(shape: [tRows, kIn], dtype: .bf16)
        xT.copyIn(from: xBits)
        let wT = Tensor.empty(shape: [nExperts, nOut, kIn / 8], dtype: .u32)
        wT.copyIn(from: weightPacked)
        let scalesT = Tensor.empty(shape: [nExperts, nOut, kIn / groupSize], dtype: .bf16)
        scalesT.copyIn(from: scalesBits)
        let biasesT = Tensor.empty(shape: [nExperts, nOut, kIn / groupSize], dtype: .bf16)
        biasesT.copyIn(from: biasesBits)
        let indicesT = Tensor.empty(shape: [tRows], dtype: .u32)
        indicesT.copyIn(from: indicesHost)

        let outBm64 = Tensor.empty(shape: [tRows, nOut], dtype: .bf16)
        runAndWait { cb in
            Ops.moeGatherDequantGemmInt4Bm64Mpp(
                input: xT, weight: wT, scales: scalesT, biases: biasesT,
                indices: indicesT,
                mTotal: tRows, nOut: nOut, kIn: kIn,
                groupSize: groupSize, on: cb, into: outBm64)
        }
        let bm64F32 = outBm64.toArray(as: UInt16.self).map { bf16ToF32($0) }

        let outBm16 = Tensor.empty(shape: [tRows, nOut], dtype: .bf16)
        runAndWait { cb in
            Ops.moeGatherDequantGemmInt4(
                input: xT, weight: wT, scales: scalesT, biases: biasesT,
                indices: indicesT,
                mTotal: tRows, nOut: nOut, kIn: kIn,
                groupSize: groupSize, on: cb, into: outBm16)
        }
        let bm16F32 = outBm16.toArray(as: UInt16.self).map { bf16ToF32($0) }

        // Build expertOffsets for m1 oracle.
        var expertOffsets = [UInt32](repeating: UInt32(tRows), count: nExperts + 1)
        for e in 0...nExperts {
            for r in 0..<tRows where Int(indicesHost[r]) >= e {
                expertOffsets[e] = UInt32(r)
                break
            }
        }
        expertOffsets[nExperts] = UInt32(tRows)

        let yM1 = m1Oracle(
            x: x, scales: scales, biases: biases, weight: weightPacked,
            indices: indicesHost, expertOffsets: expertOffsets,
            tRows: tRows, nOut: nOut, kIn: kIn, nExperts: nExperts, groupSize: groupSize)

        let cosBm64VsBm16 = cosine(bm64F32, bm16F32)
        let cosBm64VsM1 = cosine(bm64F32, yM1)
        let cosBm16VsM1 = cosine(bm16F32, yM1)
        print("[\(label)] cos: bm64_mpp vs bm16 = \(String(format: "%.6f", cosBm64VsBm16)), "
            + "bm64_mpp vs m1 = \(String(format: "%.6f", cosBm64VsM1)), "
            + "bm16 vs m1 = \(String(format: "%.6f", cosBm16VsM1))")
        print("  m1    out[0..8] = \(Array(yM1.prefix(8)))")
        print("  bm16  out[0..8] = \(Array(bm16F32.prefix(8)))")
        print("  bm64m out[0..8] = \(Array(bm64F32.prefix(8)))")
        return cosBm64VsM1
    }

    /// Upstream `..._bf16_matches_m1_clean_tile` shape.
    @Test("bm64_mpp bf16 wrapper matches bm16 reference at clean_tile shape")
    func bf16WrapperCleanTile() {
        let cosVal = Self.runWrapperCompare(
            nExperts: 4, tRows: 64, nOut: 64, kIn: 64, groupSize: 32,
            seedW: 7, seedS: 19, seedB: 29, seedX: 37,
            label: "clean_tile")
        #expect(cosVal >= 0.99,
                "bm64_mpp bf16 wrapper vs bm16 cos = \(cosVal) (want ≥ 0.99) clean_tile")
    }

    /// Multi-n-tile + multi-K-block + group_size=64 — matches the bf16
    /// multi-tile cell we added upstream that passes at the kernel level.
    @Test("bm64_mpp bf16 wrapper matches bm16 reference at multi-tile shape")
    func bf16WrapperMultiTile() {
        let cosVal = Self.runWrapperCompare(
            nExperts: 8, tRows: 128, nOut: 128, kIn: 128, groupSize: 64,
            seedW: 11, seedS: 23, seedB: 31, seedX: 41,
            label: "multi_tile")
        #expect(cosVal >= 0.99,
                "bm64_mpp bf16 wrapper vs bm16 cos = \(cosVal) (want ≥ 0.99) multi_tile")
    }

    /// Qwen3.6-A3B gate/up shape: nExperts=128, mTotal=64 (1 m-tile),
    /// nOut=768 (12 n-tiles), kIn=2048 (64 K-blocks, 32 groups at
    /// group_size=64). bf16. Sparse indices — most experts skipped.
    /// Mirrors the actual production call site that breaks
    /// `forwardManyEquivalence`.
    @Test("bm64_mpp bf16 wrapper matches bm16 reference at qwen36 gate/up shape")
    func bf16WrapperQwen36GateUp() {
        let cosVal = Self.runWrapperCompare(
            nExperts: 128, tRows: 64, nOut: 768, kIn: 2048, groupSize: 64,
            seedW: 13, seedS: 19, seedB: 23, seedX: 29,
            sparseIndices: true,
            label: "qwen36_gate_up")
        #expect(cosVal >= 0.99,
                "bm64_mpp bf16 wrapper vs bm16 cos = \(cosVal) (want ≥ 0.99) qwen36_gate_up")
    }

    /// Qwen3.6-A3B down shape: same nExperts, mTotal, but nOut=2048
    /// (32 n-tiles), kIn=768 (24 K-blocks, 12 groups). Down BGEMM is
    /// the largest n-fanout — most n-tiles per TG dispatch.
    @Test("bm64_mpp bf16 wrapper matches bm16 reference at qwen36 down shape")
    func bf16WrapperQwen36Down() {
        let cosVal = Self.runWrapperCompare(
            nExperts: 128, tRows: 64, nOut: 2048, kIn: 768, groupSize: 64,
            seedW: 17, seedS: 31, seedB: 37, seedX: 43,
            sparseIndices: true,
            label: "qwen36_down")
        #expect(cosVal >= 0.99,
                "bm64_mpp bf16 wrapper vs bm16 cos = \(cosVal) (want ≥ 0.99) qwen36_down")
    }

    /// Live-compiles the MSL source at runtime via `makeLibrary(source:)`
    /// instead of loading the pre-built `kernels.metallib`. If this path
    /// gives correct output while the metallib path stays at cos 0.816,
    /// it confirms the bug is in offline `xcrun metal → metallib` compilation
    /// of MPP cooperative-tensor kernels (likely SDK 26.5 / runtime 26.4.1
    /// header drift).
    @Test("bm64_mpp bf16 LIVE-COMPILED matches m1 at down shape")
    func bf16LiveCompiledDown() throws {
        let nExperts = 128
        let kIn = 768
        let nOut = 2048
        let groupSize = 64
        let tRows = 64
        var indicesHost = [UInt32]()
        for r in 0..<tRows { indicesHost.append(UInt32((r * 2) % nExperts)) }

        let totalWeights = nExperts * nOut * kIn
        var weightUnpacked = [UInt32](); weightUnpacked.reserveCapacity(totalWeights)
        for i in 0..<totalWeights { weightUnpacked.append(UInt32((i * 17 + 11) & 0xF)) }
        var weightPacked: [UInt32] = []; weightPacked.reserveCapacity(totalWeights / 8)
        for r in stride(from: 0, to: totalWeights, by: kIn) {
            weightPacked.append(contentsOf: Self.packRow(Array(weightUnpacked[r..<r+kIn])))
        }
        let groupsTotal = nExperts * nOut * (kIn / groupSize)
        let (scales, biases, x) = Self.sinInputs(
            scaleAmp: 0.001, scaleOff: 0.003, scaleFreq: 0.013,
            biasAmp: 0.003, biasOff: -0.015, biasFreq: 0.019,
            xAmp: 0.03, xFreq: 0.011,
            groupsTotal: groupsTotal, xCount: tRows * kIn)
        let xBits = x.map { Self.f32ToBf16($0) }
        let scalesBits = scales.map { Self.f32ToBf16($0) }
        let biasesBits = biases.map { Self.f32ToBf16($0) }

        let mtlDevice = MTLCreateSystemDefaultDevice()!
        func mkBuf<T>(_ data: [T]) -> MTLBuffer {
            let bytes = data.withUnsafeBytes { Data($0) }
            return mtlDevice.makeBuffer(bytes: (bytes as NSData).bytes, length: bytes.count, options: .storageModeShared)!
        }
        let xBuf = mkBuf(xBits)
        let wBuf = mkBuf(weightPacked)
        let sBuf = mkBuf(scalesBits)
        let bBuf = mkBuf(biasesBits)
        let iBuf = mkBuf(indicesHost)
        let outBuf = mtlDevice.makeBuffer(length: tRows * nOut * 2, options: .storageModeShared)!
        memset(outBuf.contents(), 0, tRows * nOut * 2)

        // Live-compile from the .metal source on disk. Resolve the
        // kernels dir relative to this test file (#filePath →
        // <repo>/Tests/FFAITests/…) instead of a hardcoded absolute
        // path, so the test is machine-independent.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // FFAITests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // <repo>/
        let metalURL = repoRoot.appendingPathComponent(
            "Sources/MetalTileSwift/Resources/kernels/mt_moe_gather_qmm_mma_int4_bm64_mpp_bf16.metal")
        let source = try String(contentsOf: metalURL, encoding: .utf8)
        let opts = MTLCompileOptions()
        let lib = try mtlDevice.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: "mt_moe_gather_qmm_mma_int4_bm64_mpp_bf16")!
        let pso = try mtlDevice.makeComputePipelineState(function: fn)

        let queue = mtlDevice.makeCommandQueue()!
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(sBuf, offset: 0, index: 2)
        enc.setBuffer(bBuf, offset: 0, index: 3)
        enc.setBuffer(iBuf, offset: 0, index: 4)
        enc.setBuffer(outBuf, offset: 0, index: 5)
        var mT = UInt32(tRows); enc.setBytes(&mT, length: 4, index: 6)
        var nO = UInt32(nOut);  enc.setBytes(&nO, length: 4, index: 7)
        var kI = UInt32(kIn);   enc.setBytes(&kI, length: 4, index: 8)
        var gS = UInt32(groupSize); enc.setBytes(&gS, length: 4, index: 9)
        enc.dispatchThreadgroups(
            MTLSize(width: (nOut + 63) / 64, height: (tRows + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let outRaw = outBuf.contents().bindMemory(to: UInt16.self, capacity: tRows * nOut)
        let bm64F32 = (0..<(tRows * nOut)).map { Self.bf16ToF32(outRaw[$0]) }

        var expertOffsets = [UInt32](repeating: UInt32(tRows), count: nExperts + 1)
        for e in 0...nExperts {
            for r in 0..<tRows where Int(indicesHost[r]) >= e { expertOffsets[e] = UInt32(r); break }
        }
        expertOffsets[nExperts] = UInt32(tRows)
        let yM1 = Self.m1Oracle(
            x: x, scales: scales, biases: biases, weight: weightPacked,
            indices: indicesHost, expertOffsets: expertOffsets,
            tRows: tRows, nOut: nOut, kIn: kIn, nExperts: nExperts, groupSize: groupSize)

        let cosBm64VsM1 = Self.cosine(bm64F32, yM1)
        print("[live_compile] cos: bm64 vs m1 = \(String(format: "%.6f", cosBm64VsM1))")
        print("  m1    out[0..8] = \(Array(yM1.prefix(8)))")
        print("  bm64m out[0..8] = \(Array(bm64F32.prefix(8)))")
        #expect(cosBm64VsM1 >= 0.99,
                "live-compile bm64 vs m1 cos = \(cosBm64VsM1) (want ≥ 0.99)")
    }

    /// Bypasses the wrapper entirely — builds a fresh encoder and uses
    /// `dispatchThreadgroups` (count-of-TGs semantics) instead of the
    /// wrapper's `dispatchThreads` (count-of-total-threads). If the
    /// generated wrapper has a dispatchThreads-vs-Threadgroups bug at
    /// MPP cooperative tensors specifically, this path will produce the
    /// correct output (m1-matching) and the wrapper path will produce
    /// the broken 0.816 cosine output.
    @Test("bm64_mpp bf16 raw dispatchThreadgroups matches m1 at down shape")
    func bf16RawDispatchThreadgroupsDown() throws {
        let nExperts = 128
        let kIn = 768
        let nOut = 2048
        let groupSize = 64
        let tRows = 64
        var indicesHost = [UInt32]()
        for r in 0..<tRows { indicesHost.append(UInt32((r * 2) % nExperts)) }

        let totalWeights = nExperts * nOut * kIn
        var weightUnpacked = [UInt32](); weightUnpacked.reserveCapacity(totalWeights)
        for i in 0..<totalWeights { weightUnpacked.append(UInt32((i * 17 + 11) & 0xF)) }
        var weightPacked: [UInt32] = []; weightPacked.reserveCapacity(totalWeights / 8)
        for r in stride(from: 0, to: totalWeights, by: kIn) {
            weightPacked.append(contentsOf: Self.packRow(Array(weightUnpacked[r..<r+kIn])))
        }
        let groupsTotal = nExperts * nOut * (kIn / groupSize)
        let (scales, biases, x) = Self.sinInputs(
            scaleAmp: 0.001, scaleOff: 0.003, scaleFreq: 0.013,
            biasAmp: 0.003, biasOff: -0.015, biasFreq: 0.019,
            xAmp: 0.03, xFreq: 0.011,
            groupsTotal: groupsTotal, xCount: tRows * kIn)
        let xBits = x.map { Self.f32ToBf16($0) }
        let scalesBits = scales.map { Self.f32ToBf16($0) }
        let biasesBits = biases.map { Self.f32ToBf16($0) }

        // Raw MTLBuffers — no Tensor wrapper.
        let mtlDevice = MTLCreateSystemDefaultDevice()!
        func mkBuf<T>(_ data: [T]) -> MTLBuffer {
            let bytes = data.withUnsafeBytes { Data($0) }
            return mtlDevice.makeBuffer(bytes: (bytes as NSData).bytes, length: bytes.count, options: .storageModeShared)!
        }
        let xBuf = mkBuf(xBits)
        let wBuf = mkBuf(weightPacked)
        let sBuf = mkBuf(scalesBits)
        let bBuf = mkBuf(biasesBits)
        let iBuf = mkBuf(indicesHost)
        let outBuf = mtlDevice.makeBuffer(length: tRows * nOut * 2, options: .storageModeShared)!
        memset(outBuf.contents(), 0, tRows * nOut * 2)

        // Use the FFAI MetalTileLibrary's PSO (same metallib, same kernel).
        let pso = PSOCache.shared.pipelineState(for: "mt_moe_gather_qmm_mma_int4_bm64_mpp_bf16")
        let queue = mtlDevice.makeCommandQueue()!
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(sBuf, offset: 0, index: 2)
        enc.setBuffer(bBuf, offset: 0, index: 3)
        enc.setBuffer(iBuf, offset: 0, index: 4)
        enc.setBuffer(outBuf, offset: 0, index: 5)
        // Mirror upstream Rust runtime — put constexpr scalars in MTLBuffers
        // (4 bytes each) instead of using setBytes. If this fixes the
        // output, the bug is setBytes vs setBuffer for `constant uint &`
        // params on MPP-backed kernels.
        var mT = UInt32(tRows)
        var nO = UInt32(nOut)
        var kI = UInt32(kIn)
        var gS = UInt32(groupSize)
        let mTBuf = mtlDevice.makeBuffer(bytes: &mT, length: 4, options: .storageModeShared)!
        let nOBuf = mtlDevice.makeBuffer(bytes: &nO, length: 4, options: .storageModeShared)!
        let kIBuf = mtlDevice.makeBuffer(bytes: &kI, length: 4, options: .storageModeShared)!
        let gSBuf = mtlDevice.makeBuffer(bytes: &gS, length: 4, options: .storageModeShared)!
        enc.setBuffer(mTBuf, offset: 0, index: 6)
        enc.setBuffer(nOBuf, offset: 0, index: 7)
        enc.setBuffer(kIBuf, offset: 0, index: 8)
        enc.setBuffer(gSBuf, offset: 0, index: 9)
        // dispatchThreadgroups — count-of-TGs semantics, mirrors upstream Rust.
        enc.dispatchThreadgroups(
            MTLSize(width: (nOut + 63) / 64, height: (tRows + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let outRaw = outBuf.contents().bindMemory(to: UInt16.self, capacity: tRows * nOut)
        let bm64F32 = (0..<(tRows * nOut)).map { Self.bf16ToF32(outRaw[$0]) }

        var expertOffsets = [UInt32](repeating: UInt32(tRows), count: nExperts + 1)
        for e in 0...nExperts {
            for r in 0..<tRows where Int(indicesHost[r]) >= e { expertOffsets[e] = UInt32(r); break }
        }
        expertOffsets[nExperts] = UInt32(tRows)
        let yM1 = Self.m1Oracle(
            x: x, scales: scales, biases: biases, weight: weightPacked,
            indices: indicesHost, expertOffsets: expertOffsets,
            tRows: tRows, nOut: nOut, kIn: kIn, nExperts: nExperts, groupSize: groupSize)

        let cosBm64VsM1 = Self.cosine(bm64F32, yM1)
        print("[raw_dispatchThreadgroups] cos: bm64 vs m1 = \(String(format: "%.6f", cosBm64VsM1))")
        // Per-row cosine — pinpoint which rows are correct vs broken.
        for rowIdx in [0, 1, 2, 31, 32, 33, 62, 63] {
            let m1Row = Array(yM1[rowIdx * nOut..<(rowIdx + 1) * nOut])
            let bm64Row = Array(bm64F32[rowIdx * nOut..<(rowIdx + 1) * nOut])
            let rc = Self.cosine(m1Row, bm64Row)
            print("  row \(rowIdx) cos = \(String(format: "%.6f", rc)) | "
                + "m1[0..4]=\(Array(m1Row.prefix(4))) bm64m[0..4]=\(Array(bm64Row.prefix(4)))")
        }
        #expect(cosBm64VsM1 >= 0.99,
                "raw dispatchThreadgroups bm64 vs m1 cos = \(cosBm64VsM1) (want ≥ 0.99)")
    }

    /// Down shape with the SAME sin/cos inputs the upstream Rust test
    /// uses (and where it gets cos 0.99997 vs m1). If this Swift cell
    /// reproduces the upstream cosine, the bug is input-dependent (only
    /// triggers on certain bf16 value patterns). If it still drifts at
    /// ~0.98, the bug is in the Swift dispatch path.
    @Test("bm64_mpp bf16 wrapper down shape with sin inputs (upstream match)")
    func bf16WrapperQwen36DownSinInputs() {
        let nExperts = 128
        let kIn = 768
        let nOut = 2048
        let groupSize = 64
        let tRows = 64
        var indicesHost = [UInt32]()
        indicesHost.reserveCapacity(tRows)
        for r in 0..<tRows {
            indicesHost.append(UInt32((r * 2) % nExperts))
        }

        let totalWeights = nExperts * nOut * kIn
        var weightUnpacked = [UInt32](); weightUnpacked.reserveCapacity(totalWeights)
        for i in 0..<totalWeights {
            weightUnpacked.append(UInt32((i * 17 + 11) & 0xF))
        }
        var weightPacked: [UInt32] = []
        weightPacked.reserveCapacity(totalWeights / 8)
        for r in stride(from: 0, to: totalWeights, by: kIn) {
            weightPacked.append(contentsOf: Self.packRow(Array(weightUnpacked[r..<r+kIn])))
        }
        let groupsTotal = nExperts * nOut * (kIn / groupSize)
        let (scales, biases, x) = Self.sinInputs(
            scaleAmp: 0.001, scaleOff: 0.003, scaleFreq: 0.013,
            biasAmp: 0.003, biasOff: -0.015, biasFreq: 0.019,
            xAmp: 0.03, xFreq: 0.011,
            groupsTotal: groupsTotal, xCount: tRows * kIn)

        let xBits = x.map { Self.f32ToBf16($0) }
        let scalesBits = scales.map { Self.f32ToBf16($0) }
        let biasesBits = biases.map { Self.f32ToBf16($0) }

        let xT = Tensor.empty(shape: [tRows, kIn], dtype: .bf16); xT.copyIn(from: xBits)
        let wT = Tensor.empty(shape: [nExperts, nOut, kIn / 8], dtype: .u32); wT.copyIn(from: weightPacked)
        let scalesT = Tensor.empty(shape: [nExperts, nOut, kIn / groupSize], dtype: .bf16); scalesT.copyIn(from: scalesBits)
        let biasesT = Tensor.empty(shape: [nExperts, nOut, kIn / groupSize], dtype: .bf16); biasesT.copyIn(from: biasesBits)
        let indicesT = Tensor.empty(shape: [tRows], dtype: .u32); indicesT.copyIn(from: indicesHost)

        let outBm64 = Tensor.empty(shape: [tRows, nOut], dtype: .bf16)
        runAndWait { cb in
            Ops.moeGatherDequantGemmInt4Bm64Mpp(
                input: xT, weight: wT, scales: scalesT, biases: biasesT,
                indices: indicesT,
                mTotal: tRows, nOut: nOut, kIn: kIn,
                groupSize: groupSize, on: cb, into: outBm64)
        }
        let bm64F32 = outBm64.toArray(as: UInt16.self).map { Self.bf16ToF32($0) }

        let outBm16 = Tensor.empty(shape: [tRows, nOut], dtype: .bf16)
        runAndWait { cb in
            Ops.moeGatherDequantGemmInt4(
                input: xT, weight: wT, scales: scalesT, biases: biasesT,
                indices: indicesT,
                mTotal: tRows, nOut: nOut, kIn: kIn,
                groupSize: groupSize, on: cb, into: outBm16)
        }
        let bm16F32 = outBm16.toArray(as: UInt16.self).map { Self.bf16ToF32($0) }

        var expertOffsets = [UInt32](repeating: UInt32(tRows), count: nExperts + 1)
        for e in 0...nExperts {
            for r in 0..<tRows where Int(indicesHost[r]) >= e {
                expertOffsets[e] = UInt32(r)
                break
            }
        }
        expertOffsets[nExperts] = UInt32(tRows)

        let yM1 = Self.m1Oracle(
            x: x, scales: scales, biases: biases, weight: weightPacked,
            indices: indicesHost, expertOffsets: expertOffsets,
            tRows: tRows, nOut: nOut, kIn: kIn, nExperts: nExperts, groupSize: groupSize)

        let cosBm64VsM1 = Self.cosine(bm64F32, yM1)
        let cosBm16VsM1 = Self.cosine(bm16F32, yM1)
        let cosBm64VsBm16 = Self.cosine(bm64F32, bm16F32)
        print("[qwen36_down_sin] cos: bm64 vs m1=\(String(format: "%.6f", cosBm64VsM1)), "
            + "bm16 vs m1=\(String(format: "%.6f", cosBm16VsM1)), "
            + "bm64 vs bm16=\(String(format: "%.6f", cosBm64VsBm16))")
        print("  m1    out[0..8] = \(Array(yM1.prefix(8)))")
        print("  bm16  out[0..8] = \(Array(bm16F32.prefix(8)))")
        print("  bm64m out[0..8] = \(Array(bm64F32.prefix(8)))")
        #expect(cosBm64VsM1 >= 0.99,
                "bm64_mpp bf16 vs m1 cos = \(cosBm64VsM1) (want ≥ 0.99) qwen36_down_sin")
    }
}
