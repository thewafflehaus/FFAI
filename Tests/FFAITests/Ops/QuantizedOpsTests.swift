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
// Verify the int4 dequantizing GEMV kernel against a CPU reference.
// We construct a tiny 2-row × 16-column quantized matrix, run the kernel,
// and compare to the dequantize-then-multiply result computed in Swift.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import FFAI

@Suite("Quantized GEMV (int4)")
struct QuantizedOpsTests {
    static let groupSize = 8  // smallest valid group_size (multiple of 8)
    static let inDim = 16  // 2 groups
    static let outDim = 2

    /// Pack 8 4-bit values (low nibble first) into one uint32.
    static func pack8(_ q: [UInt32]) -> UInt32 {
        precondition(q.count == 8)
        var w: UInt32 = 0
        for i in 0 ..< 8 { w |= (q[i] & 0xF) << (4 * UInt32(i)) }
        return w
    }

    /// Pack 4 8-bit values (low byte first) into one uint32.
    static func pack4Bytes(_ q: [UInt32]) -> UInt32 {
        precondition(q.count == 4)
        var w: UInt32 = 0
        for i in 0 ..< 4 { w |= (q[i] & 0xFF) << (8 * UInt32(i)) }
        return w
    }

    /// Pack 16 6-bit values into 3 uint32 (12 bytes).
    static func pack16Sixbit(_ q: [UInt32]) -> [UInt32] {
        precondition(q.count == 16, "expected 16 6-bit values")
        var bytes = [UInt8](repeating: 0, count: 12)
        for p in 0 ..< 4 {
            let v0 = q[p * 4 + 0] & 0x3F
            let v1 = q[p * 4 + 1] & 0x3F
            let v2 = q[p * 4 + 2] & 0x3F
            let v3 = q[p * 4 + 3] & 0x3F
            bytes[p * 3 + 0] = UInt8(truncatingIfNeeded: v0 | ((v1 & 0x03) << 6))
            bytes[p * 3 + 1] = UInt8(truncatingIfNeeded: ((v1 >> 2) & 0x0F) | ((v2 & 0x0F) << 4))
            bytes[p * 3 + 2] = UInt8(truncatingIfNeeded: ((v2 >> 4) & 0x03) | ((v3 & 0x3F) << 2))
        }
        return Self.bytesToUint32s(bytes)
    }

    /// Pack 32 3-bit values into 3 uint32 (12 bytes = 32*3 bits).
    /// 8 values per 3 bytes; 4 chunks span 3 uint32.
    static func pack32Threebit(_ q: [UInt32]) -> [UInt32] {
        precondition(q.count == 32, "expected 32 3-bit values")
        var bytes = [UInt8](repeating: 0, count: 12)
        for c in 0 ..< 4 {
            let v = (0 ..< 8).map { q[c * 8 + $0] & 0x07 }
            bytes[c * 3 + 0] = UInt8(truncatingIfNeeded: v[0] | (v[1] << 3) | ((v[2] & 0x03) << 6))
            bytes[c * 3 + 1] = UInt8(
                truncatingIfNeeded: ((v[2] >> 2) & 0x01) | (v[3] << 1) | (v[4] << 4)
                    | ((v[5] & 0x01) << 7))
            bytes[c * 3 + 2] = UInt8(
                truncatingIfNeeded: ((v[5] >> 1) & 0x03) | (v[6] << 2) | (v[7] << 5))
        }
        return Self.bytesToUint32s(bytes)
    }

    /// Pack 32 5-bit values into 5 uint32 (20 bytes = 32*5 bits).
    /// 8 values per 5 bytes; 4 chunks span 5 uint32.
    static func pack32Fivebit(_ q: [UInt32]) -> [UInt32] {
        precondition(q.count == 32, "expected 32 5-bit values")
        var bytes = [UInt8](repeating: 0, count: 20)
        for c in 0 ..< 4 {
            let v = (0 ..< 8).map { q[c * 8 + $0] & 0x1F }
            bytes[c * 5 + 0] = UInt8(truncatingIfNeeded: v[0] | ((v[1] & 0x07) << 5))
            bytes[c * 5 + 1] = UInt8(
                truncatingIfNeeded: ((v[1] >> 3) & 0x03) | ((v[2] & 0x1F) << 2)
                    | ((v[3] & 0x01) << 7))
            bytes[c * 5 + 2] = UInt8(
                truncatingIfNeeded: ((v[3] >> 1) & 0x0F) | ((v[4] & 0x0F) << 4))
            bytes[c * 5 + 3] = UInt8(
                truncatingIfNeeded: ((v[4] >> 4) & 0x01) | ((v[5] & 0x1F) << 1)
                    | ((v[6] & 0x03) << 6))
            bytes[c * 5 + 4] = UInt8(
                truncatingIfNeeded: ((v[6] >> 2) & 0x07) | ((v[7] & 0x1F) << 3))
        }
        return Self.bytesToUint32s(bytes)
    }

    /// Convert a byte array (length must be a multiple of 4) to little-endian uint32s.
    static func bytesToUint32s(_ bytes: [UInt8]) -> [UInt32] {
        precondition(bytes.count % 4 == 0, "byte length must be a multiple of 4")
        var out: [UInt32] = []
        out.reserveCapacity(bytes.count / 4)
        for u in 0 ..< (bytes.count / 4) {
            var w: UInt32 = 0
            for b in 0 ..< 4 {
                w |= UInt32(bytes[u * 4 + b]) << (8 * UInt32(b))
            }
            out.append(w)
        }
        return out
    }

    @Test("int4 dequant gather + matrix multiply matches CPU at group_size=64")
    func realShapeRoundTrip() {
        autoreleasepool {
            // Realistic shape: 4 rows × 128 in_dim, group_size=64 → 2 groups per row.
            let outDim = 4
            let inDim = 128
            let gs = 64
            let nGroups = inDim / gs

            // Synthetic q values: deterministic, vary across groups.
            var q = [[UInt32]](repeating: [], count: outDim)
            for r in 0 ..< outDim {
                q[r] = (0 ..< inDim).map { UInt32(($0 + r * 7) % 16) }
            }
            let scales: [Float] = (0 ..< (outDim * nGroups)).map { Float($0 + 1) * 0.01 }
            let biases: [Float] = (0 ..< (outDim * nGroups)).map { Float($0) * -0.005 }
            let input: [Float] = (0 ..< inDim).map { Float($0) * 0.1 - 6.4 }

            // Pack
            var packed: [UInt32] = []
            for r in 0 ..< outDim {
                for i in stride(from: 0, to: inDim, by: 8) {
                    let nibbles = Array(q[r][i ..< i + 8])
                    packed.append(Self.pack8(nibbles))
                }
            }

            // CPU reference
            var expected: [Float] = []
            for r in 0 ..< outDim {
                var acc: Float = 0
                for g in 0 ..< nGroups {
                    let s = scales[r * nGroups + g]
                    let b = biases[r * nGroups + g]
                    for j in 0 ..< gs {
                        let qv = Float(q[r][g * gs + j])
                        acc += (qv * s + b) * input[g * gs + j]
                    }
                }
                expected.append(acc)
            }

            // Allocate
            let weight = Tensor.empty(shape: [outDim, inDim / 8], dtype: .u32)
            weight.copyIn(from: packed)
            let scalesT = Tensor.empty(shape: [outDim, nGroups], dtype: .f32)
            scalesT.copyIn(from: scales)
            let biasesT = Tensor.empty(shape: [outDim, nGroups], dtype: .f32)
            biasesT.copyIn(from: biases)
            let inputT = Tensor.empty(shape: [inDim], dtype: .f32)
            inputT.copyIn(from: input)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.dequantGemvInt4(
                    weight: weight, scales: scalesT, biases: biasesT,
                    input: inputT, groupSize: gs, on: cb
                )
            }

            let got = out.toArray(as: Float.self)
            for i in 0 ..< outDim {
                #expect(
                    abs(got[i] - expected[i]) < 1e-2,
                    "row \(i): got \(got[i]) expected \(expected[i])")
            }
        }
    }

    @Test("int6 dequant gemv (group_size=64, bf16 scales/biases) matches CPU")
    func roundTripInt6Bf16() {
        autoreleasepool {
            let outDim = 2
            let inDim = 64  // 1 group; minimal valid (group_size = 64)
            let gs = 64

            // 6-bit values 0..63
            var q = [[UInt32]](repeating: [], count: outDim)
            for r in 0 ..< outDim {
                q[r] = (0 ..< inDim).map { UInt32(($0 + r * 11) & 0x3F) }
            }
            let scales: [Float] = (0 ..< outDim).map { Float($0 + 1) * 0.05 }
            let biases: [Float] = (0 ..< outDim).map { Float($0) * -0.1 }
            let input: [Float] = (0 ..< inDim).map { Float($0) * 0.1 - 3.2 }

            // Pack: 64 values per row → 4 chunks of 16, each chunk is 3 uint32 = 12 uint32 per row.
            var packed: [UInt32] = []
            for r in 0 ..< outDim {
                for chunk in 0 ..< (inDim / 16) {
                    let chunk16 = Array(q[r][chunk * 16 ..< (chunk + 1) * 16])
                    packed.append(contentsOf: Self.pack16Sixbit(chunk16))
                }
            }

            // CPU reference (using bf16-rounded scale/bias to compare apples-to-apples)
            func bf(_ f: Float) -> Float {
                Float(bitPattern: UInt32(UInt16(truncatingIfNeeded: f.bitPattern >> 16)) << 16)
            }
            var expected: [Float] = []
            for r in 0 ..< outDim {
                var acc: Float = 0
                for j in 0 ..< inDim {
                    let qv = Float(q[r][j])
                    acc += (qv * bf(scales[r]) + bf(biases[r])) * bf(input[j])
                }
                expected.append(acc)
            }

            let scalesBits = scales.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let biasesBits = biases.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let inputBits = input.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }

            let weight = Tensor.empty(shape: [outDim, inDim * 3 / 16], dtype: .u32)
            weight.copyIn(from: packed)
            let scalesT = Tensor.empty(shape: [outDim, 1], dtype: .bf16)
            scalesT.copyIn(from: scalesBits)
            let biasesT = Tensor.empty(shape: [outDim, 1], dtype: .bf16)
            biasesT.copyIn(from: biasesBits)
            let inputT = Tensor.empty(shape: [inDim], dtype: .bf16)
            inputT.copyIn(from: inputBits)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.dequantGemv(
                    weight: weight, scales: scalesT, biases: biasesT,
                    input: inputT, bits: 6, groupSize: gs, on: cb
                )
            }

            let got = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            for i in 0 ..< outDim {
                let tol = max(abs(expected[i]) * 0.05, 1.0)
                #expect(
                    abs(got[i] - expected[i]) < tol,
                    "row \(i): got \(got[i]) expected \(expected[i]) (tol \(tol))")
            }
        }
    }

    /// Helper: run an int<bits> dequant gemv with synthetic data and
    /// verify against a CPU reference using bf16-rounded values.
    func runDequantGemvCheck(
        bits: Int, mask: UInt32,
        packPerChunk: Int, uint32sPerChunk: Int,
        pack: ([UInt32]) -> [UInt32]
    ) {
        let outDim = 2
        let inDim = 64  // 1 group, group_size=64
        let gs = 64
        let chunkValues = packPerChunk  // values per chunk

        var q = [[UInt32]](repeating: [], count: outDim)
        for r in 0 ..< outDim {
            q[r] = (0 ..< inDim).map { UInt32(($0 + r * 7) & Int(mask)) }
        }
        let scales: [Float] = (0 ..< outDim).map { Float($0 + 1) * 0.04 }
        let biases: [Float] = (0 ..< outDim).map { Float($0) * -0.07 }
        let input: [Float] = (0 ..< inDim).map { Float($0) * 0.1 - 3.2 }

        var packed: [UInt32] = []
        for r in 0 ..< outDim {
            for chunk in 0 ..< (inDim / chunkValues) {
                let chunkVals = Array(q[r][chunk * chunkValues ..< (chunk + 1) * chunkValues])
                packed.append(contentsOf: pack(chunkVals))
            }
        }

        func bf(_ f: Float) -> Float {
            Float(bitPattern: UInt32(UInt16(truncatingIfNeeded: f.bitPattern >> 16)) << 16)
        }
        var expected: [Float] = []
        for r in 0 ..< outDim {
            var acc: Float = 0
            for j in 0 ..< inDim {
                let qv = Float(q[r][j])
                acc += (qv * bf(scales[r]) + bf(biases[r])) * bf(input[j])
            }
            expected.append(acc)
        }

        let scalesBits = scales.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        let biasesBits = biases.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        let inputBits = input.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        let u32PerRow = inDim * bits / 32
        let weight = Tensor.empty(shape: [outDim, u32PerRow], dtype: .u32)
        weight.copyIn(from: packed)
        let scalesT = Tensor.empty(shape: [outDim, 1], dtype: .bf16)
        scalesT.copyIn(from: scalesBits)
        let biasesT = Tensor.empty(shape: [outDim, 1], dtype: .bf16)
        biasesT.copyIn(from: biasesBits)
        let inputT = Tensor.empty(shape: [inDim], dtype: .bf16)
        inputT.copyIn(from: inputBits)

        var out: Tensor!
        runAndWait { cb in
            out = Ops.dequantGemv(
                weight: weight, scales: scalesT, biases: biasesT,
                input: inputT, bits: bits, groupSize: gs, on: cb
            )
        }

        let got = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
        for i in 0 ..< outDim {
            let tol = max(abs(expected[i]) * 0.05, 1.0)
            #expect(
                abs(got[i] - expected[i]) < tol,
                "bits=\(bits) row \(i): got \(got[i]) expected \(expected[i]) (tol \(tol))")
        }
        _ = uint32sPerChunk  // silence unused
    }

    @Test("int3 dequant gemv (group_size=64, bf16) matches CPU")
    func roundTripInt3Bf16() {
        autoreleasepool {
            runDequantGemvCheck(
                bits: 3, mask: 0x07, packPerChunk: 32, uint32sPerChunk: 3,
                pack: Self.pack32Threebit)
        }
    }

    @Test("int5 dequant gemv (group_size=64, bf16) matches CPU")
    func roundTripInt5Bf16() {
        autoreleasepool {
            runDequantGemvCheck(
                bits: 5, mask: 0x1F, packPerChunk: 32, uint32sPerChunk: 5,
                pack: Self.pack32Fivebit)
        }
    }

    @Test("int8 dequant gemv (group_size=64, bf16 scales/biases) matches CPU")
    func roundTripInt8Bf16() {
        autoreleasepool {
            let outDim = 2
            let inDim = 128
            let gs = 64
            let nGroups = inDim / gs

            // 8-bit values (0..255)
            var q = [[UInt32]](repeating: [], count: outDim)
            for r in 0 ..< outDim {
                q[r] = (0 ..< inDim).map { UInt32(($0 + r * 13) & 0xFF) }
            }
            let scales: [Float] = (0 ..< (outDim * nGroups)).map { Float($0 + 1) * 0.001 }
            let biases: [Float] = (0 ..< (outDim * nGroups)).map { Float($0) * -0.0005 }
            let input: [Float] = (0 ..< inDim).map { Float($0) * 0.1 - 6.4 }

            // Pack 4 8-bit values per uint32
            var packed: [UInt32] = []
            for r in 0 ..< outDim {
                for i in stride(from: 0, to: inDim, by: 4) {
                    let bytes = Array(q[r][i ..< i + 4])
                    packed.append(Self.pack4Bytes(bytes))
                }
            }

            // Reference uses bf16-rounded scale/bias values
            func bf16Round(_ f: Float) -> Float {
                Float(bitPattern: UInt32(UInt16(truncatingIfNeeded: f.bitPattern >> 16)) << 16)
            }
            var expected: [Float] = []
            for r in 0 ..< outDim {
                var acc: Float = 0
                for g in 0 ..< nGroups {
                    let s = bf16Round(scales[r * nGroups + g])
                    let b = bf16Round(biases[r * nGroups + g])
                    for j in 0 ..< gs {
                        let qv = Float(q[r][g * gs + j])
                        acc += (qv * s + b) * bf16Round(input[g * gs + j])
                    }
                }
                expected.append(acc)
            }

            let scalesBits = scales.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let biasesBits = biases.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let inputBits = input.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }

            let weight = Tensor.empty(shape: [outDim, inDim / 4], dtype: .u32)
            weight.copyIn(from: packed)
            let scalesT = Tensor.empty(shape: [outDim, nGroups], dtype: .bf16)
            scalesT.copyIn(from: scalesBits)
            let biasesT = Tensor.empty(shape: [outDim, nGroups], dtype: .bf16)
            biasesT.copyIn(from: biasesBits)
            let inputT = Tensor.empty(shape: [inDim], dtype: .bf16)
            inputT.copyIn(from: inputBits)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.dequantGemv(
                    weight: weight, scales: scalesT, biases: biasesT,
                    input: inputT, bits: 8, groupSize: gs, on: cb
                )
            }

            let got = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            for i in 0 ..< outDim {
                let tol = max(abs(expected[i]) * 0.05, 1.0)
                #expect(
                    abs(got[i] - expected[i]) < tol,
                    "row \(i): got \(got[i]) expected \(expected[i]) (tol \(tol))")
            }
        }
    }

    @Test("int4 dequant gemv with bf16 scales/biases matches CPU")
    func roundTripBf16() {
        autoreleasepool {
            // Same shapes as the real Qwen3 4B q_proj per-row (in_dim=2560,
            // group_size=64) but smaller: 2 rows × 128 in_dim, gs=64.
            let outDim = 2
            let inDim = 128
            let gs = 64
            let nGroups = inDim / gs

            var q = [[UInt32]](repeating: [], count: outDim)
            for r in 0 ..< outDim {
                q[r] = (0 ..< inDim).map { UInt32(($0 + r * 5) % 16) }
            }
            let scales: [Float] = (0 ..< (outDim * nGroups)).map { Float($0 + 1) * 0.01 }
            let biases: [Float] = (0 ..< (outDim * nGroups)).map { Float($0) * -0.005 }
            let input: [Float] = (0 ..< inDim).map { Float($0) * 0.1 - 6.4 }

            var packed: [UInt32] = []
            for r in 0 ..< outDim {
                for i in stride(from: 0, to: inDim, by: 8) {
                    let nibbles = Array(q[r][i ..< i + 8])
                    packed.append(Self.pack8(nibbles))
                }
            }

            // Reference uses the bf16-rounded scale/bias values to compare apples-to-apples
            func bf16Round(_ f: Float) -> Float {
                let bits = f.bitPattern
                let topHalf = UInt16(truncatingIfNeeded: bits >> 16)
                return Float(bitPattern: UInt32(topHalf) << 16)
            }
            var expected: [Float] = []
            for r in 0 ..< outDim {
                var acc: Float = 0
                for g in 0 ..< nGroups {
                    let s = bf16Round(scales[r * nGroups + g])
                    let b = bf16Round(biases[r * nGroups + g])
                    for j in 0 ..< gs {
                        let qv = Float(q[r][g * gs + j])
                        acc += (qv * s + b) * bf16Round(input[g * gs + j])
                    }
                }
                expected.append(acc)
            }

            // bf16 storage: shift floats >> 16 to extract top 16 bits
            let scalesBits: [UInt16] = scales.map {
                UInt16(truncatingIfNeeded: $0.bitPattern >> 16)
            }
            let biasesBits: [UInt16] = biases.map {
                UInt16(truncatingIfNeeded: $0.bitPattern >> 16)
            }
            let inputBits: [UInt16] = input.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }

            let weight = Tensor.empty(shape: [outDim, inDim / 8], dtype: .u32)
            weight.copyIn(from: packed)
            let scalesT = Tensor.empty(shape: [outDim, nGroups], dtype: .bf16)
            scalesT.copyIn(from: scalesBits)
            let biasesT = Tensor.empty(shape: [outDim, nGroups], dtype: .bf16)
            biasesT.copyIn(from: biasesBits)
            let inputT = Tensor.empty(shape: [inDim], dtype: .bf16)
            inputT.copyIn(from: inputBits)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.dequantGemvInt4(
                    weight: weight, scales: scalesT, biases: biasesT,
                    input: inputT, groupSize: gs, on: cb
                )
            }

            let got = out.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            for i in 0 ..< outDim {
                // bf16 has ~3 decimal digits — use a generous tolerance
                let tol = max(abs(expected[i]) * 0.05, 1.0)
                #expect(
                    abs(got[i] - expected[i]) < tol,
                    "row \(i): got \(got[i]) expected \(expected[i]) (tol \(tol))")
            }
        }
    }

    @Test("int4 dequant gemv matches CPU dequant + gemv (f32 scales/biases)")
    func roundTripF32() {
        autoreleasepool {
            // Build quantized weights for 2 output rows × 16 input dims, group_size=8 → 2 groups per row.
            // Per row, per group: scale and bias.
            let q: [[UInt32]] = [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  // row 0
                [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  // row 1
            ]
            let scales: [Float] = [0.1, 0.2, 0.3, 0.4]  // [row, group]
            let biases: [Float] = [0.0, -1.0, 0.5, 0.25]
            let input: [Float] = (0 ..< Self.inDim).map { Float($0) - 7 }

            // Pack weight: per row, 2 groups × 1 packed uint32 = 2 uint32s per row
            var packed: [UInt32] = []
            for row in 0 ..< Self.outDim {
                for g in 0 ..< (Self.inDim / Self.groupSize) {
                    let nibbles = Array(q[row][g * Self.groupSize ..< (g + 1) * Self.groupSize])
                    packed.append(Self.pack8(nibbles))
                }
            }

            // CPU reference
            var expected: [Float] = []
            for row in 0 ..< Self.outDim {
                var acc: Float = 0
                for g in 0 ..< (Self.inDim / Self.groupSize) {
                    let s = scales[row * (Self.inDim / Self.groupSize) + g]
                    let b = biases[row * (Self.inDim / Self.groupSize) + g]
                    for j in 0 ..< Self.groupSize {
                        let qv = Float(q[row][g * Self.groupSize + j])
                        let w = qv * s + b
                        acc += w * input[g * Self.groupSize + j]
                    }
                }
                expected.append(acc)
            }

            // Allocate tensors
            let weight = Tensor.empty(shape: [Self.outDim, Self.inDim / 8], dtype: .u32)
            weight.copyIn(from: packed)
            let scalesT = Tensor.empty(
                shape: [Self.outDim, Self.inDim / Self.groupSize], dtype: .f32)
            scalesT.copyIn(from: scales)
            let biasesT = Tensor.empty(
                shape: [Self.outDim, Self.inDim / Self.groupSize], dtype: .f32)
            biasesT.copyIn(from: biases)
            let inputT = Tensor.empty(shape: [Self.inDim], dtype: .f32)
            inputT.copyIn(from: input)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.dequantGemvInt4(
                    weight: weight, scales: scalesT, biases: biasesT,
                    input: inputT, groupSize: Self.groupSize, on: cb
                )
            }

            let got = out.toArray(as: Float.self)
            for i in 0 ..< Self.outDim {
                #expect(
                    abs(got[i] - expected[i]) < 1e-3,
                    "row \(i): got \(got[i]) expected \(expected[i])")
            }
        }
    }

    // MARK: - dequantGather (MLX-format dequantizing embedding lookup)

    /// Direct test of `Ops.dequantGatherInt4`. Builds a tiny quantized
    /// embedding table, gathers a couple of token rows, and checks each
    /// dequantized element against the CPU `q * scale + bias` formula.
    ///
    /// Layout (see metaltile `ffai/dequant_gather.rs`):
    ///   weight  [vocab, hidden*bits/32] u32
    ///   scales  [vocab, hidden/groupSize] T
    ///   biases  [vocab, hidden/groupSize] T
    ///   indices [nTokens] u32  →  out [nTokens, hidden]
    @Test("dequantGatherInt4 — gather + dequant matches CPU q*scale+bias")
    func dequantGatherInt4MatchesCPU() {
        autoreleasepool {
            let vocab = 5
            let hidden = 16
            let gs = 8  // 2 groups per row
            let nGroups = hidden / gs

            // Deterministic 4-bit values per vocab row.
            var q = [[UInt32]](repeating: [], count: vocab)
            for r in 0 ..< vocab {
                q[r] = (0 ..< hidden).map { UInt32(($0 + r * 3) % 16) }
            }
            let scales: [Float] = (0 ..< (vocab * nGroups)).map { Float($0 + 1) * 0.02 }
            let biases: [Float] = (0 ..< (vocab * nGroups)).map { Float($0) * -0.01 }

            // Pack 8 nibbles per uint32 → hidden/8 = 2 words per row.
            var packed: [UInt32] = []
            for r in 0 ..< vocab {
                for i in stride(from: 0, to: hidden, by: 8) {
                    packed.append(Self.pack8(Array(q[r][i ..< i + 8])))
                }
            }

            let tokens: [UInt32] = [3, 0, 4]  // arbitrary lookup order
            let nTokens = tokens.count

            let weight = Tensor.empty(shape: [vocab, hidden / 8], dtype: .u32)
            weight.copyIn(from: packed)
            let scalesT = Tensor.empty(shape: [vocab, nGroups], dtype: .f32)
            scalesT.copyIn(from: scales)
            let biasesT = Tensor.empty(shape: [vocab, nGroups], dtype: .f32)
            biasesT.copyIn(from: biases)
            let idsT = Tensor.empty(shape: [nTokens], dtype: .u32)
            idsT.copyIn(from: tokens)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.dequantGatherInt4(
                    weight: weight, scales: scalesT, biases: biasesT,
                    tokenIds: idsT, hidden: hidden, groupSize: gs, on: cb
                )
            }

            let got = out.toArray(as: Float.self)
            for t in 0 ..< nTokens {
                let row = Int(tokens[t])
                for d in 0 ..< hidden {
                    let g = d / gs
                    let s = scales[row * nGroups + g]
                    let b = biases[row * nGroups + g]
                    let expected = Float(q[row][d]) * s + b
                    #expect(
                        abs(got[t * hidden + d] - expected) < 1e-3,
                        "token \(t) (row \(row)) d=\(d): got \(got[t * hidden + d]) expected \(expected)"
                    )
                }
            }
        }
    }

    /// Direct test of the 8-bit path through `Ops.dequantGather`.
    @Test("dequantGather(bits=8) — gather + dequant matches CPU q*scale+bias")
    func dequantGatherInt8MatchesCPU() {
        autoreleasepool {
            let vocab = 4
            let hidden = 16
            let gs = 8
            let nGroups = hidden / gs

            // Deterministic 8-bit values per vocab row.
            var q = [[UInt32]](repeating: [], count: vocab)
            for r in 0 ..< vocab {
                q[r] = (0 ..< hidden).map { UInt32(($0 * 5 + r * 17) & 0xFF) }
            }
            let scales: [Float] = (0 ..< (vocab * nGroups)).map { Float($0 + 1) * 0.003 }
            let biases: [Float] = (0 ..< (vocab * nGroups)).map { Float($0) * -0.05 }

            // Pack 4 bytes per uint32 → hidden/4 = 4 words per row.
            var packed: [UInt32] = []
            for r in 0 ..< vocab {
                for i in stride(from: 0, to: hidden, by: 4) {
                    packed.append(Self.pack4Bytes(Array(q[r][i ..< i + 4])))
                }
            }

            let tokens: [UInt32] = [1, 3]
            let nTokens = tokens.count

            let weight = Tensor.empty(shape: [vocab, hidden / 4], dtype: .u32)
            weight.copyIn(from: packed)
            let scalesT = Tensor.empty(shape: [vocab, nGroups], dtype: .f32)
            scalesT.copyIn(from: scales)
            let biasesT = Tensor.empty(shape: [vocab, nGroups], dtype: .f32)
            biasesT.copyIn(from: biases)
            let idsT = Tensor.empty(shape: [nTokens], dtype: .u32)
            idsT.copyIn(from: tokens)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.dequantGather(
                    weight: weight, scales: scalesT, biases: biasesT,
                    tokenIds: idsT, hidden: hidden, bits: 8, groupSize: gs, on: cb
                )
            }

            let got = out.toArray(as: Float.self)
            for t in 0 ..< nTokens {
                let row = Int(tokens[t])
                for d in 0 ..< hidden {
                    let g = d / gs
                    let s = scales[row * nGroups + g]
                    let b = biases[row * nGroups + g]
                    let expected = Float(q[row][d]) * s + b
                    #expect(
                        abs(got[t * hidden + d] - expected) < 1e-3,
                        "token \(t) (row \(row)) d=\(d): got \(got[t * hidden + d]) expected \(expected)"
                    )
                }
            }
        }
    }

    // MARK: - Batched dequantGemvInt4 (Two/Three/Four-projection variants)
    //
    // The batched wrappers do N back-to-back `dequant_gemv_int4_*`
    // dispatches inside one compute encoder, sharing the input
    // binding and the `in_dim` / `group_size` constants. Correctness
    // = N independent `dequantGemvInt4` calls produce the same output
    // tensors. The tests below build one input + N (weight, scales,
    // biases) triples, run both code paths, and require element-wise
    // equality.

    /// Build one realistic (weight, scales, biases, expectedOut) shape
    /// for a single int4 projection. Returns CPU-truth output for
    /// later equality checking.
    private static func makeInt4Projection(
        outDim: Int, inDim: Int, gs: Int,
        rowSeed: Int, scaleStep: Float, biasStep: Float,
        input: [Float]
    ) -> (weight: Tensor, scales: Tensor, biases: Tensor, expected: [Float]) {
        let nGroups = inDim / gs
        var q = [[UInt32]](repeating: [], count: outDim)
        for r in 0 ..< outDim {
            q[r] = (0 ..< inDim).map { UInt32(($0 + r * rowSeed) % 16) }
        }
        let scales: [Float] = (0 ..< (outDim * nGroups)).map { Float($0 + 1) * scaleStep }
        let biases: [Float] = (0 ..< (outDim * nGroups)).map { Float($0) * biasStep }
        var packed: [UInt32] = []
        for r in 0 ..< outDim {
            for i in stride(from: 0, to: inDim, by: 8) {
                let nibbles = Array(q[r][i ..< i + 8])
                packed.append(Self.pack8(nibbles))
            }
        }
        var expected: [Float] = []
        for r in 0 ..< outDim {
            var acc: Float = 0
            for g in 0 ..< nGroups {
                let s = scales[r * nGroups + g]
                let b = biases[r * nGroups + g]
                for j in 0 ..< gs {
                    let qv = Float(q[r][g * gs + j])
                    acc += (qv * s + b) * input[g * gs + j]
                }
            }
            expected.append(acc)
        }
        let weight = Tensor.empty(shape: [outDim, inDim / 8], dtype: .u32)
        weight.copyIn(from: packed)
        let scalesT = Tensor.empty(shape: [outDim, nGroups], dtype: .f32)
        scalesT.copyIn(from: scales)
        let biasesT = Tensor.empty(shape: [outDim, nGroups], dtype: .f32)
        biasesT.copyIn(from: biases)
        return (weight, scalesT, biasesT, expected)
    }

    @Test("dequantGemvInt4Two: two projections in one encoder match CPU")
    func dequantGemvInt4TwoCorrectness() {
        autoreleasepool {
            let outDim = 4
            let inDim = 128
            let gs = 64
            let input: [Float] = (0 ..< inDim).map { Float($0) * 0.1 - 6.4 }
            let p0 = Self.makeInt4Projection(
                outDim: outDim, inDim: inDim, gs: gs,
                rowSeed: 7, scaleStep: 0.01, biasStep: -0.005, input: input)
            let p1 = Self.makeInt4Projection(
                outDim: outDim, inDim: inDim, gs: gs,
                rowSeed: 13, scaleStep: 0.02, biasStep: 0.003, input: input)
            let inputT = Tensor.empty(shape: [inDim], dtype: .f32)
            inputT.copyIn(from: input)
            let out0 = Tensor.empty(shape: [outDim], dtype: .f32)
            let out1 = Tensor.empty(shape: [outDim], dtype: .f32)
            runAndWait { cb in
                Ops.dequantGemvInt4Two(
                    input: inputT,
                    w0: p0.weight, s0: p0.scales, b0: p0.biases, out0: out0,
                    w1: p1.weight, s1: p1.scales, b1: p1.biases, out1: out1,
                    groupSize: gs, on: cb)
            }
            let r0 = out0.toArray(as: Float.self)
            let r1 = out1.toArray(as: Float.self)
            for i in 0 ..< outDim {
                #expect(abs(r0[i] - p0.expected[i]) < 1e-2)
                #expect(abs(r1[i] - p1.expected[i]) < 1e-2)
            }
        }
    }

    @Test("dequantGemvInt4Three: three projections in one encoder match CPU")
    func dequantGemvInt4ThreeCorrectness() {
        autoreleasepool {
            let outDim = 4
            let inDim = 128
            let gs = 64
            let input: [Float] = (0 ..< inDim).map { Float($0) * 0.05 - 3.2 }
            let p0 = Self.makeInt4Projection(
                outDim: outDim, inDim: inDim, gs: gs,
                rowSeed: 7, scaleStep: 0.01, biasStep: -0.005, input: input)
            let p1 = Self.makeInt4Projection(
                outDim: outDim, inDim: inDim, gs: gs,
                rowSeed: 11, scaleStep: 0.02, biasStep: 0.003, input: input)
            let p2 = Self.makeInt4Projection(
                outDim: outDim, inDim: inDim, gs: gs,
                rowSeed: 17, scaleStep: 0.015, biasStep: -0.002, input: input)
            let inputT = Tensor.empty(shape: [inDim], dtype: .f32)
            inputT.copyIn(from: input)
            let out0 = Tensor.empty(shape: [outDim], dtype: .f32)
            let out1 = Tensor.empty(shape: [outDim], dtype: .f32)
            let out2 = Tensor.empty(shape: [outDim], dtype: .f32)
            runAndWait { cb in
                Ops.dequantGemvInt4Three(
                    input: inputT,
                    w0: p0.weight, s0: p0.scales, b0: p0.biases, out0: out0,
                    w1: p1.weight, s1: p1.scales, b1: p1.biases, out1: out1,
                    w2: p2.weight, s2: p2.scales, b2: p2.biases, out2: out2,
                    groupSize: gs, on: cb)
            }
            for i in 0 ..< outDim {
                #expect(abs(out0.toArray(as: Float.self)[i] - p0.expected[i]) < 1e-2)
                #expect(abs(out1.toArray(as: Float.self)[i] - p1.expected[i]) < 1e-2)
                #expect(abs(out2.toArray(as: Float.self)[i] - p2.expected[i]) < 1e-2)
            }
        }
    }

    @Test("dequantGemvInt4Four: four projections in one encoder match CPU")
    func dequantGemvInt4FourCorrectness() {
        autoreleasepool {
            let outDim = 4
            let inDim = 128
            let gs = 64
            let input: [Float] = (0 ..< inDim).map { Float($0) * 0.04 - 2.5 }
            let p0 = Self.makeInt4Projection(
                outDim: outDim, inDim: inDim, gs: gs,
                rowSeed: 7, scaleStep: 0.01, biasStep: -0.005, input: input)
            let p1 = Self.makeInt4Projection(
                outDim: outDim, inDim: inDim, gs: gs,
                rowSeed: 11, scaleStep: 0.02, biasStep: 0.003, input: input)
            let p2 = Self.makeInt4Projection(
                outDim: outDim, inDim: inDim, gs: gs,
                rowSeed: 17, scaleStep: 0.015, biasStep: -0.002, input: input)
            let p3 = Self.makeInt4Projection(
                outDim: outDim, inDim: inDim, gs: gs,
                rowSeed: 23, scaleStep: 0.025, biasStep: 0.001, input: input)
            let inputT = Tensor.empty(shape: [inDim], dtype: .f32)
            inputT.copyIn(from: input)
            let out0 = Tensor.empty(shape: [outDim], dtype: .f32)
            let out1 = Tensor.empty(shape: [outDim], dtype: .f32)
            let out2 = Tensor.empty(shape: [outDim], dtype: .f32)
            let out3 = Tensor.empty(shape: [outDim], dtype: .f32)
            runAndWait { cb in
                Ops.dequantGemvInt4Four(
                    input: inputT,
                    w0: p0.weight, s0: p0.scales, b0: p0.biases, out0: out0,
                    w1: p1.weight, s1: p1.scales, b1: p1.biases, out1: out1,
                    w2: p2.weight, s2: p2.scales, b2: p2.biases, out2: out2,
                    w3: p3.weight, s3: p3.scales, b3: p3.biases, out3: out3,
                    groupSize: gs, on: cb)
            }
            for i in 0 ..< outDim {
                #expect(abs(out0.toArray(as: Float.self)[i] - p0.expected[i]) < 1e-2)
                #expect(abs(out1.toArray(as: Float.self)[i] - p1.expected[i]) < 1e-2)
                #expect(abs(out2.toArray(as: Float.self)[i] - p2.expected[i]) < 1e-2)
                #expect(abs(out3.toArray(as: Float.self)[i] - p3.expected[i]) < 1e-2)
            }
        }
    }

    @Test("dequantGemvInt4Many: N projections with distinct inputs match standalone calls")
    func dequantGemvInt4ManyCorrectness() {
        autoreleasepool {
            // N=3 projections, each with its own input + output.
            let outDim = 4
            let inDim = 128
            let gs = 64
            let inputs: [[Float]] = [
                (0 ..< inDim).map { Float($0) * 0.05 - 3.2 },
                (0 ..< inDim).map { Float($0) * 0.04 - 2.5 },
                (0 ..< inDim).map { Float($0) * 0.03 - 1.8 },
            ]
            var experts: [(weight: Tensor, scales: Tensor, biases: Tensor, expected: [Float])] = []
            for i in 0 ..< 3 {
                let p = Self.makeInt4Projection(
                    outDim: outDim, inDim: inDim, gs: gs,
                    rowSeed: 7 + i * 5, scaleStep: 0.01 + Float(i) * 0.005,
                    biasStep: -0.005 + Float(i) * 0.002, input: inputs[i])
                experts.append(p)
            }
            var inputTs: [Tensor] = []
            for i in 0 ..< 3 {
                let t = Tensor.empty(shape: [inDim], dtype: .f32)
                t.copyIn(from: inputs[i])
                inputTs.append(t)
            }
            let outs = (0 ..< 3).map { _ in Tensor.empty(shape: [outDim], dtype: .f32) }
            runAndWait { cb in
                Ops.dequantGemvInt4Many(
                    weights: experts.map { $0.weight },
                    scales: experts.map { $0.scales },
                    biases: experts.map { $0.biases },
                    inputs: inputTs, outputs: outs,
                    groupSize: gs, on: cb)
            }
            for i in 0 ..< 3 {
                let got = outs[i].toArray(as: Float.self)
                for j in 0 ..< outDim {
                    #expect(abs(got[j] - experts[i].expected[j]) < 1e-2)
                }
            }
        }
    }

    @Test("dequantGemvInt4ExpertIndexed: matches standalone dequantGemvInt4 for the picked expert")
    func dequantGemvInt4ExpertIndexedCorrectness() {
        autoreleasepool {
            let nExperts = 4
            let outDim = 4
            let inDim = 128
            let gs = 64
            let input: [Float] = (0 ..< inDim).map { Float($0) * 0.05 - 3.2 }
            // Build per-expert weight tensors AND a stacked weight tensor
            // that concatenates each expert's slab.
            var experts: [(weight: Tensor, scales: Tensor, biases: Tensor, expected: [Float])] = []
            for e in 0 ..< nExperts {
                let p = Self.makeInt4Projection(
                    outDim: outDim, inDim: inDim, gs: gs,
                    rowSeed: 7 + e * 5, scaleStep: 0.01 + Float(e) * 0.005,
                    biasStep: -0.005 + Float(e) * 0.002, input: input)
                experts.append(p)
            }
            // Stack: [nExperts, outDim, inDim/8]
            let packedPerRow = inDim / 8
            let nGroups = inDim / gs
            let weightsStacked = Tensor.empty(
                shape: [nExperts, outDim, packedPerRow], dtype: .u32)
            let scalesStacked = Tensor.empty(
                shape: [nExperts, outDim, nGroups], dtype: .f32)
            let biasesStacked = Tensor.empty(
                shape: [nExperts, outDim, nGroups], dtype: .f32)
            // Repack by copying each expert's data into the stacked tensor.
            var packedAll: [UInt32] = []
            var scalesAll: [Float] = []
            var biasesAll: [Float] = []
            for e in 0 ..< nExperts {
                packedAll.append(contentsOf: experts[e].weight.toArray(as: UInt32.self))
                scalesAll.append(contentsOf: experts[e].scales.toArray(as: Float.self))
                biasesAll.append(contentsOf: experts[e].biases.toArray(as: Float.self))
            }
            weightsStacked.copyIn(from: packedAll)
            scalesStacked.copyIn(from: scalesAll)
            biasesStacked.copyIn(from: biasesAll)

            let inputT = Tensor.empty(shape: [inDim], dtype: .f32)
            inputT.copyIn(from: input)

            // Pick expert 2.
            let pick: UInt32 = 2
            let pickT = Tensor.empty(shape: [1], dtype: .u32)
            pickT.copyIn(from: [pick])
            let out = Tensor.empty(shape: [outDim], dtype: .f32)
            runAndWait { cb in
                Ops.dequantGemvInt4ExpertIndexed(
                    weightsStacked: weightsStacked,
                    scalesStacked: scalesStacked,
                    biasesStacked: biasesStacked,
                    input: inputT, expertIndex: pickT,
                    groupSize: gs, on: cb, into: out)
            }
            let got = out.toArray(as: Float.self)
            let want = experts[Int(pick)].expected
            for i in 0 ..< outDim {
                #expect(abs(got[i] - want[i]) < 1e-2, "row \(i)")
            }
        }
    }

    @Test("moeRouterTopK f32 — top-K of (1, 5, 3, 4) at k=2 picks indices [1, 3]")
    func moeRouterTopKDeterministic() {
        autoreleasepool {
            // norm_topk_prob = true → weights are softmax over the two
            // selected logits and renormalise to 1.0. With logits
            // (1, 5, 3, 4), the top-2 are indices 1 and 3 (values 5, 4).
            // softmax([5, 4]) = (e^5, e^4) / (e^5 + e^4) ≈ (0.731, 0.269).
            let nExperts = 4
            let k = 2
            let logits = Tensor.empty(shape: [nExperts], dtype: .f32)
            logits.copyIn(from: [Float(1), 5, 3, 4])
            let indicesOut = Tensor.empty(shape: [k], dtype: .u32)
            let weightsOut = Tensor.empty(shape: [k], dtype: .f32)
            runAndWait { cb in
                Ops.moeRouterTopK(
                    logits: logits,
                    indicesOut: indicesOut, weightsOut: weightsOut,
                    nExperts: nExperts, k: k, normTopkProb: true, on: cb)
            }
            let idx = indicesOut.toArray(as: UInt32.self)
            let wts = weightsOut.toArray(as: Float.self)
            #expect(Set(idx) == Set([UInt32(1), 3]))
            // Sum should be ~1.0 (norm_topk_prob).
            #expect(abs(wts[0] + wts[1] - 1.0) < 1e-3)
            // Index 1 has the larger logit so its weight should dominate.
            let weightOf1 = idx[0] == 1 ? wts[0] : wts[1]
            #expect(weightOf1 > 0.7)
        }
    }
}
