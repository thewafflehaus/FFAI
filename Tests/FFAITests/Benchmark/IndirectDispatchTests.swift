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
// Correctness tests for the indirect-dispatch variants of FFAI's
// dequant-gemv kernels.
//
// Background: the GPU-router Day 1 work adds `_indirect` Swift wrappers
// that take an `MTLBuffer` carrying `MTLDispatchThreadgroupsIndirectArguments`
// (3 × u32 = threadgroup counts, NOT thread counts) instead of a host-
// computed `MTLSize` grid. Same kernel, same args, same PSO — only the
// dispatch arg source differs. The expected output is therefore bit-
// identical between the direct and indirect paths.
//
// These tests pin three risks from the Day 1 plan:
//
//   1. threadgroup count vs thread count — writing `outDim * 256` (the
//      thread count used by the direct dispatchThreads path) instead of
//      `outDim` (threadgroup count) into the indirect buffer would
//      over-dispatch by 256× and miscompute.
//   2. PSO cache name — the wrapper must look up `"dequant_gemv_int4_*"`
//      (underlying kernel name) NOT `"_indirect"`. The codegen could
//      have the suffix on either, and only the correctness test catches
//      the mismatch.
//   3. Hazard tracking on a host-written indirect buffer — the indirect
//      args are memcpy'd from the CPU into a shared-storage buffer,
//      then read by the GPU within the same command buffer. Metal's
//      default hazard tracker is per-resource; shared storage is
//      coherent so no explicit fence should be needed.

import Foundation
import Metal
import Testing
@testable import FFAI
import TestHelpers

@Suite("Indirect dispatch — dequant gemv int4")
struct IndirectDispatchTests {
    /// Helper: write `[outDim, 1, 1]` as `MTLDispatchThreadgroupsIndirectArguments`
    /// (3 × u32) into a freshly-allocated shared-storage buffer.
    static func makeIndirectBuffer(outDim: Int, on device: Device) -> MTLBuffer {
        let buf = device.makeBuffer(length: 3 * 4)
        let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: 3)
        ptr[0] = UInt32(outDim)
        ptr[1] = 1
        ptr[2] = 1
        return buf
    }

    /// Pack eight 4-bit nibbles into one u32 (little-endian, low nibble
    /// in low bits — matches mlx int4 storage).
    static func pack8(_ nibbles: [UInt32]) -> UInt32 {
        precondition(nibbles.count == 8)
        var word: UInt32 = 0
        for i in 0..<8 {
            word |= (nibbles[i] & 0xF) << (UInt32(i) * 4)
        }
        return word
    }

    @Test("int4 indirect dequant gemv matches direct dispatch bit-for-bit (bf16)")
    func bf16IndirectMatchesDirect() {
        autoreleasepool {
            // Realistic MoE expert shape: gateProj is [moeIntermediate, hidden].
            // Use a small but multi-group instance so both indirect and direct
            // exercise the reduction kernel's threadgroup tree.
            let outDim = 4
            let inDim = 128
            let gs = 64
            let nGroups = inDim / gs

            var q = [[UInt32]](repeating: [], count: outDim)
            for r in 0..<outDim {
                q[r] = (0..<inDim).map { UInt32(($0 + r * 3) % 16) }
            }
            let scales: [Float] = (0..<(outDim * nGroups)).map { Float($0 + 1) * 0.02 }
            let biases: [Float] = (0..<(outDim * nGroups)).map { Float($0) * -0.01 }
            let input: [Float] = (0..<inDim).map { Float($0) * 0.05 - 3.2 }

            var packed: [UInt32] = []
            for r in 0..<outDim {
                for i in stride(from: 0, to: inDim, by: 8) {
                    let nibbles = Array(q[r][i..<i+8])
                    packed.append(Self.pack8(nibbles))
                }
            }

            // bf16 storage: shift floats >> 16 to extract top 16 bits
            let scalesBits: [UInt16] = scales.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let biasesBits: [UInt16] = biases.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let inputBits: [UInt16]  = input.map  { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }

            let weight = Tensor.empty(shape: [outDim, inDim / 8], dtype: .u32)
            weight.copyIn(from: packed)
            let scalesT = Tensor.empty(shape: [outDim, nGroups], dtype: .bf16)
            scalesT.copyIn(from: scalesBits)
            let biasesT = Tensor.empty(shape: [outDim, nGroups], dtype: .bf16)
            biasesT.copyIn(from: biasesBits)
            let inputT = Tensor.empty(shape: [inDim], dtype: .bf16)
            inputT.copyIn(from: inputBits)

            // Direct path — reference.
            var directOut: Tensor!
            runAndWait { cb in
                directOut = Ops.dequantGemv(
                    weight: weight, scales: scalesT, biases: biasesT,
                    input: inputT, bits: 4, groupSize: gs, on: cb)
            }
            let directBits = directOut.toArray(as: UInt16.self)

            // Indirect path — same args, dispatch shape from a GPU buffer.
            let indirect = Self.makeIndirectBuffer(outDim: outDim, on: .shared)
            let indirectOut = Tensor.empty(shape: [outDim], dtype: .bf16)
            runAndWait { cb in
                Ops.dequantGemvIndirect(
                    weight: weight, scales: scalesT, biases: biasesT,
                    input: inputT, bits: 4, groupSize: gs,
                    indirectBuffer: indirect, indirectBufferOffset: 0,
                    on: cb, into: indirectOut)
            }
            let indirectBits = indirectOut.toArray(as: UInt16.self)

            // Bit-identical: same kernel, same args, different dispatch
            // arg source. Any divergence means the indirect arg buffer
            // had wrong contents (threadgroup count vs thread count) or
            // the PSO name didn't match.
            #expect(directBits == indirectBits,
                    "indirect bf16 output must bit-match direct (got: \(indirectBits.map { String($0, radix: 16) }), expected: \(directBits.map { String($0, radix: 16) }))")
        }
    }

    @Test("int4 indirect dequant gemv reads from non-zero offset correctly")
    func bf16IndirectAtOffset() {
        autoreleasepool {
            // Same setup as bf16IndirectMatchesDirect but the indirect
            // args live at offset 16 (the buffer holds two slots — slot
            // 0 is garbage [9999, 1, 1] that would over-dispatch; slot
            // 1 is the real [outDim, 1, 1]). Caller passes offset 16 so
            // the kernel reads the right slot. This pins the offset-
            // computation path that the Day 1.5 cross-layer chain will
            // rely on (one buffer, many offsets).
            let outDim = 4
            let inDim = 128
            let gs = 64
            let nGroups = inDim / gs

            var q = [[UInt32]](repeating: [], count: outDim)
            for r in 0..<outDim {
                q[r] = (0..<inDim).map { UInt32(($0 + r * 3) % 16) }
            }
            let scales: [Float] = (0..<(outDim * nGroups)).map { Float($0 + 1) * 0.02 }
            let biases: [Float] = (0..<(outDim * nGroups)).map { Float($0) * -0.01 }
            let input: [Float] = (0..<inDim).map { Float($0) * 0.05 - 3.2 }

            var packed: [UInt32] = []
            for r in 0..<outDim {
                for i in stride(from: 0, to: inDim, by: 8) {
                    let nibbles = Array(q[r][i..<i+8])
                    packed.append(Self.pack8(nibbles))
                }
            }

            let scalesBits: [UInt16] = scales.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let biasesBits: [UInt16] = biases.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let inputBits: [UInt16]  = input.map  { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }

            let weight = Tensor.empty(shape: [outDim, inDim / 8], dtype: .u32)
            weight.copyIn(from: packed)
            let scalesT = Tensor.empty(shape: [outDim, nGroups], dtype: .bf16)
            scalesT.copyIn(from: scalesBits)
            let biasesT = Tensor.empty(shape: [outDim, nGroups], dtype: .bf16)
            biasesT.copyIn(from: biasesBits)
            let inputT = Tensor.empty(shape: [inDim], dtype: .bf16)
            inputT.copyIn(from: inputBits)

            var directOut: Tensor!
            runAndWait { cb in
                directOut = Ops.dequantGemv(
                    weight: weight, scales: scalesT, biases: biasesT,
                    input: inputT, bits: 4, groupSize: gs, on: cb)
            }
            let directBits = directOut.toArray(as: UInt16.self)

            // Two-slot indirect buffer: slot 0 has garbage [9999, 1, 1],
            // slot 1 has the real [outDim, 1, 1]. Each slot = 12 bytes.
            let device = Device.shared
            let indirect = device.makeBuffer(length: 24)
            let ptr = indirect.contents().bindMemory(to: UInt32.self, capacity: 6)
            ptr[0] = 9999; ptr[1] = 1; ptr[2] = 1
            ptr[3] = UInt32(outDim); ptr[4] = 1; ptr[5] = 1

            let indirectOut = Tensor.empty(shape: [outDim], dtype: .bf16)
            runAndWait { cb in
                Ops.dequantGemvIndirect(
                    weight: weight, scales: scalesT, biases: biasesT,
                    input: inputT, bits: 4, groupSize: gs,
                    indirectBuffer: indirect, indirectBufferOffset: 12,
                    on: cb, into: indirectOut)
            }
            let indirectBits = indirectOut.toArray(as: UInt16.self)

            #expect(directBits == indirectBits,
                    "indirect bf16 at offset 12 must bit-match direct")
        }
    }
}
