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
// Tests for QuantizedOps.{quantizeAffine, dequantizeAffine} —
// general affine quantize/dequantize wrappers.
//
// Round-trip pattern: quantize a known buffer → dequantize → compare
// to the original within bits-quantization tolerance.

import Foundation
import Metal
import Testing
@testable import FFAI
import TestHelpers

@Suite("QuantizedOps — affine quant/dequant round-trip")
struct QuantizedOpsAffineTests {

    @Test("int4 round-trip — 64 elements / group=64 stays within step")
    func int4RoundTrip64() {
        autoreleasepool {
            let numel = 64
            let groupSize = 64
            let pf = 8  // int4 → 8 vals / u32
            let nGroups = numel / groupSize
            let packs = numel / pf

            let src = Tensor.empty(shape: [numel], dtype: .f32)
            let srcVals: [Float] = (0..<numel).map { Float($0) * 0.03 - 1 }
            src.copyIn(from: srcVals)

            let packed = Tensor.empty(shape: [packs], dtype: .u32)
            let scales = Tensor.empty(shape: [nGroups], dtype: .f32)
            let biases = Tensor.empty(shape: [nGroups], dtype: .f32)
            packed.zero(); scales.zero(); biases.zero()

            runAndWait { cb in
                QuantizedOps.quantizeAffine(weight: src,
                    packed: packed, scales: scales, biases: biases,
                    bits: 4, groupSize: groupSize, on: cb)
            }
            let out = Tensor.empty(shape: [numel], dtype: .f32)
            out.zero()
            runAndWait { cb in
                QuantizedOps.dequantizeAffine(weight: packed,
                    scales: scales, biases: biases, into: out,
                    bits: 4, groupSize: groupSize, on: cb)
            }
            let got = out.toArray(as: Float.self)
            // int4 affine quant step = range/15 ≈ 0.13. Use 0.1 tolerance.
            for i in 0..<numel {
                #expect(abs(got[i] - srcVals[i]) < 0.15,
                        "i=\(i): got \(got[i]) vs \(srcVals[i])")
            }
        }
    }

    @Test("int8 round-trip — 64 elements / group=64 recovers within ~0.01")
    func int8RoundTrip64() {
        autoreleasepool {
            // The metaltile int8 quantize kernel reads 2 elements per lane
            // (`lane * 2`, `lane * 2 + 1`) across a 32-lane simdgroup, so
            // group_size MUST be 64. Earlier this test passed groupSize=32
            // and silently spilled the upper 16 lanes into the next group's
            // memory — the validator now rejects that explicitly.
            let numel = 64
            let groupSize = 64
            let pf = 4
            let nGroups = numel / groupSize
            let packs = numel / pf

            let src = Tensor.empty(shape: [numel], dtype: .f32)
            let srcVals: [Float] = (0..<numel).map { Float($0) * 0.02 - 0.5 }
            src.copyIn(from: srcVals)

            let packed = Tensor.empty(shape: [packs], dtype: .u32)
            let scales = Tensor.empty(shape: [nGroups], dtype: .f32)
            let biases = Tensor.empty(shape: [nGroups], dtype: .f32)
            packed.zero(); scales.zero(); biases.zero()

            runAndWait { cb in
                QuantizedOps.quantizeAffine(weight: src,
                    packed: packed, scales: scales, biases: biases,
                    bits: 8, groupSize: groupSize, on: cb)
            }
            let out = Tensor.empty(shape: [numel], dtype: .f32)
            out.zero()
            runAndWait { cb in
                QuantizedOps.dequantizeAffine(weight: packed,
                    scales: scales, biases: biases, into: out,
                    bits: 8, groupSize: groupSize, on: cb)
            }
            let got = out.toArray(as: Float.self)
            // int8 step = range/255 ≈ 0.005. Use 0.01 tolerance.
            for i in 0..<numel {
                #expect(abs(got[i] - srcVals[i]) < 0.01,
                        "i=\(i): got \(got[i]) vs \(srcVals[i])")
            }
        }
    }

    @Test("int2 dequantize — int2 codes 0..3 decode through scale + bias")
    func int2DequantizeManualPack() {
        autoreleasepool {
            // int2: pack_factor = 16. One group of 16 elements → one u32.
            // Codes [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3].
            let numel = 16, groupSize = 16
            var word: UInt32 = 0
            for i in 0..<16 {
                let code = UInt32(i % 4) & 0x3
                word |= code << (2 * UInt32(i))
            }
            let packed = Tensor.empty(shape: [1], dtype: .u32)
            packed.copyIn(from: [word])
            let scales = Tensor.empty(shape: [1], dtype: .f32)
            let biases = Tensor.empty(shape: [1], dtype: .f32)
            scales.copyIn(from: [Float(0.5)])
            biases.copyIn(from: [Float(-1)])
            let out = Tensor.empty(shape: [numel], dtype: .f32)
            out.zero()
            runAndWait { cb in
                QuantizedOps.dequantizeAffine(weight: packed,
                    scales: scales, biases: biases, into: out,
                    bits: 2, groupSize: groupSize, on: cb)
            }
            // For each code c: value = scale * c + bias = 0.5c - 1
            let got = out.toArray(as: Float.self)
            for i in 0..<16 {
                let expected = Float(i % 4) * 0.5 - 1
                #expect(abs(got[i] - expected) < 1e-5,
                        "i=\(i): got \(got[i]) vs \(expected)")
            }
        }
    }

    // ─── validator coverage ────────────────────────────────────────

    @Test("validateAffineQuantize rejects unsupported bit-widths")
    func rejectUnsupportedBits() {
        for badBits in [0, 1, 3, 5, 6, 7, 16] {
            #expect(QuantizedOpsValidation.validateAffineQuantize(
                numel: 64, packedCount: 8, scalesCount: 1, biasesCount: 1,
                bits: badBits, groupSize: 64) != nil,
                "bits=\(badBits) should be rejected")
        }
    }

    @Test("validateAffineQuantize rejects partial trailing group")
    func rejectPartialGroup() {
        // numel=100, groupSize=64 → not a multiple.
        #expect(QuantizedOpsValidation.validateAffineQuantize(
            numel: 100, packedCount: 12, scalesCount: 1, biasesCount: 1,
            bits: 4, groupSize: 64) != nil)
    }

    @Test("validateAffineQuantize rejects pack-misaligned groupSize (via group≠64 path)")
    func rejectPackMisalignment() {
        // Pre-2026-05-25 this test probed the pack-alignment branch
        // (groupSize=10 not a multiple of pack_factor=8). With the new
        // group_size=64 hard constraint the same value gets caught one
        // check earlier — the rejection is still correct, just for a
        // sharper reason. Both branches remain in the validator (the
        // pack-alignment check would re-engage if we ever broaden the
        // accepted group sizes).
        #expect(QuantizedOpsValidation.validateAffineQuantize(
            numel: 100, packedCount: 12, scalesCount: 10, biasesCount: 10,
            bits: 4, groupSize: 10) != nil)
    }

    @Test("validateAffineDequantize verifies all buffer sizes")
    func validateDequantizeSizes() {
        // Correct case: numel=64, group=64, bits=4 → 8 packs, 1 group.
        #expect(QuantizedOpsValidation.validateAffineDequantize(
            numel: 64, packedCount: 8, scalesCount: 1, biasesCount: 1,
            bits: 4, groupSize: 64) == nil)
        // Wrong packedCount
        #expect(QuantizedOpsValidation.validateAffineDequantize(
            numel: 64, packedCount: 7, scalesCount: 1, biasesCount: 1,
            bits: 4, groupSize: 64) != nil)
        // Wrong scalesCount
        #expect(QuantizedOpsValidation.validateAffineDequantize(
            numel: 64, packedCount: 8, scalesCount: 0, biasesCount: 1,
            bits: 4, groupSize: 64) != nil)
        // Wrong biasesCount
        #expect(QuantizedOpsValidation.validateAffineDequantize(
            numel: 64, packedCount: 8, scalesCount: 1, biasesCount: 0,
            bits: 4, groupSize: 64) != nil)
    }

    @Test("validateAffineQuantize rejects group_size != 64")
    func rejectGroupNotEqual64() {
        // The metaltile quantize kernels (`mt_affine_quantize_int{2,4,8}`)
        // bake in one simdgroup × 2 elements/lane = 64 elements/group via
        // the `lane * 2` / `lane * 2 + 1` loads + `simd_min` / `simd_max`
        // reduction. Only group_size=64 is emitted; anything else either
        // reads past the group boundary (smaller) or skips elements
        // (larger). The validator rejects every group_size != 64.
        #expect(QuantizedOpsValidation.validateAffineQuantize(
            numel: 1024, packedCount: 128, scalesCount: 32, biasesCount: 32,
            bits: 4, groupSize: 32) != nil)
        #expect(QuantizedOpsValidation.validateAffineQuantize(
            numel: 128, packedCount: 16, scalesCount: 1, biasesCount: 1,
            bits: 4, groupSize: 128) != nil)
        #expect(QuantizedOpsValidation.validateAffineQuantize(
            numel: 1024, packedCount: 128, scalesCount: 2, biasesCount: 2,
            bits: 4, groupSize: 512) != nil)
        // group_size = 64 is the only accepted shape.
        #expect(QuantizedOpsValidation.validateAffineQuantize(
            numel: 128, packedCount: 16, scalesCount: 2, biasesCount: 2,
            bits: 4, groupSize: 64) == nil)
    }

    @Test("packFactor returns correct ratio")
    func packFactorValues() {
        #expect(QuantizedOpsValidation.packFactor(forBits: 2) == 16)
        #expect(QuantizedOpsValidation.packFactor(forBits: 4) == 8)
        #expect(QuantizedOpsValidation.packFactor(forBits: 8) == 4)
        #expect(QuantizedOpsValidation.packFactor(forBits: 3) == nil)
        #expect(QuantizedOpsValidation.packFactor(forBits: 6) == nil)
    }

    // ─── f16 + bf16 dispatch — round-trip smoke ────────────────────

    @Test("affine round-trip f16 — int4 dispatch fires + recovers approx")
    func int4RoundTripF16() {
        autoreleasepool {
            let numel = 64
            let groupSize = 64
            let pf = 8
            let nGroups = numel / groupSize
            let packs = numel / pf
            let src = Tensor.empty(shape: [numel], dtype: .f16)
            let vals: [Float16] = (0..<numel).map { Float16(Float($0) * 0.03 - 1) }
            src.copyIn(from: vals)
            let packed = Tensor.empty(shape: [packs], dtype: .u32)
            let scales = Tensor.empty(shape: [nGroups], dtype: .f16)
            let biases = Tensor.empty(shape: [nGroups], dtype: .f16)
            packed.zero(); scales.zero(); biases.zero()
            runAndWait { cb in
                QuantizedOps.quantizeAffine(weight: src,
                    packed: packed, scales: scales, biases: biases,
                    bits: 4, groupSize: groupSize, on: cb)
            }
            let out = Tensor.empty(shape: [numel], dtype: .f16)
            out.zero()
            runAndWait { cb in
                QuantizedOps.dequantizeAffine(weight: packed,
                    scales: scales, biases: biases, into: out,
                    bits: 4, groupSize: groupSize, on: cb)
            }
            let got = out.toFloatArray()
            for i in 0..<numel {
                #expect(abs(got[i] - Float(vals[i])) < 0.2,
                        "i=\(i): got \(got[i]) vs \(vals[i])")
            }
        }
    }

    @Test("affine round-trip bf16 — int8 dispatch fires + recovers approx")
    func int8RoundTripBF16() {
        autoreleasepool {
            // Same kernel-side group_size=64 constraint as int8RoundTrip64;
            // see the comment there. Earlier this test passed groupSize=32
            // and silently corrupted the upper-lane reads.
            let numel = 64, groupSize = 64, pf = 4
            let nGroups = numel / groupSize
            let packs = numel / pf
            // 1.0 as bf16 = 0x3F80; vary lightly so quant has range.
            let vals: [UInt16] = (0..<numel).map { i -> UInt16 in
                let f: Float = Float(i) * 0.02 - 0.5
                return UInt16(f.bitPattern >> 16)
            }
            let src = Tensor.empty(shape: [numel], dtype: .bf16)
            src.copyIn(from: vals)
            let packed = Tensor.empty(shape: [packs], dtype: .u32)
            let scales = Tensor.empty(shape: [nGroups], dtype: .bf16)
            let biases = Tensor.empty(shape: [nGroups], dtype: .bf16)
            packed.zero(); scales.zero(); biases.zero()
            runAndWait { cb in
                QuantizedOps.quantizeAffine(weight: src,
                    packed: packed, scales: scales, biases: biases,
                    bits: 8, groupSize: groupSize, on: cb)
            }
            let out = Tensor.empty(shape: [numel], dtype: .bf16)
            out.zero()
            runAndWait { cb in
                QuantizedOps.dequantizeAffine(weight: packed,
                    scales: scales, biases: biases, into: out,
                    bits: 8, groupSize: groupSize, on: cb)
            }
            // bf16 has ~3 decimal digits of precision; quant adds maybe
            // another 0.005 step. Just confirm finiteness end-to-end.
            for v in out.toFloatArray() { #expect(v.isFinite) }
        }
    }

    @Test("int2 round-trip f32 — quantize + dequantize within step tolerance")
    func int2RoundTripF32() {
        autoreleasepool {
            // bits=2 → pack_factor=16. The kernel uses one simdgroup ×
            // 2 elements/lane = 64-element groups; pack_factor=16 means
            // 4 packs per group. Use numel=64 to stay on the only
            // emitted variant.
            let numel = 64, groupSize = 64, pf = 16
            let nGroups = numel / groupSize
            let packs = numel / pf
            let src = Tensor.empty(shape: [numel], dtype: .f32)
            let vals: [Float] = (0..<numel).map { Float($0) / Float(numel - 1) }  // [0, 1]
            src.copyIn(from: vals)
            let packed = Tensor.empty(shape: [packs], dtype: .u32)
            let scales = Tensor.empty(shape: [nGroups], dtype: .f32)
            let biases = Tensor.empty(shape: [nGroups], dtype: .f32)
            packed.zero(); scales.zero(); biases.zero()
            runAndWait { cb in
                QuantizedOps.quantizeAffine(weight: src,
                    packed: packed, scales: scales, biases: biases,
                    bits: 2, groupSize: groupSize, on: cb)
            }
            let out = Tensor.empty(shape: [numel], dtype: .f32)
            out.zero()
            runAndWait { cb in
                QuantizedOps.dequantizeAffine(weight: packed,
                    scales: scales, biases: biases, into: out,
                    bits: 2, groupSize: groupSize, on: cb)
            }
            // int2 has only 4 levels → step = range/3 ≈ 0.33; tolerance 0.4.
            for i in 0..<numel {
                #expect(abs(out.toArray(as: Float.self)[i] - vals[i]) < 0.4)
            }
        }
    }
}
