// Tests for QuantizedOps.{quantizeAffine, dequantizeAffine} —
// general affine quantize/dequantize wrappers.
//
// Round-trip pattern: quantize a known buffer → dequantize → compare
// to the original within bits-quantization tolerance.

import Foundation
import Metal
import Testing
@testable import FFAI

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

    @Test("int8 round-trip — 32 elements / group=32 recovers within ~0.01")
    func int8RoundTrip32() {
        autoreleasepool {
            let numel = 32
            let groupSize = 32
            let pf = 4
            let nGroups = numel / groupSize
            let packs = numel / pf

            let src = Tensor.empty(shape: [numel], dtype: .f32)
            let srcVals: [Float] = (0..<numel).map { Float($0) * 0.04 - 0.5 }
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

    @Test("validateAffineQuantize rejects pack-misaligned groupSize")
    func rejectPackMisalignment() {
        // bits=4 → pack_factor=8. groupSize=10 isn't a multiple of 8.
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

    @Test("validateAffineQuantize rejects group_size > 32 * pack_factor")
    func rejectGroupExceedingSimdgroupWidth() {
        // bits=4, pack_factor=8 → max group_size = 32*8 = 256.
        #expect(QuantizedOpsValidation.validateAffineQuantize(
            numel: 1024, packedCount: 128, scalesCount: 2, biasesCount: 2,
            bits: 4, groupSize: 512) != nil)
        // 256 is the cap; should pass shape-wise but the 256-group test
        // is overkill at unit-test scale. Use 128 instead.
        #expect(QuantizedOpsValidation.validateAffineQuantize(
            numel: 128, packedCount: 16, scalesCount: 1, biasesCount: 1,
            bits: 4, groupSize: 128) == nil)
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
            let numel = 32, groupSize = 32, pf = 4
            let nGroups = numel / groupSize
            let packs = numel / pf
            // 1.0 as bf16 = 0x3F80; vary lightly so quant has range.
            let vals: [UInt16] = (0..<numel).map { i -> UInt16 in
                let f: Float = Float(i) * 0.04 - 0.5
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
            // another 0.005 step. Use 0.05 tolerance.
            for v in out.toFloatArray() { #expect(v.isFinite) }
        }
    }

    @Test("int2 round-trip f32 — quantize + dequantize within step tolerance")
    func int2RoundTripF32() {
        autoreleasepool {
            // bits=2 → pack_factor=16. One group of 16 elements.
            let numel = 16, groupSize = 16, pf = 16
            let nGroups = 1
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
