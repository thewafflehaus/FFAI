// Round-trip tests for the per-layer SRHT rotation that AURA Phase 5d.E
// Stage 1a wires through Qwen3 and Llama. These pin two things the
// model-level integration tests do NOT cover directly:
//
//   1. `Ops.auraRotatePerHead` truly inverts an SRHT rotation in the
//      activation dtype (bf16) — i.e. `Π^T · (Π · x) ≈ x` end-to-end on
//      the GPU. The unit-level `Ops.gemv` test only exercises tiny
//      shapes (k=2); SRHT at production headDim=128 is the actual hot
//      path and Apple's bf16 simd_sum can cumulate enough error to
//      surface as model drift, so we measure it here.
//
//   2. The AURA codec (encode + dequant) with a non-identity SRHT
//      rotation produces output that approximates `Π · input`. The
//      metaltile-side GPU correctness test for `aura_encode` only
//      exercises identity rotation; this fills the gap so a future
//      regression in the encode kernel's rotation-matmul stage is
//      caught at the Swift layer.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("AURA SRHT round-trip")
struct AURASRHTRoundTripTests {

    @Test("Π^T · (Π · x) ≈ x for bf16 at headDim=128, nHeads=2")
    func srhtInverseBf16() {
        autoreleasepool {
            let dim = 128
            let piF32 = AURARotation.srhtMatrix(dim: dim, seed: 0)
            var piTF32 = [Float](repeating: 0, count: dim * dim)
            for i in 0..<dim {
                for j in 0..<dim {
                    piTF32[j * dim + i] = piF32[i * dim + j]
                }
            }
            // Cast to bf16 bit-pattern truncation (top 16 bits of f32).
            let piBits = piF32.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let piTBits = piTF32.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let pi = Tensor.empty(shape: [dim, dim], dtype: .bf16)
            pi.copyIn(from: piBits)
            let piT = Tensor.empty(shape: [dim, dim], dtype: .bf16)
            piT.copyIn(from: piTBits)

            let nHeads = 2
            let n = nHeads * dim
            var raw = [Float](repeating: 0, count: n)
            var state: UInt64 = 0xdeadbeef
            @inline(__always) func nxt() -> Float {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                return Float(state >> 32) / Float(UInt32.max) - 0.5
            }
            for i in 0..<n { raw[i] = nxt() }
            let rawBits = raw.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
            let x = Tensor.empty(shape: [n], dtype: .bf16)
            x.copyIn(from: rawBits)

            var roundTrip: Tensor!
            runAndWait { cb in
                let rotated = Ops.auraRotatePerHead(x, rotation: pi,
                                                    nHeads: nHeads, headDim: dim, on: cb)
                roundTrip = Ops.auraRotatePerHead(rotated, rotation: piT,
                                                  nHeads: nHeads, headDim: dim, on: cb)
            }
            let outBits = roundTrip.toArray(as: UInt16.self)
            let outF32 = outBits.map { Float(bitPattern: UInt32($0) << 16) }
            let inF32 = rawBits.map { Float(bitPattern: UInt32($0) << 16) }
            var maxErr: Float = 0
            for i in 0..<n { maxErr = max(maxErr, abs(outF32[i] - inF32[i])) }
            // Two bf16 gemv passes through a headDim=128 matrix accumulate
            // ~1-2% error per element worst-case.
            #expect(maxErr < 0.05, "bf16 SRHT Π^T·(Π·x) round-trip max err=\(maxErr)")
        }
    }

    @Test("aura_encode + aura_dequant_rotated with SRHT Π reproduces Π·input")
    func auraCodecSrht() {
        autoreleasepool {
            let dim = 128
            let bits = 8  // 8-bit codec — lowest quantization error path.
            let pi = AURARotation.srhtMatrix(dim: dim, seed: 0)
            let rotation = Tensor.empty(shape: [dim, dim], dtype: .f32)
            rotation.copyIn(from: pi)

            let cb = AURACodebook.centroids(dim: dim, bits: bits)
            let bnd = AURACodebook.boundaries(dim: dim, bits: bits)
            let codebook = Tensor.empty(shape: [cb.count], dtype: .f32)
            codebook.copyIn(from: cb)
            let boundaries = Tensor.empty(shape: [bnd.count], dtype: .f32)
            boundaries.copyIn(from: bnd)

            // Synthetic Gaussian-ish unit-norm vector at production headDim.
            // L2-normalised so the encode kernel's norm correction stays
            // close to 1 — matches the codebook's calibration regime.
            var raw = [Float](repeating: 0, count: dim)
            var state: UInt64 = 0xfeedface
            @inline(__always) func nxt() -> Float {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                return Float(state >> 32) / Float(UInt32.max) - 0.5
            }
            for i in 0..<dim { raw[i] = nxt() }
            let norm = sqrt(raw.reduce(0.0) { $0 + Double($1*$1) })
            for i in 0..<dim { raw[i] = Float(Double(raw[i]) / norm) }

            let input = Tensor.empty(shape: [1, dim], dtype: .f32)
            input.copyIn(from: raw)

            let packedWidth = AURACodebook.packedWidth(dim: dim, bits: bits)
            let packedOut = Tensor.empty(shape: [1, packedWidth], dtype: .u32)
            packedOut.zero()
            let normsOut = Tensor.empty(shape: [1], dtype: .f32)

            runAndWait { cb in
                Ops.auraEncode(input: input, rotation: rotation,
                               boundaries: boundaries, codebook: codebook,
                               packedOut: packedOut, normsOut: normsOut,
                               rows: 1, dim: dim, packedWidth: packedWidth, bits: bits,
                               on: cb)
            }

            let dequant = Tensor.empty(shape: [1, 1, dim], dtype: .f32)
            runAndWait { cb in
                Ops.auraDequantRotated(packed: packedOut, norms: normsOut,
                                       codebook: codebook, into: dequant,
                                       nKVHeads: 1, dim: dim, packedWidth: packedWidth,
                                       tokens: 1, bits: bits, on: cb)
            }
            let got = dequant.toArray(as: Float.self)

            // CPU reference: Π·input.
            var piX = [Float](repeating: 0, count: dim)
            for d in 0..<dim {
                var s: Float = 0
                for j in 0..<dim { s += pi[d * dim + j] * raw[j] }
                piX[d] = s
            }

            var maxErr: Float = 0
            for i in 0..<dim { maxErr = max(maxErr, abs(got[i] - piX[i])) }
            // 8-bit codec round-trip error is ~5e-3 on synthetic unit-norm
            // vectors per AURACodecRoundTripTests; allow a small headroom.
            #expect(maxErr < 0.05, "codec+SRHT max err vs Π·input = \(maxErr)")
        }
    }
}
