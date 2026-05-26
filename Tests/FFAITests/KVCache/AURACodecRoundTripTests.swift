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
// AURACodecRoundTripTests — encode → dequant round-trip on a known
// synthetic vector at each supported bit width, asserting the
// reconstruction error stays within the per-bit-width tolerance the
// codec is supposed to deliver.
//
// Origin: a degenerate-output regression in the aura8v4 / aura8v8
// integration tests pointed at the 8-bit codec. Bisection through the
// metaltile pass pipeline tracked it to the const_fold pass — it ran
// only on nested blocks, never on `kernel.body`, so the bits=8 dequant
// kernel ended up with a fold-eligible trip count that hid from the
// unroll pass and the loop body was DCE'd out, leaving an empty
// `for (...)` in the emitted MSL. (See the
// `fix(const_fold): fold the entry block too` commit on metaltile.)
//
// After the fix, the unit-level codec is mathematically correct —
// 8-bit mean error ≈ 0.0004 vs the prior ~0.07. These tests pin that
// behaviour so any future codec regression surfaces at the kernel
// level instead of waiting on the slow model-level integration tests.
// Model-level quality at identity rotation is a separate concern
// tracked under Phase 5d.E (SRHT rotation).

import Foundation
import Metal
import Testing

@testable import FFAI

@Suite("AURA codec round-trip")
struct AURACodecRoundTripTests {

    /// Generate a synthetic [n] vector that mimics a real K/V slice:
    /// near-zero-centered with a thin tail. Deterministic so the
    /// per-bit-width error is stable across runs.
    private func makeUnitNormSlice(n: Int, seed: UInt64) -> [Float] {
        // Box-Muller pair from a Mersenne-Twister-shaped LCG. Standard
        // normal samples normalised to unit L2 norm — matches the
        // distribution the AURA Lloyd-Max boundaries were trained on
        // (rotated K/V rows are approximately unit-norm Gaussian).
        var state = seed
        @inline(__always) func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 32) / Float(UInt32.max) * 2.0 - 1.0
        }
        var vals = [Float](repeating: 0, count: n)
        var i = 0
        while i < n {
            // Box-Muller. Skip degenerate u1 to avoid log(0).
            var u1 = next() * 0.5 + 0.5
            if u1 < 1e-7 { u1 = 1e-7 }
            let u2 = next() * 0.5 + 0.5
            let r = sqrt(-2.0 * log(u1))
            let theta = 2.0 * Float.pi * u2
            vals[i] = r * cos(theta)
            if i + 1 < n { vals[i + 1] = r * sin(theta) }
            i += 2
        }
        // L2-normalise to unit norm.
        let ssq = vals.reduce(0) { $0 + $1 * $1 }
        let invNorm = 1.0 / sqrt(ssq)
        return vals.map { $0 * invNorm }
    }

    /// Run a single (input → auraEncode → auraDequantRotated → out)
    /// round-trip and return the worst-case absolute error vs the
    /// original input. headDim=128 + identity rotation matches the
    /// AURAQuantizedKVCache first-light shape.
    private func roundTripWorstError(input: [Float], bits: Int) -> Float {
        precondition(input.count == 128, "test fixture: headDim=128")
        let device = Device.shared
        let headDim = 128
        let packedWidth = AURACodebook.packedWidth(dim: headDim, bits: bits)

        // Inputs. Identity rotation means encode pipes the raw values
        // through the rotation matmul unchanged.
        let inputT = Tensor.empty(shape: [1, headDim], dtype: .f32, device: device)
        inputT.copyIn(from: input)

        let rotationData = AURARotation.identityMatrix(dim: headDim)
        let rotation = Tensor.empty(shape: [headDim, headDim], dtype: .f32, device: device)
        rotation.copyIn(from: rotationData)

        let centroids = AURACodebook.centroids(dim: headDim, bits: bits)
        let boundaries = AURACodebook.boundaries(dim: headDim, bits: bits)
        let codebookT = Tensor.empty(shape: [centroids.count], dtype: .f32, device: device)
        codebookT.copyIn(from: centroids)
        let boundariesT = Tensor.empty(shape: [boundaries.count], dtype: .f32, device: device)
        boundariesT.copyIn(from: boundaries)

        // Encode targets.
        let packedT = Tensor.empty(shape: [1, packedWidth], dtype: .u32, device: device)
        packedT.zero()
        let normsT = Tensor.empty(shape: [1], dtype: .f32, device: device)
        normsT.zero()

        // Encode.
        let cmd1 = device.makeCommandBuffer()
        Ops.auraEncode(
            input: inputT, rotation: rotation,
            boundaries: boundariesT, codebook: codebookT,
            packedOut: packedT, normsOut: normsT,
            rows: 1, dim: headDim, packedWidth: packedWidth, bits: bits,
            on: cmd1
        )
        cmd1.commit()
        cmd1.waitUntilCompleted()

        // Dequant. nKVHeads=1, tokens=1.
        let outT = Tensor.empty(shape: [1, 1, headDim], dtype: .f32, device: device)
        outT.zero()
        let cmd2 = device.makeCommandBuffer()
        Ops.auraDequantRotated(
            packed: packedT, norms: normsT, codebook: codebookT,
            into: outT,
            nKVHeads: 1, dim: headDim, packedWidth: packedWidth,
            tokens: 1, bits: bits, on: cmd2
        )
        cmd2.commit()
        cmd2.waitUntilCompleted()

        let reconstructed = outT.toArray(as: Float.self)
        var maxErr: Float = 0
        for i in 0 ..< headDim {
            let err = abs(reconstructed[i] - input[i])
            if err > maxErr { maxErr = err }
        }
        return maxErr
    }

    /// Average reconstruction error — complements the worst-case so
    /// we catch "off everywhere by 50%" failure modes that don't
    /// show up in a single max.
    private func roundTripMeanError(input: [Float], bits: Int) -> Float {
        // Reuse the worst-case path but also compute the mean. To
        // keep the file small, we just dispatch + read back twice;
        // GPU dispatch is fast on small shapes.
        let device = Device.shared
        let headDim = 128
        let packedWidth = AURACodebook.packedWidth(dim: headDim, bits: bits)

        let inputT = Tensor.empty(shape: [1, headDim], dtype: .f32, device: device)
        inputT.copyIn(from: input)
        let rotationData = AURARotation.identityMatrix(dim: headDim)
        let rotation = Tensor.empty(shape: [headDim, headDim], dtype: .f32, device: device)
        rotation.copyIn(from: rotationData)
        let centroids = AURACodebook.centroids(dim: headDim, bits: bits)
        let boundaries = AURACodebook.boundaries(dim: headDim, bits: bits)
        let codebookT = Tensor.empty(shape: [centroids.count], dtype: .f32, device: device)
        codebookT.copyIn(from: centroids)
        let boundariesT = Tensor.empty(shape: [boundaries.count], dtype: .f32, device: device)
        boundariesT.copyIn(from: boundaries)
        let packedT = Tensor.empty(shape: [1, packedWidth], dtype: .u32, device: device)
        packedT.zero()
        let normsT = Tensor.empty(shape: [1], dtype: .f32, device: device)
        normsT.zero()

        let cmd1 = device.makeCommandBuffer()
        Ops.auraEncode(
            input: inputT, rotation: rotation,
            boundaries: boundariesT, codebook: codebookT,
            packedOut: packedT, normsOut: normsT,
            rows: 1, dim: headDim, packedWidth: packedWidth, bits: bits,
            on: cmd1)
        cmd1.commit()
        cmd1.waitUntilCompleted()

        let outT = Tensor.empty(shape: [1, 1, headDim], dtype: .f32, device: device)
        outT.zero()
        let cmd2 = device.makeCommandBuffer()
        Ops.auraDequantRotated(
            packed: packedT, norms: normsT, codebook: codebookT,
            into: outT,
            nKVHeads: 1, dim: headDim, packedWidth: packedWidth,
            tokens: 1, bits: bits, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()

        let reconstructed = outT.toArray(as: Float.self)
        var sumErr: Float = 0
        for i in 0 ..< headDim {
            sumErr += abs(reconstructed[i] - input[i])
        }
        return sumErr / Float(headDim)
    }

    @Test("aura4 round-trip on a unit-norm slice")
    func aura4Roundtrip() {
        autoreleasepool {
            let input = makeUnitNormSlice(n: 128, seed: 12345)
            let worst = roundTripWorstError(input: input, bits: 4)
            let mean = roundTripMeanError(input: input, bits: 4)
            // 4-bit Lloyd-Max on a unit-norm Beta(d=128) slice should
            // achieve ≈ 0.02 mean abs error (16 levels packed densely
            // around 0, longer tails). Pinning at 0.05 mean / 0.20
            // worst-case so future codebook tweaks don't regress
            // silently.
            print("[aura4 roundtrip] mean=\(mean) worst=\(worst)")
            #expect(mean < 0.05, "aura4 mean reconstruction error \(mean) > 0.05")
            #expect(worst < 0.20, "aura4 worst-case reconstruction error \(worst) > 0.20")
        }
    }

    /// Single-vector encode-then-read-packed-buffer diagnostic.
    /// Encodes a known input, reads the packed u32 buffer back, and
    /// extracts each codebook index by hand. Then compares the
    /// extracted index against the centroid that's actually closest
    /// to the input value. Localises which side of the codec is
    /// broken when round-trip fails.
    @Test("aura8 encode produces the nearest-centroid indices for known inputs")
    func aura8EncodeIndices() {
        autoreleasepool {
            let device = Device.shared
            let headDim = 128
            let bits = 8
            let packedWidth = AURACodebook.packedWidth(dim: headDim, bits: bits)
            let levels = 1 << bits  // 256
            let centroids = AURACodebook.centroids(dim: headDim, bits: bits)
            let boundaries = AURACodebook.boundaries(dim: headDim, bits: bits)

            // Sanity: codebook + boundaries are the expected length.
            #expect(centroids.count == levels)
            #expect(boundaries.count == levels - 1)
            // Monotonic.
            for i in 1 ..< centroids.count {
                #expect(
                    centroids[i] > centroids[i - 1],
                    "centroids[\(i)]=\(centroids[i]) ≤ centroids[\(i-1)]=\(centroids[i-1])")
            }
            for i in 1 ..< boundaries.count {
                #expect(
                    boundaries[i] > boundaries[i - 1],
                    "boundaries[\(i)]=\(boundaries[i]) ≤ boundaries[\(i-1)]=\(boundaries[i-1])")
            }

            // Build a synthetic input where each dim has a known value.
            // dim 0 = 0.0, dim 1 = +0.1, dim 2 = -0.05, dim 3 = +0.32 (max),
            // dim 4 onward = 0.0.
            var input = [Float](repeating: 0, count: headDim)
            input[0] = 0.0
            input[1] = 0.1
            input[2] = -0.05
            input[3] = 0.32  // near max
            input[4] = -0.32  // near min
            // L2-normalise so the encoder's per-vector norm step
            // doesn't mangle our injected values. The codec stores
            // values in unit-norm space + a norm-correction scalar.
            let ssq = input.reduce(0) { $0 + $1 * $1 }
            let scale = 1.0 / sqrt(ssq)
            input = input.map { $0 * scale }
            // Recompute the expected post-normalize values.
            let postNorm = input
            // What centroid index do we expect for each dim?
            // The encoder's count-of-boundaries-exceeded rule gives:
            //   idx = number of boundaries i where rotated > boundaries[i]
            // Since rotation is identity, rotated = input value.
            func expectedIndex(for v: Float) -> Int {
                var idx = 0
                for b in boundaries where v > b { idx += 1 }
                return idx
            }
            let expectedIndices = postNorm.map(expectedIndex(for:))

            // Run encode.
            let inputT = Tensor.empty(shape: [1, headDim], dtype: .f32, device: device)
            inputT.copyIn(from: input)
            let rotation = Tensor.empty(shape: [headDim, headDim], dtype: .f32, device: device)
            rotation.copyIn(from: AURARotation.identityMatrix(dim: headDim))
            let codebookT = Tensor.empty(shape: [centroids.count], dtype: .f32, device: device)
            codebookT.copyIn(from: centroids)
            let boundariesT = Tensor.empty(shape: [boundaries.count], dtype: .f32, device: device)
            boundariesT.copyIn(from: boundaries)
            let packedT = Tensor.empty(shape: [1, packedWidth], dtype: .u32, device: device)
            packedT.zero()
            let normsT = Tensor.empty(shape: [1], dtype: .f32, device: device)
            normsT.zero()

            let cmd = device.makeCommandBuffer()
            Ops.auraEncode(
                input: inputT, rotation: rotation,
                boundaries: boundariesT, codebook: codebookT,
                packedOut: packedT, normsOut: normsT,
                rows: 1, dim: headDim, packedWidth: packedWidth, bits: bits,
                on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            // Read packed buffer + extract indices for the first 5 dims.
            let packed = packedT.toArray(as: UInt32.self)
            let mask: UInt32 = (1 << bits) - 1
            func extract(dim d: Int) -> Int {
                let bitOff = d * bits
                let wordIdx = bitOff / 32
                let shift = bitOff & 31
                return Int((packed[wordIdx] >> shift) & mask)
            }
            for d in 0 ..< 5 {
                let actual = extract(dim: d)
                let expected = expectedIndices[d]
                let actualCentroid = centroids[actual]
                let expectedCentroid = centroids[expected]
                print(
                    "[aura8 idx d=\(d)] input=\(input[d]) expected_idx=\(expected) (centroid=\(expectedCentroid)) actual_idx=\(actual) (centroid=\(actualCentroid))"
                )
                // Allow ±1 off (closest neighbour on either side of a
                // boundary is acceptable rounding). Off by more than
                // that = real bug.
                #expect(
                    abs(actual - expected) <= 1,
                    "dim \(d): actual_idx=\(actual) vs expected_idx=\(expected) — off by \(actual - expected)"
                )
            }
        }
    }

    @Test("aura8 round-trip on the SAME slice — should be strictly BETTER than aura4")
    func aura8Roundtrip() {
        autoreleasepool {
            // Same input as aura4 — the comparison is what we care
            // about. 8-bit has 16× more quantization levels than 4-bit
            // so the mean + worst-case error should be lower.
            let input = makeUnitNormSlice(n: 128, seed: 12345)
            let mean4 = roundTripMeanError(input: input, bits: 4)
            let mean8 = roundTripMeanError(input: input, bits: 8)
            let worst8 = roundTripWorstError(input: input, bits: 8)
            print("[aura8 roundtrip] mean=\(mean8) worst=\(worst8) (aura4 mean=\(mean4))")
            // The 8-bit codec MUST do at least as well as 4-bit. If
            // it doesn't, the codec is broken — which is exactly the
            // aura8v8 model-level regression we're chasing.
            #expect(
                mean8 <= mean4,
                "aura8 reconstruction error \(mean8) should be ≤ aura4's \(mean4) — 8-bit has 16× more levels"
            )
            // Absolute mean tolerance. 8-bit on a unit Beta slice
            // should easily reach <0.005.
            #expect(
                mean8 < 0.01,
                "aura8 mean reconstruction error \(mean8) > 0.01 — codec is broken")
        }
    }
}
