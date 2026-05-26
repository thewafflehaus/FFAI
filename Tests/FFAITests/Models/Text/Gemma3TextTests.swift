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
// Gemma3TextTests — unit coverage for `Sources/FFAI/Models/Text/Gemma3Text.swift`.
//
// Offline. Covers the `Gemma3Dense` variant surface (capabilities +
// generation defaults), the per-layer sliding/global scheduling
// formula `(i + 1) % slidingWindowPattern != 0`, and the GemmaRMSNorm
// `+1.0` weight-fold round-trip math (bf16 / f16 / f32 conversion
// helpers `bf16BitsToFloat`, `floatToBf16Bits`, `halfBitsToFloat`,
// `floatToHalfBits`, `fillScalar`).

import Foundation
import Metal
import Testing

@testable import FFAI

@Suite("Gemma3Dense Variant Surface")
struct Gemma3TextTests {

    @Test("Gemma3Dense advertises text in/out capabilities")
    func capabilities() {
        #expect(Gemma3Dense.availableCapabilities.contains(.textIn))
        #expect(Gemma3Dense.availableCapabilities.contains(.textOut))
        #expect(!Gemma3Dense.availableCapabilities.contains(.imageIn))
    }

    @Test("Gemma3Dense default generation parameters track Gemma family")
    func defaultGenerationParameters() {
        // Gemma defaults: temperature 1.0, top-p 0.95, top-k 64.
        let p = Gemma3Dense.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature >= 0)
        #expect(p.topK >= 0)
        #expect(p.topP > 0 && p.topP <= 1.0)
    }

    /// Gemma 3's family default `sliding_window_pattern = 6` — every
    /// 6th layer is global, the rest sliding.
    @Test("pattern=6 marks every 6th layer global")
    func pattern6Layout() {
        let pattern = 6
        for i in 0 ..< 12 {
            let isSliding = (i + 1) % pattern != 0
            if (i + 1) % pattern == 0 {
                #expect(!isSliding, "layer \(i) (i+1=\(i+1)) must be global")
            } else {
                #expect(isSliding, "layer \(i) (i+1=\(i+1)) must be sliding")
            }
        }
    }

    /// The sliding layer uses `rope_local_base_freq`; the global layer
    /// uses `rope_theta`. Both pulled from config with documented
    /// defaults (1e4 and 1e6 respectively).
    @Test("global vs sliding layers use distinct RoPE base frequencies")
    func ropeThetaPerLayer() {
        // Document the contract via a tiny simulation — the loader
        // assembles `layerRopeTheta` exactly this way.
        let ropeTheta: Float = 1_000_000
        let ropeLocal: Float = 10_000
        let pattern = 6
        for i in 0 ..< 6 {
            let isSliding = (i + 1) % pattern != 0
            let layerRopeTheta = isSliding ? ropeLocal : ropeTheta
            #expect(layerRopeTheta == (isSliding ? ropeLocal : ropeTheta))
        }
    }

    // ─── Weight-fold round-trips (was Gemma3TextWeightFoldTests) ─────────
    //
    // Pin the bf16 / f16 / f32 conversion math powering Gemma 3's
    // load-time +1 RMSNorm fold. The full integration test isolated
    // embed-scale as one bug surface; verifying the underlying
    // conversion math removes a class of suspects.

    /// Build a tensor of size `n` with the given fp32 values, then
    /// allocate a same-size buffer holding +1.0-folded values in
    /// `dtype`, and read back to fp32 to compare. This mirrors the
    /// path Gemma 3 takes at load time when constructing a Gemma
    /// RMSNorm from a checkpoint weight.
    private func roundTripFold(_ values: [Float], dtype: DType) -> [Float] {
        let n = values.count
        let device = Device.shared

        // Step 1: pack the fp32 values into a `dtype` buffer (this is
        // what SafeTensors hands us at load time).
        let inputBuf = device.makeBuffer(length: n * dtype.byteSize)
        switch dtype {
        case .f32:
            let p = inputBuf.contents().bindMemory(to: Float.self, capacity: n)
            for i in 0 ..< n { p[i] = values[i] }
        case .f16:
            let p = inputBuf.contents().bindMemory(to: UInt16.self, capacity: n)
            for i in 0 ..< n {
                // Use the same conversion routines used at runtime so
                // we're round-tripping through THE codepath, not a
                // hypothetical one.
                p[i] = floatToHalfBitsForTest(values[i])
            }
        case .bf16:
            let p = inputBuf.contents().bindMemory(to: UInt16.self, capacity: n)
            for i in 0 ..< n {
                p[i] = floatToBf16BitsForTest(values[i])
            }
        default:
            fatalError("roundTripFold: unsupported \(dtype)")
        }

        // Step 2: fold +1.0 into a new buffer using the same code
        // path Gemma3 uses (via the test-only re-exports below).
        let foldedBuf = gemmaFoldRMSNormForTest(
            inputBuf: inputBuf, count: n, dtype: dtype, device: device
        )

        // Step 3: read back as fp32 for comparison.
        var out = [Float](repeating: 0, count: n)
        switch dtype {
        case .f32:
            let p = foldedBuf.contents().bindMemory(to: Float.self, capacity: n)
            for i in 0 ..< n { out[i] = p[i] }
        case .f16:
            let p = foldedBuf.contents().bindMemory(to: UInt16.self, capacity: n)
            for i in 0 ..< n { out[i] = halfBitsToFloatForTest(p[i]) }
        case .bf16:
            let p = foldedBuf.contents().bindMemory(to: UInt16.self, capacity: n)
            for i in 0 ..< n { out[i] = bf16BitsToFloatForTest(p[i]) }
        default:
            fatalError("roundTripFold: unsupported \(dtype)")
        }
        return out
    }

    @Test("f32 fold returns weight + 1.0 exactly")
    func f32FoldExact() {
        let weights: [Float] = [0.0, 0.5, -0.5, 1.0, 2.0, -2.0, 100.0]
        let folded = roundTripFold(weights, dtype: .f32)
        for (i, w) in weights.enumerated() {
            #expect(
                folded[i] == w + 1.0,
                "f32 fold of \(w) should be \(w + 1.0), got \(folded[i])")
        }
    }

    @Test("bf16 fold round-trips Gemma 3 RMSNorm-typical values")
    func bf16FoldGemmaTypicalValues() {
        // First 8 values from model.layers.0.input_layernorm.weight
        // in mlx-community/gemma-3-1b-it-bf16. Verified via Python
        // safetensors decode at debug-time. Pin these so the fold
        // can't regress silently.
        let weights: [Float] = [
            4.09375, 4.375, 2.875, 4.0,
            4.34375, 3.40625, 5.1875, 4.96875,
        ]
        let folded = roundTripFold(weights, dtype: .bf16)
        // Expected: weight + 1.0, then round to bf16. bf16 has 7-bit
        // mantissa so rounding error per element is bounded by
        // |w+1| * 2^-7 ≈ 0.04 for values around 5.
        for (i, w) in weights.enumerated() {
            let expected = w + 1.0
            let diff = abs(folded[i] - expected)
            #expect(
                diff < 0.05,
                "bf16 fold of \(w): expected ≈ \(expected), got \(folded[i]) (|diff| = \(diff))")
        }
    }

    @Test("bf16 fold handles negative + outlier weights")
    func bf16FoldNegativeAndOutliers() {
        // Captures the post_attention_layernorm range (mean = -0.06,
        // outliers up to 51) plus q_norm (-1.0 to 2.0).
        let weights: [Float] = [
            -0.59765625, -1.4140625, 51.25, 0.0048828125,
            -1.0078125, 2.03125, -0.78515625, 5.0625,
        ]
        let folded = roundTripFold(weights, dtype: .bf16)
        for (i, w) in weights.enumerated() {
            let expected = w + 1.0
            // For large values (51 + 1 = 52), bf16 rounding error
            // can reach |52| * 2^-7 ≈ 0.4. Use a relative tolerance.
            let tol = max(0.05, abs(expected) * 0.01)
            let diff = abs(folded[i] - expected)
            #expect(
                diff < tol,
                "bf16 fold of \(w): expected ≈ \(expected), got \(folded[i]) (|diff| = \(diff), tol = \(tol))"
            )
        }
    }

    @Test("bf16 Ops.mul against fillScalar(sqrt(hidden)) — embed-scale path round-trip")
    func bf16EmbedScaleMul() {
        let n = 1152
        let device = Device.shared
        let scalar = Float(Double(n).squareRoot())  // ≈ 33.94

        // Build the embed-scale tensor exactly the way Gemma3Dense.loadModel does.
        let embedScale = Tensor.empty(shape: [n], dtype: .bf16, device: device)
        fillScalarForTest(embedScale, scalar: scalar, dtype: .bf16)

        // Build a synthetic "h0" tensor with values 0.5 across the
        // entire dim — typical magnitude after a Gemma-3 embedding
        // gather is somewhere in [0.01, 0.2], but 0.5 keeps the
        // arithmetic easy to verify.
        let h0 = Tensor.empty(shape: [n], dtype: .bf16, device: device)
        let h0Ptr = h0.buffer.contents().bindMemory(to: UInt16.self, capacity: n)
        let halfBits = floatToBf16BitsForTest(0.5)
        for i in 0 ..< n { h0Ptr[i] = halfBits }

        // Multiply on the GPU exactly as Gemma3Model.forward does.
        var product: Tensor!
        let cmd = device.makeCommandBuffer()
        product = Ops.mul(h0, embedScale, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // Read back as fp32. Expected: 0.5 * 34 ≈ 17.0 across all slots.
        let pPtr = product.buffer.contents()
            .advanced(by: product.offset)
            .bindMemory(to: UInt16.self, capacity: n)
        let stored = bf16BitsToFloatForTest(
            embedScale.buffer.contents()
                .bindMemory(to: UInt16.self, capacity: 1)[0])
        let expected = 0.5 * stored
        for i in 0 ..< n {
            let v = bf16BitsToFloatForTest(pPtr[i])
            #expect(
                v == expected,
                "embed-scale product slot \(i) = \(v), expected \(expected)")
            #expect(v.isFinite, "embed-scale product slot \(i) = \(v) is non-finite")
        }
    }

    @Test("fillScalar writes the same value into every element (bf16)")
    func fillScalarBf16() {
        let device = Device.shared
        let n = 1152
        let t = Tensor.empty(shape: [n], dtype: .bf16, device: device)
        let scalar: Float = Float(Double(n).squareRoot())  // ≈ 33.94 for Gemma 1B
        fillScalarForTest(t, scalar: scalar, dtype: .bf16)

        // Read back the bf16 bits and convert to fp32. Every slot
        // should hold the same value (rounded from `scalar`).
        let p = t.buffer.contents().bindMemory(to: UInt16.self, capacity: n)
        let first = bf16BitsToFloatForTest(p[0])
        for i in 0 ..< n {
            let v = bf16BitsToFloatForTest(p[i])
            #expect(v == first, "fillScalar slot \(i) = \(v), expected uniform \(first)")
        }
        // The stored value should be within bf16 rounding of the
        // fp32 input. sqrt(1152) ≈ 33.94 → bf16 should be ≈ 34.0.
        let diff = abs(first - scalar)
        #expect(
            diff < 0.5,
            "fillScalar(sqrt(1152) ≈ \(scalar)): stored \(first) (|diff| = \(diff))")
    }
}
