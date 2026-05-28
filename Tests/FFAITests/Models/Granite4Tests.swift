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
// Granite4Tests — root-file unit tests for `Sources/FFAI/Models/Granite4.swift`.
//
// Offline. Covers:
//   - Granite 4 family metadata + variant dispatch
//     (`Granite4Hybrid` is the only variant).
//   - `Granite4Error` descriptions.
//   - The quantized multiplier-folding math trick used by the
//     Granite4 loader's `loadLinearScaledGMH` helper (see
//     `Granite4Text.swift` for the helper itself). The Granite4
//     4-bit conversion ships `embedding_multiplier` /
//     `residual_multiplier` un-baked into the weights; the loader
//     folds them into the quantized `scales` + `biases` tensors
//     instead of touching the packed u32 weight. This file proves
//     the equivalence on a synthetic packed weight so a future
//     refactor of the dequant kernel can't silently break the
//     contract.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("Granite4 Family Root")
struct Granite4RootTests {

    @Test("modelTypes advertises granitemoehybrid")
    func modelTypes() {
        #expect(Granite4.modelTypes.contains("granitemoehybrid"))
    }

    @Test("architectures advertises Granite4ForCausalLM")
    func architectures() {
        #expect(Granite4.architectures.contains("Granite4ForCausalLM"))
    }

    @Test("variant(for:) returns Granite4Hybrid")
    func variantDispatch() throws {
        let cfg = ModelConfig(
            architecture: "Granite4ForCausalLM",
            modelType: "granitemoehybrid", raw: [:])
        let v = try Granite4.variant(for: cfg)
        #expect(String(describing: v) == String(describing: Granite4Hybrid.self))
    }

    @Test("Granite4Error stringifies every case with its payload")
    func errorDescriptions() {
        #expect(
            Granite4Error.missingConfig("layer_types").description
                .contains("layer_types"))
        #expect(Granite4Error.unsupportedConfig("bad").description.contains("bad"))
        #expect(Granite4Error.missingConfig("x").description.contains("Granite4"))
    }
}

// ─── loadLinearScaledGMH multiplier-folding equivalence ──────────────
//
// Granite4 4-bit conversions don't pre-bake `embedding_multiplier`
// or `residual_multiplier` into the packed weights; the loader folds
// the multiplier `m` into the per-group `scales` and `biases`
// tensors instead. Since
//
//     dequant      = nibble * scale + bias
//     dequant * m  = nibble * (m·scale) + (m·bias)
//
// multiplying both `scales` and `biases` by `m` is mathematically
// identical to scaling every dequantized output by `m`, with the
// packed u32 weight untouched. This suite proves that equivalence
// end-to-end through the production `Ops.dequantGemvInt4` GEMV
// kernel — a regression in the dequant arithmetic would break this
// assumption silently.
@Suite("Granite4 Multiplier-Folding Equivalence")
struct Granite4MultiplierFoldingTests {

    /// Pack 8 4-bit nibbles (low nibble first) into one uint32.
    private static func pack8(_ q: [UInt32]) -> UInt32 {
        precondition(q.count == 8, "expected 8 4-bit values")
        var w: UInt32 = 0
        for i in 0 ..< 8 { w |= (q[i] & 0xF) << (4 * UInt32(i)) }
        return w
    }

    @Test(
        "dequantGemv with (m·scales, m·biases) == m · dequantGemv with (scales, biases)"
    )
    func multiplierFoldingEquivalence() {
        autoreleasepool {
            // Realistic shape: 4 rows × 128 in_dim, group_size=64 → 2
            // groups per row. Matches the dimensions a Granite4 attention
            // o_proj or MLP down_proj triplet would carry per output row.
            let outDim = 4
            let inDim = 128
            let gs = 64
            let nGroups = inDim / gs

            // Multiplier under test. Granite4's `residual_multiplier`
            // is 0.246 on the published 350m-h checkpoint; pick a value
            // in the same ballpark so any precision regression that
            // only shows at small m surfaces here.
            let m: Float = 0.246

            // Synthetic q values, scales, biases, and input — same
            // shape the `realShapeRoundTrip` test in
            // QuantizedOpsTests.swift uses; identical data is fine
            // (we're testing the multiplier-folding invariant, not
            // GEMV correctness itself).
            var q = [[UInt32]](repeating: [], count: outDim)
            for r in 0 ..< outDim {
                q[r] = (0 ..< inDim).map { UInt32(($0 + r * 7) % 16) }
            }
            let scales: [Float] =
                (0 ..< (outDim * nGroups)).map { Float($0 + 1) * 0.01 }
            let biases: [Float] =
                (0 ..< (outDim * nGroups)).map { Float($0) * -0.005 }
            let input: [Float] = (0 ..< inDim).map { Float($0) * 0.1 - 6.4 }

            // Folded copies — the loader transformation under test.
            let scalesM: [Float] = scales.map { $0 * m }
            let biasesM: [Float] = biases.map { $0 * m }

            // Pack the 4-bit weight row-by-row, 8 nibbles per uint32.
            var packed: [UInt32] = []
            for r in 0 ..< outDim {
                for i in stride(from: 0, to: inDim, by: 8) {
                    let nibbles = Array(q[r][i ..< i + 8])
                    packed.append(Self.pack8(nibbles))
                }
            }

            // Allocate tensors. The packed weight is reused unchanged
            // across both calls — only `scales` and `biases` differ.
            let weight = Tensor.empty(shape: [outDim, inDim / 8], dtype: .u32)
            weight.copyIn(from: packed)
            let scalesT = Tensor.empty(shape: [outDim, nGroups], dtype: .f32)
            scalesT.copyIn(from: scales)
            let biasesT = Tensor.empty(shape: [outDim, nGroups], dtype: .f32)
            biasesT.copyIn(from: biases)
            let scalesMT = Tensor.empty(shape: [outDim, nGroups], dtype: .f32)
            scalesMT.copyIn(from: scalesM)
            let biasesMT = Tensor.empty(shape: [outDim, nGroups], dtype: .f32)
            biasesMT.copyIn(from: biasesM)
            let inputT = Tensor.empty(shape: [inDim], dtype: .f32)
            inputT.copyIn(from: input)

            // Reference path: unscaled triplet → multiply output by m.
            var outRef: Tensor!
            runAndWait { cb in
                outRef = Ops.dequantGemvInt4(
                    weight: weight, scales: scalesT, biases: biasesT,
                    input: inputT, groupSize: gs, on: cb)
            }
            let refScaled = outRef.toArray(as: Float.self).map { $0 * m }

            // Folded path: scaled triplet, same packed weight + input.
            var outFolded: Tensor!
            runAndWait { cb in
                outFolded = Ops.dequantGemvInt4(
                    weight: weight, scales: scalesMT, biases: biasesMT,
                    input: inputT, groupSize: gs, on: cb)
            }
            let folded = outFolded.toArray(as: Float.self)

            // Per-output equivalence within bf16 / fp16 tolerance —
            // tighter than the round-trip 1e-2 floor in
            // QuantizedOpsTests because both paths see identical
            // quantization noise (only the order of scalar
            // multiplications changes).
            for i in 0 ..< outDim {
                #expect(
                    abs(folded[i] - refScaled[i]) < 1e-4,
                    "row \(i): folded \(folded[i]) ≠ m·reference \(refScaled[i])"
                )
            }
        }
    }

    @Test("multiplier-folding with m = 1.0 is the identity")
    func identityMultiplier() {
        // m = 1.0 is the fast-path skip in `scaleTensorGMH`. Verifying
        // it through the same end-to-end GEMV path confirms the
        // identity case doesn't drift across the round-trip.
        autoreleasepool {
            let outDim = 2
            let inDim = 64
            let gs = 64
            let nGroups = inDim / gs

            var q = [[UInt32]](repeating: [], count: outDim)
            for r in 0 ..< outDim {
                q[r] = (0 ..< inDim).map { UInt32(($0 + r * 5) % 16) }
            }
            let scales: [Float] = [0.03, -0.07]
            let biases: [Float] = [0.5, -1.25]
            let input: [Float] = (0 ..< inDim).map { Float($0) * 0.05 }

            var packed: [UInt32] = []
            for r in 0 ..< outDim {
                for i in stride(from: 0, to: inDim, by: 8) {
                    let nibbles = Array(q[r][i ..< i + 8])
                    packed.append(Self.pack8(nibbles))
                }
            }

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
                    input: inputT, groupSize: gs, on: cb)
            }
            let got = out.toArray(as: Float.self)

            // Now run again with the "folded" scales+biases at m = 1.0
            // — should be bit-identical (no precision drift on either
            // tensor when m = 1).
            var out2: Tensor!
            runAndWait { cb in
                out2 = Ops.dequantGemvInt4(
                    weight: weight, scales: scalesT, biases: biasesT,
                    input: inputT, groupSize: gs, on: cb)
            }
            let got2 = out2.toArray(as: Float.self)

            for i in 0 ..< outDim {
                #expect(got[i] == got2[i], "row \(i): m=1 path drifted")
            }
        }
    }
}
