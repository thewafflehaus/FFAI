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
// WeightFoldingTests — the shared µP fold helpers in
// `Sources/FFAI/Loader/WeightFolding.swift`. The quantized
// scales/biases fold equivalence is proven end-to-end through the GEMV
// kernel in `Granite4Tests.swift`; these cover the host-side scalar
// scale + the MuPMultipliers value semantics the Granite 3 loader
// relies on.

import Foundation
import Testing

@testable import FFAI

@Suite("WeightFolding")
struct WeightFoldingTests {

    @Test("scaleTensorElements scales every element (f32)")
    func scaleF32() {
        let t = Tensor.empty(shape: [4], dtype: .f32)
        t.copyIn(from: [1.0, -2.0, 3.5, 0.0] as [Float])
        let scaled = scaleTensorElements(t, by: 2.0, device: .shared)
        #expect(scaled.toArray(as: Float.self) == [2.0, -4.0, 7.0, 0.0])
    }

    @Test("scaleTensorElements round-trips through bf16")
    func scaleBf16() {
        // 0.5 / -1.0 / 2.0 are exact in bf16, so the scaled values are
        // exact too — no tolerance needed.
        let t = Tensor.empty(shape: [3], dtype: .bf16)
        t.copyIn(from: [0.5, -1.0, 2.0].map { Float($0) }.map {
            UInt16(truncatingIfNeeded: $0.bitPattern >> 16)
        })
        let scaled = scaleTensorElements(t, by: 4.0, device: .shared)
        let out = scaled.toArray(as: UInt16.self).map {
            Float(bitPattern: UInt32($0) << 16)
        }
        #expect(out == [2.0, -4.0, 8.0])
        #expect(scaled.dtype == .bf16)
    }

    @Test("scaleTensorElements at m=1 is the identity fast-path")
    func scaleIdentity() {
        let t = Tensor.empty(shape: [2], dtype: .f32)
        t.copyIn(from: [5.0, 6.0] as [Float])
        let same = scaleTensorElements(t, by: 1.0, device: .shared)
        #expect(same.toArray(as: Float.self) == [5.0, 6.0])
    }

    @Test("MuPMultipliers identity + isIdentity semantics")
    func muPIdentity() {
        #expect(MuPMultipliers.identity.isIdentity)
        #expect(MuPMultipliers().isIdentity)
        #expect(MuPMultipliers(embedding: 2).isIdentity == false)
        #expect(MuPMultipliers(attention: 0.1).isIdentity == false)
        #expect(MuPMultipliers(residual: 0.5).isIdentity == false)
        #expect(MuPMultipliers(logits: 8).isIdentity == false)
    }
}
