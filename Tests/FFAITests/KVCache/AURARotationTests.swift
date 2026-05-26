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
import Testing

@testable import FFAI

@Suite("AURARotation")
struct AURARotationTests {

    @Test("hadamardMatrix is the Sylvester Hadamard at dim=2/4/8")
    func hadamardSmallDims() {
        // H_2 = [[1, 1], [1, -1]]
        #expect(AURARotation.hadamardMatrix(dim: 2) == [1, 1, 1, -1])

        // H_4 = [[1,1,1,1], [1,-1,1,-1], [1,1,-1,-1], [1,-1,-1,1]]
        let h4: [Float] = [
            1, 1, 1, 1,
            1, -1, 1, -1,
            1, 1, -1, -1,
            1, -1, -1, 1,
        ]
        #expect(AURARotation.hadamardMatrix(dim: 4) == h4)

        // Spot-check at dim=8 that the orthogonality H · H^T = 8·I holds.
        let h = AURARotation.hadamardMatrix(dim: 8)
        for i in 0 ..< 8 {
            for j in 0 ..< 8 {
                var dot: Float = 0
                for k in 0 ..< 8 { dot += h[i * 8 + k] * h[j * 8 + k] }
                let expected: Float = (i == j) ? 8.0 : 0.0
                #expect(
                    abs(dot - expected) < 1e-5,
                    "hadamardMatrix(8) not orthogonal: row \(i) · row \(j) = \(dot), expected \(expected)"
                )
            }
        }
    }

    @Test("hadamardMatrix dim=128 is orthogonal (H · H^T = 128·I)")
    func hadamardLargeDim() {
        let dim = 128
        let h = AURARotation.hadamardMatrix(dim: dim)
        // Sample the diagonal + a few off-diagonal entries; full pairwise
        // is O(dim^3) and overkill for a sanity test.
        for i in 0 ..< dim {
            var dot: Float = 0
            for k in 0 ..< dim { dot += h[i * dim + k] * h[i * dim + k] }
            #expect(abs(dot - Float(dim)) < 1e-3, "diagonal[\(i)] = \(dot), expected \(dim)")
        }
        // Off-diagonal sample
        let pairs = [(0, 1), (0, 7), (3, 11), (17, 64), (100, 127)]
        for (i, j) in pairs {
            var dot: Float = 0
            for k in 0 ..< dim { dot += h[i * dim + k] * h[j * dim + k] }
            #expect(abs(dot) < 1e-3, "off-diagonal[\(i),\(j)] = \(dot), expected 0")
        }
    }

    @Test("whtSigns returns ±1 only and is deterministic")
    func whtSignsDeterministic() {
        let a = AURARotation.whtSigns(dim: 128, seed: 42)
        let b = AURARotation.whtSigns(dim: 128, seed: 42)
        #expect(a == b, "whtSigns(seed=42) must be reproducible")

        for v in a { #expect(v == 1.0 || v == -1.0, "non-±1 entry \(v)") }

        let c = AURARotation.whtSigns(dim: 128, seed: 43)
        #expect(a != c, "different seeds should produce different sign vectors")
    }

    @Test("srhtMatrix is orthonormal (Π · Π^T = I)")
    func srhtOrthonormal() {
        let dim = 128
        let pi = AURARotation.srhtMatrix(dim: dim, seed: 42)

        // Π · Π^T should be ≈ I. Sample a few rows.
        for i in [0, 1, 17, 100, dim - 1] {
            var dot: Float = 0
            for k in 0 ..< dim { dot += pi[i * dim + k] * pi[i * dim + k] }
            #expect(abs(dot - 1.0) < 1e-4, "row \(i) · row \(i) = \(dot), expected 1.0")
        }
        for (i, j) in [(0, 1), (0, 17), (3, 50), (100, 127)] {
            var dot: Float = 0
            for k in 0 ..< dim { dot += pi[i * dim + k] * pi[j * dim + k] }
            #expect(abs(dot) < 1e-4, "row \(i) · row \(j) = \(dot), expected 0")
        }
    }

    @Test("identityMatrix is dim·dim ones-on-diagonal")
    func identityShape() {
        let dim = 16
        let id = AURARotation.identityMatrix(dim: dim)
        #expect(id.count == dim * dim)
        for i in 0 ..< dim {
            for j in 0 ..< dim {
                let want: Float = (i == j) ? 1.0 : 0.0
                #expect(id[i * dim + j] == want, "identity[\(i),\(j)] = \(id[i * dim + j])")
            }
        }
    }
}
