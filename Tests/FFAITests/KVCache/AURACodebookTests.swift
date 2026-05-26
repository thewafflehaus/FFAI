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

@Suite("AURACodebook")
struct AURACodebookTests {

    @Test("supportedBits covers 2/3/4/8")
    func supportedBits() {
        #expect(AURACodebook.supportedBits == [2, 3, 4, 8])
    }

    @Test("centroid table is 2^bits long for every supported bit width at d=128")
    func centroidsLengthD128() {
        for bits in [2, 3, 4, 8] {
            #expect(
                AURACodebook.centroids(dim: 128, bits: bits).count == 1 << bits,
                "wrong centroid count at bits=\(bits)")
            #expect(
                AURACodebook.boundaries(dim: 128, bits: bits).count == (1 << bits) - 1,
                "wrong boundary count at bits=\(bits)")
        }
    }

    @Test("centroids are sorted ascending (binary-search invariant)")
    func centroidsSorted() {
        for bits in [2, 3, 4, 8] {
            let c = AURACodebook.centroids(dim: 128, bits: bits)
            for i in 1 ..< c.count {
                #expect(c[i] >= c[i - 1], "centroid \(i) regresses at bits=\(bits)")
            }
        }
    }

    @Test("boundaries lie between adjacent centroids")
    func boundariesBetweenCentroids() {
        for bits in [2, 3, 4, 8] {
            let c = AURACodebook.centroids(dim: 128, bits: bits)
            let b = AURACodebook.boundaries(dim: 128, bits: bits)
            for i in 0 ..< b.count {
                #expect(
                    b[i] >= c[i] && b[i] <= c[i + 1],
                    "boundary \(i)=\(b[i]) not between c[\(i)]=\(c[i]) and c[\(i+1)]=\(c[i+1]) at bits=\(bits)"
                )
            }
        }
    }

    @Test("centroids are zero-mean (Lloyd-Max symmetry around 0)")
    func centroidsZeroMean() {
        // Lloyd-Max on a symmetric distribution produces a symmetric
        // codebook — the mean of the centroids should be very close
        // to zero. Tolerance is loose because the 8-bit table was
        // computed offline at finite grid resolution.
        for bits in [2, 3, 4, 8] {
            let c = AURACodebook.centroids(dim: 128, bits: bits)
            let mean = c.reduce(0, +) / Float(c.count)
            #expect(abs(mean) < 1e-3, "centroid mean=\(mean) too far from zero at bits=\(bits)")
        }
    }

    @Test("scaledCentroids(dim != 128) applies sqrt(128/dim) scale")
    func scaledCentroidsHeuristic() {
        // Spot-check a couple of dims that ship in production
        // (Mistral d=64, Llama d=128, Llama-Vision d=128, larger d=256).
        let c128 = AURACodebook.centroids(dim: 128, bits: 4)

        let c64 = AURACodebook.centroids(dim: 64, bits: 4)
        let expectedScale64 = Float((128.0 / 64.0).squareRoot())  // sqrt(2)
        for i in 0 ..< c128.count {
            let want = c128[i] * expectedScale64
            #expect(
                abs(c64[i] - want) < 1e-6,
                "d=64 scaling diverges at i=\(i): got \(c64[i]), want \(want)")
        }

        let c256 = AURACodebook.centroids(dim: 256, bits: 4)
        let expectedScale256 = Float((128.0 / 256.0).squareRoot())  // sqrt(0.5)
        for i in 0 ..< c128.count {
            let want = c128[i] * expectedScale256
            #expect(
                abs(c256[i] - want) < 1e-6,
                "d=256 scaling diverges at i=\(i): got \(c256[i]), want \(want)")
        }
    }

    @Test("packedWidth matches ceil(dim*bits/32)")
    func packedWidthFormula() {
        // (dim, bits) → expected packed_width (u32 words)
        let cases: [(Int, Int, Int)] = [
            (128, 2, 8),  // 128*2 = 256 bits → 8 words
            (128, 3, 12),  // 128*3 = 384 bits → 12 words
            (128, 4, 16),  // 128*4 = 512 bits → 16 words
            (128, 8, 32),  // 128*8 = 1024 bits → 32 words
            (64, 4, 8),
            (96, 4, 12),
            (80, 3, 8),  // 80*3 = 240 bits → ceil 8 words
        ]
        for (dim, bits, want) in cases {
            #expect(
                AURACodebook.packedWidth(dim: dim, bits: bits) == want,
                "packedWidth(\(dim), \(bits)) = \(AURACodebook.packedWidth(dim: dim, bits: bits)), want \(want)"
            )
        }
    }

    @Test("bytesPerToken = packed_width*4 + 4 (norm)")
    func bytesPerTokenFormula() {
        // 4-bit, d=128: 16 u32 words + 1 fp32 norm = 64 + 4 = 68 B
        #expect(AURACodebook.bytesPerToken(dim: 128, bits: 4) == 68)
        // 2-bit, d=128: 8 u32 words + 1 fp32 norm = 32 + 4 = 36 B
        #expect(AURACodebook.bytesPerToken(dim: 128, bits: 2) == 36)
        // 8-bit, d=128: 32 u32 words + 1 fp32 norm = 128 + 4 = 132 B
        #expect(AURACodebook.bytesPerToken(dim: 128, bits: 8) == 132)
    }
}

@Suite("AURAScheme")
struct AURASchemeTests {

    @Test("default scheme is aura4v4 (symmetric 4-bit)")
    func defaultScheme() {
        #expect(AURAScheme.default.keyBits == 4)
        #expect(AURAScheme.default.valueBits == 4)
        #expect(AURAScheme.default.name == "aura4")
    }

    @Test("asymmetric scheme keeps K/V bits separate in name")
    func asymmetricName() {
        let s = AURAScheme(keyBits: 4, valueBits: 2)
        #expect(s.name == "aura4v2")
    }

    @Test("parse 'aura' → default aura4v4")
    func parseBare() {
        #expect(AURAScheme.parse("aura") == .default)
        #expect(AURAScheme.parse("AURA") == .default)
    }

    @Test("parse 'aura{kb}' → symmetric")
    func parseSymmetric() {
        #expect(AURAScheme.parse("aura2") == AURAScheme(keyBits: 2, valueBits: 2))
        #expect(AURAScheme.parse("aura3") == AURAScheme(keyBits: 3, valueBits: 3))
        #expect(AURAScheme.parse("aura4") == AURAScheme(keyBits: 4, valueBits: 4))
        #expect(AURAScheme.parse("aura8") == AURAScheme(keyBits: 8, valueBits: 8))
    }

    @Test("parse 'aura{kb}v{vb}' → asymmetric")
    func parseAsymmetric() {
        #expect(AURAScheme.parse("aura4v2") == AURAScheme(keyBits: 4, valueBits: 2))
        #expect(AURAScheme.parse("aura8v4") == AURAScheme(keyBits: 8, valueBits: 4))
        #expect(AURAScheme.parse("aura3v2") == AURAScheme(keyBits: 3, valueBits: 2))
    }

    @Test("parse rejects unsupported bit widths and malformed inputs")
    func parseRejects() {
        #expect(AURAScheme.parse("aura5") == nil)  // 5-bit not in supportedBits
        #expect(AURAScheme.parse("aura6") == nil)
        #expect(AURAScheme.parse("aura16") == nil)
        #expect(AURAScheme.parse("aura4v5") == nil)  // 5-bit V not supported
        #expect(AURAScheme.parse("aura4v") == nil)  // empty V
        #expect(AURAScheme.parse("aurav4") == nil)  // empty K
        #expect(AURAScheme.parse("turbo4v2") == nil)  // wrong prefix
        #expect(AURAScheme.parse("") == nil)
        #expect(AURAScheme.parse("aura4-2") == nil)
    }

    @Test("aura4v2 production recipe")
    func production4v2() {
        #expect(AURAScheme.aura4v2.keyBits == 4)
        #expect(AURAScheme.aura4v2.valueBits == 2)
        #expect(AURAScheme.aura4v2.name == "aura4v2")
    }
}

@Suite("LoadOptions — AURA")
struct LoadOptionsAURATests {

    @Test("auraQuantized round-trips through KVCacheKind")
    func roundTrip() {
        let opts = LoadOptions(kvCache: .auraQuantized(scheme: .aura4v2))
        if case .auraQuantized(let scheme) = opts.kvCache {
            #expect(scheme == .aura4v2)
        } else {
            Issue.record("expected .auraQuantized(scheme: aura4v2)")
        }
    }

    @Test("auraQuantized default scheme — bare .auraQuantized() is aura4v4")
    func defaultSchemeInLoadOptions() {
        let opts = LoadOptions(kvCache: .auraQuantized())
        if case .auraQuantized(let scheme) = opts.kvCache {
            #expect(scheme == .default)
            #expect(scheme.name == "aura4")
        } else {
            Issue.record("expected .auraQuantized() to use default scheme")
        }
    }
}
