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
// FishSpeechLayersTests — unit tests for the small CPU helpers in
// FishSpeechLayers (bfloat16 conversion + the RoPE cos/sin table). The
// transformer block forward path requires real weights + GPU command
// buffers, so it is exercised in the integration suite.
//
// Validates:
//   * bfloat16ToFloat / floatToBfloat16 round-trip with representative
//     finite values and edge cases (zero, negative, small magnitudes).
//   * FishSpeechRoPECache builds a table of the expected shape and the
//     position-0 entries follow the canonical RoPE initial values
//     (cos=1, sin=0).

import Foundation
import Testing
@testable import FFAI

@Suite("FishSpeechLayers")
struct FishSpeechLayersTests {

    // ─── bfloat16 round-trip ─────────────────────────────────────────────

    @Test("bfloat16ToFloat — zero round-trips")
    func bf16Zero() {
        let bits = floatToBfloat16(0)
        #expect(bf16Roughly(0, bits: bits))
        #expect(bfloat16ToFloat(bits) == 0)
    }

    @Test("bfloat16ToFloat — positive value round-trips within bf16 precision")
    func bf16PositiveRoundTrip() {
        let original: Float = 1.5
        let restored = bfloat16ToFloat(floatToBfloat16(original))
        // bf16 has 7 mantissa bits → ~1% relative error.
        #expect(abs(restored - original) < 0.02)
    }

    @Test("bfloat16ToFloat — negative value round-trips")
    func bf16NegativeRoundTrip() {
        let original: Float = -2.5
        let restored = bfloat16ToFloat(floatToBfloat16(original))
        #expect(abs(restored - original) < 0.04)
    }

    @Test("bfloat16ToFloat — large magnitude round-trips")
    func bf16LargeRoundTrip() {
        let original: Float = 12_000
        let restored = bfloat16ToFloat(floatToBfloat16(original))
        // Allow ~1% relative error.
        #expect(abs(restored - original) / original < 0.02)
    }

    // ─── RoPE cache shape + canonical values ────────────────────────────

    @Test("FishSpeechRoPECache — table sized headDim/2 × maxSeq")
    func ropeTableShape() {
        let headDim = 128
        let maxSeq = 32
        let cache = FishSpeechRoPECache(headDim: headDim, ropeBase: 10_000,
                                        maxSeq: maxSeq)
        #expect(cache.headDim == headDim)
        #expect(cache.maxSeq == maxSeq)
        #expect(cache.cosTable.count == maxSeq * (headDim / 2))
        #expect(cache.sinTable.count == maxSeq * (headDim / 2))
    }

    @Test("FishSpeechRoPECache — position 0 row is cos=1, sin=0")
    func ropePositionZero() {
        let headDim = 16
        let maxSeq = 4
        let cache = FishSpeechRoPECache(headDim: headDim, ropeBase: 10_000,
                                        maxSeq: maxSeq)
        let half = headDim / 2
        for i in 0..<half {
            #expect(abs(cache.cosTable[i] - 1.0) < 1e-6,
                    "cos[pos=0, i=\(i)] should be 1, got \(cache.cosTable[i])")
            #expect(abs(cache.sinTable[i]) < 1e-6,
                    "sin[pos=0, i=\(i)] should be 0, got \(cache.sinTable[i])")
        }
    }

    @Test("FishSpeechRoPECache — pos=1, i=0 produces sin(1)/cos(1) for ropeBase>>headDim")
    func ropePositionOneFirstFreq() {
        let headDim = 4
        // ropeBase ^ (0 / headDim) = 1, so the first frequency is exp(0) = 1.
        let cache = FishSpeechRoPECache(headDim: headDim, ropeBase: 10_000,
                                        maxSeq: 4)
        let half = headDim / 2
        let base = 1 * half  // pos=1 row
        // At i=0: angle = 1 * 1 = 1 radian.
        #expect(abs(cache.cosTable[base + 0] - Float(cos(1.0))) < 1e-5)
        #expect(abs(cache.sinTable[base + 0] - Float(sin(1.0))) < 1e-5)
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    /// Convenience: bf16 representation of `f` "roughly" equals `bits`.
    /// (Used as a sanity check on the round-trip — not a precise compare.)
    private func bf16Roughly(_ f: Float, bits: UInt16) -> Bool {
        let restored = bfloat16ToFloat(bits)
        if f == 0 { return restored == 0 }
        return abs(restored - f) / max(abs(f), 1e-6) < 0.05
    }
}
