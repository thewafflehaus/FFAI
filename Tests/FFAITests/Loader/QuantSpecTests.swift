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
// `QuantSpec` parser + property tests. The convert CLI flag delegates
// to `QuantSpec.init(parsing:)` for every `--*-bits` value — so the
// parser is the single ground-truth for "what spec strings are valid".

import Foundation
import Testing

@testable import FFAI

@Suite("QuantSpec parser")
struct QuantSpecTests {

    // ─── Affine bit-width parsing ──────────────────────────────────

    @Test("integer literals 2 / 3 / 4 / 5 / 6 / 8 parse to .bits(N)")
    func acceptsSupportedBits() {
        for n in [2, 3, 4, 5, 6, 8] {
            let spec = QuantSpec(parsing: "\(n)")
            #expect(spec == .bits(n), "expected .bits(\(n)) for \"\(n)\", got \(String(describing: spec))")
        }
    }

    @Test("integer literals outside {2,3,4,5,6,8} are rejected")
    func rejectsUnsupportedBits() {
        // Excluded: 0 / 1 / 7 / 9 / 10 / 16 / 32 / negative.
        for n in [0, 1, 7, 9, 10, 16, 32, -1, -4] {
            #expect(
                QuantSpec(parsing: "\(n)") == nil,
                "bits=\(n) should be rejected")
        }
    }

    // ─── Downcast parsing ──────────────────────────────────────────

    @Test("fp16 aliases all parse to .fp16")
    func acceptsFp16Aliases() {
        for s in ["fp16", "f16", "float16", "half", "FP16", "F16", "Half"] {
            #expect(
                QuantSpec(parsing: s) == .fp16,
                "expected .fp16 for \"\(s)\", got \(String(describing: QuantSpec(parsing: s)))")
        }
    }

    @Test("bf16 aliases all parse to .bf16")
    func acceptsBf16Aliases() {
        for s in ["bf16", "bfloat16", "BF16", "BFloat16", "BFLOAT16"] {
            #expect(
                QuantSpec(parsing: s) == .bf16,
                "expected .bf16 for \"\(s)\", got \(String(describing: QuantSpec(parsing: s)))")
        }
    }

    @Test("unknown spec strings reject")
    func rejectsGarbage() {
        for s in ["", "int4", "q4_k_m", "fp32", "f32", "float32", "int8", "0bit", "asdf"] {
            #expect(QuantSpec(parsing: s) == nil, "expected nil for \"\(s)\"")
        }
    }

    // ─── Properties ────────────────────────────────────────────────

    @Test("isQuantized matches .bits only")
    func isQuantizedFlag() {
        #expect(QuantSpec.bits(4).isQuantized)
        #expect(QuantSpec.bits(2).isQuantized)
        #expect(QuantSpec.bits(8).isQuantized)
        #expect(!QuantSpec.fp16.isQuantized)
        #expect(!QuantSpec.bf16.isQuantized)
    }

    @Test("bits accessor matches .bits payload, nil for downcast")
    func bitsAccessor() {
        #expect(QuantSpec.bits(4).bits == 4)
        #expect(QuantSpec.bits(3).bits == 3)
        #expect(QuantSpec.bits(2).bits == 2)
        #expect(QuantSpec.fp16.bits == nil)
        #expect(QuantSpec.bf16.bits == nil)
    }

    @Test("downcastDtype maps .fp16 → .f16 and .bf16 → .bf16; nil for bits")
    func downcastDtypeAccessor() {
        #expect(QuantSpec.fp16.downcastDtype == .f16)
        #expect(QuantSpec.bf16.downcastDtype == .bf16)
        for n in QuantSpec.supportedBits {
            #expect(
                QuantSpec.bits(n).downcastDtype == nil,
                ".bits(\(n)).downcastDtype should be nil")
        }
    }

    @Test("label renders 'Nbit' for bits, 'fp16' / 'bf16' for downcast")
    func labelFormat() {
        #expect(QuantSpec.bits(4).label == "4bit")
        #expect(QuantSpec.bits(2).label == "2bit")
        #expect(QuantSpec.bits(3).label == "3bit")
        #expect(QuantSpec.bits(8).label == "8bit")
        #expect(QuantSpec.fp16.label == "fp16")
        #expect(QuantSpec.bf16.label == "bf16")
    }

    @Test("supportedBits is the exact accepted set")
    func supportedBitsCanonical() {
        #expect(QuantSpec.supportedBits == [2, 3, 4, 5, 6, 8])
    }
}
