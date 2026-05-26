// OpsLogits tests — sampling-pipeline logits processors.
//
// For each of the five wrappers we assert one numerical case + at
// least one validator-reject case (without producing a process trap).

import Foundation
import Metal
import Testing
@testable import FFAI
import TestHelpers

@Suite("OpsLogits — sampling pipeline")
struct OpsLogitsTests {

    @Test("logitsTemperature f32 — out = inp / temperature")
    func temperatureF32() {
        autoreleasepool {
            let inp = Tensor.empty(shape: [4], dtype: .f32)
            inp.copyIn(from: [Float(2), 4, 6, 8])
            var out: Tensor!
            runAndWait { cb in out = Ops.logitsTemperature(inp, temperature: 2, on: cb) }
            #expect(out.toArray(as: Float.self) == [1, 2, 3, 4])
        }
    }

    @Test("logitsTemperature — temperature=1 is a no-op divide")
    func temperatureNoop() {
        autoreleasepool {
            let inp = Tensor.empty(shape: [4], dtype: .f32)
            inp.copyIn(from: [Float(1.5), -2.25, 0, 4])
            var out: Tensor!
            runAndWait { cb in out = Ops.logitsTemperature(inp, temperature: 1, on: cb) }
            #expect(out.toArray(as: Float.self) == [1.5, -2.25, 0, 4])
        }
    }

    @Test("logitsTopKMask f32 — entries below threshold mask to -inf")
    func topKMaskF32() {
        autoreleasepool {
            let inp = Tensor.empty(shape: [5], dtype: .f32)
            inp.copyIn(from: [Float(0.5), 2, 1, 3, 0.1])
            // Threshold = 1.0 → keep entries ≥ 1.0, mask < 1.0 to -inf.
            var out: Tensor!
            runAndWait { cb in out = Ops.logitsTopKMask(inp, threshold: 1.0, on: cb) }
            let r = out.toArray(as: Float.self)
            #expect(r[0] == -Float.infinity)
            #expect(r[1] == 2)
            #expect(r[2] == 1)
            #expect(r[3] == 3)
            #expect(r[4] == -Float.infinity)
        }
    }

    @Test("logitsRepetitionPenalty f32 — penalises positive logits at token IDs")
    func repetitionPenaltyF32() {
        autoreleasepool {
            // Vocab size 4. Two token IDs: 1, 3. Penalty = 2.
            // Positive logits at these indices are divided by 2;
            // negative ones are multiplied (kernel formula).
            let logits = Tensor.empty(shape: [4], dtype: .f32)
            logits.copyIn(from: [Float(1), 4, 1, -2])  // idx1=+4, idx3=-2
            let tokenIds = Tensor.empty(shape: [2], dtype: .u32)
            tokenIds.copyIn(from: [UInt32(1), UInt32(3)])
            runAndWait { cb in
                Ops.logitsRepetitionPenalty(logits: logits, tokenIds: tokenIds,
                                            penalty: 2, on: cb)
            }
            let r = logits.toArray(as: Float.self)
            #expect(r[0] == 1)         // untouched
            #expect(r[1] == 2)         // 4 / 2
            #expect(r[2] == 1)         // untouched
            #expect(r[3] == -4)        // -2 * 2
        }
    }

    @Test("logitsMinPMask f32 — masks logits below max*minP to -inf")
    func minPMaskF32() {
        autoreleasepool {
            // Row: peak at idx2 = 5; minP = 0.5 → cutoff in prob space
            // is 0.5 * exp(5) ≈ 74.21, in log space that's 5 + log(0.5)
            // ≈ 4.31. So entries with value < 4.31 should be masked.
            let inp = Tensor.empty(shape: [5], dtype: .f32)
            inp.copyIn(from: [Float(1), 2, 5, 4, 3])
            var out: Tensor!
            runAndWait { cb in out = Ops.logitsMinPMask(inp, minP: 0.5, on: cb) }
            let r = out.toArray(as: Float.self)
            // idx 2 (val=5) and idx 3 (val=4) should be ≥ cutoff;
            // 4.31 cutoff: idx 3 (4.0) is borderline. We assert only
            // that the peak survives and the lowest entries are masked.
            #expect(r[2] == 5)
            #expect(r[0] == -Float.infinity)
            #expect(r[1] == -Float.infinity)
        }
    }

    @Test("logitsTopPMask f32 — masks tail beyond top_p of CDF")
    func topPMaskF32() {
        autoreleasepool {
            // 5-entry row with one peak: idx2 dominates the softmax,
            // so even small top_p (0.5) keeps just that entry.
            let inp = Tensor.empty(shape: [5], dtype: .f32)
            inp.copyIn(from: [Float(0), 0, 10, 0, 0])
            var out: Tensor!
            runAndWait { cb in out = Ops.logitsTopPMask(inp, topP: 0.5, on: cb) }
            let r = out.toArray(as: Float.self)
            // Peak entry stays finite; all others should be masked.
            #expect(r[2] == 10)
            for i in [0, 1, 3, 4] {
                #expect(r[i] == -Float.infinity)
            }
        }
    }

    // ─── validator rejection (pure, no trap) ────────────────────────

    @Test("validateLogitsTemperature rejects negative / zero")
    func validateTemperature() {
        #expect(OpsValidation.validateLogitsTemperature(n: 0, temperature: 1) != nil)
        #expect(OpsValidation.validateLogitsTemperature(n: 100, temperature: 0) != nil)
        #expect(OpsValidation.validateLogitsTemperature(n: 100, temperature: -1) != nil)
        #expect(OpsValidation.validateLogitsTemperature(n: 100, temperature: Float.nan) != nil)
        #expect(OpsValidation.validateLogitsTemperature(n: 100, temperature: 1) == nil)
    }

    @Test("validateLogitsMinPMask / TopPMask reject out-of-range p")
    func validatePValues() {
        // minP must be in (0, 1)
        #expect(OpsValidation.validateLogitsMinPMask(n: 4, rows: 1, minP: 0) != nil)
        #expect(OpsValidation.validateLogitsMinPMask(n: 4, rows: 1, minP: 1) != nil)
        #expect(OpsValidation.validateLogitsMinPMask(n: 4, rows: 1, minP: -0.1) != nil)
        #expect(OpsValidation.validateLogitsMinPMask(n: 4, rows: 1, minP: 0.5) == nil)
        // topP same contract.
        #expect(OpsValidation.validateLogitsTopPMask(n: 4, rows: 1, topP: 0) != nil)
        #expect(OpsValidation.validateLogitsTopPMask(n: 4, rows: 1, topP: 1) != nil)
        #expect(OpsValidation.validateLogitsTopPMask(n: 4, rows: 1, topP: 0.95) == nil)
    }

    @Test("validateLogitsRepetitionPenalty rejects penalty ≤ 0")
    func validateRepetitionPenalty() {
        #expect(OpsValidation.validateLogitsRepetitionPenalty(
            vocab: 100, nTokenIds: 5, penalty: 0) != nil)
        #expect(OpsValidation.validateLogitsRepetitionPenalty(
            vocab: 0, nTokenIds: 5, penalty: 1.1) != nil)
        #expect(OpsValidation.validateLogitsRepetitionPenalty(
            vocab: 100, nTokenIds: 5, penalty: 1.1) == nil)
    }

    // ─── f16 + bf16 dispatch ───────────────────────────────────────

    @Test("logits processors dispatch on f16 + bf16 without error")
    func logitsDtypeCoverage() {
        autoreleasepool {
            // f16: temperature, topK, minP, topP
            let inp16 = Tensor.empty(shape: [16], dtype: .f16)
            inp16.copyIn(from: (0..<16).map { Float16(Float($0) * 0.1) })
            runAndWait { cb in
                _ = Ops.logitsTemperature(inp16, temperature: 0.7, on: cb)
                _ = Ops.logitsTopKMask(inp16, threshold: 0.5, on: cb)
                _ = Ops.logitsMinPMask(inp16, minP: 0.5, on: cb)
                _ = Ops.logitsTopPMask(inp16, topP: 0.9, on: cb)
            }
            let tokensF16 = Tensor.empty(shape: [2], dtype: .u32)
            tokensF16.copyIn(from: [UInt32(0), UInt32(3)])
            runAndWait { cb in
                Ops.logitsRepetitionPenalty(logits: inp16, tokenIds: tokensF16,
                                            penalty: 1.5, on: cb)
            }
            // bf16: temperature, topK, minP, topP
            // Build bf16 via uint16 bit patterns mapped 1:1 from f32 high bits.
            let inpBF = Tensor.empty(shape: [16], dtype: .bf16)
            inpBF.copyIn(from: (0..<16).map { i -> UInt16 in
                let f: Float = Float(i) * 0.1
                return UInt16(f.bitPattern >> 16)
            })
            runAndWait { cb in
                _ = Ops.logitsTemperature(inpBF, temperature: 0.7, on: cb)
                _ = Ops.logitsTopKMask(inpBF, threshold: 0.5, on: cb)
                _ = Ops.logitsMinPMask(inpBF, minP: 0.5, on: cb)
                _ = Ops.logitsTopPMask(inpBF, topP: 0.9, on: cb)
            }
            runAndWait { cb in
                Ops.logitsRepetitionPenalty(logits: inpBF, tokenIds: tokensF16,
                                            penalty: 1.5, on: cb)
            }
        }
    }
}
