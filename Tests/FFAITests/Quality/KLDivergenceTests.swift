// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// KLDivergence math — unit tests on synthetic logit pairs to pin the
// numerical invariants of the regression gate for TQ+/AURA quality
// ports. Mirrors the canonical TQ+ harness output format.

import Foundation
import Testing

@testable import FFAI

@Suite("KLDivergence — per-position + aggregate metrics")
struct KLDivergenceTests {

    // MARK: - positionMetrics — invariants

    @Test("identical logits → kld == 0 and matching argmax")
    func identicalLogitsZeroKld() {
        let logits: [Float] = [1.0, 2.0, 0.5, -1.0, 3.0]
        let m = KLDivergence.positionMetrics(
            baselineLogits: logits, codecLogits: logits, nextTokenId: 2)
        #expect(m.kld < 1e-12, "got kld=\(m.kld), expected 0")
        #expect(m.baselineTopIdx == 4)
        #expect(m.codecTopIdx == 4)
        // Same-prob check on the target token (index 2).
        #expect(abs(m.baselineNextProb - m.codecNextProb) < 1e-12)
    }

    @Test("shifted-constant logits → kld == 0 (softmax shift-invariance)")
    func shiftInvariance() {
        let baseline: [Float] = [0.0, 1.0, 2.0, 3.0]
        let shifted: [Float] = [10.0, 11.0, 12.0, 13.0]
        let m = KLDivergence.positionMetrics(
            baselineLogits: baseline, codecLogits: shifted, nextTokenId: 1)
        #expect(m.kld < 1e-10, "got kld=\(m.kld), expected 0")
        #expect(m.baselineTopIdx == m.codecTopIdx)
    }

    @Test("uniform baseline + peaked codec → kld matches closed-form value")
    func uniformVsPeakedClosedForm() {
        // baseline: uniform over 4 tokens → P_base(i) = 1/4 for all i.
        let baseline: [Float] = [0, 0, 0, 0]
        // codec: heavily peaks at index 0. logits [3,0,0,0] →
        // softmax = (e^3, 1, 1, 1) / (e^3 + 3).
        let codec: [Float] = [3, 0, 0, 0]
        let m = KLDivergence.positionMetrics(
            baselineLogits: baseline, codecLogits: codec, nextTokenId: 0)
        // KL(U || codec) = -H(U) - E_U[log P_codec]
        //                = log(4) + (-1/4) * sum_i log P_codec(i)
        // Closed-form by hand:
        let s = exp(3.0) + 3.0
        let logQ0 = 3.0 - log(s)
        let logQother = 0.0 - log(s)
        let expectedKld =
            -log(4.0) - 0.25 * (logQ0 + 3 * logQother)
        // (KL = E_P log(P/Q) = sum_i P_i (log P_i - log Q_i))
        // P_i = 1/4, log P_i = -log 4 for all i.
        // KL = -log 4 - 1/4 * sum_i log Q_i = expected above.
        #expect(
            abs(m.kld - expectedKld) < 1e-9,
            "got \(m.kld) expected \(expectedKld)")
        #expect(m.kld > 0)  // distributions differ ⇒ kld positive
    }

    @Test("argmax indices reflect the actual logit maximums")
    func argmaxDetection() {
        let baseline: [Float] = [0, 5, 0, 0, 0]  // argmax = 1
        let codec: [Float] = [0, 0, 0, 4, 0]  // argmax = 3
        let m = KLDivergence.positionMetrics(
            baselineLogits: baseline, codecLogits: codec, nextTokenId: 0)
        #expect(m.baselineTopIdx == 1)
        #expect(m.codecTopIdx == 3)
    }

    @Test("baselineNextProb + codecNextProb pinned at the supplied next-token-id slot")
    func nextProbSlot() {
        let baseline: [Float] = [10, 0, 0, 0]
        let codec: [Float] = [0, 0, 0, 10]
        let m = KLDivergence.positionMetrics(
            baselineLogits: baseline, codecLogits: codec, nextTokenId: 0)
        // Baseline strongly prefers index 0 ≈ probability close to 1.
        #expect(m.baselineNextProb > 0.99)
        // Codec strongly prefers index 3, so its probability for
        // index 0 is near 0.
        #expect(m.codecNextProb < 0.001)
    }

    // MARK: - aggregate — invariants

    @Test("aggregate matches llama-perplexity --kl-divergence summary fields")
    func aggregateFields() {
        // Build 100 synthetic positions with controlled KLDs:
        // 99 positions at kld=0.01, 1 position at kld=10. Should give
        // mean ≈ 0.11, median = 0.01, max = 10, sameTop = 1.0.
        var positions: [KLDivergence.PositionMetrics] = []
        // 99 small-drift positions: matching argmax, low kld.
        for _ in 0 ..< 99 {
            positions.append(
                KLDivergence.PositionMetrics(
                    kld: 0.01,
                    baselineTopIdx: 0, codecTopIdx: 0,
                    baselineNextProb: 0.5, codecNextProb: 0.49))
        }
        // 1 catastrophic position.
        positions.append(
            KLDivergence.PositionMetrics(
                kld: 10.0,
                baselineTopIdx: 0, codecTopIdx: 0,
                baselineNextProb: 0.5, codecNextProb: 0.01))
        let agg = KLDivergence.aggregate(positions)
        #expect(agg.nPositions == 100)
        #expect(abs(agg.meanKld - 0.1099) < 1e-9)
        #expect(agg.medianKld == 0.01)
        #expect(agg.maxKld == 10.0)
        // Linear-interp quantile on a 100-element list:
        //   99% → pos = 99 * 0.99 = 98.01 →
        //          sorted[98] + 0.01 * (sorted[99] - sorted[98])
        //        = 0.01 + 0.01 * (10 - 0.01) = 0.1099
        //   99.9% → pos = 99 * 0.999 = 98.901 →
        //          sorted[98] + 0.901 * (sorted[99] - sorted[98])
        //        = 0.01 + 0.901 * 9.99 ≈ 9.0109
        #expect(
            abs(agg.kld99 - 0.1099) < 1e-9,
            "got kld99=\(agg.kld99)")
        #expect(
            abs(agg.kld999 - 9.0109) < 1e-3,
            "got kld999=\(agg.kld999)")
        #expect(agg.sameTopFraction == 1.0)
        // meanDp = (99 * 0.01 + 1 * 0.49) / 100 = 0.0148
        #expect(abs(agg.meanDp - 0.0148) < 1e-9)
        #expect(agg.maxDp == 0.49)
    }

    @Test("aggregate same-top fraction tracks argmax mismatches")
    func aggregateSameTopFraction() {
        var positions: [KLDivergence.PositionMetrics] = []
        // 7 of 10 positions: argmax matches.
        for _ in 0 ..< 7 {
            positions.append(
                KLDivergence.PositionMetrics(
                    kld: 0.001,
                    baselineTopIdx: 1, codecTopIdx: 1,
                    baselineNextProb: 0.5, codecNextProb: 0.5))
        }
        // 3 of 10: codec picks something else.
        for _ in 0 ..< 3 {
            positions.append(
                KLDivergence.PositionMetrics(
                    kld: 1.0,
                    baselineTopIdx: 1, codecTopIdx: 5,
                    baselineNextProb: 0.5, codecNextProb: 0.1))
        }
        let agg = KLDivergence.aggregate(positions)
        #expect(abs(agg.sameTopFraction - 0.7) < 1e-12)
    }

    @Test("summaryLine includes the canonical TQ+ harness fields")
    func summaryLineFormat() {
        let positions = (0 ..< 5).map { _ in
            KLDivergence.PositionMetrics(
                kld: 0.1,
                baselineTopIdx: 0, codecTopIdx: 0,
                baselineNextProb: 0.5, codecNextProb: 0.5)
        }
        let agg = KLDivergence.aggregate(positions)
        let line = KLDivergence.summaryLine(label: "q8_0/turbo3", metrics: agg)
        #expect(line.contains("q8_0/turbo3"))
        #expect(line.contains("mean_kld="))
        #expect(line.contains("same_top="))
        #expect(line.contains("n=5"))
    }
}
