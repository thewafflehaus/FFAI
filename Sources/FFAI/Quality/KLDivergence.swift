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
// KLDivergence — per-position + aggregate metrics for comparing a
// codec's next-token distribution against an fp16/fp16 baseline. The
// regression gate for every TQ+ / AURA quality port (matched-norm L2,
// InnerQ equalization, per-group FP8 scale, sparse-V threshold, etc.)
// — without these metrics we can't tell if a codec change improved or
// degraded distributional fidelity.
//
// Mirrors the canonical TQ+ harness output from
// `bench-tq+/harness/kld_vs_baseline.py` in
// /Users/tom/local_llms/llama.cpp: same field names, same percentile
// list, so plot scripts + cross-codec comparisons remain compatible.

import Foundation

public enum KLDivergence {

    /// Per-position quality metrics from comparing one codec's logits
    /// to a baseline at the same context position.
    public struct PositionMetrics: Sendable {
        /// KL(baseline ‖ codec). Always ≥ 0.
        public let kld: Double
        /// argmax over baseline log-probs.
        public let baselineTopIdx: Int
        /// argmax over codec log-probs.
        public let codecTopIdx: Int
        /// `exp(logSoftmax(baselineLogits)[nextTokenId])` — baseline's
        /// assigned probability of the next ground-truth token.
        public let baselineNextProb: Double
        /// Same but under the codec's distribution.
        public let codecNextProb: Double
    }

    /// Aggregate metrics across all measured positions. Same field set
    /// as llama-perplexity's `--kl-divergence` output so downstream
    /// plotting / summary scripts can consume FFAI runs.
    public struct AggregateMetrics: Sendable {
        public let meanKld: Double
        public let medianKld: Double
        public let maxKld: Double
        /// 99.9 / 99 / 95 / 90 / 10 / 5 / 1 percentile KLDs.
        public let kld999: Double
        public let kld99: Double
        public let kld95: Double
        public let kld90: Double
        public let kld10: Double
        public let kld05: Double
        public let kld01: Double
        /// Fraction of positions where baseline argmax == codec argmax.
        /// Greedy decode is preserved when this stays > 0.99 even if
        /// KLD is non-trivial — distributional drift only matters at
        /// sampling temperature > 0.
        public let sameTopFraction: Double
        /// Mean |P_base(next) − P_codec(next)|.
        public let meanDp: Double
        public let rmsDp: Double
        public let maxDp: Double
        public let nPositions: Int
    }

    /// Compute per-position metrics from raw logit vectors. `nextTokenId`
    /// is the ground-truth token at `position + 1` in the corpus —
    /// what the model should have predicted at `position`.
    ///
    /// Both inputs must be the same length (vocab size). Uses
    /// numerically-stable log-softmax in Double precision regardless
    /// of the logits' input dtype.
    public static func positionMetrics(
        baselineLogits: [Float], codecLogits: [Float],
        nextTokenId: Int
    ) -> PositionMetrics {
        precondition(
            baselineLogits.count == codecLogits.count,
            "KLDivergence.positionMetrics: logits vocab size mismatch "
                + "(\(baselineLogits.count) vs \(codecLogits.count))")
        precondition(
            nextTokenId >= 0 && nextTokenId < baselineLogits.count,
            "KLDivergence.positionMetrics: nextTokenId \(nextTokenId) "
                + "out of [0, \(baselineLogits.count))")

        let logPbase = logSoftmax(baselineLogits)
        let logPcodec = logSoftmax(codecLogits)

        var kld: Double = 0
        var topBase = 0
        var topCodec = 0
        var bestBase: Double = -.infinity
        var bestCodec: Double = -.infinity
        for i in 0 ..< logPbase.count {
            let pb = exp(logPbase[i])
            kld += pb * (logPbase[i] - logPcodec[i])
            if logPbase[i] > bestBase {
                bestBase = logPbase[i]
                topBase = i
            }
            if logPcodec[i] > bestCodec {
                bestCodec = logPcodec[i]
                topCodec = i
            }
        }
        // KL ≥ 0 in theory; clamp tiny negative drift from fp rounding.
        if kld < 0 { kld = 0 }
        let pBaseNext = exp(logPbase[nextTokenId])
        let pCodecNext = exp(logPcodec[nextTokenId])
        return PositionMetrics(
            kld: kld,
            baselineTopIdx: topBase, codecTopIdx: topCodec,
            baselineNextProb: pBaseNext, codecNextProb: pCodecNext)
    }

    /// Aggregate per-position metrics into the summary stats reported
    /// by llama-perplexity's `--kl-divergence` mode. Sorts the KLD
    /// list once internally; cost is O(N log N).
    public static func aggregate(_ perPosition: [PositionMetrics]) -> AggregateMetrics {
        precondition(
            !perPosition.isEmpty,
            "KLDivergence.aggregate: positions list must not be empty")
        let klds = perPosition.map { $0.kld }
        let sorted = klds.sorted()
        let n = perPosition.count

        let meanKld = klds.reduce(0, +) / Double(n)
        let dps = perPosition.map { abs($0.baselineNextProb - $0.codecNextProb) }
        let sameTop = perPosition.reduce(into: 0) { acc, m in
            if m.baselineTopIdx == m.codecTopIdx { acc += 1 }
        }

        return AggregateMetrics(
            meanKld: meanKld,
            medianKld: quantile(sorted, 0.5),
            maxKld: sorted.last ?? 0,
            kld999: quantile(sorted, 0.999),
            kld99: quantile(sorted, 0.99),
            kld95: quantile(sorted, 0.95),
            kld90: quantile(sorted, 0.90),
            kld10: quantile(sorted, 0.10),
            kld05: quantile(sorted, 0.05),
            kld01: quantile(sorted, 0.01),
            sameTopFraction: Double(sameTop) / Double(n),
            meanDp: dps.reduce(0, +) / Double(n),
            rmsDp: (dps.map { $0 * $0 }.reduce(0, +) / Double(n)).squareRoot(),
            maxDp: dps.max() ?? 0,
            nPositions: n)
    }

    /// Human-readable single-line summary matching the format
    /// `kld_vs_baseline.py` prints per codec row.
    public static func summaryLine(
        label: String, metrics: AggregateMetrics
    ) -> String {
        func fmt(_ v: Double) -> String { String(format: "%.4f", v) }
        return "\(label): mean_kld=\(fmt(metrics.meanKld)) "
            + "med=\(fmt(metrics.medianKld)) "
            + "95%=\(fmt(metrics.kld95)) "
            + "99%=\(fmt(metrics.kld99)) "
            + "99.9%=\(fmt(metrics.kld999)) "
            + "max=\(fmt(metrics.maxKld)) "
            + "same_top=\(String(format: "%.4f", metrics.sameTopFraction)) "
            + "mean_dp=\(fmt(metrics.meanDp)) "
            + "n=\(metrics.nPositions)"
    }

    // MARK: - Private helpers

    /// Numerically-stable log-softmax in Double. Subtract max for
    /// stability, then `x - log(Σ exp(x'))`.
    private static func logSoftmax(_ logits: [Float]) -> [Double] {
        var maxL: Float = -.infinity
        for v in logits where v > maxL { maxL = v }
        var sumExp: Double = 0
        let centered = logits.map { Double($0 - maxL) }
        for v in centered { sumExp += exp(v) }
        let logSumExp = log(sumExp)
        return centered.map { $0 - logSumExp }
    }

    /// Linear-interpolation quantile (numpy default; matches what the
    /// llama-perplexity `--kl-divergence` percentiles report). The
    /// high-percentile rows (99 / 99.9) need to surface heavy-tail
    /// outliers, which a nearest-rank-lower variant would lose.
    /// `q` in [0, 1].
    private static func quantile(_ sorted: [Double], _ q: Double) -> Double {
        if sorted.isEmpty { return 0 }
        let pos = Double(sorted.count - 1) * q
        let lo = Int(pos.rounded(.down))
        let hi = min(sorted.count - 1, lo + 1)
        let frac = pos - Double(lo)
        return sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }
}
