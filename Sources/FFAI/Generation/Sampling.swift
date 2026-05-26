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
// Sampling — CPU sampling pipeline over a vocab-sized logits tensor.
//
// The greedy fast path is handled by Ops.argmax + LanguageModel.forwardSample
// (single 4-byte readback, no logits cross CPU↔GPU). When the user picks
// non-greedy sampling — temperature > 0 with any of top-K / top-P / min-P /
// repetition-penalty — we read the full vocab to CPU once per token and run
// the pipeline here. At Qwen3 vocab=151_936 fp16 that's ~304 KB per token,
// negligible bandwidth (~18 MB/s at 60 tok/s) on Apple Silicon's unified
// memory. A GPU softmax+categorical sample kernel is on the metaltile
// `ek/sampling-kernels` branch for the future fully-on-GPU path.
//
// Pipeline order (matches mlx-swift-lm / mlx-lm):
//
//   1. Repetition penalty (logit / penalty for seen tokens with positive
//      logits; logit * penalty for negative ones — Hugging Face convention)
//   2. Temperature scaling (logit / T; T == 0 → argmax shortcut)
//   3. Top-K filter (keep K largest logits)
//   4. Top-P / nucleus filter (keep tokens until cumulative softmax ≥ P)
//   5. Min-P filter (keep tokens with prob ≥ min_p × max_prob)
//   6. Softmax + categorical sample with a uniform draw from the RNG

import Foundation

public enum Sampling {

    // ─── Debug helpers ───────────────────────────────────────────────

    /// Debug: top-N highest logits as (index, value) pairs.
    public static func topN(_ logits: Tensor, n: Int) -> [(Int, Float)] {
        let values = decodeF32(logits)
        let indexed = values.enumerated().sorted { $0.element > $1.element }
        return indexed.prefix(n).map { ($0.offset, $0.element) }
    }

    /// Greedy: argmax over a 1D logits tensor. Tie-breaks toward the
    /// smallest index, matching the GPU argmax kernel.
    public static func argmax(_ logits: Tensor) -> Int {
        let values = decodeF32(logits)
        var best = 0
        var bestVal = values[0]
        for i in 1..<values.count where values[i] > bestVal {
            bestVal = values[i]
            best = i
        }
        return best
    }

    // ─── Full CPU sampling pipeline ──────────────────────────────────

    /// Sample a token id from `logits` per the given `parameters`. When
    /// `parameters.temperature == 0` the call short-circuits to a CPU
    /// argmax (deterministic, no RNG draw). `tokenHistory` is the
    /// already-generated prefix (prompt + decode so far); only used
    /// when `repetitionPenalty != 1.0`.
    public static func sample(
        _ logits: Tensor,
        parameters: GenerationParameters,
        rng: inout some RandomNumberGenerator,
        tokenHistory: [Int] = []
    ) -> Int {
        var values = decodeF32(logits)
        let vocab = values.count

        // (1) Repetition penalty
        if parameters.repetitionPenalty != 1.0 && !tokenHistory.isEmpty {
            let p = parameters.repetitionPenalty
            var seen = Set<Int>()
            for t in tokenHistory where t >= 0 && t < vocab {
                seen.insert(t)
            }
            for i in seen {
                let v = values[i]
                values[i] = v > 0 ? v / p : v * p
            }
        }

        // Greedy short-circuit. Skips the softmax / RNG draw entirely.
        if parameters.temperature == 0 {
            return argmaxOf(values)
        }

        // (2) Temperature
        if parameters.temperature != 1.0 {
            let invT = 1.0 / parameters.temperature
            for i in 0..<vocab { values[i] *= invT }
        }

        // Build a working list of (logit, index) so we can filter.
        // For top-K, sort once and keep the leading K. For top-P, sort
        // once and walk until cumulative softmax ≥ P. For min-P, no
        // sort needed.
        var indexed: [(idx: Int, logit: Float)] = []
        indexed.reserveCapacity(vocab)
        for i in 0..<vocab { indexed.append((i, values[i])) }

        let needsSort = parameters.topK > 0 || parameters.topP < 1.0
        if needsSort {
            indexed.sort { $0.logit > $1.logit }
        }

        // (3) Top-K — keep K largest
        if parameters.topK > 0 && parameters.topK < indexed.count {
            indexed.removeLast(indexed.count - parameters.topK)
        }

        // (4) Top-P — keep until cumulative softmax ≥ topP
        if parameters.topP < 1.0 && !indexed.isEmpty {
            // Stable softmax over the (sorted) candidates.
            let maxL = indexed.first?.logit ?? 0
            var cum = 0.0
            var sumExp = 0.0
            for x in indexed { sumExp += Double(expf(x.logit - maxL)) }
            var cutoff = indexed.count
            for i in 0..<indexed.count {
                cum += Double(expf(indexed[i].logit - maxL)) / sumExp
                if cum >= Double(parameters.topP) { cutoff = i + 1; break }
            }
            indexed.removeLast(indexed.count - cutoff)
        }

        // (5) Min-P — keep tokens with prob ≥ min_p × max_prob.
        // exp(logit - maxL) is the unnormalized prob ratio; threshold
        // becomes logit ≥ maxL + log(min_p).
        if parameters.minP > 0 && !indexed.isEmpty {
            let maxL = indexed.first?.logit ?? indexed.map(\.logit).max() ?? 0
            let threshLogit = maxL + logf(parameters.minP)
            indexed = indexed.filter { $0.logit >= threshLogit }
        }

        if indexed.isEmpty {
            // Defensive: every filter rejected everything. Fall back to
            // the unfiltered argmax so we still emit something sensible.
            return argmaxOf(values)
        }

        // (6) Softmax + categorical sample
        let maxL = indexed.map(\.logit).max() ?? 0
        var weights = [Double](repeating: 0, count: indexed.count)
        var total = 0.0
        for i in 0..<indexed.count {
            let w = Double(expf(indexed[i].logit - maxL))
            weights[i] = w
            total += w
        }
        let u = Double.random(in: 0..<1, using: &rng) * total
        var acc = 0.0
        for i in 0..<indexed.count {
            acc += weights[i]
            if acc >= u { return indexed[i].idx }
        }
        return indexed.last!.idx
    }

    // ─── Internal helpers ────────────────────────────────────────────

    /// Decode logits to a fp32 Swift array. Handles fp32 / fp16 / bf16.
    public static func decodeF32(_ logits: Tensor) -> [Float] {
        let n = logits.elementCount
        var out = [Float](repeating: 0, count: n)
        switch logits.dtype {
        case .f32:
            let arr = logits.toArray(as: Float.self)
            for i in 0..<n { out[i] = arr[i] }
        case .f16:
            let arr = logits.toArray(as: Float16.self)
            for i in 0..<n { out[i] = Float(arr[i]) }
        case .bf16:
            let bits = logits.toArray(as: UInt16.self)
            for i in 0..<n { out[i] = Float(bitPattern: UInt32(bits[i]) << 16) }
        default:
            fatalError("Sampling: unsupported logits dtype \(logits.dtype)")
        }
        return out
    }

    private static func argmaxOf(_ values: [Float]) -> Int {
        var best = 0
        var bestVal = values[0]
        for i in 1..<values.count where values[i] > bestVal {
            bestVal = values[i]; best = i
        }
        return best
    }
}

// MARK: - Seeded RNG

/// SplitMix64 PRNG — same seed always produces the same sequence.
/// Used by `GenerationParameters.makeRNG()` when `seed != nil` so
/// non-greedy sampling is reproducible run-to-run.
public struct SeededRandomNumberGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { self.state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

/// Type-erased RNG so callers can hand around a concrete `RandomNumberGenerator`
/// without leaking whether it's seeded vs system-random.
public struct AnyRandomNumberGenerator: RandomNumberGenerator {
    private var fill: () -> UInt64
    public mutating func next() -> UInt64 { fill() }

    public init<R: RandomNumberGenerator>(_ wrapped: R) {
        var w = wrapped
        self.fill = { w.next() }
    }
}

public extension GenerationParameters {
    /// Build an RNG for this generation: seeded + deterministic when
    /// `seed != nil`, system random otherwise.
    func makeRNG() -> AnyRandomNumberGenerator {
        if let s = seed {
            return AnyRandomNumberGenerator(SeededRandomNumberGenerator(seed: s))
        }
        return AnyRandomNumberGenerator(SystemRandomNumberGenerator())
    }
}
