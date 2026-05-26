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
// Perplexity — log-likelihood / perplexity / KL-divergence over a
// fixed token sequence.
//
// Standalone helper. Not folded into `generate(...)` because perplexity
// computation requires capturing logits at every step (extra work +
// memory) which would tax the greedy fast path. The bench harness
// calls this directly when `--method wikitext2` runs; the CLI / your
// own code can opt into `--stats` plus a separate
// `Perplexity.compute(model:tokens:)` call.
//
// ─── Why fp32 ────────────────────────────────────────────────────────
// log_softmax computes log(Σ exp(x_i − max_x)) over the full vocab. With
// 128–152K vocab and a peaky distribution, the partial sum lives in
// roughly [1, 10⁵]. bf16's 7 mantissa bits can't represent that range
// without precision loss that pollutes the perplexity 3rd decimal —
// enough to obscure the differences between quantization tiers we're
// usually trying to measure. fp16 has 11 mantissa bits, also marginal.
// fp32 it is. The extra work is ~1µs per token; we only do this
// offline for evaluation, so the cost is irrelevant.
//
// ─── KL divergence (paired-run) ──────────────────────────────────────
// `Perplexity.klDivergence(reference:candidate:tokens:)` runs both
// models forward in lockstep, capturing per-position log-softmax
// distributions and accumulating KL(p_ref || q_cand) per position.
// Both models must share the same vocabulary — the natural pairing is a
// quantized variant against its bf16 unquantized parent. The reference
// should be the unquantized variant if it fits in memory; using a
// smaller reference makes the KL number a measure of family closeness
// rather than quantization fidelity.

import Foundation

public enum Perplexity {
    public struct Result: Sendable, Equatable {
        /// `exp(-mean(log p(token_t | tokens_<t)))`.
        public let perplexity: Double
        /// Mean negative log-likelihood (nats).
        public let meanNegLogLikelihood: Double
        /// Number of scored positions (tokens.count - 1).
        public let scoredTokens: Int
    }

    public struct KLDResult: Sendable, Equatable {
        /// Mean KL(p_ref || q_cand) across scored positions, in nats.
        public let meanKLDivergence: Double
        /// Number of positions scored. Equal to `tokens.count - 1`
        /// minus any positions skipped due to vocab-size mismatch.
        public let scoredTokens: Int
    }

    /// Score the sequence against the model. Token at index `t+1` is
    /// the target for the prediction conditioned on tokens `[0..=t]`.
    /// Uses `forward(...)` (not `forwardSample`) so we get logits, then
    /// reads them back to CPU once per step. Cost is one extra readback
    /// per token vs greedy; acceptable for offline evaluation.
    public static func compute(
        model: Model,
        tokens: [Int],
        device: Device = .shared
    ) -> Result {
        precondition(
            tokens.count >= 2,
            "Perplexity.compute requires at least 2 tokens (one context + one target)")
        let caches = model.engine.makeLayerCaches(maxSeq: nil, device: device)
        var nll = 0.0
        var scored = 0
        for t in 0 ..< (tokens.count - 1) {
            let logits = model.engine.forward(
                tokenId: tokens[t], position: t,
                caches: caches, device: device)
            let target = tokens[t + 1]
            nll += negLogSoftmaxAt(logits: logits, index: target)
            scored += 1
        }
        let mean = nll / Double(scored)
        return Result(
            perplexity: exp(mean),
            meanNegLogLikelihood: mean,
            scoredTokens: scored)
    }

    /// KL(reference || candidate) over a fixed token sequence. Both
    /// models must use the same tokenizer / vocab. Runs them in
    /// lockstep — at each position, compute reference probs +
    /// candidate log-probs from full logits, then accumulate
    /// `Σ p_ref(v) * (log p_ref(v) − log q_cand(v))`.
    ///
    /// Memory cost: two simultaneously-loaded models + two KV caches.
    /// Pick the reference and candidate sizes such that both fit in
    /// device memory. The recommended pairing is the bf16 variant of
    /// the same checkpoint as `reference` against the quantized
    /// variant as `candidate`.
    public static func klDivergence(
        reference: Model,
        candidate: Model,
        tokens: [Int],
        device: Device = .shared
    ) -> KLDResult {
        precondition(
            tokens.count >= 2,
            "klDivergence requires at least 2 tokens")
        let refCaches = reference.engine.makeLayerCaches(maxSeq: nil, device: device)
        let candCaches = candidate.engine.makeLayerCaches(maxSeq: nil, device: device)

        var totalKL = 0.0
        var scored = 0
        for t in 0 ..< (tokens.count - 1) {
            let refLogits = reference.engine.forward(
                tokenId: tokens[t], position: t,
                caches: refCaches, device: device)
            let candLogits = candidate.engine.forward(
                tokenId: tokens[t], position: t,
                caches: candCaches, device: device)
            // Skip positions where vocab sizes mismatch (defensive — KL
            // over heterogeneous vocabs isn't meaningful).
            let refV = refLogits.shape.last ?? 0
            let candV = candLogits.shape.last ?? 0
            guard refV > 0, refV == candV else { continue }
            totalKL += klAtPosition(refLogits: refLogits, candLogits: candLogits)
            scored += 1
        }
        let mean = scored > 0 ? totalKL / Double(scored) : 0
        return KLDResult(meanKLDivergence: mean, scoredTokens: scored)
    }

    // MARK: - Internal: log-softmax(logits)[index], in fp32
    // Exposed at internal access for direct test coverage (see
    // PerplexityTests). The public surface remains compute(...) / klDivergence(...).

    internal static func negLogSoftmaxAt(logits: Tensor, index: Int) -> Double {
        let n = logits.shape.last ?? 0
        precondition(index >= 0 && index < n, "target index out of range")
        let ptr = logits.buffer.contents().advanced(by: logits.offset)

        var maxV: Double = -.infinity
        var targetLogit: Double = 0
        switch logits.dtype {
        case .f32:
            let f = ptr.assumingMemoryBound(to: Float.self)
            for i in 0 ..< n {
                let v = Double(f[i])
                if v > maxV { maxV = v }
            }
            targetLogit = Double(f[index])
        case .f16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n {
                let v = Double(float16ToFloat32(f[i]))
                if v > maxV { maxV = v }
            }
            targetLogit = Double(float16ToFloat32(f[index]))
        case .bf16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n {
                let v = Double(bfloat16ToFloat32(f[i]))
                if v > maxV { maxV = v }
            }
            targetLogit = Double(bfloat16ToFloat32(f[index]))
        default:
            fatalError("Perplexity: unsupported logits dtype \(logits.dtype)")
        }

        var sum: Double = 0
        switch logits.dtype {
        case .f32:
            let f = ptr.assumingMemoryBound(to: Float.self)
            for i in 0 ..< n { sum += exp(Double(f[i]) - maxV) }
        case .f16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n { sum += exp(Double(float16ToFloat32(f[i])) - maxV) }
        case .bf16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n { sum += exp(Double(bfloat16ToFloat32(f[i])) - maxV) }
        default:
            fatalError("Perplexity: unsupported logits dtype \(logits.dtype)")
        }
        let logZ = maxV + log(sum)
        return logZ - targetLogit  // = -log p(target)
    }

    /// Per-position KL(p_ref || q_cand). Decodes both logit vectors to
    /// fp32, computes log-softmax of each, then accumulates
    /// `Σ p_ref(v) * (log p_ref(v) − log q_cand(v))`. Internal access
    /// for direct test coverage.
    internal static func klAtPosition(refLogits: Tensor, candLogits: Tensor) -> Double {
        let n = refLogits.shape.last ?? 0
        precondition((candLogits.shape.last ?? 0) == n, "KL: vocab size mismatch")
        let refLog = decodeLogSoftmax(logits: refLogits)
        let candLog = decodeLogSoftmax(logits: candLogits)
        var kl = 0.0
        for v in 0 ..< n {
            let p = exp(refLog[v])
            kl += p * (refLog[v] - candLog[v])
        }
        return kl
    }

    /// Decode logits to a fp32 log-softmax vector (`log p(v)` for all v).
    /// Internal access for direct test coverage.
    internal static func decodeLogSoftmax(logits: Tensor) -> [Double] {
        let n = logits.shape.last ?? 0
        var raw = [Double](repeating: 0, count: n)
        let ptr = logits.buffer.contents().advanced(by: logits.offset)
        switch logits.dtype {
        case .f32:
            let f = ptr.assumingMemoryBound(to: Float.self)
            for i in 0 ..< n { raw[i] = Double(f[i]) }
        case .f16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n { raw[i] = Double(float16ToFloat32(f[i])) }
        case .bf16:
            let f = ptr.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n { raw[i] = Double(bfloat16ToFloat32(f[i])) }
        default:
            fatalError("Perplexity: unsupported logits dtype \(logits.dtype)")
        }
        var maxV = -Double.infinity
        for v in raw where v > maxV { maxV = v }
        var sum = 0.0
        for v in raw { sum += exp(v - maxV) }
        let logZ = maxV + log(sum)
        for i in 0 ..< n { raw[i] -= logZ }
        return raw
    }

    private static func float16ToFloat32(_ x: UInt16) -> Float {
        var v = x
        return withUnsafePointer(to: &v) {
            $0.withMemoryRebound(to: Float16.self, capacity: 1) { Float($0.pointee) }
        }
    }

    private static func bfloat16ToFloat32(_ x: UInt16) -> Float {
        var bits = UInt32(x) << 16
        return withUnsafePointer(to: &bits) {
            $0.withMemoryRebound(to: Float.self, capacity: 1) { $0.pointee }
        }
    }
}
