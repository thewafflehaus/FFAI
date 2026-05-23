// Drafter — interface for speculative-decode candidate proposal.
//
// A drafter takes the token history so far and proposes one or more
// candidate next tokens. The target model then verifies via
// `forwardManyAllLogits([lastAccepted, ...candidates])` and either
// accepts (commit candidates) or rejects (restore caches, commit the
// model's actual choice from the verify logits).
//
// Implementations live alongside this protocol:
//   * `NGramDrafter` — prompt-lookup n-gram. Zero ML cost.
//   * `ANEMTPDrafter` — runs the trained MTP head on the ANE via the
//     mlpackage built at /Users/tom/models/Qwen3.6-35B-A3B-mtp.mlpackage.
//     (Implementation lands when the ANEDrafter Swift wrapper is wired.)
//   * `GreedyDrafter` — stub that always returns nil. Useful for
//     bench-comparing the spec-decode driver overhead vs raw decode.

import Foundation

/// A drafter proposes candidate next tokens for the target model to
/// verify in a speculative-decode loop.
public protocol Drafter: AnyObject {
    /// Propose up to `gamma` candidate tokens. Implementations may
    /// return fewer (or `nil` / empty) if they can't make a confident
    /// proposal — the driver falls back to a plain decode step in
    /// that case.
    ///
    /// `history` is the full sequence so far (prompt + generated).
    /// `gamma` is the maximum candidates the driver is asking for.
    func propose(history: [Int], gamma: Int) -> [Int]
}

/// Prompt-lookup n-gram drafter — zero ML cost; works well on
/// repetitive contexts (code, structured chat).
///
/// Algorithm (matches the typical prompt-lookup decoding paper):
///   1. Take the last `nMatch` tokens of `history` as the lookup key.
///   2. Scan backwards through `history` looking for a previous
///      occurrence of that key.
///   3. If found, return the next `gamma` tokens AFTER that earlier
///      occurrence — those are the candidate next tokens.
///   4. If not found, fall back to shorter keys (`nMatch - 1`,
///      `nMatch - 2`, ...) before giving up.
///
/// Common heuristic: start with `nMatch = 3` (trigrams) and fall back
/// to bigrams + unigrams.
public final class NGramDrafter: Drafter {
    /// Largest match length to try first. Falls back to shorter lengths
    /// if no longer match is found. Typical: 3 (trigram).
    public let maxNMatch: Int
    /// Smallest match length the drafter will try before giving up.
    /// Default 1 (unigram lookup is usually too noisy; raising to 2
    /// avoids spurious draft.)
    public let minNMatch: Int

    public init(maxNMatch: Int = 3, minNMatch: Int = 2) {
        precondition(maxNMatch >= minNMatch && minNMatch >= 1,
                     "NGramDrafter: maxNMatch (\(maxNMatch)) must be ≥ minNMatch (\(minNMatch)) ≥ 1")
        self.maxNMatch = maxNMatch
        self.minNMatch = minNMatch
    }

    public func propose(history: [Int], gamma: Int) -> [Int] {
        precondition(gamma >= 0, "NGramDrafter.propose: gamma must be ≥ 0")
        guard gamma > 0, !history.isEmpty else { return [] }

        // Try longest match first, fall back to shorter. Returns the
        // FIRST (most-recent) occurrence's continuation. High accept
        // probability when the trigram repeats.
        //
        // Empirically (Qwen3.6-A3B + bench in SpecDecodeBenchTests):
        // trigram acceptance ~83%, bigram ~50%, unigram-frequency ~35%.
        // Spec-decode break-even acceptance at γ=2 with the current
        // verify-cost ratio is ~70% — so unigram is a NET LOSS even
        // though it increases proposal rate. Better to return [] and
        // let the driver run a single decode step (52 ms) than waste
        // a 77 ms batched verify on a low-confidence proposal.
        for nMatch in stride(from: maxNMatch, through: minNMatch, by: -1) {
            guard history.count >= nMatch else { continue }
            let keyStart = history.count - nMatch
            let key = Array(history[keyStart..<history.count])
            // Scan backwards from just-before the key (so we don't
            // match the key against itself).
            var probe = keyStart - 1
            while probe >= nMatch - 1 {
                if matches(history, at: probe - (nMatch - 1), key: key) {
                    let candidateStart = probe + 1
                    let candidateEnd = Swift.min(candidateStart + gamma,
                                                  history.count)
                    if candidateEnd > candidateStart {
                        return Array(history[candidateStart..<candidateEnd])
                    }
                }
                probe -= 1
            }
        }
        return []
    }

    @inline(__always)
    private func matches(_ history: [Int], at start: Int, key: [Int]) -> Bool {
        if start < 0 || start + key.count > history.count { return false }
        for i in 0..<key.count {
            if history[start + i] != key[i] { return false }
        }
        return true
    }
}

/// Stub drafter that never proposes anything. Used to measure pure
/// spec-decode driver overhead against the no-spec baseline.
public final class NeverDrafter: Drafter {
    public init() {}
    public func propose(history: [Int], gamma: Int) -> [Int] { [] }
}
