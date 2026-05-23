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

// ─── Tree drafter (research candidate #3 — tree-verify) ─────────────
//
// Linear γ=2 spec decode caps at 1.83 expected accepted tokens per
// verify cycle (83% acceptance × 2 candidates + 1 verified). Tree
// drafters (EAGLE-2 / Sequoia / DDTree-MLX) expand multiple
// continuations per draft step, verifying ALL of them in one forward
// pass with a tree-causal attention mask. At γ=8 tree (e.g., 2-branch
// at depth 3) with 50-65% per-branch acceptance, expected accepted
// tokens jumps to 3-4 → **1.7-2× decode** over linear.
//
// The scaffolding below is the API contract. The SpecDecode driver
// integration (tree-causal SDPA mask + walk-longest-accepted-path
// + cache rollback to accepted depth) lands in a follow-up iter.

/// One node in a draft tree. The root represents the first proposed
/// token following `history`; each child represents a possible
/// continuation after its parent. A linear chain (one child per node)
/// degenerates to the existing `[Int]` candidate sequence.
public struct DraftTreeNode: Sendable {
    /// The token this node proposes.
    public let token: Int
    /// Continuations after this token. Empty at leaves.
    public let children: [DraftTreeNode]

    public init(token: Int, children: [DraftTreeNode] = []) {
        self.token = token
        self.children = children
    }

    /// Total node count (root + all descendants).
    public var size: Int {
        return 1 + children.reduce(0) { $0 + $1.size }
    }

    /// Maximum depth (root alone = 1).
    public var depth: Int {
        return 1 + (children.map(\.depth).max() ?? 0)
    }
}

/// A drafter that proposes a tree of candidate continuations. The
/// SpecDecode driver walks the tree, flattens to a tree-causal
/// attention mask, and verifies all candidates in ONE forward pass —
/// accepting the longest matching prefix.
///
/// Implementations adapter-wrap existing `Drafter`s for the degenerate
/// linear case (see `Drafter.proposeTreeLinear`).
public protocol TreeDrafter: AnyObject {
    /// Propose a tree rooted at the first token after `history`.
    /// Returns `nil` when the drafter has no confident proposal — the
    /// driver falls back to a plain decode step.
    ///
    /// `maxDepth` caps the depth of the returned tree (root counts as
    /// depth 1). `maxNodes` caps the total node count so the driver
    /// can budget the verify forward pass.
    func proposeTree(history: [Int], maxDepth: Int, maxNodes: Int) -> DraftTreeNode?
}

extension Drafter {
    /// Convert a linear `propose` result into a degenerate tree (one
    /// child per node). Used to expose any existing `Drafter` as a
    /// `TreeDrafter` without adding a real branching policy.
    public func proposeTreeLinear(history: [Int], maxDepth: Int) -> DraftTreeNode? {
        let linear = propose(history: history, gamma: maxDepth)
        guard let last = linear.last else { return nil }
        // Build from leaf inward so children-of-children compose correctly.
        var node = DraftTreeNode(token: last, children: [])
        for t in linear.dropLast().reversed() {
            node = DraftTreeNode(token: t, children: [node])
        }
        return node
    }
}

/// Adapter: any linear `Drafter` becomes a `TreeDrafter` that emits a
/// degenerate (single-branch) tree. Useful for A/B'ing the tree-verify
/// driver against the linear baseline using the existing NGramDrafter.
public final class LinearTreeAdapter: TreeDrafter {
    public let inner: Drafter
    public init(_ inner: Drafter) { self.inner = inner }
    public func proposeTree(history: [Int], maxDepth: Int, maxNodes _: Int) -> DraftTreeNode? {
        inner.proposeTreeLinear(history: history, maxDepth: maxDepth)
    }
}
