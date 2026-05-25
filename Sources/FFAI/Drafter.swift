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

    // ─── ITER 69 (Bagel 2): tree-flatten + tree-causal mask synthesis ──
    //
    // The verify forward needs (a) a linear sequence of tokens to feed
    // the model and (b) a [T, T] attention mask over those tokens
    // describing the tree structure. These pure-Swift helpers produce
    // both; the future tree-causal SDPA kernel consumes them.

    /// Depth-first flatten of the tree. Returns:
    /// - `tokens[i]`: the token at flat position `i` (root at 0).
    /// - `parentIndex[i]`: the flat-index of node `i`'s parent, or
    ///   `-1` for the root (`i == 0`).
    /// - `pathFromRoot[i]`: indices of node `i`'s ancestors INCLUDING
    ///   `i` itself, root-first (length = node `i`'s depth in the
    ///   tree). Used by the verify driver to walk the accepted prefix.
    public func flatten() -> (tokens: [Int],
                               parentIndex: [Int],
                               pathFromRoot: [[Int]]) {
        var tokens: [Int] = []
        var parent: [Int] = []
        var paths: [[Int]] = []
        var indexStack: [Int] = []  // current DFS path-to-this-node, by flat-index
        func recurse(_ node: DraftTreeNode, parentIdx: Int) {
            let myIdx = tokens.count
            tokens.append(node.token)
            parent.append(parentIdx)
            indexStack.append(myIdx)
            paths.append(indexStack)
            for child in node.children {
                recurse(child, parentIdx: myIdx)
            }
            indexStack.removeLast()
        }
        recurse(self, parentIdx: -1)
        return (tokens, parent, paths)
    }

    /// Tree-causal additive attention mask for the in-tree positions.
    /// Returns a flat `[T*T]` array where `mask[i*T + j]` is:
    ///   - `0.0` if flat-index `j` is an ancestor of `i` in the tree,
    ///     or `j == i` itself (the diagonal — every token attends to
    ///     itself).
    ///   - `-Float.infinity` otherwise (siblings / cousins / disjoint
    ///     branches — must NOT attend across alternative paths).
    ///
    /// Caller adds this mask onto the attention scores BEFORE softmax.
    /// The cached-prefix portion of attention (positions < `base_kv`)
    /// is always-attended (full causal-to-cache) and is NOT included
    /// here — wrap this mask into the kernel's mask param only for the
    /// in-block region.
    public func treeCausalMask() -> (mask: [Float], t: Int) {
        let (_, parent, _) = flatten()
        let t = parent.count
        var mask = [Float](repeating: -Float.infinity, count: t * t)
        for i in 0..<t {
            // Walk i's ancestor chain and set mask[i, ancestor] = 0.
            var node = i
            while node != -1 {
                mask[i * t + node] = 0.0
                node = parent[node]
            }
        }
        return (mask, t)
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

/// ITER 68 (Bagel 2): real branching n-gram drafter.
///
/// Extends `NGramDrafter`'s "scan history for n-gram matches" lookup
/// from "first match → linear chain" to "all matches → top-K
/// continuations per depth → branching tree." Each node expands its
/// `branchingFactor` most-frequent continuations from past occurrences
/// of the current key, recursively up to `maxDepth` or `maxNodes`
/// total.
///
/// Algorithm (depth-first build):
/// 1. Compose key = last `nMatch` tokens of `history + accumulated
///    branch tokens up to current node`.
/// 2. Scan history for ALL occurrences of `key`; collect the token
///    that came immediately after each occurrence, with frequency.
/// 3. Take top-K (= `branchingFactor`) most-frequent tokens as the
///    current node's children.
/// 4. For each child, recurse with `extended_history = history +
///    branch_tokens_so_far + child_token`.
/// 5. Stop at `maxDepth`, no candidates, or `maxNodes` reached.
///
/// The fallback to shorter n-grams (`maxNMatch → minNMatch`) mirrors
/// `NGramDrafter` — longer keys give higher per-branch acceptance but
/// fewer matches.
///
/// Cost / payoff sketch (Qwen3.6-A3B baseline NGram acceptance ~83%
/// per-step at γ=2):
/// - Linear γ=2 ⇒ ≤2 candidate tokens, E[accepted] ≈ 1.83.
/// - Tree depth=2, branching=2 ⇒ 7 candidates (1+2+4), E[accepted] up
///   to ~3 if top-2 each cover the target's argmax.
/// - Verify cost rises with candidate count (causal-tree attention
///   over T candidates ≈ T·QK·V); the win comes if `accepted/verify`
///   ratio improves vs linear.
///
/// Conforms to both `Drafter` (linear γ fallback via `propose`) and
/// `TreeDrafter` (real branching via `proposeTree`).
public final class NGramTreeDrafter: Drafter, TreeDrafter {
    public let maxNMatch: Int
    public let minNMatch: Int
    /// Number of children per node (top-K continuations).
    public let branchingFactor: Int

    public init(maxNMatch: Int = 3, minNMatch: Int = 2,
                branchingFactor: Int = 2) {
        precondition(maxNMatch >= minNMatch && minNMatch >= 1,
                     "NGramTreeDrafter: maxNMatch (\(maxNMatch)) must be ≥ minNMatch (\(minNMatch)) ≥ 1")
        precondition(branchingFactor >= 1,
                     "NGramTreeDrafter: branchingFactor must be ≥ 1")
        self.maxNMatch = maxNMatch
        self.minNMatch = minNMatch
        self.branchingFactor = branchingFactor
    }

    // MARK: Drafter — linear γ fallback (top-1 chain).

    public func propose(history: [Int], gamma: Int) -> [Int] {
        precondition(gamma >= 0, "NGramTreeDrafter.propose: gamma must be ≥ 0")
        guard gamma > 0, !history.isEmpty else { return [] }
        var chain: [Int] = []
        chain.reserveCapacity(gamma)
        var extended = history
        for _ in 0..<gamma {
            guard let token = topKContinuations(of: extended, k: 1).first else { break }
            chain.append(token)
            extended.append(token)
        }
        return chain
    }

    // MARK: TreeDrafter — branching tree.

    public func proposeTree(history: [Int], maxDepth: Int,
                             maxNodes: Int) -> DraftTreeNode? {
        guard maxDepth > 0, maxNodes > 0, !history.isEmpty else { return nil }
        var nodeBudget = maxNodes
        let roots = topKContinuations(of: history, k: branchingFactor)
        guard let rootTok = roots.first else { return nil }
        nodeBudget -= 1
        // Build root subtree first; siblings are inserted as additional
        // children of the *root* (i.e. the root has up to K children).
        // The TREE shape is: root token = the most-likely next, then
        // both `roots[0]` and `roots[1..K]` become its children-line.
        // To keep things simple + matching the API (single root),
        // we emit the top-1 as the root's token and expand each of the
        // top-K (including #1) as children of the root recursively
        // grown downwards.
        var rootChildren: [DraftTreeNode] = []
        rootChildren.reserveCapacity(roots.count)
        // First-level expansion: each `roots[i]` becomes a child node
        // (depth-1) of the root. Recurse for `maxDepth - 1`.
        for tok in roots {
            guard nodeBudget > 0 else { break }
            nodeBudget -= 1  // RESERVE this child's slot before recursing.
            var path = history
            path.append(tok)
            let subtree = buildSubtree(extendedHistory: path,
                                        remainingDepth: maxDepth - 1,
                                        nodeBudget: &nodeBudget)
            // The path's last token IS the child's token; subtree
            // hangs off it.
            rootChildren.append(DraftTreeNode(token: tok,
                                               children: subtree?.children ?? []))
        }
        return DraftTreeNode(token: rootTok, children: rootChildren)
    }

    /// Recursive helper: build a chain-or-tree rooted at the implicit
    /// next-token, capped by `remainingDepth` and `nodeBudget`. Returns
    /// `nil` if no continuation is found at this depth. Budget is
    /// reserved BEFORE recursing so the cap is a hard upper bound on
    /// `tree.size`.
    private func buildSubtree(extendedHistory: [Int],
                               remainingDepth: Int,
                               nodeBudget: inout Int) -> DraftTreeNode? {
        guard remainingDepth > 0, nodeBudget > 0 else { return nil }
        let toks = topKContinuations(of: extendedHistory, k: branchingFactor)
        guard let firstTok = toks.first else { return nil }
        var children: [DraftTreeNode] = []
        children.reserveCapacity(toks.count)
        for tok in toks {
            guard nodeBudget > 0 else { break }
            nodeBudget -= 1  // RESERVE before recursing.
            var deeper = extendedHistory
            deeper.append(tok)
            let sub = buildSubtree(extendedHistory: deeper,
                                    remainingDepth: remainingDepth - 1,
                                    nodeBudget: &nodeBudget)
            children.append(DraftTreeNode(token: tok,
                                           children: sub?.children ?? []))
        }
        return DraftTreeNode(token: firstTok, children: children)
    }

    /// Top-K most frequent continuations after `history` (longest
    /// available n-gram → fall back to shorter). Returns up to `k`
    /// tokens sorted by occurrence count descending; ties broken by
    /// token id ascending for determinism.
    private func topKContinuations(of history: [Int], k: Int) -> [Int] {
        guard !history.isEmpty, k > 0 else { return [] }
        for nMatch in stride(from: maxNMatch, through: minNMatch, by: -1) {
            guard history.count >= nMatch else { continue }
            let keyStart = history.count - nMatch
            let key = Array(history[keyStart..<history.count])
            var counts: [Int: Int] = [:]
            // Scan ALL prior occurrences (not just the most recent).
            var probe = keyStart - 1
            while probe >= nMatch - 1 {
                if matches(history, at: probe - (nMatch - 1), key: key) {
                    let candidateIdx = probe + 1
                    if candidateIdx < history.count {
                        counts[history[candidateIdx], default: 0] += 1
                    }
                }
                probe -= 1
            }
            if !counts.isEmpty {
                return counts.sorted {
                    if $0.value != $1.value { return $0.value > $1.value }
                    return $0.key < $1.key
                }.prefix(k).map(\.key)
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
