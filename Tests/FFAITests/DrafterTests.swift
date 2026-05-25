// Host-side unit test for n-gram drafter — no GPU needed.

import Foundation
import Testing
@testable import FFAI

@Suite("TreeDrafter — scaffolding (ITER 61)")
struct TreeDrafterScaffoldTests {

    @Test("DraftTreeNode size + depth on a linear chain")
    func linearChainStructure() {
        // Build: 1 → 2 → 3 (depth 3, size 3, single child each).
        let leaf = DraftTreeNode(token: 3)
        let mid = DraftTreeNode(token: 2, children: [leaf])
        let root = DraftTreeNode(token: 1, children: [mid])
        #expect(root.size == 3)
        #expect(root.depth == 3)
        #expect(root.children.count == 1)
    }

    @Test("DraftTreeNode size + depth on a branching tree")
    func branchingTreeStructure() {
        // Root → [A, B]; A → [C]; B → [D, E]. size=6, depth=3.
        let c = DraftTreeNode(token: 3)
        let d = DraftTreeNode(token: 4)
        let e = DraftTreeNode(token: 5)
        let a = DraftTreeNode(token: 1, children: [c])
        let b = DraftTreeNode(token: 2, children: [d, e])
        let root = DraftTreeNode(token: 0, children: [a, b])
        #expect(root.size == 6)
        #expect(root.depth == 3)
    }

    @Test("Drafter.proposeTreeLinear converts a linear propose to a chain tree")
    func linearAdapterMatchesChain() {
        let the = 1, cat = 2, sat = 3, on = 4
        let history = [the, cat, sat, on, the, cat, sat]
        let nGram = NGramDrafter(maxNMatch: 3, minNMatch: 2)
        // Linear: should propose [on, ?]; we only care that the tree
        // adapter wraps it as a single-child chain.
        let linear = nGram.propose(history: history, gamma: 2)
        guard let tree = nGram.proposeTreeLinear(history: history, maxDepth: 2) else {
            Issue.record("expected a non-nil tree")
            return
        }
        #expect(tree.size == linear.count)
        #expect(tree.depth == linear.count)
        // Walk single-child chain comparing tokens.
        var node: DraftTreeNode? = tree
        for tok in linear {
            #expect(node?.token == tok)
            node = node?.children.first
        }
    }

    @Test("LinearTreeAdapter wraps any Drafter as a TreeDrafter")
    func linearAdapterIsTreeDrafter() {
        let drafter = NGramDrafter(maxNMatch: 3, minNMatch: 1)
        let tree: TreeDrafter = LinearTreeAdapter(drafter)
        let history = [1, 2, 3, 4, 1, 2, 3]
        let result = tree.proposeTree(history: history, maxDepth: 2, maxNodes: 16)
        #expect(result != nil)
        if let r = result {
            var node: DraftTreeNode? = r
            while let n = node {
                #expect(n.children.count <= 1)
                node = n.children.first
            }
        }
    }

    @Test("LinearTreeAdapter returns nil when underlying drafter has nothing")
    func linearAdapterEmptyProposal() {
        let drafter = NeverDrafter()
        let tree = LinearTreeAdapter(drafter)
        let result = tree.proposeTree(history: [1, 2, 3], maxDepth: 4, maxNodes: 16)
        #expect(result == nil)
    }
}

@Suite("NGramTreeDrafter — branching tree from n-gram history (ITER 68)")
struct NGramTreeDrafterTests {

    @Test("proposeTree branches on multiple continuations of the same n-gram")
    func branchesOnMultipleContinuations() {
        // History where trigram "the cat sat" appears 3 times with
        // different continuations: ["on", "on", "down"]. Top-2 should be
        // "on" (count=2) then "down" (count=1).
        let the = 1, cat = 2, sat = 3, on = 4, down = 5
        let pad = 99 // filler tokens that don't interfere
        // [the,cat,sat, on, pad, the,cat,sat, on, pad, the,cat,sat, down, pad, the,cat,sat]
        let history = [the, cat, sat, on, pad,
                       the, cat, sat, on, pad,
                       the, cat, sat, down, pad,
                       the, cat, sat]
        let drafter = NGramTreeDrafter(maxNMatch: 3, minNMatch: 3,
                                        branchingFactor: 2)
        guard let tree = drafter.proposeTree(history: history,
                                              maxDepth: 1, maxNodes: 16) else {
            Issue.record("expected a non-nil tree")
            return
        }
        // Root token = top-1 = "on" (count=2)
        #expect(tree.token == on)
        // depth=1, branching=2 → root has 2 children: "on" and "down".
        #expect(tree.children.count == 2)
        #expect(tree.children.map(\.token).sorted() == [on, down].sorted())
    }

    @Test("proposeTree returns nil when n-gram never repeats")
    func nilOnNoMatch() {
        let history = [1, 2, 3, 4, 5]
        let drafter = NGramTreeDrafter(maxNMatch: 3, minNMatch: 2,
                                        branchingFactor: 2)
        let tree = drafter.proposeTree(history: history,
                                        maxDepth: 2, maxNodes: 16)
        #expect(tree == nil)
    }

    @Test("propose (linear) falls back to top-1 chain")
    func linearFallbackProposeChain() {
        // Bigram "a b" → "c" appears 3×, then once more.
        let a = 10, b = 11, c = 12
        let history = [a, b, c, 99, a, b, c, 99, a, b, c, 99, a, b]
        let drafter = NGramTreeDrafter(maxNMatch: 2, minNMatch: 2,
                                        branchingFactor: 3)
        let linear = drafter.propose(history: history, gamma: 1)
        #expect(linear == [c])
    }

    @Test("nodeBudget caps tree growth")
    func nodeBudgetCap() {
        // Construct a history with lots of trigram repeats.
        let t = [1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 1, 2, 3, 4, 7, 1, 2, 3, 4]
        let drafter = NGramTreeDrafter(maxNMatch: 3, minNMatch: 2,
                                        branchingFactor: 3)
        guard let tree = drafter.proposeTree(history: t,
                                              maxDepth: 4, maxNodes: 4) else {
            Issue.record("expected non-nil")
            return
        }
        // Tree size must respect the maxNodes cap.
        #expect(tree.size <= 4)
    }

    @Test("flatten + treeCausalMask: linear chain produces lower-triangular mask")
    func flattenLinearChain() {
        // chain 1 → 2 → 3 → 4 (depth 4, branching 1)
        let leaf = DraftTreeNode(token: 4)
        let n3 = DraftTreeNode(token: 3, children: [leaf])
        let n2 = DraftTreeNode(token: 2, children: [n3])
        let root = DraftTreeNode(token: 1, children: [n2])
        let (tokens, parent, paths) = root.flatten()
        #expect(tokens == [1, 2, 3, 4])
        #expect(parent == [-1, 0, 1, 2])
        // Path from root to each node (root-inclusive).
        #expect(paths[0] == [0])
        #expect(paths[1] == [0, 1])
        #expect(paths[2] == [0, 1, 2])
        #expect(paths[3] == [0, 1, 2, 3])
        // Linear chain mask = lower triangular (each token attends to
        // every prior).
        let (mask, t) = root.treeCausalMask()
        #expect(t == 4)
        for i in 0..<t {
            for j in 0..<t {
                let m = mask[i * t + j]
                let expected: Float = (j <= i) ? 0.0 : -Float.infinity
                #expect(m == expected)
            }
        }
    }

    // ─── ITER 71: tree-verify acceptance algorithm ───────────────────

    /// Helper to build the branching tree from earlier tests for verify cases.
    private func cousinsTree() -> DraftTreeNode {
        //         0 (root, tok=10)
        //        / \
        //      1     2     (tok=20, tok=30)
        //     /|     |
        //    3 4     5     (tok=40, tok=50, tok=60)
        let n3 = DraftTreeNode(token: 40)
        let n4 = DraftTreeNode(token: 50)
        let n5 = DraftTreeNode(token: 60)
        let n1 = DraftTreeNode(token: 20, children: [n3, n4])
        let n2 = DraftTreeNode(token: 30, children: [n5])
        return DraftTreeNode(token: 10, children: [n1, n2])
    }

    @Test("verify: root mismatch returns empty accepted + the bonus token")
    func verifyRootMismatch() {
        let tree = cousinsTree()
        // Oracle says the target wants token 99 first (not the root's 10).
        let oracleHistoryEnd = 99
        let result = tree.verify(oracleAtHistoryEnd: oracleHistoryEnd,
                                  oracle: { _ in 0 /* unused — root rejected */ })
        #expect(result.acceptedTokens.isEmpty)
        #expect(result.bonusToken == 99)
    }

    @Test("verify: oracle agrees with path 0→1→3, then diverges")
    func verifyAcceptsPathToN3() {
        let tree = cousinsTree()
        // After history, target wants 10 (root). At root (flat 0),
        // target wants 20 (descend into n1). At n1 (flat 1), target
        // wants 40 (descend into n3). At n3 (flat 2), target wants 999
        // (off-tree) — stop.
        let oracle: (Int) -> Int = { flat in
            switch flat {
            case 0: return 20      // pick n1 (token=20)
            case 1: return 40      // pick n3 (token=40)
            case 2: return 999     // off-tree, bonus
            default: return -1
            }
        }
        let result = tree.verify(oracleAtHistoryEnd: 10, oracle: oracle)
        #expect(result.acceptedTokens == [10, 20, 40])
        #expect(result.bonusToken == 999)
    }

    @Test("verify: oracle picks a sibling at depth 1 → only root accepted")
    func verifySiblingOnlyRoot() {
        let tree = cousinsTree()
        // Target wants 10 (root accepted), then at root wants 30
        // (descend into n2). At n2 (flat 4 in DFS) wants 60 (n5).
        // At n5 (flat 5) wants 999 (off-tree).
        let oracle: (Int) -> Int = { flat in
            switch flat {
            case 0: return 30   // pick n2 over n1
            case 4: return 60   // descend to n5
            case 5: return 999  // off-tree bonus
            default: return -1
            }
        }
        let result = tree.verify(oracleAtHistoryEnd: 10, oracle: oracle)
        #expect(result.acceptedTokens == [10, 30, 60])
        #expect(result.bonusToken == 999)
    }

    @Test("verify: full path accept on a linear chain tree")
    func verifyFullLinearPath() {
        let leaf = DraftTreeNode(token: 4)
        let n3 = DraftTreeNode(token: 3, children: [leaf])
        let n2 = DraftTreeNode(token: 2, children: [n3])
        let root = DraftTreeNode(token: 1, children: [n2])
        // Oracle agrees with every step + adds a bonus past the leaf.
        let oracle: (Int) -> Int = { flat in
            switch flat {
            case 0: return 2
            case 1: return 3
            case 2: return 4
            case 3: return 999  // bonus after the leaf
            default: return -1
            }
        }
        let result = root.verify(oracleAtHistoryEnd: 1, oracle: oracle)
        #expect(result.acceptedTokens == [1, 2, 3, 4])
        #expect(result.bonusToken == 999)
    }

    @Test("flatten + treeCausalMask: branching tree masks cousins/siblings")
    func flattenBranchingTreeMasksCousins() {
        //         0 (root)
        //        / \
        //       1   2
        //      /|   |
        //     3 4   5
        // DFS order: 0, 1, 3, 4, 2, 5
        let n3 = DraftTreeNode(token: 30)
        let n4 = DraftTreeNode(token: 40)
        let n5 = DraftTreeNode(token: 50)
        let n1 = DraftTreeNode(token: 10, children: [n3, n4])
        let n2 = DraftTreeNode(token: 20, children: [n5])
        let root = DraftTreeNode(token: 0, children: [n1, n2])
        let (tokens, parent, _) = root.flatten()
        // DFS order: root, n1, n3, n4, n2, n5
        #expect(tokens == [0, 10, 30, 40, 20, 50])
        #expect(parent == [-1, 0, 1, 1, 0, 4])

        let (mask, t) = root.treeCausalMask()
        #expect(t == 6)
        // Helper: which flat-indices is each node allowed to attend?
        // root (0): only self
        let allow0: Set<Int> = [0]
        // n1 (1): self + root
        let allow1: Set<Int> = [0, 1]
        // n3 (2): self + n1 + root
        let allow2: Set<Int> = [0, 1, 2]
        // n4 (3): self + n1 + root  — NOT n3 (sibling)
        let allow3: Set<Int> = [0, 1, 3]
        // n2 (4): self + root  — NOT n1 / n3 / n4 (sibling-branch + cousins)
        let allow4: Set<Int> = [0, 4]
        // n5 (5): self + n2 + root  — NOT n1 / n3 / n4
        let allow5: Set<Int> = [0, 4, 5]
        let allowed: [Set<Int>] = [allow0, allow1, allow2, allow3, allow4, allow5]

        for i in 0..<t {
            for j in 0..<t {
                let m = mask[i * t + j]
                let expected: Float = allowed[i].contains(j) ? 0.0 : -Float.infinity
                #expect(m == expected)
            }
        }
    }

    @Test("frequency-tie deterministic order")
    func frequencyTieDeterministicOrder() {
        // Bigram "a b" → "c" once, → "d" once (tie). Top-2 returns both;
        // tie-break is by token id ascending, so order is [c, d] when c < d.
        let a = 1, b = 2, c = 3, d = 4
        let history = [a, b, c, 99, a, b, d, 99, a, b]
        let drafter = NGramTreeDrafter(maxNMatch: 2, minNMatch: 2,
                                        branchingFactor: 2)
        guard let tree = drafter.proposeTree(history: history,
                                              maxDepth: 1, maxNodes: 8) else {
            Issue.record("expected non-nil")
            return
        }
        // Both c and d have count=1; deterministic tie-break = ascending id.
        // Root token is the top-1 (= c on ties).
        #expect(tree.token == c)
        #expect(tree.children.map(\.token) == [c, d])
    }
}

@Suite("Drafter — n-gram prompt-lookup")
struct DrafterTests {

    @Test("NGramDrafter proposes the continuation after a repeated trigram")
    func nGramFindsTrigramContinuation() throws {
        // History: "the cat sat ... the cat sat on" — trigram "the cat sat"
        // appeared earlier, then "on" comes after; expect the next call
        // (with key "the cat sat" at end) to propose "on", "the", "mat"
        // depending on gamma.
        // Use integer IDs for clarity.
        let the = 1, cat = 2, sat = 3, on = 4, mat = 5, dog = 6, ran = 7
        // Construct: [the, cat, sat, on, the, mat, dog, ran, the, cat, sat]
        // Latest 3 = "the cat sat"; earliest occurrence at index 0..2;
        // next 3 after that = "on, the, mat" → candidates.
        let history = [the, cat, sat, on, the, mat, dog, ran, the, cat, sat]

        let drafter = NGramDrafter(maxNMatch: 3, minNMatch: 2)
        let proposal = drafter.propose(history: history, gamma: 3)
        #expect(proposal == [on, the, mat],
                "expected [on, the, mat], got \(proposal)")
    }

    @Test("NGramDrafter falls back to shorter match when longest absent")
    func nGramBigramFallback() throws {
        // History has only a bigram match, no trigram.
        let a = 1, b = 2, c = 3, d = 4, e = 5
        // [a, b, c, d, a, b]  — bigram "a, b" repeats; trigram absent.
        let history = [a, b, c, d, a, b]
        let drafter = NGramDrafter(maxNMatch: 3, minNMatch: 2)
        let proposal = drafter.propose(history: history, gamma: 2)
        // After the FIRST "a, b" we have "c, d". So candidates: c, d.
        #expect(proposal == [c, d],
                "expected bigram fallback [c, d], got \(proposal)")
    }

    @Test("NGramDrafter returns empty when no match is found")
    func nGramNoMatchReturnsEmpty() throws {
        let history = [1, 2, 3, 4, 5]
        let drafter = NGramDrafter(maxNMatch: 3, minNMatch: 2)
        let proposal = drafter.propose(history: history, gamma: 3)
        #expect(proposal.isEmpty, "expected empty, got \(proposal)")
    }

    @Test("NeverDrafter always proposes empty")
    func neverDrafterEmpty() throws {
        let drafter = NeverDrafter()
        #expect(drafter.propose(history: [1, 2, 3], gamma: 4).isEmpty)
    }
}
