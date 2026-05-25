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
