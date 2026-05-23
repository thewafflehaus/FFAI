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
