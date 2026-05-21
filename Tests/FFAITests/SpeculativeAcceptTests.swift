// SpeculativeAcceptTests — pure-CPU coverage of the greedy
// longest-matching-prefix acceptance helper used by speculative
// decoding (Nemotron-Labs-Diffusion self-speculation first).

import Testing
@testable import FFAI

@Suite("SpeculativeAccept — greedy prefix acceptance")
struct SpeculativeAcceptTests {

    @Test("partial accept stops at the first mismatch and takes the verifier token")
    func partialAccept() {
        let outcome = SpeculativeAccept.verify(
            draft:          [10, 11, 12, 13],
            verifierTokens: [10, 11, 99, 14],
            bonusToken: 7)
        #expect(outcome.acceptedDraft == [10, 11])
        #expect(outcome.bonusToken == 99)            // verifier token at first mismatch
        #expect(outcome.committedCount == 3)
        #expect(outcome.committedTokens == [10, 11, 99])
    }

    @Test("full accept uses the caller-supplied bonus token")
    func fullAccept() {
        let outcome = SpeculativeAccept.verify(
            draft:          [1, 2, 3],
            verifierTokens: [1, 2, 3],
            bonusToken: 42)
        #expect(outcome.acceptedDraft == [1, 2, 3])
        #expect(outcome.bonusToken == 42)
        #expect(outcome.committedCount == 4)
        #expect(outcome.committedTokens == [1, 2, 3, 42])
    }

    @Test("immediate mismatch accepts nothing but still commits one verifier token")
    func zeroAccept() {
        let outcome = SpeculativeAccept.verify(
            draft:          [5, 6, 7],
            verifierTokens: [8, 6, 7],
            bonusToken: 0)
        #expect(outcome.acceptedDraft == [])
        #expect(outcome.bonusToken == 8)
        #expect(outcome.committedCount == 1)
        #expect(outcome.committedTokens == [8])
    }

    @Test("empty draft commits exactly the bonus token")
    func emptyDraft() {
        let outcome = SpeculativeAccept.verify(
            draft: [], verifierTokens: [], bonusToken: 13)
        #expect(outcome.acceptedDraft == [])
        #expect(outcome.bonusToken == 13)
        #expect(outcome.committedTokens == [13])
    }
}
