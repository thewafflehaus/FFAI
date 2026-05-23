// Host-side unit test for n-gram drafter — no GPU needed.

import Foundation
import Testing
@testable import FFAI

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
