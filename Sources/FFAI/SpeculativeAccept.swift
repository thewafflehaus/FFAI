// SpeculativeAccept — greedy longest-matching-prefix acceptance for
// speculative decoding.
//
// A draft block of speculative tokens is verified against a verifier's
// own tokens. The longest matching prefix is accepted; the verifier's
// token at the first mismatch (or one token past a fully-accepted
// draft) is committed as a bonus, so each verify step commits between
// 1 and draft.count + 1 tokens.
//
// First consumer is Nemotron-Labs-Diffusion self-speculation (diffusion
// draft + AR verify). The helper is deliberately agnostic to how the
// draft and verifier tokens are produced so the planned n-gram (spec
// 013) and MTP/EAGLE (spec 030) speculative paths can reuse it. The
// planned non-greedy Leviathan accept/reject sampler (spec 023) slots
// in later as a sibling entry point.

import Foundation

/// Greedy longest-matching-prefix acceptance for speculative decoding.
public enum SpeculativeAccept {

    /// Outcome of verifying one draft block.
    public struct Outcome: Equatable, Sendable {
        /// Draft tokens accepted — the longest prefix that matched the
        /// verifier, in order.
        public let acceptedDraft: [Int]
        /// One bonus token from the verifier: its token at the first
        /// rejected position, or — when the whole draft was accepted —
        /// the caller-supplied `bonusToken` that follows the draft.
        public let bonusToken: Int

        public init(acceptedDraft: [Int], bonusToken: Int) {
            self.acceptedDraft = acceptedDraft
            self.bonusToken = bonusToken
        }

        /// Tokens committed this step = `acceptedDraft.count + 1`.
        public var committedCount: Int { acceptedDraft.count + 1 }
        /// All tokens to commit: the accepted draft followed by the
        /// bonus token.
        public var committedTokens: [Int] { acceptedDraft + [bonusToken] }
    }

    /// Verify `draft` against `verifierTokens`, greedily.
    ///
    /// - `draft[j]` — the speculatively-drafted token for position `j`.
    /// - `verifierTokens[j]` — the verifier's own token for the same
    ///   position (e.g. the AR-argmax of the verifier's logits there).
    ///   The caller is responsible for aligning the two arrays.
    /// - `bonusToken` — the verifier's token for the position
    ///   immediately *after* the draft. Used only when every draft
    ///   token is accepted, so an all-accept block still yields the
    ///   standard +1 token.
    ///
    /// Accepts the longest prefix where `draft[j] == verifierTokens[j]`.
    public static func verify(draft: [Int],
                              verifierTokens: [Int],
                              bonusToken: Int) -> Outcome {
        precondition(draft.count == verifierTokens.count,
                     "SpeculativeAccept.verify: draft (\(draft.count)) and "
                     + "verifierTokens (\(verifierTokens.count)) must be equal length")
        var accepted = 0
        while accepted < draft.count && draft[accepted] == verifierTokens[accepted] {
            accepted += 1
        }
        let bonus = accepted < draft.count ? verifierTokens[accepted] : bonusToken
        return Outcome(acceptedDraft: Array(draft.prefix(accepted)), bonusToken: bonus)
    }
}
