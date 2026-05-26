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
// Text-side test helpers — coherent-output assertion shared by every
// text / VLM / audio integration suite (any model that produces a
// decoded token stream).
//
// Why a "coherent" check and not a golden-fixture comparison:
//
//   An earlier `expectGoldenMatch` flow compared FFAI's token stream
//   against mlx-lm's stream for the same prompt at temperature=0.
//   That signal degraded as FFAI added kernels and features mlx-lm
//   doesn't have (AURA, GDN, Mamba 2 hybrid layers, sinks-fused SDPA,
//   …) — token-by-token parity vs another implementation became more
//   a measure of how aligned our rounding modes happened to be (RoPE
//   precision, RMSNorm epsilon, SDPA accumulator dtype) than a real
//   correctness signal. Neither implementation has a bug — they made
//   different reasonable choices.
//
//   Per-kernel correctness now comes from metaltile's GPU
//   correctness tests (compared against a naive CPU oracle, much
//   tighter than cross-impl comparison). The job of these
//   integration tests is just to verify that the pipeline — load,
//   prefill, KV cache, sampling — produces coherent text on a real
//   model. That's a meaningful signal that survives diverging from
//   mlx-lm.

import Foundation
import Testing

/// Assert that a stream of decoded token IDs looks like coherent
/// model output rather than a degenerate failure mode. Catches:
///
/// - **Truncation**: model exited the decode loop before producing
///   `minTokens` tokens (almost always means EOS was hit immediately
///   or the cache/sampler crashed silently). Treated as a failure
///   because real prompts should yield ≥ 50 tokens at temp=0.
/// - **Stuck output**: a run of `maxConsecutiveRepeat + 1` identical
///   tokens. The textbook signature of "empty kernel" or "stuck
///   argmax" regressions — the model emits one token forever.
/// - **Degenerate distribution**: unique-token ratio below
///   `minUniqueRatio`. Catches the case where the model alternates
///   between a tiny handful of tokens (a degenerate cycle).
///
/// All thresholds are deliberately loose — the goal is to catch
/// catastrophic regressions, not to police generation quality. A
/// model emitting an unusual but valid pattern (e.g. lots of
/// punctuation) should still pass.
///
/// `label` is just for diagnostic messages; pass the model name.
public func expectCoherentOutput(
    _ tokens: [Int],
    minTokens: Int = 50,
    maxConsecutiveRepeat: Int = 5,
    minUniqueRatio: Double = 0.2,
    label: String = "",
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let tag = label.isEmpty ? "" : "[\(label)] "

    // Token-count floor — catches early-exit / silent crash regressions.
    guard tokens.count >= minTokens else {
        let msg = "\(tag)generated only \(tokens.count) tokens, expected ≥ \(minTokens). " +
            "Forward pass likely exited early (EOS at first step, sampler crash, " +
            "or cache desync). Tokens: \(tokens)"
        Issue.record(Comment(rawValue: msg), sourceLocation: sourceLocation)
        return
    }

    // No run of identical tokens longer than `maxConsecutiveRepeat`.
    var run = 1
    var prev = tokens[0]
    for (i, tok) in tokens.enumerated().dropFirst() {
        run = (tok == prev) ? run + 1 : 1
        prev = tok
        if run > maxConsecutiveRepeat {
            let context = max(0, i - run + 1)..<min(tokens.count, i + 5)
            let msg = "\(tag)degenerate output: \(run) consecutive copies of token \(tok) " +
                "starting near index \(i - run + 1). Context: \(Array(tokens[context]))…"
            Issue.record(Comment(rawValue: msg), sourceLocation: sourceLocation)
            return
        }
    }

    // Token-diversity floor — catches alternating-cycle degenerate output.
    let unique = Set(tokens).count
    let ratio = Double(unique) / Double(tokens.count)
    guard ratio >= minUniqueRatio else {
        let pct = String(format: "%.0f%%", ratio * 100)
        let msg = "\(tag)low token diversity: \(unique) unique tokens of \(tokens.count) " +
            "(\(pct)), expected ratio ≥ \(minUniqueRatio). Likely degenerate output. " +
            "First 16: \(Array(tokens.prefix(16)))"
        Issue.record(Comment(rawValue: msg), sourceLocation: sourceLocation)
        return
    }

    // Diagnostic — always print so reviewers can see what the model
    // produced (and spot quality regressions even when the floor passes).
    let pct = String(format: "%.0f%%", ratio * 100)
    print("COHERENT \(tag)tokens=\(tokens.count) unique=\(unique) (\(pct))")
}

/// Assert a decoded text contains every substring in `phrases` (case-
/// and punctuation-insensitive). Useful for "the output should mention
/// dog and golden" style assertions where exact wording varies by model.
public func expectTextContains(
    _ text: String,
    _ phrases: [String],
    label: String,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let normalized = normalizeForMatch(text)
    for phrase in phrases {
        let needle = normalizeForMatch(phrase)
        let comment = Comment(
            rawValue: "\(label): expected to find \"\(phrase)\" in output: \(text)"
        )
        #expect(
            normalized.contains(needle),
            comment,
            sourceLocation: sourceLocation
        )
    }
}

/// Normalise a string for loose phrase matching: lowercase, strip
/// punctuation, collapse whitespace. Used by `expectTextContains` and
/// the STT phrase-match assertion in AudioTestHelpers.
public func normalizeForMatch(_ text: String) -> String {
    let lowered = text.lowercased()
    let punct: Set<Character> = [".", ",", "!", "?", ";", ":", "\"", "'", "`",
                                 "—", "–", "(", ")", "[", "]", "{", "}"]
    let stripped = lowered.filter { !punct.contains($0) }
    // Collapse all whitespace runs to a single space.
    let parts = stripped.split(whereSeparator: { $0.isWhitespace })
    return parts.joined(separator: " ")
}
