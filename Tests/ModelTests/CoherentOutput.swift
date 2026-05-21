// Shared helper for ModelTests/*IntegrationTests.swift — asserts that
// FFAI's greedy decode produced *coherent* output, independent of any
// other implementation.
//
// Why not compare to mlx-lm / mlx-vlm goldens any more:
//
//   The earlier `GoldenFixture` + `expectGoldenMatch` flow compared
//   FFAI's token stream against mlx-lm's stream for the same prompt
//   at temperature=0. That signal degraded as FFAI added kernels and
//   features mlx-lm doesn't have (AURA, GDN, Mamba 2 hybrid layers,
//   sinks-fused SDPA, …) — token-by-token parity vs another
//   implementation became more a measure of how aligned our rounding
//   modes happened to be with theirs (RoPE precision, RMSNorm epsilon,
//   SDPA accumulator dtype) than a real correctness signal. The
//   4-bit Qwen3 case hit 15/32 token match against mlx-lm; Llama 3.2
//   1B fp16 hit 1/32. Neither indicates a bug in FFAI — they indicate
//   that two reasonable implementations made different choices.
//
//   Per-kernel correctness now comes from the metaltile-side GPU
//   correctness tests under `crates/metaltile-std/tests/` (compared
//   against a naive CPU oracle, much tighter than cross-impl
//   comparison). The job of these integration tests is just to
//   verify that the model pipeline — load, prefill, KV cache,
//   sampling — produces coherent text on a real model. That's a
//   meaningful signal that survives diverging from mlx-lm.

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
func expectCoherentOutput(
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
    // Empty kernel / NaN logits / stuck argmax usually manifests as
    // the same token repeated forever.
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

    // Token-diversity floor — catches alternating-cycle degenerate output
    // (the empty-kernel-but-not-stuck-on-one-token failure mode).
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
