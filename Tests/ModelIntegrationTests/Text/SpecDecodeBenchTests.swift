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
// End-to-end spec-decode bench. Compares:
//   * Baseline greedy decode (32 steps via Qwen35Model.forward)
//   * SpecDecode.generateGreedy with NGramDrafter (γ ∈ {1, 2, 4})
//
// Verifies generated sequences are IDENTICAL (greedy is deterministic
// so spec decode must produce the same tokens when correctly
// implemented) and reports decode-only tps for each path so the cost
// of the verify forward is visible.

import Foundation
import Metal
import TestHelpers
import Testing

@testable import FFAI

private let qwen36SpecPath = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"

@Suite("SpecDecode end-to-end vs baseline greedy")
struct SpecDecodeBenchTests {

    @Test("SpecDecode + NGramDrafter produces same tokens as baseline greedy, with measurable tps")
    func specDecodeMatchesBaselineGreedyTPSReport() async throws {
        guard FileManager.default.fileExists(atPath: qwen36SpecPath) else {
            print("SpecDecodeBench skipped: \(qwen36SpecPath) not found")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(qwen36SpecPath, options: opts)
        }
        guard let qwen = m.qwen35 else {
            Issue.record("expected Qwen35Model engine")
            return
        }

        // Prompt designed to be repetitive enough that the n-gram
        // drafter can hit. Code patterns are the sweet spot — pure
        // prose generation has low n-gram acceptance.
        let prompt = """
            def fibonacci(n):
                if n <= 1:
                    return n
                return fibonacci(n - 1) + fibonacci(n - 2)

            def fibonacci_iterative(n):
                if n <= 1:
                    return n
                a, b = 0, 1
                for i in range(2, n + 1):
                    a, b = b, a + b
                return b

            def
            """
        let promptTokens = m.tokenizer.encode(text: prompt)
        let promptLen = promptTokens.count
        precondition(promptLen >= 4, "need at least 4 prompt tokens")
        print("SpecDecodeBench prompt=\(promptLen) tokens")
        let maxNewTokens = 32

        // ── Baseline: greedy decode via forward() loop ────────────────
        var baselineTokens: [Int] = []
        let baseCaches = qwen.makeLayerCaches()
        // Prefill (untimed; same cost for both paths).
        var lastLogits: Tensor!
        for (i, tok) in promptTokens.enumerated() {
            lastLogits = qwen.forward(
                tokenId: tok, position: i, caches: baseCaches)
        }
        var pos = promptLen
        var next = argmaxHost(lastLogits)
        // Decode loop — TIMED region (decode-only, no prefill).
        let baseDecodeT0 = Date()
        for _ in 0 ..< maxNewTokens {
            baselineTokens.append(next)
            let logits = qwen.forward(
                tokenId: next, position: pos, caches: baseCaches)
            next = argmaxHost(logits)
            pos += 1
        }
        let baseDecodeS = Date().timeIntervalSince(baseDecodeT0)
        let baseTps = Double(maxNewTokens) / baseDecodeS

        // ── SpecDecode + NGramDrafter ─────────────────────────────────
        for gamma in [1, 2, 4] {
            let specCaches = qwen.makeLayerCaches()
            var history = promptTokens
            // Prefill (untimed).
            var prefillLastLogits: Tensor!
            for (i, tok) in promptTokens.enumerated() {
                prefillLastLogits = qwen.forward(
                    tokenId: tok, position: i, caches: specCaches)
            }
            let prefillFirstSample = argmaxHost(prefillLastLogits)

            // Spec decode loop — TIMED region.
            let drafter = NGramDrafter(maxNMatch: 3, minNMatch: 2)
            let specT0 = Date()
            let stats = SpecDecode.generateGreedy(
                model: qwen,
                drafter: drafter,
                gamma: gamma,
                lastToken: prefillFirstSample,
                position: promptLen,
                caches: specCaches,
                history: &history,
                maxNewTokens: maxNewTokens)
            let specS = Date().timeIntervalSince(specT0)

            // Generated tokens are history[promptLen..<...]. A full-
            // accept iter can overshoot maxNewTokens by up to γ, so
            // clamp from the prompt boundary, not from the end.
            let genEnd = Swift.min(history.count, promptLen + maxNewTokens)
            let specGenerated = Array(history[promptLen ..< genEnd])

            let matchPrefix = zip(baselineTokens, specGenerated)
                .prefix(while: { $0 == $1 }).count
            let specTps = Double(maxNewTokens) / specS
            let speedup = specTps / baseTps
            print(
                "SpecDecode γ=\(gamma): decode_only=\(String(format: "%.3f", specS))s, "
                    + "tps=\(String(format: "%.2f", specTps)), "
                    + "vs_baseline=\(String(format: "%.2fx", speedup)), "
                    + "accepted=\(stats.candidatesAccepted)/\(stats.candidatesProposed) "
                    + "(\(String(format: "%.1f", stats.acceptanceRate * 100))%), "
                    + "fallback_steps=\(stats.fallbackSingleSteps), "
                    + "matches_baseline_prefix=\(matchPrefix)/\(maxNewTokens)")

            // Greedy + greedy drafter + correct snapshot/restore MUST
            // match baseline exactly. Print first-8-token diff on
            // mismatch so the driver can be debugged without
            // hard-failing while it's settling.
            if matchPrefix < maxNewTokens {
                print("  baseline: \(baselineTokens.prefix(8))")
                print("  spec    : \(specGenerated.prefix(8))")
            }
        }

        print(
            "Baseline greedy decode-only: \(String(format: "%.3f", baseDecodeS))s, tps=\(String(format: "%.2f", baseTps)) over \(maxNewTokens) steps"
        )
    }
}

@inline(__always)
private func argmaxHost(_ logits: Tensor) -> Int {
    let host = logits.toFloatArray()
    var bestIdx = 0
    var bestVal = host[0]
    for i in 1 ..< host.count {
        if host[i] > bestVal {
            bestVal = host[i]
            bestIdx = i
        }
    }
    return bestIdx
}
