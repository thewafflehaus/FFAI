// Copyright 2026 Tom Turney (@TheTom)
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
// SpecDecode — greedy speculative-decode driver for any model that
// conforms to `SpeculativeVerifier` (Qwen3.5 / 3.6 today; the driver is
// no longer family-specific).
//
// One iteration at γ candidates:
//   1. Drafter proposes up to γ candidate next tokens c_0..c_{γ-1}.
//   2. If γ' = drafter's actual proposal length is 0, fall back to a
//      regular single-token forward.
//   3. Otherwise:
//        a. Snapshot every layer cache.
//        b. Run `forwardManyAllLogits([lastAccepted, c_0..c_{γ'-1}],
//           startPos=pos)` — this advances every cache by γ' + 1 tokens
//           and returns logits at each of the γ' + 1 input positions.
//        c. Greedy verify: for i in 0..γ'-1, check whether
//           `argmax(logits[i]) == c_i`. Accept consecutive matches up
//           to the first mismatch.
//        d. If first mismatch at index k:
//             * Restore caches from snapshot.
//             * Re-run single-step `forward(...)` over `[prev, c_0..c_{k-1}]`
//               to advance caches by exactly k+1 tokens (with the
//               bit-identical-to-baseline path).
//             * Take the next-prev token from the LAST single-step
//               forward — `forwardManyAllLogits` can drift slightly
//               (different kernel paths / accumulation order) vs the
//               single-step decode, so the sampling source for `prev`
//               must be the single-step forward output to keep the
//               greedy stream bit-identical to the baseline loop.
//             * tokens committed this iter: k + 1.
//           If no mismatch (all γ' accepted):
//             * Commit [c_0..c_{γ'-1}] + the bonus token from the
//               trailing logits[γ'] (the model's prediction for what
//               comes AFTER the last candidate).
//             * tokens committed: γ' + 1.
//             * Cache state is already correct — discard snapshot.
//
// Greedy-only for v0. Temperature sampling / nucleus / top-K extensions
// follow the same shape but compare sampled tokens against the
// drafter's proposed tokens via probability-ratio rejection (see the
// original spec-decode paper).

import Foundation
import Metal

public struct SpecDecodeStats {
    public let tokensGenerated: Int
    public let stepsRun: Int
    public let candidatesProposed: Int
    public let candidatesAccepted: Int
    public let fallbackSingleSteps: Int
    public let wallclockSeconds: Double

    public var acceptanceRate: Double {
        guard candidatesProposed > 0 else { return 0 }
        return Double(candidatesAccepted) / Double(candidatesProposed)
    }
    public var tps: Double {
        guard wallclockSeconds > 0 else { return 0 }
        return Double(tokensGenerated) / wallclockSeconds
    }
}

/// A model that can drive speculative decoding. Two forwards are needed:
/// the single-token decode used for the bit-identical baseline replay,
/// and a per-position "verify" forward over a batch of candidate tokens.
/// Any family implementing both can use `SpecDecode.generateGreedy` — it
/// is no longer Qwen-specific. Qwen3.5 / 3.6 conform today; MTP / EAGLE /
/// other-family drivers conform as they land.
public protocol SpeculativeVerifier: AnyObject {
    /// Single-token decode. Advances `caches` by one token and returns
    /// the next-token logits `[vocab]`. MUST be the bit-identical
    /// baseline decode path — the accept loop replays through it so the
    /// greedy stream matches a non-speculative run exactly.
    func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor

    /// Verify forward over `tokenIds` starting at `startPosition`.
    /// Advances `caches` by `tokenIds.count` tokens and returns logits at
    /// EACH input position — shape `[tokenIds.count, vocab]`.
    func forwardManyAllLogits(
        tokenIds: [Int], startPosition: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor
}

public enum SpecDecode {
    /// Run a greedy speculative-decode loop. The caller is responsible
    /// for prefill — `caches` must already reflect `prompt`, `lastToken`
    /// is the most recent (already-sampled) token, `position` is the
    /// position where `lastToken` would be inserted on the next decode
    /// step (i.e. promptTokens.count if you just finished prefill).
    ///
    /// Greedy-only — `argmax` everywhere. Stops when `maxNewTokens` is
    /// reached or a token in `stopTokens` is emitted.
    public static func generateGreedy(
        model: any SpeculativeVerifier,
        drafter: Drafter,
        gamma: Int,
        lastToken: Int,
        position: Int,
        caches: [any LayerCacheProtocol],
        history: inout [Int],
        maxNewTokens: Int,
        stopTokens: Set<Int> = [],
        device: Device = .shared
    ) -> SpecDecodeStats {
        precondition(gamma >= 1, "SpecDecode.generateGreedy: gamma must be ≥ 1")
        precondition(
            maxNewTokens >= 0,
            "SpecDecode.generateGreedy: maxNewTokens must be ≥ 0")

        var pos = position
        var prev = lastToken
        var tokensGenerated = 0
        var stepsRun = 0
        var candidatesProposed = 0
        var candidatesAccepted = 0
        var fallbackSingleSteps = 0
        let t0 = Date()

        loop: while tokensGenerated < maxNewTokens {
            // ── Draft proposal ────────────────────────────────────────
            let candidates = drafter.propose(history: history + [prev], gamma: gamma)
            if candidates.isEmpty {
                // No proposal → fall back to single-token decode.
                let cmd = device.makeCommandBuffer()
                let logits = model.forward(
                    tokenId: prev, position: pos,
                    caches: caches, on: cmd, device: device)
                cmd.commit()
                cmd.waitUntilCompleted()
                let next = argmax(logits)
                history.append(prev)
                pos += 1
                prev = next
                tokensGenerated += 1
                stepsRun += 1
                fallbackSingleSteps += 1
                if stopTokens.contains(next) { break loop }
                continue
            }
            let gp = candidates.count
            candidatesProposed += gp

            // ── Snapshot caches before the speculative forward ────────
            let snap = caches.snapshotAll(device: device)

            // ── Verify: run forwardManyAllLogits([prev, c_0..c_{gp-1}]) ─
            let inputIds = [prev] + candidates
            let cmd = device.makeCommandBuffer()
            let allLogits = model.forwardManyAllLogits(
                tokenIds: inputIds, startPosition: pos,
                caches: caches, on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            stepsRun += 1

            // allLogits shape [gp + 1, vocab]. Read host-side argmax per row.
            let flat = allLogits.toFloatArray()
            let vocab = flat.count / (gp + 1)
            precondition(
                flat.count == (gp + 1) * vocab,
                "SpecDecode: allLogits has \(flat.count) elements; expected (gp+1)*vocab = \((gp + 1) * vocab)"
            )

            // ── Greedy accept loop ────────────────────────────────────
            var acceptedCount = 0
            var firstMismatchTok: Int? = nil
            for i in 0 ..< gp {
                let rowStart = i * vocab
                let modelChoice = argmax(flat, offset: rowStart, count: vocab)
                if modelChoice == candidates[i] {
                    acceptedCount += 1
                } else {
                    firstMismatchTok = modelChoice
                    break
                }
            }

            if firstMismatchTok != nil {
                // ── Partial accept (could be 0): restore + replay ────
                // We DO NOT trust the batched verify's logits for
                // sampling the next-prev — forwardManyAllLogits can
                // drift slightly (different kernel paths, accumulation
                // order) vs single forward(). After replay, take the
                // next-prev from the LAST single-step forward — that is
                // bit-identical to what the baseline greedy loop
                // produces.
                caches.restoreAll(from: snap, device: device)
                let toReplay = [prev] + Array(candidates.prefix(acceptedCount))
                var lastLogits: Tensor!
                for (i, tok) in toReplay.enumerated() {
                    let stepCmd = device.makeCommandBuffer()
                    lastLogits = model.forward(
                        tokenId: tok, position: pos + i,
                        caches: caches, on: stepCmd, device: device)
                    stepCmd.commit()
                    stepCmd.waitUntilCompleted()
                }
                let nextProvenTok = argmax(lastLogits)
                history.append(prev)
                for c in candidates.prefix(acceptedCount) { history.append(c) }
                let totalCommitted = 1 + acceptedCount
                pos += totalCommitted
                prev = nextProvenTok
                tokensGenerated += totalCommitted
                candidatesAccepted += acceptedCount
                if stopTokens.contains(nextProvenTok) { break loop }
            } else {
                // ── Full accept: commit all γ' candidates + bonus ────
                // Bonus token = argmax(logits[gp]) — model's prediction
                // at position pos + gp + 1.
                let bonusStart = gp * vocab
                let bonusTok = argmax(flat, offset: bonusStart, count: vocab)
                history.append(prev)
                for c in candidates { history.append(c) }
                let totalCommitted = 1 + gp
                pos += totalCommitted
                prev = bonusTok
                tokensGenerated += totalCommitted
                candidatesAccepted += gp
                if stopTokens.contains(bonusTok) { break loop }
            }
        }
        // `prev` is the next-iteration's input (not yet a generated
        // token — its KV slot isn't in the cache). Caller continues
        // from here by calling again with this `prev`, OR can append it
        // themselves if they're done generating. We do NOT append it
        // here — `tokensGenerated` and `history` reflect ONLY tokens
        // committed to the cache, matching the baseline greedy loop's
        // semantics.

        return SpecDecodeStats(
            tokensGenerated: tokensGenerated,
            stepsRun: stepsRun,
            candidatesProposed: candidatesProposed,
            candidatesAccepted: candidatesAccepted,
            fallbackSingleSteps: fallbackSingleSteps,
            wallclockSeconds: Date().timeIntervalSince(t0))
    }
}

// ─── argmax helpers ──────────────────────────────────────────────────

@inline(__always)
private func argmax(_ logits: Tensor) -> Int {
    let host = logits.toFloatArray()
    return argmax(host, offset: 0, count: host.count)
}

@inline(__always)
private func argmax(_ flat: [Float], offset: Int, count: Int) -> Int {
    precondition(count > 0)
    var bestIdx = 0
    var bestVal = flat[offset]
    for i in 1 ..< count {
        let v = flat[offset + i]
        if v > bestVal {
            bestVal = v
            bestIdx = i
        }
    }
    return bestIdx
}
