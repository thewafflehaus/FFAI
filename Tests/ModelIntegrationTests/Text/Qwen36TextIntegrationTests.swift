// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// Integration tests covering the Qwen3.6-35B-A3B local checkpoint —
// end-to-end load, prefill/decode bench, batched forwardMany
// equivalence, and first-token greedy decode.
//
// The whole suite is gated on the local-checkpoint path existing —
// it ships with FFAI as a reference for how to load Qwen3.6-A3B from
// a local snapshot. When the path doesn't resolve, the suite is
// disabled (visibly, by Swift Testing) rather than silently returning
// from each test.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

private let qwen36LocalPath = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
private let qwen36CheckpointAvailable =
    FileManager.default.fileExists(atPath: qwen36LocalPath)

@Suite(
    "Qwen3.6 Text Integration", .serialized,
    .enabled(
        if: qwen36CheckpointAvailable && IntegrationGroupGating.enableTextSuites,
        "Qwen3.6 integration requires a local checkpoint at \(qwen36LocalPath) "
            + "AND IntegrationGroupGating.enableTextSuites = true")
)
struct Qwen36TextIntegrationTests {

    @Test("Qwen3.6-35B-A3B local checkpoint loads")
    func loadLocal() async throws {
        let path = qwen36LocalPath
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false  // skip prewarm to isolate load failures
        let opts = optsBuilder
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        print("LOAD OK")
        #expect(m.qwen35 != nil, "expected Qwen35Model engine")
        if let q = m.qwen35 {
            print(
                "hidden=\(q.hidden) layers=\(q.nLayers) heads=\(q.nHeads) kv=\(q.nKVHeads) headDim=\(q.headDim)"
            )
            print(
                "GDN dims: Hk=\(q.numKeyHeads) Hv=\(q.numValueHeads) Dk=\(q.keyHeadDim) Dv=\(q.valueHeadDim)"
            )
            print("hasMoE=\(q.hasMoE) vocab=\(q.vocab) dtype=\(q.dtype)")
            let gdnCount = q.layers.filter { $0 is Qwen35GDNLayer }.count
            let attnCount = q.layers.filter { $0 is Qwen35AttentionLayer }.count
            print("layers: gdn=\(gdnCount) attn=\(attnCount)")
            #expect(q.nLayers == 40)
            #expect(gdnCount == 30)
            #expect(attnCount == 10)
        }
    }

    @Test("Qwen3.6-35B-A3B bench — short prefill + decode steady-state")
    func benchShort() async throws {
        // FFAI's Qwen35 model does single-token forward steps for both
        // prefill and decode (no batched prefill on this branch), so a
        // 4K/32K prefill takes prohibitively long. A 128-token prefill +
        // 64 decode steps is enough to see steady-state decode tps
        // (the cold first 1-2 tokens absorb PSO JIT) and short-ctx
        // prefill cost. Use 32-token prompt + 32 steady decode.
        try await runBench(targetPromptTokens: 32, decodeSteps: 32, label: "T=32")
    }

    @Test("Qwen3.6-35B-A3B bench — T=4K (slow, ~10min on M5 Max)")
    func bench4k() async throws {
        try await runBench(targetPromptTokens: 4096, decodeSteps: 16, label: "T=4K")
    }

    @Test("Qwen3.6-35B-A3B bench — T=32K (very slow, ~85min on M5 Max)")
    func bench32k() async throws {
        try await runBench(targetPromptTokens: 32_768, decodeSteps: 16, label: "T=32K")
    }

    /// Generate a deterministic prompt to a target token count by
    /// repeating a base sentence. Returns (promptTokens, prefillSecs,
    /// decodeSecs, decodeTokens).
    private func runBench(targetPromptTokens: Int, decodeSteps: Int, label: String) async throws {
        let path = qwen36LocalPath
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        // Build a prompt of approximately the target length by repeating
        // a paragraph until we hit it.
        let base =
            "The quick brown fox jumps over the lazy dog. "
            + "Pack my box with five dozen liquor jugs. "
        var text = ""
        var tokens = m.tokenizer.encode(text: base)
        while tokens.count < targetPromptTokens {
            text += base
            tokens = m.tokenizer.encode(text: text)
        }
        if tokens.count > targetPromptTokens {
            tokens = Array(tokens.prefix(targetPromptTokens))
        }
        print("\(label): prompt=\(tokens.count) tokens")

        let caches = m.engine.makeLayerCaches()

        // ── Prefill ──────────────────────────────────────────────────
        let prefillStart = Date()
        var lastLogits: Tensor!
        for (i, tok) in tokens.enumerated() {
            lastLogits = m.engine.forward(tokenId: tok, position: i, caches: caches)
        }
        let prefillSecs = Date().timeIntervalSince(prefillStart)
        let prefillMs = prefillSecs * 1000
        let prefillTps = Double(tokens.count) / prefillSecs
        print(
            "\(label): prefill \(String(format: "%.0f", prefillMs))ms (\(String(format: "%.1f", prefillTps)) tok/s)"
        )

        // Greedy sample first generated token.
        var logits = lastLogits.toFloatArray()
        var nextTok = logits.enumerated().max(by: { $0.element < $1.element })!.offset

        // ── Decode loop ──────────────────────────────────────────────
        var stepTimes: [Double] = []
        var pos = tokens.count
        for _ in 0 ..< decodeSteps {
            let t0 = Date()
            lastLogits = m.engine.forward(tokenId: nextTok, position: pos, caches: caches)
            logits = lastLogits.toFloatArray()
            nextTok = logits.enumerated().max(by: { $0.element < $1.element })!.offset
            pos += 1
            stepTimes.append(Date().timeIntervalSince(t0))
        }
        let decodeSecs = stepTimes.reduce(0, +)
        let decodeTps = Double(decodeSteps) / decodeSecs
        // Steady-state: skip first 4 steps (PSO JIT) if we have enough samples.
        let steadyCutoff = min(4, max(0, decodeSteps - 4))
        let steadySteps = stepTimes.dropFirst(steadyCutoff)
        let steadySecs = steadySteps.reduce(0, +)
        let steadyTps = Double(steadySteps.count) / max(steadySecs, 1e-9)
        print(
            "\(label): decode \(decodeSteps) steps in \(String(format: "%.2f", decodeSecs))s (avg \(String(format: "%.2f", decodeTps)) tok/s, steady \(String(format: "%.2f", steadyTps)) tok/s)"
        )
        print(
            "\(label): per-step ms (first 8): \(stepTimes.prefix(8).map { String(format: "%.0f", $0 * 1000) })"
        )
        print(
            "\(label): RESULT prefill_ms=\(String(format: "%.0f", prefillMs)) decode_tps=\(String(format: "%.2f", decodeTps)) steady_tps=\(String(format: "%.2f", steadyTps)) prefill_tps=\(String(format: "%.1f", prefillTps))"
        )
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=32 prefill, batched vs per-token")
    func forwardManyBench() async throws {
        try await runForwardManyBench(targetT: 32)
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=128 prefill scaling")
    func forwardManyBench128() async throws {
        try await runForwardManyBench(targetT: 128)
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=512 long-context scaling")
    func forwardManyBench512() async throws {
        try await runForwardManyBench(targetT: 512)
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=2048 long-context scaling")
    func forwardManyBench2K() async throws {
        try await runForwardManyBench(targetT: 2048)
    }

    private func runForwardManyBench(targetT: Int) async throws {
        let path = qwen36LocalPath
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        let qwen = try #require(m.qwen35, "expected Qwen35Model engine")
        // Seed prompt — 32 tokens. Tile until we hit targetT, then trim.
        let seed =
            "The history of the printing press began when European craftsmen of the 15th century combined movable metal type with oil based ink screw presses paper to mass produce printed books pamphlets and broadsheets revolutionising communication"
        let seedEncoded = m.tokenizer.encode(text: seed)
        var encoded = seedEncoded
        while encoded.count < targetT {
            encoded.append(contentsOf: seedEncoded)
        }
        encoded = Array(encoded.prefix(targetT))
        let T = encoded.count
        print("forwardManyBench T=\(T)")

        // Warm up Metal PSO + first-token JIT for **both** paths.
        // mlx-lm bench convention is `model(batch[:1]); mx.eval()` once;
        // here we do 2 iters of each path so the second iter benches at
        // steady-state-ish (PSOs compiled, page caches warm).
        for warmIter in 0 ..< 2 {
            let warmCachesP = qwen.makeLayerCaches()
            for (i, tok) in encoded.prefix(2).enumerated() {
                _ = qwen.forward(tokenId: tok, position: i, caches: warmCachesP)
            }
            let warmCachesB = qwen.makeLayerCaches()
            let warmCmd = Device.shared.makeCommandBuffer()
            _ = qwen.forwardMany(
                tokenIds: encoded, startPosition: 0,
                caches: warmCachesB, on: warmCmd, device: Device.shared)
            warmCmd.commit()
            await warmCmd.awaitCompletion()
            _ = warmIter  // silence
        }

        // Per-token loop baseline (5 runs, median).
        var perTokenSecs: [Double] = []
        for _ in 0 ..< 5 {
            let caches = qwen.makeLayerCaches()
            let t0 = Date()
            for (i, tok) in encoded.enumerated() {
                _ = qwen.forward(tokenId: tok, position: i, caches: caches)
            }
            perTokenSecs.append(Date().timeIntervalSince(t0))
        }
        perTokenSecs.sort()
        let perTokenMedian = perTokenSecs[perTokenSecs.count / 2]
        print(
            "per-token T=\(T): runs=\(perTokenSecs.map { String(format: "%.3f", $0) }) median=\(String(format: "%.3f", perTokenMedian))s = \(String(format: "%.2f", Double(T)/perTokenMedian)) tps"
        )

        // Batched forwardMany (5 runs, median).
        var batchedSecs: [Double] = []
        for _ in 0 ..< 5 {
            let caches = qwen.makeLayerCaches()
            let bCmd = Device.shared.makeCommandBuffer()
            let t0 = Date()
            _ = qwen.forwardMany(
                tokenIds: encoded, startPosition: 0,
                caches: caches, on: bCmd, device: Device.shared)
            bCmd.commit()
            await bCmd.awaitCompletion()
            batchedSecs.append(Date().timeIntervalSince(t0))
        }
        batchedSecs.sort()
        let batchedMedian = batchedSecs[batchedSecs.count / 2]
        print(
            "batched T=\(T): runs=\(batchedSecs.map { String(format: "%.3f", $0) }) median=\(String(format: "%.3f", batchedMedian))s = \(String(format: "%.2f", Double(T)/batchedMedian)) tps"
        )

        let speedup = perTokenMedian / batchedMedian
        print(
            "forwardManyBench RESULT T=\(T): per_token=\(String(format: "%.0f", perTokenMedian*1000))ms batched=\(String(format: "%.0f", batchedMedian*1000))ms speedup=\(String(format: "%.2fx", speedup))"
        )
    }

    @Test("Qwen3.6-35B-A3B forwardMany matches per-token forward")
    func forwardManyEquivalence() async throws {
        let path = qwen36LocalPath
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        let qwen = try #require(m.qwen35, "expected Qwen35Model engine, got \(type(of: m.engine))")
        let prompt = "The history of the printing press began when"
        let encoded = m.tokenizer.encode(text: prompt)
        precondition(
            encoded.count >= 4,
            "forwardManyEquivalence: prompt encoded to \(encoded.count) tokens; need ≥ 4")

        // ── Reference path: T per-token `forward` calls on fresh caches.
        let refCaches = qwen.makeLayerCaches()
        var refLastLogits: Tensor!
        for (i, tok) in encoded.enumerated() {
            refLastLogits = qwen.forward(tokenId: tok, position: i, caches: refCaches)
        }
        let refLogits = refLastLogits.toFloatArray()
        let refArgmax = refLogits.enumerated().max(by: { $0.element < $1.element })!.offset

        // ── Batched path: one `forwardMany` over the whole prompt.
        let manyCaches = qwen.makeLayerCaches()
        let manyCmd = Device.shared.makeCommandBuffer()
        let manyLogitsTensor = qwen.forwardMany(
            tokenIds: encoded, startPosition: 0,
            caches: manyCaches, on: manyCmd, device: Device.shared)
        manyCmd.commit()
        await manyCmd.awaitCompletion()
        let manyLogits = manyLogitsTensor.toFloatArray()
        let manyArgmax = manyLogits.enumerated().max(by: { $0.element < $1.element })!.offset

        print(
            "forwardManyEquivalence T=\(encoded.count): ref argmax=\(refArgmax) batched argmax=\(manyArgmax)"
        )
        let refTop5 = refLogits.enumerated().sorted { $0.element > $1.element }.prefix(5)
            .map { (id: $0.offset, logit: $0.element) }
        let manyTop5 = manyLogits.enumerated().sorted { $0.element > $1.element }.prefix(5)
            .map { (id: $0.offset, logit: $0.element) }
        print("  ref top5: \(refTop5)")
        print("  many top5: \(manyTop5)")

        #expect(
            refArgmax == manyArgmax,
            "forwardMany batched argmax \(manyArgmax) ≠ per-token forward argmax \(refArgmax)")
        let refTopLogit = refLogits[refArgmax]
        let manyTopLogit = manyLogits[manyArgmax]
        let absDelta = abs(refTopLogit - manyTopLogit)
        #expect(
            absDelta < 0.5,
            "forwardMany batched top-1 logit \(manyTopLogit) drifted \(absDelta) from per-token \(refTopLogit)"
        )
    }

    @Test("Qwen3.6-35B-A3B forward pass — first-token greedy decode")
    func firstTokenForward() async throws {
        let path = qwen36LocalPath
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        // 11-token probe (matches the existing Qwen3.6 baseline prompt
        // used in the prefill-hypothesis sweeps). Greedy decode of one
        // token; compare against the mlx-swift-lm-produced baseline
        // first-token = 11.
        let prompt = "The history of the printing press began when"
        let tokenizer = m.tokenizer
        let encoded = tokenizer.encode(text: prompt)
        print("encoded prompt (\(encoded.count) tokens): \(encoded.prefix(20))")

        // Run the prefill manually so we can sample the first token
        // before any maxTokens-loop kicks in.
        let caches = m.engine.makeLayerCaches()
        var lastLogits: Tensor!
        let prefillStart = Date()
        for (i, tok) in encoded.enumerated() {
            lastLogits = m.engine.forward(tokenId: tok, position: i, caches: caches)
        }
        let prefillSecs = Date().timeIntervalSince(prefillStart)
        print("prefill \(encoded.count) tokens: \(String(format: "%.3f", prefillSecs))s")

        let logits = lastLogits.toFloatArray()
        precondition(
            logits.count == m.engine.vocab,
            "logits length \(logits.count) != vocab \(m.engine.vocab)")
        let argmax = logits.enumerated().max(by: { $0.element < $1.element })!.offset
        print("first-token argmax = \(argmax)")
        print("decoded first token: \(tokenizer.decode(tokens: [argmax]))")
        // top-5 for sanity
        let top5 = logits.enumerated().sorted { $0.element > $1.element }.prefix(5)
            .map { (id: $0.offset, logit: $0.element, tok: tokenizer.decode(tokens: [$0.offset])) }
        print("top5: \(top5)")
    }
}
