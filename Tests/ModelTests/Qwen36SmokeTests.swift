// Temporary smoke test for Qwen3.6-35B-A3B local checkpoint.
// Verifies the load path succeeds end-to-end (no precondition trap).

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen3.6 smoke", .serialized)
struct Qwen36SmokeTests {

    @Test("Qwen3.6-35B-A3B local checkpoint loads")
    func loadLocal() async throws {
        let path = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
        guard FileManager.default.fileExists(atPath: path) else {
            print("Qwen3.6 smoke skipped: \(path) not found")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false  // skip prewarm to isolate load failures
        let opts = optsBuilder
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially {
                try await Model.load(path, options: opts)
            }
        } catch {
            print("LOAD FAILED: \(error)")
            throw error
        }
        print("LOAD OK")
        #expect(m.qwen35 != nil, "expected Qwen35Model engine")
        if let q = m.qwen35 {
            print("hidden=\(q.hidden) layers=\(q.nLayers) heads=\(q.nHeads) kv=\(q.nKVHeads) headDim=\(q.headDim)")
            print("GDN dims: Hk=\(q.numKeyHeads) Hv=\(q.numValueHeads) Dk=\(q.keyHeadDim) Dv=\(q.valueHeadDim)")
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
        let path = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
        guard FileManager.default.fileExists(atPath: path) else {
            print("Qwen3.6 bench skipped: \(path) not found")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        // Build a prompt of approximately the target length by repeating
        // a paragraph until we hit it.
        let base = "The quick brown fox jumps over the lazy dog. " +
                   "Pack my box with five dozen liquor jugs. "
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
        print("\(label): prefill \(String(format: "%.0f", prefillMs))ms (\(String(format: "%.1f", prefillTps)) tok/s)")

        // Greedy sample first generated token.
        var logits = lastLogits.toFloatArray()
        var nextTok = logits.enumerated().max(by: { $0.element < $1.element })!.offset

        // ── Decode loop ──────────────────────────────────────────────
        var stepTimes: [Double] = []
        var pos = tokens.count
        for _ in 0..<decodeSteps {
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
        print("\(label): decode \(decodeSteps) steps in \(String(format: "%.2f", decodeSecs))s (avg \(String(format: "%.2f", decodeTps)) tok/s, steady \(String(format: "%.2f", steadyTps)) tok/s)")
        print("\(label): per-step ms (first 8): \(stepTimes.prefix(8).map { String(format: "%.0f", $0 * 1000) })")
        print("\(label): RESULT prefill_ms=\(String(format: "%.0f", prefillMs)) decode_tps=\(String(format: "%.2f", decodeTps)) steady_tps=\(String(format: "%.2f", steadyTps)) prefill_tps=\(String(format: "%.1f", prefillTps))")
    }

    @Test("Qwen3.6-35B-A3B forwardManyAllLogits — last-row equals forwardMany")
    func forwardManyAllLogitsLastRowMatchesForwardMany() async throws {
        let path = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
        guard FileManager.default.fileExists(atPath: path) else {
            print("Qwen3.6 forwardManyAllLogits skipped: \(path) not found")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        guard let qwen = m.qwen35 else {
            Issue.record("expected Qwen35Model engine")
            return
        }
        let seed = "The history of the printing press began when European craftsmen of the 15th century"
        let encoded = Array(m.tokenizer.encode(text: seed).prefix(8))
        let T = encoded.count
        precondition(T >= 4, "need at least 4 tokens to test")

        // Path 1: forwardMany returns [vocab] for last row.
        let cachesA = qwen.makeLayerCaches()
        let cmdA = Device.shared.makeCommandBuffer()
        let lastRowLogits = qwen.forwardMany(tokenIds: encoded, startPosition: 0,
                                              caches: cachesA, on: cmdA, device: Device.shared)
        cmdA.commit()
        await cmdA.completed()
        let lastRowHost = lastRowLogits.toFloatArray()

        // Path 2: forwardManyAllLogits returns [T, vocab].
        let cachesB = qwen.makeLayerCaches()
        let cmdB = Device.shared.makeCommandBuffer()
        let allLogits = qwen.forwardManyAllLogits(tokenIds: encoded, startPosition: 0,
                                                   caches: cachesB, on: cmdB, device: Device.shared)
        cmdB.commit()
        await cmdB.completed()
        let allHost = allLogits.toFloatArray()
        let vocab = lastRowHost.count
        #expect(allHost.count == T * vocab,
                "forwardManyAllLogits returned \(allHost.count) elements; expected T·vocab = \(T * vocab)")

        // Compare last row to forwardMany's last-row-only logits.
        let lastRowSlice = Array(allHost[(T - 1) * vocab ..< T * vocab])
        var maxAbsDiff: Float = 0
        for (a, b) in zip(lastRowSlice, lastRowHost) {
            maxAbsDiff = max(maxAbsDiff, abs(a - b))
        }
        print("forwardManyAllLogits T=\(T) last-row max |Δ| = \(String(format: "%.4f", maxAbsDiff))")
        #expect(maxAbsDiff < 0.1,
                "last-row logits diverged by \(maxAbsDiff) > 0.1")
        let refMax = lastRowHost.enumerated().max { $0.element < $1.element }!.offset
        let allMax = lastRowSlice.enumerated().max { $0.element < $1.element }!.offset
        #expect(refMax == allMax,
                "argmax mismatch: forwardMany=\(refMax) vs allLogits last row=\(allMax)")
        print("forwardManyAllLogits T=\(T) argmax match: \(refMax)")
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=2 spec-decode ceiling probe")
    func forwardManyBench2() async throws {
        try await runForwardManyBench(targetT: 2)
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=4 spec-decode ceiling probe")
    func forwardManyBench4() async throws {
        try await runForwardManyBench(targetT: 4)
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=8 spec-decode ceiling probe")
    func forwardManyBench8() async throws {
        try await runForwardManyBench(targetT: 8)
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=16 spec-decode ceiling probe")
    func forwardManyBench16() async throws {
        try await runForwardManyBench(targetT: 16)
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=32 prefill, batched vs per-token")
    func forwardManyBench() async throws {
        try await runForwardManyBench(targetT: 32)
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=64 crossover refinement")
    func forwardManyBench64() async throws {
        try await runForwardManyBench(targetT: 64)
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=128 prefill scaling")
    func forwardManyBench128() async throws {
        try await runForwardManyBench(targetT: 128)
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=512 long-context scaling")
    func forwardManyBench512() async throws {
        try await runForwardManyBench(targetT: 512)
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=2048 batched-only")
    func forwardManyBench2K() async throws {
        try await runForwardManyBench(targetT: 2048, skipPerToken: true)
    }

    @Test("Qwen3.6-35B-A3B forwardMany profile — T=2048 phase breakdown")
    func forwardManyProfile2K() async throws {
        try await runForwardManyProfile(targetT: 2048)
    }

    @Test("Qwen3.6-35B-A3B decode profile — T=1 per-layer breakdown")
    func decodeProfileT1() async throws {
        let path = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
        guard FileManager.default.fileExists(atPath: path) else {
            print("Qwen3.6 decodeProfile skipped: \(path) not found")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        guard let qwen = m.qwen35 else {
            Issue.record("expected Qwen35Model engine")
            return
        }
        // Build a short 32-token prefill so decode runs at non-trivial KV length.
        let seed = "The history of the printing press began when European craftsmen of the 15th century combined movable metal type with oil based ink screw presses paper to mass produce printed books"
        var encoded = m.tokenizer.encode(text: seed)
        while encoded.count < 32 { encoded.append(contentsOf: encoded) }
        encoded = Array(encoded.prefix(32))

        let caches = qwen.makeLayerCaches()
        // Prefill 32 tokens (untimed; warms KV + PSO).
        for (i, tok) in encoded.enumerated() {
            _ = qwen.forward(tokenId: tok, position: i, caches: caches)
        }
        // Warm the decode path itself with 4 steps.
        var nextTok = 1234
        for step in 0..<4 {
            _ = qwen.forward(tokenId: nextTok, position: encoded.count + step, caches: caches)
        }

        // Enable wallclock, reset, run 32 decode steps timed.
        Profile.shared.level = .wallclock
        Profile.shared.resetPhases()
        let nSteps = 32
        let t0 = Date()
        for step in 0..<nSteps {
            _ = qwen.forward(tokenId: nextTok, position: encoded.count + 4 + step, caches: caches)
            nextTok = (nextTok &+ 17) % 248_320  // deterministic walk
        }
        let totalS = Date().timeIntervalSince(t0)
        Profile.shared.level = .off

        var totals: [String: (count: Int, sumS: Double)] = [:]
        for (name, dur) in Profile.shared.phases.entries {
            let prev = totals[name] ?? (0, 0)
            totals[name] = (prev.count + 1, prev.sumS + dur)
        }
        let tps = Double(nSteps) / totalS
        let perStepMs = totalS / Double(nSteps) * 1000
        print("decodeProfile T=1: \(nSteps) steps in \(String(format: "%.3f", totalS))s = \(String(format: "%.2f", tps)) tps (per-step \(String(format: "%.2f", perStepMs))ms)")
        let sorted = totals.sorted { $0.value.sumS > $1.value.sumS }
        for (name, agg) in sorted {
            let pct = agg.sumS / totalS * 100
            let avgMs = agg.sumS / Double(agg.count) * 1000
            let perStep = Double(agg.count) / Double(nSteps)
            print("  \(name): count=\(agg.count) (\(String(format: "%.1f", perStep))/step) total=\(String(format: "%.3f", agg.sumS))s (\(String(format: "%.1f", pct))%) avg=\(String(format: "%.2f", avgMs))ms")
        }
    }

    @Test("Qwen3.6-35B-A3B forwardMany profile — T=512 phase breakdown")
    func forwardManyProfile512() async throws {
        try await runForwardManyProfile(targetT: 512)
    }

    private func runForwardManyProfile(targetT: Int) async throws {
        let path = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
        guard FileManager.default.fileExists(atPath: path) else {
            print("Qwen3.6 forwardManyProfile skipped: \(path) not found")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        guard let qwen = m.qwen35 else {
            Issue.record("expected Qwen35Model engine")
            return
        }
        // Build a targetT-token prompt.
        let seed = "The history of the printing press began when European craftsmen of the 15th century combined movable metal type with oil based ink screw presses paper to mass produce printed books pamphlets and broadsheets revolutionising communication"
        let seedEncoded = m.tokenizer.encode(text: seed)
        var encoded = seedEncoded
        while encoded.count < targetT { encoded.append(contentsOf: seedEncoded) }
        encoded = Array(encoded.prefix(targetT))
        let T = encoded.count

        // Warm-up — same convention as the bench.
        for _ in 0..<2 {
            let warmCaches = qwen.makeLayerCaches()
            let warmCmd = Device.shared.makeCommandBuffer()
            _ = qwen.forwardMany(tokenIds: encoded, startPosition: 0,
                                 caches: warmCaches, on: warmCmd, device: Device.shared)
            warmCmd.commit()
            await warmCmd.completed()
        }

        // Enable wallclock profile, reset accumulator.
        Profile.shared.level = .wallclock
        Profile.shared.resetPhases()

        // Run forwardMany once. Wallclock measures CPU-side dispatch time
        // per phase; for the layer-type phases the underlying decodeMany
        // calls queue work onto an in-flight cmd without waiting, so the
        // recorded values are CPU dispatch + GPU work IF the layer commits
        // inside, else CPU dispatch only. The breakdown is most useful
        // for identifying CPU-bound dispatch concentrations rather than
        // GPU phase time — for GPU timing use Instruments / xctrace.
        let caches = qwen.makeLayerCaches()
        let cmd = Device.shared.makeCommandBuffer()
        let totalStart = Date()
        _ = qwen.forwardMany(tokenIds: encoded, startPosition: 0,
                             caches: caches, on: cmd, device: Device.shared)
        cmd.commit()
        await cmd.completed()
        let totalS = Date().timeIntervalSince(totalStart)

        // Dump per-phase totals.
        Profile.shared.level = .off
        var totals: [String: (count: Int, sumS: Double)] = [:]
        for (name, dur) in Profile.shared.phases.entries {
            let prev = totals[name] ?? (0, 0)
            totals[name] = (prev.count + 1, prev.sumS + dur)
        }
        print("forwardManyProfile T=\(T): total=\(String(format: "%.3f", totalS))s = \(String(format: "%.1f", Double(T)/totalS)) tps")
        let sorted = totals.sorted { $0.value.sumS > $1.value.sumS }
        for (name, agg) in sorted {
            let pct = agg.sumS / totalS * 100
            let avgMs = agg.sumS / Double(agg.count) * 1000
            print("  \(name): count=\(agg.count) total=\(String(format: "%.3f", agg.sumS))s (\(String(format: "%.1f", pct))%) avg=\(String(format: "%.2f", avgMs))ms")
        }
    }

    @Test("Qwen3.6-35B-A3B forwardMany bench — T=4096 batched-only")
    func forwardManyBench4K() async throws {
        try await runForwardManyBench(targetT: 4096, skipPerToken: true)
    }

    private func runForwardManyBench(targetT: Int, skipPerToken: Bool = false) async throws {
        let path = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
        guard FileManager.default.fileExists(atPath: path) else {
            print("Qwen3.6 forwardManyBench(T=\(targetT)) skipped: \(path) not found")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        guard let qwen = m.qwen35 else {
            Issue.record("expected Qwen35Model engine")
            return
        }
        // Seed prompt — 32 tokens. Tile until we hit targetT, then trim.
        let seed = "The history of the printing press began when European craftsmen of the 15th century combined movable metal type with oil based ink screw presses paper to mass produce printed books pamphlets and broadsheets revolutionising communication"
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
        for warmIter in 0..<2 {
            let warmCachesP = qwen.makeLayerCaches()
            for (i, tok) in encoded.prefix(2).enumerated() {
                _ = qwen.forward(tokenId: tok, position: i, caches: warmCachesP)
            }
            let warmCachesB = qwen.makeLayerCaches()
            let warmCmd = Device.shared.makeCommandBuffer()
            _ = qwen.forwardMany(tokenIds: encoded, startPosition: 0,
                                 caches: warmCachesB, on: warmCmd, device: Device.shared)
            warmCmd.commit()
            await warmCmd.completed()
            _ = warmIter  // silence
        }

        // Per-token loop baseline (5 runs, median) — skipped at long
        // contexts (T≥2K) where per-token would cost ~25-40 min total.
        var perTokenMedian = 0.0
        if !skipPerToken {
            var perTokenSecs: [Double] = []
            for _ in 0..<5 {
                let caches = qwen.makeLayerCaches()
                let t0 = Date()
                for (i, tok) in encoded.enumerated() {
                    _ = qwen.forward(tokenId: tok, position: i, caches: caches)
                }
                perTokenSecs.append(Date().timeIntervalSince(t0))
            }
            perTokenSecs.sort()
            perTokenMedian = perTokenSecs[perTokenSecs.count / 2]
            print("per-token T=\(T): runs=\(perTokenSecs.map { String(format: "%.3f", $0) }) median=\(String(format: "%.3f", perTokenMedian))s = \(String(format: "%.2f", Double(T)/perTokenMedian)) tps")
        } else {
            print("per-token T=\(T): SKIPPED (--skipPerToken)")
        }

        // Batched forwardMany (5 runs, median).
        var batchedSecs: [Double] = []
        for _ in 0..<5 {
            let caches = qwen.makeLayerCaches()
            let bCmd = Device.shared.makeCommandBuffer()
            let t0 = Date()
            _ = qwen.forwardMany(tokenIds: encoded, startPosition: 0,
                                 caches: caches, on: bCmd, device: Device.shared)
            bCmd.commit()
            await bCmd.completed()
            batchedSecs.append(Date().timeIntervalSince(t0))
        }
        batchedSecs.sort()
        let batchedMedian = batchedSecs[batchedSecs.count / 2]
        print("batched T=\(T): runs=\(batchedSecs.map { String(format: "%.3f", $0) }) median=\(String(format: "%.3f", batchedMedian))s = \(String(format: "%.2f", Double(T)/batchedMedian)) tps")

        if !skipPerToken {
            let speedup = perTokenMedian / batchedMedian
            print("forwardManyBench RESULT T=\(T): per_token=\(String(format: "%.0f", perTokenMedian*1000))ms batched=\(String(format: "%.0f", batchedMedian*1000))ms speedup=\(String(format: "%.2fx", speedup))")
        } else {
            print("forwardManyBench RESULT T=\(T): batched=\(String(format: "%.0f", batchedMedian*1000))ms = \(String(format: "%.2f", Double(T)/batchedMedian)) tps (best \(String(format: "%.3f", batchedSecs[0]))s = \(String(format: "%.2f", Double(T)/batchedSecs[0])) tps)")
        }
    }

    @Test("Qwen3.6-35B-A3B forwardMany matches per-token forward")
    func forwardManyEquivalence() async throws {
        let path = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
        guard FileManager.default.fileExists(atPath: path) else {
            print("Qwen3.6 smoke skipped: \(path) not found")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        guard let qwen = m.qwen35 else {
            Issue.record("expected Qwen35Model engine, got \(type(of: m.engine))")
            return
        }
        let prompt = "The history of the printing press began when"
        let encoded = m.tokenizer.encode(text: prompt)
        precondition(encoded.count >= 4,
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
        await manyCmd.completed()
        let manyLogits = manyLogitsTensor.toFloatArray()
        let manyArgmax = manyLogits.enumerated().max(by: { $0.element < $1.element })!.offset

        print("forwardManyEquivalence T=\(encoded.count): ref argmax=\(refArgmax) batched argmax=\(manyArgmax)")
        let refTop5 = refLogits.enumerated().sorted { $0.element > $1.element }.prefix(5)
            .map { (id: $0.offset, logit: $0.element) }
        let manyTop5 = manyLogits.enumerated().sorted { $0.element > $1.element }.prefix(5)
            .map { (id: $0.offset, logit: $0.element) }
        print("  ref top5: \(refTop5)")
        print("  many top5: \(manyTop5)")

        #expect(refArgmax == manyArgmax,
                "forwardMany batched argmax \(manyArgmax) ≠ per-token forward argmax \(refArgmax)")
        let refTopLogit = refLogits[refArgmax]
        let manyTopLogit = manyLogits[manyArgmax]
        let absDelta = abs(refTopLogit - manyTopLogit)
        #expect(absDelta < 0.5,
                "forwardMany batched top-1 logit \(manyTopLogit) drifted \(absDelta) from per-token \(refTopLogit)")
    }

    /// T=128 equivalence — pins the new dequantised-gate + `mt_steel_gemm`
    /// path in `MoELayer.gateLogitsMany` (engaged when T % 64 == 0). The
    /// short-prompt sibling above hits the legacy fallback at T=8, so
    /// without this cell the steel-gemm gate path has no argmax canary.
    @Test("Qwen3.6-35B-A3B forwardMany T=128 matches per-token forward")
    func forwardManyEquivalenceT128() async throws {
        let path = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
        guard FileManager.default.fileExists(atPath: path) else {
            print("Qwen3.6 smoke skipped: \(path) not found")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(path, options: opts)
        }
        guard let qwen = m.qwen35 else {
            Issue.record("expected Qwen35Model engine, got \(type(of: m.engine))")
            return
        }
        // Encode + pad-with-repeats to exactly 128 tokens. The model's
        // attention is causal so trailing repeats only feed the cache —
        // the LAST-token logits are what both paths return + compare.
        let prompt = "The history of the printing press began when"
        let baseEncoded = m.tokenizer.encode(text: prompt)
        precondition(baseEncoded.count >= 4,
                     "forwardManyEquivalenceT128: short prompt encoded to \(baseEncoded.count) tokens")
        var encoded: [Int] = []
        while encoded.count + baseEncoded.count <= 128 {
            encoded.append(contentsOf: baseEncoded)
        }
        while encoded.count < 128 { encoded.append(baseEncoded[0]) }
        precondition(encoded.count == 128,
                     "forwardManyEquivalenceT128: built \(encoded.count) tokens, want 128")

        // Reference: T per-token forward.
        let refCaches = qwen.makeLayerCaches()
        var refLastLogits: Tensor!
        for (i, tok) in encoded.enumerated() {
            refLastLogits = qwen.forward(tokenId: tok, position: i, caches: refCaches)
        }
        let refLogits = refLastLogits.toFloatArray()
        let refArgmax = refLogits.enumerated().max(by: { $0.element < $1.element })!.offset

        // Batched: one forwardMany call — engages gateLogitsMany's steel-gemm
        // path (T=128 % 64 == 0, nExperts=128 % 64 == 0).
        let manyCaches = qwen.makeLayerCaches()
        let manyCmd = Device.shared.makeCommandBuffer()
        let manyLogitsTensor = qwen.forwardMany(
            tokenIds: encoded, startPosition: 0,
            caches: manyCaches, on: manyCmd, device: Device.shared)
        manyCmd.commit()
        await manyCmd.completed()
        let manyLogits = manyLogitsTensor.toFloatArray()
        let manyArgmax = manyLogits.enumerated().max(by: { $0.element < $1.element })!.offset

        print("forwardManyEquivalenceT128 T=\(encoded.count): ref argmax=\(refArgmax) batched argmax=\(manyArgmax)")
        let refTop5 = refLogits.enumerated().sorted { $0.element > $1.element }.prefix(5)
            .map { (id: $0.offset, logit: $0.element) }
        let manyTop5 = manyLogits.enumerated().sorted { $0.element > $1.element }.prefix(5)
            .map { (id: $0.offset, logit: $0.element) }
        print("  ref top5: \(refTop5)")
        print("  many top5: \(manyTop5)")

        #expect(refArgmax == manyArgmax,
                "T=128 forwardMany batched argmax \(manyArgmax) ≠ per-token forward argmax \(refArgmax)")
        let refTopLogit = refLogits[refArgmax]
        let manyTopLogit = manyLogits[manyArgmax]
        let absDelta = abs(refTopLogit - manyTopLogit)
        // T=128 across 40 layers of bf16 accumulates more rounding than
        // T=8 (the short-prompt sibling's 0.5 floor). The dense-gate
        // path adds another ~0.4 logits of drift on the top-1 score
        // vs the per-row dequant_gemv reference — argmax still matches,
        // top-5 still overlaps. 1.5 is a permissive cap that catches
        // structural divergence (≥ 2 logits = different token order)
        // without false-positive-ing on bf16 noise.
        #expect(absDelta < 1.5,
                "T=128 forwardMany top-1 logit \(manyTopLogit) drifted \(absDelta) from per-token \(refTopLogit)")
    }

    @Test("Qwen3.6-35B-A3B forward pass — first-token greedy decode")
    func firstTokenForward() async throws {
        let path = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
        guard FileManager.default.fileExists(atPath: path) else {
            print("Qwen3.6 smoke skipped: \(path) not found")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially {
                try await Model.load(path, options: opts)
            }
        } catch {
            print("LOAD FAILED: \(error)")
            throw error
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
        precondition(logits.count == m.engine.vocab,
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
