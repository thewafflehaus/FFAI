// Generate — autoregressive text generation, both buffered and streaming.
//
// Streaming is the primitive: `generateStream(prompt:parameters:)` yields
// one `GenerationChunk` per generated token plus a final chunk carrying
// the full `GenerationStats`. The buffered `generate(prompt:parameters:)`
// is a thin collector over the same stream so there's one source of truth
// for the prefill + decode loop.
//
// Phase 4 strategy: simple slow prefill (decode each prompt token
// sequentially through the same forward path used at decode time), then
// greedy GPU argmax decode. Sampling parameters on `GenerationParameters`
// (temperature, topP, …) are no-ops on the greedy fast path until GPU
// sampling kernels land in Phase 5. `prefillStepSize` is honored once
// chunked prefill ships; today's per-token prefill ignores it.
//
// Stats: a `PhaseMemoryTracker` samples GPU memory at each token boundary
// and at the prefill→decode transition. Cost is one
// `MTLDevice.currentAllocatedSize` read per token (sub-µs);
// `GenerationStats` is always populated.

import Foundation
import Tokenizers

// MARK: - Public types

public struct GenerationResult: Sendable {
    public let promptTokens: [Int]
    public let generatedTokens: [Int]
    public let text: String
    public let stats: GenerationStats

    public init(promptTokens: [Int], generatedTokens: [Int],
                text: String, stats: GenerationStats) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.text = text
        self.stats = stats
    }

    // Convenience read-throughs to the underlying stats.
    public var tokensPerSecond: Double { stats.decodeTokensPerSecond }
    public var prefillTokensPerSecond: Double { stats.prefillTokensPerSecond }
    public var prefillTimeS: Double { stats.prefillTimeS }
    public var decodeTimeS: Double { stats.decodeTimeS }
    public var timeToFirstTokenS: Double { stats.timeToFirstTokenMs / 1000.0 }
}

/// One yield from the streaming generation API. For decode tokens,
/// `text` and `tokens` carry the delta since the last yield. The
/// stream's final chunk has `tokens.isEmpty`, `text == ""`, and a
/// non-nil `stats` populated with full memory + timing numbers.
public struct GenerationChunk: Sendable {
    public let text: String
    public let tokens: [Int]
    public let position: Int
    public let stats: GenerationStats?

    public var isFinal: Bool { stats != nil }

    public init(text: String, tokens: [Int], position: Int,
                stats: GenerationStats? = nil) {
        self.text = text
        self.tokens = tokens
        self.position = position
        self.stats = stats
    }
}

// MARK: - Public API

public extension Model {
    /// Buffered generation. Collects every chunk from `generateStream`
    /// and returns the final result. Use this when you want the full
    /// text in one shot; use `generateStream` for token-by-token UI
    /// streaming.
    /// - Parameter profile: telemetry sink for this generation. Defaults
    ///   to the process-wide `Profile.shared`; pass a dedicated instance
    ///   to accumulate phase timings + signposts independently (the
    ///   prerequisite for per-sequence telemetry under Phase 8's batched
    ///   decode, where one shared singleton would conflate sequences).
    func generate(prompt: String,
                  parameters: GenerationParameters? = nil,
                  profile: Profile = .shared) async throws -> GenerationResult {
        let promptTokens = encodePrompt(prompt)
        let params = parameters ?? defaultGenerationParameters
        let stream = generateStreamInternal(promptTokens: promptTokens,
                                            parameters: params, profile: profile)
        return try await collectStream(stream, promptTokens: promptTokens)
    }

    /// Streaming generation. Yields one `GenerationChunk` per generated
    /// token (with the decoded delta text), then a final chunk with the
    /// full `GenerationStats`. Cancel the consuming `Task` to abort
    /// generation early — the producer task notices the cancellation at
    /// the next token boundary.
    ///
    /// - Parameter profile: telemetry sink for this generation. See
    ///   `generate(prompt:parameters:profile:)` for the rationale.
    func generateStream(prompt: String,
                        parameters: GenerationParameters? = nil,
                        profile: Profile = .shared)
        -> AsyncThrowingStream<GenerationChunk, Error> {
        let promptTokens = encodePrompt(prompt)
        let params = parameters ?? defaultGenerationParameters
        return generateStreamInternal(promptTokens: promptTokens,
                                      parameters: params, profile: profile)
    }

    // MARK: - Prompt encoding

    /// Encode a raw prompt string to token ids, prepending the model's
    /// `<bos>` token when the engine declares `requiresLeadingBOS` and
    /// the tokenizer's post-processor did not already add one.
    ///
    /// Gemma 4 is the motivating case: it is BOS-critical, but its
    /// `tokenizer.json` post-processor's `single` template is bare, so
    /// `Tokenizer.encode` returns no leading BOS. Without this prefix
    /// the model generates incoherent text. Gemma 3 and most other
    /// families list `<bos>` in their post-processor and need no fixup
    /// here. The guard against a BOS already being present keeps the
    /// helper idempotent and safe for families whose post-processor
    /// does add one.
    internal func encodePrompt(_ prompt: String) -> [Int] {
        var tokens = tokenizer.encode(text: prompt)
        guard engine.requiresLeadingBOS,
              let bos = tokenizer.bosTokenId ?? config.bosTokenId
        else { return tokens }
        if tokens.first != bos {
            tokens.insert(bos, at: 0)
        }
        return tokens
    }

    // MARK: - Internal entry points (shared by chat overloads)

    /// Build the producer stream from already-encoded prompt tokens.
    /// Used by both the `prompt:` and `messages:` public overloads so
    /// the tokenizer is invoked exactly once per call.
    internal func generateStreamInternal(promptTokens: [Int],
                                         parameters params: GenerationParameters,
                                         profile: Profile = .shared)
        -> AsyncThrowingStream<GenerationChunk, Error> {
        runStream(promptTokens: promptTokens, parameters: params, profile: profile)
    }

    /// Drain a generation stream into a buffered `GenerationResult`.
    internal func collectStream(_ stream: AsyncThrowingStream<GenerationChunk, Error>,
                                promptTokens: [Int]) async throws -> GenerationResult {
        var generated: [Int] = []
        var text = ""
        var stats: GenerationStats?
        for try await chunk in stream {
            generated.append(contentsOf: chunk.tokens)
            text += chunk.text
            if let s = chunk.stats { stats = s }
        }
        guard let stats else {
            throw GenerationError.streamEndedWithoutFinalChunk
        }
        return GenerationResult(
            promptTokens: promptTokens,
            generatedTokens: generated,
            text: text, stats: stats
        )
    }

    // MARK: - Core stream producer

    private func runStream(promptTokens: [Int],
                           parameters params: GenerationParameters,
                           profile: Profile)
        -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            // Capture the values we need on the producer task. `Model`
            // is @unchecked Sendable, so this is fine.
            let model = self
            Task {
                do {
                    try await model.driveGeneration(
                        promptTokens: promptTokens,
                        parameters: params,
                        profile: profile,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func driveGeneration(
        promptTokens: [Int],
        parameters params: GenerationParameters,
        profile: Profile,
        continuation: AsyncThrowingStream<GenerationChunk, Error>.Continuation
    ) async throws {
        let caches = engine.makeLayerCaches()
        // Gemma 3+, several Qwen variants, and a few llama-tuned models
        // publish `eos_token_id` as a *list* (model EOS plus end-of-turn
        // tokens like `<|im_end|>`). Stop on any of them.
        let eosIds = config.eosTokenIds
        let stopSet: Set<Int> = {
            var s = params.extraStopTokens
            if params.stopOnEOS { s.formUnion(eosIds) }
            return s
        }()
        let weightsBytes = engine.parameters().reduce(0) { $0 + $1.1.byteCount }
        let memTracker = PhaseMemoryTracker()

        // Path selector for the sampling step:
        //
        //   greedy-GPU       T == 0, no filters — existing argmax kernel
        //                    (4-byte readback per token, fastest path).
        //   gpu-categorical  T > 0, no filters — forward + GPU softmax +
        //                    categorical sample kernel. Logits stay on GPU;
        //                    only the chosen token id flows back (4 bytes).
        //   cpu-sample       Any of top-K / top-P / min-P / rep-penalty:
        //                    full vocab readback + Sampling.sample pipeline.
        let hasFilters = params.topK > 0 || params.topP < 1.0 || params.minP > 0
            || params.repetitionPenalty != 1.0
        enum Path: String { case greedyGPU, gpuCategorical, cpuSample }
        let path: Path = {
            if hasFilters             { return .cpuSample }
            if params.temperature == 0 { return .greedyGPU }
            return .gpuCategorical
        }()

        var rng = params.makeRNG()
        var tokenHistory = promptTokens   // grows as decode produces tokens; used by cpu-sample

        func sampleNext(tokenId t: Int, position i: Int) -> Int {
            switch path {
            case .greedyGPU:
                return engine.forwardSample(tokenId: t, position: i, caches: caches)
            case .gpuCategorical:
                let u = Float(Double.random(in: 0..<1, using: &rng))
                return engine.forwardSampleCategorical(
                    tokenId: t, position: i, caches: caches,
                    temperature: params.temperature, uniformDraw: u
                )
            case .cpuSample:
                let logits = engine.forward(tokenId: t, position: i, caches: caches)
                return Sampling.sample(logits, parameters: params,
                                       rng: &rng, tokenHistory: tokenHistory)
            }
        }

        Debug.log(.generate, "begin prefill: \(promptTokens.count) tokens, maxTokens=\(params.maxTokens), path=\(path.rawValue)")

        // ─── Prefill ─────────────────────────────────────────────────
        let prefillStart = Date()
        var nextToken = 0
        try profile.signpost("prefill") {
            for (i, t) in promptTokens.enumerated() {
                try Task.checkCancellation()
                nextToken = sampleNext(tokenId: t, position: i)
                memTracker.sample()
            }
        }
        memTracker.endPrefill()
        let prefillTime = Date().timeIntervalSince(prefillStart)
        profile.recordPhase("prefill", durationS: prefillTime)
        profile.event("ttft")
        Debug.log(.generate, String(format: "prefill done in %.3fs (%.1f tok/s)", prefillTime, Double(promptTokens.count) / max(prefillTime, 1e-9)))

        if promptTokens.isEmpty {
            let stats = makeStats(
                promptTokens: promptTokens, generatedCount: 0,
                contextSize: engine.maxSeq, prefillTime: 0, decodeTime: 0,
                ttftMs: 0, perTokenWallclock: [],
                memTracker: memTracker, caches: caches,
                weightsBytes: weightsBytes, splitTokens: nil
            )
            continuation.yield(GenerationChunk(text: "", tokens: [],
                                               position: 0, stats: stats))
            return
        }

        // TTFT for the slow per-token prefill is identical to prefill time.
        let ttftMs = prefillTime * 1000

        // ─── Decode ──────────────────────────────────────────────────
        let decodeStart = Date()
        var generated: [Int] = []
        var perTokenWallclock: [Double] = []
        var lastStep = decodeStart
        var pos = promptTokens.count
        for _ in 0..<params.maxTokens {
            try Task.checkCancellation()
            if stopSet.contains(nextToken) { break }
            generated.append(nextToken)
            let chunkText = tokenizer.decode(tokens: [nextToken],
                                             skipSpecialTokens: true)
            continuation.yield(GenerationChunk(text: chunkText,
                                               tokens: [nextToken],
                                               position: pos + 1, stats: nil))
            let priorToken = nextToken
            nextToken = profile.signpost("decode_step") {
                sampleNext(tokenId: priorToken, position: pos)
            }
            tokenHistory.append(priorToken)
            memTracker.sample()
            let now = Date()
            perTokenWallclock.append(now.timeIntervalSince(lastStep))
            lastStep = now
            pos += 1
        }
        let decodeTime = Date().timeIntervalSince(decodeStart)
        memTracker.endDecode()
        profile.recordPhase("ttft", durationS: ttftMs / 1000)
        profile.recordPhase("decode", durationS: decodeTime)
        profile.recordPhase("generation_total", durationS: prefillTime + decodeTime)
        Debug.log(.generate, String(format: "decode done: %d tokens in %.3fs (%.1f tok/s)", generated.count, decodeTime, Double(generated.count) / max(decodeTime, 1e-9)))

        let split = ThinkingSplit.split(tokens: generated, model: self)
        let stats = makeStats(
            promptTokens: promptTokens, generatedCount: generated.count,
            contextSize: engine.maxSeq, prefillTime: prefillTime,
            decodeTime: decodeTime, ttftMs: ttftMs,
            perTokenWallclock: perTokenWallclock,
            memTracker: memTracker, caches: caches,
            weightsBytes: weightsBytes, splitTokens: split
        )

        continuation.yield(GenerationChunk(text: "", tokens: [],
                                           position: pos, stats: stats))
    }

    // MARK: - Stats assembly

    private func makeStats(
        promptTokens: [Int], generatedCount: Int, contextSize: Int,
        prefillTime: Double, decodeTime: Double, ttftMs: Double,
        perTokenWallclock: [Double],
        memTracker: PhaseMemoryTracker, caches: [any LayerCacheProtocol],
        weightsBytes: Int, splitTokens: ThinkingSplit.Split?
    ) -> GenerationStats {
        let steady: Double? = {
            guard perTokenWallclock.count > 10 else { return nil }
            let tail = perTokenWallclock.dropFirst(10)
            let totalS = tail.reduce(0, +)
            return totalS > 0 ? Double(tail.count) / totalS : nil
        }()

        return GenerationStats(
            promptTokens: promptTokens.count,
            generatedTokens: generatedCount,
            contextSize: contextSize,
            prefillTimeS: prefillTime,
            decodeTimeS: decodeTime,
            timeToFirstTokenMs: ttftMs,
            steadyTokensPerSecond: steady,
            baselineGPUBytes: memTracker.baseline.gpuBytes,
            postPrefillGPUBytes: memTracker.postPrefill?.gpuBytes ?? memTracker.baseline.gpuBytes,
            postDecodeGPUBytes: memTracker.postDecode?.gpuBytes ?? memTracker.baseline.gpuBytes,
            prefillPeakGPUBytes: memTracker.prefillPeakBytes,
            decodePeakGPUBytes: memTracker.decodePeakBytes,
            wiredTicketBytes: memTracker.baseline.wiredTicketBytes,
            weightsBytes: weightsBytes,
            kvCacheAllocatedBytes: caches.totalBytesAllocated,
            kvCacheUsedBytes: caches.totalBytesInUse,
            thinkPerplexity: nil,
            genPerplexity: nil,
            thinkKLDivergence: nil,
            genKLDivergence: nil,
            thinkTokenCount: splitTokens.map { $0.thinkTokens.count },
            genTokenCount: splitTokens.map { $0.genTokens.count }
        )
    }
}

public enum GenerationError: Error, CustomStringConvertible {
    case streamEndedWithoutFinalChunk
    public var description: String {
        switch self {
        case .streamEndedWithoutFinalChunk:
            return "GenerationStream finished without yielding the final stats chunk"
        }
    }
}
