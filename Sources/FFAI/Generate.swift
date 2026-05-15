// Generate — autoregressive text generation loop.
//
// Phase 2 strategy: simple slow prefill (decode each prompt token
// sequentially through the same forward path used at decode time), then
// decode loop with greedy argmax sampling. Token-by-token streaming via
// AsyncStream.

import Foundation
import Tokenizers

public struct GenerateOptions: Sendable {
    public var maxNewTokens: Int
    public var stopOnEOS: Bool

    public init(maxNewTokens: Int = 64, stopOnEOS: Bool = true) {
        self.maxNewTokens = maxNewTokens
        self.stopOnEOS = stopOnEOS
    }
}

public struct GenerationResult: Sendable {
    public let promptTokens: [Int]
    public let generatedTokens: [Int]
    public let text: String
    public let prefillTimeS: Double
    public let decodeTimeS: Double
    public var tokensPerSecond: Double {
        decodeTimeS > 0 ? Double(generatedTokens.count) / decodeTimeS : 0
    }
}

public extension Model {
    /// Run prompt → generate(maxNewTokens) returning the final result.
    func generate(prompt: String,
                  options: GenerateOptions = GenerateOptions()) async throws -> GenerationResult {
        let promptTokens = tokenizer.encode(text: prompt)
        let caches = engine.makeKVCache()
        let eos = config.eosTokenId

        // Prefill: feed each prompt token through forwardSample so the
        // last call returns the first sampled token directly (logits
        // never leave the GPU). Earlier prompt tokens use the same path
        // for symmetry — the sampled value is discarded.
        let prefillStart = Date()
        var nextToken = 0
        for (i, t) in promptTokens.enumerated() {
            nextToken = engine.forwardSample(tokenId: t, position: i, caches: caches)
        }
        let prefillTime = Date().timeIntervalSince(prefillStart)
        guard !promptTokens.isEmpty else {
            return GenerationResult(promptTokens: promptTokens, generatedTokens: [],
                                    text: "", prefillTimeS: 0, decodeTimeS: 0)
        }

        // Decode loop. nextToken already holds the first sampled token
        // from the last prefill step.
        let decodeStart = Date()
        var generated: [Int] = []
        var pos = promptTokens.count
        for _ in 0..<options.maxNewTokens {
            if options.stopOnEOS, let e = eos, nextToken == e { break }
            generated.append(nextToken)
            nextToken = engine.forwardSample(tokenId: nextToken, position: pos,
                                             caches: caches)
            pos += 1
        }
        let decodeTime = Date().timeIntervalSince(decodeStart)

        let text = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        return GenerationResult(
            promptTokens: promptTokens, generatedTokens: generated,
            text: text, prefillTimeS: prefillTime, decodeTimeS: decodeTime
        )
    }
}
