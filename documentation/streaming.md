# Streaming

Streaming is the primitive in FFAI's generation API; the buffered `Model.generate(...)` is a thin collector over the same stream so there's one source of truth for the prefill + decode loop.

```swift
let stream = model.generateStream(prompt: "Why is the sky blue?")
for try await chunk in stream {
    print(chunk.text, terminator: "")
}
```

Each chunk carries the decoded delta `text`, the new token id(s) since the last yield, and the absolute sequence position. The **final** chunk has empty `text` / `tokens` and a non-nil `stats: GenerationStats` with the full memory + timing numbers — same shape `--stats` prints.

## `GenerationChunk` shape

```swift
public struct GenerationChunk: Sendable {
    public let text: String              // decoded delta since last chunk
    public let tokens: [Int]             // new token id(s) in this chunk
    public let position: Int             // absolute sequence position after this chunk
    public let stats: GenerationStats?   // populated only on the final chunk
    public var isFinal: Bool { stats != nil }
}
```

## Cancellation

The stream-producing task honors Swift's structured cancellation. Drop the consuming task (or call `task.cancel()`) and the producer notices at the next token boundary, flushes the stream cleanly, and finishes — no zombie command buffers, no leaked KV cache.

```swift
let task = Task {
    for try await chunk in model.generateStream(prompt: "...") {
        if userInterrupted { break }
        print(chunk.text, terminator: "")
    }
}
// elsewhere:
task.cancel()
```

## Chat / multi-turn streaming

`generateStream` has a chat-templated overload that takes `[ChatMessage]` and renders through the tokenizer's chat template. Same chunk shape on the output side:

```swift
let messages: [ChatMessage] = [
    .init(role: .system, content: "You are concise."),
    .init(role: .user, content: "Why is the sky blue?"),
]
let stream = try model.generateStream(messages: messages)
for try await chunk in stream {
    print(chunk.text, terminator: "")
}
```

See [`chat-templates.md`](chat-templates.md) for how `ChatMessage`
+ `ChatTemplateOptions` map onto the tokenizer's Jinja template.

## Buffered collection

`Model.generate(prompt:parameters:)` is implemented as roughly:

```swift
public func generate(prompt: String,
                     parameters: GenerationParameters? = nil) async throws -> GenerationResult {
    var generated: [Int] = [], text = ""
    var stats: GenerationStats?
    for try await chunk in generateStream(prompt: prompt, parameters: parameters) {
        generated.append(contentsOf: chunk.tokens)
        text += chunk.text
        if let s = chunk.stats { stats = s }
    }
    return GenerationResult(
        promptTokens: tokenizer.encode(text: prompt),
        generatedTokens: generated, text: text, stats: stats!
    )
}
```

So you get the same `GenerationResult` shape — including all the stats — whether you call the buffered API or build your own collector around the stream.

## Why streaming is the primitive

Three reasons:

1. **The most common UI pattern.** Token-as-it-arrives chat / completion demos can't wait for the full response.
2. **Fits batch decoding naturally.** When Phase 8+ batching lands, `generateStream(...)` for a single sequence becomes one of N sequences sharing the same kernel dispatch loop; the consumer side stays unchanged.
3. **One loop.** Buffered + streaming sharing the same producer means the prefill / decode / stats path is exercised by every call shape — no parallel "almost-the-same" implementations to keep in sync.

## See also

- [Quick start](quickstart.md) — basic usage including the streaming example.
- [Chat templates](chat-templates.md) — how `messages:` overloads render through the tokenizer.
- [`generation-parameters.md`](generation-parameters.md) — the knobs that control generation; same parameters apply to streaming and buffered.
- [Observability](observability.md) — the `[STATS]` block streaming yields on its final chunk.
