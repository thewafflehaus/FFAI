# Chat Templates

Most modern chat / instruct models ship with a Jinja chat template in their `tokenizer_config.json`. FFAI calls into swift-transformers' `Tokenizer.applyChatTemplate(...)` to render those templates; you pass typed `ChatMessage` values + a typed `ChatTemplateOptions` and FFAI threads the right variables into the Jinja context.

The Phase-2/2.5 plain `Model.generate(prompt:...)` API takes a raw string — no chat template applied — which means *you* are responsible for rendering. Use the `messages:` overloads instead when working with chat / instruct models.

## Buffered

```swift
let messages: [ChatMessage] = [
    .init(role: .system, content: "You are concise."),
    .init(role: .user,   content: "Why is the sky blue?"),
]

let result = try await model.generate(messages: messages)
print(result.text)
```

## Streaming

```swift
let stream = try model.generateStream(messages: messages)
for try await chunk in stream {
    print(chunk.text, terminator: "")
}
```

Same chunk shape as the `prompt:` streaming variant — see [`streaming.md`](streaming.md).

## `ChatMessage`

```swift
public struct ChatMessage: Sendable, Equatable {
    public enum Role: String { case system, user, assistant, tool }
    public var role: Role
    public var content: String
    public var thinking: String?   // re-emit reasoning trace in multi-turn
}
```

The `thinking` field is for multi-turn conversations where the prior assistant turn included a thinking segment that the template wants to re-emit (Qwen 3 / DeepSeek-R1 do this).

## `ChatTemplateOptions`

```swift
public struct ChatTemplateOptions: Sendable {
    public var addGenerationPrompt: Bool        // = true
    public var enableThinking: Bool             // = false
    public var reasoningEffort: ReasoningEffort?// = nil  (.low | .medium | .high)
    public var maxLength: Int?                  // = nil
    public var truncation: Bool                 // = false
    public var extraContext: [String: any Sendable]
}
```

| Field | Maps to | When to set |
|---|---|---|
| `addGenerationPrompt` | template's "now generate the assistant reply" suffix | `true` (default) for the typical chat-completion case; `false` when scoring an existing assistant turn (e.g. perplexity over a fixed conversation). |
| `enableThinking` | `enable_thinking` Jinja variable | `true` to turn on the model's reasoning mode (Qwen 3 emits `<think>...</think>` blocks). Harmless on templates that don't reference the variable. |
| `reasoningEffort` | `reasoning_effort` Jinja variable | GPT-OSS Harmony reasoning levels (`low` / `medium` / `high`). |
| `maxLength` / `truncation` | swift-transformers' template-side truncation | Hard cap on the templated token count. `truncation: false` (default) throws on overflow; `true` truncates leading turns. |
| `extraContext` | additional Jinja variables | Anything else the template reads. |

## Format quirks

The template does the per-family rendering. We pass typed inputs through the well-known variable names; the rest is in the model's `tokenizer_config.json`. Specific behaviours:

| Family | Notes |
|---|---|
| **Qwen 3** | `enable_thinking: true` → `<think>...</think>` block before the answer. Pair with [`ThinkingSplit`](observability.md#think-vs-gen-split) for per-segment stats. |
| **DeepSeek-R1** | Same `<think>...</think>` convention as Qwen 3. |
| **GPT-OSS (Harmony)** | `reasoning_effort: "high"` (etc.) → analysis + final channel structure. The GPT-OSS family ships; pair with the `ThinkingSplit.harmony` scanner to partition the analysis + final channels. |
| **Gemma 3 / 4** | `<channel\|reasoning\|>` markers when reasoning is enabled. The Gemma 3 / 4 family files ship; the ThinkingSplit scanner partitions the reasoning segment. |
| **Llama 3 instruct** | Standard chat template, no reasoning hooks. |
| **Tools / function calling** | Not yet wired — `tools:` is plumbed through swift-transformers but FFAI's `ChatMessage` doesn't carry `toolCalls` yet. Lands when `.toolCalling` capability ships. |

## Errors

```swift
public enum ChatTemplateError: Error {
    case noTemplateOnTokenizer  // tokenizer_config.json had no chat_template
    case renderFailed(any Error) // wraps the underlying Jinja error
}
```

`noTemplateOnTokenizer` typically means you've loaded a base (non-chat) checkpoint and should either pass a raw prompt via `generate(prompt:)`, or use a different checkpoint (e.g. `*-Instruct`).

## Rendering without generating

For testing / debugging the templated input, render to token ids without running the model:

```swift
let ids = try model.renderChatTemplate(
    messages: messages,
    options: ChatTemplateOptions(enableThinking: true)
)
print(ids)
print(model.tokenizer.decode(tokens: ids, skipSpecialTokens: false))
```

## See also

- [Quickstart](quickstart.md) — `prompt:` vs `messages:` decision.
- [Streaming](streaming.md) — both overloads support streaming.
- [Observability § think vs gen split](observability.md#think-vs-gen-split) — what `enable_thinking: true` enables on the stats side.
