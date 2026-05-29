# Quick Start

Downloading, loading a model and generating text in 5 lines in Swift!

```swift
import FFAI

let model = try await Model.load("mlx-community/Qwen3.5-0.8B-MLX-4bit")
let result = try await model.generate(prompt: "Once upon a time")
print(result.text)
```

The first call resolves and downloads the checkpoint on demand (cached under `~/.cache/huggingface/hub/`), parses `config.json`, loads weights into per-tensor MTLBuffers, attaches the tokenizer, and prewarms the PSO cache. `generate(...)` defaults to the model's family-declared [`defaultGenerationParameters`](generation-parameters.md) — Llama and Qwen 3 each carry their own. The returned `GenerationResult` carries the prompt + generated tokens, the decoded text, and prefill / decode timings.

```swift
print("\(result.promptTokens.count) prompt + \(result.generatedTokens.count) generated tokens")
print(String(format: "%.2f tok/s", result.tokensPerSecond))
```

## CLI

The same surface ships as the `ffai` executable. See [using-the-cli.md](using-the-cli.md) for how to build the binary and get it on `PATH`; once that's done:

```bash
ffai --model mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "Once upon a time"
ffai --model mlx-community/Qwen3-4B-4bit --prompt "Hello" --max-tokens 128
```

Tokens are streamed to stdout as they're generated. Pass `--no-streaming` to print the full text once at the end (matches the buffered API exactly). Pass `--stats` for the post-run `[STATS]` block (per-phase memory, TTFT, KV cache, wired ticket — see [observability.md](observability.md)). Pass `--verbose` to print the top-5 next-token distribution from a single prefill instead of generating.

## Customizing the generation

The second argument to `generate` is a [`GenerationParameters`](generation-parameters.md). Omit it (or pass `nil`) to use the family default — overriding a single field with the `with(_:)` copy-mutator preserves the family-tuned baseline:

```swift
let result = try await model.generate(
    prompt: "Once upon a time",
    parameters: model.defaultGenerationParameters.with { $0.maxTokens = 64 }
)
```

For the full field table (sampling temp, top-p / top-k, repetition penalty, prefill chunk size, …) and which fields are honored today vs staged for Phase 5, see [`generation-parameters.md`](generation-parameters.md).

## Streaming

Streaming is the primitive — buffered `generate(...)` collects from the same stream:

```swift
for try await chunk in model.generateStream(prompt: "Why is the sky blue?") {
    print(chunk.text, terminator: "")
}
```

The final chunk carries the full `GenerationStats` (peak GPU, KV cache size, TTFT, …) on its `stats` property. Cancel the consuming task to stop generation early — the producer notices at the next token boundary. See [streaming.md](streaming.md).

## Chat / multi-turn

For instruct / chat models, pass `[ChatMessage]` and FFAI applies the tokenizer's chat template:

```swift
let messages: [ChatMessage] = [
    .init(role: .system, content: "You are concise."),
    .init(role: .user,   content: "Why is the sky blue?"),
]
let result = try await model.generate(messages: messages)
print(result.text)
```

For reasoning-tuned models (Qwen 3, DeepSeek-R1, GPT-OSS), opt into the model's thinking / reasoning hooks:

```swift
let result = try await model.generate(
    messages: messages,
    templateOptions: ChatTemplateOptions(enableThinking: true)
)
```

See [chat-templates.md](chat-templates.md) for the full options surface and per-family quirks.

## Using Different Model Quantizations

The same call works for any [mlx-format](quantization.md) checkpoint — 3 / 4 / 5 / 6 / 8-bit:

```swift
let model = try await Model.load("mlx-community/Qwen3-4B-8bit")
let result = try await model.generate(prompt: "What is the capital of France?")
print(result.text)
```

## Customizing the model loading

`Model.load(_:options:)` takes a `LoadOptions`:

```swift
let model = try await Model.load(
    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    options: LoadOptions(
        capabilities: [.textIn, .textOut],
        kvCache: .raw,
        prewarm: true,
        revision: "main"
    )
)
```

| Field | Default | Notes |
|---|---|---|
| `capabilities` | `[.textIn, .textOut]` | Which capabilities to load. Disabled modalities skip weight allocation entirely (relevant for VLMs in Phase 6). |
| `kvCache` | `.raw` | KV cache scheme. `.raw` keeps unquantized fp16 / bf16 K/V tensors (the default — zero overhead per step). `.affineQuantized(bits:groupSize:)` compresses K/V to int4 / int8 with a shared per-layer working buffer + a `bulk_dequant_kv` pre-pass before SDPA (~45 % memory savings at 8-bit, ~65 % at 4-bit; CLI: `--kv-cache affine8` / `affine4`). `.auraQuantized(scheme:)` is AURA — rotated + Lloyd-Max scalar-quantized + bit-packed, with compressed-domain attention via the `aura_flash_p1` / `aura_flash_pass2` kernel pair (~4× memory savings at `aura4v2`; CLI: `--kv-cache aura4v2` / `aura4` / `aura3`). See [kv-cache.md](kv-cache.md) for the per-scheme trade-offs. |
| `dispatchMode` | `.eager` | Standard `MTLComputeCommandEncoder` per kernel. `.argumentBuffers` / `.icb` land in Phase 8+ if profiles justify. |
| `prewarm` | `true` | Run one no-op forward to compile the PSOs before the first user-visible decode. |
| `lazyCapabilities` | `true` | Allow runtime `enable(_:)` / `disable(_:)` after load. |
| `revision` | `"main"` | HF branch / tag / commit. |
| `cacheDirectory` | `nil` | Override the HF cache root for this load. `nil` honors `HF_HOME` then `~/.cache/huggingface/hub/`. See [§ Custom model cache path](#custom-model-cache-path). |
| `maxContextLength` | `nil` | KV-cache growth ceiling. `nil` uses the model's `max_position_embeddings`. The cache grows incrementally up to this; the over-allocation guard clamps it further so weights + max-KV + margin fit the wired-memory budget. See [kv-cache.md § Memory budget](kv-cache.md#memory-budget--the-wired-memory-ticket). |
| `preallocateKVCache` | `false` | Reserve the full context's KV memory up front instead of growing on demand. |
| `initialKVCacheCapacity` | `nil` | Override the incremental cache's starting depth (default `KVCache.defaultInitialCapacity` = 2048). |
| `wiredLimitBytes` | `nil` | Override the wired-memory budget the over-allocation guard works against. `nil` uses `recommendedMaxWorkingSetSize` (~75% of unified memory). Any value is bounded by the safe maximum = physical RAM − an 8 GB OS reserve; an explicit value above that is rejected at load (`ModelError.wiredLimitTooHigh`). |

## Downloading a model without automatic loading

`Model.load(_:options:)` lazy-downloads the checkpoint on first use, which is the right behaviour for most apps but the wrong tool when you want to **warm the cache without paying the GPU prewarm cost** — for example, a test-runner setup script that pre-fetches every checkpoint the integration suite touches, or a UI that downloads a model in the background and only mounts it into the inference engine later.

`ModelDownloader` is the bare cache-warm primitive: network + disk only, no model load, no GPU dispatch:

```swift
import FFAI

let downloader = ModelDownloader()      // shares the runtime loader's HubClient + cache

// One repo, default revision.
let snapshotURL = try await downloader.download(
    id: "mlx-community/Qwen3.5-0.8B-MLX-4bit"
)
// snapshotURL → ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-4bit/snapshots/<sha>/

// Specific revision, with progress callbacks on the main actor:
let pinnedURL = try await downloader.download(
    id: "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    revision: "v1.2",
    progressHandler: { progress in
        print("downloaded \(progress.completedUnitCount) / \(progress.totalUnitCount) bytes")
    }
)

// Verify the snapshot is already on disk without hitting the network:
do {
    _ = try await downloader.download(
        id: "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        localFilesOnly: true
    )
    // Already cached — no network call happened.
} catch {
    // Snapshot is missing or incomplete; fetch it without `localFilesOnly`.
}

// Point the cache at an external SSD (otherwise shares the loader's cache):
let onSSD = ModelDownloader(cacheDirectory: URL(fileURLWithPath: "/Volumes/Scratch/hf"))
_ = try await onSSD.download(id: "mlx-community/Qwen3-1.7B-4bit")
```

The CLI equivalent is `ffai download <repo-id> [...]` — see [using-the-cli.md § `download`](using-the-cli.md#download--pre-fetch-checkpoints-into-the-cache) for the full flag surface (multi-repo batches, `--cache`, `--continue-on-error`, `--local-files-only`). Both surfaces use the same `HubClient` under the hood, so a cache populated by either is consumed by both.

After a pre-fetch, `Model.load(...)` on the same `id` finds the snapshot already on disk and skips straight to the load + prewarm path.

## Using a custom model cache path

By default FFAI shares a snapshot cache with Python's `huggingface_hub`. The standard discovery order is:

1. **`HF_HOME` env var** — if set, the cache lives under `$HF_HOME/hub/` (or `$HF_HOME` if it's already a `hub` dir).
2. **`~/.cache/huggingface/hub/`** — the default fallback.

If you store your models somewhere else there are three ways to point FFAI to your preferred location, easiest first:

### 1. `HF_HOME` env var (CLI + library)

Cleanest for ad-hoc relocation — works for both the `ffai` CLI and any Swift code calling `Model.load(...)`. Same env var Python's `huggingface_hub` honors, so the cache stays shared with `mlx-lm`, `huggingface-cli`, etc.

```bash
export HF_HOME=/Volumes/Big/hf-cache
ffai --model mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "Once upon a time"
```

### 2. `LoadOptions.cacheDirectory` (programmatic)

Override per `Model.load(...)` call without touching the process env:

```swift
let model = try await Model.load(
    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    options: LoadOptions(
        cacheDirectory: URL(fileURLWithPath: "/Volumes/Big/hf-cache")
    )
)
```

`nil` (the default) keeps the standard `HF_HOME` → `~/.cache/...` discovery order. Useful when one process needs to read from multiple cache roots, or when you want to keep the user's normal cache untouched while a background pipeline downloads to its own location.

### 3. Fully local snapshot path

Skip HF entirely — `Model.load(...)` accepts a local directory containing the snapshot files (`config.json`, `tokenizer.json`, `*.safetensors`, etc.):

```swift
let model = try await Model.load("/Volumes/Big/models/llama-3.2-1B-snapshot")
```

```bash
ffai --model /Volumes/Big/models/llama-3.2-1B-snapshot --prompt "Once upon a time"
```

`ModelLocator.isLocalPath(_:)` decides this — anything that starts with `/`, `./`, `../`, or `~` (or just exists on disk) routes to the local-path branch and never hits the network. The directory needs at minimum:

- `config.json`
- `tokenizer.json` (or the multi-file tokenizer the model uses)
- `*.safetensors` (one or more shard files)
- `tokenizer_config.json` if you'll be using chat templates

This is also how you'd point at a snapshot you've already downloaded with `huggingface-cli download` or `mlx-lm`.

## Lifecycle events

`Model.events` is an `AsyncStream<ModelLifecycleEvent>` that emits `idle → downloading → loading → loaded → ready`, plus `failed(Error)` from any state. Useful for UI progress bars:

```swift
let model = try await Model.load("mlx-community/Qwen3.5-0.8B-MLX-4bit")
Task {
    for await event in model.events {
        print("model state: \(event.state)")
    }
}
```

## Lower level API for customizing model usage

`Model.generate(...)` is a thin wrapper. To drive the loop yourself (e.g. custom sampling, streaming hooks, multi-turn cache reuse) drop to the `LanguageModel` protocol:

```swift
let caches = model.engine.makeLayerCaches()

// Prefill: feed each prompt token through the same forward path.
var nextToken = 0
let promptTokens = model.tokenizer.encode(text: "Once upon a time")
for (i, t) in promptTokens.enumerated() {
    nextToken = model.engine.forwardSample(tokenId: t, position: i, caches: caches)
}

// Decode loop. forwardSample returns the next token id (GPU argmax) —
// no logits readback to CPU.
var pos = promptTokens.count
for _ in 0..<64 {
    if nextToken == model.config.eosTokenId { break }
    print(model.tokenizer.decode(tokens: [nextToken]), terminator: "")
    nextToken = model.engine.forwardSample(tokenId: nextToken, position: pos, caches: caches)
    pos += 1
}
```

`forward(tokenId:position:caches:)` returns the logits `Tensor` if you need them on CPU; `forwardSample` keeps them on the GPU and only returns the sampled token id (4 bytes across CPU↔GPU per token).

## Next steps

| Want to … | Read |
|---|---|
| Add FFAI to your project | [installation.md](installation.md) |
| See which models are supported | [models.md](models.md) |
| Understand the three-layer stack | [architecture.md](architecture.md) |
| Stream tokens to a UI | [streaming.md](streaming.md) |
| Use chat / instruct models | [chat-templates.md](chat-templates.md) |
| Tune sampling / prefill / `maxTokens` | [generation-parameters.md](generation-parameters.md) |
| See per-phase memory + tok/s in `--stats` | [observability.md](observability.md) |
| Run benchmarks (`ffai bench`) | [benchmarking.md](benchmarking.md) |
| Pick a KV cache strategy | [kv-cache.md](kv-cache.md) |
| Use 3 / 4 / 5 / 6 / 8-bit quantized weights | [quantization.md](quantization.md) |
| Check current `tok/s` numbers | [performance.md](performance.md) |
| Port a new architecture | [developing/adding-a-model.md](developing/adding-a-model.md) |
