# Observability — `--stats`, `--debug`, `--profiling`

Three CLI flags to introspect what FFAI is doing per token. The underlying types live in [`Sources/FFAI/Stats/`](../Sources/FFAI/Stats/) and are usable from your own code without going through the CLI.

This page covers `--stats` today; `--debug` and `--profiling` land in follow-up commits and are described under [Coming next](#coming-next).

## `--stats`

Prints a `[STATS]` block after generation completes. Always-on instrumentation: a `PhaseMemoryTracker` samples `MTLDevice.currentAllocatedSize` at every token boundary (cost: one property read per token, ~sub-µs). The fields and shape are deliberately portable to / from mlx-swift-lm's bench-row schema so analysis tooling stays compatible.

```bash
ffai --model mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "Once upon a time" --stats
```

```
[STATS]
  prompt:           5 tokens
  generated:        64 tokens
  context:          131072 tokens
  ttft:             82.34 ms
  prefill:          0.08s (60.71 tok/s)
  decode:           1.00s (64.06 tok/s)
  decode (steady):  65.20 tok/s   (tokens 11+)
  baseline GPU:     1240.0 MB
  post-prefill GPU: 1252.0 MB     (+ 12.0 MB)
  post-decode  GPU: 1268.0 MB     (+ 16.0 MB)
  prefill peak:     1255.5 MB
  decode  peak:     1269.0 MB
  weights:          1180.0 MB
  KV cache (alloc): 64.0 MB
  KV cache (used):  4.5 MB
  wired ticket:     16.00 GB
```

### What each field means

| Field | Source | Notes |
|---|---|---|
| `prompt` / `generated` / `context` | counts | `context` is the model's max sequence length. |
| `ttft` | wall clock | Time-to-first-token (ms). On the slow per-token prefill, equal to total prefill time. |
| `prefill`, `decode` | wall clock | Phase totals + tok/s. |
| `decode (steady)` | wall clock | Mean per-token decode rate from token 11 onward. Drops PSO compile + autorelease pool warm-up. `nil` when fewer than 11 generated tokens. |
| `baseline GPU` | `MTLDevice.currentAllocatedSize` at the start of `generate(...)` | Resident weights + warm caches. |
| `post-prefill GPU`, `post-decode GPU` | same source | Includes a `+ delta` column attributing the growth to the corresponding phase — answers *"where did the GB come from?"*. |
| `prefill peak`, `decode peak` | per-token sample max | The actual high-water mark inside each phase, separately reported. |
| `weights` | sum of `Tensor.byteCount` over `engine.parameters()` | Resident model weights only. |
| `KV cache (alloc)` | sum of `KVCache.bytesAllocated` | Capacity allocated up-front at the model's `maxSeq`. |
| `KV cache (used)` | sum of `KVCache.bytesInUse` | The live `length / maxSeq` slice — the part you're actually paying for in attention math. |
| `wired ticket` | `MTLDevice.recommendedMaxWorkingSetSize` | The OS's wired-memory budget for this device. |

### Programmatic access

Stats are always populated on `GenerationResult.stats`. The CLI flag only controls the printout.

```swift
let result = try await model.generate(prompt: "...")

print(result.stats.peakGPUBytes)
print(result.stats.prefillGrowthBytes)   // post-prefill – baseline
print(result.stats.decodeGrowthBytes)    // post-decode  – post-prefill
print(result.stats.kvCacheUsedBytes)
print(result.stats.steadyTokensPerSecond ?? "<warm-up only>")
```

For a custom report, call `result.stats.formatted()` — that's what the CLI prints.

### Perplexity (opt-in)

Perplexity computation requires a per-step logits readback (extra cost on top of greedy decode), so it's not folded into `generate(...)`. Use the standalone helper:

```swift
let r = try await model.generate(prompt: "Once upon a time")
let ppl = Perplexity.compute(model: model,
                             tokens: r.promptTokens + r.generatedTokens)
print(ppl.perplexity)   // exp(-mean log p(token | prefix))
```

All perplexity math runs in **fp32** — bf16's 7 mantissa bits can't represent the partial `Σ exp(x_i − max_x)` over a 128–152K vocab without losing precision in the third decimal of perplexity, which is exactly the resolution we need for distinguishing quantization tiers. fp32 costs ~1µs per token; only matters offline. Header in [`Stats/Perplexity.swift`](../Sources/FFAI/Stats/Perplexity.swift) has the longer rationale.

The bench harness's `--method wikitext2` runs this over the WikiText-2 corpus and writes the resulting `genPerplexity` into the report row.

### KL divergence vs a reference model

`Perplexity.klDivergence(reference:candidate:tokens:)` runs both models forward in lockstep over a fixed token sequence and returns the mean per-position `KL(p_ref || q_cand)` in nats. Both models must share the same tokenizer / vocab.

```swift
let reference = try await Model.load("mlx-community/Qwen3-4B-bf16")
let candidate = try await Model.load("mlx-community/Qwen3-4B-4bit")
let kld = Perplexity.klDivergence(
    reference: reference, candidate: candidate,
    tokens: tokenizer.encode(text: "Sample evaluation text…")
)
print(kld.meanKLDivergence)
```

**Pick the right reference.** The number is only meaningful when the reference is a higher-fidelity sibling of the candidate — same architecture, same tokenizer. **Use the bf16 unquantized variant if it fits in device memory.** A smaller / different-family reference turns the KL into a measure of family closeness rather than quantization fidelity.

Memory: both models live resident simultaneously, plus their KV caches. For Qwen 3 4B, that's roughly 8GB (bf16) + 2.5GB (4-bit) + a few hundred MB of caches — fits on any 16GB+ Mac. For Qwen 3 14B the bf16 variant alone is ~28GB; pair with `Qwen3-14B-8bit` as the candidate against `Qwen3-14B-bf16` as the reference if you have the RAM, otherwise drop down to the next size class.

CLI `--ref-model <repo>` plumbing ships with the bench subcommand in the next chunk.

### Thinking vs Generation Split

Models that emit reasoning segments (Qwen 3 / DeepSeek-R1 ChatML, GPT-OSS Harmony, Gemma 3/4 channels) get a separate `think_tokens / gen_tokens` count line. Format auto-detection runs on every `generate(...)` call via `ThinkingSplit.detectFormat(model:)`:

| Detected from | Format | Implementation |
|---|---|---|
| Tokenizer has `<think>` / `</think>` ids | `.chatML` | ✅ Implemented (Qwen 3, DeepSeek-R1 distills). |
| `model_type` contains `gpt-oss` | `.harmony` | 🚧 TODO — scanner reaches for the multi-token `<\|channel\|>` `analysis` / `final` `<\|message\|>` subsequence in the Harmony tokenizer; format is detected today but `splitHarmony` returns `nil`. |
| `model_type` contains `gemma3` / `gemma4` | `.gemmaChannel` | 🚧 TODO — scanner reaches for the `<channel\|reasoning\|>` / `<channel\|final\|>` markers; format is detected today but `splitGemmaChannel` returns `nil`. |
| Otherwise | `.none` | Whole generation counts as gen. |

For models that don't emit thinking markers, the split is silently omitted. See [`Stats/ThinkingSplit.swift`](../Sources/FFAI/Stats/ThinkingSplit.swift) for the per-format scanners. Wiring the two TODO scanners is tracked in [`planning/session-plan.md`](../planning/session-plan.md).

Per-segment perplexity is `nil` from `generate(...)` alone — the bench harness will run perplexity over each segment separately when it lands. For now you can do it yourself:

```swift
let split = ThinkingSplit.split(tokens: result.generatedTokens, model: model)
if let s = split {
    let thinkPPL = Perplexity.compute(model: model,
                                      tokens: result.promptTokens + Array(s.thinkTokens))
    let genPPL   = Perplexity.compute(model: model,
                                      tokens: result.promptTokens + Array(s.thinkTokens)
                                              + [open] + Array(s.genTokens))
    print(thinkPPL.perplexity, genPPL.perplexity)
}
```

### Scaffolded fields (commented today)

The `GenerationStats` struct reserves space — currently commented — for capabilities that haven't shipped yet:

- **Batch decoding**: `batchSize`, `perSequenceDecodeTokensPerSecond`.
- **Speculative decoding**: `acceptanceRate`, `draftTokensPerSecond`, `draftAcceptedTokens`.

When those modes land, uncomment the fields, populate them in `Generate.swift`, and the formatted output + bench writer pick them up automatically.

## `--debug`

Gates verbose log output to **stderr**, tagged by subsystem. Off by default. The CLI flag flips the global gate; the same gate is controllable via env vars so other entry points (Xcode test runs, your own Swift code) can opt in without editing source.

```bash
ffai --debug --model mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "Hi" 2>debug.log
```

Sample output (heavily abbreviated):

```
[ffai:load] Model.load id-or-path=mlx-community/Qwen3.5-0.8B-MLX-4bit
[ffai:loader] resolved snapshot dir: /Users/.../snapshots/abcd…
[ffai:load] config: arch=LlamaForCausalLM model_type=llama hidden=2048 layers=16
[ffai:generate] begin prefill: 5 tokens, maxTokens=64
[ffai:generate] prefill done in 0.082s (60.7 tok/s)
[ffai:generate] decode done: 64 tokens in 1.000s (64.0 tok/s)
```

### Per-subsystem gates

`--debug` enables every subsystem. To enable just one, set the matching env var instead of passing `--debug`:

| Env var | Subsystem | What it logs |
|---|---|---|
| `FFAI_DEBUG=1` | (all) | Global gate; every subsystem on. |
| `FFAI_DEBUG_LOADER=1` | `loader` | `ModelLocator` / `ModelDownloader` snapshot resolution + cache hits. |
| `FFAI_DEBUG_LOAD=1` | `load` | `Model.load` + family loaders, config decode. |
| `FFAI_DEBUG_GENERATE=1` | `generate` | Prefill + decode loop boundaries, per-call timing summary. |
| `FFAI_DEBUG_SAMPLING=1` | `sampling` | Sampling decisions — greedy-GPU / GPU-categorical / CPU-sample path choice + the resulting token. |
| `FFAI_DEBUG_KVCACHE=1` | `kvcache` | KV cache append + slice events. |
| `FFAI_DEBUG_KERNELS=1` | `kernels` | Per-kernel dispatch chatter — very loud, opt-in only. |
| `FFAI_DEBUG_DISPATCH=1` | `dispatch` | Per-`MTLCommandBuffer` commit / wait. |
| `FFAI_DEBUG_BENCH=1` | `bench` | `BenchRunner` method dispatch + sub-phase timing. |

The closure passed to `Debug.log(...)` is `@autoclosure` — when the subsystem is off, the message string isn't built and the call is near-free. Safe to leave instrumentation in hot paths.

```swift
Debug.log(.kvcache, "append k+v at pos \(pos), live bytes=\(caches.totalBytesInUse)")
```

## `--profiling N`

Three levels. `0` is the default (off, zero overhead anywhere).

```bash
ffai --profiling 1 --model mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "Hi"
```

### Level 1 — wallclock breakdown

Captures wallclock durations at every phase boundary (`model_load`, `prewarm`, `prefill`, `ttft`, `decode`, `generation_total`) and prints a `[PROFILE]` block at the end of the run. Cost is a few `Date()` calls per generation — negligible.

```
[PROFILE]
  model_load            2.34 s
  prewarm             480.12 ms
  prefill              82.30 ms
  ttft                 82.30 ms
  decode                1.00 s
  generation_total      1.08 s
```

### Level 2 — level 1 + `os_signpost` intervals

Wraps each phase + the inner decode-step loop in [`OSSignposter`](https://developer.apple.com/documentation/os/ossignposter) intervals under subsystem `ai.ffai`, capturable by Instruments (*Profile → Logging → os_signpost*) or `xctrace record`.

**Zero overhead when no tracer is attached** — `OSSignposter` checks a flag at the start of each `beginInterval(...)` and bails before constructing any state. ~40 ns per call when nobody's listening.

```bash
xctrace record --template 'os_signpost' --launch -- \
    .build/release/ffai --profiling 2 --model mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "Hi"
```

The recorded trace shows FFAI's `prefill`, `decode_step`, `prewarm`, `model_load` spans on the same timeline as Apple's Metal subsystem spans (`com.apple.Metal`). Each Metal kernel dispatch shows up automatically there — no need to wrap individual kernels on our side. Use Instruments' Metal System Trace template to drill into GPU vs CPU time per phase.

### Pattern for new call sites

Wrapping a call site in a signpost is a one-liner — passes through the body's value, no signpost overhead at level 0/1:

```swift
Profile.signpost("MyOp.compute") {
    // existing implementation
}

// Async variant
await Profile.signpostAsync("loader.fetch") {
    try await fetchSnapshot(...)
}

// Wallclock-only timing (level 1)
let result = Profile.time("custom_phase") {
    expensiveSyncWork()
}

// Point-in-time event (no duration)
Profile.event("first_decode_token")
```

The top-level CLI sets `Profile.shared.level` once before any FFAI work runs; the same global is used by the bench subcommand so trace recordings line up across `ffai generate` and `ffai bench`.

### Programmatic access

```swift
Profile.shared.level = .signposts
Profile.shared.resetPhases()

let result = try await model.generate(prompt: "...")

print(Profile.shared.phases.formatted())   // [PROFILE] block
```

`PhaseTimings` is a `Sendable` struct — safe to grab a snapshot from any context.

## See also

- [`generation-parameters.md`](generation-parameters.md) — the knobs that control what gets generated; stats describes how it ran.
- [`performance.md`](performance.md) — current `tok/s` numbers per model + the wave-by-wave perf history.
- [`kv-cache.md`](kv-cache.md) — what the `KV cache (alloc)` and `(used)` fields actually account for.
