# `GenerationParameters`

Every `Model.generate(...)` call is configured by a
`GenerationParameters` value. Each family declares its own defaults
via the family Variant protocol; the user either uses them as-is,
mutates fields, or constructs their own.

## The fields

| Field | Default | Honored today | Notes |
|---|---|---|---|
| `maxTokens: Int` | `256` | ✅ | Hard cap on generated tokens. |
| `stopOnEOS: Bool` | `true` | ✅ | Stop at the model's `eosTokenId`. |
| `extraStopTokens: Set<Int>` | `[]` | ✅ | Additional stop ids beyond EOS. |
| `prefillStepSize: Int` | `1024` | 🚧 Phase 5+ | Honored once chunked prefill ships; today's per-token prefill ignores it. |
| `temperature: Float` | `0.6` | ✅ | `0` → greedy (GPU argmax fast path, no logits readback). `> 0` → CPU sample (one logits readback per token, ~30% decode-tok/s tax). |
| `topP: Float` | `1.0` | ✅ | Nucleus cutoff. `1.0` = disabled. Forces CPU sample path when set. |
| `topK: Int` | `0` | ✅ | `0` = disabled. Forces CPU sample path when set. |
| `minP: Float` | `0.0` | ✅ | Qwen-style min-P cutoff: keep tokens with prob ≥ `min_p × max_prob`. Forces CPU sample path. |
| `repetitionPenalty: Float` | `1.0` | ✅ | Hugging-Face convention — divide logit by penalty when seen + logit > 0; multiply when seen + logit < 0. `1.0` = disabled. Forces CPU sample path. |
| `presencePenalty: Float` | `0.0` | 🚧 Phase 5+ | Additive. `0` = disabled. |
| `seed: UInt64?` | `nil` | ✅ | When set, the CPU sample path is reproducible run-to-run (SplitMix64 PRNG seeded by this). Ignored on the greedy path (no RNG draw). |

Greedy (the family defaults' `temperature: 0`) stays on the
existing GPU argmax fast path — only 4 bytes cross CPU↔GPU per
token. Setting any sampling knob (`temperature > 0`, `topK > 0`,
`topP < 1`, `minP > 0`, `repetitionPenalty != 1`) switches to the
CPU sample path: the forward pass returns logits to CPU, then
`Sampling.sample(...)` runs the rep-penalty → temperature →
top-K → top-P → min-P → categorical pipeline. Costs one
vocab-sized readback per token (~300 KB on Qwen3 vocab=151_936 fp16,
trivial on unified memory) and ~30% of decode tok/s. A GPU sample
kernel is on the metaltile `ek/sampling-kernels` branch for the
fully-on-GPU path later.

## Family defaults

Each family's Variant protocol declares a static
`defaultGenerationParameters: GenerationParameters` that captures the
values that family ships with. The `Model` instance carries the
resolved value as `model.defaultGenerationParameters`.

```swift
let model = try await Model.load("mlx-community/Qwen3-4B-4bit")

print(model.defaultGenerationParameters.topP)          // 0.95   (Qwen 3)
print(model.defaultGenerationParameters.topK)          // 20     (Qwen 3)
print(model.defaultGenerationParameters.prefillStepSize) // 1024
```

Current values:

| Family | `temperature` | `topP` | `topK` | `minP` | `repPenalty` | `prefillStepSize` | `maxTokens` |
|---|---|---|---|---|---|---|---|
| `LlamaDense` | 0.6 | 1.0 | 0 | 0.0 | 1.0 | 1024 | 256 |
| `Qwen3Dense` | 0.6 | 0.95 | 20 | 0.0 | 1.0 | 1024 | 256 |

These match mlx-swift-lm's per-family `GenerationParameters` baseline
and `defaultPrefillStepSize` for the same architectures. As new
families land (Qwen 3.5 hybrid, Qwen 3.5 MoE, Mistral, Phi, Gemma,
etc.) they declare their own defaults — see
[developing/adding-a-model.md § Step 4: family defaults](developing/adding-a-model.md#step-4--declare-family-defaults).

## Three ways to call `generate`

### 1. Family default

```swift
let result = try await model.generate(prompt: "Once upon a time")
```

`parameters` defaults to `nil`, which falls back to
`model.defaultGenerationParameters`.

### 2. Override one field

The `with(_:)` copy-mutator keeps the family-tuned baseline and
edits a single knob:

```swift
let result = try await model.generate(
    prompt: "Once upon a time",
    parameters: model.defaultGenerationParameters.with { $0.maxTokens = 64 }
)
```

This is the recommended call shape — you don't lose the family-tuned
sampling values just because you wanted a shorter generation.

### 3. Custom from scratch

```swift
let params = GenerationParameters(
    maxTokens: 1024,
    temperature: 0.0,        // greedy
    topP: 1.0,
    repetitionPenalty: 1.05
)
let result = try await model.generate(prompt: "...", parameters: params)
```

Any field you don't pass picks the `GenerationParameters.init` default,
not the family default. Use `with(_:)` (#2) when you want the
family-tuned baseline.

## CLI behaviour

Each CLI flag overrides only its corresponding `GenerationParameters`
field; every other knob still picks up the family default. Omit all
sampling flags to use the family value:

```bash
# Family defaults — Qwen 3 ships temperature=0.6 / top-p=0.95 / top-k=20.
# Routes through the CPU sample path (non-greedy):
ffai --model mlx-community/Qwen3-1.7B-4bit --prompt "Hello"

# Greedy fast path (GPU argmax, 4-byte readback per token, no sampling):
ffai --model mlx-community/Qwen3-1.7B-4bit --prompt "Hello" \
     --temperature 0 --max-tokens 64

# Seeded non-greedy — reproducible run-to-run:
ffai --model mlx-community/Qwen3-1.7B-4bit --prompt "Hello" \
     --temperature 0.7 --top-p 0.9 --seed 42 --max-tokens 64

# Full sampling pipeline:
ffai --model mlx-community/Qwen3-1.7B-4bit --prompt "Hello" \
     --temperature 0.8 --top-k 40 --top-p 0.95 --min-p 0.05 \
     --repetition-penalty 1.05 --seed 12345
```

Available flags: `--temperature`, `--top-k`, `--top-p`, `--min-p`,
`--repetition-penalty`, `--seed`. Plus `--max-tokens`.

## See also

- [Quick start](quickstart.md) — basic `Model.load` + `generate`
  flow.
- [Models](models.md) — supported families and which defaults each
  carries.
- [developing/adding-a-model.md](developing/adding-a-model.md) — how
  to declare defaults when porting a new family.
- [`planning/roadmap.md`](../planning/roadmap.md) — Phase 5 sampling
  + chunked prefill commitment.
