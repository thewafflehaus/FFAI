# `GenerationParameters`

Every `Model.generate(...)` call is configured by a `GenerationParameters` value. Each family declares its own defaults via the family Variant protocol; the user either uses them as-is, mutates fields, or constructs their own.

## The fields

| Field | Default | Honored today | Notes |
|---|---|---|---|
| `maxTokens: Int` | `256` | ✅ | Hard cap on generated tokens. |
| `stopOnEOS: Bool` | `true` | ✅ | Stop at the model's `eosTokenId`. |
| `extraStopTokens: Set<Int>` | `[]` | ✅ | Additional stop ids beyond EOS. |
| `prefillStepSize: Int` | `1024` | 🚧 Phase 5+ | Honored once chunked prefill ships; today's per-token prefill ignores it. |
| `temperature: Float` | `0.6` | ✅ | `0` → greedy (GPU argmax fast path, no logits readback). `> 0` with no filters → GPU `softmax_categorical_sample` kernel, no logits readback. `> 0` with any filter (top-K / top-P / min-P / rep-penalty) → CPU sample (one logits readback per token, ~30% decode-tok/s tax). |
| `topP: Float` | `1.0` | ✅ | Nucleus cutoff. `1.0` = disabled. Forces CPU sample path when set. |
| `topK: Int` | `0` | ✅ | `0` = disabled. Forces CPU sample path when set. |
| `minP: Float` | `0.0` | ✅ | Qwen-style min-P cutoff: keep tokens with prob ≥ `min_p × max_prob`. Forces CPU sample path. |
| `repetitionPenalty: Float` | `1.0` | ✅ | Hugging-Face convention — divide logit by penalty when seen + logit > 0; multiply when seen + logit < 0. `1.0` = disabled. Forces CPU sample path. |
| `presencePenalty: Float` | `0.0` | 🚧 Phase 5+ | Additive. `0` = disabled. |
| `seed: UInt64?` | `nil` | ✅ | When set, the CPU sample path is reproducible run-to-run (SplitMix64 PRNG seeded by this). Ignored on the greedy path (no RNG draw). |

Generate picks the cheapest path that produces correct output for the supplied parameters:

| Path | Triggered by | Cost per token |
|---|---|---|
| `greedy-GPU` | `temperature == 0`, no filters | GPU argmax + 4-byte readback (fastest). |
| `gpu-categorical` | `temperature > 0`, no filters | Forward + `softmax_categorical_sample` GPU kernel + 4-byte readback. Two cmdbufs today (forward + sample); per-family fusion is a follow-up. |
| `cpu-sample` | Any of `topK > 0`, `topP < 1`, `minP > 0`, `repetitionPenalty != 1` | Forward + full vocab readback (~300 KB at Qwen 3 fp16, trivial on unified memory) + `Sampling.sample(...)` pipeline. |

The GPU categorical kernel itself (`softmax_categorical_sample`) lives in metaltile on the `ek/sampling-kernels` branch — cooperative 256-thread reduction for max + sum-exp, then a single-threaded inverse-CDF walk. The single-thread walk is the ~150µs bottleneck at vocab=152K; a parallel prefix-scan version is the natural follow-up alongside per-family `forwardSampleCategorical` fusion.

GPU top-K / top-P / min-P / rep-penalty kernels are deferred — they need a sort or radix-select, which is a substantial follow-up. Until those land, setting any filter falls back to the CPU sample path.

## Family defaults

Each family's Variant protocol declares a static `defaultGenerationParameters: GenerationParameters` that captures the values that family ships with. The `Model` instance carries the resolved value as `model.defaultGenerationParameters`.

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
| `LFM2Dense` / `LFM2MoE` | 0.0 | 1.0 | 0 | 0.0 | 1.0 | 1024 | 256 |

The full model family set ships today — dense text, hybrid (SSM / GDN / conv+attention), MoE, vision-language, and audio families have all landed, and each variant struct declares its own `defaultGenerationParameters`. The values above are the ones whose exact defaults are spelled out here; for any other family, read its variant struct in `Sources/FFAI/Models/<Family>.swift` or run `ffai inspect <repo>` to see the resolved values. The Llama / Qwen 3 defaults match mlx-swift-lm's per-family `GenerationParameters` baseline and `defaultPrefillStepSize` for the same architectures. When porting a new family, declare its defaults — see [developing/adding-a-model.md § Step 4: family defaults](developing/adding-a-model.md#step-4--declare-family-defaults).

## Three ways to call `generate`

### 1. Family default

```swift
let result = try await model.generate(prompt: "Once upon a time")
```

`parameters` defaults to `nil`, which falls back to `model.defaultGenerationParameters`.

### 2. Override one field

The `with(_:)` copy-mutator keeps the family-tuned baseline and edits a single knob:

```swift
let result = try await model.generate(
    prompt: "Once upon a time",
    parameters: model.defaultGenerationParameters.with { $0.maxTokens = 64 }
)
```

This is the recommended call shape — you don't lose the family-tuned sampling values just because you wanted a shorter generation.

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

Any field you don't pass picks the `GenerationParameters.init` default, not the family default. Use `with(_:)` (#2) when you want the family-tuned baseline.

## CLI behaviour

Each CLI flag overrides only its corresponding `GenerationParameters` field; every other knob still picks up the family default. Omit all sampling flags to use the family value:

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

Available flags: `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--repetition-penalty`, `--seed`. Plus `--max-tokens`.

## See also

- [Quick start](quickstart.md) — basic `Model.load` + `generate` flow.
- [Models](models.md) — supported families and which defaults each carries.
- [developing/adding-a-model.md](developing/adding-a-model.md) — how to declare defaults when porting a new family.
- [`planning/roadmap.md`](../planning/roadmap.md) — Phase 5 sampling
  + chunked prefill commitment.
