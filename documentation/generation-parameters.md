# `GenerationParameters`

Every `Model.generate(...)` call is configured by a `GenerationParameters` value. Each family declares its own defaults via the family Variant protocol; the user either uses them as-is, mutates fields, or constructs their own.

## The fields

| Field | Default | Honored today | Notes |
|---|---|---|---|
| `maxTokens: Int` | `256` | ✅ | Hard cap on generated tokens. |
| `stopOnEOS: Bool` | `true` | ✅ | Stop at the model's `eosTokenId`. |
| `extraStopTokens: Set<Int>` | `[]` | ✅ | Additional stop ids beyond EOS. |
| `prefillStepSize: Int` | `1024` | ✅ | Chunk size for `engine.forwardMulti(...)` during prefill. Per-family defaults are tuned (Gemma 4 = 4096, Mamba2 / FalconH1 / Granite4 / Jamba / NemotronH = 256, …); override only if you know the family. |
| `temperature: Float` | `0.6` | ✅ | `0` → greedy (GPU argmax fast path, no logits readback). `> 0` with no filters → GPU `softmax_categorical_sample` kernel, no logits readback. `> 0` with any filter (top-K / top-P / min-P / rep-penalty) → CPU sample (one logits readback per token, ~30% decode-tok/s tax). |
| `topP: Float` | `1.0` | ✅ | Nucleus cutoff. `1.0` = disabled. Forces CPU sample path when set. |
| `topK: Int` | `0` | ✅ | `0` = disabled. Forces CPU sample path when set. |
| `minP: Float` | `0.0` | ✅ | Qwen-style min-P cutoff: keep tokens with prob ≥ `min_p × max_prob`. Forces CPU sample path. |
| `repetitionPenalty: Float` | `1.0` | ✅ | Hugging-Face convention — divide logit by penalty when seen + logit > 0; multiply when seen + logit < 0. `1.0` = disabled. Forces CPU sample path. |
| `presencePenalty: Float` | `0.0` | 🚧 planned | Additive penalty applied once a token id has been emitted at all. Field is on `GenerationParameters` today but the CPU sampler doesn't read it yet; the wiring lands alongside the remaining filtered-sampling GPU kernels. Tracked in [`planning/plan.md`](../planning/plan.md). |
| `seed: UInt64?` | `nil` | ✅ | When set, the CPU sample path is reproducible run-to-run (SplitMix64 PRNG seeded by this). Ignored on the greedy path (no RNG draw). |

Generate picks the cheapest path that produces correct output for the supplied parameters. **Sampling is GPU-resident on the fast paths** — `Ops.argmax` and `Ops.softmaxCategoricalSample` both write a single uint32 (the chosen token id) on-device, so only 4 bytes cross CPU↔GPU per decode token. Logits never leave the GPU on these paths. The CPU sample path is the fallback for filters we don't yet have GPU kernels for:

| Path | Triggered by | Cost per token |
|---|---|---|
| `greedy-GPU` | `temperature == 0`, no filters | GPU `argmax` + 4-byte readback (fastest). |
| `gpu-categorical` | `temperature > 0`, no filters | Forward + `softmax_categorical_sample` GPU kernel + 4-byte readback. Two cmdbufs today (forward + sample); per-family fusion is a follow-up. |
| `cpu-sample` | Any of `topK > 0`, `topP < 1`, `minP > 0`, `repetitionPenalty != 1` | Forward + full vocab readback (~300 KB at Qwen 3 fp16, trivial on unified memory) + `Sampling.sample(...)` pipeline. |

The GPU categorical kernel (`softmax_categorical_sample`) runs a cooperative 256-thread reduction for max + sum-exp, then a single-threaded inverse-CDF walk. The single-thread walk is the ~150µs bottleneck at vocab=152K; a parallel prefix-scan version is the natural follow-up alongside per-family `forwardSampleCategorical` fusion.

GPU top-K / top-P / min-P / rep-penalty kernels are deferred — they need a sort or radix-select, which is a substantial follow-up. Until those land, setting any filter falls back to the CPU sample path.

## Family defaults

Each family's Variant protocol declares a static `defaultGenerationParameters: GenerationParameters` that captures the values that family ships with. The `Model` instance carries the resolved value as `model.defaultGenerationParameters`.

```swift
let model = try await Model.load("mlx-community/Qwen3-4B-4bit")

print(model.defaultGenerationParameters.topP)          // 0.95   (Qwen 3)
print(model.defaultGenerationParameters.topK)          // 20     (Qwen 3)
print(model.defaultGenerationParameters.prefillStepSize) // 1024
```

Current values (pulled from each family's `defaultGenerationParameters` literal in `Sources/FFAI/Models/`):

| Family | `maxTokens` | `prefillStepSize` | `temperature` | `topP` | `topK` | `repPenalty` |
|---|---|---|---|---|---|---|
| `LlamaDense` | 256 | 1024 | 0.6 | 1.0 | 0 | 1.0 |
| `MistralDense` | 256 | 1024 | 0.6 | 1.0 | 0 | 1.0 |
| `PhiDense` | 256 | 1024 | 0.6 | 1.0 | 0 | 1.0 |
| `Qwen2Dense` | 256 | 1024 | 0.6 | 1.0 | 0 | 1.0 |
| `Qwen3Dense` | 256 | 1024 | 0.6 | 0.95 | 20 | 1.0 |
| `Qwen35*` (Hybrid / MoE / GDN) | 256 | 1024 | 0.0 | 1.0 | 0 | 1.0 |
| `Gemma2Dense` | 256 | 1024 | 1.0 | 0.95 | 64 | 1.0 |
| `Gemma3Dense` | 256 | 1024 | 1.0 | 0.95 | 64 | 1.0 |
| `Gemma4*` (Dense / E / MoE) | 256 | 4096 | 1.0 | 0.95 | 64 | 1.0 |
| `GPTOSSText` | 256 | 2048 | 0.0 | 1.0 | 0 | 1.0 |
| `LFM2Dense` / `LFM2MoE` | 256 | 1024 | 0.0 | 1.0 | 0 | 1.0 |
| `NemotronHText` | 256 | 256 | 0.0 | 1.0 | 0 | 1.0 |
| `NemotronDiffusionText` | 256 | 1024 | 0.0 | 1.0 | 0 | 1.0 |
| `FalconH1Text` | 256 | 256 | 0.0 | 1.0 | 0 | 1.0 |
| `Granite4Text` | 256 | 256 | 0.0 | 1.0 | 0 | 1.0 |
| `JambaText` | 256 | 256 | 0.0 | 1.0 | 0 | 1.0 |
| `Mamba2Text` | 256 | 256 | 0.0 | 1.0 | 0 | 1.0 |
| `Idefics3` (VL) | 256 | 1024 | 0.0 | 1.0 | 0 | 1.0 |
| `Paligemma` (VL) | 256 | 512 | 0.0 | 1.0 | 0 | 1.0 |
| `SmolVLM2` (VL) | 256 | 1024 | 0.0 | 1.0 | 0 | 1.0 |

`minP` is `0.0` for every family today (a deliberate opt-in); CLI / API callers can pass it explicitly. The GPT-OSS `reasoningLevel` defaults to `.none`. For any family not listed, read its variant struct in `Sources/FFAI/Models/Text/<F>Text.swift` (or the `Models/<F>.swift` root for VL orchestrators) — or run `ffai inspect <repo>` to dump the resolved values. The Llama / Qwen 3 defaults match mlx-swift-lm's per-family `GenerationParameters` baseline and `defaultPrefillStepSize` for the same architectures. Audio families don't ship `GenerationParameters` (they're not text generators); STT / TTS / VAD configuration lives on their own modality-specific options surface. When porting a new family, declare its defaults — see [developing/adding-a-model.md § Step 4: family defaults](developing/adding-a-model.md#step-4--declare-family-defaults).

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
- [`planning/roadmap.md`](../planning/roadmap.md) — remaining sampling work (presence-penalty wiring, GPU filtered-sampling kernels).
