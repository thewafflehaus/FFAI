# Benchmarking

`ffai bench` is the canonical perf harness — same set of methods
mlx-swift-lm exposes via its `--method` flag, same `BenchRow` schema
in the markdown + JSON sidecar so analysis tooling stays
cross-compatible.

```bash
ffai bench --method simple --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
           --prompt "Once upon a time" --max-tokens 64 --quantization bf16
```

Per-day report files land under `--report-dir` (default `./benchmarks`):

```
./benchmarks/
  apple-m1-max-2026-05-15.md           ← markdown table (regenerated each append)
  .apple-m1-max-2026-05-15.state.json  ← JSON source-of-truth
```

The JSON sidecar is the source of truth — the markdown is
re-rendered deterministically from it on every `--method` run, so
manual `.md` edits don't survive. Hack the JSON if you need to
backfill.

## Method matrix

| `--method` | Status | What it measures |
|---|---|---|
| `simple` | ✅ shipped | Single-prompt generation, throughput + memory |
| `summarization` | ✅ shipped | Fixed-shape long-prompt run (multi-context sweep lands when `--ctx <list>` ships) |
| `wikitext2` | ✅ shipped | Perplexity over WikiText-2 — uses [`Perplexity.compute(...)`](observability.md#perplexity-opt-in) |
| `niah` | 🚧 stub | Needle-in-a-haystack at multiple depths — needs sliding-window attention mask |
| `multi-turn` | 🚧 stub | Multi-turn conversation, replies fed back — needs `ChatSession`-style cache reuse helper |
| `tool-calling` | 🚧 stub | Tool call generation + validation — needs tool-spec rendering in `ChatTemplate` (Phase 8+) |
| `ngram-spot` / `ngram-sweep` / `ngram-sweep-summary` | 🚧 stub | n-gram speculative-decoding sweeps — need n-gram lookup (Phase 8+) |
| `vision` | 🚧 stub | VLM smoke test — needs vision encoder + multi-modal generate path (Phase 6) |

Stubs fail fast with the dependency name they're waiting on rather
than producing garbage:

```
$ ffai bench --method niah --model mlx-community/Qwen3.5-0.8B-MLX-4bit
Error: ffai bench --method niah: not implemented yet — needs
sliding-window attention mask + needle-position bookkeeping. Tracked
alongside its parent feature in planning/plan.md.
```

## Common flags

| Flag | Default | Notes |
|---|---|---|
| `--model <repo>` | required | HuggingFace id or local path. Same shape as `Model.load(...)`. |
| `--method <name>` | `simple` | One of the table rows above. |
| `--prompt <text>` | — | Required for prompt-based methods (`simple`, `summarization`). |
| `--max-tokens <N>` | `64` | Cap on generated tokens. |
| `--quantization <label>` | — | Free-form column value (`4bit`, `bf16`, `5bit`, …). Documents what you're benching. |
| `--report-dir <path>` | `./benchmarks` | Where the per-day report lives. |
| `--ref-model <repo>` | — | Reference model for KLD computation. See [§ KLD comparison](#kld-comparison-with---ref-model). |
| `--wikitext2-corpus <path>` | — | Required for `--method wikitext2`. Path to the `wiki.test.raw` file from the WikiText-2 dataset. |
| `--wikitext2-max-tokens <N>` | `2048` | Truncate the corpus before scoring. Trade fidelity for runtime. |
| `--debug` | off | See [observability.md § --debug](observability.md#--debug). |
| `--profiling N` | `0` | See [observability.md § --profiling N](observability.md#--profiling-n). |

## KLD comparison with `--ref-model`

KL divergence vs a reference distribution is the right way to
quantify "how much fidelity does this quantization cost vs the bf16
parent". `ffai bench --ref-model <bf16-repo> --method simple <quantized-repo>`
loads both models simultaneously and runs them in lockstep over the
same generated tokens.

```bash
ffai bench --method simple \
           --model     mlx-community/Qwen3-4B-4bit \
           --ref-model mlx-community/Qwen3-4B-bf16 \
           --prompt "Once upon a time" --max-tokens 64 \
           --quantization 4bit
```

The resulting report row's `Gen KLD` column carries the mean
per-position KL in nats. Lower is closer to the reference.

### Picking a reference model

The number is only meaningful when the reference is a higher-fidelity
sibling of the candidate — same architecture, same tokenizer.

- **Use the bf16 unquantized variant if it fits in device memory.**
  That's what makes the KLD a measure of *quantization fidelity*
  rather than *family closeness*.
- For Qwen 3 4B: `mlx-community/Qwen3-4B-bf16` (~8 GB) +
  `mlx-community/Qwen3-4B-4bit` (~2.5 GB) ≈ **10.5 GB** + a few
  hundred MB of caches. Fits on any 16 GB+ Mac.
- For Qwen 3 14B: bf16 alone is ~28 GB. If you can't fit it, drop
  to `mlx-community/Qwen3-14B-8bit` as a *higher-precision*
  reference against `Qwen3-14B-4bit` as the candidate. The number
  is still useful — just interpret it as "4bit vs 8bit" instead of
  "4bit vs ground truth".
- Cross-architecture references (e.g. comparing Qwen 3 against Llama
  3) produce noise, not signal — the tokenizers differ, so
  `KL(p_Qwen || q_Llama)` is mostly measuring tokenizer mismatch.

## Per-day report shape

The markdown re-renders from the JSON sidecar on every append:

```markdown
# FFAI Bench — apple-m1-max

- System RAM: 64.00 GB
- OS: 15.4.0
- Created: 2026-05-15T13:00:00Z

| Model | Method | Quant | Ctx | Prompt | Prefill tok/s | Decode tok/s | Steady tok/s | TTFT (ms) | Gen tokens | Baseline GPU | Peak GPU | KV used | Weights | Gen PPL | Gen KLD | Sample |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| unsloth/Llama-3.2-1B | simple | bf16 | 131072 | 5 | 60.71 | 64.06 | 65.20 | 82.34 | 64 | 1.21 GB | 1.27 GB | 4.5 MB | 1.15 GB | - | - | Once upon a time, ... |
```

Field-for-field, the row schema mirrors mlx-swift-lm's `ResultRow` so
analysis scripts (jq queries, plotting tools) work cross-repo without
modification. The full schema lives in
[`Sources/FFAI/Benchmark/BenchmarkWriter.swift`](../Sources/FFAI/Benchmark/BenchmarkWriter.swift).

## CLI streaming semantics

`ffai bench simple` consumes the generation stream silently (no
per-token `print`), then writes the row + prints a summary at the
end. This eliminates per-token I/O from the measurement — important
for short-prompt high-tok/s methods where stdout flushing would
otherwise be a meaningful fraction of decode time. `ffai generate`
does print incrementally because that's the user-facing UX.

## Method-specific notes

### `wikitext2`

Uses [`Perplexity.compute(...)`](observability.md#perplexity-opt-in).
Requires `--wikitext2-corpus </path/to/wiki.test.raw>` — the
WikiText-2 test split from
[salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext).

Cost: roughly N forward passes for N tokens (greedy is one forward
per token; perplexity captures logits at each step). Default cap of
2048 tokens runs in ~30s on Llama 3.2 1B 4-bit; bump
`--wikitext2-max-tokens` for tighter PPL estimates.

### Methods that aren't shipped yet

`niah`, `multi-turn`, `tool-calling`, `ngram-*`, and `vision` exist
as enum cases + CLI strings but throw `BenchRunnerError.notImplemented`
when invoked. The dependency each one needs is named in the error
message + the [BenchMethod.dependency](../Sources/FFAI/Benchmark/BenchMethod.swift)
enum. They ship alongside their parent features per
[`planning/plan.md`](../planning/plan.md).

## Programmatic access

The bench harness pieces are public — you can drive a method from
your own code without going through the CLI:

```swift
let model = try await Model.load("mlx-community/Qwen3-4B-4bit")
let runner = BenchRunner(model: model, modelLabel: "mlx-community/Qwen3-4B-4bit")
let row = try await runner.run(
    method: .simple,
    options: BenchOptions(prompt: "Once upon a time", maxTokens: 64,
                          quantization: "4bit")
)
let writer = BenchmarkWriter(reportDirectory: URL(fileURLWithPath: "benchmarks"))
let urls = try writer.append(row)
print(urls.markdown.path)
```

## See also

- [Observability](observability.md) — `--stats`, `--debug`,
  `--profiling`, perplexity helpers, KLD math.
- [Performance](performance.md) — current `tok/s` baseline numbers
  per model + the Phase 4 wave-by-wave perf history.
- [`planning/plan.md`](../planning/plan.md) — what's in / out of scope
  per phase, including the bench methods that ship alongside their
  parent features.
- mlx-swift-lm's [benchmarks/README.md](https://github.com/ml-explore/mlx-swift-lm/blob/main/benchmarks/README.md)
  — the upstream method matrix + report shape FFAI mirrors.
