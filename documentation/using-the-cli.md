# Using the CLI

The `ffai` executable is a SwiftPM product, not a Homebrew formula —
there's no global install step. After cloning the repo, build it with
`swift build` and invoke it through SwiftPM (`swift run ffai …`), the
built binary path, or by symlinking onto `PATH`.

## Build

```bash
git clone https://github.com/thewafflehaus/FFAI
cd FFAI

swift build -c release        # binary lands at .build/release/ffai
```

Use `-c debug` (the SwiftPM default) for faster compile + slower run;
`-c release` for the inference numbers you'd quote.

## Run

Pick one of three invocations — they're equivalent, just trade-offs
on ergonomics.

```bash
# (a) Via SwiftPM — no setup, recompiles if the source changed.
swift run -c release ffai generate -m unsloth/Llama-3.2-1B -p "Once upon a time"

# (b) Direct binary path — no recompile check, fastest start-up.
.build/release/ffai generate -m unsloth/Llama-3.2-1B -p "Once upon a time"

# (c) Symlink onto PATH (one-time) so plain `ffai …` works from anywhere.
ln -s "$PWD/.build/release/ffai" /usr/local/bin/ffai
ffai generate -m unsloth/Llama-3.2-1B -p "Once upon a time"
```

`generate` is the default subcommand, so the `-m / -p` flags can be
passed directly to `ffai` (`ffai -m … -p …` is equivalent to
`ffai generate -m … -p …`).

## Subcommands

| Subcommand | One-liner | More |
|---|---|---|
| `generate` (default) | Stream a single prompt's continuation to stdout. | `ffai generate --help` |
| `models` | List every supported model family with copy-paste example repo IDs (bf16 / 8-bit / 4-bit). | `ffai models` |
| `inspect` | Load a model and dump architecture + tokenization + top-K logits for a fixed probe prompt. The first thing to reach for when a new model produces broken output. | `ffai inspect --help` |
| `convert` | Quantize a bf16/fp16 HuggingFace checkpoint to MLX 4-bit affine format using FFAI's own GPU kernels — no Python / mlx-lm dependency. Optionally upload to HF. | [§ `convert`](#convert--quantize-a-checkpoint-to-mlx-4-bit) |
| `bench` | Run a benchmark method against a model, append to a per-day report. | [benchmarking.md](benchmarking.md) |

### `models` — what can I run?

`ffai models` prints every supported architecture family grouped by
kind (dense / MoE / SSM-GDN hybrid / diffusion), each with a one-line
summary and a few example HuggingFace repo IDs you can paste straight
into `generate` or `bench`:

```bash
ffai models
# ── Dense text ──────────────────────────────────────
#   Qwen 3  [qwen3]
#     Qwen 3 dense — per-head q/k RMSNorm before RoPE.
#     • mlx-community/Qwen3-1.7B-bf16
#     • mlx-community/Qwen3-1.7B-8bit
#     • mlx-community/Qwen3-1.7B-4bit
#   …
```

The example IDs include bf16, 8-bit, and 4-bit conversions where
published. Any mlx-format 3/4/5/6/8-bit conversion of a listed
architecture also loads — the IDs are just convenient starting points.
For sizes exercised + known gaps, see [models.md](models.md).

### `inspect` — model bring-up diagnostic

When a new model checkpoint isn't producing coherent text, run
`ffai inspect <repo>` before anything else. The output is structured
in six sections:

1. **Architecture** — family + dtype + every shape the loader inferred from
   `config.json` (hidden / nLayers / nHeads / nKVHeads / head_dim / vocab
   / max_position_embeddings). Tells you instantly whether the loader
   parsed the right config.
2. **Capabilities** — what the family declares it can do vs what
   `LoadOptions` enabled.
3. **Tokenizer** — per-token decode of a fixed prompt. Catches
   tokenization regressions (wrong special-token IDs, missing merges,
   model-vs-tokenizer mismatch) at a glance.
4. **KV cache** — bytes allocated, per-layer stride, eviction policy.
   For Gemma 3 / GPT-OSS, per-layer eviction shows up here.
5. **Special tokens** — BOS / EOS / chat-turn / reasoning / tool-
   calling / multimodal / utility tokens parsed from
   `tokenizer_config.json` and bucketed by role. Shows the
   `chat_template` markers when present. This is the first thing
   to look at when a chat-template / tool-calling / multi-turn
   loop is misbehaving:

   ```
   │ EOS / end-of-turn 128001 "<|end_of_text|>", 128008 "<|eom_id|>", 128009 "<|eot_id|>"
   │ Chat turn         128006 "<|start_header_id|>", 128007 "<|end_header_id|>"
   │ Reasoning         151667 "<think>", 151668 "</think>"
   │ Tool calling      151657 "<tool_call>", 151658 "</tool_call>", …
   │ chat_template     present — mentions: <|im_start|>, <|im_end|>, <think>, <tool_call>
   ```

6. **Top-K next-token logits** — runs prefill and prints the K
   most-likely continuations of the probe prompt. NaN logits get
   flagged with a debug-checklist hint; values are model-comparable
   (e.g. for `Once upon a time, in a quiet` you want to see `" village"`,
   `" little"`, `" forest"`, `" valley"`, not `"<pad>"`).

```bash
ffai inspect -m mlx-community/gemma-3-1b-it-bf16 -p "Once upon a time, in a quiet"
# → top-5: " village" +34.0, " little" +31.25, " valley" +29.88, …
```

Pair with `--debug` (per-subsystem trace dump) and `--profiling 1`
(wallclock breakdown) for full visibility into where a problem hides.

#### `--layer-trace` — per-layer intermediate-value dumps

For deeper triage when the top-K logits come back NaN / corrupted,
add `--layer-trace` and (optionally) `--trace-layers N,M,...` to
print min/max/nan/inf/first-4 statistics at every layer boundary
during prefill. Each model's `forward(...)` calls `InspectTap`
(`Sources/FFAI/Inspect/InspectTap.swift`) at the layer-out
boundary, so the trace is uniform across families:

```bash
ffai inspect -m <broken-model> --layer-trace --trace-layers 0,1,5,15
# [L0 layer_out] n=1152 min=-1.52 max=+1.55 nan=0 inf=0 first=[...]
# [L1 layer_out] n=1152 min=— max=— nan=1152 inf=0 first=[nan, nan, …]
# ↑ Layer 1's output is all NaN — the bug is in layer 1's forward.
```

This is the diagnostic that found the Gemma 3 bf16 GELU NaN in two
runs (one to localise the failing layer, one to confirm the fix).
The taps are zero-cost when the flag isn't set — they're a single
`if active` compare on the hot path. To wire them into a new model
family, see [`developing/adding-a-model.md` § Inspect
hooks](developing/adding-a-model.md#step-7--inspect-hooks).

Common cross-cutting flags (`--stats`, `--debug`, `--profiling`) are
documented in [observability.md](observability.md).

### `convert` — quantize a checkpoint to MLX 4-bit

`ffai convert` quantizes a bf16/fp16 HuggingFace checkpoint to MLX
affine-quantized format (the same `.weight` + `.scales` + `.biases`
triplet layout that `mlx-community/*-4bit` checkpoints use) and writes
the result as a drop-in directory `Model.load(...)` can consume. The
quantize work runs through FFAI's own `QuantizedOps.quantizeAffine`
GPU kernel — there is **no dependency on Python, `mlx-lm`, or
`mlx-vlm`** at conversion time.

```bash
# Pull a bf16 repo, quantize to 4-bit, drop the result in
# ~/.cache/ffai/converts/HuggingFaceTB--SmolLM2-360M-Instruct-4bit/.
ffai convert HuggingFaceTB/SmolLM2-360M-Instruct

# Convert + upload to a HF repo you control (requires `hf` CLI
# authenticated — `hf auth login`).
ffai convert HuggingFaceTB/SmolLM2-360M-Instruct \
    --upload-repo ekryski/SmolLM2-360M-Instruct-4bit

# Convert a model already on disk (e.g. a local fine-tune).
ffai convert /path/to/my-finetune --output /path/to/my-finetune-4bit

# 8-bit instead of 4-bit (also runs through QuantizedOps.quantizeAffine).
ffai convert mlx-community/gemma-3-1b-it-bf16 --bits 8
```

End-to-end on the SmolLM2-360M case: download → quantize → write →
upload measures at **~1.4 seconds**.

#### Flags

| Flag | Default | Meaning |
|---|---|---|
| `<source>` (positional) | — | HF repo id (`org/repo`) or local directory path. Local paths must start with `/`, `./`, `../`, or `~`. |
| `-b` / `--bits` | `4` | Bits per weight. Currently `4`. Other widths (`2`, `8`) are kernel-level supported but should be validated per-model before relying on them. |
| `--output` | `~/.cache/ffai/converts/<safe-name>-<bits>bit` | Destination directory. Created if missing; overwrites if present. |
| `--upload-repo` | — | After convert, shell out to `hf upload <repo> <output-dir>`. Requires the `hf` CLI authenticated with write access to `<repo>`. The local output is kept regardless of upload outcome. |
| `--quantize-embeddings` | off | Also quantize `embed_tokens.weight`. Off by default to match mlx-lm convention; turn on for further size reduction at a small quality cost. |
| `--quantize-lm-head` | off | Also quantize `lm_head.weight`. Off by default; mlx-lm skips this because most models tie `lm_head` to `embed_tokens` and quantizing the tied pair compounds error. |
| `--revision` | `main` | HF revision (branch / tag / commit) to download. |

#### What gets quantized vs copied through

- **Quantized**: every 2D weight matrix whose name ends in `.weight`
  whose last dim divides `64` (the kernel's group_size constraint) and
  isn't one of `embed_tokens.weight` / `lm_head.weight` (unless the
  matching `--quantize-…` flag is set). The triplet
  `name.weight` (packed u32) + `name.scales` (bf16) + `name.biases`
  (bf16) is written to the output.
- **Copied unchanged**: 1D norms, conv1d kernels, biases, RoPE
  `inv_freq` tables, anything that isn't a Linear-shaped 2D weight.
- **Patched**: `config.json` gets `quantization` (mlx-lm convention)
  and `quantization_config` (transformers convention) blocks added,
  both with `{bits, group_size: 64, mode: "affine"}`. Other config
  keys are preserved. Non-finite numbers (Python `json.allow_nan=True`
  artifacts — e.g. NemotronH's `time_step_limit: [0.0, Infinity]`)
  are sanitized to JSON-legal sentinels (`1e308` / `NSNull`) so
  `JSONSerialization` can re-encode the dict.
- **Copied alongside**: `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, `chat_template.jinja`, `tokenizer.model`,
  `vocab.txt`, `merges.txt`, and any other top-level `*.json` /
  `*.txt` file in the source. HF Hub snapshot directories store these
  as relative symlinks into a `blobs/` store — the convert resolves
  the symlinks before copying so the destination is self-contained.

#### When `ffai convert` succeeds where `mlx_lm.convert` / `mlx_vlm.convert` fails

`mlx-lm` and `mlx-vlm` import the source model via `AutoConfig` /
`AutoModel`, which triggers Python's full transformers + custom
`modeling_*.py` import chain for the family. That fails for several
architectures FFAI loads natively:

| Model | mlx-lm error | `ffai convert` |
|---|---|---|
| Soprano-1.1-80M | `Model type 'soprano' not supported` | ✅ — `ekryski/Soprano-1.1-80M-4bit` |
| Nemotron-H-4B-Base-8K | Mamba GQA `q_proj.weight` shape `(3072, 3072)` vs `(4096, 3072)` | ✅ — `ekryski/Nemotron-H-4B-Base-8K-4bit` |
| FastVLM-0.5B (Apple) | metaclass conflict on FastVLM's custom `LlavaQwen2ForCausalLM` | ✅ — `ekryski/FastVLM-0.5B-4bit` |

The reason: `ffai convert` doesn't load the upstream model code. It
reads weight tensors out of safetensors, classifies them by shape, and
quantizes via the same GPU kernel FFAI uses at inference. If FFAI can
*load* a checkpoint, it can also *convert* it.

## See also

- [Quick start](quickstart.md) — the 5-line library equivalent.
- [Benchmarking](benchmarking.md) — `ffai bench --method <name>`, KLD
  comparisons, per-day report shape.
- [Installation](installation.md) — adding FFAI to your own SwiftPM
  package (no CLI required).
