# FFAI

**Fucking Fast Apple Inference.**

A minimal, dependency-light LLM inference library for Apple Silicon, built on
pre-compiled Metal kernels generated from the [metaltile](https://github.com/thewafflehaus/metaltile)
DSL. No Python. No MLX. No C compilation. No JIT. No four-repo dependency chain.

**Just really fucking fast AI!** 🚀

## Status

Early bootstrap — Phase 4 complete (end-to-end inference + perf pass).

- [`planning/plan.md`](planning/plan.md) — phased build-out, what we're
  shipping when
- [`planning/architecture.md`](planning/architecture.md) — visual
  reference for kernel generation, model loading, and the inference
  dispatch loop
- [`documentation/`](documentation/README.md) — user-facing docs

## Features

| Functionality | Description | Status |
|---|---|---|
| **Apple Silicon native** | Built ground-up for M-series GPUs. No fallbacks dragging it down. | ✅ |
| **Pre-compiled kernels** | The Metal kernels ship ready-to-run. No JIT delay the first time you load a model. | ✅ |
| **One-line model loading** | `Model.load("org/repo")` and you're generating. Download, cache, tokenizer, prewarm — one async call. | ✅ |
| **HuggingFace native** | Pull any compatible model straight from the Hub. Same cache as Python. | ✅ |
| **3 / 4 / 5 / 6 / 8-bit quantization** | Run beefy models on lean machines. The `mlx-community` quants you're already using. | ✅ |
| **Single-buffer-per-token dispatch** | Forward + sample on one Metal command buffer per token. Just 4 bytes cross CPU↔GPU. | ✅ |
| **Capability-driven hot loading/unloading** | Only load what you'll use. Add and remove vision and audio encoders as you need them. | ✅ |
| **Async lifecycle stream** | Real progress for your UI — download, load, ready — as an `AsyncStream`. | ✅ |
| **Built in performance profiling** | Run benchmarks using the FFAI CLI and get performance telemtry data as you do inference. | ✅ |
| **Streaming generation** | Streaming inference support across all models. | ✅ |
| **Quantized KV cache** | Squeeze long contexts into a fraction of the memory. Affine 4/6/8-bit + TurboQuant. | 🚧 Phase 5 |
| **Hybrid models (GDN + SSM)** | Qwen 3.5, Mamba, NemotronH — the families that mix attention with recurrence. | 🚧 Phase 5 |
| **Vision (multi-modal)** | Drop in an image, get text back. Qwen 2.5-VL / 3.5-VL first. | 🚧 Phase 6 |
| **Audio in / out** | Whisper-style speech-to-text and text-to-speech. | 🚧 Phase 8+ |
| **Speculative decoding** | Faster generation via n-gram lookup + draft models. | 🚧 Phase 8+ |
| **Autotuner** | Per-shape kernel tuning so you never leave perf on the table. | 🚧 Phase 7 |
| **GGUF support** | Run llama.cpp's quants directly. | 🚧 Phase 8+ |

For the longer-form view of what's shipped vs planned, see
[`planning/roadmap.md`](planning/roadmap.md). For the per-topic
deep-dives (KV cache, quantization, performance, capabilities) see
[`documentation/`](documentation/README.md).

## Quick Start

Install via SwiftPM:

```swift
.package(url: "https://github.com/thewafflehaus/FFAI", from: "0.1.0")
```

Then generate text in five lines:

```swift
import FFAI

let model = try await Model.load("unsloth/Llama-3.2-1B")
let result = try await model.generate(
    prompt: "Once upon a time",
    parameters: model.defaultGenerationParameters.with { $0.maxTokens = 64 }
)
print(result.text)
print("\(result.tokensPerSecond) tok/s")
```

`Model.load` resolves the HuggingFace repo, downloads the snapshot
(or hits the cache), parses `config.json`, mmap-loads weights into
per-tensor MTLBuffers, attaches the tokenizer, and prewarms the
PSO cache. The first call costs a few seconds; subsequent loads
of the same repo are near-instant.

CLI equivalent (the `ffai` executable target):

```bash
ffai --model unsloth/Llama-3.2-1B --prompt "Once upon a time"
```

See [`documentation/quickstart.md`](documentation/quickstart.md) for
streaming, chat templates, capability gating, and lower-level
forward APIs. Using a non-default cache directory (external SSD,
shared cache between Python tools, etc.)? See
[Custom model cache path](documentation/quickstart.md#custom-model-cache-path).

## Models Supported

Two architecture families ship today; both run real HuggingFace
checkpoints end-to-end. Adding a new family is one Swift file plus
test fixtures — see
[`documentation/adding-a-model.md`](documentation/adding-a-model.md).

| Family | Variants | Sizes | Quantizations |
|---|---|---|---|
| **Llama 3.x** (`Llama.swift`) | `LlamaDense` (GQA + RoPE3 scaling + RMSNorm + SwiGLU MLP) | 1B / 3B / 8B / 70B | bf16 / 8bit / 6bit / 5bit / 4bit / 3bit |
| **Qwen 3** (`Qwen3.swift`) | `Qwen3Dense` (Llama core + per-head q_norm/k_norm) | 0.6B / 1.7B / 4B / 8B / 14B / 32B | bf16 / 8bit / 6bit / 5bit / 4bit / 3bit |

Quant layouts follow the **mlx-community** packed-uint32 format
(weights + scales + biases per group). Pass any HuggingFace repo ID
and the loader resolves architecture, downloads the snapshot, and
routes to the right family. See
[`documentation/models.md`](documentation/models.md) for the full
matrix and known gaps.

**Coming next** (per [`planning/plan.md`](planning/plan.md)): Qwen 3.5
hybrid (GDN + attention), Qwen 3.5 MoE, Mistral, Phi, Gemma, vision
(Qwen 2.5/3.5-VL), audio (Whisper / Qwen-Omni).

## High Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│  FFAI (Swift)                                           │
│   • Tensor (MTLBuffer-backed)                           │
│   • Module / Linear / Embedding / RMSNorm               │
│   • Model definitions (Llama, Qwen, …)                  │
│   • SafeTensors loader                                  │
│   • KV cache, sampling, generate loop                   │
└────────────────────────┬────────────────────────────────┘
                         │ calls
┌────────────────────────▼────────────────────────────────┐
│  MetalTileSwift (Swift, in-repo)                        │
│   • Loads kernels.metallib (pre-compiled at build time) │
│   • PSO cache, function-constant specialization         │
│   • Generated typed wrappers (one per kernel)           │
└────────────────────────┬────────────────────────────────┘
                         │ resources from
┌────────────────────────▼────────────────────────────────┐
│  metaltile (Rust, sibling repo)                         │
│   • #[kernel] DSL → IR → MSL                            │
│   • `tile build --emit all` (metaltile-cli) produces:   │
│       kernels.metallib   (compiled by xcrun metal)      │
│       manifest.json      (kernel metadata)              │
│       MetalTileKernels.swift  (typed wrappers)          │
└─────────────────────────────────────────────────────────┘
```

For the longer-form view (build pipeline, model load sequence,
inference dispatch loop) see
[`planning/architecture.md`](planning/architecture.md) and
[`documentation/architecture.md`](documentation/architecture.md).

## Contributing

Read **[`CONTRIBUTING.md`](CONTRIBUTING.md)** first — it covers the
issue-first rule, what good PRs look like, and how to disclose
AI-assisted contributions.

### Setup

```bash
git clone https://github.com/thewafflehaus/FFAI && cd FFAI
git clone https://github.com/thewafflehaus/metaltile ../metaltile   # sibling
./scripts/setup-dev.sh                                        # toolchains + first build
make test                                                     # full unit suite
```

`setup-dev.sh` verifies Xcode CLI tools + `xcrun metal`, the Swift
toolchain, Cargo (for the metaltile `tile` CLI), and the sibling
metaltile checkout; resolves SPM deps; and runs the first build to
populate `kernels.metallib`.

### Common Make targets

| Target | What |
|---|---|
| `make build` | Regenerate kernels + `swift build` (debug) |
| `make build-release` | Same, release config |
| `make test` | Regenerate kernels + `swift test` |
| `make coverage` | `swift test --enable-code-coverage` + summary |
| `make regenerate-kernels` | Run `tile build --emit all` only |
| `make format` | `swift-format` the repo in place |
| `make docs` | Lint markdown + (if `../ffai-website` exists) preview the docs site locally |
| `make clean` | Remove `.build/` + generated artifacts |

### Where to read next

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution guidelines, AI disclosure.
- [`documentation/developing/`](documentation/developing/) — dev workflow, testing, adding a model, publishing.
- [`planning/architecture.md`](planning/architecture.md) — architectural invariants.
- [`planning/roadmap.md`](planning/roadmap.md) — what's shipped vs planned.

User-facing documentation lives at
[**ffai.dev**](https://thewafflehaus.github.io/ffai-website/) (built
from the markdown in this repo's [`documentation/`](documentation/),
the top-level `README.md`, and `planning/architecture.md` +
`planning/roadmap.md`). Site source:
[thewafflehaus/ffai-website](https://github.com/thewafflehaus/ffai-website).
For the release → docs publishing flow see
[`documentation/developing/publishing.md`](documentation/developing/publishing.md).

## License

Apache-2.0.
