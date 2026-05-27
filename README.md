# FFAI

**F*cking Fast Apple Inference.**

A minimal, dependency-light LLM inference library for Apple Silicon, built on pre-compiled Metal kernels generated from the [metaltile](https://github.com/thewafflehaus/metaltile) DSL. No Python. No MLX. No C compilation. No JIT. No four-repo dependency chain.

**Just really f*cking fast AI on your Mac!** 🚀

## Status

Early bootstrap — the dense-text, hybrid, vision-language, and audio model waves have all landed; end-to-end inference runs real HuggingFace checkpoints across every shipped family.

- [`planning/plan.md`](planning/plan.md) — phased build-out, what we're shipping when
- [`planning/architecture.md`](planning/architecture.md) — visual reference for kernel generation, model loading, and the inference dispatch loop
- [`documentation/`](documentation/README.md) — user-facing docs

## Features

| Functionality | Description | Status |
|---|---|---|
| **Apple Silicon native** | Built ground-up for M-series GPUs. No fallbacks dragging it down. | ✅ |
| **Pre-compiled kernels** | The Metal kernels ship ready-to-run. No JIT delay the first time you load a model. | ✅ |
| **One-line model loading** | `Model.load("org/repo")` and you're generating. Download, cache, tokenizer, prewarm — one async call. | ✅ |
| **HuggingFace native** | Pull any compatible model straight from the Hub. Same cache as Python. | ✅ |
| **3 / 4 / 5 / 6 / 8-bit quantization** | Run beefy models on lean machines. The `mlx-community` quants you're already using. | ✅ |
| **Native MLX 4-bit conversion** | `ffai convert <repo>` quantizes any FFAI-loadable checkpoint to MLX 4-bit using our own GPU kernel — no Python, no `mlx-lm`, no `mlx-vlm`. Optional one-flag HF upload. | ✅ |
| **Single-buffer-per-token dispatch** | Forward + sample on one Metal command buffer per token. Just 4 bytes cross CPU↔GPU. | ✅ |
| **Capability-driven hot loading/unloading** | Only load what you'll use. Add and remove vision and audio encoders as you need them. | ✅ |
| **Async lifecycle stream** | Real progress for your UI — download, load, ready — as an `AsyncStream`. | ✅ |
| **Built in performance profiling** | Run benchmarks using the FFAI CLI and get performance telemtry data as you do inference. | ✅ |
| **Streaming generation** | Streaming inference support across all models. | ✅ |
| **Quantized KV cache** | Squeeze long contexts into a fraction of the memory. Affine 4/8-bit + AURA compressed. | ✅ |
| **Hybrid models (GDN + SSM)** | Qwen 3.5, Mamba 2, NemotronH, Jamba, GraniteMoeHybrid, FalconH1 — attention mixed with recurrence. | ✅ |
| **Mixture-of-experts** | GPT-OSS-20B, Qwen 3.5 MoE, Gemma 4 MoE — sparse top-K expert routing. | ✅ |
| **Vision (multi-modal)** | Drop in an image or video, get text back. Gemma 3/4-VL, Qwen 2.5/3-VL, Qwen3-VL-MoE, Nemotron-VLM. | ✅ |
| **Audio in / out** | Whisper-style speech-to-text, text-to-speech, omni audio, VAD — plus 7 neural audio codecs. | ✅ |
| **Speculative decoding** | Faster generation via n-gram lookup + draft models. | 🚧 Phase 8 |
| **Autotuner** | Per-shape kernel tuning so you never leave perf on the table. | 🚧 Phase 9 |
| **GGUF support** | Run llama.cpp's quants directly. | 🚧 Phase 10 |

For the longer-form view of what's shipped vs planned, see [`planning/roadmap.md`](planning/roadmap.md). For the per-topic deep-dives (KV cache, quantization, performance, capabilities) see [`documentation/`](documentation/README.md).

## Quick Start

Install via SwiftPM:

```swift
.package(url: "https://github.com/thewafflehaus/FFAI", from: "0.1.0")
```

Then generate text in five lines:

```swift
import FFAI

let model = try await Model.load("mlx-community/Qwen3.5-0.8B-MLX-4bit")
let result = try await model.generate(
    prompt: "Once upon a time",
    parameters: model.defaultGenerationParameters.with { $0.maxTokens = 64 }
)
print(result.text)
print("\(result.tokensPerSecond) tok/s")
```

`Model.load` resolves the HuggingFace repo, downloads the snapshot (or hits the cache), parses `config.json`, mmap-loads weights into per-tensor MTLBuffers, attaches the tokenizer, and prewarms the PSO cache. The first call costs a few seconds; subsequent loads of the same repo are near-instant.

CLI equivalent (the `ffai` executable target):

```bash
ffai --model mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "Once upon a time"
```

See [`quickstart.md`](documentation/quickstart.md) for streaming, chat templates, capability gating, and lower-level forward APIs. Using a non-default cache directory (external SSD, shared cache between Python tools, etc.)? See [Custom model cache path](documentation/quickstart.md#custom-model-cache-path).

## Models Supported

FFAI ships the most comprehensive Apple Silicon model coverage of any single library — **LLMs, VLMs, vision, STT, STS, TTS, and Omni models** all running real HuggingFace checkpoints end-to-end through one loader. Pass a repo ID, the registry resolves the architecture, downloads the snapshot, and routes to the right family.

Run `ffai models` for the live list with copy-paste repo IDs, or browse the full breakdown by family + capability in [`documentation/models.md`](documentation/models.md).

**Quantization.** Affine **2 / 3 / 4 / 5 / 6 / 8-bit** (mlx-community packed-uint32 format) ships today, with per-tensor bit-width derivation for mixed-precision checkpoints. `ffai convert` accepts per-tensor specs — any of `2 / 3 / 4 / 5 / 6 / 8 / fp16 / bf16` independently for `--bits` / `--embedding-bits` / `--lm-head-bits` / `--vision-bits`, so a single conversion can mix bit-widths across roles. **GGUF format is the remaining gap** — see [`quantization.md`](documentation/quantization.md) for what's wired up now vs queued.

### Adding a model

Porting a new family is **one Swift file plus an integration test**. The `Models/` tree mirrors itself in `Tests/` so the diff lands in two focused places, and the loader auto-routes on the `model_type` / `architectures[0]` strings the family enum advertises.

Step-by-step walkthrough with copy-pasteable templates: [`documentation/developing/adding-a-model.md`](documentation/developing/adding-a-model.md).

### Quantize a Model

`ffai convert` quantizes any bf16/fp16 HuggingFace checkpoint to MLX 4-bit or 8bit affine format using FFAI's own GPU kernels — no Python deps, no `mlx-lm` / `mlx-vlm` install, and it works on architectures `mlx-lm` rejects (custom-modeling-code families like Soprano, Nemotron-H, FastVLM):

```bash
# Pull, quantize, and write to ~/.cache/ffai/converts/.
ffai convert HuggingFaceTB/SmolLM2-360M-Instruct

# Also upload to a HF repo you control (uses your `hf` CLI auth).
ffai convert HuggingFaceTB/SmolLM2-360M-Instruct \
    --upload-repo ekryski/SmolLM2-360M-Instruct-4bit
```

Full flag list + recipes: [`using-the-cli.md` § `convert`](documentation/using-the-cli.md#convert--quantize-a-checkpoint-to-mlx-4-bit).

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
│   • `tile emit` (metaltile-cli) produces:               │
│       kernels.metallib   (compiled by xcrun metal)      │
│       manifest.json      (kernel metadata)              │
│       MetalTileKernels.swift  (typed wrappers)          │
└─────────────────────────────────────────────────────────┘
```

For the longer-form view (build pipeline, model load sequence, inference dispatch loop) see [`planning/architecture.md`](planning/architecture.md) and [`documentation/architecture.md`](documentation/architecture.md).

## Contributing

Read **[`CONTRIBUTING.md`](CONTRIBUTING.md)** first — it covers:

- the community guidelines;
- issue-first rule;
- what good PRs look like;
- how we deal with AI-assisted contributions; and
- how to get started! 🚀

## License

Apache-2.0.
