# FFAI

**Fucking Fast Apple Inference.**

A minimal, dependency-light LLM inference library for Apple Silicon, built on
pre-compiled Metal kernels generated from the [metaltile](https://github.com/thewafflehaus/metaltile)
DSL. No Python. No MLX. No C compilation. No JIT. No four-repo dependency chain.

**Just really fucking fast AI!** 🚀

## Status

Early bootstrap — the dense-text, hybrid, vision-language, and audio
model waves have all landed; end-to-end inference runs real
HuggingFace checkpoints across every shipped family.

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
| **Quantized KV cache** | Squeeze long contexts into a fraction of the memory. Affine 4/8-bit + AURA compressed. | ✅ |
| **Hybrid models (GDN + SSM)** | Qwen 3.5, Mamba 2, NemotronH, Jamba, GraniteMoeHybrid, FalconH1 — attention mixed with recurrence. | ✅ |
| **Mixture-of-experts** | GPT-OSS-20B, Qwen 3.5 MoE, Gemma 4 MoE — sparse top-K expert routing. | ✅ |
| **Vision (multi-modal)** | Drop in an image, get text back. Gemma 3/4-VL, Qwen 2.5/3-VL, Qwen3-VL-MoE, Nemotron-VLM. | ✅ |
| **Audio in / out** | Whisper-style speech-to-text, text-to-speech, omni audio, VAD — plus 7 neural audio codecs. | ✅ |
| **Speculative decoding** | Faster generation via n-gram lookup + draft models. | 🚧 Phase 8 |
| **Autotuner** | Per-shape kernel tuning so you never leave perf on the table. | 🚧 Phase 9 |
| **GGUF support** | Run llama.cpp's quants directly. | 🚧 Phase 10 |

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

See [`quickstart.md`](documentation/quickstart.md) for
streaming, chat templates, capability gating, and lower-level
forward APIs. Using a non-default cache directory (external SSD,
shared cache between Python tools, etc.)? See
[Custom model cache path](documentation/quickstart.md#custom-model-cache-path).

## Models Supported

The full text-LLM family set ships today — dense, mixture-of-experts,
and SSM/GDN hybrid architectures, all running real HuggingFace
checkpoints end-to-end. Run `ffai models` for the live list with
copy-paste repo IDs. Adding a family is one Swift file plus an
integration test — see
[`adding-a-model.md`](documentation/developing/adding-a-model.md).

| Family | `model_type` | Example repo (bf16 · 8-bit · 4-bit) |
|---|---|---|
| **Llama 3.x** | `llama` | `unsloth/Llama-3.2-1B` · `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| **Llama-compatible zoo** | `smollm` / `olmo2` / `starcoder2` / … | `mlx-community/SmolLM2-360M-Instruct-bf16` |
| **Qwen 2** | `qwen2` | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Qwen 3** | `qwen3` | `mlx-community/Qwen3-1.7B-bf16` · `-8bit` · `-4bit` |
| **Qwen 3.5** (GDN hybrid · MoE) | `qwen3_5` | `mlx-community/Qwen3.5-0.8B-MLX-bf16` · `-MLX-8bit` · `-MLX-4bit` |
| **Mistral** | `mistral` | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |
| **Phi 3** | `phi3` | `mlx-community/Phi-3-mini-4k-instruct-4bit` |
| **Gemma 3** | `gemma3` | `mlx-community/gemma-3-1b-it-bf16` |
| **Gemma 4** (Dense · E-PLE · MoE) | `gemma4` | `mlx-community/gemma-4-e2b-it-bf16` · `gemma-4-26b-a4b-it-8bit` · `gemma-4-31b-it-4bit` |
| **GPT-OSS-20B** (MoE) | `gpt_oss` | `mlx-community/gpt-oss-20b-MXFP4-Q8` |
| **Mamba 2** | `mamba2` | `mlx-community/mamba2-130m` |
| **FalconH1** (hybrid) | `falcon_h1` | `mlx-community/Falcon-H1-Tiny-90M-Instruct-bf16` |
| **NemotronH** (hybrid) | `nemotron_h` | `nvidia/Nemotron-H-4B-Base-8K` |
| **GraniteMoeHybrid** | `granitemoehybrid` | `mlx-community/granite-4.0-h-350m-bf16` |
| **Jamba** (hybrid) | `jamba` | `mlx-community/AI21-Jamba-Reasoning-3B-bf16` |
| **LFM2 / LFM2.5** (conv+attention hybrid · MoE) | `lfm2` / `lfm2_moe` | `LiquidAI/LFM2-1.2B` · `LiquidAI/LFM2-8B-A1B` |
| **Nemotron-Labs-Diffusion** | `nemotron_labs_diffusion` | `nvidia/Nemotron-Labs-Diffusion-3B` |

Quantization follows the **mlx-community** packed-uint32 format
(3/4/5/6/8-bit affine — weights + scales + biases per group), with
per-tensor bit-width derivation for mixed-precision checkpoints. Pass
any HuggingFace repo ID and the loader resolves architecture,
downloads the snapshot, and routes to the right family. See
[`models.md`](documentation/models.md) for sizes exercised + known
gaps (hybrid families load raw bf16/f16 only).

Vision-language families (Gemma 3/4-VL, Qwen 2.5/3-VL, Qwen3-VL-MoE,
Nemotron-VLM) and audio families (Whisper STT, SenseVoice STT, Kokoro
TTS, LlamaTTS / Marvis / Qwen3-TTS, Qwen-Omni, Silero/SmartTurn VAD)
plus seven neural audio codecs are all in tree — see
[`models.md`](documentation/models.md).

**Coming next** (per [`planning/roadmap.md`](planning/roadmap.md)):
chunked (batched) prefill, AURA compressed-domain attention,
speculative decoding + serving (Phase 8).

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

For the longer-form view (build pipeline, model load sequence,
inference dispatch loop) see
[`planning/architecture.md`](planning/architecture.md) and
[`documentation/architecture.md`](documentation/architecture.md).

## Contributing

Read **[`CONTRIBUTING.md`](CONTRIBUTING.md)** first — it covers:

- the community guidelines;
- issue-first rule;
- what good PRs look like;
- how we deal with AI-assisted contributions; and
- how to get started! 🚀

## License

Apache-2.0.
