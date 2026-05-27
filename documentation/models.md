# Supported Models

FFAI runs every family below directly from HuggingFace through
`Model.load("org/repo")` (or `AudioModel.load(...)` / `VADModel.load(...)`
for the non-text-decoder contracts). The `ffai models` CLI lists the
verified repo IDs per family; pass any other mlx-format conversion of a
supported architecture and the loader picks it up via
`ModelRegistry.dispatchAndLoad`.

This page is the canonical landing for:

- which architectures are in tree,
- which sizes / quantizations we have shipped checkpoints for,
- the one-line architectural quirk per family.

For porting a new architecture, see
[developing/adding-a-model.md](developing/adding-a-model.md). For the
per-family Swift type prefix and source layout, see the
`Sources/FFAI/Models/<X>.swift` family roots.

## Supported families

The Sizes column lists the parameter scales we ship in the CLI catalog
(`ffai models`); the Quants column lists the bit-widths we know are
published. Any 3 / 4 / 5 / 6 / 8-bit mlx affine conversion of a listed
architecture also loads.

Grouped alphabetically by family name. Modality is one of: **Text**,
**MoE-Text**, **Hybrid-Text** (SSM / GDN / conv), **Diffusion-Text**,
**VL** (vision-language), **STT** / **TTS** / **Omni** / **VAD** (audio
contract), **STS** (speech-to-speech).

| Family | Modality | `model_type` | Sizes shipped | Quants | Notes |
|---|---|---|---|---|---|
| **FalconH1** | Hybrid-Text | `falcon_h1` | 90M / 0.5B / 1.5B / 3B / 7B | bf16 / 8 / 6 / 5 / 4 | Per-layer Mamba 2 + attention summed into the residual (within-layer hybrid). |
| **FastVLM** | VL | `llava_qwen2` | 0.5B | bf16 | Apple FastVLM — LLaVA-style Qwen 2 backbone + FastViT depthwise-conv tower. |
| **Gemma 2** | Text | `gemma2`, `gemma2_text` | 2B / 9B / 27B | bf16 / 8 / 6 / 4 | Llama-shaped GQA + final-logit soft-cap; backbone for PaliGemma 2. |
| **Gemma 3** | Text | `gemma3`, `gemma3_text` | 270M / 1B / 4B / 12B / 27B | bf16 / 8 / 6 / 4 | Four per-block norms, per-head q/k RMSNorm, `sqrt(hidden)` embed scale. |
| **Gemma 3 VL** | VL | `gemma3` (+ `vision_config`) | 4B / 12B / 27B | bf16 / 8 / 6 / 4 | SigLIP ViT + multi-modal projector + Gemma 3 backbone. |
| **Gemma 4** | MoE-Text | `gemma4`, `gemma4_text` | E2B / E4B / 26B-A4B / 31B | bf16 / 8 / 6 / 4 / mxfp4 | Sliding (256-wide) + global (512-wide) SDPA, ProportionalRoPE, Per-Layer Embeddings, MoE on 26B-A4B. |
| **Gemma 4 VL** | VL | `gemma4` (+ `vision_config`) | E2B / E4B / 26B-A4B / 31B | bf16 / 8 / 6 / 4 / mxfp4 | Bespoke Gemma 4 ViT (multi-dim RoPE, attention pool) + Gemma 4 backbone. |
| **GPT-OSS** | MoE-Text | `gpt_oss` | 20B / 120B | MXFP4-Q8 / mxfp4-bf16 / 4 | 32-expert top-4 MoE, learned per-head attention sinks, alternating sliding / full attention, mixed-precision (MXFP4 experts). |
| **Granite 3** | Text | `granite` | 2B / 8B | fp16 / 8 / 6 / 4 | IBM Granite v3 dense — Llama-shaped GQA backbone (reuses Llama loader). |
| **Granite 4** | Hybrid-Text | `granitemoehybrid` | 350M / 1B / micro / tiny / small | bf16 / 8 / 6 / 4 | Stack-interleaved Mamba 2 / attention + dense or block-sparse MoE FFN (top-K SwiGLU + shared expert). No RoPE. |
| **Idefics3** | VL | `idefics3` | 8B | bf16 / 8 / 6 / 4 / 3 | HuggingFace Idefics3 — Llama 3 backbone + SigLIP-So400m ViT. |
| **InternLM 2** | Text | `internlm2` | 7B | bf16 / 8 / 4 | Shanghai AI Lab InternLM 2 / 2.5 — Llama-shaped GQA (reuses Llama loader). |
| **Jamba** | Hybrid-Text | `jamba` | 3B | bf16 | Stack-interleaved Mamba 1 (CPU scan — 2-D `A_log`) + attention + dense or MoE FFN. No RoPE. |
| **Kokoro** | TTS | `kokoro` | 82M | bf16 / 8 / 6 / 4 | StyleTTS2 acoustic + iSTFTNet GPU vocoder tail (`Ops.vocoderISTFT`). |
| **LFM2 / LFM2.5** | Hybrid-Text | `lfm2`, `lfm2_moe` | 350M / 700M / 1.2B / 2.6B / 8B-A1B / 24B-A2B | bf16 / fp16 / 8 / 6 / 5 / 4 / 3 | Stack-interleaved double-gated short-conv + attention (no SSM). MoE variant is block-sparse with biased-router top-K. Host-side Q/K norm (head_dim 64). |
| **Llama 3.x** | Text | `llama` | 1B / 3B / 8B / 70B / 405B | bf16 / 8 / 6 / 4 / 3 | Meta Llama 3 / 3.1 / 3.2 / 3.3 dense GQA transformer. Auto-detects Llama 3 RoPE scaling. |
| **Mamba 2** | Hybrid-Text | `mamba2` | 130M / 370M / 780M / 1.3B / 2.7B | 8 / 4 | Dense Mamba 2 selective-SSM (no attention). |
| **MiniCPM-V** | VL | `minicpmv4_6` | 4.6 (≈ 8B class) | bf16 / 8 / 5 / 4 | Qwen 3 backbone + SigLIP ViT, resampler projector. |
| **Mistral** | Text | `mistral` | 7B | bf16 / 8 / 4 | Llama-shaped GQA backbone — loads through the Llama-compatible loader. |
| **Mistral Small 3 VL** | VL | `mistral3` | 24B (3.1, 3.2) | bf16 / 8 / 6 / 4 / 3 | Mistral Small 3.1 / 3.2 — bespoke ViT + Mistral text backbone. |
| **Nemotron-Labs-Diffusion** | Diffusion-Text | `nemotron_labs_diffusion` | 3B / 8B / 14B | bf16 | Tri-mode — autoregressive / block diffusion / linear self-speculation. Hot-swappable `linear_spec_lora` drafter adapter. |
| **Nemotron-Labs-Diffusion VLM** | VL | `nemotron_labs_diffusion_vlm` | 8B | bf16 | Tri-mode diffusion text backbone + Pixtral ViT vision tower. |
| **Nemotron H** | Hybrid-Text | `nemotron_h` | 4B / 8B / 47B / 56B + Cascade-2 (8B / 14B / 30B-A3B) + Nemotron-3 (30B-A3B) | bf16 | Stack-interleaved Mamba 2 / attention / dense-MLP via `hybrid_override_pattern`. No RoPE on attention. Grouped-B/C Mamba (`n_groups`), gated mixer RMSNorm. MoE variants (Cascade-2 / Nemotron-3): sigmoid group-expert routing + squared-ReLU experts. |
| **Nemotron H VL** | VL | VL (`text_config.model_type = nemotron_h`) | 8B / 12B | bf16 | Shared SigLIP ViT + GELU projector + NemotronH text backbone. |
| **OLMo** | Text | `olmo`, `olmo2` | 1B / 7B / 13B / 32B | bf16 / 8 / 6 / 4 | AI2 OLMo 1 / 2 — open Llama-shaped research models (reuses Llama loader). |
| **Orpheus (LlamaTTS)** | TTS | `llama_tts`, `orpheus` | 3B | bf16 / 8 / 6 / 4 | Llama 3 acoustic backbone + Orpheus SNAC-code decode loop (`generateCodes`). SNAC codec is a separate codec port. |
| **PaliGemma** | VL | `paligemma` | 3B (PG1) / 3B / 10B / 28B (PG2) | bf16 / 8 / 6 / 4 / 3 | SigLIP ViT + Gemma backbone (Gemma 1 on PG1, Gemma 2 on PG2). |
| **Phi 3** | Text | `phi3` | mini / small / medium (4K / 8K / 128K ctx) | bf16 / 8 / 6 / 4 | Microsoft Phi-3 / 3.5 dense transformer. |
| **Pixtral** | VL | `pixtral` | 12B | bf16 / 8 / 4 | Bespoke RoPE-2D ViT + Mistral text backbone. |
| **Qwen-Omni** | Omni | `qwen2_5_omni`, `qwen3_omni` | Q2.5-Omni 3B / 7B, Q3-Omni 30B-A3B | bf16 / 8 / 6 / 5 / 4 | Whisper-style audio encoder projecting into the text backbone hidden dim. Vision path reuses Qwen-VL. |
| **Qwen 2** | Text | `qwen2` | 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B | bf16 / 8 / 4 | Qwen 2 / 2.5 dense transformer. |
| **Qwen 2-VL** | VL | `qwen2_vl` | 2B / 7B / 72B | bf16 / 8 / 4 | Dynamic-resolution windowed-attention ViT (head_dim 96) + Qwen 2 backbone. |
| **Qwen 2.5-VL** | VL | `qwen2_5_vl` | 3B / 7B / 32B / 72B | bf16 / 8 / 6 / 4 / 3 | Dynamic-resolution windowed-attention ViT (head_dim 80) + Qwen 2.5 backbone. |
| **Qwen 3** | Text | `qwen3` | 0.6B / 1.7B / 4B / 8B / 14B / 32B | bf16 / 8 / 6 / 5 / 4 / 3 | Per-head q/k RMSNorm applied before RoPE (only delta vs Llama). |
| **Qwen 3-ASR** | STT | `qwen3_asr` | 0.6B / 1.7B | bf16 / 8 / 6 / 5 / 4 | Qwen 3 backbone + audio encoder front-end. |
| **Qwen 3-TTS** | TTS | `qwen3_tts`, `qwen3_tts_base` | 0.6B / 1.7B | bf16 / 8 / 6 / 5 / 4 | Four-part TTS — talker + code predictor + ECAPA speaker encoder + intrinsic codec. Staged port — synthesis throws until later stages land. |
| **Qwen 3-VL** | VL | `qwen3_vl` | 2B / 4B / 8B / 32B | bf16 / 8 / 6 / 5 / 4 / 3 | Dynamic-resolution full-attention ViT + Qwen 3 dense backbone. |
| **Qwen 3-VL MoE** | VL | `qwen3_vl_moe` | 30B-A3B / 235B-A22B | bf16 / 8 / 6 / 4 / 3 | Qwen 3-VL ViT + Qwen 3.5-shaped GDN ↔ attention MoE backbone. |
| **Qwen 3.5 / 3.6** | Hybrid-Text | `qwen3_5`, `qwen3_5_moe` | 3.5 dense: 0.8B / 2B / 4B / 9B / 27B; 3.5 MoE: 35B-A3B / 122B-A10B / 397B-A17B; 3.6: 27B / 35B-A3B | bf16 / mxfp4 / 8 / 6 / 5 / 4 / 3 / **2** | Stack-interleaved Gated Delta Net ↔ attention; dense SwiGLU or block-sparse MoE with sigmoid-gated always-on shared expert. Gated attention (`attn_output_gate`), partial RoPE. Auto-folds the centered-RMSNorm `+1` on raw HF releases (mlx-community pre-folds during conversion). Qwen 3.6 ships under the same `qwen3_5*` keys. |
| **Qwen 3.5-VL** | VL | `qwen3_5` + `vision_config` | 0.8B (more on the way) | bf16 / 8 / 6 / 5 / 4 / 3 / **2** | Qwen 3-VL vision tower + Qwen 3.5 hybrid text backbone. Shares the `Qwen3_5ForConditionalGeneration` architecture string with the text-only release; the loader disambiguates by probing for the actual `model.visual.*` / `vision_tower.*` tower in the safetensors. Inherits the centered-RMSNorm auto-fold for raw HF layouts. |
| **SenseVoice** | STT | `sensevoice` | Small | bf16 | SAN-M encoder (fused QKV + FSMN depthwise-conv memory) + CTC head — non-autoregressive STT. |
| **SmolLM** | Text | `smollm`, `smollm2`, `smollm3` | 135M / 360M / 1.7B / 3B | bf16 / fp16 / 8 / 6 / 5 / 4 / 3 | HuggingFace SmolLM 1 / 2 / 3 — small Llama-shaped models (reuses Llama loader). |
| **SmolVLM** | VL | `smolvlm` | 256M / 500M / 2.2B / Instruct | bf16 / 8 / 6 / 4 / 3 | HuggingFace SmolVLM 1 / 2 — SmolLM backbone + SigLIP-So400m ViT. |
| **Soprano** | TTS | `soprano` | 80M | bf16 / 8 / 6 / 5 / 4 | Compact StyleTTS2-flavored on-device synth (Soprano + Soprano 1.1). |
| **Starcoder 2** | Text | `starcoder2` | 3B / 7B / 15B | 8 / 4 | BigCode Starcoder 2 — Llama-shaped code model (attention biases). |
| **Voxtral** | STT | `voxtral_realtime` | Mini 3B / 4B (realtime + TTS-2603) | bf16 / fp16 / 8 / 6 / 4 | Mistral Voxtral — realtime STT / TTS (streaming). |
| **Whisper** | STT | `whisper` | tiny / base / small / medium / large-v1 / large-v2 / large-v3 / large-v3-turbo | bf16 / fp16 / 8 / 6 / 5 / 4 | OpenAI Whisper — `AudioEncoder` + cross-attending causal text decoder. |
| **Yi** | Text | `yi` | 6B / 9B / 34B | bf16 / 8 / 4 | 01.AI Yi — Llama-shaped dense backbone (reuses Llama loader). |

### Audio dispatch

The audio families do **not** route through `ModelRegistry` /
`LanguageModel` — those describe a pure text-in / text-out causal
decoder. STT is audio-in / text-out, TTS is text-in / audio-out. They
load through [`AudioModelRegistry`](../Sources/FFAI/CLI/AudioModelRegistry.swift)
which inspects `config.json`, picks the family, and reports the audio
[`Capability`](../Sources/FFAI/Capability.swift) set.

Whisper, Kokoro and Qwen-Omni share the
[`AudioEncoder`](../Sources/FFAI/AudioEncoder.swift) module (Whisper-style
conv stem + bidirectional transformer) and the
[`AudioPreprocessing`](../Sources/FFAI/AudioPreprocessing.swift) front-end
(log-Mel STFT framing). The three FFAI audio kernels — `mel_spectrogram`,
`audio_conv1d`, `vocoder_istft` — are wrapped by `Ops.melSpectrogram`,
`Ops.audioConv1d`, `Ops.vocoderISTFT`. SenseVoice carries its own
Kaldi-style FBANK + LFR front-end and a depthwise FSMN memory block on
the CPU path.

### VAD dispatch

VAD models have a third contract — audio waveform in, per-frame
speech-probability stream out. They load through
[`VADModelRegistry`](../Sources/FFAI/CLI/VADModelRegistry.swift) which
exposes `loadFromDirectory` / `fromPretrained` and a
`detect(audio:sampleRate:)` returning a
[`VADOutput`](../Sources/FFAI/VADOutput.swift).

| Family | `model_type` | Notes |
|---|---|---|
| **Silero VAD** | `silero_vad` | Streaming VAD — STFT + small gated-conv encoder, 16 kHz and 8 kHz branch configs. |
| **SmartTurn** | `smart_turn`, `smart_turn_v3` | Conversational endpoint / turn detection (Pipecat). |

Sortformer (multi-speaker diarization), TenVAD, and FireRedVAD are
recognised by the registry; full loaders are follow-on ports.

### Neural audio codecs

Codecs (encoder + quantizer + decoder) turn a waveform into discrete
codes and back; autoregressive TTS families (Orpheus / LlamaTTS, Marvis,
…) emit codes that a codec renders to audio. They live under
[`Sources/FFAI/Models/Audio/`](../Sources/FFAI/Models/Audio) — see the
codec table in [audio.md](audio.md) for SNAC, EnCodec, Mimi, Descript
DAC, Vocos, BigVGAN, DACVAE.

## Quantization

| Bit width | Format | Status |
|---|---|---|
| **bf16 / fp16** | Plain safetensors | ✅ |
| **8-bit** | mlx-format affine | ✅ |
| **6-bit** | mlx-format affine (byte-packed) | ✅ |
| **5-bit** | mlx-format affine (byte-packed) | ✅ |
| **4-bit** | mlx-format affine | ✅ |
| **3-bit** | mlx-format affine (byte-packed) | ✅ |
| **MXFP4** | GPT-OSS expert blocks (transcoded to int4 at load) | ✅ |
| **MXFP4-Q8** | GPT-OSS attention / router / embedding in 8-bit affine | ✅ |

See [quantization.md](quantization.md) for the packing layout, the
sub-group split dispatch that closed the 4-bit Qwen 3 perf gap, and how
the loader auto-detects mlx-format weights.

## Hybrid checkpoint caveats

A subset of hybrid (SSM / GDN) families only accept **raw bf16 / f16**
checkpoints today — quantized variants are rejected with a clear error
at load:

- **FalconH1** — µP scaling interacts with packed-weight dequant.
- **Jamba** — quantized MoE-expert slicing not wired.
- **Nemotron H** — quantized variants not supported.
- **Granite 4** — non-quantized only.

The published `mlx-community/granite-4.0-h-*-{3,4,5,6,8}bit` checkpoints
exist (we list them above for completeness — they round-trip the
loader's shape inspector), but the runtime path through the quantized
hybrid stack is still gated; load failures surface the same error
message. The unblock is shared with the AURA performance pass.

## Loading any other repo

Pass any HuggingFace repo ID with one of the in-tree `model_type` /
`architectures` strings to `Model.load`:

```swift
let model = try await Model.load("mlx-community/Qwen3-14B-4bit")
```

The loader resolves the snapshot, parses `config.json`, picks the right
family via `ModelRegistry.dispatchAndLoad` and builds the variant. If
the architecture isn't in the registry yet, you get a
`ModelError.unsupportedArchitecture(...)`.

## Known gaps

| Item | Status |
|---|---|
| GPU sampling filters | Greedy (argmax) and categorical (`temperature`) run fully on-GPU. `top-K` / `top-P` / `min-P` fall back to the CPU-sample path. |
| AURA performance | AURA KV schemes decode coherently but still run the dequant-then-`sdpaDecode` path with a working-buffer mirror. Compressed-domain attention is in flight. |
| Cross-request prompt caching | The KV cache lives for one `generate(...)` call. Prefix-cache reuse across requests is a follow-on. |
| Quantized hybrid checkpoints | NemotronH / Granite4 / Jamba / FalconH1 load raw bf16 / f16 only; quantized variants are rejected with a clear error. |
| Diffusion sampler / quadratic self-spec / Nemotron-Labs-Diffusion VLM tests | Trained diffusion sampler weights aren't in the public checkpoint; quadratic self-spec is paper-only; the VLM variant has the family file but no integration test. |
| Qwen 3-TTS synthesis | Stage 1 ships config decoding + family detection. The talker, code predictor, and intrinsic codec are follow-on stages — `synthesize` throws `synthesisNotWired` until then. |
| MiniCPM-V / Gemma 3 VL grounding | Loaders work, but generation has known content / loop bugs being tracked. |
