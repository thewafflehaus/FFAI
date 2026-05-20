# Supported Models

FFAI ships several architecture families today; all run real
HuggingFace checkpoints end-to-end through `Model.load("org/repo")`.
Beyond the dense transformer families (Llama, Qwen 3, Mistral, Phi,
Gemma 3, …) the in-tree set now includes the dense SSM family Mamba 2
and the first *hybrid* family, FalconH1 (Mamba 2 + attention).

This page is the canonical landing for:

- which architectures are in tree
- which sizes / quantizations have been exercised
- known gaps per family

For porting a new architecture, see
[developing/adding-a-model.md](developing/adding-a-model.md).

## In-tree families

| Family | File | `model_type` | `architectures` | Variants |
|---|---|---|---|---|
| **Llama 3.x** | [`Models/Llama.swift`](../Sources/FFAI/Models/Llama.swift) | `llama` | `LlamaForCausalLM` | `LlamaDense` |
| **Qwen 3** | [`Models/Qwen3.swift`](../Sources/FFAI/Models/Qwen3.swift) | `qwen3` | `Qwen3ForCausalLM` | `Qwen3Dense` |
| **Mamba 2** | [`Models/Mamba2.swift`](../Sources/FFAI/Models/Mamba2.swift) | `mamba2` | `Mamba2ForCausalLM` | `Mamba2Dense` |
| **FalconH1** | [`Models/FalconH1.swift`](../Sources/FFAI/Models/FalconH1.swift) | `falcon_h1` | `FalconH1ForCausalLM` | `FalconH1Hybrid` |

**FalconH1** is FFAI's first *hybrid* family: every decoder layer runs
BOTH a Mamba 2 selective-SSM mixer AND a grouped-query attention path
on the same normalized input, sums their outputs into the residual,
then applies a SwiGLU MLP. There is no layer-schedule interleave — all
layers are identically shaped (the hybrid-ness is *within* the layer).
It is the proving ground for the `DecoderLayer` protocol: the engine
holds `[any DecoderLayer]` and walks it in lockstep with a per-layer
`FalconH1LayerCache` bundling a Mamba SSM/conv state and an attention
KV cache. FalconH1 reuses the shipped Mamba 2 SSM kernels (`ssm_step`,
`conv1d_causal_step`) and the `sdpaDecode` attention path — no new
kernels were needed.

Both variants share the same Llama-shaped core: GQA attention with
RoPE, RMSNorm, SwiGLU MLP. Qwen 3 adds per-head q_norm / k_norm
RMSNorms applied to queries/keys *before* RoPE — the only structural
difference vs Llama. No new kernels were needed for Qwen 3; just an
extra RMSNorm site.

## Sizes exercised

These are the checkpoints regression-swept in the test suite and used
for the [performance](performance.md) numbers. Any other Llama 3.x or
Qwen 3 size from HuggingFace should work — the family files don't hard-code
size-specific paths.

### Llama 3.x

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `unsloth/Llama-3.2-1B` | 1B | bf16 | Phase 2 reference. |
| `unsloth/Llama-3.2-3B` | 3B | bf16 | |
| `mlx-community/Llama-3.2-1B-4bit` | 1B | mlx 4-bit | |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 3B | mlx 4-bit | |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | 8B | mlx 4-bit | |

Llama 3 RoPE scaling (`factor`, `low_freq_factor`, `high_freq_factor`,
`original_max_position`) is honored; checkpoints with `rope_type:
"llama3"` in `config.json` route through the scaled variant
automatically.

### Qwen 3

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `mlx-community/Qwen3-0.6B-bf16` | 0.6B | bf16 | |
| `mlx-community/Qwen3-1.7B-bf16` | 1.7B | bf16 | Integration-test baseline. |
| `mlx-community/Qwen3-1.7B-3bit` | 1.7B | mlx 3-bit | Integration-tested. |
| `mlx-community/Qwen3-1.7B-4bit` | 1.7B | mlx 4-bit | Integration-tested. |
| `mlx-community/Qwen3-1.7B-5bit` | 1.7B | mlx 5-bit | Integration-tested. Published by us — mlx-community didn't ship a plain text-only 5-bit Qwen3-1.7B (only TTS / ASR variants existed). |
| `mlx-community/Qwen3-1.7B-6bit` | 1.7B | mlx 6-bit | Integration-tested. |
| `mlx-community/Qwen3-1.7B-8bit` | 1.7B | mlx 8-bit | Integration-tested. |
| `mlx-community/Qwen3-4B-bf16` | 4B | bf16 | |
| `mlx-community/Qwen3-4B-4bit` | 4B | mlx 4-bit | |
| `mlx-community/Qwen3-4B-8bit` | 4B | mlx 8-bit | |
| `mlx-community/Qwen3-8B-4bit` | 8B | mlx 4-bit | |
| `mlx-community/Qwen3-14B-4bit` | 14B | mlx 4-bit | |

The integration-test suite uses Qwen 3 1.7B across every supported
bit width to keep CI fast — the per-bit-width quantization paths
don't depend on model size, so 1.7B (~3.5 GB bf16) covers the same
codepaths as 4B / 8B / 14B at a fraction of the download cost.

mlx-community didn't have a plain text-only 5-bit Qwen3-1.7B
(only TTS / ASR variants), so we converted + uploaded one ourselves:

```bash
mlx_lm.convert --hf-path Qwen/Qwen3-1.7B \
               --mlx-path ./Qwen3-1.7B-5bit \
               -q --q-bits 5 --q-group-size 64 \
               --upload-repo mlx-community/Qwen3-1.7B-5bit
```

### FalconH1

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `mlx-community/Falcon-H1-Tiny-90M-Instruct-bf16` | 90M | bf16 | Integration-test baseline (~173 MB). |
| `mlx-community/Falcon-H1-0.5B-Instruct-bf16` | 0.5B | bf16 | |
| `mlx-community/Falcon-H1-1.5B-Instruct-bf16` | 1.5B | bf16 | |
| `mlx-community/Falcon-H1-3B-Instruct-bf16` | 3B | bf16 | |
| `mlx-community/Falcon-H1-7B-Instruct-bf16` | 7B | bf16 | |

The integration suite uses Falcon-H1-Tiny-90M-Instruct — the smallest
published FalconH1 checkpoint — so the hybrid decode path (dual
Mamba+attention mixers, `FalconH1LayerCache`, the `DecoderLayer`
protocol stack) is exercised at minimal download cost. The architecture
is identical in shape across 0.5B / 1.5B / 3B / 7B.

**Known gaps.** Only raw bf16 / f16 FalconH1 checkpoints are supported
today — quantized (`-4bit` / `-8bit`) variants are rejected with a
clear error (the µP scaling interacts with packed-weight dequant in a
way that needs dedicated handling). `mamba_rms_norm=true` checkpoints
(gated mixer RMSNorm) and `mamba_n_groups > 1` are likewise rejected;
the shipped 90M / 0.5B / 1.5B all use `mamba_rms_norm=false` +
`n_groups=1`. mlx-community checkpoints ship *pre-sanitized* (the
scalar multipliers are already folded into the saved weights); the
loader detects this via a conv1d-weight-shape probe and skips
re-folding to avoid double-applying the multipliers.

## Quantization

All in-tree families support every bit width FFAI implements:

| Bit width | Format | Status |
|---|---|---|
| **bf16 / fp16** | Plain safetensors | ✅ |
| **8-bit** | mlx-format affine | ✅ |
| **6-bit** | mlx-format affine (byte-packed) | ✅ |
| **5-bit** | mlx-format affine (byte-packed) | ✅ |
| **4-bit** | mlx-format affine | ✅ |
| **3-bit** | mlx-format affine (byte-packed) | ✅ |

See [quantization.md](quantization.md) for the packing layout, the
sub-group split dispatch trick that closed the perf gap on 4-bit
Qwen3 4B, and how the loader auto-detects mlx-format weights.

## Loading any other repo

Pass any HuggingFace repo ID with one of the in-tree
`model_type` / `architectures` strings to `Model.load`:

```swift
let model = try await Model.load("mlx-community/Qwen3-14B-4bit")
```

The loader resolves the snapshot, parses `config.json`, picks the
right family via `ModelRegistry.dispatchAndLoad`, and builds the
variant. If the architecture isn't in the registry yet, you get a
`ModelError.unsupportedArchitecture(...)`.

## Known gaps

| Item | Status |
|---|---|
| Multi-modal (vision, audio) | Capability infrastructure in place from Phase 2; first real exercise lands in Phase 6 (Qwen 2.5/3.5-VL). |
| Chat templates | Tokenizer's chat template is not auto-applied by `generate(...)` yet — pass the templated prompt yourself. Auto-apply lands alongside the first instruct-tuned VL model. |
| Sampling | Greedy argmax only on the GPU path. Top-k / top-p / temperature exist as CPU helpers in `Sampling.swift`; GPU kernels for these land in Phase 5. |
| Quantized KV cache | Raw fp16/bf16 only. Affine + AURA land in Phase 5 — see [kv-cache.md](kv-cache.md). |
| Hybrid models | Qwen 3.5 (GDN + attention) and Mamba/Mamba 2 families need new SSM kernels; Phase 5. |
| MoE | Qwen 3.5 MoE and similar need fused-expert kernels; Phase 5. |
| MoE / vision-tied checkpoints | Detected as `unsupportedArchitecture` until their family files land. |
| Prompt caching across requests | Not yet — the cache lives for one `generate(...)` call. Multi-turn cache reuse is straightforward via the lower-level API (see [quickstart.md § Lower-level API](quickstart.md#lower-level-api)). |

## Coming next

Per [`planning/plan.md`](../planning/plan.md):

- **Phase 5** — AURA KV cache + GDN + SSM. Unlocks Qwen 3.5
  hybrid (GDN + attention), Qwen 3.5 MoE, NemotronH, Mamba families.
- **Phase 6** — first multi-modal model: Qwen 2.5-VL or Qwen 3.5-VL,
  exercising `Capability.visionIn` end-to-end.
- **Phase 7** — autotuner over kernel parameters
  (`tile_dims`, `threads`, `unroll`, `simd_matrix`, `async_copy`).
- **Phase 8+** — audio, additional families (Mistral, Phi, Gemma,
  GPT-OSS), gguf format support, dispatch-mode upgrades
  (`.argumentBuffers`, `.icb`).
