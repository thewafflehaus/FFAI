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
| **NemotronH** | [`Models/NemotronH.swift`](../Sources/FFAI/Models/NemotronH.swift) | `nemotron_h` | `NemotronHForCausalLM` | `NemotronHHybrid` |
| **GraniteMoeHybrid** | [`Models/GraniteMoeHybrid.swift`](../Sources/FFAI/Models/GraniteMoeHybrid.swift) | `granitemoehybrid` | `GraniteMoeHybridForCausalLM` | `GraniteMoeHybridHybrid` |
| **Jamba** | [`Models/Jamba.swift`](../Sources/FFAI/Models/Jamba.swift) | `jamba` | `JambaForCausalLM` | `JambaHybrid` |

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

**NemotronH** is FFAI's first *stack-interleaved* hybrid: a
`hybrid_override_pattern` string assigns each decoder layer exactly one
mixer kind — `M` = Mamba 2 selective-SSM mixer, `*` = multi-head
attention, `-` = dense squared-ReLU MLP — and the kinds genuinely vary
down the stack (NemotronH-4B is 24 Mamba / 4 attention / 24 dense MLP).
Every layer shares one pre-mixer RMSNorm + one residual add; there is
no separate pre-FF norm. The layer array is therefore *heterogeneous*:
the engine holds `[any DecoderLayer]` and walks it in lockstep with a
per-index cache array (`Mamba2LayerCache` for `M`, `KVCache` for `*`,
`StatelessLayerCache` for `-`). Two structural deltas vs other
families: NemotronH attention uses **no positional embedding** (no
RoPE — the Mamba layers carry sequence order), and its Mamba layers run
**`n_groups = 8` grouped B/C** plus a **gated mixer RMSNorm**. The
grouped SSM reuses the shipped scalar `ssm_step` kernel by dispatching
it once per group over a contiguous head sub-slab — no new kernel. The
`E` (mixture-of-experts) layer kind is recognised but rejected at load:
NemotronH's MoE diverges from the shipped SwiGLU `MoELayer`, and no
small published checkpoint exercises it.

**GraniteMoeHybrid** is a *stack-interleaved* hybrid like NemotronH — a
`layer_types` array assigns each decoder layer one mixer kind (`mamba`
or `attention`) — but the feed-forward half of every layer is uniform:
either a block-sparse **MoE** (top-K SwiGLU experts plus an always-on
shared SwiGLU expert) when `num_local_experts > 0`, or a dense SwiGLU
MLP when it is `0`. Each layer is two pre-norm + residual blocks
(`input_layernorm` → mixer, `post_attention_layernorm` → FFN), unlike
NemotronH's single norm/residual. Attention uses **no positional
embedding** (`position_embedding_type: "nope"`) like NemotronH; the
Mamba mixer runs a single full-width gated RMSNorm over `d_inner`.
Granite's four scalar multipliers are handled without double-folding:
`embedding_multiplier` folds into a dedicated scaled embedding copy,
`residual_multiplier` folds into every mixer/FFN output projection,
`attention_multiplier` is the SDPA scale, and `logits_scaling` divides
the final logits. The MoE feed-forward reuses the shared `MoELayer`
(`.topKThenSoftmax` gating). It is the first family to exercise the
MoE command-buffer contract end-to-end: `MoELayer.decode` commits the
command buffer, so `GraniteMoeHybridModel.forward` allocates a fresh
buffer after every MoE-bearing layer.

**Jamba** is a *stack-interleaved* hybrid like NemotronH /
GraniteMoeHybrid — a `layers_block_type` schedule (derived from
`attn_layer_period` / `attn_layer_offset` when not given explicitly)
assigns each decoder layer one mixer kind (`mamba` or `attention`), and
every layer carries a feed-forward half: a dense SwiGLU MLP
(`num_experts == 1`) or a block-sparse **MoE** block. Each layer is two
pre-norm + residual blocks (`input_layernorm` → mixer,
`pre_ff_layernorm` → FFN). Attention uses **no positional embedding**
(no RoPE) like NemotronH. The structural delta vs every other FFAI
hybrid: Jamba's mixer is the *original* **Mamba 1** selective SSM, not
the Mamba 2 SSD form. Mamba 1's `A` is per-`(channel, state)` (a 2-D
`A_log` of shape `[d_inner, d_state]`) and `dt` is per-channel, so the
recurrence decay `exp(A·dt)` varies with the state index — which the
shipped Mamba 2 `ssm_step` Metal kernel (one scalar `A`/`dt` per head)
cannot express. Jamba therefore runs the selective-scan core on the
**CPU** (the per-token cost is `d_inner · d_state` ≈ 82K MACs,
negligible); the GPU still owns every projection, attention, and the
MLP. Because the scan is host-side, every Jamba *mamba* layer commits
the command buffer mid-`decode`, so `JambaModel.forward` refreshes the
command buffer after each such layer — the same contract
GraniteMoeHybrid's MoE layers use. The MoE feed-forward reuses the
shared `MoELayer` (`.topKThenSoftmax` gating).

Both Llama / Qwen 3 variants share the same Llama-shaped core: GQA
attention with RoPE, RMSNorm, SwiGLU MLP. Qwen 3 adds per-head q_norm / k_norm
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

### NemotronH

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `nvidia/Nemotron-H-4B-Base-8K` | 4B | bf16 | Integration-test baseline. |
| `nvidia/Nemotron-H-4B-Instruct-128K` | 4B | bf16 | Same shape, instruction-tuned. |
| `nvidia/Nemotron-H-8B-Base-8K` | 8B | bf16 | |

The integration suite uses Nemotron-H-4B-Base-8K — the smallest
published NemotronH checkpoint whose config the shipped FFAI path can
run end-to-end. It exercises the heterogeneous `[any DecoderLayer]`
decode loop, the per-index cache array, the grouped-B/C Mamba path,
the gated mixer RMSNorm, and no-RoPE attention.

**Known gaps.** Only raw bf16 / f16 NemotronH checkpoints are supported
— quantized variants are rejected with a clear error. The `E`
(mixture-of-experts) layer kind — used by the Nemotron-Cascade-2 /
Nemotron-3 MoE checkpoints — is rejected at load: NemotronH MoE uses
squared-ReLU experts with sigmoid group-expert-select routing, both of
which diverge from the shipped SwiGLU `MoELayer`. A NemotronH Mamba
layer whose per-group RMSNorm row size (`d_inner / n_groups`) is not a
multiple of 128 (e.g. Nemotron-3-Nano-4B's 960) is also rejected — the
`rmsNormRows` kernel requires a 128-aligned row.

### GraniteMoeHybrid

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `mlx-community/granite-4.0-h-350m-bf16` | 350M | bf16 | Integration-test baseline (dense FFN). |
| `mlx-community/granite-4.0-h-1b-bf16` | 1B | bf16 | Dense FFN, same hybrid shape. |
| `ibm-granite/granite-4.0-h-tiny` | 7B | bf16 | 64-expert MoE FFN + shared expert. |

The integration suite uses granite-4.0-h-350m — the smallest published
GraniteMoeHybrid checkpoint (32 layers, 28 Mamba + 4 attention,
`num_local_experts = 0` → dense SwiGLU FFN). It exercises the
heterogeneous `[any DecoderLayer]` decode loop, the per-index cache
array, no-RoPE attention, the gated mixer RMSNorm, and the four Granite
scalar multipliers.

**Known gaps.** Only raw bf16 / f16 GraniteMoeHybrid checkpoints are
supported — quantized variants are rejected with a clear error. The MoE
feed-forward path (block-sparse experts + shared expert) is implemented
and unit-covered via `MoELayerTests`, but the published MoE checkpoints
(H-Tiny / H-Small, 7B+) ship only quantized on mlx-community, so the
integration suite cannot exercise the MoE path on a small raw
checkpoint. A Mamba layer whose `d_inner` is not a multiple of 128 or
exceeds 4096 is rejected — the gated mixer RMSNorm uses the single-row
`rmsNorm` reduction kernel.

### Jamba

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `mlx-community/AI21-Jamba-Reasoning-3B-bf16` | 3B | bf16 | Integration-test baseline (dense FFN). |

The integration suite uses AI21-Jamba-Reasoning-3B — the smallest
published Jamba checkpoint that runs end-to-end without quantized-MoE
expert slicing (28 layers, 26 Mamba 1 + 2 attention, `num_experts = 1`
→ dense SwiGLU FFN on every layer). It exercises the heterogeneous
`[any DecoderLayer]` decode loop, the per-index cache array
(`JambaMambaLayerCache` / `KVCache`), the host-side Mamba 1 selective
scan, the 2-D `A_log` handling, and no-RoPE attention.

**Known gaps.** Only raw bf16 / f16 Jamba checkpoints are supported —
quantized variants (the `-4bit` / `-6bit` conversions) are rejected
with a clear error. The MoE feed-forward path (`num_experts > 1`,
e.g. the original Jamba-v0.1's 16-expert blocks) is implemented and
reuses the shared `MoELayer`, but every small raw Jamba checkpoint on
mlx-community is dense, so the integration suite exercises only the
dense FFN. The Mamba 1 selective scan runs on the CPU — the shipped
Mamba 2 `ssm_step` kernel cannot express Mamba 1's per-`(channel,
state)` decay — which makes every Jamba mamba layer commit the command
buffer mid-decode.

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
