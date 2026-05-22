# Design — Cohere Command A+ (`command-a-plus-05-2026-w4a4`)

Implementation plan for HF repo
`CohereLabs/command-a-plus-05-2026-w4a4`. Concise; sacrifices grammar
for density. Issue files: `planning/issues/features/F-001..F-006`.

---

## 1. What the checkpoint actually is

Pulled the real `config.json`. **Not** a quantized text LLM — a
**vision-language MoE model**.

- Top-level: `architectures: ["Cohere2VisionForConditionalGeneration"]`,
  `model_type: cohere2_vision`. = SigLIP vision tower (27 layers,
  hidden 1152) + multimodal projector + `Cohere2MoeForCausalLM` text
  decoder.
- Text decoder `text_config` (`model_type: cohere2_moe`):
  - 32 layers, hidden 4096, vocab 262144.
  - Attention: 128 q-heads / 8 kv-heads, head_dim 128.
    `attention_bias: false`. `use_qk_norm: false`.
  - MoE FFN every layer (`first_k_dense_replace: 0`): 128 experts,
    top-8 routed, 4 shared experts, expert intermediate 4096,
    `expert_selection_fn: sigmoid`, `norm_topk_prob: true`,
    `shared_expert_combination_strategy: average`. SwiGLU experts.
  - Block: `transformer_block_type: parallel` — one `input_layernorm`
    feeds BOTH attention and MoE; outputs sum into the residual.
  - Norm: `norm_type: layer_norm` — mean-centered LayerNorm,
    weight-only, `layer_norm_eps: 1e-5`. NOT RMSNorm.
  - RoPE: `position_embedding_type: rope_gptj`, `rotary_pct: 1.0`,
    `rope_theta: 50000`. GPT-J = interleaved adjacent pairs.
  - Attention schedule: `layer_types` interleaves
    `sliding_attention` (window 4096, RoPE) and `full_attention`
    (every 4th layer). **Global/full layers use NoPE.**
  - `use_embedding_sharing: true` → lm_head tied to embed_tokens.
  - `logit_scale: 1.0` (final logits × logit_scale).
- Size: ~218B total params, ~25B active.
- Quant `quantization_config`: `format: nvfp4-pack-quantized`,
  `quant_method: compressed-tensors`. **NVFP4 W4A4** — 4-bit float
  (e2m1) weights, fp8-e4m3 per-group-of-16 scale + fp32 global
  scale. `ignore` list ⇒ **only MoE expert Linears are NVFP4**;
  everything else bf16.

## 2. Gap analysis vs current FFAI

| Need | State | Issue |
|---|---|---|
| `cohere2_moe` family file | absent | F-001 |
| NVFP4 (`nvfp4-pack-quantized`) load | unsupported | F-002 |
| `cohere2_vision` routing | absent; VLM wave | F-003 |
| Sigmoid top-k routing + N averaged shared experts | partial | F-004 |
| GPT-J interleaved RoPE | `Ops.rope` is NeoX rotate-half | F-005 |
| Mean-centered LayerNorm | only `RMSNorm` exists | F-006 |
| Parallel transformer block | new layer shape | F-001 |
| NoPE on global-attention layers | new | F-001 |

## 3. Hard constraint — verification

The integration test ("model output coherent") **cannot be run in the
current dev environment**: 218B params; w4a4 weights ≈ 110 GB+. No
small `cohere2_moe` checkpoint exists. Component unit tests are
runnable; end-to-end coherence + golden fixtures need a ≥128 GB host.

## 4. Implementation strategy

### 4.1 Weight-only NVFP4 (F-002)

Transcode at load — no GPU kernel, the GPT-OSS MXFP4 trick. CPU-decode
each NVFP4 Linear (e2m1 nibble × e4m3 group scale × fp32 global) → fp32
→ re-quantize to mlx-affine int4 → `QuantizedLinear` → existing
`dequant_gemv_4`. Activations stay bf16 (w4a16 read of a w4a4
checkpoint — correctness-preserving).

### 4.2 GPT-J RoPE (F-005) + LayerNorm (F-006)

Host-side first — decode commits the command buffer every MoE layer,
so a host-side GPT-J rotation / LayerNorm is free (GPT-OSS host-norm
precedent). No metaltile worktree needed for v1.

### 4.3 Sigmoid MoE (F-004)

DeepSeek-style sigmoid routing with a load-balancing bias; 4 shared
experts averaged.

### 4.4 Family file (F-001)

`Sources/FFAI/Models/Cohere2Moe.swift`. Closest template = GPT-OSS
(MoE + sliding/full schedule + per-layer commit). Parallel block,
LayerNorm, NoPE on full layers.

### 4.5 `cohere2_vision` (F-003)

Phase A: load text-only from `text_config`. Phase B: full SigLIP
vision tower — the Phase 6.5 VLM wave.

## 5. Sequencing

F-006 → F-005 → F-004 → F-002 → F-001 → F-003-A. F-003-B defers to the
VLM wave.

## 6. Unresolved questions

1. Router bias tensor name + whether bias affects selection only.
2. Expert weight layout — stacked vs per-expert.
3. Shared-expert intermediate size + whether NVFP4-quantized.
4. A machine with ≥128 GB unified memory for golden-fixture capture.
5. Priority of F-003-B (full VLM).
6. Native W4A4 activation quant vs the w4a16 simplification.
