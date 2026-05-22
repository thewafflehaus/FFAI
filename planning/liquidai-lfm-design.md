# Design — LiquidAI LFM2 / LFM2.5 / LFM2.5-VL

Implementation plan for the three LiquidAI collections. Concise;
density over grammar. Issue files: `planning/issues/features/F-013..F-015`.
Companion to the other `planning/*-design.md` docs.

---

## 1. What the collections are

Configs pulled: `LFM2-1.2B`, `LFM2-8B-A1B`, `LFM2.5-1.2B-Thinking`,
`LFM2.5-VL-1.6B`.

### LFM2 (`Lfm2ForCausalLM`, `model_type: lfm2`)

Liquid Foundation Models 2 — a **stack-interleaved hybrid** of
short-range gated convolution layers and GQA attention layers.

- `layer_types` per layer = `conv` or `full_attention` (LFM2-1.2B:
  16 layers, ~10 conv / 6 attention). The split is config-driven.
- Hidden 2048, 32 q-heads / 8 kv-heads (GQA), head_dim 64, vocab
  65536, RMSNorm `norm_eps` 1e-5, rope_theta 1e6, ctx 128000.
- **Conv mixer** — LFM2's "LIV" double-gated short convolution
  (`conv_L_cache: 3` → kernel 3, `conv_bias: false`, `conv_dim`
  2048). Block: in-projection → 3 chunks `(B, C, x)`; gate
  `Bx = B·x`; **causal depthwise conv1d** (kernel 3) over `Bx`;
  gate `y = C·conv`; out-projection. Per-layer conv state, depth 3.
- Attention layers — standard GQA + RoPE, per-head Q/K RMSNorm.
- MLP — SwiGLU (`block_use_swiglu`), FF dim auto-adjusted
  (`block_auto_adjust_ff_dim`, `block_multiple_of 256`).
- Sizes: 350M / 700M / 1.2B / 2.6B (+ `2.6B-Exp`).

### LFM2.5 (`Lfm2ForCausalLM`, `model_type: lfm2`)

**Architecturally identical to LFM2** — `LFM2.5-1.2B-Thinking`'s
config is `model_type: lfm2`, `Lfm2ForCausalLM`, the same
conv/attention `layer_types`, `tie_embedding: true`,
`rope_parameters` nested dict. "Thinking" vs "Instruct" vs "Base"
vs "JP" = training run + chat template, **not architecture**.
⇒ **No new family** — LFM2.5 loads through the `lfm2` family.

### LFM2-MoE (`Lfm2MoeForCausalLM`, `model_type: lfm2_moe`)

LFM2 conv+attention backbone with a **block-sparse MoE FFN**.
`LFM2-8B-A1B`, `LFM2-24B-A2B`.

- 24 layers, `layer_types` conv/full_attention as LFM2.
- MoE: `num_experts 32`, `num_experts_per_tok 4`,
  `moe_intermediate_size 1792`, `num_dense_layers 2` (first 2
  layers keep a dense SwiGLU FFN), `norm_topk_prob`,
  `use_expert_bias: true`, `routed_scaling_factor`.
- Routing (per `mlx-lm/lfm2_moe.py`): softmax over all experts →
  add `expert_bias` → top-K of the biased values → `norm_topk_prob`
  re-normalisation. The biased value is BOTH the selector and the
  combine weight.

### LFM2.5-VL (`Lfm2VlForConditionalGeneration`, `model_type: lfm2_vl`)

VLM — `LFM2.5-VL-450M`, `LFM2.5-VL-1.6B`.

- Vision: `siglip2_vision_model` — hidden 1152, 27 layers, patch 16,
  `gelu_pytorch_tanh`, intermediate 4304, `vision_use_head: false`.
- Text: `lfm2` (the LFM2-1.2B decoder).
- Projector: pixel-shuffle `downsample_factor: 2` → Linear (gelu,
  `projector_hidden_size 2048`, `projector_bias`, optional LN).
- Image tiling: `do_image_splitting`, `min/max_tiles 2/10`,
  `tile_size 512`, `use_thumbnail`, 64–256 image tokens.
  `image_token_id 396`.

## 2. FFAI fit

| Model | Family | In FFAI's wheelhouse? |
|---|---|---|
| LFM2 / LFM2.5 (350M–2.6B) | new `lfm2` | ✅ stack-interleaved hybrid |
| LFM2-MoE (8B-A1B / 24B-A2B) | `lfm2` MoE variant | ✅ reuses `MoELayer` |
| LFM2.5-VL (450M / 1.6B) | new `lfm2_vl` | needs the VLM wave |

LFM2's conv/attention interleave is exactly the heterogeneous
`[any DecoderLayer]` pattern FFAI already runs for NemotronH /
GraniteMoeHybrid / Jamba / FalconH1. The shipped `conv1d_causal_step`
kernel + `ConvStateCache` provide the depthwise causal conv1d the
LFM2 gated-conv mixer needs — **no new kernel** for LFM2/LFM2.5/-MoE.

## 3. Strategy

- **F-013 LFM2 family** — `Sources/FFAI/Models/LFM2.swift`. Closest
  templates: GraniteMoeHybrid / Jamba. Conv mixer = in-proj →
  `(B,C,x)` → `B·x` → `Ops.conv1dCausalStep` → `C·conv` → out-proj,
  with a `ConvStateCache`; attention layer = GQA+RoPE with host-side
  per-head Q/K norm (head_dim 64 is not 128-aligned). LFM2.5 routes
  through the same family unchanged.
- **F-014 LFM2-MoE** — an `LFM2MoE` variant in the same family file:
  swap the dense SwiGLU FFN for `MoELayer` on layers ≥
  `num_dense_layers`. `use_expert_bias` ⇒ a per-expert additive bias
  on `MoERouter`'s `.softmaxThenTopK`.
- **F-015 LFM2.5-VL** — `lfm2_vl` family: SigLIP2 encoder + the LFM2
  text decoder (F-013) + pixel-shuffle projector + image tiling.
  Needs the VLM wave.

## 4. Sequencing

F-013 (LFM2 dense — independent, ship first) → F-014 (LFM2-MoE —
depends F-013) ; F-015 (LFM2.5-VL — depends the VLM wave + F-013).

## 5. Testing

- LFM2-350M / LFM2.5-350M (~0.4B) — tiny, fully runnable; the
  integration-test baseline for the family.
- LFM2-MoE — `LFM2-8B-A1B` is the smallest MoE; ~8B bf16 ≈ 16 GB,
  runnable on a dev Mac.
- LFM2.5-VL-450M — tiny VLM; runnable once the VLM wave lands.

## 6. Unresolved questions

1. LFM2 conv block — exact tensor names + the `(B,C,x)` split order.
   (Resolved during F-013: `conv.in_proj` / `conv.conv` /
   `conv.out_proj`; split B, C, x.)
2. `block_auto_adjust_ff_dim` — the FF-dim formula. (Irrelevant for
   loading — the `w1/w3/w2` weight shapes are authoritative.)
3. LFM2-MoE `use_expert_bias` semantics. (Resolved during F-014:
   softmax-then-bias, bias is selector AND combine weight.)
4. `tie_embedding` vs `tie_word_embeddings` — LFM2 always ties.
5. LFM2.5-VL — pixel-shuffle projector + image-tiling layout. Needs
   the HF `modeling_lfm2_vl.py` / mlx-vlm reference.
6. Conv state cache depth — `conv_L_cache: 3` ⇒ depth-3 cache.
