# Quantization

FFAI supports the **mlx-format** affine group-quantized weight layout at every bit width MLX itself ships: 3 / 4 / 5 / 6 / 8.

This is **weight-only** quantization — it shrinks the on-disk + on-GPU footprint of the model and (usually) speeds up decode by lowering memory-bandwidth pressure. It's a different axis from **[KV cache quantization](kv-cache.md)**, which compresses the attention K/V tensors at runtime (the affine 4/8-bit and AURA compressed KV caches both ship today).

## What's supported

| Bit width | Group sizes | Status | Notes |
|---|---|---|---|
| **bf16 / fp16** | n/a | ✅ | Reference. `unsloth/Llama-3.2-1B`, `mlx-community/Qwen3-4B-bf16`, etc. |
| **8-bit** | 32 / 64 / 128 | ✅ | One byte per weight, packed 4 per uint32. |
| **6-bit** | 32 / 64 / 128 | ✅ | Byte-level pack — 4 weights per 3 bytes. |
| **5-bit** | 32 / 64 / 128 | ✅ | Byte-level pack — 8 weights per 5 bytes. |
| **4-bit** | 32 / 64 / 128 | ✅ | Eight weights per uint32. The mlx-community standard. |
| **3-bit** | 32 / 64 / 128 | ✅ | Byte-level pack — 8 weights per 3 bytes. |

mlx-community ships `*-3bit`, `*-4bit`, `*-5bit`, `*-6bit`, `*-8bit` variants of most models; see [models.md](models.md) for the curated set we regression-sweep.

## File format

mlx-format is the **same `*.safetensors` file** as a plain HF checkpoint, just with different *contents*. Each quantized linear layer ships as a triplet of tensors:

```
model.layers.0.self_attn.q_proj.weight   uint32   (out, in/pack_factor)
model.layers.0.self_attn.q_proj.scales   fp16     (out, n_groups)
model.layers.0.self_attn.q_proj.biases   fp16     (out, n_groups)
```

`pack_factor = 32 / bits` for the 4-bit and 8-bit "uint32-aligned" case. For the byte-packed widths (3 / 5 / 6) the storage shape is chosen so the row stride lands on a byte boundary — kernels handle the unpack.

`config.json` carries:

```json
{
  "quantization": {
    "bits": 4,
    "group_size": 64
  }
}
```

The loader (`SafeTensors.swift`) detects this block and routes the matching layers through the dequant-gemv kernels. Plain `*.bin` / non-quantized `.safetensors` keep the bf16/fp16 path.

## How it dispatches

A single decode-time matvec on a quantized linear is one `dequant_gemv_<bits>` kernel call:

```
input row (1 × in)        →  hidden state for this layer
weight uint32 buffer      →  packed quant
scales fp16 buffer        →  per-group scale
biases fp16 buffer        →  per-group bias (zero point)
                              │
                              ▼
                  per-(out_row, group) sum  →  reduce  →  output (1 × out)
```

On 4-bit and 8-bit, the uint32-packed layout means each thread can fetch a packed word and emit `pack_factor` partial-products inside the inner loop — the kernel parallelizes over both rows and packs. On 3 / 5 / 6-bit, the unpack happens byte-by-byte; the **sub-group split** trick (one SIMD subgroup per pack within a row) closes the gap to the uint32-aligned widths.

## Choosing a bit width

Rule of thumb (subject to per-model variation):

- **8-bit** — closest to bf16 quality; modest 2× memory savings vs bf16. Use when quality is the priority.
- **6-bit** — middle ground. Slightly worse PPL than 8-bit, ~25% more memory savings.
- **4-bit** — the mlx-community standard. Best mainstream speed/memory/quality balance for most chat tasks.
- **5-bit** — compromise between 4-bit and 6-bit. Less commonly shipped by mlx-community.
- **3-bit** — most aggressive. Notable PPL hit; useful when memory is the binding constraint.

For a given model, pick the highest bit width that fits your memory budget. If you can fit 8-bit, bf16 is rarely worth the extra space on Apple Silicon.

## Loading quantized models

Same call as bf16:

```swift
let model = try await Model.load("mlx-community/Qwen3-4B-4bit")
```

The loader reads `config.json`, sees `quantization.bits = 4`, and routes the linear layers through `dequant_gemv_4`. No flag, no extra field on `LoadOptions`.

## MXFP4

GPT-OSS-20B publishes its MoE experts **MXFP4**-quantized (Microscaling FP4 with FP8-block scales) while the attention / router / embedding / lm_head tensors stay mlx affine-quantized. FFAI handles this via a load-time **transcode** path ([`GPTOSSMoE.swift`](../Sources/FFAI/Models/GPTOSSMoE.swift)): the MXFP4 experts are converted to FFAI's affine-int4 format at load, so the decode kernels are the same `dequant_gemv_4` path the rest of the model uses — no separate MXFP4 inference kernel. `nvfp4` is not handled.

## What's not supported (yet)

- **2-bit affine quantization** — the Ops-layer 2-bit affine quantize / dequantize wrappers exist (parity sweep across f32/f16/bf16) but the inference-side `dequant_gemv_2bit` kernel + loader path haven't landed. Tracked in [`planning/plan.md`](../planning/plan.md).
- **GGUF** quantizations (`Q4_K_M`, `Q5_K_M`, `Q8_0`, …) — different binary layout, different per-block scales, different tensor naming. Planned: a per-arch name mapper alongside the GGUF reader. Not currently scheduled.
- **Native mxfp4 / nvfp4 inference** — FFAI transcodes GPT-OSS's MXFP4 experts to affine-int4 at load (see above). A native MXFP4-scale-layout decode kernel — keeping the FP8-block scales rather than transcoding — is not implemented. `nvfp4` is not handled at all.
- **Mixed-bit per-layer** — config-driven per-layer bit budgets (`quantization_config` block). Planned alongside the autotuner.

## See also

- [Models](models.md) — checkpoints regression-swept per family.
- [Performance](performance.md) — current `tok/s` baseline and the perf history.
- [KV cache](kv-cache.md) — runtime quantization of the attention K/V (a different axis).
- [Architecture](architecture.md) — where dequant-gemv sits in the per-token dispatch loop.
