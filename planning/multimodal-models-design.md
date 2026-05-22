# Design — Marlin-2B / MiniCPM-V-4.6 / SAM 3.1 / Falcon-OCR

Implementation plan for four requested models. Concise; density over
grammar. Issue files: `planning/issues/features/F-007..F-012`.

> NOTE (post-clobber recovery, 05/21/2026): this doc was written
> before the FFAI VLM wave landed on `ek/aura-port`. Several pieces it
> describes as "not built" — `VisionEncoder`, `ImagePreprocessing`,
> `VLModel`, the `conv2d`/`patch_embed`/`rope_2d` Ops, SigLIP, and the
> Qwen3-VL tower — are now IN TREE. Treat F-007/F-009 as largely done
> and re-scope F-008/F-010/F-011 against the current codebase before
> implementing.

---

## 1. What each model is

### Marlin-2B (`NemoStation/Marlin-2B`)

- **Video VLM** — a fine-tune of **Qwen3.5-2B** with the Qwen3.5-VL
  video-capable visual tower. bf16, ~2B params.
- `modeling_marlin.py` is a thin wrapper adding `caption()` (dense
  video captioning) and `find()` (text query → `(start,end)` span)
  — prompt templates + output parsers, not architecture.
- FFAI fit: text decoder = Qwen3.5, already in tree. Net new = the
  Qwen3.5-VL visual tower + video frame sampling + the parsers.

### MiniCPM-V-4.6 (`openbmb/MiniCPM-V-4.6`)

- **Image/video VLM** — `MiniCPMV4_6ForConditionalGeneration`
  (`model_type: minicpmv4_6`).
- Vision `minicpmv4_6_vision`: SigLIP2-400M (hidden 1152, 27 layers,
  patch 14, image 980, `gelu_pytorch_tanh`).
- Text `qwen3_5_text`: **Qwen3.5** LLM — already in tree (`Qwen35`).
- A MiniCPM perceiver resampler projects vision tokens into the LLM
  stream; `insert_layer_id: 6`.
- FFAI fit: text decoder in tree. Net new = SigLIP encoder + the
  MiniCPM resampler + image/video preprocessing.

### Falcon-OCR (`tiiuae/Falcon-OCR`)

- **300M early-fusion document-OCR VLM** — `FalconOCRForCausalLM`
  (`model_type: falcon_ocr`). Image-in, text-out, autoregressive.
- LLM: dim 768, 22 layers, 16 q-heads / 8 kv-heads (GQA), head_dim
  64, vocab 65536, `ffn_dim 2304`, RMSNorm eps 1e-5, rope_theta
  10000.
- **No separate vision encoder** — one transformer processes image
  patches AND text tokens from layer 1 (early fusion). Vision
  frontend = `_patchify_and_project`.
- Deltas vs Llama: squared-ReLU gated MLP (`relu(gate)^2 * up`);
  hybrid 3D RoPE (1D temporal + 2D spatial golden-ratio); mixed mask
  (image bidirectional, text causal).

### SAM 3.1 (`facebook/sam3.1`, `mlx-community/sam3.1-bf16`)

- **Segment Anything 3.1** — open-vocabulary detection +
  segmentation + video tracking. `Sam3VideoModel`
  (`model_type: sam3.1_video`), ~860M params.
- **Not a language model.** Outputs boxes + masks, not tokens. No
  autoregressive decode, no token KV cache, no vocab head.

## 2. FFAI fit matrix

| Model | Text decoder | New work | Fits `LanguageModel`? |
|---|---|---|---|
| Falcon-OCR | new `falcon_ocr` | family + patch-embed frontend | ✅ |
| Marlin-2B | Qwen3.5 ✅ | Qwen3.5-VL tower + parsers | ✅ |
| MiniCPM-V-4.6 | Qwen3.5 ✅ | SigLIP encoder + resampler | ✅ |
| SAM 3.1 | none | whole new model category | ❌ |

## 3. Shared infrastructure — the VLM wave (F-007)

Image/video preprocessing, interpolation kernels
(`nearest`/`bicubic`/`grid_sample`), `conv2d`/`patch_embed`/`rope_2d`
metaltile kernels, the vision-token splice, `Capability.visionIn`.

## 4. Per-model strategy

- **F-008 Falcon-OCR** — new family. LLM ≈ Llama + a `reluSquared`
  gated MLP + hybrid-3D RoPE + a mixed image/text mask. Early fusion
  ⇒ no separate ViT — just patch-embed + projector. 300M ⇒ best
  end-to-end test target.
- **F-009 SigLIP encoder** — reusable ViT building block.
- **F-010 MiniCPM-V-4.6** — SigLIP (F-009) + MiniCPM resampler + the
  in-tree Qwen3.5 decoder.
- **F-011 Qwen3.5-VL tower + Marlin-2B** — port the visual tower;
  Marlin = tower + Qwen3.5 + caption/find parsers.
- **F-012 SAM 3.1** — out-of-scope assessment. A new non-autoregressive
  model category; needs a project-scope decision before any code.

## 5. Sequencing

F-007 → F-008 ‖ F-009 → F-010 ; F-011 after F-007. F-012 gated on a
scope decision.

## 6. Unresolved questions

1. Marlin-2B `config.json` — exact `architectures` / `model_type`
   (raw fetch returned HTTP 401).
2. MiniCPM-V-4.6 resampler shape + `insert_layer_id` semantics.
3. Falcon-OCR hybrid-3D RoPE exact axis split + golden-ratio formula.
4. Falcon-OCR `img_projector` — conv2d or reshape+linear.
5. **SAM 3.1 scope** — is FFAI chartered for non-LLM segmentation
   models? Project-owner decision.
6. Verification hosts for the `mlx-vlm` golden captures.
