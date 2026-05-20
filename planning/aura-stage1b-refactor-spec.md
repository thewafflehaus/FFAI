# AURA Stage 1b refactor spec — two codecs, two-phase, compressed-domain default + opt-in B-path

Derived from the `mlx-swift-lm@alpha` TurboQuant reference
(`Libraries/MLXLMCommon/TurboQuantKVCache.swift`) per the user's
2026-05-19 guidance. Companion to `aura-audit-and-task-inventory-2026-05-19.md`.

**Sequencing:** the AURA index-50 coherence bug is being hunted first
(against the current simple path) so this refactor lands on a known-good
baseline. If the bug hunt finds the dequant+sdpaDecode path is itself the
defect, item 3 below subsumes the fix.

## 1. Two codecs (separate K + V rotation/seed)

- Today `AURAQuantizedKVCache` holds one `rotation` used for both K and V.
- Reference uses two independent `MSECodec`s — `keyMSECodec`, `valueMSECodec`
  — each with its own seed → decorrelated K/V quantization noise.
- FFAI change: per layer, build two SRHT rotations with distinct seeds
  (e.g. `2*layerIdx` for K, `2*layerIdx+1` for V). Encode K with the K
  rotation, V with the V rotation. Q is rotated with the **K** rotation
  (must match K for score cancellation). Attention output is un-rotated
  with the **V** rotation's transpose.
- `AURAQuantizedKVCacheRotations.build` already constructs the four-tensor
  bundle (f32 + activation-dtype, rotation + rotationT) — extend it to a
  K-bundle + V-bundle pair.

## 2. Two-phase prefill (raw fp16 → batch-compress)

Reference shape (`update` / `compressRawCache` / `encodeNewToken`):

- **Phase 1 — prefill.** `update(keys:values:)` stores raw fp16 K/V into a
  growable `rawKeys` / `rawValues` buffer and returns them. Attention during
  prefill runs on raw fp16 via standard SDPA. No compression yet.
- **Transition.** On the first decode-step call, `compressRawCache()`
  batch-compresses the entire raw cache into packed form in one shot, then
  frees the raw buffers.
- **Phase 2 — decode.** `encodeNewToken` per-token-encodes each new token
  into the compressed store; attention runs compressed-domain.

FFAI today encodes every token on arrival (no raw phase). Adopt the
two-phase shape: `AURAQuantizedKVCache` grows a raw fp16 buffer during
prefill, compresses once at the prefill→decode boundary, and per-token
encodes thereafter. Benefit: encode cost is hidden in TTFT, and prefill
attention is exact (no prefill-token quantization error).

## 3. Compressed-domain attention (DEFAULT) + opt-in B-path

This is the core of the user's intent: respect the compression algorithm
the caller asked for — do NOT keep a persistent exploded fp16 mirror.

### Default path — compressed-domain via aura_flash_p1 + aura_flash_pass2

- Drop the persistent `sharedWorkingK` / `sharedWorkingV` buffers entirely.
- Decode attention dispatches `aura_flash_p1` (per-block online-softmax over
  packed K/V; emits `o/m/l` partials) then `aura_flash_pass2` (cross-block
  reduce) directly on the packed K/V — no dequant buffer.
- New Ops wrappers needed: `Ops.auraFlashP1`, `Ops.auraFlashPass2`.
  Kernel sigs: `aura_flash_p1` takes `q_rot, key_packed, key_norms,
  key_codebook, val_packed, val_norms, val_codebook` → `o_partials,
  m_partials, l_partials`; Grid3D `(lane, q_idx, block_idx)`, tg 32 lanes.
  `aura_flash_pass2` reduces partials → `[q_heads, dim]` output.
- Output is in V-rotated space → un-rotate with V rotation transpose
  (or fold into oProj later — Stage 1a does runtime un-rotation).

### Opt-in B-path — short-lived dequant buffer + sdpaDecode

- Selected by a `LoadOptions` flag (mirror reference's `useDequantSDPA`).
- Per-layer-per-step: bulk-dequant the live K/V slice into a **local**
  working buffer sized `[nKVHeads, tokenCount, headDim]` for THAT layer
  only, run `Ops.sdpaDecode`, then let the buffer be released before the
  next layer. NOT a persistent `maxSeq`-sized mirror.
- Memory discipline (the user's explicit requirement): the fp16 working
  buffer must be short-lived — allocated inside the attention call, dropped
  when it returns. Reference relies on MLX lazy-eval + ARC; FFAI must do it
  explicitly — allocate from a pool with a tight scope, or a per-layer
  buffer that's overwritten each layer (one layer's worth resident at a
  time, not the whole cache × nLayers).
- Investigate whether to dequant only the needed heads/tokens rather than
  the full live slice — the user recalls per-layer/per-head narrowing in
  mlx-swift-lm; confirm + adopt + try to improve.

## 4. Norm correction — keep FFAI's (for now)

FFAI's always-applied `corrected_norm = ‖x‖/‖recon‖` stays. Revisit only if
the bug hunt implicates it, or later via an A/B test (PPL/KLD + speed) vs
the reference's WHT-path raw-norm approach.

## 5. Q-rotation dispatch — bug-hunt input

Flagged as a coherence suspect. The bug hunt checks whether the per-head
`Ops.gemv` Q-rotation loses precision (bf16 accumulation) vs the reference's
fp32-accumulate matmul. If so, the fix (fp32 Q-rotation) lands in the bug
hunt, not here.

## Verification

Both paths must produce coherent English on Qwen3-1.7B-bf16 (and ideally
Qwen3.5 0.8B/2B once GDN/SSM lands — the reference was validated on
Qwen3.5 + Gemma 4, not Qwen3-1.7B, so Qwen3-1.7B is an extra variable).
Re-enable the AURA integration tests in `KVCacheSchemeIntegrationTests`.

## Fallback

If Qwen3-1.7B can't be made coherent after the bug hunt + this refactor,
implement Phase 5e (GDN + SSM) and validate AURA on Qwen3.5 0.8B/2B, which
the reference was actually tested against.
