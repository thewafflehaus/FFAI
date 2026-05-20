# Phase 5e — SSM / GDN hybrid models: execution plan

`plan.md` § Phase 5e holds the *what*. This is the *how* — scope,
dependency graph, execution order, delegation strategy. Goal: get
the five hybrid model families decoding coherent text.

## Scope — and what is explicitly NOT in 5e

**In scope (5e ships the forward / decode path):**

- GDN forward kernel + `GDNStateCache` (forward-only).
- MoE inference infrastructure (top-K router + per-expert dispatch).
- Per-layer mixer alternation scaffolding.
- Five family files: NemotronH, FalconH1, Jamba, GraniteMoeHybrid,
  Qwen3.5 — each decoding coherent text on a real checkpoint.

**Deferred to Phase 8 (speculative decoding) — do NOT build in 5e:**

- `gated_delta_step_record`, `state_replay`, `ssm_step_record`,
  `ssm_replay` — partial-accept rollback kernels.
- The `StateReplayCache` protocol + `GDNStateCache.record()` /
  `.rollback(acceptedPrefix:)` hooks.

`plan.md` § 5e currently lists those replay items under 5e; they
belong with Phase 8. `GDNStateCache` in 5e is a plain forward state
holder, mirroring the shipped `SSMStateCache`.

**Deferred to a perf pass (note, don't build now):**

- `ssm_step` chunked-prefill parallel-scan variant — the shipped
  decode-step kernel works for prefill, just slow on long prompts.
- `conv1d_causal_prefill` — same; the decode-step variant suffices.

**Conditional:** `ssm_step` `n_groups > 1` (grouped B/C) — only if a
target checkpoint needs it. Check each model's `config.json` for
`n_groups` during 5e.D; generalise the kernel only if some model
sets it `> 1`.

## Already shipped (the Mamba 2 foundation 5e builds on)

`ssm_step`, `conv1d_causal_step` kernels + `Ops` wrappers;
`SSMStateCache`, `ConvStateCache`, `Mamba2LayerCache`;
`LayerCacheProtocol` / `KVCacheProtocol` split; `Ops.softplus`;
`Models/Mamba2.swift` (dense Mamba 2 decoding end-to-end). The four
Mamba-2-based hybrids reuse all of this — they need **no new SSM
kernel** for the basic path.

## Dependency graph

```
5e.A  layer-mixer scaffolding ──┬─→ FalconH1   (Mamba2 + attn/MLP, no MoE)
 (FFAI, pure Swift)             │
                                ├─→ 5e.B MoE infra ──┬─→ NemotronH
5e.C  GDN kernel + GDNStateCache│   (kernels + FFAI)  ├─→ Jamba
 (metaltile + FFAI) ────────────┴──────────┐          └─→ GraniteMoeHybrid
                                           └─────────→ Qwen3.5 (GDN + MoE)
```

- **FalconH1** depends only on 5e.A → it is the proving ground for
  the mixer scaffolding (no MoE, no GDN).
- **NemotronH / Jamba / GraniteMoeHybrid** add 5e.B (MoE).
- **Qwen3.5** is the headline and the most-coupled — needs 5e.A +
  5e.B + 5e.C. Do it last.

## Execution order

### 5e.A — Per-layer mixer scaffolding  (FFAI, small)

A model is a sequence of heterogeneous layers (Mamba 2 / attention /
MoE-MLP / dense-MLP). Add a per-layer mixer abstraction so a family
file declares a layer-type schedule and the decode loop dispatches
the right mixer + cache per layer. NemotronH's layer-type string
(`M` / `*` / `E` / `-`) is the canonical driver; the scaffolding
must be general enough for all five. `makeLayerCaches` already
returns `[any LayerCacheProtocol]` — extend that, don't refactor it.

### 5e.B — MoE inference infrastructure  (metaltile + FFAI, medium)

No MoE support exists in FFAI today. Needs: a top-K expert router
(gating logits → top-K expert ids + normalised weights) and
per-expert FFN dispatch (gather the K selected experts' weights,
run the SwiGLU, weighted-sum). The router top-K can likely reuse
the existing `arg_reduce` / sampling reduction machinery; per-expert
weight gather reuses the `dequant_gather` family for quantised
expert weights. Decide: dense loop over all experts with masking
(simple, fine for first light) vs a true sparse gather (faster).
Start dense; optimise later.

### 5e.C — GDN kernel + `GDNStateCache`  (metaltile + FFAI, medium)

Port `gated_delta_step` (the forward recurrence
`S_t = g_t·S_{t-1} + β_t·k_t·(v_t − k_tᵀ·S_{t-1})ᵀ`, fp32 state
throughout) from the `ekryski/mlx@alpha` `gated_delta.metal`
reference into `crates/metaltile-std/src/ffai/`. Macro over the
`(Dk, Dv, Hk, Hv)` combinatorics. Pair with a GPU correctness test
(naive CPU oracle, same commit). Then `Ops.gatedDeltaStep` +
`GDNStateCache` (forward-only, mirrors `SSMStateCache`).

### 5e.D — Family files  (FFAI, one per family + integration test)

Order = fewest dependencies first; each is the proving ground for
the capability it adds:

1. **FalconH1** — Mamba 2 + attention/MLP, per-layer multipliers.
   Needs only 5e.A. First family end-to-end.
2. **NemotronH** — layer-type string parsing; Mamba 2 + attention +
   MoE/MLP. Needs 5e.A + 5e.B.
3. **GraniteMoeHybrid** — Mamba 2 + MoE + attention.
4. **Jamba** — Mamba 2 + attention/MoE. Handle the 2D `A_log` shape
   (kernel-side generalisation OR Swift-side reshape — pick the
   cheaper once GDN/MoE have landed).
5. **Qwen3.5** — `Qwen35Dense`, `Qwen35MoE`, `Qwen35GDN` variants;
   GDN ↔ full-attention alternation every `fullAttentionInterval`
   layers. Needs 5e.A + 5e.B + 5e.C. The headline; last.

Each family: family file in `Sources/FFAI/Models/`, `ModelRegistry`
entry, variant detection from `config.json`, and a
`Tests/ModelTests/<Family>IntegrationTests.swift` that downloads
from mlx-community and asserts `expectCoherentOutput`. Doc rows in
`models.md` / `capabilities.md` / `quantization.md` same commit.

## Verification

- Each new kernel: paired GPU correctness test (naive CPU oracle) in
  metaltile, same commit.
- Each new `Ops` wrapper: a direct per-Op test + a validator if it
  is reduction-mode (per `CLAUDE.md`).
- Each family: coherent-output integration test on a real
  checkpoint.
- `make test-unit` green after every FFAI commit; `cargo test
  --workspace` green after every metaltile commit.

## Delegation strategy

5e.A is small + judgement-heavy (a protocol shape that all five
families live on) — do it directly. 5e.B (MoE) and 5e.C (GDN
kernel) are well-bounded kernel/infra ports — good agent
candidates once 5e.A's shape is fixed. Family files (5e.D) are
mechanical once their dependencies exist — delegate per family,
but verify each integration test's actual decoded text, not just
the pass/fail.

## Realistic size

Five model families + GDN kernel + MoE infrastructure is multi-
session. Landable checkpoints: (1) 5e.A + FalconH1, (2) 5e.B +
NemotronH, (3) GraniteMoeHybrid + Jamba, (4) 5e.C + Qwen3.5. Each
checkpoint is a coherent, shippable increment.
