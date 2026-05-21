# Optimizing Kernels for Apple M-series Architecture

**Authors:** Eric Kryski
**Hardware target:** Apple M1 → M5 series (Max class)
**Software target:** [FFAI](https://github.com/ekryski/FFAI) on [metaltile](https://github.com/0xClandestine/metaltile)
**Status:** Living reference — extend when we discover new optimization patterns.

This document is the working reference for everyone writing kernels in metaltile or wrappers in FFAI. It covers:

- A **glossary** of Metal / Apple-GPU vocabulary used throughout the codebase.
- The **hardware**: unified memory, GPU cache hierarchy, per-generation specs, the M5 matrix unit.
- The **inference model**: what an op is, how decode tokens flow CPU→GPU, where time goes.
- **Optimization strategies** with concrete recommendations for our kernel families.
- **FFAI / metaltile conventions** that consumers of the codegen surface need to know.

Sections 2–3 are adapted from earlier Sam-project performance notes; sections 1, 4–6 are new and FFAI-specific.

---

## 1. Glossary

Terms with specific meanings in Apple's Metal vocabulary that show up across our kernels, dispatch wrappers, and benchmark output.

| Term | Meaning |
|---|---|
| **Lane** | One thread inside a simdgroup. Apple simdgroups are 32 lanes wide. The `simd_lane` intrinsic gives this index. |
| **Simdgroup** | A hardware-fixed group of 32 lanes that execute in lockstep on one shader-core scheduler slot. `simd_sum`, `simd_max`, `simd_broadcast`, `simd_shuffle_xor` all reduce/communicate across these 32 lanes. NVIDIA equivalent: warp. |
| **Threadgroup** | A software-defined group of 1–32 simdgroups (32–1024 threads). Shares `threadgroup`-scoped memory and `threadgroup_barrier()`. Cooperative cross-simdgroup work happens here. |
| **TPG** | **Threads Per Threadgroup.** The width of the threadgroup, in threads. Hard-capped at 1024 on M-series. Must be a multiple of 32 if the kernel uses any `simd_*` intrinsic (else only partial simdgroups participate, which is undefined). Set at dispatch time via `threadsPerThreadgroup`. |
| **Grid** | The total dispatch geometry, expressed as a 3D box of threadgroups. Metal hierarchically subdivides: `grid` → `threadgroups` → `simdgroups` → `lanes`. metaltile's `dispatch_with_grid([gx,gy,gz], [tx,ty,tz])` takes grid in *threadgroups* and tg size in *threads*; total threads = `gx*gy*gz*tx*ty*tz`. |
| **Dispatch** | One `dispatchThreadgroups()` (or `dispatchThreads()`) call on a `MTLComputeCommandEncoder`. Encoded into a command buffer. Carries a PSO, a binding table, and the grid geometry. |
| **Op** | An MLX / metaltile abstraction over a dispatch. One op = set PSO + bind buffers + set scalars + dispatch. ~5–10 µs of CPU encoding cost per op. Reshape / transpose / slice are NOT ops — they create array view metadata with zero GPU cost. |
| **Command Buffer (CB)** | An ordered sequence of dispatches and resource bindings, committed to a `MTLCommandQueue` for GPU execution. Metal keeps every buffer referenced by a CB alive until the CB completes — peak memory tracks CB granularity. |
| **Pipeline State Object (PSO)** | A compiled-kernel + binding-layout artifact. Compiled from MSL once per `(kernel, function-constants)` pair, then cached in `PSOCache.shared`. Per-dispatch cost amortizes to a hash lookup. |
| **Register file** | Per-GPU-core SRAM holding per-thread variables. ~208 KB/core on M1 Max; ~210 KB/core × 40 cores ≈ 8.2 MiB total on M5 Max. Cleared between dispatches. Register pressure caps occupancy. |
| **Threadgroup memory** | Per-threadgroup SRAM (~32 KB usable on M1-M4; larger on M5). Persists for the threadgroup's lifetime; accessible to every thread in the group via `threadgroup_alloc(...)` and `threadgroup_load/store`. |
| **L1 / L2 / SLC** | GPU L1 (8 KB / core), GPU L2 (512 KB shared by all GPU cores), SLC = System Level Cache (48 MB on M1 Max; shared between CPU, GPU, ANE). Only SLC is big enough to amortize anything model-scale. |
| **Unified Memory** | CPU and GPU read the same physical LPDDR memory — no copy, no PCIe. Bandwidth: 400 GB/s on M1 Max, 614 GB/s on M5 Max. Decode at our model sizes is bandwidth-bound on weight reads. |
| **ANE (Apple Neural Engine)** | Dedicated 16-core neural accelerator with 32 MB SRAM. Native int8 / fp16 / bf16 / fp32. Runs in parallel with the GPU — ops on ANE don't consume GPU cycles. Accessed via CoreML (public) or `_ANEClient` (private). |
| **Reduction mode** | metaltile `KernelMode::Reduction`: kernel uses `simd_*` intrinsics and/or `threadgroup_alloc` to cooperate across threads in a group. `program_id<0>` lowers to `tgid_x` (threadgroup index). Carries strict TPG invariants. |
| **Grid3D mode** | metaltile `KernelMode::Grid3D`: "one thread per output element" pattern. `program_id<i>` lowers to `thread_position_in_grid.{x,y,z}`. No simdgroup arithmetic; no TPG invariant. |
| **Occupancy** | Active simdgroups per GPU core, ÷ peak simdgroups the core can hold. Bound by register pressure and threadgroup-memory pressure. Higher occupancy = better latency hiding for memory loads. |
| **Vector / Steel attention** | MLX naming for two SDPA variants. *Vector* is the per-query-vector kernel used at decode (Q is 1 row); *Steel* is the tiled flash-attention kernel used at prefill (Q is many rows). |
| **GQA / Grouped-Query Attention** | Multiple Q heads share a single KV head, so the cache shrinks by the GQA factor. `heads_per_group = nQHeads / nKVHeads`. |

---

## 2. Apple M-series Hardware Architecture

### 2.1 Unified Memory

There is **no separate GPU VRAM**. CPU and GPU share the same physical LPDDR memory. The GPU reads model weights directly from the same memory addresses the CPU loaded them to — zero-copy access.

```
┌─────────────────────────────────────────────────┐
│                 M-series SoC                    │
│                                                 │
│  ┌─────────┐  ┌──────────┐  ┌───────────────┐   │
│  │ CPU     │  │ GPU      │  │ Neural Engine │   │
│  │ 10 core │  │ 32 core  │  │ 16 core       │   │
│  └────┬────┘  └────┬─────┘  └───────┬───────┘   │
│       │            │                │           │
│  ┌────┴────────────┴────────────────┴───────┐   │
│  │       System Level Cache (SLC)           │   │
│  │              48 MB (M1 Max)              │   │
│  └──────────────────┬───────────────────────┘   │
│                     │                           │
│  ┌──────────────────┴───────────────────────┐   │
│  │    Memory Controller (512-bit LPDDR5)    │   │
│  │   400 GB/s (M1 Max) / 614 GB/s (M5 Max)  │   │
│  └──────────────────┬───────────────────────┘   │
└─────────────────────┼───────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │    Unified Memory       │
         │   16 / 32 / 64 / 128 GB │
         │  model weights +        │
         │  KV cache + activations │
         └─────────────────────────┘
```

### 2.2 GPU Cache Hierarchy

| Level | Size (M1 Max) | Can cache model weights across tokens? |
|---|---|---|
| Register file | 208 KB × 32 cores = 6.6 MB | No — cleared between dispatches |
| Threadgroup memory | ~32 KB per core | No — per-threadgroup, not persistent |
| GPU L1 | 8 KB × 32 = 256 KB | No — far too small, thrashed instantly |
| GPU L2 | 512 KB (shared by all GPU cores) | No — 0.05% of a 1 GB model |
| **SLC (L3)** | **48 MB** (shared CPU + GPU + ANE) | Partially — holds ~5% of a 1 GB model |

**Implication for decode.** A typical 4-bit-quantized 2B model (~1 GB) cycles through ~20× the SLC capacity per token. By the time layer 30's weights finish loading, layer 1's are evicted. Decode is therefore weight-bandwidth-bound, not compute-bound. **Theoretical floor: 1 GB ÷ 400 GB/s = 2.5 ms / token = 400 tok/s.**

### 2.3 Per-Generation Specs

Pulled together so you can size kernel parameters against the right target.

| Spec | M1 Max (2021) | M2/M3 Max | M4 Max | M5 Max (2026) |
|---|---|---|---|---|
| GPU cores | 32 | 38–40 | 40 | 40 |
| Register file (total) | ~6.6 MB | ~7 MB | ~7 MB | ~8.2 MB |
| Threadgroup memory | 32 KB | 32 KB | 32 KB | 32+ KB |
| Memory bandwidth | 400 GB/s | 400–410 GB/s | 546 GB/s | 614 GB/s |
| Native bf16 | No | Yes | Yes | Yes |
| Dedicated matrix unit | No | No | No | **Yes — 1024 FMA/core/cycle** |
| FP32 TFLOPS | 10.4 | ~14 | ~16 | ~20 (general), ~57 (matrix) |
| ANE cores | 16 | 16 | 16 | 16 |
| ANE SRAM | ~32 MB | ~32 MB | ~32 MB | ~32 MB |

### 2.4 M5 Max: the Matrix Unit changes the math

Each M5 GPU core now includes a hardware block specifically for matrix multiplication with its own register-adjacent buffers. The Metal compiler routes `simdgroup_matrix` operations and our `MMATile` / `tile_matmad` ops to this unit automatically. The practical effect:

**On M1 Max, BD=512 (head_dim=512) Steel SDPA gets ~6% occupancy** — the per-thread Otile accumulator is 2 KB, pushing per-simdgroup register usage to ~69 KB. The GPU stalls on memory reads constantly.

**On M5 Max, the same kernel can get ~92% occupancy** — the matrix unit holds Otile in its own buffers, so general register pressure drops to ~152 bytes/thread. No code changes needed; the same `MMATile` ops compile to `simdgroup_matrix` and route there.

This is why "the same kernel" can be barely viable on M1 Max and excellent on M5 Max. Don't over-tune for M1 if our hardware target spans the generation.

---

## 3. Inference on Apple Silicon

### 3.1 What's an "Op"?

An **op** is one Metal compute dispatch — a single `dispatchThreads()` call encoded into a command buffer. Each op launches one kernel on the GPU. Reshape, transpose, and slice are NOT ops — they just create array view metadata with zero GPU cost.

Per-op encoding cost (on the CPU side):

1. Set pipeline state — PSO lookup (~1 µs, cached)
2. Bind buffers — point the kernel at inputs/outputs (~2–5 µs per buffer)
3. Set parameters — threadgroup size, grid dimensions, scalar constexprs (~1 µs)
4. Dispatch — append to command buffer (~1 µs)

**Total: ~5–10 µs per op.** A typical decode token has ~200–500 ops. So **~1–5 ms of pure CPU encoding cost per token**, regardless of what the GPU does.

This is the reason kernel fusion is so high-leverage: every op you eliminate removes 5–10 µs of CPU cost AND one kernel launch (with its scheduling overhead).

### 3.2 Per-decode-token op counts (representative models)

Counts include kernel fusion (RMSNormRoPE, compiledNormResidual, etc.). Numbers are CPU dispatch counts, not GPU cycles.

| Model | Layers | Hidden | Total ops/token | Weight reads/token | Theoretical floor |
|---|---|---|---|---|---|
| Gemma4 E2B | 30 | 2816 | ~420 | ~1 GB | 2.5 ms / 400 tok/s |
| Gemma4 26B-A4B (MoE) | 30 | 2816 | ~900 | ~14 GB | 35 ms / 28 tok/s |
| Gemma4 31B dense | 60 | 5376 | ~1260 | ~17 GB | 42 ms / 24 tok/s |
| Qwen3.5-27B (GDN hybrid) | 64 | 5120 | ~1120 | ~14 GB | 35 ms / 28 tok/s |
| Qwen3.5-35B-A3B (GDN+MoE) | 40 | 2048 | ~1020 | ~18 GB | 45 ms / 22 tok/s |
| GPT-OSS-20B (MoE) | 24 | 2880 | ~480 | ~10 GB | 25 ms / 40 tok/s |

MoE models read fewer weights per token than total model size — only the top-K routed experts. For Gemma4 26B with top-8 of 128 experts, that's ~6.25% of expert weights per token plus shared attention + router.

### 3.3 Decode token: CPU↔GPU flow

```
 CPU (Swift + MLX C++)                          GPU (Metal)
 ═══════════════════                            ═══════════

 ① Model forward pass (LAZY — builds graph, no GPU work)
 ┌─────────────────────────────────────┐
 │ for layer in 0..<L:                 │
 │   norm(x)        → graph node       │       (idle)
 │   qProj(x)       → graph node       │
 │   rope(q)        → graph node       │
 │   cache.update() → graph node       │
 │   sdpa(q,k,v)    → graph node       │
 │   mlp(h)         → graph node       │
 │ sample(logits)   → graph node       │
 └─────────────────┬───────────────────┘
                   │
 ② asyncEval(token) — walks graph, encodes into Metal CBs
 ┌─────────────────┴───────────────────┐
 │  CB1 (ops 1-100):                   │
 │  ┌────────────────────────────────┐ │
 │  │ set pipeline: qgemv_f16        │ │
 │  │ bind buffers (x, weights)      │ │
 │  │ dispatch                       │──────▶ GPU starts CB1 immediately
 │  │ ... (98 more ops)              │ │      while CPU encodes CB2
 │  └──────────────┬─────────────────┘ │
 │  CB2 (ops 101-200):                 │
 │  ┌────────────────────────────────┐ │
 │  │ set pipeline: sdpa_vector      │ │
 │  │ dispatch                       │ │      ┌──────────────────┐
 │  │ ...                            │ │      │ GPU executing    │
 │  └──────────────┬─────────────────┘ │      │ CB1 while CPU    │
 └─────────────────┴───────────────────┘      │ encodes CB2      │
                                              └────────┬─────────┘
 ③ .item() — CPU waits for final result               │
 ┌─────────────────────────────────────┐               │
 │ block until GPU finishes CB2  ◄─────────────────────┘
 │ read sampled token from GPU buffer  │
 └─────────────────────────────────────┘
```

### 3.4 The pipeline problem

In the ideal pipeline, CPU graph-building for token N+1 overlaps with GPU execution of token N. In our current shape, `.item()` blocks the CPU mid-loop, so the GPU goes idle between tokens while the CPU rebuilds the graph:

```
CPU: [build][sub][████ WAIT ████][build][sub][████ WAIT ████]
GPU:              [████ exec ████]            [████ exec ████]
                                  ↑                          ↑
                                  GPU idle while CPU builds next graph
```

Two ways out:
- **Pre-encoded decode loops (ICBs).** Pre-encode the per-token op chain once; replay with parameter updates each token. Eliminates per-token encoding cost entirely (~2 ms).
- **Double-buffered graph build.** Start building token N+1's graph in a background thread while the GPU still executes token N. Trades complexity for overlap.

### 3.5 Command buffers and peak memory

Metal keeps every buffer referenced by a CB alive until the CB completes. MLX batches multiple ops per CB (controlled by `max_ops_per_buffer`). The total number of allocations is the same regardless of batch size — what changes is **how many are alive simultaneously**.

| `max_ops_per_buffer` | Decode | Prefill peak memory |
|---|---|---|
| 25 (frequent commits) | -11% throughput | lowest peak |
| 100 (default) | baseline | moderate |
| 300 | +5% throughput | high peak — prefill can OOM |
| 500 | plateaus | very high peak |

**The actionable knob is memory-based commit triggers.** Track total referenced bytes (inputs + outputs); commit early during prefill when large tensors accumulate; let it run high during decode where intermediates are tiny.

---

## 4. Kernel Optimization Strategies for FFAI

This is the prescriptive section — apply these patterns when porting or writing a new kernel.

### 4.1 Pick the right dispatch mode

**Grid3D** if the kernel is "one thread per output element," no cross-thread cooperation:

```rust
#[kernel] pub fn mul<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let i = program_id::<0>();
    store(out[i], load(a[i]) * load(b[i]));
}
// dispatch grid=[1,1,1] tg=[N,1,1] OR grid=[ceil(N/TPG),1,1] tg=[TPG,1,1]
```

**Reduction** if the kernel needs `simd_sum` / `simd_max` / `threadgroup_alloc` / `threadgroup_barrier`. The kernel reads `program_id<0>` as the **threadgroup index** (`tgid_x`), and threads within each TG cooperate via simdgroup intrinsics and shared memory:

```rust
#[kernel] pub fn rms_norm<T>(x: Tensor<T>, ...) {
    let row = program_id::<0>();  // = tgid_x
    let base = row * n + tid * 4;
    // ... reads x[base..base+3], reduce_sum across the TG, scale ...
}
// dispatch grid=[rows,1,1] tg=[n/4, 1, 1]
```

**The mistake to avoid:** dispatching `grid=[N,1,1] tg=[N,1,1]` for a Grid3D kernel — that's `N²` threads in flight, most of them garbage. Confirmed via the 2026-05-19 conv1d_causal_step test (see `post-mortem-2026-05-19-dispatch-shape-gpu-freeze.md` §11 for the systemic version).

### 4.2 Respect simdgroup boundaries

If you use *any* `simd_*` intrinsic:
- **TPG must be a multiple of 32.** A partial simdgroup has undefined participation.
- **Cross-simdgroup reduction** needs `threadgroup_barrier()` + threadgroup-shared memory, not `simd_*` alone. Pattern: each simdgroup reduces via `simd_sum`, writes its partial to `threadgroup_alloc("scratch", n_simd)`, barrier, then any thread reads the partials and combines.

If you don't use simdgroup intrinsics and use Grid3D mode, you can dispatch any TPG up to 1024.

### 4.3 Maximize occupancy

Occupancy = (active simdgroups per core) / (max simdgroups per core). Bounded by:

1. **Register pressure.** Each thread's locals (including loop-carried `let mut acc = 0.0f32` accumulators) consume registers. ~210 KB / core total. Big per-thread footprint → fewer simdgroups in flight.
2. **Threadgroup memory.** ~32 KB / core. Big `threadgroup_alloc` → fewer simdgroups in flight.

Rules of thumb:
- Prefer `simd_sum` to `threadgroup_alloc + barrier` when both express the reduction — simd intrinsics use hardware buffers, not threadgroup memory.
- Keep loop-carried state lean. A `[1024]` float accumulator burns 4 KB/thread.
- On M5+, route matrix accumulators to the matrix unit via `MMATile` / `tile_matmad` ops — they don't compete for general registers.

### 4.4 Fuse aggressively

Per-op CPU cost is 5–10 µs. Every op you eliminate is a measurable win.

High-leverage fusions:
- **NormRoPE** — RMSNorm + RoPE in one kernel. Already done; saves ~60 ops on a Gemma4 E2B token.
- **NormResidual** — RMSNorm + residual add. Saves ~30 ops.
- **GEGLU / SwiGLU** — gate × silu(up). Saves ~30 ops.
- **QKV projection** — three GEMVs as one kernel. Saves ~60 ops on a model with 30 layers.
- **MoE gather + GEMV** — pre-gathered expert weights × hidden. Already done in `dequant_gemv`.

Each fusion adds register pressure; check occupancy doesn't drop below the unfused baseline.

### 4.5 Quantize the KV cache, not just the weights

Decode is bandwidth-bound. Weights are 4-bit already; the KV cache is often the next-largest bandwidth consumer. AURA (FFAI's bit-pack codec) compresses K and V to 2–8 bits with a rotated-codebook quantizer; long-context attention reads compressed K/V from DRAM and dequantizes on-the-fly in the attention kernel. See `papers/aura-compression-algorithm.md`.

Rough numbers: at 16K context, Gemma4 E2B's full-precision KV cache is ~3 MB / layer / token (read once per token). AURA at 4-bit drops that 4×.

### 4.6 Use the ANE for parallelizable ops

The Neural Engine runs in parallel with the GPU. Anything you can move off-GPU is a strict win for tok/s — no GPU cycles, no GPU bandwidth, no memory pressure.

Good candidates:
- **LM head (hidden → logits)**: 402M multiply-accumulates for Gemma4 (1536 × 262K vocab), runs once per token after all transformer layers complete. Pipeline: GPU finishes transformer → hidden state copied to ANE via IOSurface → ANE computes logits + softmax → CPU samples → GPU starts next token's layer 0. Caveat: ANE has a 32K channel limit; large vocab requires tiling (262K / 32K ≈ 9 tiles).
- **PLE (per-layer embeddings)** in Gemma4 E2B: a matmul that runs every step and could compute layer N+1 while the GPU runs layer N's attention.
- **Softmax**: 33.8× faster on ANE than CPU per Orion benchmarks.

Costs:
- IOSurface round-trip is ~2.3 ms via CoreML (high); ~0.1 ms via direct `_ANEClient`.
- `_ANEClient` is a private API — pin macOS version compatibility.
- ANE working set must fit in its 32 MB SRAM or performance drops ~30%.

Decision criterion: pursue ANE offload if parallel offloading achieves > 5% per-token latency reduction with < 1 week of implementation effort.

### 4.7 Cap command buffers by memory, not just op count

Default `max_ops_per_buffer = 100` works fine for decode (tiny intermediates) but hurts at high op counts. Default `max_mb_per_buffer` exists but only tracks input buffer sizes — output allocations are invisible. **Prefill's big tensors are outputs**, so the memory trigger doesn't fire when it should.

Fix: track total referenced bytes (inputs + outputs) and use that as the commit trigger. Set ops/CB high (300–500) for decode speed; let a ~200 MB memory limit force early commits during prefill.

### 4.8 Indirect Command Buffers (ICBs)

For decode loops with fixed shape, pre-encode the entire per-token op chain into an ICB once. Each token is then a parameter update (new token id, new KV position) + replay. Eliminates the ~2 ms per-token encoding cost entirely.

Trade-off: locks the decode-graph shape. Works for typical greedy / temperature decode; harder when shape varies (speculative decoding with variable accept length, dynamic batching).

### 4.9 Watch for asyncEval memory-pressure stalls

MLX checks memory on every op during the encoding tape walk:

```cpp
if (get_active_memory() > get_memory_limit()) {
    gpu::finalize(s);
    scheduler::wait_for_one();   // ← CPU BLOCKS waiting for GPU to free memory
}
```

If active memory exceeds the memory limit (set at allocator init to roughly `min(1.5 × recommendedMaxWorkingSetSize, 0.95 × totalSystemMemory)`), `asyncEval` stops encoding and waits for the GPU to finish. The "async" eval becomes synchronous mid-encoding. On a 64 GB machine the limit is ~60 GB; on a 32 GB machine ~30 GB.

If you see asyncEval times that don't track CPU graph-build work, this is the first thing to check.

---

## 5. FFAI / metaltile Conventions

### 5.1 DISPATCH INVARIANTS blocks

Every reduction-mode kernel in `metaltile-std/src/{ffai,mlx}/` carries a `## DISPATCH INVARIANTS` docblock at the top of its `.rs` source. Example from `sdpa_decode.rs`:

```rust
//! ## DISPATCH INVARIANTS
//!
//! - TPG = 1024 threads (32 simdgroups × 32 lanes). Smaller TPG
//!   makes n_simd = TPG/32 = 0, infinite-loop in the K walk.
//! - head_dim == 128. Each lane owns 4 consecutive Q/K/V elements;
//!   loads are unconditional.
//! - Grid: 1 threadgroup per q_head. Wrapper uses
//!   grid=(nQHeads*1024, 1, 1), tg=(1024, 1, 1).
//! - nQHeads % nKVHeads == 0.
//! - n_kv ≤ kv_stride.
```

Consumers (FFAI's `OpsValidation`, GPU correctness tests) cite this block in their error messages. New reduction kernels MUST include one.

### 5.2 metaltile `dispatch_with_grid` semantics

`Context::dispatch_with_grid(kernel, buffers, constexprs, grid_xyz, tg_xyz)` calls `dispatchThreadgroups_threadsPerThreadgroup(grid_xyz, tg_xyz)`. So:

- `grid_xyz` is in **threadgroups**, not threads. Total threads = `grid.x * grid.y * grid.z * tg.x * tg.y * tg.z`.
- For Grid3D mode: `program_id<i>()` lowers to `gid.{x,y,z} = thread_position_in_grid.{x,y,z}` — *thread index*, not threadgroup index. So for N total threads, dispatch `grid=[1,1,1]` and `tg=[N,1,1]` (or split across grid + tg, but the product must be exactly N).
- For Reduction mode: `program_id<i>()` lowers to `tgid_{x,y,z} = threadgroup_position_in_grid` — *threadgroup index*. So for N reductions of TPG threads each, dispatch `grid=[N,1,1]` and `tg=[TPG,1,1]`.

This caught us in the 2026-05-19 hardening pass; the Grid3D tests were over-dispatching `N²` threads. See post-mortem.

### 5.3 Three layers of correctness verification

When porting a new kernel:

| Layer | Catches | Location |
|---|---|---|
| **Codegen smoke** | DSL → MSL emission, `xcrun metal` accepts the output | `cargo test -p metaltile-std --lib` + `make emit-all` |
| **MSL snapshots** (insta) | Codegen pass output drift; reviewable text diffs in PRs | `crates/metaltile-codegen/tests/msl_snapshots.rs` |
| **GPU correctness** | Numerical disagreement vs naive CPU reference, on real Metal | `crates/metaltile-std/tests/<kernel>_gpu_correctness.rs` |
| **MLX side-by-side** (bench) | Numerical parity vs upstream MLX kernel | `make bench-vv` |

The first three are CI-runnable; the bench is local-only (needs MLX checkout). For an FFAI-only kernel (no MLX counterpart), MLX side-by-side doesn't apply — rely on the first three plus the FFAI integration test.

### 5.4 Wrapping a kernel in FFAI

`Sources/FFAI/Ops.swift` is the consumer-facing Swift API over `MetalTileKernels.*`. For reduction-mode kernels, the wrapper MUST:

1. Cite the kernel's `DISPATCH INVARIANTS` block as a source of truth.
2. Encode every invariant as a `precondition` via the matching `OpsValidation.validate*` function.
3. Compute dispatch geometry from invariants, never from `elementwiseGrid`.

See `Sources/FFAI/OpsValidation.swift` for the pattern and `FFAI/CLAUDE.md §"Wrapping kernels in FFAI"` for the longer-form rationale.

---

## 6. Per-Kernel Optimization Tiers

When the question is "how good is this kernel right now," it usually falls into one of these tiers. Promote up the ladder when bench evidence justifies the next tier's cost.

### Tier 0 — Correctness

- Naive algorithm, matches a CPU reference within fp32 noise.
- DISPATCH INVARIANTS block + Swift wrapper precondition.
- GPU correctness test against naive CPU reference.
- Codegen smoke green (`make emit-all`).

**Acceptance criterion:** integration test on a real model produces coherent output.

### Tier 1 — Occupancy

- TPG chosen to maximize active simdgroups per core (typically full-occupancy point given register usage).
- Cross-thread cooperation via simdgroup intrinsics (`simd_sum`, `simd_broadcast`) rather than `threadgroup_alloc + barrier` where both express the reduction.
- Per-thread loop-carried state minimized.

**Acceptance criterion:** within ~2× of theoretical bandwidth floor for the kernel's I/O pattern (measure via `make bench-vv`).

### Tier 2 — Fusion

- Combined with adjacent ops that share inputs.
- Common pairs: norm+rope, norm+residual, gate+silu+mul, q_proj+k_proj+v_proj.
- Output kept in registers between phases where shape allows.

**Acceptance criterion:** measurable tok/s improvement on a real model, justified by op-count reduction.

### Tier 3 — Hardware-specific

- M5 matrix-unit routing for accumulator-heavy reductions (use `MMATile` / `tile_matmad`).
- Larger tiles on M5 where the larger register file + matrix unit allows.
- Runtime arch detection if both M1 and M5 are supported.

**Acceptance criterion:** ≥ 1.5× improvement on M5 over the Tier-2 version, no regression on M1.

### Tier 4 — Dispatch-mode

- Argument-buffer dispatch (Mode 2 in our architecture): bind many buffers at once via a single `MTLArgumentBuffer`.
- ICB / pre-encoded dispatch (Mode 3): the full per-token chain encoded once at model-load.
- ANE offload for ops where parallel execution overlaps GPU work.

**Acceptance criterion:** ≥ 5% per-token latency reduction with bounded implementation effort.

---

## 7. References

External:

- Apple Developer documentation — *Metal Shading Language Specification* (TPG limits, simdgroup intrinsics, `simdgroup_matrix`)
- MLX Swift — `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp` (vector vs Steel dispatch)
- Steel attention sources — `mlx-swift/.../kernels/steel/attn/kernels/steel_attention.{h,metal}`
- Orion (ANE direct access) — <https://github.com/mechramc/Orion> and arXiv 2603.06728

Internal:

- `papers/post-mortem-2026-05-19-dispatch-shape-gpu-freeze.md` — the wrong-dispatch GPU freeze; sections §6, §11 are required reading before writing a new wrapper.
- `papers/aura-compression-algorithm.md` — KV cache codec.
- `papers/beyond-quadratic-attention-on-apple-silicon.md` — long-context attention strategies.
- `papers/speculative-decoding-on-apple-silicon.md` — Phase 8 batching + speculative decode plans.
- `FFAI/CLAUDE.md §"Wrapping kernels in FFAI"` — required-reading checklist for Ops wrapper authors.
- `metaltile/CLAUDE.md` and `metaltile/CONTRIBUTING.md` — DSL idioms and review conventions.
