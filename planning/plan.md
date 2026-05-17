# FFAI — Phased Implementation Plan

> For the visual end-to-end architecture (build pipeline, model
> loading, inference dispatch loop) see
> [`architecture.md`](architecture.md). This document is the
> *schedule*; that one is the *picture*.

**Goal:** A single-repo, dependency-light LLM inference library for Apple
Silicon, built on pre-compiled Metal kernels generated from the metaltile
Rust DSL. Drop the `mlx-swift-lm → mlx-swift → mlx-c → mlx` chain entirely.
Optimize for fastest inference with the least complexity. Longer term goal 
to also support CoreML/ANE inference. Later phase.

**Non-goals:** Training. Autograd. Distributed. CUDA. CPU inference.
Backwards compatibility with `mlx-swift-lm`.

---

## Target architecture

Three layers, all in this repo except metaltile (sibling Rust repo,
already forked at `~/Development/personal/ai/metaltile`):

1. **metaltile** (Rust) — `#[kernel]` DSL, IR, MSL codegen, CPU interpreter.
   Ships a `tile` CLI (`metaltile-cli`) whose `build --emit all`
   subcommand produces `kernels.metallib`, `manifest.json`, and
   `MetalTileKernels.swift`. Historical note: Phase 0 originally
   wired a standalone `metaltile-emit` bin for this; that pipeline
   was rehomed into `metaltile-codegen::emit` + `tile build` in
   May 2026 when upstream consolidated the build/bench/test commands
   under one CLI.
2. **MetalTileSwift** (Swift, in this repo at `Sources/MetalTileSwift/`) —
   loads the pre-compiled `.metallib`, manages a PSO cache with function
   constants, exposes typed Swift wrappers per kernel.
3. **FFAI** (Swift, in this repo at `Sources/FFAI/`) — `Tensor`
   abstraction over `MTLBuffer`, module/parameter system, model
   definitions, SafeTensors loading, KV cache, sampling, generate loop.

No JIT compilation at runtime. No Rust at runtime. No `MLX*` imports.

---

## Decisions already made

| Decision | Choice |
|---|---|
| Approach | Option B: build-time metallib emission, pure-Swift runtime |
| Swift package location | Inside FFAI (Sources/MetalTileSwift) |
| Kernel author flow | Rust `#[kernel]` → `make regenerate-kernels` runs `tile build --emit all --out Sources/MetalTileSwift` (an SPM build plugin would automate this if the `tile` binary ships via Homebrew) |
| Authoring DSL | Rust (metaltile). Swift macro frontend deferred indefinitely; metaltile IR is serde-serializable so this stays open. |
| MTLBuffer access | Direct (we own the tensor type, no MLXArray middle-man) |
| Testing | 100% line coverage target. Every kernel, every Swift function, every model layer gets unit tests. CI gates on coverage + correctness. |
| ANE / CoreML backend | Deferred to v0.3+. Add a `mil/` codegen sibling to `msl/` in metaltile-codegen when it's time. |

---

## Quality bar

**Test coverage: 100% line coverage of FFAI + MetalTileSwift Swift code.**
Measured via `swift test --enable-code-coverage` and the `xccov` /
`llvm-cov` toolchain. CI fails any PR that drops coverage below the
configured threshold. Phase 0 sets up the tooling; every subsequent
phase adds tests alongside code, not after.

**What "100%" means here:**

- Every public function: at least one happy-path test
- Every branch in business logic: at least one test per side
- Every kernel in metaltile: numerical correctness test (CPU
  interpreter reference) + Swift wrapper integration test (fixed
  inputs → fixed outputs)
- Every model: token-by-token determinism test against a reference
  for at least one prompt + seed combination

**What it doesn't mean:**

- Mocking out the GPU. Tests run real Metal dispatches on the CI
  runner (Apple Silicon).
- Property/fuzz testing — out of scope for v0.1; revisit later.
- Defensive-error paths that can't actually be triggered (Swift
  `fatalError` on programmer bugs etc.) are excluded from the
  coverage denominator via `// coverage:ignore` markers.

**Test layout:**

```
Tests/
  MetalTileSwiftTests/   # one test file per kernel
  FFAITests/             # Tensor, Module, Linear, KVCache, Sampling, …
  ModelTests/            # one folder per model (Llama, Qwen3, …) with
                         # forward-pass determinism + token-output tests
```

Every PR that adds production code without corresponding tests is
rejected at review. CI publishes coverage diff per PR.

---

## Model architecture conventions

**One file per family**, not per variant. A "family" is a major
architectural lineage: `Qwen3.swift` covers Qwen3, Qwen3.5 dense,
Qwen3.5 MoE, Qwen3.5-VL, Qwen3.5-Omni, Qwen3.6, etc. Different major
generations get separate files (Qwen2 ≠ Qwen3).

Each family file uses **protocol + per-variant struct** internally so
adding a variant doesn't bloat a giant switch statement. See
`architecture.md` for the diagram.

**Capability-driven loading.** Multi-modal models declare
capabilities (`.textIn`, `.textOut`, `.visionIn`, `.audioIn`,
`.audioOut`, `.toolCalling`). At load time the user picks which to
enable; disabled modalities skip weight allocation entirely. Vision
encoder of a 9B VL model is ~600MB you don't pay for if you only need
text.

**Async lifecycle with progress.** Loading is async and observable:

```
ModelLifecycleState:
  idle → downloading(Progress) → loading(LoadProgress)
       → loaded → ready
       (or failed(Error) at any stage)
```

The `Model` exposes an `AsyncStream<ModelLifecycleEvent>` for UI
progress bars and observability. `enable(_ capability:)` /
`disable(_ capability:)` are async and emit per-capability lifecycle
events through the same stream.

Full API surface lands in **Phase 2** (with only `.textIn`/`.textOut`
exercised), gets stress-tested in **Phase 6** with the first VL model.

---

## Model file formats

| Format | What | When |
|---|---|---|
| **safetensors** (fp16/bf16) | HF default. Header (JSON) + raw tensor bytes. mmap-friendly. | **Phase 2** — required |
| **mlx-format** (quantized safetensors) | Same file format; mlx-community quant layout (weight + scale + zero per group). | **Phase 3** alongside our quant kernels |
| **gguf** | llama.cpp single-file format, embeds quant + tokenizer, different naming convention. | **Phase 7+** if there's user demand |
| **onnx** | Graph format, embedded weights. Wrong fit — would need a graph executor. | **Skip.** Doesn't align with static-kernel approach. |

---

## Dispatch modes

Three CPU-side dispatch strategies, selectable per-load via
`LoadOptions.dispatchMode`. We architect for all three from Phase 2
but only ship Mode 1 initially. See `architecture.md §4a` for the
diagrams and tradeoffs.

| Mode | What | Phase |
|---|---|---|
| `.eager` | Standard MTLComputeCommandEncoder per kernel call. Simple, debuggable. | **Phase 2 — default and only mode** |
| `.argumentBuffers` | Per-layer argument buffers pre-bind weights. Per-token, only activations + KV offset are bound. ~5x fewer setBuffer calls. | **Phase 5** if profiles justify |
| `.icb` | Indirect Command Buffer pre-records the entire forward pass. Per-token, only an arg buffer update + one execute. | **Phase 5+** if `.argumentBuffers` isn't enough |

**Architecture invariants we maintain from day 1 to keep all three
modes viable:** (1) weight MTLBuffers are immutable post-load,
(2) activation MTLBuffers come from a `BufferPool` with stable
handles, (3) no CPU readback during forward pass — sampling runs on
the GPU, (4) fused kernels for hot paths to minimize memory bandwidth.

---

## Phase 0 — Plumbing (this repo + metaltile)

**Goal:** Round-trip a single trivial kernel from Rust `#[kernel]` to
Swift dispatch, with the SPM build plugin auto-invoking the emit step.

**Deliverables:**

- `metaltile-emit` bin in the metaltile workspace
  - Walks a registered set of kernels (initially: `vector_add`, `rms_norm`)
  - Runs `MslGenerator` to produce one `.metal` file per kernel
  - Shells out to `xcrun -sdk macosx metal` and `xcrun metallib` to build
    `kernels.metallib`
  - Writes `manifest.json` with per-kernel metadata (name, params with
    dtype/shape kind, function constants, default grid/threadgroup rule)
  - Generates `MetalTileKernels.swift` with one strongly-typed func per
    kernel (function-constant specialization keyed in PSO cache)
- FFAI `Package.swift`
  - Targets: `MetalTileSwift`, `FFAI`, `FFAITests`
  - Build plugin `MetalTileEmitPlugin` runs `cargo run -p metaltile-emit`
    against the local metaltile path before each build
  - Resources: `kernels.metallib`, `manifest.json` bundled into
    `MetalTileSwift`
- `Sources/MetalTileSwift/`
  - `MetalTileLibrary.swift` — singleton MTLDevice + MTLLibrary loader
  - `PSOCache.swift` — keyed by `(name, MTLFunctionConstantValues)`
  - `Resources/` — generated artifacts
  - `Generated/MetalTileKernels.swift` — generated typed wrappers
- `Sources/FFAI/` gains the **model download/cache plumbing** (no
  inference yet, just file fetching):
  - `ModelDownloader.swift` — thin wrapper over `swift-huggingface`'s
    `HubClient.downloadSnapshot(of:revision:matching:)`. Caches in the
    standard `~/.cache/huggingface/hub/` location, supports glob
    patterns (e.g. `["*.safetensors", "*.json", "*.jinja"]`), resumes
    partial downloads, returns a local directory URL.
  - `ModelLocator.swift` — accepts either a HF repo ID
    (`"meta-llama/Llama-3.2-1B"`) or a local path. Resolves to a
    local directory.
  - `TokenizerLoader.swift` — thin wrapper over
    `swift-transformers`'s `AutoTokenizer.from(pretrained:)`, sourced
    from the resolved directory.
  - `ModelConfig.swift` — decodes `config.json` (architecture name,
    hidden_size, num_layers, etc.) into a typed Swift struct
- `Tests/MetalTileSwiftTests/` — round-trip test for `vector_add` and
  `rms_norm` against fixed inputs/outputs
- `Tests/FFAITests/ModelDownloaderTests.swift` — download `config.json`
  only from a known small HF repo (e.g. `meta-llama/Llama-3.2-1B`),
  verify cache hit on second call, verify pattern filter excludes
  weight files, verify offline fallback when cache exists
- **Test infrastructure:**
  - `swift test --enable-code-coverage` wired up
  - Coverage threshold enforced in CI (start at 100% for the small
    Phase 0 surface; subsequent phases maintain it)
  - `scripts/coverage.sh` to print local coverage summary
- **CI:**
  - `.github/workflows/ci.yml` — Apple Silicon runner, runs `swift
    test`, uploads coverage report, fails on coverage drop
  - `.github/workflows/auto-label.yml` — conventional-commit PR
    labeling (adapted from mlx-swift-lm)
  - `.github/release.yml` — release notes categorization
  - `.github/pull_request_template.md` — adapted from mlx-swift-lm
- **Dev tooling (ported and adapted from mlx-swift-lm):**
  - `scripts/setup-dev.sh` — verify Xcode CLI tools, `xcrun metal`,
    Swift, Cargo, and the sibling metaltile checkout; resolve SPM
    deps; run first build to populate metallib
  - `scripts/verify-docs.sh` — `swift package generate-documentation
    --warnings-as-errors` over each library target
  - `scripts/coverage.sh` — local coverage summary via `llvm-cov`
  - `Makefile` — common targets: `build`, `test`, `clean`,
    `regenerate-kernels`, `coverage`, `format`
  - `.swift-format` — copied verbatim from mlx-swift-lm
  - `.pre-commit-config.yaml` — copied verbatim
  - `.spi.yml` — Swift Package Index docs config, FFAI targets
  - `.gitignore` — already in place
  - `LICENSE` — Apache-2.0
- **Testing reference convention (decided up front):**
  - Numerical references for tests are **golden fixtures**, not live
    Python/PyTorch invocations.
  - `Tools/capture-fixtures.py` — Python script (only used to
    *generate* fixtures, never run during `swift test`). Uses
    `mlx-lm` for text-only families and `mlx-vlm` for vision-language
    families (closer architectural match than PyTorch for the model
    variants we target). `mlx-vlm` lists `mlx-lm` as a runtime
    dependency, so a single `pip install mlx-vlm` covers both
    backends; the script picks the right one per model. Writes
    activations + token sequences to `Tests/Fixtures/<model>/`.
  - Tests load the fixtures and compare with tolerance.
  - When a fixture needs regeneration: developer runs the capture
    script locally on a verified setup, commits the new files. The
    fixture file's `metadata.json` records the `mlx-lm` / `mlx-vlm`
    version + capture date.
  - **Result:** zero Python dependency for `swift test`, fully
    reproducible CI on a stock Apple Silicon runner.

**Done when:** `swift test` runs the build plugin (which runs cargo,
which produces the metallib), loads the metallib, dispatches `vector_add`
and `rms_norm` against known inputs, verifies output, **downloads
`config.json` for Llama 3.2 1B from HF and parses it**, and reports
100% line coverage on the Phase 0 Swift surface. End-to-end with zero
MLX involved.

**Open items for Phase 0:**

- Confirm SPM build plugin can shell out to `cargo` and access the
  sibling metaltile repo. If not, fall back to a Makefile target invoked
  manually before `swift build`.
- Decide manifest.json schema (versioned).
- Confirm `xcrun metal` toolchain availability assumption (Xcode required).
- Decide coverage tooling: `xccov` (Xcode-native, macOS-only) vs
  `llvm-cov` via `swift test --enable-code-coverage` (portable). Default
  to the latter unless we hit limitations.

---

## Phase 1 — Foundation kernels

**Goal:** Have enough kernels in metaltile + Swift wrappers to express a
basic Llama-style forward pass without quantization.

**Kernels to port from existing metaltile bench/ops + add Swift wrappers:**

- `rms_norm` (already in metaltile-bench)
- `layer_norm`
- `rope` (Llama3 variant; SuScaled, Proportional later)
- `silu`, `gelu`, `relu` (activation primitives)
- `softmax`
- `matmul` / `gemv` (fp16, bf16, fp32) — already in bench
- `scaled_dot_product_attention` (vector decode variant exists; tile
  prefill variant may need work)
- `gather` (embedding lookup)
- Element-wise add, mul (residual + gating)
- **GPU sampling kernels** (avoid the per-token logits→CPU readback):
  - `sample_argmax` — greedy
  - `sample_temperature` — scale logits by 1/T then sample
  - `sample_top_k` — top-k mask + sample
  - `sample_top_p` — cumulative-softmax + threshold + sample
  - All output a single token id (4 bytes), no logits readback

**Deliverables:**

- All above registered in `metaltile-emit` registry
- Each gets a generated Swift wrapper
- `Sources/FFAI/` gains:
  - `Device.swift` — MTLDevice + MTLCommandQueue singleton
  - `Tensor.swift` — `MTLBuffer` + shape + dtype + strides + byte offset
  - `BufferPool.swift` — slab allocator for activation tensors
  - Bare-minimum `Module` protocol with named parameter discovery
- **Tests (mandatory, ship with code):**
  - One test file per kernel in `Tests/MetalTileSwiftTests/`, covering
    fp32 + fp16 + bf16 dtypes, edge shapes, and numerical accuracy vs
    a Swift CPU reference (or fixed reference vectors)
  - `Tests/FFAITests/TensorTests.swift` — shape, dtype, stride, slice,
    reshape correctness
  - `Tests/FFAITests/BufferPoolTests.swift` — alloc/free, fragmentation,
    reuse semantics
  - `Tests/FFAITests/ModuleTests.swift` — parameter discovery on nested
    modules
  - 100% line coverage maintained

**Done when:** A hand-built Llama-shaped layer (RMSNorm → QKV → RoPE →
SDPA → out-proj → RMSNorm → gate/up/down → residual) runs end-to-end
on randomly-initialized weights and produces stable outputs. Numerical
accuracy verified against a golden fixture (captured from mlx-lm,
see Phase 0 testing reference convention). All tests pass, coverage
at threshold.

---

## Phase 2 — First model end-to-end (Llama 3.2 1B)

**Target model: Llama 3.2 1B.** ~1.2GB fp16, standard transformer
(GQA + RoPE + RMSNorm + SwiGLU MLP). Maps 1:1 to the Phase 1 kernel
list. Reference implementations exist in PyTorch, llama.cpp, MLX, and
mlx-swift-lm — token-by-token cross-checks are easy. No exotic
features (no MoE, no q/k norm, no attention sinks, no hybrid
recurrent layers).

**Goal:** Load a real fp16/bf16 Llama 3.2 1B checkpoint and generate
text. No quantization yet.

**Deliverables:**

- `Sources/FFAI/SafeTensors.swift` — SafeTensors loader →
  `[String: Tensor]` (memory-mapped where possible)
- `Sources/FFAI/Module.swift` — full Module/parameter system, weight
  loading via key-path mapping
- `Sources/FFAI/Linear.swift`, `Embedding.swift`, `RMSNorm.swift` —
  basic layer types calling MetalTileSwift kernels
- **Model API (capability + lifecycle infrastructure)** — even though
  Phase 2 only exercises text-in/text-out, the API surface lands now:
  - `Sources/FFAI/Capability.swift` — Capability enum
  - `Sources/FFAI/LoadOptions.swift` — capability set, kvCache kind,
    precision, lazyCapabilities flag
  - `Sources/FFAI/ModelLifecycle.swift` — `ModelLifecycleState`,
    `ModelLifecycleEvent`, `LoadProgress`
  - `Sources/FFAI/ModelFamily.swift` — protocol for family files
  - `Sources/FFAI/Model.swift` — concrete `Model` class with
    `events: AsyncStream<ModelLifecycleEvent>`, `enable(_:)`,
    `disable(_:)`, `prewarm()`
  - `Sources/FFAI/ModelRegistry.swift` — config.architectures /
    config.model_type → `ModelFamily.Type` lookup
- `Sources/FFAI/Models/Llama.swift` — first family file. Uses the
  protocol+per-variant convention even with one variant (`LlamaDense`)
  so the pattern is established
- `Sources/FFAI/KVCache.swift` — fp16/bf16 raw cache, append + slice
- `Sources/FFAI/Sampling.swift` — argmax, top-k, top-p, temperature
- `Sources/FFAI/Generate.swift` — generate loop, prefill + decode
- `Sources/FFAICLI/main.swift` — CLI: `ffai --model <id-or-path> --prompt "…"`
  (uses Phase 0's `ModelDownloader` + `TokenizerLoader`; no new HF code)
- **Tests:**
  - `Tests/FFAITests/SafeTensorsTests.swift` — load + roundtrip a small
    synthetic file
  - `Tests/FFAITests/SamplingTests.swift` — argmax/top-k/top-p with
    fixed seeds
  - `Tests/FFAITests/KVCacheTests.swift` — append, slice, eviction
  - `Tests/FFAITests/ModelLifecycleTests.swift` — state machine
    transitions (idle → downloading → loading → loaded → ready, plus
    error paths from each), AsyncStream event ordering
  - `Tests/FFAITests/CapabilityTests.swift` — load with subset of
    capabilities, verify others not loaded; `enable` / `disable`
    round-trip (uses a synthetic model since Llama is text-only)
  - `Tests/ModelTests/Llama/LlamaForwardTests.swift` —
    randomly-initialized layer numerical match vs golden fixture
    (captured from mlx-lm)
  - `Tests/ModelTests/Llama/LlamaGenerateTests.swift` — token-by-token
    determinism on a real Llama 3.2 1B checkpoint with a fixed prompt
    and seed (small reference fixture committed to repo, or downloaded
    via test setup script)
  - 100% line coverage maintained

**Bootstrap from mlx-swift-lm:**

Copy and adapt (do NOT preserve git history; treat as new code):

- Tokenizer integration patterns from `Libraries/MLXLMCommon/Tokenizers.swift`
- Sampling code (top-k, top-p logic) from
  `Libraries/MLXLMCommon/Sampling.swift` or equivalent
- KV cache slice/append patterns from
  `Libraries/MLXLMCommon/KVCache.swift` (drop the MLX dependency)
- Generate loop from `Libraries/MLXLMCommon/Generate.swift`
- `Libraries/MLXLLM/Models/Llama.swift` as a structural template —
  rewrite all `MLXArray` ops as FFAI `Tensor` ops calling
  MetalTileSwift kernels

Do NOT copy:

- All 56 model architectures (one is enough for Phase 2)
- VLM / Embedders
- GigaQuant / state-replay (Phase 5d/5e — TurboQuant in mlx-swift-lm
  was renamed to GigaQuant in FFAI)
- Quantization (Phase 3)

**Done when:** `ffai --model llama-3.2-1B --prompt "Hello"` produces
coherent text on M-series, token-by-token output matches a reference
implementation for a fixed seed, and tokens/sec is measured and
recorded as baseline.

---

## Phase 2.5 — Second model (Qwen3 4B)

**Target model: Qwen3 4B.** Same kernel set as Llama plus one
structural addition: per-head q_norm and k_norm RMSNorms applied to
queries/keys *before* RoPE (see
`Libraries/MLXLLM/Models/Qwen3.swift:74` in mlx-swift-lm). No new
kernels needed — just an extra RMSNorm site in the attention block.

**Goal:** Validate that the Phase 2 kernel set + module system
generalize to a second modern architecture without core changes. By
the time we hit Phase 2.5, RMSNorm/RoPE/SDPA/reshape paths are
battle-tested, so adding q_norm/k_norm becomes a small structural
edit, not a debugging session.

**Deliverables:**

- `Sources/FFAI/Models/Qwen3.swift` — second family file. Adds
  `Qwen3Dense` variant. Sets the precedent for the protocol +
  per-variant struct pattern (later variants —
  `Qwen35HybridDense`, `Qwen35MoE`, `Qwen35VL`, `Qwen35Omni` —
  land in their respective phases).
- Any tokenizer/config quirks for Qwen3 (different chat template,
  vocab size, etc.) — handled by the family's Variant detection
- Updated `ModelRegistry` entries for Qwen3 model_type strings
- **Tests:**
  - `Tests/ModelTests/Qwen3/Qwen3ForwardTests.swift` — q_norm/k_norm
    application correctness
  - `Tests/ModelTests/Qwen3/Qwen3GenerateTests.swift` — token-by-token
    determinism
  - 100% line coverage maintained

**Done when:** `ffai --model qwen3-4B --prompt "…"` generates
coherent text and matches a reference implementation token-by-token
for a fixed seed.

**Philosophy from this point on:** core kernels and Tensor/Module
infrastructure are stable. Adding a new model = porting its
forward-pass shape from mlx-swift-lm and wiring it to existing
kernels. New *model-specific* kernels (e.g. attention sinks for
GPT-OSS, fused MoE expert kernels, GDN steps, GigaQuant codecs) get
added to the metaltile DSL as needed — driven by which model we want
to support next, not speculatively.

---

## Phase 3 — Quantization

**Goal:** Match MLX's standard 4-bit / 8-bit affine quantization for
weight-only quantized inference.

**Kernels:**

- `affine_quantize` / `affine_dequantize` (group-wise, scale + zero)
- `quantized_gemv` (4-bit, 8-bit, group sizes 32/64/128)
- `rms_norm_quantized_gemv` (fused — matches MLX's perf trick)
- `batched_qkv_qgemv` (fused QKV from RMSNormed input)

**FFAI changes:**

- `QuantizedLinear` layer
- Weight loading path that recognizes **mlx-format** pre-quantized
  safetensors (mlx-community layout: weight + scale + zero per group,
  MLX naming conventions). Same file format as plain safetensors,
  just different *contents*. Detect at load time, route to quantized
  GEMV kernels.
- `Sources/FFAI/SafeTensors.swift` extended with mlx-format detection
- Memory savings verified

**Done when:** A 4-bit quantized Llama runs and produces same-quality
output as the fp16 baseline at significantly lower memory + faster
decode.

---

## Phase 4 — Performance optimizations

**Goal:** Close the gap between Phase 0-3 correctness-first kernels
and the M-series GPU's actual ceiling. The Phase 0-3 implementation
hit ~5 tok/s on Llama 3.2 1B / 4-bit Qwen3 4B; M1 Max hardware
should support 80-200 tok/s on these workloads.

**Profile-driven targets, in priority order:**

1. **Eliminate per-layer CPU↔GPU sync.** The Phase 2 KV cache append
   is a CPU memcpy from rotated K/V into the cache buffer, requiring
   `cmd.commit(); cmd.waitUntilCompleted()` mid-layer. Replace with a
   `kv_cache_update` Metal kernel; never sync inside a layer.
2. **Single MTLCommandBuffer per token.** All layers + sampling on
   one command buffer; one commit + wait per token (down from
   ~30-70 currently).
3. **GPU sampling.** Argmax / top-k / top-p / temperature kernels.
   Eliminates the per-token logits→CPU readback (~50-300 KB
   transfer + sync). Only the chosen token id (4 bytes) crosses
   CPU↔GPU.
4. **Activate the BufferPool.** Per-token activations come from
   `BufferPool.shared` instead of fresh `Tensor.empty(...)` per op.
   Drop pool at end of each token. ~100-150 fewer `makeBuffer` calls
   per token.
5. **Cooperative-thread gemv.** Replace `gemv_naive` (one thread per
   output row, serial inner loop) with the strided-reduce-dot pattern
   from `metaltile-bench/src/ops/gemv.rs` (one threadgroup per row,
   simd_sum reduction across the in_dim axis).
6. **Multi-row RMSNorm.** Qwen3's per-head q_norm/k_norm currently
   dispatches one `rmsNorm` per head (32 + 8 launches per layer ×
   36 layers = 1440 launches per token). Replace with a single
   multi-row dispatch.
7. **Better SDPA decode.** Port the bench's online-softmax SDPA
   (`metaltile-bench/src/ops/scaled_dot_product_attention.rs`) — one
   simdgroup per Q head with cooperative reduction, instead of the
   naive per-thread recompute-everything approach.

**Targets:**

- Llama 3.2 1B bf16: ≥ 60 tok/s (12× current)
- Qwen3 4B bf16: ≥ 30 tok/s (18× current)
- Qwen3 4B 4-bit: ≥ 100 tok/s (20× current)

**Tests:**

- Synthetic correctness for `kv_cache_update`, GPU sampling kernels,
  cooperative gemv (all against existing CPU/CPU-shadowed reference)
- All existing integration tests must continue to pass with the new
  dispatch path
- New `Tests/PerfTests/` measures tok/s at fixed seed; CI publishes
  the numbers per commit (regression-tracker)

**Out of scope (deferred to Phase 7 autotuner):**

- Per-shape kernel parameter selection
- Argument buffer / ICB dispatch modes
- Multi-token batched prefill

---

## Phase 5 — Advanced kernels (sampling, GigaQuant, GDN, SSM)

**Goal:** Port the high-value custom kernels currently in mlx-swift-lm
plus close the user-visible sampling gap. The custom kernels were
the original motivator for this project — the 4-repo dance to ship
a new GigaQuant variant (TurboQuant in mlx-swift-lm, renamed to
GigaQuant in FFAI) is the pain we're eliminating.

Phase 5 is now sub-divided since the full scope is many sessions.
Each sub-phase ships independently with its own commit + verification.

### Phase 5a — Sampling pipeline ✅ SHIPPED

CPU sampling pipeline (temperature, top-K, top-P, min-P, repetition
penalty, seeded reproducible RNG) wired through
`GenerationParameters` + CLI flags. Greedy stays on the GPU argmax
fast path; non-greedy reads logits to CPU once per token and runs
the pipeline there. ~30% decode-tok/s tax vs greedy.

Done: `feat(phase-5): CPU sampling pipeline (...)` —
[`Sources/FFAI/Sampling.swift`](../Sources/FFAI/Sampling.swift) +
`Sources/FFAI/Generate.swift` + `Sources/FFAICLI/GenerateCommand.swift`
+ `Tests/FFAITests/SamplingTests.swift` + sampling sections in
`documentation/generation-parameters.md`. Verified end-to-end
against `mlx-community/Qwen3-1.7B-4bit`: greedy deterministic at
~53 tok/s; seeded non-greedy reproducible at ~36 tok/s.

### Phase 5b — GPU softmax + categorical sample kernel ✅ SHIPPED

One fused metaltile kernel: `softmax_categorical_sample` —
cooperative 256-thread reduction (max + sum-exp) followed by a
single-threaded inverse-CDF walk. Used by the Generate decode
loop's `gpu-categorical` path for the pure-temperature case
(T > 0, no top-K / top-P / min-P / rep-penalty). Logits stay on
GPU; only the chosen token id (4 bytes) flows back. Top-K / top-P /
min-P / rep-penalty kernels deferred — those need a sort or
radix-select kernel which is a separate follow-up.

Done: metaltile `feat(phase-5b): softmax_categorical_sample kernel`
on the `ek/sampling-kernels` branch (commit `e663118`); FFAI
`feat(phase-5b): GPU softmax+categorical sample path for
pure-temperature decoding` on main (commit `507095a`). Verified
end-to-end against `mlx-community/Qwen3-1.7B-4bit` with synthetic
correctness sweeps at vocab=4, 32, and 152K (model-shaped).

**Follow-ups not yet done:**

- Per-family `forwardSampleCategorical` fusion (single cmdbuf for
  forward + sample). Default impl uses two cmdbufs today, so
  `gpu-categorical` runs at the same tok/s as `cpu-sample` — the
  perf win lands with fusion.
- Parallel prefix-scan replacement for the single-thread CDF walk
  (currently ~150µs at vocab=152K).
- GPU top-K / top-P / min-P / rep-penalty kernels.

### Phase 5c — Affine-quantized KV cache (int8 + int4) ✅ SHIPPED

int8 affine group-quantization with a shared working buffer pair
across all layers. Real memory savings on
`mlx-community/Qwen3-1.7B-4bit` at maxSeq=40960: KV cache 4.38 GB
→ 2.32 GB (−47%), peak GPU 5.28 GB → 3.38 GB (−36%), with a 7%
decode tok/s tax (46.7 → 43.6 tok/s on M1 Max).

Shipped:

- metaltile `ek/sampling-kernels` (commit 87ecbd3): two new
  kernels — `quantize_kv_int8` (one thread per group; find min/max,
  derive scale+bias, pack 4 int8 per uint32) and `bulk_dequant_kv_int8`
  (one thread per output element; reads packed weights + scales +
  biases, writes fp16/bf16 directly into the SDPA-ready layout)
- FFAI `main` (commit 5109b9b): `KVCacheProtocol` extracted from
  the existing `KVCache`; new `AffineQuantizedKVCache` conformance;
  `LoadOptions.kvCache = .affineQuantized(bits:groupSize:)`;
  `--kv-cache affine8` CLI flag; `LlamaModel` + `Qwen3Model` carry
  `kvCacheKind` and switch on it in `makeLayerCaches(...)`. All
  layers built in one `makeLayerCaches` call share a single pair of
  working buffers (Metal hazard tracking serialises the buffer reuse
  across layers within a cmdbuf — that's the architectural unlock
  that makes the memory savings real vs per-layer working buffers).

4-bit shipped: metaltile commit `67ab7a3` (quantize_kv_int4 +
bulk_dequant_kv_int4 kernels) + FFAI commit `31589b4`. Activate
via `LoadOptions.kvCache = .affineQuantized(bits: 4, groupSize: 32)`
or CLI `--kv-cache affine4`. Measured ~70% KV memory savings vs raw
on Qwen3 1.7B at maxSeq=40960 (4.38 GB → 1.37 GB) with no further
tok/s tax beyond `affine8`. Uses group_size=32 default — group_size=64
at 4-bit loses too much discriminative power and decode degenerates.

Follow-ups not yet done:

- **int6 variant** — byte-packed sub-byte storage (mirror the
  existing `dequant_gather_int6` pattern).
- **Fused `bulk_dequant + sdpa_decode`** — today each attention
  step pays one extra dequant kernel dispatch. Fusing removes
  the working-buffer materialisation.

### Phase 5d — GigaQuant compressed-domain attention

The original motivator (TurboQuant in mlx-swift-lm; **renamed
GigaQuant** in FFAI — kernel names, env vars, CLI flags, and docs
all use `giga*`). ~6-8× memory at `giga4v2`. Substantial
research-grade codec port (many sessions).

**KV scheme naming.** Schemes are named `giga{kb}v{vb}` where `kb`
and `vb` are the K-side and V-side bit widths. Symmetric aliases
(`giga3`, `giga4`, `giga6`, `giga8`) map to `giga3v3`, `giga4v4`, …
Asymmetric examples: `giga8v4`, `giga4v2`, `giga3v2`. The CLI accepts
the same strings: `--kv-cache giga4v2`.

**Metaltile DSL prerequisites** (must land in `metaltile` first):

- Sub-byte packed dtypes (`Packed4` — 4-bit, 2 per byte; `Packed2` —
  2-bit, 4 per byte) with bit-unpack ops. Extend `DType` enum.
- `simd_shuffle_xor` intrinsic (needed for FWHT butterfly).
- Function-constant integration in the typed `launch` builder
  (`#[constexpr]` already works; extend to typed launch).
- Persistent state-buffer convention (in/out aliasing the same
  `MTLBuffer`) — needed for the rotating index buffer.
- Type-checked launch builder ("shape algebra" gap — closes
  `metaltile-codegen::launch_builder`).

**GigaQuant kernels** (port from
`Libraries/MLXLMCommon/TurboQuantKernels.swift` and
`turbo_quant.metal` / `turbo_flash_sdpa.metal` upstream; the FFAI
copies are renamed `giga*`):

- `giga_encode_{kb}_{dim}` — dense rotation Π + Lloyd-Max +
  bit-pack + norm correction.
- `giga_encode_wht_{kb}_{dim}` — FWHT butterfly variant, no
  correction.
- `giga_bulk_dequant_rotated_{kb}_{dim}` — dequant into the
  SDPA-ready layout.
- `giga_score_{kb}_{dim}` + `giga_value_{vb}_{dim}` —
  compressed-domain attention reductions.
- `giga_flash_pass1_{kb}_{vb}_{dim}` + `giga_flash_pass2` (with
  causal / NR0 variants).
- `giga_flash_sdpa_v_{kb}_{vb}_{dim}` — single-dispatch fused, with
  optional attention-sinks fold (consumed by Phase 5f).
- MSE codec internals (`giga_mse_score`, `giga_mse_weighted_sum`)
  for the rotation/codebook calibration path.
- **DC-bias correction in the MSE codec.** Per-vector mean
  subtraction baked into the encode path; recovers the structured
  DC offset from `RMSNorm → Linear(bias=True)` flows that otherwise
  blows up the residual energy after rotation.

**FFAI changes:**

- `Sources/FFAI/GigaQuantizedKVCache.swift` — protocol conformance,
  per-layer compressed K/V storage, rotating index buffer, two-phase
  prefill+compress, dequant working-buffer pool shared across layers.
- `LoadOptions.kvCache = .gigaQuantized(scheme:)` where `scheme`
  parses strings of the form `giga{kb}v{vb}` and the symmetric
  aliases.
- CLI flag `--kv-cache giga4v2` (and aliases `giga3`, `giga4`,
  `giga6`, `giga8`, `giga8v4`, `giga3v2`).
- `Sources/FFAI/Models/{Llama,Qwen3}.swift` cast `kvCacheKind` and
  build `GigaQuantizedKVCache` in `makeLayerCaches`.
- `documentation/kv-cache.md` — full scheme table + bench numbers.

**Tests:**

- `Tests/MetalTileSwiftTests/Giga*` — one file per kernel,
  numerical correctness vs CPU reference.
- `Tests/FFAITests/GigaQuantizedKVCacheTests.swift` — encode/decode
  round-trip, multi-layer shared working buffer, serialize/hydrate.
- `Tests/ModelTests/GigaQuantIntegrationTests.swift` —
  `--kv-cache giga4v2` on Qwen 3 1.7B-4bit; coherent output +
  measured memory savings.

### Phase 5e — SSM / GDN hybrid models

Unlocks Qwen 3.5 (GDN + attention), Mamba 2 families (NemotronH,
GraniteMoeHybrid, FalconH1).

**Foundation shipped** (metaltile `ek/sampling-kernels` 1224890 +
FFAI main f58a801):

- `ssm_step` kernel — Mamba 2 selective-scan single-token decode.
  Per-head A, shared B/C, fp32 recurrent state h
  `[nHeads, stateDim, headDim]`. One thread per (head, channel).
  Registered for f32/f16/bf16.
- `Ops.ssmStep(...)` Swift wrapper.
- `SSMStateCache` class — per-layer recurrent state holder.
  O(1) memory per layer (state size doesn't grow with seq length —
  selective scan compresses history into fixed-size state).
- 6 correctness tests in `Tests/FFAITests/SSMStateCacheTests.swift`:
  cache plumbing (3) + single-step kernel vs CPU reference +
  12-step recurrence (catches per-step drift) + bf16-input variant.
  All pass.

**1D depthwise causal conv shipped** (metaltile a9f6787 + FFAI
ad04737):

- `conv1d_causal_step` kernel — Mamba 2 input-projection conv in
  streaming-decode form. One thread per channel; state shifts
  in-place after compute.
- `Ops.conv1dCausalStep(...)` Swift wrapper.
- `ConvStateCache` class — per-layer rolling window of
  `[K-1, nChannels]`, constant size w.r.t. sequence length.
- 7 correctness tests including 8-step sequential verification.

Both Mamba 2 building blocks (`ssm_step` + `conv1d_causal_step`) +
both caches (`SSMStateCache` + `ConvStateCache`) are now in place,
and `Models/Mamba2.swift` ships the end-to-end dense path.

**Shipped on top of those building blocks (Phase 5e initial drop):**

- `Models/Mamba2.swift` — `Mamba2` family, `Mamba2Dense` variant,
  `Mamba2Layer` mixer block (RMSNorm → in_proj → conv1d_causal_step
  + SiLU → softplus(dt) → ssm_step → D·x skip → SiLU(z) gate →
  mixer norm → out_proj), `Mamba2Model` with single-cmdbuf decode.
- `Mamba2LayerCache` bundling `SSMStateCache` + `ConvStateCache`,
  conforming to the new `LayerCacheProtocol`.
- `LayerCacheProtocol` parent extracted so SSM caches don't need
  no-op attention methods; `KVCacheProtocol` is now a sub-protocol
  with the attention surface.
- `LanguageModel.makeKVCache` → `makeLayerCaches`, returning
  `[any LayerCacheProtocol]`. Llama / Qwen3 / Mamba 2 cast back to
  their concrete cache type internally.
- `Ops.softplus` + numerically-stable `softplus_elem` kernel for
  Mamba 2's dt computation.
- `ssm_step` updated to per-head `dt[n_heads]` (Mamba 2 spec; was
  scalar in the initial kernel drop). `SSMStateCacheTests` updated
  accordingly.
- `ModelConfig.load` switched to `.json5Allowed` so the published
  Mamba 2 configs with `Infinity` literals parse.
- `TokenizerLoader` calls `AutoTokenizer.from(modelFolder:, strict:
  false)` so swift-transformers falls back to plain BPE for
  tokenizers it doesn't have a class for (Mamba 2 ships
  `GPTNeoXTokenizer`).
- Integration test: `Tests/ModelTests/Mamba2IntegrationTests.swift`
  loads `mlx-community/mamba2-130m`, verifies shapes match config,
  runs greedy decode to completion (~130 tok/s on M-series).

**Still planned for 5e (SSM / GDN hybrid foundations):**

- **GDN kernels** (port from
  `mlx-swift/Source/Cmlx/mlx-generated/metal/{gated_delta,gated_delta_replay}.metal`):
  - `gated_delta_step_{Dk}_{Dv}_{Hk}_{Hv}` — recurrence
    `S_t = g_t·S_{t-1} + β_t·k_t·(v_t − k_tᵀ·S_{t-1})ᵀ`, state in
    fp32 throughout.
  - `gated_delta_step_record_*` — same forward + tape per-step
    delta for speculative-decoding rollback.
  - `state_replay_*` — re-fold the delta-log tape on partial accept.
- **SSM kernels** (extend the shipped `ssm_step`):
  - `ssm_step_record_*` — Mamba 2 forward + tape.
  - `ssm_replay_*` — Mamba 2 re-fold for partial accept.
  - Chunked-prefill parallel-scan variant of `ssm_step` (today's
    kernel is decode-only — usable but slow for long prompts).
  - `conv1d_causal_prefill_*` — non-streaming 1D depthwise (today
    only the decode-step variant ships).
  - Generalise `ssm_step` to support `n_groups > 1` (grouped B / C
    tensors).
- **`StateReplayCache` protocol** — parent of `SSMStateCache` +
  `GDNStateCache`. Caches declare `canStateReplay = true` without
  bolting on no-op KV methods. Lands in
  `Sources/FFAI/StateReplayCache.swift`.
- **`GDNStateCache`** — `Sources/FFAI/GDNStateCache.swift`. Per-layer
  GDN recurrent state with `record(...)` + `rollback(acceptedPrefix:)`
  hooks; mirrors the shipped `SSMStateCache` pattern.
- **Family files**:
  - `Sources/FFAI/Models/Qwen35.swift` — `Qwen35Dense`, `Qwen35MoE`,
    `Qwen35GDN` variants. Layer-type alternation between GDN and
    full attention every `fullAttentionInterval` layers. MoE routing
    uses sparse top-K (fused gating kernel reuses the existing
    `dequant_gather` family for expert weights).
  - `Sources/FFAI/Models/NemotronH.swift` — layer-type string
    parsing (`M` Mamba, `*` attention, `E` MoE, `-` MLP); per-layer
    mixer protocol.
  - `Sources/FFAI/Models/Jamba.swift` — Mamba 2 + attention / MoE
    alternation. Handle Jamba's 2D `A_log` shape (kernel-side
    generalisation OR Swift-side reformulation; pick whichever is
    cheaper after the GDN kernels land).
  - `Sources/FFAI/Models/GraniteMoeHybrid.swift` — Mamba 2 + MoE +
    attention.
  - `Sources/FFAI/Models/FalconH1.swift` — Mamba 2 + attention / MLP
    with per-layer multipliers (chunk-size tuning).

Each new family ships with one `Tests/ModelTests/<Family>IntegrationTests.swift`
that downloads from mlx-community and runs a coherent-output assertion;
each new kernel + cache gets a unit test.

### Phase 5f — Attention sinks + sliding window + GPT-OSS-20B

**Kernels:**

- Symbolic sliding-window mask in SDPA decode + prefill (no buffer
  allocation; computed per-step from `(seq_offset, window_size)`).
- `giga_flash_sdpa_v` extension: attention-sinks fold via
  numerically-stable softmax `max(scores)` clamping + dequant.
- Hybrid sliding-FP16 layer policy (GPT-OSS-20B): full-attention
  layers stay on `GigaQuantizedKVCache(useBias: true)`;
  sliding-window layers cap at 128 tokens and stay raw FP16
  (~1.5 MB total).

**FFAI changes:**

- `Sources/FFAI/SlidingWindowMask.swift` — symbolic mask helper.
- `Sources/FFAI/Models/GPTOSS.swift` — family file with the
  alternating layer schedule + sinks parameter + bias-correcting
  K/V projections.
- `LoadOptions.kvCache = .gigaQuantized(scheme:, sinks: true)`
  plumbing.
- `documentation/kv-cache.md`, `documentation/models.md` updated.

**Tests:**

- `Tests/MetalTileSwiftTests/SlidingWindowMaskTests.swift`.
- `Tests/ModelTests/GPTOSSIntegrationTests.swift` — coherent
  `--kv-cache giga4v2` decode at 1k + 8k prompts.

---

## Phase 6 — Dense-text model wave

For each model: family file (consuming existing kernels), config-key
plumbing, registry entry, one integration test, doc row update.

- **Qwen3.5 / 3.6 dense** (0.8B, 2B, 4B, 9B, 27B) — `Qwen35Dense`
  variant. Already partially scaffolded by Phase 5e for the hybrid
  variants.
- **Qwen3.5 / 3.6 MoE** (35B-A3B) — `Qwen35MoE` variant. Sparse
  top-K gating + shared expert + per-expert dequant.
- **Gemma 3** — `Gemma3Dense` variant. Reuses Llama-style backbone.
- **Gemma 4 dense** (E2B, E4B, 31B) — `Gemma4Dense` and `Gemma4E`
  variants. Soft-capped logits (`finalLogitSoftcapping`), per-layer
  embedding (PLE: `hiddenSizePerLayerInput`, `vocabSizePerLayerInput`),
  sliding window every other layer (reuses Phase 5f mask),
  4096-token prefill chunk.
- **Gemma 4 MoE** (26B-A4B) — `Gemma4MoE` variant.
- **Nemotron Cascade 2** — `NemotronCascade2` variant inside
  `NemotronH.swift` (the layer-type string makes the cascade
  scheduling data-driven).
- **Mistral** — single family file `Mistral.swift`; reuses
  Llama-style GQA backbone.
- **Phi** — single family file `Phi.swift`.

Each model gets:

- Family file in `Sources/FFAI/Models/`.
- Registry entry in `ModelRegistry`.
- `Tests/ModelTests/<Family>IntegrationTests.swift` downloading from
  mlx-community + asserting coherent output.
- Doc row updates in `documentation/models.md`,
  `documentation/capabilities.md`, `documentation/quantization.md`.

---

## Phase 6.5 — Vision (VLM)

**Goal:** Stress-test the Capability + lifecycle infrastructure with
real multi-modal models. Validate that disabled-by-default modalities
genuinely don't allocate, that lazy `enable(.visionIn)` works, and
that lifecycle events stream correctly to consumers.

**Kernels (new in metaltile):**

- `conv2d_{kh}_{kw}_{stride}` — fp16 / bf16. Im2col + tiled GEMM
  (no fused depthwise variant for now; add if profiles show
  patch-embed is the bottleneck).
- `patch_embed_*` — combined unfold + linear.
- Vision-specific RoPE 2D positional embedding.
- Cross-modal token splice helper (CPU-fine; image tokens
  interleaved with text tokens).

**FFAI changes:**

- `Sources/FFAI/VisionEncoder.swift` module type (declared in
  Phase 2 for the capability API; lit up here).
- `Sources/FFAI/ImagePreprocessing.swift` — resize / normalize /
  patchify (CPU initially; Metal later if it shows up in profiles).
- Family-level VL variants in their existing family files:
  - `Qwen3.swift` → `Qwen25VL` and `Qwen35VL`.
  - `Qwen35.swift` → `Qwen35VL` + `Qwen35VLMoE`.
  - `Gemma3.swift` → `Gemma3VL`.
  - `Gemma4.swift` → `Gemma4VL` (composes Gemma 4 backbone +
    vision encoder).
- `Capability.visionIn` exercised end-to-end (load with text-only,
  `enable(.visionIn)`, `disable(.visionIn)`).

**Tests:**

- Capability matrix tests (every subset).
- Vision encoder forward correctness vs `mlx-vlm`-captured fixture.
- Multi-modal generation determinism on each VLM.

---

## Phase 7 — Audio (STT + TTS + Omni)

Audio modality is interleaved with vision rather than deferred:
Whisper STT + Qwen-Omni audio + at least one TTS family
(Kokoro or Bark) ship alongside the first VLM wave.

**Kernels (new in metaltile):**

- `mel_spectrogram_*` — log-Mel filterbank (fp32 / fp16 output).
- `audio_conv1d_{kh}_{stride}` — wide-stride 1D conv for STT
  patch embedding.
- Reuse the existing SDPA / RMSNorm / RoPE for transformer stacks.

**FFAI changes:**

- `Sources/FFAI/AudioEncoder.swift` module type.
- `Sources/FFAI/AudioPreprocessing.swift` — Mel computation + framing.
- `Sources/FFAI/Models/Whisper.swift` — Whisper family
  (tiny → large-v3) STT.
- `Sources/FFAI/Models/Kokoro.swift` and/or `Bark.swift` (TTS).
  Pick whichever has the cleaner mlx-audio-swift reference; can
  ship one and queue the other.
- `Sources/FFAI/Models/QwenOmni.swift` — Qwen3.5-Omni
  (text + vision + audio). Uses `Capability.audioIn`, `.audioOut`.
- `Capability.audioIn` + `.audioOut` exercised end-to-end.

**Tests:**

- `Tests/ModelTests/WhisperIntegrationTests.swift` — coherent
  transcription on a known sample.
- `Tests/ModelTests/KokoroIntegrationTests.swift` (or Bark) —
  coherent generated speech (frame-by-frame deterministic with a
  fixed seed).

---

## Phase 8 — Speculative + cache + serving wave (specs 013–043)

Each sub-phase = one or more PRs + integration test + doc update.
Sub-phases land in priority order per
`~/Development/personal/ai/mlx-swift-lm/specs/IMPLEMENTATION-PLAN.md`.

### 8.0 — Foundational infrastructure (specs 018, 023, 020)

- `--method ngram-spot` / `ngram-sweep-summary` bench mode (spec 018).
- Leviathan accept/reject sampling for non-greedy ngram (spec 023).
- `StateReplayCache` protocol formalised (spec 020 — partly shipped
  in 5e; this finalises the protocol surface).

### 8.1 — n-gram speculative decoding (spec 013)

- `NGramSpeculativeTokenIterator` port from
  `Libraries/MLXLMCommon/NgramSpeculativeDecoding.swift`.
- `NGramLookup` multi-size hash, min-hits filter, fallback.
- `ngramRouteDecision()` eligibility predicate + env-var defaults
  (`FFAI_NGRAM_ENABLED`, `ngramSize`, `maxNgramDraftTokens`).
- Auto-disengage on regressive regimes.

### 8.2 — Prefix KV cache (spec 017 — all phases incl. L2 disk)

- `PrefixKVCache` + `PrefixKey` + LRU + stats.
- Per-class `serialise()` / `hydrate(from:)` on every shipped cache:
  `KVCache`, `AffineQuantizedKVCache`, `GigaQuantizedKVCache` (incl.
  compressed-mode), `SSMStateCache`, `GDNStateCache`,
  `Mamba2LayerCache`.
- `LastAssistantOpenerPolicy` for Qwen / Gemma / GPT-OSS chat
  templates.
- L2 disk persistence (opt-in `FFAI_PREFIX_CACHE_DISK=1`) at
  `~/.cache/ffai/prefix/`.
- `generate(...)` wraps a stream that snapshots post-prefill.
- Integration test verifies warm-turn TTFT speedup.

### 8.3 — Compressed-domain prefix KV cache (spec 039)

- Reuses `GigaQuantizedKVCache.fusedEncodeDispatch` for snapshot-time
  batch encode. Bumps `PrefixKey.formatVersion` to 3.

### 8.4 — Batched decoding (`generateBatched`)

- `BatchedKVCache` flat `[B, kv_heads, max_seq, head_dim]`.
- `BatchedHybridCache` for GDN + attention.
- `generateBatched(...)` API + `BatchedGenerateCompletionInfo`.
- Variable-length prompts + per-sequence EOS.
- Continuous batching.

### 8.5 — Cross-request n-gram cache (spec 016)

- Three-tier (`nc_context` / `nc_dynamic` / `nc_static`) per
  llama.cpp.
- Registry + tiered cache.

### 8.6 — Deterministic-stretch acceleration (spec 022)

- `ChatTemplateGrammar` protocol + `BigramTable` + per-family
  grammars.
- Highest win on GPT-OSS harmony channel transitions.

### 8.7 — Native MTP / EAGLE-3 draft heads (spec 030)

- Variant A: stop stripping `mtp.*` in sanitize; ship
  `MTPSelfSpeculativeTokenIterator` + `scripts/mtp_convert.py`.
- Variant B: companion EAGLE-style assistant draft +
  `AssistantDraftRegistry`.

### 8.8 — Tree attention (spec 014, phase 1: K=2 root branches)

### 8.9 — PLD+ attention-weighted span selection (spec 019)

### 8.10 — DuoAttention retrieval / streaming head split (spec 036) + block-sparse SDPA (spec 033)

- Calibration pass on synthetic NIAH.
- Two-cache-per-layer dispatch.
- Block-sparse SDPA Metal kernel (spec 033) consumed here.

### 8.11 — Decode-side K-side top-k / Quest (spec 034) + spec 035 K_max/K_min refinement

### 8.12 — TEAL activation thresholding (spec 037)

- `threshold_and_mask` hook + `scripts/teal_calibrate.py`.
- Block-sparse Metal kernel for `(masked_act, down_proj) → out`.

### 8.13 — Sparse prefill (spec 031 vertical-slash + spec 032 speculative prefill)

### 8.14 — DFlash on GPU (spec 015, phases 1–3)

- Phase 2: draft model from `z-lab/Qwen3.5-*-DFlash`.
- Phase 3: refactor onto `StateReplayCache` protocol.

### 8.15 — Mirror SD (spec 021) + ANE concurrency primitives (spec 025)

- ANE + GPU concurrency primitives (spec 025) land first.
- Then `MirrorSpeculativeLoop` glue.
- Decision point: ANE + GPU truly concurrent on Apple Silicon?
  Failure here kills 021 + spec 029. Document and pivot.

### 8.16 — KV cache write fusion (spec 024)

- Eliminate the `copy_bfloat16` dispatches per decode token.

### 8.17 — Profile-guided Morton-order expert reorder (spec 026)

### 8.18 — Adaptive per-layer mixed-precision (spec 027)

- Recipe-driven framework via JSON sidecar + glob-pattern matching.

### 8.19 — Quadratic / chunkwise WY GDN prefill (spec 028)

- Highest research bet. Could regress if it doesn't work.

### 8.20 — ANE-offloaded LM head + Gemma 4 PLE projection (spec 029)

- Blocked on spec 025 + Mirror SD measurement.

### 8.21 — Active KV cache SSD offload (spec 038)

- Long-context memory reduction. Multi-month. Only justified if
  long-context single-request use cases matter.

### 8.22 — Flash-quantized SDPA + Metal kernel SIMD audit (specs 041, 042)

- Spec 041: drop-in Flash-tiled fused kernel for the affine
  quantized SDPA path.
- Spec 042: cross-kernel SIMD audit — convert GigaFlash + affine
  flash + `giga_dequant_rotated` + `mse_*` to
  `simdgroup_matrix_multiply_accumulate` MMAs.

### 8.23 — GigaFlash decode-time kernel uplift (spec 043)

- Renamed from TurboFlash. Per-simdgroup bit-unpack reuse + bf16
  V accumulator + headDim-aware tile autotune + bias-aware kernel.

---

## Phase 9 — Performance / dispatch modes / autotuner

- **Argument-buffer dispatch mode** (Mode 2 in
  `architecture.md §4a`).
- **ICB dispatch mode** (Mode 3).
- **Metaltile autotuner** — grid search over
  `(tile_dims, threads, unroll, simd_matrix, async_copy)`.
  Persist to `~/.cache/metaltile/tuning_cache.json`. CI: nightly
  autotune on a reference machine; commit results.

**Done when:** generated kernels are within ≤2% of hand-tuned
variants for representative shapes per kernel, and argument-buffer +
ICB modes are selectable via `LoadOptions.dispatchMode` behind the
same Model API.

---

## Phase 10 — Polish

- **gguf format support** — per-architecture name mapper. Single-file
  format from llama.cpp, embeds quantization (Q4_K_M, Q5_K_M, Q8_0,
  etc.) and tokenizer. Worth doing if community gguf quants are
  valuable to users; skip if all target checkpoints are mlx-format
  or safetensors.
- **Distribution** — Homebrew formula for the `tile` binary, SPM
  consumer instructions.
- **Benchmarks** — full sweep vs MLX baseline across the model zoo.
- **Documentation site polish.**

---

## Out of scope / deferred

- **CoreML / ANE backend.** Realistic only for boring kernels (RMSNorm,
  RoPE, layer norm, plain GEMV at fp16/int8). GigaQuant, FWHT, online
  softmax, recurrent SSM/GDN do not fit ANE constraints. Add a `mil/`
  codegen sibling to `msl/` in metaltile-codegen when v0.3 demand justifies it.
- **Swift macro frontend** for kernel authoring. metaltile IR is
  serde-serializable; a Swift `@kernel` macro emitting IR JSON could feed
  the same backend later. Don't build it preemptively — wait for demand.
- **Training / autograd.** Different project.
- **CUDA / Linux backends.** Different project.

---

## What we are explicitly betting on

1. **Eager dispatch with kernel-level fusion is enough.** MLX's lazy
   graph + automatic op fusion gets replaced by hand-written fused
   kernels in metaltile. For inference's known hot paths this is fine
   and gives predictable timing. If we discover Phase 2 perf regressions
   trace to lost lazy-eval pipelining, we re-evaluate.
2. **MTLCommandBuffer batching at the layer level recovers what little
   pipelining we lose.** One `MTLCommandBuffer` per token (or per N
   layers), not per kernel.
3. **Pre-compiled metallib eliminates first-call latency that JIT
   imposed.** No 1-2ms warmup per never-before-seen kernel.
4. **Direct MTLBuffer ownership eliminates copies that MLX hid behind
   `MLXArray`.** Memory residency is ours to manage.
5. **Rust kernel authoring is fine for a small expert audience.** Most
   contributors will work in Swift on layers/models, not kernels.

If any of these bets break, the plan changes. Stage commits so we can
roll back to the last working phase boundary.

---

## Open questions to settle before Phase 0 implementation

1. SPM build plugin can shell out to arbitrary commands (`cargo`)? Or
   do we need a pre-build Makefile step?
2. Where does metaltile live for FFAI to consume — sibling path
   dependency (current), git submodule, or eventually a published Rust
   crate?
3. metallib per-arch (arm64 only, or universal)? Per-Metal-version
   (separate metallib for Metal 3.0 vs 3.1+ for native bf16)?
4. Manifest.json schema versioning strategy — how do we evolve it
   without breaking the generated Swift?
