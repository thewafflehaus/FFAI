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

1. **metaltile** (Rust) — `#[kernel]` DSL, IR, MSL codegen.
   Ships a `tile` CLI (`metaltile-cli`) whose `emit`
   subcommand produces `kernels.metallib`, `manifest.json`, and
   `MetalTileKernels.swift`. Historical note: Phase 0 originally
   wired a standalone `metaltile-emit` bin for this; that pipeline
   was rehomed into `metaltile-codegen::emit` + the `tile build --emit all`
   subcommand when upstream consolidated the build/bench/test
   commands under one CLI.
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

**Test coverage: ≥ 80 % line coverage on FFAI + MetalTileSwift Swift
code (was 100 % aspirational; the realised gate is the 80 % ratchet in
ci.yml).** Measured via `swift test --enable-code-coverage` driven
through `make coverage` / `scripts/coverage.sh` (matches ci.yml). CI
fails any PR that drops coverage below the ratchet. Every phase adds
tests alongside code, not after.

**What the quality bar means here:**

- Every public function: at least one happy-path test
- Every branch in business logic: at least one test per side
- Every kernel in metaltile: a paired GPU correctness test under
  `crates/metaltile-std/tests/<kernel>_gpu_correctness.rs` comparing
  against a naive CPU reference (pattern from
  [TheTom PR #35](https://github.com/0xClandestine/metaltile/pull/35)),
  **plus** an MLX side-by-side check when the kernel has an upstream
  counterpart (kernels under `metaltile-std/src/mlx/`, via `tile bench`
  + `check_equiv`), **plus** an FFAI integration test that exercises
  the kernel on a real model (kernels under `metaltile-std/src/ffai/`
  where there's no MLX upstream counterpart). The `metaltile-interp`
  CPU interpreter crate is gone (dropped in PRs #16 / #17); GPU
  correctness + MLX side-by-side are the two correctness paths now.
- Every codegen emit path: `insta` MSL golden snapshot under
  `crates/metaltile-codegen/tests/msl_snapshots.rs` (pattern from
  [PR #25](https://github.com/0xClandestine/metaltile/pull/25)).
- Every model: token-by-token determinism test against a reference
  for at least one prompt + seed combination

**What it doesn't mean:**

- Mocking out the GPU. Tests run real Metal dispatches on the CI
  runner (Apple Silicon).
- Property/fuzz testing — out of scope for v0.1; revisit later.
- Defensive-error paths that can't actually be triggered (Swift
  `fatalError` on programmer bugs etc.) are excluded from the
  coverage denominator via `// coverage:ignore` markers.

---

## Performance & testing gaps (current state, 2026-05-23)

After Phase 6.5 (Vision) and Phase 7 (Audio) landed, this is the
honest list of where the codebase is soft. Tracked in detail in
`planning/session-plan.md` under "Performance gaps" + "Testing gaps".

### Performance
1. **One-token-per-dispatch prefill** — `Generate.swift` ignores
   `prefillStepSize`. TTFT scales linearly with prompt length.
   Closed by Phase 6.6.
2. **Sliding-window SDPA full-attention fall-through** — `Ops.sdpaDecode`
   passes `sink_end = 0`, `window_start = 0` even when the cache is
   sliding-window. 4×–8× decode tax at 16K–32K context.
   Closed by Phase 6.1.
3. **VLM cold inference on Idefics3 / PaliGemma / GlmOcr / FastVLM**
   — vision-tower attention + depthwise conv run on parallelised CPU,
   not GPU. Minutes-per-first-image at 1024px on FastVLM.
   Closed by Phase 6.5b.
4. **AURA persistent K/V mirror** — Stage 1a snapshots a full
   `[maxSeq, kvHeads, headDim]` mirror per layer. Closed by Phase 6.3
   Stage 1b (compressed-domain `aura_flash` as default decode path).
5. **GPU 100% pin (unknown root cause)** — deferred per user; needs
   a Metal System Trace to localise. Mitigation: `FFAI_MAX_COMMAND_BUFFERS=16`
   caps the queue so it stays annoying-but-survivable.
6. **Per-token `commit + wait` + 4-byte readback**  [#93] — partially
   shipped under Phase C #6 (1-cmdbuf default path); full GPU-resident
   decode is a Phase 9 dispatch-mode win.
7. **FishSpeech Conv1d on CPU** — codec ports run dilated /
   transposed Conv1d in Swift until the metaltile kernels land.
   Synthesise path produces a waveform but slowly.
8. **Marvis Mimi codec — not wired** — `MarvisModel` frame-generates
   but `mimiDecoder` is nil. Hook in `Audio/Mimi.swift`.

### Testing
1. **Integration tests written but unrun** — every family added in
   the Phase 6.5 (16 VLMs) + Phase 7 (35+ audio families) wave
   ships an assertive integration test that has not yet been run
   against a cached checkpoint. The first
   `make test-integration --filter <Family>` pass against each gives
   the real coherence verdict.
2. **No GPU correctness tests** for the CPU vision towers (Phase 6.5b
   prerequisite). Each new metaltile kernel under that phase needs
   its paired `*_gpu_correctness.rs`.
3. **No per-layer forward tests** for most new families — they ship
   config-parse + registry-detection unit tests but no
   `<Family>ForwardTests.swift` running one decoder layer against a
   known input. These catch regressions earlier than the
   integration suite.
4. **No AURA MSL snapshots** — Phase 6.2 still open.
5. **FishSpeech integration test asserts the staged path** — flip
   the assertion once the Conv1d codec primitives land.
6. **VLMTestSupport has one fixture** — `dog.jpeg`. A text-rendering
   fixture (for GLM-OCR) and an alpha-channel fixture (for SmolVLM)
   would harden the preprocessing pipeline.
7. **`steel_gemm_splitk` flakiness**  [#92] — intermittent under the
   full suite; needs a deterministic repro.

**Test layout:**

```
metaltile/
  crates/metaltile-codegen/tests/msl_snapshots.rs   # insta MSL goldens
  crates/metaltile-codegen/tests/snapshots/         # .snap files
  crates/metaltile-std/tests/<kernel>_gpu_correctness.rs  # one per non-trivial kernel
  crates/metaltile/tests/error/*.rs                 # trybuild compile-fail fixtures

FFAI/
  Tests/MetalTileSwiftTests/   # PSO manifest + per-kernel Swift wrapper smoke
  Tests/FFAITests/             # Tensor, Module, Linear, KVCache, Sampling, … (parallel-safe)
  Tests/ModelIntegrationTests/ # one folder per model — forward-pass determinism
                               # vs mlx-lm / mlx-vlm golden fixture (serialized)
```

Every PR that adds production code without corresponding tests is
rejected at review. CI publishes coverage diff per PR. See
[`CLAUDE.md`](../CLAUDE.md#tooling-cheat-sheet--local-dev-loop) for the
dev-loop cheat-sheet.

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

## Phase 0 — Plumbing (this repo + metaltile) ✅ SHIPPED

**Goal:** Round-trip a single trivial kernel from Rust `#[kernel]` to
Swift dispatch, with the SPM build plugin auto-invoking the emit step.

**Deliverables:**

- `tile build --emit all` subcommand in the metaltile CLI
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
  - `make regenerate-kernels` runs `tile build --emit all` against the local
    metaltile checkout before each build
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
  - `.github/workflows/ci.yml` — Apple Silicon runner. Runs
    `swift test --enable-code-coverage --filter "FFAITests|MetalTileSwiftTests"`
    (unit suite only; mirrors `make test-unit`). Uploads coverage
    report, fails on coverage drop. Integration tests are deliberately
    excluded from PR CI — they download multi-GB HF snapshots and
    OOM the 7 GB runner; release.yml runs them right before tagging.
  - `.github/workflows/release.yml` — Apple Silicon runner. Runs both
    `swift test … --filter "FFAITests|MetalTileSwiftTests"` and
    `swift test … --filter "ModelIntegrationTests" --parallel --num-workers 1`
    (matches `make test-integration`) right before cutting a release.
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
  - `Makefile` — common targets: `build`, `build-release`, `test`,
    `test-unit`, `test-integration`, `clean`, `regenerate-kernels`,
    `coverage`, `format`, `format-check`, `docs`. The `test-unit` /
    `test-integration` split mirrors `.github/workflows/{ci,release}.yml`
    — integration tests are gated to `--parallel --num-workers 1` so
    only one HuggingFace snapshot is GPU-resident at a time.
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

## Phase 1 — Foundation kernels ✅ SHIPPED

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

- All above registered in the metaltile kernel registry that `tile
  emit` walks
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

## Phase 2 — First model end-to-end (Llama 3.2 1B) ✅ SHIPPED

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
  - `Tests/ModelIntegrationTests/Llama/LlamaForwardTests.swift` —
    randomly-initialized layer numerical match vs golden fixture
    (captured from mlx-lm)
  - `Tests/ModelIntegrationTests/Llama/LlamaGenerateTests.swift` — token-by-token
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
- AURA / state-replay (Phase 5d/5e — TurboQuant in mlx-swift-lm
  was renamed to AURA in FFAI)
- Quantization (Phase 3)

**Done when:** `ffai --model llama-3.2-1B --prompt "Hello"` produces
coherent text on M-series, token-by-token output matches a reference
implementation for a fixed seed, and tokens/sec is measured and
recorded as baseline.

---

## Phase 2.5 — Second model (Qwen3 4B) ✅ SHIPPED

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
  - `Tests/ModelIntegrationTests/Qwen3/Qwen3ForwardTests.swift` — q_norm/k_norm
    application correctness
  - `Tests/ModelIntegrationTests/Qwen3/Qwen3GenerateTests.swift` — token-by-token
    determinism
  - 100% line coverage maintained

**Done when:** `ffai --model qwen3-4B --prompt "…"` generates
coherent text and matches a reference implementation token-by-token
for a fixed seed.

**Philosophy from this point on:** core kernels and Tensor/Module
infrastructure are stable. Adding a new model = porting its
forward-pass shape from mlx-swift-lm and wiring it to existing
kernels. New *model-specific* kernels (e.g. attention sinks for
GPT-OSS, fused MoE expert kernels, GDN steps, AURA codecs) get
added to the metaltile DSL as needed — driven by which model we want
to support next, not speculatively.

---

## Phase 3 — Quantization ✅ SHIPPED

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

## Phase 4 — Performance optimizations ✅ SHIPPED

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

**Deferred out of Phase 4:**

- Per-shape kernel parameter selection → Phase 9 autotuner.
- Argument-buffer / ICB dispatch modes → Phase 9.
- Multi-token batched (chunked) prefill → Phase 6.6.

---

## Phase 5 — Advanced kernels (sampling, AURA, GDN, SSM)

**Goal:** Port the high-value custom kernels currently in mlx-swift-lm
plus close the user-visible sampling gap. The custom kernels were
the original motivator for this project — the 4-repo dance to ship
a new AURA variant (TurboQuant in mlx-swift-lm, renamed to
AURA in FFAI) is the pain we're eliminating.

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

### Phase 5d — AURA compressed-domain attention ✅ Part 1 (correctness) SHIPPED

TurboQuant codec ported from mlx-swift-lm, **renamed AURA** in FFAI
(kernels, env vars, CLI flags, docs all use `aura*`). Schemes are
`aura{kb}v{vb}` (K-side / V-side bit widths); symmetric aliases
`aura3` / `aura4` / `aura6` / `aura8`, asymmetric `aura8v4`,
`aura4v2`, `aura3v2`. CLI: `--kv-cache aura4v2`. Codec lineage +
design rationale: `papers/aura-compression-algorithm.md`.

**5d.A — metaltile DSL prerequisites ✅** — `simd_shuffle_xor` /
`simd_broadcast`, atomic ops on threadgroup memory, the persistent
state-buffer convention. All landed in `metaltile`.

**5d.B — AURA kernels ✅** — `aura_encode`, `aura_dequant_rotated`,
`aura_score`, `aura_value`, `aura_flash_p1`, `aura_flash_pass2`
under `crates/metaltile-std/src/ffai/`, bits ∈ {2,3,4,6,8}.

**5d.C — FFAI integration ✅** — `AURAQuantizedKVCache`
(`KVCacheProtocol`, per-layer compressed K/V, per-layer eviction),
`LoadOptions.kvCache = .auraQuantized(scheme:)`, the `--kv-cache`
flag + aliases, `Llama` / `Qwen3` `makeLayerCaches` wiring.

**5d.D — per-layer SRHT rotation (Stage 1a) ✅** —
`Ops.auraRotatePerHead`, a per-layer SRHT rotation Π, Q rotated
post-RoPE + the attention output un-rotated in the model forward.
The "coherent-then-collapse around token 50" bug was a
dequant-kernel stride mismatch — `aura_dequant_rotated` keyed
per-head offsets off its `tokens` constexpr instead of the buffer's
`maxSeq` stride; fixed by passing an explicit `cacheStride`. All
four recipes (`aura4v4` / `aura4v2` / `aura8v4` / `aura8v8`) now
produce coherent text on Qwen3-1.7B; `aura8v8` is near-lossless vs
raw bf16.

**5d.E — kernel test coverage ✅** — per-kernel GPU correctness
tests (naive-CPU oracle) for every AURA kernel, including the
non-identity SRHT rotation path; FFAI-side codec round-trip tests.

**Audit finding — do not re-litigate.** FFAI's `AURACodebook` is
byte-identical to the working mlx-swift-lm reference, and that
reference produces coherent Qwen3 output with `useBias: false`. The
earlier "codebook recalibration / DC-bias correction needed"
hypothesis was **refuted** — DC-bias is a GPT-OSS-only feature
(`RMSNorm → Linear(bias=True)` projections), not a Qwen3
requirement.

**AURA performance — deferred to Phase 6.3.** FFAI currently runs
AURA through a dequant-then-`sdpaDecode` path with a persistent
working buffer. Compressed-domain attention (`aura_flash_p1/p2`),
two separate K/V codecs, two-phase prefill, the W_o offline fold,
strided-output encode, and the cache-layout flip are all perf /
architecture work — **correctness does not depend on any of them**.
Picked up after the rest of Phase 5 + Phase 6 — full scope in
Phase 6.3 below.

### Phase 5e — SSM / GDN hybrid models ✅ SHIPPED

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
- Integration test: `Tests/ModelIntegrationTests/Mamba2IntegrationTests.swift`
  loads `mlx-community/mamba2-130m`, verifies shapes match config,
  runs greedy decode to completion (~130 tok/s on M-series).

**Shipped — Phase 5e complete.** The forward / decode path for all
five hybrid families landed, each with a coherent-output integration
test:

- **GDN forward kernel** — `mt_gated_delta_step` (recurrence
  `S_t = g_t·S_{t-1} + β_t·k_t·(v_t − k_tᵀ·S_{t-1})ᵀ`, fp32 state)
  + `Ops.gatedDeltaStep` + `GDNStateCache` (forward-only, mirrors
  `SSMStateCache`).
- **MoE inference infrastructure** — `MoERouter` + `MoELayer`
  (top-K router + per-expert SwiGLU dispatch).
- **Per-layer mixer scaffolding** — the `DecoderLayer` protocol +
  `StatelessLayerCache`, driving a heterogeneous `[any DecoderLayer]`
  decode loop.
- **Family files** — `Models/{FalconH1,NemotronH,GraniteMoeHybrid,
  Jamba,Qwen35}.swift`, with `Jamba`'s 2D `A_log` selective scan
  handled host-side (a GPU 2D-`A` `ssm_step` variant is a tracked
  perf follow-up — see metaltile `docs/KERNEL_AUDIT.md`).

**Deferred out of 5e:**

- **→ Phase 8 (speculative decoding):** the partial-accept rollback
  infra — `gated_delta_step_record`, `state_replay`,
  `ssm_step_record`, `ssm_replay` kernels; the `StateReplayCache`
  protocol; `GDNStateCache.record()` / `.rollback(acceptedPrefix:)`.
- **→ a perf pass:** chunked-prefill parallel-scan `ssm_step`;
  `conv1d_causal_prefill` (the shipped decode-step variants cover
  prefill, just slower).
- **Conditional:** generalise `ssm_step` to `n_groups > 1` (grouped
  B / C) only if a target checkpoint's `config.json` needs it.

### Phase 5f — Attention sinks + sliding window + GPT-OSS-20B ✅ SHIPPED

**Kernels:**

- Symbolic sliding-window mask in SDPA decode + prefill (no buffer
  allocation; computed per-step from `(seq_offset, window_size)`).
- `aura_flash_sdpa_v` extension: attention-sinks fold via
  numerically-stable softmax `max(scores)` clamping + dequant.
- Hybrid sliding-FP16 layer policy (GPT-OSS-20B): full-attention
  layers stay on `AURAQuantizedKVCache(useBias: true)`;
  sliding-window layers cap at 128 tokens and stay raw FP16
  (~1.5 MB total).

**FFAI changes:**

- `Sources/FFAI/SlidingWindowMask.swift` — symbolic mask helper.
- `Sources/FFAI/Models/GPTOSS.swift` — family file with the
  alternating layer schedule + sinks parameter + bias-correcting
  K/V projections.
- `LoadOptions.kvCache = .auraQuantized(scheme:, sinks: true)`
  plumbing.
- `documentation/kv-cache.md`, `documentation/models.md` updated.

**Tests:**

- `Tests/MetalTileSwiftTests/SlidingWindowMaskTests.swift`.
- `Tests/ModelIntegrationTests/GPTOSSIntegrationTests.swift` — coherent
  `--kv-cache aura4v2` decode at 1k + 8k prompts.

---

## Phase 6 — Dense-text model wave ✅ SHIPPED

For each model: family file (consuming existing kernels), config-key
plumbing, registry entry, one integration test, doc row update.

Shipped: Mistral, Phi, Gemma 3, Gemma 4 (`Gemma4Dense` / `Gemma4E` /
`Gemma4MoE` — incl. the 26B-A4B MoE), Qwen 3.5 dense + MoE (via
`Qwen35.swift`), and `NemotronLabsDiffusion` (NVIDIA Nemotron-Labs-
Diffusion tri-mode text backbone). The `ModelKVCacheMatrixIntegrationTests`
cross-product (model family × weight-bitwidth × KV-cache scheme) also
landed in this wave.

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
- `Tests/ModelIntegrationTests/<Family>IntegrationTests.swift` downloading from
  mlx-community + asserting coherent output.
- Doc row updates in `documentation/models.md`,
  `documentation/capabilities.md`, `documentation/quantization.md`.

---

## Phase 6.1 — Sliding-window SDPA fast path  ❌ ARCHITECTURALLY MOOT (2026-05-23)

The original plan was to thread `sink_end` + `window_start` (metaltile
PR #50's `ffai_sdpa_decode` head_dim=128 constexprs) through
`Ops.sdpaDecode` so the kernel could skip the stale K/V slot range
`[sink_end, window_start)` at the loop-bound level.

**Audit finding:** the kernel fast path only delivers a speedup when
the cache layout has *stale slots* — i.e. linear-grow-with-skip,
where positions beyond the window stay in the buffer but should be
masked out of attention. FFAI's sliding-window cache uses
**rotating-eviction** instead (`KVEviction.window(maxSize:keep:)` in
`KVCacheEviction.swift`): when the buffer fills, the oldest non-sink
slot is overwritten by the next append. Every slot in `[0, length)`
is a live K/V vector we want to attend. The kernel's skip range is
always empty for our cache; threading `sinkEnd`/`windowStart` would
be a no-op.

Compounding factor: the two sliding-window consumers in tree don't
qualify anyway — **GPT-OSS uses `head_dim = 64`** and **Gemma 3 / 4
use `head_dim = 256`**; the kernel fast path only exists on the
head_dim=128 variant.

Real wins still available in this space (re-scoping Phase 6.1):

1. **Port `has_sink + sink_logit` to head_dim=64 in metaltile.** GPT-OSS
   today commits the cmdbuf mid-attention, reads K + Q back to CPU,
   computes a per-head sink-correction factor, scales the SDPA output
   on the host. The kernel feature exists on head_dim=128; porting it
   to head_dim=64 eliminates one per-token CPU readback per attention
   layer.
2. **Switch sliding-window cache to linear-grow-with-skip.** Enables
   the existing head_dim=128 kernel fast path on Llama / Mistral /
   Phi / Qwen. Cost: cache memory grows from `min(maxSize, length)` to
   `length` (up to `maxSeq`). Probably not worth it for our memory
   constraints; revisit only if the head_dim=128 sliding-window
   layers become a measurable bottleneck.

Decision: drop the "Phase 6.1" milestone from the active plan, leave
the rescoped wins (#1, #2) on the backlog. The architecturally-moot
nature of the original phase is documented here so a future reader
doesn't re-pick it up under the old framing.

---

## Phase 6.2 — AURA MSL snapshot tests

Add `insta` MSL snapshot fixtures for the AURA kernels under
`crates/metaltile-codegen/tests/msl_snapshots.rs` (the metaltile
PR #25 pattern). Pins the emitted MSL so any codegen change to an
AURA emit path surfaces as a reviewable text diff.

---

## Phase 6.3 — AURA performance

Deferred from Phase 5d. **Correctness does not depend on any of
this** — it is the architecture + perf pass. (The AURA index-50
coherence collapse was a separate correctness bug — a stride
mismatch in `aura_dequant_rotated` — and is already fixed; every
AURA scheme decodes coherently in `ModelKVCacheMatrixIntegrationTests`.)

**Stage 1b — compressed-domain attention.**

- Two separate K/V codecs — independent SRHT seeds per layer
  (e.g. `2·layerIdx` for K, `2·layerIdx+1` for V); Q rotated with
  the K rotation (score cancellation), output un-rotated with the V
  rotation. Decorrelated quantization noise; matches the
  mlx-swift-lm reference.
- Two-phase prefill — raw fp16 buffer during prefill, batch-compress
  at the prefill→decode boundary, per-token encode after.
- Compressed-domain attention via `aura_flash_p1` / `aura_flash_pass2`
  as the **default** decode path — drop the persistent
  `sharedWorkingK/V` mirror buffers.
- Opt-in B-path — short-lived per-layer dequant buffer + `sdpaDecode`
  for callers with the memory headroom who want matrix-engine SDPA.
  The dequant buffer must be tight-scoped (one layer's worth resident
  at a time, not a persistent `maxSeq`-sized mirror).
- W_o offline fold — replaces Stage 1a's runtime output un-rotation.
- Norm correction — keep FFAI's always-applied `‖x‖/‖recon‖`;
  revisit only via an A/B test (PPL/KLD + speed) vs the reference's
  raw-norm WHT path.

**Stage 3 — encode + layout perf.**

- Strided-output `aura_encode` — one dispatch writes all heads
  (today: one dispatch per head).
- Cache-layout flip to `[maxSeq, nKVHeads, packedWidth]` — makes the
  decode-time append a single contiguous write.

---

## Phase 6.4 — Profile injectable

Make `Profile` injectable instead of a `.shared` singleton — each
`Model.generate(...)` takes `profile: Profile = .shared`.
Prerequisite for per-sequence telemetry under the batched /
continuous decode Phase 8 introduces (see the concurrency audit
`planning/concurrency-and-cache-readiness-audit-2026-05-19.md` §2.D).

---

## Phase 6.5 — Vision (VLM) ✅ SHIPPED

**16 VL families landed**, well past the original plan scope.
Shipped: Qwen 2-VL (`Qwen2VL.swift`), Qwen 2.5-VL (`Qwen25VL.swift`),
Qwen 3-VL (`Qwen3VL.swift`), Qwen 3-VL-MoE (`Qwen3VLMoe.swift`),
Gemma 3-VL (`Gemma3VL.swift`), Gemma 4-VL (`Gemma4VL.swift`),
LFM2-VL (`LFM2VL.swift`), MiniCPM-V 4.6 (`MiniCPMV.swift`),
NemotronVL (`NemotronVL.swift`), SmolVLM2 (`SmolVLM2.swift`),
Pixtral (`Pixtral.swift` — Mistral 2D-RoPE ViT), Mistral 3
(`Mistral3.swift`), FastVLM (`FastVLM.swift` — Apple FastViTHD),
GlmOcr (`GlmOcr.swift`), Idefics3 (`Idefics3.swift`),
Paligemma (`Paligemma.swift`).

Phase 6.5 deliverables ✅ on disk:
- `Sources/FFAI/VisionEncoder.swift` (with parallelised CPU
  bidirectional `cpuAttention` — fix for the original VLM "image
  hang").
- `Sources/FFAI/ImagePreprocessing.swift` (resize / normalize /
  patchify + CHW conversion).
- `Tests/ModelIntegrationTests/VLMTestSupport.swift` with
  `dogImageCHW(targetSize:)` /
  `dogImageCHWNormalized(targetSize:normalization:)` /
  `expectMentionsDog(...)`.
- Metaltile kernels: `conv2d`, `patch_embed`, `audio_conv1d`,
  `rope_2d`.
- Every VL family ships with a `<Family>IntegrationTests.swift`
  asserting "dog" in the caption of `dog.jpeg`. **The tests are
  written but most have not yet been run against a cached
  checkpoint** — see session-plan testing gaps.

**Phase 6.5b deferred follow-up (perf):** Idefics3, PaliGemma,
GlmOcr, and FastVLM still run their vision-tower attention +
depthwise conv on CPU (parallelised but not GPU). FastVLM cold
inference at 1024px is the loudest signal. Slot a GPU
vision-attention kernel + depthwise conv2d port under 6.5b before
Phase 8 starts.

Original 6.5 goal text preserved for context below.

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

## Phase 6.5b — VLM vision-tower GPU port

**Why this exists.** Phase 6.5 shipped 16 VL families, but four of
them — Idefics3, PaliGemma, GlmOcr, FastVLM — still run their
vision-tower attention + depthwise conv on CPU (parallelised via
`DispatchQueue.concurrentPerform`, but not GPU). FastVLM cold
inference at 1024px is flagged at minutes per first-image in the
agent return notes; the others have similar tails.

The Phase 6.5 vision encoder reused FFAI's `VisionEncoder` for any VL
family whose tower is a standard ViT + LayerNorm + GELU stack. The
four CPU-bound stragglers each ship a custom tower:
- **Idefics3** — bidirectional pre-norm SigLIP-ish with pixel-shuffle
  connector; CPU bidirectional attention in `Idefics3VisionEncoder`.
- **PaliGemma** — SigLIP variant; CPU bidirectional in
  `PaligemmaModel`'s embedded vision branch.
- **GlmOcr** — dynamic-resolution ViT; CPU bidirectional in
  `GlmOcrVisionBlock`.
- **FastVLM** — Apple FastViTHD; depthwise + pointwise conv chain
  with CPU depthwise (pointwise = `Ops.gemm`, GPU); CPU SE gates +
  CPU bidirectional MHSA.

**Deliverables:**
- New metaltile kernel(s):
  - `conv2d_depthwise_{kh}_{kw}_{stride}` (fp16 / bf16) — depthwise
    pass via direct sliding-window MAC; no im2col blow-up.
  - Vision bidirectional attention (no causal mask, no KV cache) —
    likely a thin variant of `sdpa_decode` or `sdpa_decode_batched_prefill`
    with the mask disabled, exposed as `Ops.sdpaBidirectional(...)`.
- FFAI:
  - `Ops.conv2dDepthwise(...)` wrapper.
  - Migrate each custom tower's CPU attention to `Ops.sdpaBidirectional`.
  - Migrate FastVLM ConvFFN depthwise to `Ops.conv2dDepthwise`.
- Tests: each kernel ships `*_gpu_correctness.rs` in metaltile;
  per-family integration test confirms speedup + identical caption
  vs CPU oracle.

**Done-gate:** FastVLM cold inference at 1024px completes inside
30 s on a representative M-series device. The other three see their
vision-encode wall-time drop by ≥ 4× vs the parallelised-CPU baseline.

---

## Phase 6.6 — Chunked (batched) prefill  🟡 SCAFFOLD SHIPPED (2026-05-23)

**Scaffold shipped** (commit `017e954`):
- `LanguageModel.forwardMulti(tokenIds:startingAt:caches:on:device:) -> Tensor`
  protocol method (default loop on a single command buffer, returns
  tail-position logits).
- `Generate.driveGeneration` prefill restructured to call
  `engine.forwardMulti(chunk, startingAt: pos, on: cmd)` per
  `prefillStepSize`-sized chunk, then `sampleNext` only on the final
  position. **Commit count drops from N to ceil(N/chunkSize) + 1** —
  for a 1024-token prompt with the default `prefillStepSize=1024`
  that's 1024 → 2 commits, eliminating ~milliseconds of CPU↔GPU
  sync per discarded commit.

**Next session — family-optimised `forwardMulti` overrides.** The
default loop calls `forward(tokenId:)` N times on the same cmdbuf;
the real Phase 6.6 perf win is overriding `forwardMulti` per family
to batch the QKV / MLP projections via `Ops.gemm(weight:, input:
[N, hidden], nRows: N)` and collapse the N per-token SDPA dispatches
into one `Ops.sdpaMulti(causal: true)` call. All primitives exist
(`Ops.gemm` ships batched matmul; `Ops.rmsNormRows` ships batched
RMSNorm; `Ops.sdpaMulti` ships causal multi-query SDPA). The work is:

1. **`LlamaLayer.forwardMulti(chunk:positions:cache:cmd:device:)`** —
   restructure the per-token layer forward to take `[N, hidden]`
   inputs and produce `[N, hidden]` outputs. The hot wins are
   `Ops.gemm(...)` for {q,k,v,o,gate,up,down}_proj, `Ops.sdpaMulti`
   for attention, `Ops.rmsNormRows` for the two layer norms.
2. **`KVCacheProtocol.appendChunkOnGPU(kChunk:vChunk:positions:on:)`** —
   loop the existing single-position `kv_cache_update` N times on
   the same cmdbuf (cheap; the K/V append isn't the bottleneck) OR
   add a batched kernel to metaltile if the loop overhead shows up
   in profiles.
3. **`LlamaModel.forwardMulti(...)`** — embed N tokens, run the layer
   stack via `LlamaLayer.forwardMulti`, final norm batched, lm_head
   only on the tail position.
4. **Test** — chunked-vs-per-token logit equality (within numerical
   tolerance) on a tiny Llama. Smoke at production size: a TTFT
   regression check on Llama 3.2 1B with 2048-token prompt.

The same shape applies to Qwen 3, Mistral, Phi, Gemma 3 / 4 dense.
Hybrid families (NemotronH, Jamba, GraniteMoeHybrid, FalconH1, LFM2)
can chunk the attention layers but keep SSM / GDN steps per-token —
their recurrence is inherently sequential.

Original phase text preserved below.

Today FFAI prefills a prompt one token per dispatch (`Generate.swift`
— `prefillStepSize` is a no-op placeholder). Batch the prompt into
chunks so prefill processes N tokens per forward — a large TTFT win
on long prompts.

- Drive a multi-token forward — `forwardMulti(tokenIds:positions:
  caches:)` — over a chunk of prompt tokens.
- Chunk attention uses the existing
  `ffai/sdpa_decode_batched_prefill` metaltile kernel; the KV-cache
  append writes the whole chunk's K/V in one shot.
- Honor `GenerationParameters.prefillStepSize` (Gemma 4's
  4096-token prefill chunk becomes a real path, not a placeholder).
- Hybrid families: chunked prefill for the attention layers; the
  SSM/GDN recurrent layers still step per token (their recurrence is
  inherently sequential) until the deferred chunked-scan kernels land.

**Prioritized** — wanted before the Phase 6.5 Vision wave.
`forwardMulti` is also the Phase 8.0 speculative-decoding prereq, so
this de-risks Phase 8.

**Tests:** prefill correctness (chunked vs per-token prefill produce
identical logits) + a TTFT regression check.

---

## Phase 7 — Audio (STT + TTS + Omni) ✅ SHIPPED (overshipped)

Phase 7 was scoped as Whisper + Kokoro + Qwen-Omni. Reality shipped
the full mlx-audio-swift surface plus VAD + STS (audio enhancement /
source separation / segmentation), plus the FishSpeech dual-AR
family and its FishS1DAC codec.

Shipped families:
- **STT:** `Whisper.swift`, `SenseVoice.swift`, `Parakeet.swift`,
  `FireRedASR2.swift`, `Qwen3ASR.swift`, `VoxtralRealtime.swift`
  (Mistral streaming), `GLMASR.swift`, `CohereTranscribe.swift`,
  `GraniteSpeech.swift`.
- **TTS:** `Kokoro.swift`, `LlamaTTS.swift`, `Marvis.swift` (CSM
  acoustic; Mimi codec wired separately, codec-port follow-up),
  `Qwen3TTS.swift`, `Qwen3TTSBase.swift`, `EchoTTS.swift`,
  `Chatterbox.swift` (Resemble T3 + S3Gen + HiFi-GAN),
  `MossTTS.swift`, `MossTTSNano.swift`, `PocketTTS.swift`,
  `Soprano.swift`, `StyleTTS2.swift`, `FishSpeech.swift` +
  `FishS1DAC.swift` codec.
- **Omni:** `QwenOmni.swift`, `LFMAudio.swift` (Liquid AI Conformer
  + LFM2 backbone).
- **VAD (separate `VADModelRegistry`):** `SileroVAD.swift`,
  `SmartTurn.swift`, `Sortformer.swift` (diarization),
  `TenVAD.swift` (TEN-framework), `FireRedVAD.swift`.
- **STS / audio enhancement** (new `Capability.speechToSpeech`):
  `DeepFilterNet.swift`, `MossFormer2SE.swift`, `SAMAudio.swift`
  (audio segmentation).
- **Codecs:** `BigVGAN.swift`, `Vocos.swift`, `DACVAE.swift`,
  `DescriptDAC.swift`, `Encodec.swift`, `Mimi.swift`, `SNAC.swift`,
  `FishS1DAC.swift`.
- **Infrastructure:** `AudioEncoder.swift` (with parallelised
  `cpuAttention` — Whisper transcribe fix), `AudioPreprocessing.swift`
  (mel + framing + log10 / Slaney variants), `AudioGenerationModel`
  protocol + `AudioGenerationParameters` + `AudioGenerationError`,
  shared `AudioFixtures.swift` (clean_001.wav fixture +
  `resolveCheckpoint(mlxAudioSlugs:repoIds:)`).
- **Kernels (metaltile `ffai/`):** `mel_spectrogram`, `audio_conv1d`.

Same testing-gap caveat as Phase 6.5: every family ships an assertive
integration test, but most have not yet been run against a cached
checkpoint. The first `make test-integration --filter <Family>` pass
will surface any remaining bugs.

Codec follow-ups:
- Marvis Mimi-decoder wiring — `MarvisModel.mimiDecoder` is nil;
  hook the existing `Sources/FFAI/Audio/Mimi.swift` in.
- FishSpeech end-to-end — codec wired, integration test still
  asserts the staged path because dilated/transposed Conv1d
  primitives haven't landed. Switch the assertion once those kernels
  ship.

Original 7 goal text preserved below for context.

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

- `Tests/ModelIntegrationTests/WhisperIntegrationTests.swift` — coherent
  transcription on a known sample.
- `Tests/ModelIntegrationTests/KokoroIntegrationTests.swift` (or Bark) —
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
  `KVCache`, `AffineQuantizedKVCache`, `AURAQuantizedKVCache` (incl.
  compressed-mode), `SSMStateCache`, `GDNStateCache`,
  `Mamba2LayerCache`.
- `LastAssistantOpenerPolicy` for Qwen / Gemma / GPT-OSS chat
  templates.
- L2 disk persistence (opt-in `FFAI_PREFIX_CACHE_DISK=1`) at
  `~/.cache/ffai/prefix/`.
- `generate(...)` wraps a stream that snapshots post-prefill.
- Integration test verifies warm-turn TTFT speedup.

### 8.3 — Compressed-domain prefix KV cache (spec 039)

- Reuses `AURAQuantizedKVCache.fusedEncodeDispatch` for snapshot-time
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

### 8.14 — KV cache write fusion (spec 024)

- Eliminate the `copy_bfloat16` dispatches per decode token.

### 8.15 — Profile-guided Morton-order expert reorder (spec 026)

### 8.16 — Adaptive per-layer mixed-precision (spec 027)

- Recipe-driven framework via JSON sidecar + glob-pattern matching.

### 8.17 — Quadratic / chunkwise WY GDN prefill (spec 028)

- Highest research bet. Could regress if it doesn't work.

### 8.18 — DFlash on GPU (spec 015, phases 1–3)

- Phase 2: draft model from `z-lab/Qwen3.5-*-DFlash`.
- Phase 3: refactor onto `StateReplayCache` protocol.

### 8.19 — Mirror SD (spec 021) + ANE concurrency primitives (spec 025)

- ANE + GPU concurrency primitives (spec 025) land first.
- Then `MirrorSpeculativeLoop` glue.
- Decision point: ANE + GPU truly concurrent on Apple Silicon?
  Failure here kills 021 + spec 029. Document and pivot.

### 8.20 — ANE-offloaded LM head + Gemma 4 PLE projection (spec 029)

- Blocked on spec 025 + Mirror SD measurement.

### 8.21 — Active KV cache SSD offload (spec 038)

- Long-context memory reduction. Multi-month. Only justified if
  long-context single-request use cases matter.

### 8.22 — Flash-quantized SDPA + Metal kernel SIMD audit (specs 041, 042)

- Spec 041: drop-in Flash-tiled fused kernel for the affine
  quantized SDPA path.
- Spec 042: cross-kernel SIMD audit — convert AURAFlash + affine
  flash + `aura_dequant_rotated` + `mse_*` to
  `simdgroup_matrix_multiply_accumulate` MMAs.

### 8.23 — AURAFlash decode-time kernel uplift (spec 043)

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
- **Metaltile housekeeping** —
  - Runtime dispatch-shape validator: reject degenerate threadgroup
    geometry (the < 32-threads-per-group freeze class) before the
    dispatch reaches the GPU.
  - Codegen i32-signedness preservation through lowering (signed
    operands currently widen to unsigned in some paths).
  - Fix the flaky `matches_cpu_reference_f16_chained_resident_gqa`
    GPU correctness test.

**Done when:** generated kernels are within ≤2% of hand-tuned
variants for representative shapes per kernel, and argument-buffer +
ICB modes are selectable via `LoadOptions.dispatchMode` behind the
same Model API.

---

## Phase 10 — Polish

- **`ffai convert` v1** ✅ SHIPPED — Swift-native MLX 4-bit affine
  quantizer (`Sources/FFAI/{SafeTensorsWriter,ConvertDriver}.swift` +
  `Sources/FFAICLI/ConvertCommand.swift`, commit
  `d3b67c5`). Accepts HF repo id or local path, reads source
  safetensors, dispatches `QuantizedOps.quantizeAffine` per Linear-
  shaped 2D weight, writes the `.weight` / `.scales` / `.biases`
  triplet to a fresh safetensors file, patches `config.json` with
  the `quantization` + `quantization_config` blocks, copies
  tokenizer + aux files (symlink-resolved), and optionally uploads
  via the `hf` CLI. Closes the gap that blocked Soprano-1.1,
  Nemotron-H, and FastVLM — `mlx-lm` / `mlx-vlm` fail those because
  they import the model's custom `modeling_*.py` chain;
  `ffai convert` reads raw safetensors only.

- **`ffai convert` v2 — format + precision coverage**
  - **Mixed-precision recipes.** The FFAI loader already supports
    per-tensor bit-width via the `affineBits` map (see Phase 5c).
    Expose this through `ffai convert`:
      - `--bits-embed 8 --bits-linear 4 --bits-lm-head 8` —
        per-role overrides for the three common cases.
      - `--recipe {mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6}` —
        mlx-lm-compatible recipes that bucket tensors by
        sensitivity (attention vs MLP, first/last layers, etc.).
      - Custom recipe file (`--recipe-file <path>`) — JSON map
        from tensor-name glob to bit-width.
  - **GGUF read.** Add `Sources/FFAI/GGUFReader.swift` — parse the
    GGUF v3 header + the K-quant blocks (Q4_K_M, Q5_K_M, Q6_K, Q8_0,
    Q4_0, Q5_0). For each K-quant block, dequantize to bf16 in a
    scratch buffer, then re-quantize through
    `QuantizedOps.quantizeAffine` to the FFAI mlx-4bit layout. Pairs
    with the existing **gguf format support** row below — Phase 10
    "load GGUF directly" lands as `ffai convert <gguf-file> -o
    <mlx-4bit-dir>` first, then a direct GGUF→model loader once we
    confirm there's user demand for skipping the conversion.
  - **GGUF write.** Inverse of the read path — emit a single-file
    `model.gguf` with the FFAI affine weights → Q4_K_M /
    Q5_K_M / Q8_0 conversion. Lets FFAI act as a bridge to
    llama.cpp consumers.
  - **Streaming convert.** Today `SafeTensorsWriter` buffers every
    quantized tensor in RAM before flushing. For 30B+ models this
    overflows. Add a streaming mode that opens the output file
    once, writes a placeholder header, appends each tensor's bytes
    as soon as the quantize kernel returns, and back-patches the
    header at the end. Optionally shard into `model-00001-of-N.safetensors`
    when the total exceeds a configurable size (5 GB default —
    matches HF's chunking convention).
  - **Source coverage.** Today `ffai convert` accepts local paths
    and HF repo ids. v2 adds: (a) a `--source-url <url>` path that
    pulls a raw safetensors URL (S3, GitHub Releases, etc.) without
    a repo wrapper, useful for un-published fine-tunes; (b) a
    direct path to a single `.safetensors` file (not a directory)
    that auto-discovers the matching `config.json` / tokenizer
    files from the same directory.
  - **VLM-aware skip lists.** Several VLM checkpoints (Qwen 2.5-VL,
    LFM2-VL, etc.) ship a `skip_vision: true` quantization hint —
    the vision tower stays bf16 while only the text backbone gets
    quantized. Honor that flag in `ConvertDriver` and add
    `--skip-vision` / `--quantize-vision` overrides.

- **gguf format support** — per-architecture name mapper. Single-file
  format from llama.cpp, embeds quantization (Q4_K_M, Q5_K_M, Q8_0,
  etc.) and tokenizer. Worth doing if community gguf quants are
  valuable to users; skip if all target checkpoints are mlx-format
  or safetensors. The `ffai convert` v2 GGUF read/write paths above
  give us the per-block dequant arithmetic; loading GGUF directly
  is just wiring that into the `Model.load` dispatch instead of
  routing through a convert step.

- **Distribution** — Homebrew formula for the `tile` binary, SPM
  consumer instructions.
- **Benchmarks** — full sweep vs MLX baseline across the model zoo.
- **Documentation site polish.**

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
