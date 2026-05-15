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
   Adds a new `metaltile-emit` bin that produces `kernels.metallib`,
   `manifest.json`, and `MetalTileKernels.swift`.
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
| Kernel author flow | Rust `#[kernel]` → SPM build plugin invokes `metaltile-emit` automatically |
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
    `mlx-lm` for capture (closer architectural match than PyTorch
    for the model variants we target). Writes activations + token
    sequences to `Tests/Fixtures/<model>/`.
  - Tests load the fixtures and compare with tolerance.
  - When a fixture needs regeneration: developer runs the capture
    script locally on a verified setup, commits the new files. The
    fixture file's `metadata.json` records mlx-lm version + capture
    date.
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
- TurboQuant / state-replay (Phase 4)
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
GPT-OSS, fused MoE expert kernels, GDN steps, TurboQuant codecs) get
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

## Phase 4 — Advanced kernels (TurboQuant, GDN, SSM)

**Goal:** Port the high-value custom kernels currently in mlx-swift-lm.
These were the original motivator for this project — the 4-repo dance to
ship a new TurboQuant variant is the pain we're eliminating.

**DSL prerequisites in metaltile (must land first):**

- Sub-byte packed dtypes (`Packed4`, `Packed2`) with bit-unpack ops
- `simd_shuffle_xor` (needed for FWHT butterfly)
- Function-constant integration in the `launch` builder
- Persistent state-buffer convention (in/out aliasing the same MTLBuffer)
- Type-checked launch builder (closes the v0.2 "shape algebra" gap)

**TurboQuant kernels (port from `Libraries/MLXLMCommon/TurboQuantKernels.swift`):**

- `turbo_encode` (dense rotation Π + Lloyd-Max + bit-pack + norm correction)
- `turbo_encode_wht` (FWHT butterfly variant)
- `turbo_bulk_dequant_rotated`
- `turbo_score`, `turbo_value` (compressed-domain attention)
- `turbo_flash_pass1` / `turbo_flash_pass2` + causal/NR0 variants
- `turbo_flash_sdpav` (single-dispatch fused)

**SSM / GDN kernels:**

- `gated_delta_step` + `gated_delta_step_record` (with delta tape)
- `state_replay` (refold accepted prefix from tape)
- `ssm_kernel` (Mamba selective scan)
- Tests for masked-timestep correctness + branchless SIMD pattern

**FFAI changes:**

- `TurboQuantizedKVCache` (port the two-phase prefill+compress logic)
- `SSMStateCache` with rollback
- `GatedDelta`, `SSM` model layers
- Hybrid model support (Qwen 3.5 etc.)

**Done when:** Qwen 3.5 (hybrid GDN+attention with TurboQuant KV) runs
end-to-end with measured tokens/sec ≥ current mlx-swift-lm baseline.

---

## Phase 5 — First multi-modal model (vision)

**Goal:** Stress-test the Capability + lifecycle infrastructure with a
real multi-modal model. Validate that disabled-by-default modalities
genuinely don't allocate, that lazy `enable(.visionIn)` works, and
that lifecycle events stream correctly to consumers.

**Target model:** Qwen2.5-VL or Qwen3.5-VL (decide closer to Phase 5
based on what's most demanded; we don't need to commit now).

**Kernels (new):**

- Vision encoder primitives: 2D conv, patch embedding, vision
  positional embedding (often just learned), vision-specific attention
- Cross-modal token splicing (image tokens interleaved with text tokens)
- Any vision-specific normalization layers

**FFAI changes:**

- New `VisionEncoder` module type
- `Models/Qwen3.swift` gains `Qwen35VL` variant composing
  `Qwen35HybridDense` (text backbone) + `VisionEncoder`
- `Capability.visionIn` exercised end-to-end:
  - Load with `[.textIn, .textOut]` only → no vision weights allocated
  - `await model.enable(.visionIn)` → mmaps vision weights, builds
    encoder, prewarms vision kernels, emits lifecycle events
  - `await model.disable(.visionIn)` → releases MTLBuffer refs and
    encoder, frees GPU residency
- Image preprocessing pipeline (resize, normalize, patchify) — CPU
  for now, can move to Metal later if it shows up in profiles

**Tests:**

- Capability matrix: every (textIn, textOut, visionIn) subset
- Vision encoder forward correctness vs golden fixture (captured
  from mlx-vlm)
- Multi-modal generation (image + prompt → text) determinism
- Memory footprint reported correctly per capability

---

## Phase 5 — Autotuner

Implement `metaltile-runtime`'s autotuner for real (currently stubbed):

- Grid search over `(tile_dims, threads, unroll, simd_matrix, async_copy)`
- Persist to `~/.cache/metaltile/tuning_cache.json`
- Shape-bucket lookup at emit time
- CI: nightly autotune on a reference machine, commit results

**Done when:** Generated kernels are within ≤2% of hand-tuned variants
for representative shapes per kernel.

---

## Phase 7+ — Audio, more model families, polish

- Audio capability (`.audioIn` for STT like Whisper, `.audioOut` for
  TTS) — first audio target TBD
- Omni models (Qwen3.5-Omni or similar) — vision + audio simultaneously
- Port additional model families as demanded (Mistral, Phi, Gemma,
  GPT-OSS MoE, Bitnet, etc.) — each is a single new family file
- Embedding-only models (`Qwen3Embedding`, etc.)
- **gguf format support** (optional) — single-file format from
  llama.cpp, embeds quantization (Q4_K_M, Q5_K_M, Q8_0, etc.) and
  tokenizer. Different binary layout from safetensors and different
  tensor naming conventions; needs a per-architecture name mapper.
  Worth doing if community gguf quants are valuable to users. Skip
  if all checkpoints we care about are mlx-format or safetensors.
- Dispatch mode upgrades: ship `.argumentBuffers` and `.icb` modes
  if Phase 5 profiling shows Mode 1 encoding cost is a real
  bottleneck on larger models
- Documentation, examples, distribution polish, perf benchmarking
  vs MLX baseline across the model zoo

---

## Out of scope / deferred

- **CoreML / ANE backend.** Realistic only for boring kernels (RMSNorm,
  RoPE, layer norm, plain GEMV at fp16/int8). TurboQuant, FWHT, online
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
