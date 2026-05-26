# Architecture

FFAI is three layers, all in this repo except `metaltile` (a sibling
Rust crate). The longer-form diagrams live in
[`planning/architecture.md`](../planning/architecture.md); this
page covers the user-facing model — what each layer is responsible
for and how a single token moves through the stack.

## The three layers

```
┌─────────────────────────────────────────────────────────┐
│  FFAI (Swift)                                           │
│   • Tensor (MTLBuffer-backed)                           │
│   • Module / Linear / Embedding / RMSNorm               │
│   • Model definitions (Llama, Qwen, …)                  │
│   • SafeTensors loader                                  │
│   • KV cache, sampling, generate loop                   │
└────────────────────────┬────────────────────────────────┘
                         │ calls
┌────────────────────────▼────────────────────────────────┐
│  MetalTileSwift (Swift, in-repo)                        │
│   • Loads kernels.metallib (pre-compiled at build time) │
│   • PSO cache, function-constant specialization         │
│   • Generated typed wrappers (one per kernel)           │
└────────────────────────┬────────────────────────────────┘
                         │ resources from
┌────────────────────────▼────────────────────────────────┐
│  metaltile (Rust, sibling repo)                         │
│   • #[kernel] DSL → IR → MSL                            │
│   • `tile build --emit all` (metaltile-cli) produces:               │
│       kernels.metallib   (compiled by xcrun metal)      │
│       manifest.json      (kernel metadata)              │
│       MetalTileKernels.swift  (typed wrappers)          │
└─────────────────────────────────────────────────────────┘
```

### `metaltile` (Rust)

A `#[kernel]` proc-macro lowers a small Rust DSL into an IR; the
codegen back-end emits Metal Shading Language. Authoring lives here:
new kernels are Rust `pub fn`s in `crates/metaltile-std/src/ops/`,
annotated with `#[bench_kernel(...)]` so the registry picks them up.
End users never touch Rust — they consume the artifacts shipped in
`Sources/MetalTileSwift/Resources/`.

### `MetalTileSwift`

A thin Swift wrapper that loads `kernels.metallib` once
(`MetalTileLibrary.shared`), maintains a PSO cache keyed on
`(name, MTLFunctionConstantValues)`, and exposes one typed
function per kernel via the generated `MetalTileKernels.swift`.
Function-constant specialization lets us produce a single MSL source
that spans dtype/shape variants — at PSO instantiation time the
constants get baked into the pipeline.

### `FFAI`

The user-facing layer:

- **`Tensor`** — `MTLBuffer` + shape + dtype + strides + byte offset.
  Owns memory directly; no `MLXArray` middle-man.
- **`Module`** — protocol with named-parameter discovery.
- **Layers** — `Linear`, `Embedding`, `RMSNorm`, `RoPE`, attention
  blocks. Each is a thin call to `MetalTileSwift` kernels.
- **Models** — one Swift file per family per folder. Text-only
  families live at `Models/Text/<F>.swift` (`Models/Text/Llama.swift`,
  `Models/Text/Mistral.swift`). VL families add a `Models/<F>.swift`
  orchestrator + `Models/Vision/<F>Vision.swift` tower; the paired
  text impl lives at `Models/Text/<F>Text.swift` (e.g.
  `Models/Text/Qwen3Text.swift`). Audio families live under
  `Models/Audio/{STT,TTS,STS,VAD,Omni}/<F>.swift`. Family files use
  a protocol + per-variant struct pattern so adding `Qwen35MoE` etc.
  doesn't bloat a switch. See
  [developing/adding-a-model.md](developing/adding-a-model.md) for
  the full layout rule.
- **Loader** — `Model.load(...)` resolves an HF id (or local path),
  downloads via `swift-huggingface`, parses `config.json`, mmap-loads
  weights into per-tensor MTLBuffers, dispatches to the right family,
  attaches the tokenizer.
- **Inference** — `KVCache`, `Sampling`, `Generate` (the prefill +
  decode loop).

## The build pipeline

```
┌──────────────┐ tile build        ┌──────────────────┐  xcrun metal   ┌────────────────────┐
│  Rust kernels│ --emit all --out  │  *.metal sources │   + metallib   │  kernels.metallib  │
│  (#[kernel]) │ ─────────────────▶│  manifest.json   │ ──────────────▶│  manifest.json     │
└──────────────┘                   │  MetalTileKernels│                │  MetalTileKernels  │
                                   │  .swift (typed)  │                │  .swift (typed)    │
                                   └──────────────────┘                └────────────────────┘
                                                                              │
                                                                              ▼
                                                                ┌──────────────────────────┐
                                                                │ Sources/MetalTileSwift/  │
                                                                │   Resources/             │
                                                                │   Generated/             │
                                                                └──────────────────────────┘
                                                                              │
                                                                              ▼
                                                                  Bundled into the SwiftPM
                                                                  package — end users get
                                                                  a pre-compiled metallib
```

Kernel regeneration is `make regenerate-kernels` (which `make build`
runs automatically). End users adding FFAI as a SwiftPM dep don't run
this — they consume the metallib that ships in the package.

## A single token, end-to-end

This is the dispatch path for one decode step on a Llama-shaped
model. The same path runs for every prompt token (slow prefill) and
every generated token (decode loop):

```
User: model.engine.forwardSample(tokenId: t, position: pos, caches: caches)
            │
            ▼
   ┌─────────────────────────────────────────────────────┐
   │  open one MTLCommandBuffer                          │
   │                                                      │
   │  gather    (token id → embedding vector)            │
   │                                                      │
   │  for each transformer layer:                         │
   │    rms_norm                                          │
   │    Q/K/V projections (gemv or dequant_gemv)          │
   │    rope                                              │
   │    [Qwen3 only: per-head q_norm / k_norm RMSNorm]    │
   │    kv_cache_update    (append K/V on the GPU)        │
   │    sdpa_decode        (one Q-row × cached K/V)       │
   │    O projection                                      │
   │    add  (residual)                                   │
   │    rms_norm                                          │
   │    SwiGLU MLP: gate, up, silu, mul, down             │
   │    add  (residual)                                   │
   │                                                      │
   │  rms_norm (final)                                    │
   │  LM-head gemv → logits                               │
   │  argmax  (GPU-side; writes a single uint32)          │
   │                                                      │
   │  commit + waitUntilCompleted                         │
   └─────────────────────────────────────────────────────┘
            │
            ▼
   read 4 bytes → return next token id
```

**Invariants the code maintains:**

1. **One `MTLCommandBuffer` per token.** No mid-token sync. Every
   layer's kernels enqueue onto the same buffer.
2. **No CPU↔GPU sync inside a layer.** KV cache append is the
   `kv_cache_update` Metal kernel — not a CPU memcpy.
3. **No logits readback.** Sampling runs on the GPU
   (`argmax` today; top-k / top-p / temperature land in Phase 5+).
   Only the chosen token id (4 bytes) crosses CPU↔GPU per token.
4. **Weights are immutable post-load.** Per-tensor MTLBuffers are
   allocated once, never resized. Activations come from a
   `BufferPool` so per-token allocation doesn't grow.

## Capability-driven loading

A `Model` has two `Capability` sets:

- `availableCapabilities` — what the family declares it can do
  (`Llama` is `[.textIn, .textOut]`; the VL families add `.visionIn`).
- `enabledCapabilities` — what the user opted into via
  `LoadOptions.capabilities`.

Disabled modalities skip weight allocation entirely — the vision
encoder of a 9B VL model is ~600MB you don't pay for if you only need
text. The infrastructure has been in place since Phase 2; the
vision-language and audio families now exercise it end-to-end.

## File layout

```
Sources/
  FFAI/                    User-facing library
    Tensor.swift           MTLBuffer + shape/dtype/strides
    BufferPool.swift       Per-token activation slab allocator
    Device.swift           MTLDevice + MTLCommandQueue singleton
    Module.swift           Parameter discovery protocol
    Layers.swift           Linear / Embedding / RMSNorm / etc.
    Ops.swift              Public ops (gemv, rope, sdpa, argmax, …)
    KVCache.swift          Raw fp16/bf16 cache + GPU append
    Sampling.swift         argmax / top-k / top-p (CPU paths)
    Generate.swift         Prefill + decode loop
    SafeTensors.swift      *.safetensors loader
    Model.swift            High-level Model.load(...) entry point
    ModelConfig.swift      config.json decoder
    ModelDownloader.swift  HF Hub snapshot download/cache
    ModelLocator.swift     Repo id ↔ local dir resolver
    ModelLifecycle.swift   AsyncStream<Event> state machine
    Capability.swift       .textIn / .visionIn / etc.
    LoadOptions.swift      What the user requests at load
    LanguageModel.swift    Protocol implemented by family models
    TokenizerLoader.swift  AutoTokenizer.from(modelFolder:) wrapper
    Models/
      Llama.swift          Llama 3.x (LlamaDense)
      Qwen3.swift          Qwen 3 (Qwen3Dense)

  MetalTileSwift/          Pre-compiled kernels + dispatch wrappers
    MetalTileLibrary.swift  Singleton MTLDevice + MTLLibrary loader
    PSOCache.swift          (name, function-constants) → PSO
    Resources/              kernels.metallib + manifest.json
    Generated/              MetalTileKernels.swift (typed wrappers)

  FFAICLI/                 ffai executable
    main.swift

Tests/
  MetalTileSwiftTests/         One file per kernel
  FFAITests/                   Tensor, Module, KVCache, Sampling, …
                               (mirrors Sources/FFAI/ — every source file
                               has a sibling Tests/FFAITests/<X>Tests.swift)
  ModelIntegrationTests/       Per-family integration tests — load,
                               greedy-decode, assert coherent output
```

## Where to read more

- [`planning/architecture.md`](../planning/architecture.md) — fuller
  diagrams (build pipeline, model load sequence, dispatch loop,
  threadgroup mapping per kernel).
- [`planning/plan.md`](../planning/plan.md) — phased build-out and
  the rationale for what's in / out of scope per phase.
- [Models](models.md) — what's actually supported today.
- [KV cache](kv-cache.md) — current cache and the planned
  affine / AURA / SSM variants.
