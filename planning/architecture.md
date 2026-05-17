# FFAI — Architecture

This document is a **visual** reference for how FFAI works end-to-end:
how Metal kernels are generated, how kernels are loaded into the
process, how a model is downloaded and bound to GPU memory, and how a
single inference token flows through the system.

For the *what we're building and when* see [`plan.md`](plan.md).

---

## 1. Build-time pipeline (kernel generation)

Runs once per build (or whenever a Rust `#[kernel]` definition
changes). Produces three artifacts that the Swift package consumes as
resources.

```
                           BUILD TIME
                           ──────────

  ┌──────────────────────────────────────────────────────────────┐
  │  metaltile (Rust workspace, sibling repo)                    │
  │                                                              │
  │   #[kernel] fn rms_norm<T>(x, w, out, eps, n) { … }          │
  │   #[kernel] fn rope(q, k, freqs, offset, n) { … }            │
  │   …                                                          │
  │                                                              │
  │              │ proc-macro expansion → IR                     │
  │              ▼                                               │
  │   ┌──────────────────────────┐                               │
  │   │  metaltile-core          │   SSA IR (Kernel, Op,         │
  │   │  metaltile-codegen       │   Block, ConstExpr, …)        │
  │   │                          │                               │
  │   │  6-pass pipeline:        │                               │
  │   │  TypeCheck → ConstFold   │                               │
  │   │  → TileLowering          │                               │
  │   │  → Fusion → Schedule     │                               │
  │   │  → Vectorize             │                               │
  │   └─────────────┬────────────┘                               │
  │                 ▼                                            │
  │           MSL source text (.metal)                           │
  └───────────────────┬──────────────────────────────────────────┘
                      │
                      ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  tile build --emit all --out Sources/MetalTileSwift          │
  │  (metaltile-cli, dev sibling repo)                           │
  │                                                              │
  │  Walks the inventory of `BenchSpec`-registered kernels,      │
  │  for each (kernel × dtype):                                  │
  │   1. Codegen MSL → write Resources/kernels/<name>.metal      │
  │   2. Append to Resources/manifest.json (name, params,        │
  │      dtype, constexprs, mode)                                │
  │   3. Append to Generated/MetalTileKernels.swift              │
  │      (one strongly-typed Swift func per kernel)              │
  │                                                              │
  │  Then: shell out to                                          │
  │   xcrun -sdk macosx metal -c   *.metal → *.air               │
  │   xcrun -sdk macosx metallib   *.air   → kernels.metallib    │
  │                                                              │
  │  Emit logic lives in metaltile-codegen::emit so other        │
  │  tooling (build scripts, future SPM plugins) can consume     │
  │  it without going through the CLI binary.                    │
  └───────────────────┬──────────────────────────────────────────┘
                      │
                      ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Sources/MetalTileSwift/                                     │
  │   ├─ Resources/                                              │
  │   │   ├─ kernels.metallib       ← compiled Metal binary      │
  │   │   ├─ manifest.json          ← per-kernel metadata        │
  │   │   └─ kernels/*.metal        ← MSL sources (debug aid)    │
  │   └─ Generated/                                              │
  │       └─ MetalTileKernels.swift ← typed Swift wrappers       │
  └──────────────────────────────────────────────────────────────┘

           │
           │  triggered automatically by SPM build plugin
           │  (MetalTileEmitPlugin) on every `swift build`
           ▼
   swift build / swift test  →  ready Swift package
```

**Key properties:**

- Kernel changes never require Swift source edits — the wrappers and
  metallib are regenerated.
- The `.metal` files are checked in (per `.gitignore` policy: optional;
  default is regenerate). This makes diffs reviewable and lets
  contributors inspect what got generated without running cargo.
- All compilation happens at build time. **Zero runtime JIT.**

---

## 2. Process startup (kernel loading)

Happens once when the FFAI process launches.

```
                          PROCESS STARTUP
                          ───────────────

  app starts
      │
      ▼
  ┌────────────────────────────────────────┐
  │  MetalTileLibrary.shared (lazy)        │
  │                                        │
  │  1. MTLCreateSystemDefaultDevice()     │
  │  2. device.makeLibrary(URL: bundle     │
  │       resource "kernels.metallib")     │
  │  3. parse manifest.json                │
  │                                        │
  │  Result:                               │
  │   • MTLDevice                          │
  │   • MTLCommandQueue                    │
  │   • MTLLibrary (all kernels resident)  │
  │   • [String: KernelDescriptor]         │
  └─────────────────┬──────────────────────┘
                    │
                    ▼
  ┌────────────────────────────────────────┐
  │  PSOCache.shared                       │
  │                                        │
  │  Empty at start. Populated lazily on   │
  │  first call to each kernel/spec combo. │
  │                                        │
  │  key  = (kernelName, sortedFnConsts)   │
  │  val  = MTLComputePipelineState        │
  │                                        │
  │  miss → MTLLibrary.makeFunction(       │
  │           name:, constantValues:)      │
  │       → device.makeComputePipelineState│
  │       → store in cache                 │
  └────────────────────────────────────────┘

  Cost on first call to a (kernel, fn-consts) combo:
   • ~100-500µs PSO compilation (Metal driver, no MSL JIT —
     the MSL is already pre-compiled into kernels.metallib;
     the cost is just specialization with constants + reflection)
  Cost on subsequent calls:
   • ~1µs hash lookup
```

**Why no MSL JIT:** the `.metallib` already contains the parsed,
type-checked, optimized AIR for every kernel. The only per-spec cost
is creating the PSO with the right function-constant values, which
Metal does without invoking the front-end compiler.

---

## 3. Model loading (capability-driven)

Models are organized into **family files** (one per major
architectural lineage: `Llama.swift`, `Qwen3.swift`, `Whisper.swift`)
that contain multiple **variants** (each a struct conforming to a
family-specific protocol). At load time the user picks which
**capabilities** to enable; disabled modalities skip weight allocation
entirely.

```
                       MODEL FILE LAYOUT
                       ─────────────────

  Sources/FFAI/Models/
   ├─ Llama.swift          (Llama 3.x: 1B, 3B, 8B, 70B)
   ├─ Qwen3.swift          (Qwen3, Qwen3.5 dense+hybrid+MoE+VL+Omni,
   │                        Qwen3.6 …)
   ├─ Mistral.swift        (Mistral 7B, Mixtral 8x7B/8x22B)
   ├─ Phi.swift            (Phi-3, Phi-3.5)
   ├─ GPTOSS.swift         (GPT-OSS 20B, …)
   ├─ Whisper.swift        (STT family)
   └─ Bark.swift           (TTS family)

  Inside Qwen3.swift:
  ┌──────────────────────────────────────────────────────────┐
  │ public enum Qwen3: ModelFamily {                         │
  │   static func variant(for: ModelConfig) -> any           │
  │       Qwen3Variant.Type { … }                            │
  │ }                                                        │
  │                                                          │
  │ protocol Qwen3Variant {                                  │
  │   static var availableCapabilities: Set<Capability>      │
  │   static func load(...) async throws -> Module           │
  │ }                                                        │
  │                                                          │
  │ struct Qwen3Dense:         Qwen3Variant { … }            │
  │ struct Qwen35HybridDense:  Qwen3Variant { … }            │
  │ struct Qwen35HybridMoE:    Qwen3Variant { … }            │
  │ struct Qwen35VL:           Qwen3Variant { … }  // text+vision
  │ struct Qwen35Omni:         Qwen3Variant { … }  // text+vision+audio
  └──────────────────────────────────────────────────────────┘
```

```
                       LOADING FLOW
                       ────────────

  user provides:
   • repo id "Qwen/Qwen3.5-9B-VL"          OR
   • local path "/path/to/local-model"
   • LoadOptions(capabilities: [.textIn, .textOut],
                 kvCache: .gigaQuantized(scheme: "giga4v2"),
                 lazyCapabilities: true)
       │
       ▼
  ┌────────────────────────────────────────────────────────┐
  │  Model.load(...) — async                               │
  │  emits ModelLifecycleEvent on the AsyncStream as it    │
  │  progresses through each stage                         │
  └─────────────────┬──────────────────────────────────────┘
                    │
                    ▼
  ┌────────────────────────────────────────────────────────┐
  │  STAGE: downloading(Progress)                          │
  │                                                        │
  │  ModelLocator.resolve(id-or-path) →                    │
  │    ModelDownloader (swift-huggingface HubClient)            │
  │    HubClient.downloadSnapshot(                         │
  │      of: repoID, matching: [...],                      │
  │      progressHandler: → emits .downloading events)     │
  │    Cache: ~/.cache/huggingface/hub/                    │
  └─────────────────┬──────────────────────────────────────┘
                    │
                    ▼
   /Users/.../hub/models--Qwen--Qwen3.5-9B-VL/snapshots/<sha>/
     config.json
     tokenizer.json, tokenizer_config.json, *.jinja
     model-00001-of-00004.safetensors, …, model.safetensors.index.json
                    │
                    ▼
  ┌────────────────────────────────────────────────────────┐
  │  STAGE: loading(LoadProgress) — config phase           │
  │                                                        │
  │  ModelConfig.decode(config.json)                       │
  │    → architecture / model_type                         │
  │    → all hyperparameters                               │
  │    → has("vision_config")? has("audio_config")?        │
  │                                                        │
  │  ModelRegistry.family(for: config) → Qwen3.self        │
  │  Qwen3.variant(for: config) → Qwen35VL.self            │
  │                                                        │
  │  Qwen35VL.availableCapabilities                        │
  │    = [.textIn, .textOut, .visionIn, .toolCalling]      │
  │                                                        │
  │  effectiveCaps = options.capabilities                  │
  │                ∩ availableCapabilities                 │
  │                ∪ [.textIn, .textOut]   // mandatory    │
  └─────────────────┬──────────────────────────────────────┘
                    │
                    ▼
  ┌────────────────────────────────────────────────────────┐
  │  STAGE: loading(LoadProgress) — weights phase          │
  │                                                        │
  │  SafeTensors.openBundle(directory)                     │
  │    → mmap each .safetensors file                       │
  │    → parse header, build tensor key → (file, offset)   │
  │      lookup table (no allocation yet)                  │
  │                                                        │
  │  For each requested capability, build the corresponding│
  │  sub-module and bind only its weights:                 │
  │                                                        │
  │   ┌─ ALWAYS ───────────────────────────────────────┐   │
  │   │ backbone (Qwen35HybridDense):                  │   │
  │   │   embed_tokens, layers (GDN+attn hybrid),      │   │
  │   │   norm, lm_head                                │   │
  │   │   weight keys: "model.embed_tokens.*",         │   │
  │   │     "model.layers.*", "model.norm.*",          │   │
  │   │     "lm_head.*"                                │   │
  │   └────────────────────────────────────────────────┘   │
  │   ┌─ IF .visionIn requested ──────────────────────┐    │
  │   │ vision encoder:                                │   │
  │   │   patch_embed, vision layers, vision_norm     │   │
  │   │   weight keys: "vision_model.*"                │   │
  │   └────────────────────────────────────────────────┘   │
  │   ┌─ IF .audioIn requested ───────────────────────┐    │
  │   │ audio encoder: "audio_model.*"                 │   │
  │   └────────────────────────────────────────────────┘   │
  │                                                        │
  │  Skipped weight keys are NOT mmapped, NOT allocated;   │
  │  the corresponding submodule is never instantiated.    │
  └─────────────────┬──────────────────────────────────────┘
                    │
                    ▼
  ┌────────────────────────────────────────────────────────┐
  │  STAGE: loaded                                         │
  │                                                        │
  │  TokenizerLoader.load(directory)                       │
  │   → swift-transformers AutoTokenizer.from(...)         │
  │                                                        │
  │  Model is built. Weights resident in MTLBuffers.       │
  │  No PSOs compiled yet (lazy on first dispatch).        │
  └─────────────────┬──────────────────────────────────────┘
                    │
                    ▼
  ┌────────────────────────────────────────────────────────┐
  │  STAGE: ready (after prewarm — default ON)             │
  │                                                        │
  │  Model.prewarm() runs by default                       │
  │  (LoadOptions.prewarm = true).                         │
  │   • Iterate every kernel needed by enabled capabilities│
  │   • Compile PSO with the right function constants      │
  │   • Touch each weight MTLBuffer to page it in          │
  │   • Now first-token latency = pure GPU work, no        │
  │     compilation cost, no cold-cache page faults        │
  │                                                        │
  │  Opt out with LoadOptions(prewarm: false) if loading   │
  │  many models at once and warming on demand.            │
  └────────────────────────────────────────────────────────┘
```

### 3a. Lifecycle state machine

```
                  MODEL LIFECYCLE STATES
                  ──────────────────────

           idle
            │
            │ Model.load(...)
            ▼
       downloading(Progress)
            │
            │ download complete (or cache hit)
            ▼
       loading(LoadProgress)
            │
            │ weights mmap'd, modules built
            ▼
         loaded
            │
            │ optional: await model.prewarm()
            ▼
          ready
            │
            │ — generation happens here —
            │
            │ user later: await model.enable(.visionIn)
            ▼
       loading(LoadProgress, capability: .visionIn)
            │
            │ vision weights mmap'd, encoder built, vision
            │ PSOs prewarmed, vision pages touched
            ▼
          ready (now with .visionIn)


  Any state can transition to:
       failed(Error)
  (e.g. network error, missing weight key, OOM, etc.)

  Note: prewarm is part of getting to .ready — both at
  initial load and at enable(_ capability:). The model
  is never .ready unless its kernels are compiled and
  weights are paged in.


  Observation API:
   ┌───────────────────────────────────────────────────────┐
   │ public final class Model {                            │
   │   public let events:                                  │
   │       AsyncStream<ModelLifecycleEvent>                │
   │   public var currentState: ModelLifecycleState        │
   │ }                                                     │
   │                                                       │
   │ public struct ModelLifecycleEvent {                   │
   │   public let capability: Capability?  // nil = whole  │
   │   public let state: ModelLifecycleState               │
   │ }                                                     │
   └───────────────────────────────────────────────────────┘

  Typical UI consumption:
   ┌───────────────────────────────────────────────────────┐
   │ Task {                                                │
   │   for await event in model.events {                   │
   │     switch event.state {                              │
   │     case .downloading(let p): updateBar(p)            │
   │     case .loading(let p):     updateBar(p)            │
   │     case .ready:              hideBar()               │
   │     case .failed(let e):      showError(e)            │
   │     default: break                                    │
   │     }                                                 │
   │   }                                                   │
   │ }                                                     │
   └───────────────────────────────────────────────────────┘
```

### 3b. Capability enable/disable at runtime

```
                      DYNAMIC CAPABILITIES
                      ────────────────────

  Loaded text-only:
                          ┌──────────────────────────┐
                          │ Model (Qwen35VL)         │
                          │  ├─ backbone   ✓ loaded  │  9.0 GB
                          │  ├─ vision     ✗ skipped │
                          │  └─ audio       —        │
                          │ enabled = [textIn,       │
                          │            textOut]      │
                          └──────────────────────────┘
                                     │
                                     │  await model.enable(.visionIn)
                                     ▼
                          ┌──────────────────────────┐
                          │ Model (Qwen35VL)         │
                          │  ├─ backbone   ✓ loaded  │  9.6 GB
                          │  ├─ vision     ✓ loaded  │  (+0.6 GB)
                          │  └─ audio       —        │
                          │ enabled = [textIn,       │
                          │            textOut,      │
                          │            visionIn]     │
                          └──────────────────────────┘
                                     │
                                     │  await model.disable(.visionIn)
                                     ▼
                          ┌──────────────────────────┐
                          │ Model (Qwen35VL)         │
                          │  ├─ backbone   ✓ loaded  │  9.0 GB
                          │  ├─ vision     ✗ released│  (page cache
                          │  └─ audio       —        │   eventually
                          │                          │   reclaims)
                          └──────────────────────────┘

  enable() does:           disable() does:
   1. mmap vision weights   1. drop submodule references
   2. build VisionEncoder   2. release MTLBuffers
   3. prewarm vision PSOs   3. (no kernel uninstall — PSOs stay
   4. emit .ready event        in cache, cheap to re-enable)
   5. atomically add to
      enabledCapabilities
```

**Cost characteristics:**

- Download: dominated by network. First time a 1B model: ~30s on
  reasonable bandwidth. Cached: ~1s (just metadata revalidation).
- mmap: O(1). The 1.2GB of fp16 weights live in the page cache;
  Metal reads them on demand.
- Module instantiation: O(num_layers), microseconds.
- `enable(.visionIn)` on warm cache: ~100ms (mostly module
  construction). Cold cache (after disable + GC): re-mmap is fast.

### 3c. Modality strip — vision + audio encoders hooking into the backbone

Multi-modal models compose a text backbone with optional
`VisionEncoder` and `AudioEncoder` sub-modules. The `Capability`
system decides which encoders are built; the backbone receives
**already-tokenised activations** from each enabled encoder via a
splice helper. Encoders never see the text token stream and the
backbone never sees raw pixels or audio frames — the interface
between them is `[seq, hidden]` activations plus a position map.

```
                       MODALITY STRIP
                       ──────────────

  ┌──────────────────────────────────────────────────────────┐
  │  raw inputs (per request)                                │
  │                                                          │
  │  text:   "describe this image: <img>"                    │
  │  image:  CGImage / Data (PNG, JPEG, …)                   │
  │  audio:  AVAudioPCMBuffer / Float32 [samples]            │
  └──────────────────────────────────────────────────────────┘
        │                  │                    │
        │                  │                    │
        ▼                  ▼                    ▼
  ┌──────────┐    ┌──────────────────┐  ┌──────────────────┐
  │ Tokenizer│    │ ImagePreprocess  │  │ AudioPreprocess  │
  │          │    │  resize/normalize│  │  resample/mel    │
  │          │    │  patchify        │  │  frame           │
  └────┬─────┘    └────────┬─────────┘  └────────┬─────────┘
       │                   │                     │
       ▼                   ▼                     ▼
  text_ids        patches [P, c, h, w]    mel [T, n_mels]
       │                   │                     │
       │                   │                     │
       │          ┌────────▼─────────┐  ┌────────▼─────────┐
       │          │ VisionEncoder    │  │ AudioEncoder     │
       │          │  patch_embed     │  │  audio_conv1d    │
       │          │  + ViT layers    │  │  + transformer   │
       │          │  + vision_proj   │  │  + audio_proj    │
       │          │   to backbone    │  │   to backbone    │
       │          │   hidden dim     │  │   hidden dim     │
       │          └────────┬─────────┘  └────────┬─────────┘
       │                   │                     │
       │            image_emb [Pi, H]      audio_emb [Pa, H]
       │                   │                     │
       ▼                   ▼                     ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Cross-modal splice                                      │
  │                                                          │
  │  Walk text token stream; at each <img> / <audio>         │
  │  placeholder, splice the corresponding encoder output    │
  │  into the activation sequence. Position map records      │
  │  where each token came from so RoPE / position embeds    │
  │  use the right offsets.                                  │
  │                                                          │
  │  Output: spliced_hidden [seq, H],  position_map [seq]    │
  └──────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Text backbone (Llama / Qwen3.5 / Gemma 4 / GPT-OSS / …) │
  │                                                          │
  │  Receives spliced_hidden directly — no embedding lookup  │
  │  on this path (embed_tokens used only for plain text     │
  │  spans). All subsequent layers (attention, MLP, MoE,     │
  │  GDN, SSM, sliding-window) are identical to the          │
  │  text-only forward pass.                                 │
  │                                                          │
  │  Output: token stream — text only, or interleaved with   │
  │  audio tokens when .audioOut is enabled (Omni models).   │
  └──────────────────────────────────────────────────────────┘

  Output side (optional, Omni / TTS models):
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────┐
  │  AudioDecoder (TTS families: Kokoro / Bark)              │
  │                                                          │
  │  Token stream → vocoder → audio waveform.                │
  │  Only built when Capability.audioOut is enabled.         │
  └──────────────────────────────────────────────────────────┘
```

**What the Capability system gates:**

- `.visionIn` → builds `VisionEncoder`, mmaps `vision_model.*` weights,
  prewarms vision PSOs.
- `.audioIn` → builds `AudioEncoder`, mmaps `audio_model.*` weights,
  prewarms audio PSOs.
- `.audioOut` → builds `AudioDecoder` (vocoder + decoder layers),
  mmaps `audio_decoder.*` weights.
- `.toolCalling` → no weight load; toggles the chat template's
  tool-call grammar.

Encoder modules are independent of the backbone's choice of cache
type — vision and audio activations always live in plain fp16 / bf16
since they're encoder-resident and don't get cached across tokens.
GigaQuant / affine quant only apply to the backbone KV cache.

---

## 4. Inference — single token decode

This is the hot loop. For each token to generate, we run one forward
pass through the network. With KV cache, decode does ~N kernel
dispatches per layer (N = a small constant).

```
                       SINGLE-TOKEN DECODE
                       ───────────────────

  prev_token (Int)
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  let cmdBuf = device.commandQueue.makeCommandBuffer()   │
  │  // ALL the layer's kernels go on ONE command buffer    │
  │  // → fewer GPU↔CPU sync points                         │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  embed_tokens(prev_token)                               │
  │   → MetalTileKernels.gather(                            │
  │       table:  embed_tokens.weight,                      │
  │       index:  prev_token,                               │
  │       out:    h /* [1, hidden] */,                      │
  │       on:     cmdBuf)                                   │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ▼
   for each TransformerBlock layer:
   ┌─────────────────────────────────────────────────────────┐
   │  // -- attention --                                     │
   │  MetalTileKernels.rms_norm(                             │
   │     x: h, weight: layer.input_layernorm.weight,         │
   │     out: x_norm, eps:, on: cmdBuf)                      │
   │                                                         │
   │  MetalTileKernels.gemv_fp16(                            │
   │     x: x_norm, w: layer.self_attn.q_proj.weight,        │
   │     out: q, on: cmdBuf)                                 │
   │  // … same for k, v                                     │
   │                                                         │
   │  MetalTileKernels.rope(                                 │
   │     q: q, k: k, freqs: rope_freqs,                      │
   │     offset: cache.length, on: cmdBuf)                   │
   │                                                         │
   │  cache.append(k, v)   // writes into pre-alloc'd        │
   │                       // KV MTLBuffer at offset         │
   │                                                         │
   │  MetalTileKernels.sdpa_decode(                          │
   │     q: q,                                               │
   │     k_cache: cache.k_buffer,                            │
   │     v_cache: cache.v_buffer,                            │
   │     length: cache.length,                               │
   │     out: attn_out, on: cmdBuf)                          │
   │                                                         │
   │  MetalTileKernels.gemv_fp16(                            │
   │     x: attn_out, w: layer.self_attn.o_proj.weight,      │
   │     out: o, on: cmdBuf)                                 │
   │                                                         │
   │  MetalTileKernels.add(h, o, out: h, on: cmdBuf)         │
   │                                                         │
   │  // -- mlp --                                           │
   │  MetalTileKernels.rms_norm(                             │
   │     x: h, weight: layer.post_attention_layernorm.weight,│
   │     out: x_norm, eps:, on: cmdBuf)                      │
   │                                                         │
   │  MetalTileKernels.gemv_fp16(                            │
   │     x: x_norm, w: layer.mlp.gate_proj.weight,           │
   │     out: gate, on: cmdBuf)                              │
   │  MetalTileKernels.gemv_fp16(                            │
   │     x: x_norm, w: layer.mlp.up_proj.weight,             │
   │     out: up, on: cmdBuf)                                │
   │                                                         │
   │  MetalTileKernels.silu_mul(                             │
   │     a: gate, b: up, out: g, on: cmdBuf)                 │
   │                                                         │
   │  MetalTileKernels.gemv_fp16(                            │
   │     x: g, w: layer.mlp.down_proj.weight,                │
   │     out: d, on: cmdBuf)                                 │
   │                                                         │
   │  MetalTileKernels.add(h, d, out: h, on: cmdBuf)         │
   └────────────────────────┬────────────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────────┐
  │  MetalTileKernels.rms_norm(h, model.norm.weight, …)     │
  │  MetalTileKernels.gemv_fp16(h, lm_head.weight, → logits)│
  └────────────────────────┬────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  cmdBuf.commit()                                        │
  │  cmdBuf.waitUntilCompleted()  // single sync point      │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Sampling.swift (CPU)                                   │
  │   • read logits (small: vocab_size floats)              │
  │   • argmax / top-k / top-p / temperature                │
  │   • → next_token (Int)                                  │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ▼
                       next_token
                  (feed back to next loop iter)
```

**What each `MetalTileKernels.*` call does internally:**

```
   MetalTileKernels.rms_norm(x:, weight:, out:, eps:, on: cmdBuf)
       │
       ▼
   pso = PSOCache.shared.get(
            "rms_norm",
            constants: [n: x.shape.last])
       │
       ▼
   enc = cmdBuf.makeComputeCommandEncoder()
   enc.setComputePipelineState(pso)
   enc.setBuffer(x.mtlBuffer, offset: x.offset, index: 0)
   enc.setBuffer(weight.mtlBuffer, offset: weight.offset, index: 1)
   enc.setBuffer(out.mtlBuffer, offset: out.offset, index: 2)
   enc.setBytes(&eps, length: 4, index: 3)
   enc.dispatchThreadgroups(
        gridFromManifest(rows = x.shape[0]),
        threadsPerThreadgroup: tgFromManifest)
   enc.endEncoding()
```

**Observations:**

- One `MTLCommandBuffer` per token, not per kernel. Most layers'
  worth of dispatches queue up before any GPU↔CPU sync happens.
- KV cache is a pre-allocated `MTLBuffer` that grows by writing into
  the next slice — no reallocation per token.
- All weight tensors are mmap'd from disk at load time; Metal reads
  pages on demand.
- Activation tensors (`h`, `x_norm`, `q`, `k`, etc.) come from a
  `BufferPool` that reuses MTLBuffers across layers (slabs sized for
  the max activation per layer).
- Logits read-back is the only mandatory CPU↔GPU transfer per token —
  and even that is **eliminated** when sampling runs on the GPU
  (default; see "GPU sampling" below). With on-GPU sampling, only the
  chosen token id (4 bytes) crosses CPU↔GPU per token.

---

## 4a. Dispatch modes — three options, evolve as needed

The per-token decode shown above is **Mode 1: Eager**. It's what we
ship in Phase 2 because it's the simplest and most debuggable. As
profiles show CPU encoding cost growing on bigger models, we have two
upgrade paths that the architecture is designed to support without
rewrites.

```
  ─────────────── Mode 1: Eager (default, Phase 2) ────────────────

   per token:
     cmdBuf = queue.makeCommandBuffer()
     for each kernel call (~100s per token):
       enc = cmdBuf.makeComputeCommandEncoder()
       enc.setComputePipelineState(pso)
       enc.setBuffer(...) × N      ← repetitive: same weights
       enc.setBytes(constants, …)     every token
       enc.dispatchThreadgroups(...)
       enc.endEncoding()
     cmdBuf.commit()

   CPU cost: ~10µs × N kernels per token.
   ~800µs for small Llama, ~8ms for Qwen 9B.


  ──────── Mode 2: Argument buffers (Phase 5 if needed) ──────────

   at load time, per layer:
     argEncoder = device.makeArgumentEncoder(...)
     Bind ALL weights once into a per-layer arg buffer:
       q_proj.weight, k_proj.weight, v_proj.weight, o_proj.weight,
       gate.weight, up.weight, down.weight,
       input_norm.weight, post_attn_norm.weight,
       kv_cache_buffer

   per token:
     For each layer, encoder still exists, but only:
       enc.setBuffer(perTokenArgBuf, ...)  ← activations + KV offset
       enc.useResources([weightArgBuf], usage: .read)
       enc.dispatchThreadgroups(...)
     ~3-4 setBuffer calls per layer instead of ~30

   Wins:
   • Massive setBuffer reduction (CPU-bound work)
   • useResources() declares residency once, no per-call validation
   • ~5x reduction in per-kernel encoding cost expected


  ──────── Mode 3: ICB pre-recorded (Phase 5+, last resort) ──────

   at load time:
     icb = device.makeIndirectCommandBuffer(
              descriptor: ..., maxCommandCount: ~1000)
     For each kernel needed in one full forward pass:
       cmd = icb.indirectComputeCommandAtIndex(i)
       cmd.setComputePipelineState(pso)
       cmd.setKernelBuffer(weightArgBuf, ...)
       cmd.setKernelBuffer(perTokenArgBuf, ...)
       cmd.dispatchThreadgroups(...)

   per token:
     Update perTokenArgBuf with current activation pointers + KV len.
     enc.executeCommandsInBuffer(icb, range: 0..<commandCount)
     One call instead of ~800.

   Wins:
   • Collapses ~800 encoder operations into one ICB execute
   • Only CPU work per token: write to one argument buffer
   Caveats — and these are why the mlx-swift-lm experiment was rough:
   • Dynamic KV length must come through arg buffer, not encoded
   • Can't easily skip kernels conditionally per token
   • Debugging is harder (Xcode GPU debugger has limited ICB tools)
   • Memory residency edge cases with arg buffer + ICB combos
```

**Lessons from a prior ICB experiment in mlx-swift-lm**

A previous attempt at ICB + argument buffers in mlx-swift-lm didn't
yield the expected throughput win. Two takeaways inform the FFAI
design:

1. **Per-token state must be ping-ponged (or triple-buffered).** The
   per-token arg buffer (activation pool slots, KV length, RNG state)
   is updated by the CPU *while* the GPU is still reading the
   previous token's arg buffer. Without distinct buffers for token N,
   N+1, N+2, the CPU stalls or the GPU reads stale data. FFAI's
   `.argumentBuffers` and `.icb` modes triple-buffer the per-token
   arg buffer by construction.

2. **Dispatch optimization is a smaller lever than memory bandwidth.**
   The mlx-swift-lm experiment cut command-buffer build time
   significantly but token throughput barely moved — because GPU
   memory bandwidth, not CPU dispatch, was the actual bottleneck on
   the hot path. This is why FFAI's bigger optimization lever is
   **fused kernels** (one activation read, multiple ops happen in
   registers/threadgroup memory) rather than dispatch mode upgrades.

3. **The 4-repo interface change cost was a separate killer.** Adding
   argument buffers to MLX kernels meant changing kernel signatures
   across mlx → mlx-c → mlx-swift → mlx-swift-lm. In FFAI a kernel
   signature change is a Rust edit + cargo rebuild. The complexity
   cost of dispatch mode upgrades drops by an order of magnitude.

**How we keep all three options open from Phase 2:**

1. **Weight MTLBuffers are immutable post-load.** Bound once into arg
   buffers (Mode 2/3), never re-bound. Trivially true since we mmap
   safetensors.
2. **Activations come from a `BufferPool` with stable handles.** Per-token
   arg buffer references handle IDs, not fresh MTLBuffers.
3. **No CPU readback during forward pass.** Sampling on GPU (next
   section) eliminates the one mandatory sync point per token.
4. **Fused kernels for hot paths.** A `rms_norm_qkv` fused kernel reads
   the activation tensor *once* and produces three outputs. This is
   the **memory bandwidth lever** — typically more impactful than
   dispatch optimization on Apple Silicon since GPU memory is the
   actual bottleneck once dispatch is efficient.

`LoadOptions.dispatchMode` selects between modes at load time. Phase
2 supports `.eager` only; Phases 5/6 add `.argumentBuffers` and `.icb`
behind the same Model API.

---

## 4b. GPU sampling — eliminate the per-token sync point

In MLX/mlx-swift-lm, sampling reads the full logits vector to CPU
every token (~60-200KB), runs argmax/top-k/top-p in Swift, and
returns the next token id. That's a mandatory CPU↔GPU sync per
token, plus the actual memory transfer.

FFAI runs sampling on the GPU as a final kernel in the per-token
command buffer:

```
   ┌─────────────────────────────────────────────────────────┐
   │  logits     ─→  sample_kernel(logits, rng_state)        │
   │   (vocab_size × dtype, lives on GPU)                    │
   │                                                         │
   │  greedy:    argmax along vocab axis                     │
   │  temp:      logits *= 1/temp                            │
   │  top-k:     bitonic sort top k, mask others to -inf     │
   │  top-p:     cumulative softmax, threshold mask          │
   │  sample:    inverse CDF sample with rng_state           │
   │                                                         │
   │  output:    next_token (Int, 4 bytes)                   │
   └─────────────────────────────────────────────────────────┘
       │
       │ 4-byte readback (vs 60-200KB)
       ▼
   Swift advances the loop with next_token
```

This is a **Phase 1 deliverable** (alongside the other foundation
kernels), not a Phase 5 optimization. It costs us very little to
implement up front and removes the biggest per-token sync point from
day one.

---

## 4c. GDN / SSM hybrid layer dispatch

Hybrid families (Qwen 3.5 GDN, NemotronH, Jamba, GraniteMoeHybrid,
FalconH1) interleave **attention layers** with **recurrent layers**
(GDN or Mamba 2 SSM) in a fixed schedule. The schedule is data-driven:
NemotronH ships a layer-string like `M*EM-*M…` where each char picks
a mixer; Qwen 3.5 ships an integer `fullAttentionInterval` so attention
fires every N layers and GDN runs in between.

The per-token decode loop branches on layer type at dispatch time —
each layer's mixer protocol selects the right kernel set + cache
type without the outer loop knowing the difference.

```
                 HYBRID LAYER DISPATCH
                 ─────────────────────

  ModelConfig (NemotronH example):
    layer_pattern = "M*EM-*M-*EM-*M-*EM-*M-*EM-*M-*…"
                     │ │  │ │ │ │
                     │ │  │ │ │ └── '*' attention
                     │ │  │ │ └──── '-' MLP residual
                     │ │  │ └────── 'M' Mamba 2
                     │ │  └──────── '-' MLP
                     │ └─────────── '*' attention
                     └───────────── 'M' Mamba 2

  Qwen 3.5 GDN example:
    fullAttentionInterval = 6
    decoder_layers = 60
    layer i mixer = (i % 6 == 0) ? attention : gdn

  At load time, each layer instantiates its mixer protocol:
    LayerMixer = AttentionMixer | GDNMixer | Mamba2Mixer | MLPOnly

  Per-layer cache also picks its type:
    cache[i] = KVCache | AffineQuantizedKVCache | GigaQuantizedKVCache
                                       (attention layers)
             | GDNStateCache            (GDN layers)
             | Mamba2LayerCache         (SSM layers — conv + state)
             | nil                      (MLP-only layers)

  All caches conform to LayerCacheProtocol; attention ones extend
  with KVCacheProtocol; SSM/GDN ones extend with StateReplayCache.


                 PER-TOKEN DECODE LOOP
                 ─────────────────────

  for each layer in model.layers:
    switch layer.mixer {

    case .attention(let attn):
      // Standard attention path — RMSNorm → QKV → RoPE →
      // SDPA (against KV cache) → o_proj → residual
      h = attn.forward(h, cache: cache[i], on: cmdBuf)

    case .gdn(let gdn):
      // Gated DeltaNet — RMSNorm → q/k/v + β/g projections →
      // gated_delta_step (state in fp32) → out_proj → residual
      h = gdn.forward(h, state: cache[i], on: cmdBuf)
      // GDNStateCache.state holds the recurrent matrix S_t

    case .mamba2(let ssm):
      // Mamba 2 mixer — RMSNorm → in_proj →
      // conv1d_causal_step + SiLU → softplus(dt) →
      // ssm_step → D·x skip → SiLU(z) gate → out_proj
      h = ssm.forward(h, state: cache[i], on: cmdBuf)
      // Mamba2LayerCache wraps SSMStateCache + ConvStateCache

    case .mlpOnly:
      // No mixer; pass residual to the MLP block only.
      pass
    }

    // MLP path is shared (or routed through MoE if the family
    // ships an expert layer at this position).
    h = layer.mlpOrMoE.forward(h, on: cmdBuf)


  Key invariant: the outer decode loop is single-cmdbuf per token
  regardless of mixer type. Mixers queue kernels onto the shared
  cmdBuf; only after every layer commits does the GPU sync.
```

Speculative decoding rollback (Phase 8) leans on this dispatch:
GDN + SSM caches expose `record(...)` + `rollback(acceptedPrefix:)`
via `StateReplayCache`. Attention caches truncate the KV slice;
`MLPOnly` is stateless and needs nothing. The verifier walks the
draft, accepts a prefix, and per-layer caches roll back to that
prefix — the loop body above is unchanged.

---

## 4d. Speculative decoding loop

Phase 8 ships several speculative-decoding strategies that share a
common skeleton: a **drafter** proposes K candidate tokens; a
**verifier** runs one forward pass to score all K (batched along the
sequence axis); an **accept/reject** stage decides how many to keep;
caches roll back to the accepted prefix and the loop continues.

```
                 SPECULATIVE DECODING LOOP
                 ─────────────────────────

  state:  prev_token, layer caches (KV + GDN + SSM)
       │
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Drafter — pick one (or stack multiple):                 │
  │                                                          │
  │  • ngram lookup (spec 013)                               │
  │      └─ multi-size hash on recent context → K tokens     │
  │                                                          │
  │  • MTP head / EAGLE-3 (spec 030)                         │
  │      └─ small draft head shares weights w/ verifier      │
  │                                                          │
  │  • PLD+ attention-weighted span (spec 019)               │
  │      └─ pick the span the verifier "wanted to copy"      │
  │                                                          │
  │  • DFlash draft model (spec 015)                         │
  │      └─ small companion model with its own KV state      │
  │                                                          │
  │  Output: draft_tokens[K]                                 │
  └──────────────────────────────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Verifier forward — one cmdBuf, K positions              │
  │                                                          │
  │  Per layer:                                              │
  │    record_caches.record(...)   ◄── tape per-token delta  │
  │                                    on every KV/GDN/SSM   │
  │                                    cache for rollback    │
  │    layer.forward(positions: K, cmdBuf)                   │
  │                                                          │
  │  Tree attention (spec 014): K positions live in a        │
  │  branching tree, not a flat sequence. SDPA uses a        │
  │  tree mask so each branch sees only its ancestors.       │
  │                                                          │
  │  Output: logits[K, vocab]                                │
  └──────────────────────────────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Accept / reject — pick one:                             │
  │                                                          │
  │  • Greedy: accept while draft[i] == argmax(logits[i])    │
  │                                                          │
  │  • Leviathan (spec 023): non-greedy accept-reject —      │
  │      sample u ~ U[0,1]; accept iff                       │
  │        u ≤ min(1, p_target(draft[i]) / p_draft(draft[i]))│
  │      reject + resample at the first failure.             │
  │                                                          │
  │  • Tree-aware: walk the branch with the longest accept   │
  │      prefix; rejected branches' caches roll back.        │
  │                                                          │
  │  Output: accepted_prefix_len (m where 0 ≤ m ≤ K),        │
  │           plus one bonus token sampled from the          │
  │           failing position's residual distribution.      │
  └──────────────────────────────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Cache rollback                                          │
  │                                                          │
  │  for each layer cache:                                   │
  │    switch cache {                                        │
  │      case KVCache:               truncate to prefix_len  │
  │      case GigaQuantizedKVCache:  truncate compressed slab│
  │      case GDNStateCache:         state_replay(prefix_len)│
  │      case Mamba2LayerCache:      ssm_replay(prefix_len)  │
  │      case MLPOnly:               (stateless)             │
  │    }                                                     │
  │                                                          │
  │  All replay kernels re-fold the recorded delta tape      │
  │  back to the accepted boundary; KV caches just adjust    │
  │  their length cursor.                                    │
  └──────────────────────────────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Emit accepted_prefix + bonus_token to the consumer      │
  │  AsyncStream. Update prev_token = bonus_token. Loop.     │
  └──────────────────────────────────────────────────────────┘


  Batched verify (spec 8.4 — `generateBatched` integration):
  B requests verify K candidates each in one cmdBuf.
  K·B positions go through SDPA together; per-request caches
  (BatchedKVCache + BatchedHybridCache) roll back independently.
```

The throughput win comes from amortising one verifier forward over
multiple accepted tokens. Worst case (everything rejects) costs one
forward + the drafting overhead — auto-disengage logic (issue #153)
kills the speculation path when the accept rate drops below a
floor.

---

## 5. Where this differs from MLX

| Concern | MLX path | FFAI path |
|---|---|---|
| Kernel source | MSL string passed to `metal_kernel()` | Rust `#[kernel]` → MSL at build time |
| Compilation | JIT on first call (~1-2ms) | Pre-compiled into `.metallib`, only PSO specialization at runtime (~100-500µs once per fn-const combo) |
| Tensor type | `MLXArray` (lazy graph node, hides MTLBuffer) | `Tensor` (direct MTLBuffer + offset + shape) |
| Dispatch timing | Lazy: graph traversal + planning + encode + submit on `eval()` | Eager: Swift code is the dispatch order, no graph traversal step |
| Dispatch modes | One mode (lazy graph) | Three modes (Eager / Argument buffers / ICB) selectable at load time |
| Op fusion | Automatic via lazy graph compiler | Manual via fused kernels in metaltile (compile-time, profileable) |
| Sampling | CPU readback per token (~60-200KB) | On-GPU sampling kernel, 4-byte readback |
| Custom kernel cost | ~55-95µs per call (Swift→C→C++→Metal marshaling) | ~5-10µs per call eager / sub-µs amortized in ICB mode |
| Kernel iteration | Edit MLX C++ → mlx-c → mlx-swift → mlx-swift-lm | Edit Rust `#[kernel]` → rebuild Swift package |
| Model download | swift-transformers via mlx-swift-lm wrappers | swift-huggingface direct, bypass MLX entirely |

---

## 6. Layered dependency graph

```
  ┌─────────────────────────────────────────────────────────┐
  │  app / FFAICLI                                          │
  └────────────────────┬────────────────────────────────────┘
                       │ calls
  ┌────────────────────▼────────────────────────────────────┐
  │  FFAI                                                   │
  │                                                         │
  │  Core text path                                         │
  │   Tensor • Module • Linear • Embedding • RMSNorm •      │
  │   SafeTensors • ModelDownloader • TokenizerLoader •     │
  │   Sampling • Generate • ChatTemplate                    │
  │                                                         │
  │  Caches                                                 │
  │   KVCache • AffineQuantizedKVCache •                    │
  │   GigaQuantizedKVCache • SSMStateCache •                │
  │   GDNStateCache • Mamba2LayerCache •                    │
  │   BatchedKVCache • BatchedHybridCache •                 │
  │   PrefixKVCache • StateReplayCache                      │
  │                                                         │
  │  Modality encoders (Capability-gated)                   │
  │   VisionEncoder • ImagePreprocessing                    │
  │   AudioEncoder  • AudioPreprocessing                    │
  │                                                         │
  │  Speculative decoding                                   │
  │   NGramSpeculativeTokenIterator •                       │
  │   MTPSelfSpeculativeTokenIterator •                     │
  │   AssistantDraftRegistry • MirrorSpeculativeLoop        │
  │                                                         │
  │  Models/{Llama, Qwen3, Qwen35, Mamba2, GPTOSS,          │
  │          Gemma3, Gemma4, NemotronH, Jamba,              │
  │          GraniteMoeHybrid, FalconH1, Mistral, Phi,      │
  │          Whisper, Kokoro/Bark, QwenOmni, …}             │
  └────────┬─────────────────────┬──────────────────────────┘
           │ kernels             │ HF + tokenizers
           ▼                     ▼
  ┌────────────────────┐   ┌──────────────────────────────┐
  │  MetalTileSwift    │   │  swift-huggingface           │
  │  (in-repo)         │   │  swift-transformers          │
  │                    │   │  (external SPM dependencies) │
  │  • Library loader  │   └──────────────────────────────┘
  │  • PSO cache       │
  │  • Generated wraps │
  │  • kernels.metallib│
  └────────┬───────────┘
           │ talks to
           ▼
  ┌────────────────────┐
  │  Metal framework   │   ← Apple system framework, only
  │  (MTLDevice etc.)  │     external runtime dependency
  └────────────────────┘

  Build-time only (not in runtime graph):
  ┌──────────────────────────────────────────────────────────┐
  │  metaltile (Rust workspace, sibling repo)                │
  │  Generates kernels.metallib + manifest + Swift wrappers  │
  │  via the `tile build --emit all --out <dir>` CLI         │
  │  command (metaltile-cli).                                │
  └──────────────────────────────────────────────────────────┘
```

**Runtime dependencies:** Metal framework, swift-huggingface,
swift-transformers. That's it.

**Build-time dependencies:** add Xcode (for `xcrun metal`), a Rust
toolchain (for `tile`), and the metaltile sibling repo checkout
until the `tile` binary ships via Homebrew.
