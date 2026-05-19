# Adding a Model

How to port a new architecture into FFAI. Both in-tree families
(`Llama`, `Qwen3`) were added with this flow; copy whichever is
structurally closest to your target.

## Decide if it's a new family or a new variant

A **family** is a major architectural lineage. A **variant** is a
shape inside a family.

- **New family.** A different family file. Examples: Mistral, Phi,
  Gemma, Mamba, GPT-OSS — none of these share Llama's exact
  attention shape. New file in `Sources/FFAI/Models/`, registered in
  `ModelRegistry.dispatchAndLoad`.
- **New variant.** Same family file, new struct conforming to the
  family's `Variant` protocol. Examples: `Qwen35Dense` alongside
  `Qwen3Dense`, `Qwen35MoE`, `Qwen35VL` — they go in
  `Models/Qwen3.swift` and dispatch in `Qwen3.variant(for:)`.

The convention is **one file per family, not per variant**. A
family file with five variant structs is fine; five files for one
family is not.

## Step 1 — read the reference

Pick a reference implementation that's close to your target. Good
sources:

- `mlx-swift-lm/Libraries/MLXLLM/Models/<Family>.swift` — the
  structural template we work from.
- The model's HuggingFace `modeling_<family>.py`.
- llama.cpp's `convert-*.py` for tensor-name mapping.

What you want from the reference:

- The forward-pass shape (RMSNorm → QKV → RoPE → ...).
- Any structural quirks (Qwen 3's per-head q_norm/k_norm, GPT-OSS'
  sliding-window layers, Gemma's per-layer rope scaling, etc.).
- The expected tensor-key naming in the safetensors files.
- Which `config.json` fields drive the shape.

## Step 2 — confirm the kernel coverage

Open `Sources/FFAI/Ops.swift` and verify every op the forward pass
needs exists. Most modern LLMs reuse the existing kernel set:

- `gather` (embedding lookup)
- `rms_norm` + `multiRowRMSNorm`
- `gemv` (bf16/fp16 dense matvec)
- `dequant_gemv_<bits>` (mlx-format quantized matvec, 3/4/5/6/8-bit)
- `rope` (Llama 3 scaled + non-scaled)
- `silu`, `mul`, `add`
- `sdpa_decode`
- `kv_cache_update`
- `argmax`

If your model needs something new (attention sinks, fused MoE,
sliding-window mask, GDN step, SSM scan), that's a metaltile-side
addition first — see [developing.md § Writing a new
kernel](developing.md#writing-a-new-kernel).

## Step 3 — write the family file

Copy `Sources/FFAI/Models/Llama.swift` as a template. Structure:

```swift
public enum <Family> {
    public static let modelTypes: Set<String> = ["<type>"]
    public static let architectures: Set<String> = ["<Arch>ForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any <Family>Variant.Type {
        // Inspect config.json fields and pick the right variant.
        return <Family>Dense.self
    }
}

public protocol <Family>Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> <Family>Model
}

public struct <Family>Dense: <Family>Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]
    public static let defaultGenerationParameters = GenerationParameters(...)  // see Step 4

    public static func loadModel(...) throws -> <Family>Model { ... }
}

public final class <Family>Model: LanguageModel, @unchecked Sendable {
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] { ... }
    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol], device: Device) -> Tensor { ... }
    public func forwardSample(tokenId: Int, position: Int,
                              caches: [any LayerCacheProtocol], device: Device) -> Int { ... }
}
```

Key invariants the implementation must preserve:

1. **One `MTLCommandBuffer` per token.** Open it at the start of
   `forward` / `forwardSample`, commit + wait at the end. No
   mid-token sync.
2. **No CPU readback inside the forward pass.** `forwardSample`
   returns a sampled token id; `forward` returns logits but does the
   readback at the end, not mid-layer.
3. **Per-tensor MTLBuffers, allocated once at load.** Weights are
   immutable. Activations come from `BufferPool`.
4. **Capability-gated loading.** If the family supports
   `.visionIn` etc., skip those weights when the user didn't enable
   the capability.

## Step 4 — declare family defaults

Each variant struct carries a static
`defaultGenerationParameters: GenerationParameters` that captures the
sampling + length values that family ships with. The `Model`
instance threads this through as `model.defaultGenerationParameters`,
and `Model.generate(prompt:parameters:)` falls back to it when the
caller passes `nil`. This is what the user gets when they do
`ffai --model <repo>` with no sampling flags.

Pull the values from the reference implementation. mlx-swift-lm's
`Libraries/MLXLLM/Models/<Family>.swift` declares
`defaultPrefillStepSize`; per-checkpoint sampling defaults live in
`LLMRegistry`. Pull representative-checkpoint values for each
variant. Reasonable starting points if the reference is silent:

- `temperature: 0.6`, `topP: 1.0`, `topK: 0` — matches mlx-swift-lm's
  baseline `GenerationParameters.init` for text-only LLMs.
- `prefillStepSize: 1024` — dense attention models. Bump to `4096`
  for pure-attention models with long contexts (Gemma 4, Mistral) per
  mlx-swift-lm's per-family override pattern.
- `maxTokens: 256` — sane default; users override per call.

Example:

```swift
public struct <Family>Dense: <Family>Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Defaults pulled from <reference> for the <family> dense
    /// checkpoints. Future variants (<Family>MoE, <Family>VL, …)
    /// declare their own when they land.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 1024,
        temperature: 0.6,
        topP: 0.95,
        topK: 20,
        repetitionPenalty: 1.0
    )

    public static func loadModel(...) throws -> <Family>Model { ... }
}
```

Sampling fields are no-ops on the greedy decode path until Phase 5 —
declare them anyway so per-family defaults don't churn when GPU
sampling kernels land. See [generation-parameters.md](../generation-parameters.md)
for the full field table.

## Step 5 — register the family

In `Sources/FFAI/Model.swift`, `ModelRegistry.dispatchAndLoad`:

```swift
if let arch = config.architecture, <Family>.architectures.contains(arch) {
    return try load<Family>(config: config, weights: weights,
                            options: options, device: device)
}
if let mt = config.modelType, <Family>.modelTypes.contains(mt) {
    return try load<Family>(config: config, weights: weights,
                            options: options, device: device)
}
```

And a `load<Family>` helper that delegates to the variant's
`loadModel` and returns a `ModelRegistry.Loaded` carrying both the
engine and the variant's `defaultGenerationParameters`:

```swift
public static func load<Family>(...) throws -> Loaded {
    let variant = try <Family>.variant(for: config)
    let engine = try variant.loadModel(...)
    return Loaded(engine: engine,
                  defaultGenerationParameters: variant.defaultGenerationParameters)
}
```

## Step 6 — capture golden fixtures

Golden fixtures are how we pin correctness without a Python
dependency in CI. The capture script runs the new model through our
reference — `mlx-lm` for text-only families, `mlx-vlm` for
vision-language families — and dumps tokens + per-layer activations
to disk; tests then load those files and compare against our forward
pass within tolerance. One commit pins the reference; every
subsequent `swift test` is fully reproducible on a stock Apple Silicon
runner with no Python in the loop.

`mlx-vlm` installs `mlx-lm` as a transitive dependency, so a single
`pip install mlx-vlm` gets you both backends. The capture script
picks the right one per model based on whether the config declares a
vision encoder.

```bash
pip install mlx-vlm        # also installs mlx-lm
python Tools/capture-fixtures.py \
    --model <hf-repo-id> \
    --output Tests/Fixtures/<family>/ \
    --prompts "Once upon a time,The capital of France is" \
    --max-tokens 16
```

The script invokes `mlx_lm.generate` (or `mlx_vlm.generate` for VLMs)
and dumps:

- `metadata.json` — mlx-lm version, capture date, config hash
- `tokens-<prompt-hash>.json` — token sequences for each prompt
- `activations-layer-0.npy` — first-layer activations (optional, for
  forward-pass tests)

Commit the fixtures with the model code.

## Step 7 — wire inspect hooks

`ffai inspect` is the first command every dev reaches for when a
new model produces broken output. Every family file calls the
shared `InspectTap` utility (`Sources/FFAI/Inspect/InspectTap.swift`)
at layer boundaries inside `<Family>Model.forward(...)` so the
`--layer-trace` diagnostic surface works uniformly across families.

The pattern, lifted from `LlamaModel.forward` (the canonical
reference — `Sources/FFAI/Models/Llama.swift`):

```swift
public func forward(tokenId: Int, position: Int,
                    caches: [any LayerCacheProtocol],
                    on cmd: MTLCommandBuffer, device: Device) -> Tensor {
    // 1. Pick up the env-driven tap state. Cached at the first
    //    call site so subsequent forwards pay a static-load cost
    //    only — no per-token env-dict allocation. No-op when
    //    FFAI_INSPECT isn't set.
    let tap = InspectTap.fromEnvironment

    // 2. When taps are active, route work onto a *private* cmdbuf
    //    so per-op commit+wait sync points don't double-commit
    //    the caller's cmd. `makeWorkCmd` returns the caller's
    //    cmd unchanged in production mode (single branch, no
    //    allocation).
    var workCmd = tap.makeWorkCmd(from: cmd, device: device)

    // 3. Embed lookup — tap fires at the residual stream that
    //    feeds layer 0. `dumpLayerBoundary` returns the cmdbuf
    //    the caller should continue queueing on; production
    //    mode collapses to `workCmd = workCmd` and the optimizer
    //    folds the call out.
    var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
    workCmd = tap.dumpLayerBoundary(h, label: "embed", layer: -1,
                                    cmd: workCmd, device: device)

    // 4. Per-layer forward — tap at the OUTPUT of each layer.
    //    The first layer's input is the embed dump above; every
    //    subsequent layer's input is the prior layer's output,
    //    so one dump per boundary is enough.
    for (i, layer) in layers.enumerated() {
        h = layer.forward(h, position: position,
                          cache: caches[i] as! any KVCacheProtocol,
                          cmd: workCmd, device: device)
        workCmd = tap.dumpLayerBoundary(h, label: "layer_out", layer: i,
                                        cmd: workCmd, device: device)
    }

    // 5. Final norm + lm head — tap both.
    let normed = finalNorm(h, on: workCmd)
    workCmd = tap.dumpLayerBoundary(normed, label: "final_norm", layer: -1,
                                    cmd: workCmd, device: device)
    let logits = lmHead(normed, on: workCmd)
    workCmd = tap.dumpLayerBoundary(logits, label: "logits", layer: -1,
                                    cmd: workCmd, device: device)

    // 6. When taps are active, flush the private cmdbuf — the
    //    caller's cmd has no work queued so their commit() is a
    //    fast no-op. In production mode this branch is skipped.
    if tap.active {
        workCmd.commit()
        workCmd.waitUntilCompleted()
    }
    return logits
}
```

The `<Family>Layer.forward(...)` stays clean — no taps inside the
layer's hot path. That's deliberate: layer-boundary taps localise
*which* layer is failing in two `ffai inspect --layer-trace` runs.
For inside-layer triage (which op in layer N produced the NaN),
drop temporary fine-grained `tap.dumpLayerBoundary(...)` calls
between ops as you debug, then remove them before merging. See
`papers/gemma3-coherence-investigation-2026-05-19.md` for the
canonical example — a single GELU NaN inside layer 1's MLP that
the standardized boundary tap localised in seconds once `ffai
inspect --layer-trace --trace-layers 0,1,2,3,4` was the first
thing the dev ran.

**Required:** every new family file MUST follow this pattern. The
`InspectSmokeTests` integration test (Tests/ModelTests/) asserts
the inspect path runs end-to-end against a representative model
from each family; that test will fail loudly if a family file
forgets to wire `InspectTap` calls.

## Step 8 — add tests

`Tests/ModelTests/<Family>/` gets at minimum:

- **`<Family>ForwardTests.swift`** — one prompt, run the forward
  pass once, compare the per-layer activations against the
  captured golden values within tolerance (`1e-3` for bf16,
  `1e-4` for fp16, `1e-2` for 4-bit quant).
- **`<Family>GenerateTests.swift`** — one prompt with greedy
  argmax, generate N tokens, compare exact token sequence against
  the golden capture.

Determinism check: both tests should be exactly reproducible run-to-
run on the same machine. Floating-point drift is fine within the
forward-pass tolerance; greedy decode must be bit-exact (the same
argmax produces the same token).

## Step 9 — wire into CI

`make test` will pick up new tests automatically. Update
[`documentation/models.md`](../models.md) with the new family + the
checkpoints you exercised.

## Common gotchas

- **Tensor naming.** mlx-format checkpoints sometimes rename layers
  (`model.layers` vs `transformer.h`, `q_proj` vs `attn.q.weight`).
  Read the safetensors header (`SafeTensorsBundle.keys`) and write
  the mapping in your family's loader.
- **RoPE scaling.** Llama 3 uses `rope_type: "llama3"` with `factor`,
  `low_freq_factor`, `high_freq_factor`, `original_max_position` in
  `config.rope_scaling`. Other families have other schemes
  (`linear`, `dynamic`, `su`, etc.) — check the config before
  defaulting to plain RoPE.
- **GQA head count.** `num_key_value_heads` may be missing from
  `config.json`; default to `num_attention_heads` (MHA).
- **Tied embeddings.** `tie_word_embeddings: true` means the LM head
  reuses the embedding weight transposed. Many small models tie
  (Llama 3.2 1B, Qwen 3 0.6B); larger ones often don't.
- **Bias terms on `q_proj` / `k_proj` / `v_proj`.** Llama is
  bias-less. Qwen 2 has biases. Qwen 3 dropped them again. Check
  `config.attention_bias`.

## See also

- [Developing](developing.md) — repo layout, `make` workflow, kernel
  regeneration.
- [Testing](testing.md) — golden fixtures, coverage targets.
- [Architecture](../architecture.md) — where family files sit in the
  stack, the per-token dispatch loop they implement.
- [`planning/plan.md`](../../planning/plan.md) — what's in / out of
  scope per phase.
