# Capabilities & Lifecycle

FFAI models declare what they *can* do via `Capability`, the user
picks what to *enable* at load time via `LoadOptions`, and the model
exposes its load progress + hot capability changes via an
`AsyncStream<ModelLifecycleEvent>`.

The infrastructure is in place from Phase 2; the first multi-modal
model that exercises it end-to-end (vision encoder, hot
`enable(.visionIn)`, etc.) lands in Phase 6.

## The `Capability` enum

```swift
public enum Capability: String, Sendable, Hashable, CaseIterable, Codable {
    case textIn
    case textOut
    case visionIn
    case audioIn
    case audioOut
    case toolCalling
}
```

| Capability | Today | When |
|---|---|---|
| `.textIn` / `.textOut` | ✅ Always on for LLMs. | Phase 2 |
| `.visionIn` | Declared on family files but no family supports it yet. | Phase 6 (Qwen 2.5/3.5-VL) |
| `.audioIn` / `.audioOut` | Not declared by any family. | Phase 8+ |
| `.toolCalling` | Not declared by any family. | Phase 8+ |

Convenience sets:

```swift
Capability.textOnly       // [.textIn, .textOut]
Capability.textWithTools  // [.textIn, .textOut, .toolCalling]
```

## What each family declares

| Family | `availableCapabilities` |
|---|---|
| `Llama.LlamaDense` | `[.textIn, .textOut]` |
| `Qwen3.Qwen3Dense` | `[.textIn, .textOut]` |
| `Mamba2.Mamba2Dense` | `[.textIn, .textOut]` |
| `FalconH1.FalconH1Hybrid` | `[.textIn, .textOut]` |
| `NemotronH.NemotronHHybrid` | `[.textIn, .textOut]` |
| `NemotronLabsDiffusion.NemotronLabsDiffusionDense` | `[.textIn, .textOut]` |
| `GraniteMoeHybrid.GraniteMoeHybridHybrid` | `[.textIn, .textOut]` |
| `Jamba.JambaHybrid` | `[.textIn, .textOut]` |
| `Qwen35.Qwen35Hybrid` | `[.textIn, .textOut]` |
| `Gemma4.Gemma4Dense` / `Gemma4E` / `Gemma4MoE` | `[.textIn, .textOut]` |
| `GPTOSS.GPTOSSMoEVariant` | `[.textIn, .textOut]` |

When a family adds a capability (e.g. `Qwen35VL` adds `.visionIn`),
the family file declares it and the loader allocates the
corresponding subnet only if the user opts in.

## `LoadOptions`

```swift
let model = try await Model.load(
    "unsloth/Llama-3.2-1B",
    options: LoadOptions(
        capabilities: [.textIn, .textOut],
        kvCache: .raw,
        dispatchMode: .eager,
        prewarm: true,
        lazyCapabilities: true,
        revision: "main"
    )
)
```

| Field | Default | Notes |
|---|---|---|
| `capabilities` | `Capability.textOnly` | What to load. `textIn` + `textOut` are always implicitly on. Disabled modalities skip weight allocation. |
| `kvCache` | `.raw` | Cache compression scheme — see [kv-cache.md](kv-cache.md). |
| `dispatchMode` | `.eager` | Standard `MTLComputeCommandEncoder` per kernel. `.argumentBuffers` / `.icb` deferred. |
| `prewarm` | `true` | Run one no-op forward to compile PSOs before the first user-visible decode. |
| `lazyCapabilities` | `true` | Allow runtime `enable(_:)` / `disable(_:)` after load. Phase 6 wires this end-to-end. |
| `revision` | `"main"` | HF branch / tag / commit. |

## Inspecting a loaded model

```swift
let model = try await Model.load("mlx-community/Qwen3-4B-4bit")

print(model.availableCapabilities)   // what the family supports
print(model.enabledCapabilities)     // what you opted into
print(model.config.modelType)        // "qwen3"
print(model.modelDirectory)          // resolved local snapshot
```

If you ask for a capability the family doesn't expose, the loader
throws `ModelError.capabilityNotAvailable(.visionIn)`.

## Lifecycle states

```
ModelLifecycleState:
  idle → downloading(Progress) → loading(LoadProgress)
       → loaded → ready
       (or failed(Error) at any stage)
```

`Model.events` is an `AsyncStream<ModelLifecycleEvent>` that emits
each transition. The stream is multi-consumer-safe and finishes when
the `Model` is deinitialized.

```swift
let model = try await Model.load("unsloth/Llama-3.2-1B")

Task {
    for await event in model.events {
        switch event.state {
        case .downloading(let progress): print("downloading \(progress.fractionCompleted)")
        case .loading(let p):            print("loading \(p)")
        case .loaded:                    print("weights resident")
        case .ready:                     print("ready to generate")
        case .failed(let err):           print("failed: \(err)")
        default:                         break
        }
    }
}

print(model.currentState)  // sync snapshot — typically .ready by the time load() returns
```

`currentState` is a thread-safe snapshot of the latest emitted event.
The stream is the source of truth for fine-grained progress.

## Hot capability changes (Phase 6)

The API surface is in place from Phase 2; the implementation lands
alongside the first VL family:

```swift
// Phase 6:
try await model.enable(.visionIn)    // mmaps vision weights, builds encoder, prewarms
// ... use the model with images ...
try await model.disable(.visionIn)   // releases MTLBuffers, frees GPU residency
```

Each call emits per-capability lifecycle events through the same
`events` stream. If `lazyCapabilities = false` was passed at load
time, both calls throw — capabilities are then frozen at the load-time
set.

## See also

- [Quick start](quickstart.md) — the basic `Model.load` + `generate`
  flow.
- [Models](models.md) — what each family declares for `availableCapabilities`.
- [Architecture](architecture.md) — where capability-driven loading
  sits in the load sequence.
