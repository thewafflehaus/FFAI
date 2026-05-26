# Testing

FFAI tests in three layers — kernel-wrapper correctness, Swift unit tests, and per-model integration tests — using **Swift Testing** (`@Suite` / `@Test` / `#expect`), not XCTest. CI gates on ≥ 80 % line coverage of the Swift surface plus a green integration sweep.

## Running tests

**Go through `make` — do not run bare `swift test`.** Each `ModelIntegrationTests` suite downloads a multi-GB checkpoint; an unconstrained parallel run loads several at once and OOMs the box (and can pin the GPU). The `make` targets cap parallelism correctly.

```bash
make test-unit          # FFAITests + MetalTileSwiftTests — fast, parallel-safe
make test-integration   # ModelIntegrationTests — serialized (--num-workers 1)
make test               # both in sequence — the full local CI gate
make coverage           # unit-suite line coverage (≥ 80 %)
make test-stress        # production cap, uncapped parallelism — run after dispatch changes
```

`make test` runs `make regenerate-kernels` first, so you never test against stale kernels.

### Filtering to one suite or test

To iterate on a single suite, run `swift test` directly **but keep the memory cap** — `--parallel --num-workers 1` loads one model at a time:

```bash
swift test --parallel --num-workers 1 --filter Qwen3TextIntegrationTests
swift test --parallel --num-workers 1 --filter ModelKVCacheMatrixIntegrationTests
swift test --filter OpsTests          # unit suite — fast, no cap needed
```

`--filter` matches a regex against suite + test names.

## Test layout

```
Tests/
  MetalTileSwiftTests/      One file per kernel wrapper — numerical
                            correctness vs a CPU reference across
                            fp32 / fp16 / bf16. Plus
                            KernelManifestSmokeTests.
  FFAITests/                Mirrors Sources/FFAI/ — every source file
                            has a sibling test. Audio/, Benchmark/,
                            Generation/, KVCache/, Loader/, Models/,
                            Ops/, Stats/, Telemetry/, Vision/ are the
                            top-level groups.
  ModelIntegrationTests/    Per-family end-to-end checkpoint runs,
                            grouped by modality (Text/, Vision/,
                            Audio/Omni/, Audio/STT/, Audio/STS/,
                            Audio/TTS/, Audio/VAD/), plus the
                            cross-cutting suites (see below).
  Helpers/                  CommonTestHelpers (`loadModel`,
                            `expectCoherentOutput`, `ModelLoadLock`),
                            TextTestHelpers, VisionTestHelpers,
                            AudioTestHelpers, RunAndWait.
  Resources/                Test inputs (dog.jpeg, cat.mp4, audio
                            clips, … shared via the helpers).
```

There are **no golden fixtures**. Cross-implementation token-parity vs mlx-lm proved to be a measure of rounding-mode alignment, not correctness — it was dropped. Numerical correctness now comes from the metaltile-side per-kernel GPU-correctness tests (compared to a naive CPU oracle); the FFAI integration tests assert that the model *pipeline* produces coherent text.

## Integration testing

Every model family has a `Tests/ModelIntegrationTests/<Family>IntegrationTests.swift` that downloads the smallest published checkpoint from mlx-community, greedy-decodes, and asserts `expectCoherentOutput(...)` (token-count floor, no degenerate repeat run, minimum token diversity). A checkpoint that can't be fetched (offline, gated repo) prints a skip line and **passes** — integration tests never hard-fail on a missing download.

Cross-cutting suites:

- `ModelKVCacheMatrixIntegrationTests` — the model family × weight bitwidth × KV-cache scheme cross-product.
- `Quantized{3,4,5,6,8}bitIntegrationTests` — the weight-bitwidth ladder.
- `ModelDeterminismIntegrationTests` — temp = 0 greedy decode is stable across runs.
- `ModelInspectionIntegrationTests` — `ffai inspect` end-to-end against a representative model from each family (verifies every family wired its `InspectTap` hooks).
- `SlidingWindowIntegrationTests` — sliding-window KV eviction composes correctly across cache schemes.

### Not every model runs by default — env-gated tests

The largest checkpoints are too heavy (or too slow) for the routine gate, so they are gated behind environment variables. With the var unset the test skips (and passes); set it to opt in:

| Env var | Unlocks |
|---|---|
| `FFAI_BUILD_MACHINE` | The heavy generation checks — `GPTOSSIntegrationTests` (~20B MoE), the Gemma 4 31B / 26B-A4B decode in `Gemma4TextIntegrationTests` (load + shape checks still run unconditionally), and every non-smallest cell of `ModelKVCacheMatrixIntegrationTests`. Intended for a dedicated build machine. |
| `FFAI_MATRIX_FAMILY=<family>` | Restricts `ModelKVCacheMatrixIntegrationTests` to one family's row (e.g. `FFAI_MATRIX_FAMILY=Gemma4`) — fast targeted re-runs. |

```bash
# Run the whole matrix incl. env-gated cells, on a build machine:
FFAI_BUILD_MACHINE=1 swift test --parallel --num-workers 1 \
    --filter ModelKVCacheMatrixIntegrationTests

# Re-run just the Llama row of the matrix:
FFAI_MATRIX_FAMILY=Llama swift test --parallel --num-workers 1 \
    --filter ModelKVCacheMatrixIntegrationTests
```

The default `make test-integration` runs only the always-on cells: the smallest checkpoint per family.

## Writing a test

```swift
import Testing
@testable import FFAI

@Suite("Ops.add")
struct OpsAddTests {
    @Test("elementwise add matches a CPU reference")
    func addMatchesCPU() {
        let a = Tensor.empty(shape: [4], dtype: .f32)
        a.copyIn(from: [1, 2, 3, 4])
        // … dispatch, then:
        #expect(out.toArray(as: Float.self) == [2, 4, 6, 8])
    }
}
```

A model integration test loads through `ModelLoadLock.shared` (serializes the multi-GB load across suites) and asserts coherence. The canonical text-model pattern uses the `loadModel(_:)` helper from [`Tests/Helpers/CommonTestHelpers.swift`](../../Tests/Helpers/CommonTestHelpers.swift) — it wraps `ModelLoadLock.shared.loadSerially { … }` and fails the test on load failure instead of silently skipping:

```swift
import Testing
@testable import FFAI

@Suite("Qwen3 Text Integration", .serialized)
struct Qwen3TextIntegrationTests {
    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        let m = try await loadModel("mlx-community/Qwen3-1.7B-4bit")
        let r = try await m.generate(
            prompt: "Once upon a time, in a quiet village",
            parameters: GenerationParameters(maxTokens: 200, temperature: 0))
        expectCoherentOutput(r.generatedTokens, label: "Qwen3-1.7B")
    }
}
```

Audio + VL families typically need a typed cast after loading. The pattern is to resolve the snapshot through `ModelLocator()` first, then build the typed model directly so the test can call its modality-specific API (codec decode, vision-tower preprocess, …):

```swift
@Suite("FishSpeech Integration", .serialized)
struct FishSpeechIntegrationTests {
    private static let repoId = "fishaudio/openaudio-s1-mini"

    @Test("synthesises a coherent waveform")
    func synthesise() async throws {
        let dir = try await ModelLoadLock.shared.loadSerially {
            try await ModelLocator().resolve(idOrPath: Self.repoId)
        }
        let model = try await Model.load(dir.path)
        // … call the typed audio API on `model.audio`
    }
}
```

Both patterns serialize through the same lock so the multi-GB downloads / GPU footprint stay capped regardless of how many integration suites the runner picks up.

A non-trivial kernel lands with a paired metaltile GPU-correctness test in the **same commit** (see metaltile `docs/testing.md`).

## CI

CI runs on Apple Silicon: the unit gate, then the serialized integration gate (matching `make test-integration`), and uploads the coverage report — a PR that drops Swift-surface coverage below the threshold fails.

## What we don't test

- **Property / fuzz testing** — revisit post-v0.2.
- **GPU mocking** — all tests run real Metal dispatches.
- **Cross-implementation token parity** — dropped (see above).
- **Multi-GPU / Linux / CUDA** — different project.

## See also

- [Developing](developing.md) — the `make` workflow, kernel regen.
- [Adding a model](adding-a-model.md) — which tests a new family adds.
- [Performance](../performance.md) — `Tests/PerfTests/` thresholds.
