# Testing

FFAI tests in three layers — kernel-wrapper correctness, Swift unit
tests, and per-model integration tests — using **Swift Testing**
(`@Suite` / `@Test` / `#expect`), not XCTest. CI gates on ≥ 80 % line
coverage of the Swift surface plus a green integration sweep.

## Running tests

**Go through `make` — do not run bare `swift test`.** Each
`ModelIntegrationTests` suite downloads a multi-GB checkpoint; an unconstrained
parallel run loads several at once and OOMs the box (and can pin the
GPU). The `make` targets cap parallelism correctly.

```bash
make test-unit          # FFAITests + MetalTileSwiftTests — fast, parallel-safe
make test-integration   # ModelIntegrationTests — serialized (--num-workers 1)
make test               # both in sequence — the full local CI gate
make coverage           # unit-suite line coverage (≥ 80 %)
make test-stress        # production cap, uncapped parallelism — run after dispatch changes
```

`make test` runs `make regenerate-kernels` first, so you never test
against stale kernels.

### Filtering to one suite or test

To iterate on a single suite, run `swift test` directly **but keep
the memory cap** — `--parallel --num-workers 1` loads one model at a
time:

```bash
swift test --parallel --num-workers 1 --filter Qwen3IntegrationTests
swift test --parallel --num-workers 1 --filter ModelKVCacheMatrixIntegrationTests
swift test --filter OpsTests          # unit suite — fast, no cap needed
```

`--filter` matches a regex against suite + test names.

## Test layout

```
Tests/
  MetalTileSwiftTests/   One file per kernel wrapper — numerical
                         correctness vs a CPU reference across
                         fp32 / fp16 / bf16.
  FFAITests/             Tensor, Module, Linear, BufferPool, the
                         Ops* / *StateCache / KVCache / Layers /
                         Sampling / Capability / ModelConfig units.
  ModelIntegrationTests/            Flat files — one <Family>IntegrationTests
                         per model family, plus the cross-cutting
                         suites (see below).
```

There are **no golden fixtures**. Cross-implementation token-parity
vs mlx-lm proved to be a measure of rounding-mode alignment, not
correctness — it was dropped. Numerical correctness now comes from
the metaltile-side per-kernel GPU-correctness tests (compared to a
naive CPU oracle); the FFAI integration tests assert that the model
*pipeline* produces coherent text.

## Integration testing

Every model family has a `Tests/ModelIntegrationTests/<Family>IntegrationTests.swift`
that downloads the smallest published checkpoint from mlx-community,
greedy-decodes, and asserts `expectCoherentOutput(...)` (token-count
floor, no degenerate repeat run, minimum token diversity). A
checkpoint that can't be fetched (offline, gated repo) prints a skip
line and **passes** — integration tests never hard-fail on a missing
download.

Cross-cutting suites:

- `ModelKVCacheMatrixIntegrationTests` — the model family × weight
  bitwidth × KV-cache scheme cross-product.
- `Quantized{3,4,5,6,8}bitIntegrationTests` — the weight-bitwidth
  ladder.
- `DeterminismSmokeTests` — temp = 0 is stable across runs.

### Not every model runs by default — env-gated tests

The largest checkpoints are too heavy (or too slow) for the routine
gate, so they are gated behind environment variables. With the var
unset the test skips (and passes); set it to opt in:

| Env var | Unlocks |
|---|---|
| `FFAI_BUILD_MACHINE` | The heavy generation checks — `GPTOSSIntegrationTests` (~20B MoE), the Gemma 4 31B / 26B-A4B decode in `Gemma4IntegrationTests` (load + shape checks still run unconditionally), and every non-smallest cell of `ModelKVCacheMatrixIntegrationTests`. Intended for a dedicated build machine. |
| `FFAI_MATRIX_FAMILY=<family>` | Restricts `ModelKVCacheMatrixIntegrationTests` to one family's row (e.g. `FFAI_MATRIX_FAMILY=Gemma4`) — fast targeted re-runs. |

```bash
# Run the whole matrix incl. env-gated cells, on a build machine:
FFAI_BUILD_MACHINE=1 swift test --parallel --num-workers 1 \
    --filter ModelKVCacheMatrixIntegrationTests

# Re-run just the Llama row of the matrix:
FFAI_MATRIX_FAMILY=Llama swift test --parallel --num-workers 1 \
    --filter ModelKVCacheMatrixIntegrationTests
```

The default `make test-integration` runs only the always-on cells:
the smallest checkpoint per family.

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

A model integration test loads through `ModelLoadLock.shared`
(serializes the multi-GB load across suites) and asserts coherence:

```swift
@Suite("Qwen3 integration", .serialized)
struct Qwen3IntegrationTests {
    @Test("load + greedy generate produces coherent output")
    func loadAndGenerate() async throws {
        let m: Model
        do { m = try await ModelLoadLock.shared.loadSerially {
                 try await Model.load("mlx-community/Qwen3-1.7B-bf16") } }
        catch { print("skipped: \(error)"); return }
        let r = try await m.generate(
            prompt: "Once upon a time",
            parameters: GenerationParameters(maxTokens: 64, temperature: 0))
        expectCoherentOutput(r.generatedTokens, label: "Qwen3-1.7B")
    }
}
```

A non-trivial kernel lands with a paired metaltile GPU-correctness
test in the **same commit** (see metaltile `docs/testing.md`).

## CI

CI runs on Apple Silicon: the unit gate, then the serialized
integration gate (matching `make test-integration`), and uploads the
coverage report — a PR that drops Swift-surface coverage below the
threshold fails.

## What we don't test

- **Property / fuzz testing** — revisit post-v0.2.
- **GPU mocking** — all tests run real Metal dispatches.
- **Cross-implementation token parity** — dropped (see above).
- **Multi-GPU / Linux / CUDA** — different project.

## See also

- [Developing](developing.md) — the `make` workflow, kernel regen.
- [Adding a model](adding-a-model.md) — which tests a new family adds.
- [Performance](../performance.md) — `Tests/PerfTests/` thresholds.
