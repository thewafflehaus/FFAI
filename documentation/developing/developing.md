# Developing in FFAI

Repo layout, the `make` workflow, and how to regenerate kernels.

## One-time setup

```bash
cd ~/Development
git clone https://github.com/thewafflehaus/FFAI
git clone https://github.com/thewafflehaus/metaltile     # sibling repo, required
cd FFAI
./scripts/setup-dev.sh
```

`setup-dev.sh` verifies:

- Xcode CLI tools (`xcode-select -p`)
- `xcrun metal` (the full Xcode IDE, not just CLI tools, is required)
- Swift toolchain (`swift --version`)
- Cargo (Rust, for `metaltile`)
- The sibling `metaltile` checkout at `../metaltile`

Then resolves SPM deps and runs `make build` to produce
`kernels.metallib`.

## Repo layout

```
Sources/
  FFAI/                    User-facing library
  MetalTileSwift/          Pre-compiled kernels + dispatch wrappers
    Resources/             kernels.metallib + manifest.json (generated)
    Generated/             MetalTileKernels.swift (generated)
  FFAICLI/                 ffai executable

Tests/
  MetalTileSwiftTests/     One file per kernel
  FFAITests/               Tensor, Module, KVCache, Sampling, ...
  ModelTests/              Per-family integration tests — load,
                           greedy-decode, assert coherent output

planning/                  Phased build-out + architecture diagrams
documentation/             User-facing docs (you are here)
scripts/                   setup-dev.sh, coverage.sh, verify-docs.sh
```

For the per-Sources-file purpose see
[`documentation/architecture.md` § File layout](../architecture.md#file-layout).

## The `make` workflow

```bash
make build              # regenerate kernels + swift build (debug)
make build-release      # regenerate kernels + swift build -c release
make regenerate-kernels # run `tile build --emit all` only
make test               # regenerate kernels + swift test
make coverage           # swift test --enable-code-coverage + summary
make format             # swift-format the repo in place
make format-check       # lint without modifying
make docs               # verify swift-docc builds clean
make clean              # remove .build + generated artifacts
```

`make build` and `make test` always run `regenerate-kernels` first —
no out-of-date kernels in CI or local dev.

## How kernel regeneration works

```
~/Development/metaltile          ←  Rust kernel source
   cargo run --bin tile          ←  generates →  Sources/MetalTileSwift/
   -- emit                                           Resources/kernels.metallib
   --out Sources/MetalTileSwift                     Resources/kernels/*.metal
                                                    Resources/manifest.json
                                                    Generated/MetalTileKernels.swift
                                                                  ↓
                                                    Sources/FFAI/Ops.swift uses
                                                    the generated typed wrappers
```

`make regenerate-kernels` runs `cargo run --release --bin tile -- emit
--out Sources/MetalTileSwift` from the sibling metaltile repo. Cargo
runs from the metaltile dir so its `rust-toolchain.toml` (nightly,
2024 edition) is honored. Eventually the `tile` binary will
ship via Homebrew so this won't need a metaltile checkout — only
kernel authors will need the repo.

The generated artifacts are checked into the FFAI repo so end-user SPM
consumers don't need Cargo or the metaltile checkout.

## Writing a new kernel

Kernels are Rust functions in metaltile, not Swift. The flow:

1. Add a `#[kernel]` Rust function to
   `crates/metaltile-std/src/ops/<file>.rs` in the metaltile sibling repo.
2. Annotate it with `#[bench_kernel(op="…", subop="…", class=…, …)]`
   so the registry picks it up. For kernels without an MLX-comparable
   bench, submit a hand-rolled `BenchSpec` with `shapes: &[]` and the
   right `kernel_mode: Some(KernelMode::…)` instead.
3. `make regenerate-kernels` from the FFAI repo — picks up the new
   kernel, regenerates `kernels.metallib` + `MetalTileKernels.swift`.
4. Add a thin `Ops.swift` wrapper if the typed `MetalTileKernels`
   signature isn't ergonomic enough for callers.
5. Add a `Tests/MetalTileSwiftTests/` test against fixed inputs /
   outputs.

## Adding a model family

See [adding-a-model.md](adding-a-model.md) for the full walk-through.
TL;DR:

1. New `Sources/FFAI/Models/<Family>.swift` with a `<Family>` enum
   declaring `modelTypes` + `architectures`, a `<Family>Variant`
   protocol, and one or more variant structs.
2. Register the family in `Sources/FFAI/Model.swift` →
   `ModelRegistry.dispatchAndLoad`.
3. Add `Tests/ModelTests/<Family>IntegrationTests.swift` — load the
   smallest published checkpoint, greedy-decode, and assert
   `expectCoherentOutput(...)`.

## Testing

See [testing.md](testing.md) for running tests, the
`expectCoherentOutput` integration model, and coverage targets.

## Coding conventions

- **Swift formatting** — `swift-format` per `.swift-format`. CI gates
  on `make format-check`.
- **Comments** — sparing. Lead with WHY, not WHAT. The
  Tensor/Module/Layer naming carries the WHAT.
- **No mocking the GPU.** Every test runs real Metal dispatches on
  the CI runner (Apple Silicon). Numerical correctness comes from the
  metaltile-side per-kernel GPU-correctness tests (vs a naive CPU
  oracle); FFAI integration tests assert the model pipeline produces
  coherent text. There are no golden fixtures.
- **No unused / speculative code.** Build only what the active phase
  needs. Future-phase fields go into `LoadOptions` with a comment
  pointer to the phase, not stubs in the call path.

## Common pitfalls

- **Forgot the metaltile checkout.** `make regenerate-kernels` will
  fail with `metaltile not found at ../metaltile`. Clone it.
- **Used Cargo from the FFAI repo.** Will fail with edition=2024
  errors. Always `cd metaltile && cargo run …` (the Makefile does
  this for you).
- **Hit the page-alignment crash** when adding a new tensor loader.
  `MTLBuffer.bytesNoCopy` requires page-aligned (16KiB) pointers; per-
  tensor offsets aren't aligned. Use `device.makeBuffer(bytes:length:)`
  instead — the existing SafeTensors path already does.
- **Sendable warnings on Metal types.** `MTLBuffer` / `MTLDevice` aren't
  `Sendable`. Wrap holders in `@unchecked Sendable` rather than
  fighting the compiler.

## See also

- [Testing](testing.md) — running tests, integration coherence
  checks, coverage.
- [Adding a model](adding-a-model.md) — porting a new architecture.
- [`planning/plan.md`](../../planning/plan.md) — what's in / out of
  scope per phase.
