# Installation

FFAI ships as a SwiftPM package with three products:

| Product | Use it for |
|---|---|
| `FFAI` | The main inference library — `Model.load(...)`, `generate(...)`, KV cache, sampling, tokenizer integration. |
| `MetalTileSwift` | Lower-level: the pre-compiled `kernels.metallib`, PSO cache, and generated typed kernel wrappers. Pulled in transitively by `FFAI`; depend on it directly only when authoring new layers against raw kernels. |
| `ffai` | Executable target — the `ffai --model <id> --prompt "..."` CLI. |

## SwiftPM `Package.swift`

```swift
.package(url: "https://github.com/thewafflehaus/FFAI", from: "0.1.0"),

.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "FFAI", package: "FFAI"),
    ]
)
```

## Xcode

1. **Project → Package Dependencies → `+`**
2. Enter `https://github.com/thewafflehaus/FFAI` and pick a version / branch.
3. Add the `FFAI` product to your target.

Apple's [Adding package dependencies to your app](https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app)
docs cover the UI flow.

## Transitive dependencies

`FFAI` itself pulls in:

- [`swift-huggingface`](https://github.com/huggingface/swift-huggingface)
  (≥ 0.9.0) — HF Hub snapshot download / cache.
- [`swift-transformers`](https://github.com/huggingface/swift-transformers)
  (≥ 1.3.0) — `AutoTokenizer.from(modelFolder:)`.
- [`swift-argument-parser`](https://github.com/apple/swift-argument-parser)
  (≥ 1.5.0) — only the `ffai` executable target depends on this.

You don't add these yourself unless you want to use them directly in
your own code.

## Platform requirements

- **macOS 14.0+** / **iOS 17+** / **visionOS 1+**
- **Apple Silicon** (M-series). FFAI is Metal-only in the inference
  hot path; CPU fallback is for small auxiliary tensors only.

The HuggingFace cache lives at
`~/.cache/huggingface/hub/` — same path Python's
`huggingface_hub` uses, so model snapshots are shared between
languages.

## Local checkout (contributors)

Cloning the repo to hack on FFAI itself requires the sibling
[`metaltile`](https://github.com/thewafflehaus/metaltile) checkout (the
Rust DSL that compiles kernels) and Cargo:

```bash
git clone https://github.com/thewafflehaus/FFAI
git clone https://github.com/thewafflehaus/metaltile
cd FFAI
./scripts/setup-dev.sh
```

`setup-dev.sh` verifies Xcode CLI tools + `xcrun metal` + Swift +
Cargo, resolves SPM deps, and runs the first build to populate
`kernels.metallib`. End users adding FFAI as a SwiftPM dependency
do **not** need Cargo or the metaltile repo — they consume the
pre-built metallib bundled into the package.

## After install

- [Quick start](quickstart.md) — generate text in 5 lines.
- [Models](models.md) — supported architectures and quantizations.
- [KV cache](kv-cache.md) — what's shipped today.
