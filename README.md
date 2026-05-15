# FFAI

**Fucking Fast Apple Inference.**

A minimal, dependency-light LLM inference library for Apple Silicon, built on
pre-compiled Metal kernels generated from the [metaltile](https://github.com/metaltile/metaltile)
DSL. No MLX. No JIT. No four-repo dependency chain.

## Status

Early bootstrap.

- [`planning/plan.md`](planning/plan.md) — phased build-out, what we're
  shipping when
- [`planning/architecture.md`](planning/architecture.md) — visual
  reference for kernel generation, model loading, and the inference
  dispatch loop

## Architecture (target)

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
│   • metaltile-emit produces:                            │
│       kernels.metallib   (compiled by xcrun metal)      │
│       manifest.json      (kernel metadata)              │
│       MetalTileKernels.swift  (typed wrappers)          │
└─────────────────────────────────────────────────────────┘
```

## License

Apache-2.0.
