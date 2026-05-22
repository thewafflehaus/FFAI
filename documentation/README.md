# Documentation

Table of contents for the FFAI documentation. The top-level
[`README`](../README.md) is the curated landing page; this index lists
every page in the tree so you can jump straight to a topic.

## Getting started

- [Installation](installation.md) — SwiftPM / Xcode setup, platform
  requirements, sibling-metaltile checkout.
- [Quick start](quickstart.md) — generate text in 5 lines.
- [Using the CLI](using-the-cli.md) — build the `ffai` binary and run
  it via `swift run`, the built path, or a `PATH` symlink.
- [Architecture](architecture.md) — the three-layer stack
  (`metaltile` Rust → `MetalTileSwift` → `FFAI`), build pipeline, and
  per-token dispatch loop.
- [Models](models.md) — the supported architecture families (dense
  text, SSM / GDN / conv+attention hybrids, MoE, vision-language,
  audio, neural codecs), per-family known gaps, and adding a new
  family.

## Cross-cutting topics

- [`GenerationParameters` reference](generation-parameters.md) — every
  generation knob, per-family defaults table, the three call shapes
  (default, with-override, custom).
- [Streaming](streaming.md) — `generateStream(...)`,
  `GenerationChunk` shape, cancellation, why streaming is the
  primitive over which buffered `generate(...)` is built.
- [Chat templates](chat-templates.md) — `ChatMessage` +
  `ChatTemplateOptions`, `enableThinking` / `reasoningEffort` hooks,
  per-family quirks (Qwen 3 / DeepSeek-R1 / GPT-OSS / Gemma).
- [KV cache](kv-cache.md) — the raw fp16 / bf16 cache, GPU-side
  `kv_cache_update` kernel, and what's coming (affine, AURA,
  SSM/GDN).
- [Quantization](quantization.md) — mlx-format coverage (3 / 4 / 5 / 6
  / 8-bit), packing layout, sub-group split dispatch.
- [Performance](performance.md) — current `tok/s` numbers per model,
  what each Phase 4 wave got us, where the remaining headroom is.
- [Observability](observability.md) — `--stats` (per-phase memory,
  TTFT, KV cache, wired ticket), `--debug` (subsystem-tagged stderr
  logs), `--profiling` (wallclock + `os_signpost`), perplexity /
  think-vs-gen split helpers.
- [Benchmarking](benchmarking.md) — `ffai bench --method <name>` +
  `--ref-model` for KLD, per-day markdown + JSON sidecar reports
  (mlx-swift-lm-compatible row schema).
- [Capabilities & lifecycle](capabilities.md) — the
  `Capability` enum, `LoadOptions`, `ModelLifecycleEvent` stream.

## Local development

- [Developing in FFAI](developing/developing.md) — repo layout, the
  `make` workflow, regenerating kernels.
- [Adding a model](developing/adding-a-model.md) — porting a new
  architecture from a reference implementation.
- [Testing](developing/testing.md) — running tests, the
  `expectCoherentOutput` integration model, coverage targets.
- [Publishing](developing/publishing.md) — dev → main release flow,
  release notes, and how the docs site rebuilds.

## See also

- Top-level [`README`](../README.md) — project landing page.
- [`planning/plan.md`](../planning/plan.md) — phased build-out, what
  ships when.
- [`planning/architecture.md`](../planning/architecture.md) —
  longer-form architecture diagrams.

## How these docs get published

The user-facing site at **https://thewafflehaus.github.io/ffai-website/**
is built from the markdown in *this* repo (`documentation/*.md`,
`README.md`, `planning/architecture.md`, `planning/roadmap.md`) by a
separate site repo,
[thewafflehaus/ffai-website](https://github.com/thewafflehaus/ffai-website).
The site fetches FFAI's markdown at build time — there's no manual
copy step.

**The published site always builds against a real, immutable FFAI
release tag — never main HEAD.** So unreleased doc changes that land
on this repo's main are intentionally invisible to the published site
until the next release.

### When the site rebuilds

| Trigger | What happens |
|---|---|
| **A new release is published on this repo** | `.github/workflows/notify-docs.yml` calls `workflow_dispatch` on ffai-website's `deploy.yml`, passing the release tag, name, body, and url as inputs. ffai-website pins its FFAI checkout to that tag, renders the release body as the Changelog page, updates the version label in the site title + hero, then deploys. |
| Push to `main` on `ffai-website` | Site source changed (CSS, layout, new page). ffai-website rebuilds against FFAI's *latest published release* (via `gh release view`). |
| Manual dispatch on either repo | Same — `ffai-website` always builds against the latest release (or against a specific tag if you pass `--field ffai_tag=...`). |

### The token

The cross-repo dispatch needs a `WEBSITE_DISPATCH_TOKEN` secret on
**this repo** (FFAI). Use a fine-grained PAT scoped to *only*
`thewafflehaus/ffai-website` with **`Actions: Read and write`** plus
the implicit `Contents: Read-only` + `Metadata: Read-only`. The token
does **not** need `Contents: write` — the dispatch uses
`workflow_dispatch` (not the older `repository_dispatch`) so the
worst a leaked token could do is spam-trigger the deploy workflow
or cancel runs; it can't modify ffai-website's repo contents.

### Releasing → publishing flow

1. Land doc changes on `dev` alongside the code changes they describe.
2. Open a PR `dev` → `main` titled `Release vX.Y.Z`. Wait for CI;
   merge.
3. Trigger the **Release** workflow from the GitHub UI (Actions →
   Release → Run workflow on `main`). Pick the right `bump_type` +
   `prerelease_tag`; it runs tests, then
   [`scripts/release.sh`](../scripts/release.sh) tags the merge commit
   and `gh release create --generate-notes` publishes the release.
4. The `release: published` event automatically fires
   `notify-docs.yml`; the site rebuilds within a minute or two and
   the Changelog gets a new section from the release body.

You can also kick a rebuild manually without cutting a release:

```bash
# Re-publish against the latest release (e.g. you want to verify the
# site build before / between releases).
gh workflow run deploy.yml --repo thewafflehaus/ffai-website

# Force a rebuild against a specific past release.
gh workflow run notify-docs.yml --repo thewafflehaus/FFAI --field tag=v0.1.0

# Rebuild ffai-website against an arbitrary FFAI tag directly (skips
# the FFAI notify hop — handy for debugging the site itself).
gh workflow run deploy.yml --repo thewafflehaus/ffai-website \
  --field ffai_tag=v0.1.0
```

### Previewing unreleased doc changes locally

The published site won't show unreleased docs, but the local Astro
dev server can build against your FFAI working tree:

```bash
git clone https://github.com/thewafflehaus/ffai-website ../ffai-website
cd ../ffai-website
pnpm install
FFAI_REPO_PATH=$(pwd)/../FFAI pnpm dev   # → http://localhost:4321
```

`make docs` from this repo prints the same commands if the
`../ffai-website` checkout exists.
