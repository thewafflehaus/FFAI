# Publishing: branches, releases, docs

How FFAI's branching, release, and docs-publishing pipeline fit
together. **Read this before cutting your first release.**

## Branches

| Branch | Role |
|---|---|
| `dev`  | Day-to-day work. Every push + PR runs CI ([`ci.yml`](../../.github/workflows/ci.yml)). |
| `main` | Stable. Only advances via "release PR" merges from `dev`. |
| `release/v<X.Y.Z>` | Cut by the release workflow on each release. Kept open for hotfixes against that version line. |

Day-to-day: branch off `dev`, PR back to `dev`.
Release-time: PR `dev` → `main`.

## Release process

1. **Open a PR `dev` → `main`** titled `Release vX.Y.Z`. CI runs on
   the PR. Get a review, merge.
2. **Trigger the release workflow.** GitHub UI → Actions → **Release**
   → "Run workflow" on `main`. Inputs:
   - `bump_type` — `patch` / `minor` / `major`
   - `prerelease_tag` — `alpha` / `beta` / `rc` / `none`
   - `override_version` — bypass auto-bump (e.g. `0.2.0-alpha`)
   - `tag_prefix` — `v` (default) or `auto` (inherit from last tag)
3. The workflow does the rest: clean test pass on macOS, then
   [`scripts/release.sh`](../../scripts/release.sh) computes the next
   version from `git describe`, **rewrites `FFAI.version` in
   `Sources/FFAI/FFAI.swift` to match the new tag** (and commits the
   bump on `main` with `[skip ci]`), creates `release/<tag>` branch
   + an annotated tag pointing at the bump commit, pushes branch +
   tag + the updated `main`. Finally `gh release create
   --generate-notes` publishes the GitHub Release targeting the
   release branch.
4. The `release: published` event automatically fires
   [`notify-docs.yml`](../../.github/workflows/notify-docs.yml), which
   dispatches the [ffai-website](https://github.com/thewafflehaus/ffai-website)
   build against the new tag.

Always run the Release workflow **from `main`** (the branch picker in
the workflow_dispatch UI). The workflow guards against running off
any other branch.

## What the script does to `FFAI.version`

There's exactly one in-code version string,
[`FFAI.version`](../../Sources/FFAI/FFAI.swift), surfaced by `ffai
--version` + on every CLI invocation + recorded in bench reports.
`release.sh` keeps it in lockstep with the tag automatically:

- **If `FFAI.version` already matches** the computed `NEW_VERSION`
  (e.g. a contributor manually bumped it in the release PR), the
  script logs `no bump needed` and proceeds.
- **Otherwise** it rewrites the string literal, commits the change
  on the current branch with `chore: bump FFAI.version to <tag>
  [skip ci]`, and `git push origin <branch>`. The release branch +
  tag are created on the bump commit, so any checkout of the tag
  carries the matching version string.

Convention for `dev` between releases: suffix `-dev` (e.g.
`"0.2.0-dev"` after v0.1.0 ships) so stale dev builds are easy to
spot in CLI output and bench logs. This bump is **not** automated
today — open a manual PR on dev after each release.

## Dry-running the version bump locally

`scripts/release.sh` honors a `PUSH=0` env var for dry-runs:

```bash
PUSH=0 BUMP_TYPE=minor PRERELEASE_TAG=alpha ./scripts/release.sh
# FFAI.version already at 0.1.0 — no bump needed.
# (or:)
# [dry-run] would bump FFAI.version: 0.0.5-dev → 0.1.0-alpha + commit.
# tag=v0.1.0-alpha
# version=0.1.0-alpha
# release_branch=release/v0.1.0-alpha
# commit=<sha of HEAD>
```

Dry-run doesn't write to `FFAI.swift` or make any commits — it only
creates the local tag + release branch (which point at unchanged
HEAD). Clean up with:

```bash
git tag -d v0.1.0-alpha
git branch -D release/v0.1.0-alpha
```

Env vars the script understands:

- `BUMP_TYPE` — `major` / `minor` / `patch` (default `minor`)
- `PRERELEASE_TAG` — `alpha` / `beta` / `rc` / `none` (default `none`)
- `OVERRIDE_VERSION` — bypass auto-bump entirely (e.g. `0.2.0-alpha`)
- `TAG_PREFIX` — `v` / `auto` / empty string (default `auto`, which
  inherits from the last reachable tag and falls back to `v`)
- `PUSH` — `1` to push branch + tag, `0` for dry-run (default `1`)

## Release notes generation

The release workflow passes `--generate-notes` to `gh release create`.
GitHub generates the notes from PR titles + labels since the previous
reachable tag, grouped per
[`.github/release.yml`](../../.github/release.yml).

Labels get applied automatically by
[`auto-label.yml`](../../.github/workflows/auto-label.yml) based on
the PR title's conventional-commit prefix:

| PR title prefix | Label | Section in notes |
|---|---|---|
| `feat:` / `feature:`     | `feature`     | ✨ Features        |
| `fix:` / `bugfix:`       | `bug`         | 🐛 Bug Fixes       |
| `perf:`                  | `performance` | 🚀 Performance     |
| `docs:` / `doc:`         | `documentation` | 📚 Documentation |
| `test:` / `tests:`       | `test`        | 🧪 Tests           |
| `chore:` / `ci:` / `build:` / `refactor:` / `style:` | `ignore-for-release` | (hidden) |
| Any prefix with `!`      | `breaking`    | 💥 Breaking Changes |

So title PRs intentionally — `feat: …` for features, `fix: …` for
bugs, `chore: …` for refactors that shouldn't appear in user-facing
notes.

## Docs site publishing chain

The user-facing site at
[**https://thewafflehaus.github.io/ffai-website/**](https://thewafflehaus.github.io/ffai-website/)
**only rebuilds against immutable FFAI release tags** — never `main`
HEAD. Unreleased doc changes land on dev → main but stay invisible
to the published site until the next release.

```
FFAI release published
        │
        ▼
notify-docs.yml      (FFAI / .github/workflows/)
        │
        │ workflow_dispatch (Actions: write on ffai-website)
        ▼
deploy.yml           (ffai-website / .github/workflows/)
        │
        │ checkouts FFAI@<tag>, syncs markdown, builds Astro,
        │ deploys to GH Pages
        ▼
https://thewafflehaus.github.io/ffai-website/
```

Site rebuilds also happen when:
- You push to `main` on ffai-website (site source changed — Astro
  components, CSS, sidebar). Pins to FFAI's latest published release.
- You manually trigger ffai-website's `deploy.yml` from the Actions
  tab. Same — pins to latest release.

Either way the docs content stays pinned to a real release; you can
iterate on the site itself between FFAI releases.

## The cross-repo dispatch token

`notify-docs.yml` calls `gh workflow run deploy.yml` on ffai-website,
which requires authentication. The `WEBSITE_DISPATCH_TOKEN` repo
secret holds a fine-grained PAT scoped to **only** `ffai-website`
with:

| Permission | Level |
|---|---|
| Actions  | Read and write |
| Contents | Read-only |
| Metadata | Read-only |

**No `Contents: write` on ffai-website.** The dispatch uses
`workflow_dispatch` (not `repository_dispatch`) so the worst a
leaked token could do is spam-trigger the deploy workflow or cancel
runs — it can't modify ffai-website's repo contents.

If the secret is missing the notify workflow logs a warning and
exits 0 gracefully (no failed Actions run); the release still gets
created, the site just won't rebuild until you set the token.

## Manual rebuild commands

When you need to force a rebuild without cutting a release:

```bash
# Re-deploy the site against whatever the latest FFAI release is.
gh workflow run deploy.yml --repo thewafflehaus/ffai-website

# Re-deploy against a specific FFAI release (e.g. roll back the docs).
gh workflow run deploy.yml --repo thewafflehaus/ffai-website \
  --field ffai_tag=v0.1.0

# Re-run the full notify → dispatch chain (useful to test the token).
gh workflow run notify-docs.yml --repo thewafflehaus/FFAI \
  --field tag=v0.1.0
```

## See also

- [Architecture](../architecture.md) — where in the pipeline kernel
  generation, model load, and inference dispatch live.
- [Testing](testing.md) — what gets tested, where fixtures live.
- [`planning/roadmap.md`](../../planning/roadmap.md) — what we're
  shipping in upcoming releases.
