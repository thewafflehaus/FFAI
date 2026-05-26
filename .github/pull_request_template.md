<!--
  Read CONTRIBUTING.md before opening this PR:
    https://github.com/thewafflehaus/FFAI/blob/main/CONTRIBUTING.md

  PRs without a linked, discussed issue will be closed.
-->

## Linked issue

<!--
  REQUIRED. Use `Fixes #123` or `Closes #123` (auto-closes the issue
  on merge), or `Refs #123` if the PR only partially addresses it.
  PRs with no linked issue will be closed.
-->

Fixes #

## Type of change

<!-- Check all that apply. PR title prefix (feat: / fix: / perf: /
     docs: / test: / chore: / refactor:) drives release-notes
     categorization via .github/release.yml — see CONTRIBUTING.md. -->

- [ ] 🐛 Bug fix (no API change)
- [ ] ✨ Feature (new API surface or capability)
- [ ] 🤖 New model (architecture / variant / size)
- [ ] 🚀 Performance (faster / smaller, no API change)
- [ ] 📚 Documentation
- [ ] 🧪 Tests only
- [ ] 🔧 Chore / refactor / CI (`ignore-for-release` label, hidden from notes)
- [ ] 💥 Breaking change (any of the above with `!` suffix in commit prefix)

## Summary

<!-- One short paragraph: what does this PR do, and why. Keep it tight —
     reviewers should know if the change is in their wheelhouse
     without scrolling. -->

## What changed

<!-- Bulleted list of the substantive changes, grouped by file or
     subsystem. Skip mechanical things (whitespace, lint). -->

-

## Test plan

<!-- How you verified this works. Required for behavior changes. -->

- [ ] `make test` passes locally
- [ ] Added/updated tests in `Tests/FFAITests/` or `Tests/ModelTests/`
- [ ] Manual verification (describe below)

<details>
<summary>Manual test steps + observed output</summary>

```

```

</details>

## Documentation

<!-- Required for user-visible changes. The docs site rebuilds
     against the next release tag — see
     documentation/developing/publishing.md. -->

- [ ] No user-visible change (skip)
- [ ] Updated `documentation/` for the affected surface
- [ ] Updated `planning/architecture.md` (load-time / kernel / dispatch flow changed)
- [ ] Updated `planning/roadmap.md` (shipped vs planned status changed)

## AI assistance disclosure

<!-- Check each category that applies, or "No AI used" if hand-written.
     Transparency, not gatekeeping. Agentic PRs are welcome — but the
     diff + description still need to read as if hand-written. -->

- [ ] 🔍 **Research** — searching docs, prior art, related issues
- [ ] 💡 **Ideation** — brainstorming approach, weighing trade-offs
- [ ] ⌨️ **Implementation** — writing the actual code / documentation
- [ ] 🧪 **Testing** — writing or running test cases
- [ ] 📚 **Documentation** — writing this PR's content
- [ ] ✋ **No AI used** — this is hand-written

## Final checklist

- [ ] Linked issue above is real, open, and was discussed before I opened this PR
- [ ] PR title follows conventional-commit prefix (`feat:` / `fix:` / `perf:` / `docs:` / `test:` / `chore:` / `refactor:`, optionally with `!` for breaking)
- [ ] Scope is tight — one logical change (split if not)
- [ ] CI is green
- [ ] Tests + docs updated where applicable
- [ ] I read [`CONTRIBUTING.md`](../blob/main/CONTRIBUTING.md)
- [ ] I agree with and authorize my contribution to be distributed under the [`LICENSE`](../blob/main/LICENSE)
