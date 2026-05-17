# Contributing to FFAI

Thanks for your interest. FFAI is a small, focused project — these
guidelines keep contributions coherent and the maintainer queue sane.

**TL;DR:**
- Open an issue **before** opening a PR.
- PRs without a linked issue **will be closed**.
- Tests + docs land with the code that needs them.
- AI-assisted contributions are welcome (preferred, even) — disclose
  how you used AI in the issue/PR template.

## Open an issue first

Even small changes. The issue is where we discuss scope, surface
related work, and align on the approach. A 5-minute exchange there
saves a lot of rework on a PR.

PRs that aren't linked to a discussed issue **will be closed**. No
exceptions — the rule applies equally to maintainers and external
contributors.

Use the [issue template](.github/ISSUE_TEMPLATE/issue.yml). Tick the
type checkboxes (bug, feature, new model, performance, discussion)
and the AI-usage checkboxes so the maintainer queue has consistent
metadata.

## What a good PR looks like

- **Linked to an open, discussed issue** (`Fixes #123` in the PR body).
- **Scoped tightly.** One logical change per PR. If it touches three
  unrelated things, that's three PRs.
- **Tests for behavior changes**, **docs for user-visible changes**.
  See [`documentation/developing/testing.md`](documentation/developing/testing.md)
  for what kind of test goes where; see
  [`documentation/`](documentation/README.md) for the docs surface.
- **Passes CI** (`make test`) before requesting review.
- **Uses the [PR template](.github/pull_request_template.md).**

PRs without tests or docs get feedback to add them before review.

## Agentic contributions

We **prefer** AI-assisted contributions. Claude, Cursor, Aider,
whatever you use — bring it. Agents often produce PRs with tighter
descriptions, better test coverage, and clearer reasoning than
hand-written ones.

Two rules:

1. **Disclose.** The issue and PR templates have an "AI assistance
   disclosure" checkbox group with four categories: **research**,
   **ideation**, **implementation**, **testing**. Tick the ones that
   apply. This is for transparency — not gatekeeping.
2. **Curate before opening.** An AI-assisted PR should read no
   differently from a hand-written one: tight description, linked
   issue, scoped diff, tests, docs. Don't paste raw assistant output.
   If the diff is sprawling or the description is vague, tighten it
   before opening.

The same applies to issues — if your assistant produces a
2000-word writeup, condense to what's actually relevant before
filing.

## Code of conduct

The usual: no spam, no off-topic content, no harassment, no
back-seat-driving on closed issues. Maintainer discretion on what
counts. Repeated violations → blocked from the org.

## Setup

See the top-level [`README.md` § Contributing](README.md#contributing)
for clone + sibling-metaltile + `make test` instructions.

## Deeper reading

- [`documentation/developing/developing.md`](documentation/developing/developing.md)
  — Make workflow, repo layout, kernel regeneration, writing new kernels.
- [`documentation/developing/testing.md`](documentation/developing/testing.md)
  — Test discovery, golden fixtures, coverage targets.
- [`documentation/developing/adding-a-model.md`](documentation/developing/adding-a-model.md)
  — Porting a new model family.
- [`documentation/developing/publishing.md`](documentation/developing/publishing.md)
  — Cutting releases + how the docs site rebuilds.
- [`planning/architecture.md`](planning/architecture.md) —
  Architectural invariants that constrain what can land where.
- [`planning/roadmap.md`](planning/roadmap.md) — What's shipped
  vs planned. Read before proposing a feature.

## License

By contributing you agree your contribution is licensed under
[Apache-2.0](LICENSE).
