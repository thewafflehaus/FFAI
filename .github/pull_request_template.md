## Proposed changes

Please describe the problem or feature this PR addresses. Link any
relevant issue with `#<issue-number>`.

## Checklist

- [ ] I have read [`planning/plan.md`](../planning/plan.md) and
      confirmed this change fits the current phase or is explicitly
      out of scope
- [ ] I have run `pre-commit run --all-files` (or installed pre-commit)
- [ ] I have added tests that exercise the new code (100% line coverage
      target — see `planning/plan.md` Quality bar)
- [ ] I have updated `planning/architecture.md` if this changes a
      load-time, kernel, or dispatch flow
- [ ] If this PR adds a new kernel, I regenerated artifacts via
      `cargo run -p metaltile-emit` and the build still passes

## Conventional commit prefix

PR title prefix is used by `auto-label.yml` for release-notes
categorization. Use one of:

`feat: …` `fix: …` `perf: …` `docs: …` `test: …`
`chore: …` `ci: …` `build: …` `refactor: …` `style: …`

Add `!` for breaking changes (`feat!: …`).
