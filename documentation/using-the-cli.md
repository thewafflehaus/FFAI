# Using the CLI

The `ffai` executable is a SwiftPM product, not a Homebrew formula —
there's no global install step. After cloning the repo, build it with
`swift build` and invoke it through SwiftPM (`swift run ffai …`), the
built binary path, or by symlinking onto `PATH`.

## Build

```bash
git clone https://github.com/thewafflehaus/FFAI
cd FFAI

swift build -c release        # binary lands at .build/release/ffai
```

Use `-c debug` (the SwiftPM default) for faster compile + slower run;
`-c release` for the inference numbers you'd quote.

## Run

Pick one of three invocations — they're equivalent, just trade-offs
on ergonomics.

```bash
# (a) Via SwiftPM — no setup, recompiles if the source changed.
swift run -c release ffai generate -m unsloth/Llama-3.2-1B -p "Once upon a time"

# (b) Direct binary path — no recompile check, fastest start-up.
.build/release/ffai generate -m unsloth/Llama-3.2-1B -p "Once upon a time"

# (c) Symlink onto PATH (one-time) so plain `ffai …` works from anywhere.
ln -s "$PWD/.build/release/ffai" /usr/local/bin/ffai
ffai generate -m unsloth/Llama-3.2-1B -p "Once upon a time"
```

`generate` is the default subcommand, so the `-m / -p` flags can be
passed directly to `ffai` (`ffai -m … -p …` is equivalent to
`ffai generate -m … -p …`).

## Subcommands

| Subcommand | One-liner | More |
|---|---|---|
| `generate` (default) | Stream a single prompt's continuation to stdout. | `ffai generate --help` |
| `bench` | Run a benchmark method against a model, append to a per-day report. | [benchmarking.md](benchmarking.md) |

Common cross-cutting flags (`--stats`, `--debug`, `--profiling`) are
documented in [observability.md](observability.md).

## See also

- [Quick start](quickstart.md) — the 5-line library equivalent.
- [Benchmarking](benchmarking.md) — `ffai bench --method <name>`, KLD
  comparisons, per-day report shape.
- [Installation](installation.md) — adding FFAI to your own SwiftPM
  package (no CLI required).
