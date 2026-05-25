# Design — `ffai bench --mactop` integration

Spawn `mactop` alongside `ffai bench`, capture CPU / GPU / memory /
power / temperature samples for the bench window, pin fans high
during the run, and write the resulting time series next to the
existing markdown + JSON sidecar. Density over grammar; companion to
the other `planning/*-design.md` docs.

---

## 1. Why

Bench rows today (`BenchCommand.swift`) report `prefill tok/s`,
`decode tok/s`, `TTFT`, optional `gen perplexity` / `gen KLD`. They
don't capture the thermal regime, GPU utilisation, or CPU
saturation — so a number measured with the chassis at 95 °C and the
GPU thermally throttled looks identical to one measured cool. On
sustained benches (wikitext2, niah, long-context decode) the chip
**will** throttle if fans aren't pinned. mactop already exposes
every signal we need; we just need to spawn it, parse its headless
JSON, and gate `--fan-control` to keep the GPU/CPU in their
non-throttling regime for the full bench window.

This also gives us a way to surface the
[CPU pin — vision / audio models](known-issues.md#cpu-pin--vision--audio-models-burn-all-cpu-cores)
issue with real data: a bench row with GPU usage low + every CPU
core pinned is exactly the silent-CPU-fallback signature.

## 2. mactop surface we'll use

mactop ships with the flags we need today (verified via
`mactop --help` on 0.x).

| Flag | Use |
|---|---|
| `--headless` | No TUI — emits structured samples to stdout |
| `--format json --pretty` | Each sample is a JSON object; `--pretty` lets us tail with `jq` |
| `-i <ms>` | Sample interval; default 1000 ms |
| `--count <n>` | Cap sample count (0 = infinite). We pass `0` and stop via SIGINT when the bench finishes |
| `--pid <PID>` | Restrict per-process CPU / memory accounting to the bench process |
| `--fan-control` | Writes SMC fan keys; requires root (sudo) on macOS |
| `--unit-temp celsius` | Standardise temp output |
| `--dump-temps` | Diagnostic only — handy if `--fan-control` misbehaves on a new chip |

Not used: `--prometheus` (out of scope), `--overlay` /
`--menubar` (TUI-only), `--foreground` / `--bg` (theming).

## 3. CLI surface

New flags on `BenchCommand` (`Sources/FFAICLI/BenchCommand.swift`):

```
--mactop                       Spawn mactop alongside the bench. Captures
                               CPU / GPU / memory / power / temp samples
                               and writes <chip>-YYYY-MM-DD-<method>-mactop.json
                               next to the bench report.

--mactop-interval-ms <N>       mactop sample interval. Default 500.

--mactop-pin-fans              During the bench window, pin fans to their
                               maximum SMC value. Requires running ffai
                               under sudo so mactop's --fan-control can
                               write SMC. Restores prior fan mode on exit.
                               Implies --mactop.

--mactop-binary <PATH>         Override the mactop executable. Default:
                               first `mactop` on $PATH; error if missing.
```

`--mactop-pin-fans` is the user-facing knob for "stats + fans pinned
high" mode. No threshold logic in v1 — fans pinned for the entire
bench window, restored on exit. The noise tradeoff is acceptable for
bench mode.

## 4. Process lifecycle

`BenchCommand.run()` adds a `MactopProbe` actor (new file
`Sources/FFAICLI/MactopProbe.swift`) wrapped around the bench
invocation:

```text
1. Resolve mactop binary (--mactop-binary or first on PATH).
   Fail fast with a clear error if missing.

2. If --mactop-pin-fans:
   - Require euid == 0 (geteuid). If not root, error:
     "ffai bench --mactop-pin-fans must run under sudo
      (mactop --fan-control writes SMC)."
   - Snapshot current fan mode via `mactop --dump-temps` so we can
     log the pre-bench thermal regime in the sidecar.

3. Spawn mactop subprocess:
     mactop --headless --format json --pretty \
            -i <interval-ms> --count 0 \
            --pid <ffai-pid> \
            --unit-temp celsius \
            [--fan-control]   # only when --mactop-pin-fans

   Capture stdout via Pipe; line-buffer; parse each line as one
   JSON sample; append to an in-memory ring buffer. stderr goes
   straight to the FFAI process's stderr.

4. Run the bench (existing BenchRunner.run path).

5. On bench completion (success OR failure):
   - SIGINT the mactop process; wait up to 2s; SIGTERM if still
     alive; SIGKILL after another 1s. mactop releases SMC writes
     on signal, restoring prior fan mode.
   - Flush ring buffer to <reportDir>/<chip>-YYYY-MM-DD-<method>-mactop.json
     using the same chip + date + method tags as the bench sidecar.
   - Append a `mactopSummary` block to the bench JSON sidecar with
     min / max / mean of CPU% / GPU% / memory pressure / package
     power / max temps over the bench window.

6. If mactop crashed mid-bench: log the stderr tail, mark the
   sidecar `mactopStatus: crashed`, but DO NOT fail the bench row —
   stats are observational, not load-bearing.
```

## 5. Output layout

### 5a. Per-sample file

`<reportDir>/<chip>-YYYY-MM-DD-<method>-mactop.json` — newline-
delimited JSON, one record per mactop sample, raw mactop fields
preserved. Easy to `jq` or replay.

```json
{"timestamp": 1716580800123, "cpu_pct": 12.4, "gpu_pct": 87.2,
 "gpu_freq_mhz": 1398, "memory_used_mb": 18342, "power_w": 24.1,
 "temp_cpu_c": 71.0, "temp_gpu_c": 78.0, "fan_rpm": [2680, 2700]}
```

### 5b. Summary in bench sidecar

The existing bench JSON sidecar (one row per bench invocation) gets
a new optional `mactop` block:

```json
"mactop": {
  "samples": 412,
  "intervalMs": 500,
  "durationMs": 206000,
  "cpu":  {"min": 1.2, "mean": 14.8, "max": 96.4},
  "gpu":  {"min": 0.0, "mean": 81.3, "max": 98.7},
  "tempCpuC": {"min": 51.0, "mean": 68.4, "max": 79.0},
  "tempGpuC": {"min": 54.0, "mean": 74.2, "max": 84.0},
  "powerW":   {"min": 8.1, "mean": 22.4, "max": 31.2},
  "fanPinned": true,
  "fanRpmMax": 6500,
  "status": "ok"
}
```

The markdown report keeps its existing row schema for cross-compat
with mlx-swift-lm tooling — no new columns. The summary block goes
to the JSON sidecar only.

## 6. Fan control & permissions

`--fan-control` writes the `F<N>Md` / `F<N>Tg` SMC keys. macOS
locks these to root. Implications:

- `--mactop-pin-fans` requires `sudo ffai bench …`. Document this
  prominently in the flag's help and in `documentation/`.
- mactop is invoked with `--fan-control` only when `--mactop-pin-fans`
  is set. Plain `--mactop` runs unprivileged (read-only sensors).
- On exit (clean, signal, or crash) mactop restores the prior fan
  mode by releasing the SMC override. We trust mactop here rather
  than re-implementing fan management. Cross-check: `--dump-temps`
  before + after to confirm restoration; warn if the post-bench
  mode doesn't match the pre-bench snapshot.
- M1 / M2 / M3 / M4 fan SMC keys differ; mactop handles the
  per-chip mapping. We do **not** thread chip family into our own
  code — mactop is the abstraction.

## 7. Failure modes

| Failure | Behaviour |
|---|---|
| mactop binary missing | `--mactop` errors before model load; clear "install via `brew install mactop`" hint |
| `--mactop-pin-fans` without sudo | Errors before model load |
| mactop spawn fails (`posix_spawn` ENOENT) | Errors before model load |
| mactop crashes mid-bench | Bench continues; sidecar marks `mactopStatus: crashed`; stderr tail logged |
| mactop hangs on shutdown | SIGINT → 2s → SIGTERM → 1s → SIGKILL escalation |
| SMC restore fails (fans stuck high) | Warn loudly post-bench; surface in sidecar `fanRestoreOk: false`; user can `sudo mactop --fan-control` to manually release |
| JSON parse error on a sample | Drop the sample, increment a `parseErrors` counter in the summary block; never abort the bench |

## 8. Out of scope (v1)

- **Threshold-based fan control.** v1 pins fans for the entire
  bench window; the `temp ≥ 85 °C` ramp variant is a v2 once we
  have baseline temp histograms across the model zoo.
- **Prometheus / external scrape.** mactop's `--prometheus <port>`
  is left unwired; this design ships file-based capture only.
  Hooking a scrape later is a one-line addition (forward the flag).
- **Per-kernel attribution.** mactop samples are wall-clock global
  — they don't attribute power / temp to specific Metal kernels.
  That belongs to a `--profiling 2` / `os_signpost` + Metal System
  Trace integration, separate roadmap item.
- **Continuous-bench thermal logging** outside `ffai bench`. The
  flag is bench-scoped; the inference engine itself stays
  observation-free in v1.

## 9. Implementation files

| File | Change |
|---|---|
| `Sources/FFAICLI/BenchCommand.swift` | Add four `@Flag` / `@Option`s; wrap `runner.run(...)` with `MactopProbe.withProbe(...)`; thread sidecar summary into the existing `BenchmarkWriter.append(...)` |
| `Sources/FFAICLI/MactopProbe.swift` | NEW. Actor managing subprocess lifecycle, stdout parsing, SIGINT/SIGTERM/SIGKILL escalation, root-check + SMC pre/post snapshot |
| `Sources/FFAI/Bench/BenchmarkRow.swift` (or wherever the row + sidecar codable lives) | Add optional `mactop: MactopSummary` field; new `MactopSummary` codable |
| `Tests/FFAICLITests/MactopProbeTests.swift` | NEW. Stub-binary tests: spawn a `bash -c 'echo {...}; sleep'` fake mactop, confirm we parse, SIGINT, summarise. No real mactop dep in CI |
| `documentation/cli/bench.md` (or equivalent) | Document the four flags, the sudo requirement, the sidecar schema |

## 10. Unresolved

- Should `--mactop-pin-fans` *also* re-pin fans every N seconds
  defensively in case mactop's SMC write decays? Probably no — the
  fan setpoint persists until explicitly released — but worth a
  bench-during-development check.
- Does `--pid <ffai-pid>` correctly attribute GPU% on Apple Silicon?
  mactop's GPU sampling is via IOReport, which is system-global on
  the Apple GPU. `--pid` likely only narrows the CPU% / memory
  columns. Verify and document.
- Where does the mactop NDJSON live when `--report-dir` isn't
  writable? Default behaviour today errors on the existing markdown
  / sidecar write; keep that contract.
