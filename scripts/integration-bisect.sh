#!/bin/bash
# integration-bisect.sh
#
# Run every Tests/ModelTests/*IntegrationTests.swift suite ONE AT A TIME
# in a fresh process, capturing:
#   - pass / fail / timeout
#   - wall-clock duration
#   - GPU utilization shortly after the test finishes (to detect the
#     "GPU stays at 100% after test exits" pinned-command-buffer signature)
#
# Output: a markdown table printed to stdout + saved to
# planning/integration-bisect-<UTC>.md. Each row is one test suite.
#
# Why this script exists:
#   The full `make test-integration` runs every suite in one swift-test
#   process (serialized via --num-workers 1 + ModelLoadLock). If any
#   suite hangs or leaves the GPU pinned, the failure mode is hidden
#   inside a single multi-hour process. Running each suite in its own
#   process means:
#     - A pinned GPU at the END of a suite implicates THAT suite's
#       kernels, not its successors.
#     - A test that NEVER completes can be timeout-killed without
#       sacrificing the runs that followed.
#     - We can attach Instruments / xctrace to any specific suite
#       after this script localises the culprit.
#
# Usage:
#   ./scripts/integration-bisect.sh                  # all suites
#   ./scripts/integration-bisect.sh Whisper Llama   # named suites only
#   PER_TEST_TIMEOUT=600 ./scripts/integration-bisect.sh
#
# Env vars:
#   PER_TEST_TIMEOUT  Seconds before a suite is treated as hung
#                     (SIGTERM, then SIGKILL). Default 900 (15 min).
#   GPU_SETTLE_SEC    Seconds to wait after a suite exits before
#                     sampling GPU util. Default 3.
#   GPU_PIN_THRESHOLD Percent above which the GPU counts as pinned.
#                     Default 50.
#   OUTPUT            Override the markdown output path.
#
# GPU sampling uses `powermetrics` (requires sudo). If sudo is not
# available the script still reports pass/fail/duration; the
# GPU-pinned column degrades to "?".

set -uo pipefail

cd "$(dirname "$0")/.."

PER_TEST_TIMEOUT="${PER_TEST_TIMEOUT:-900}"
GPU_SETTLE_SEC="${GPU_SETTLE_SEC:-3}"
GPU_PIN_THRESHOLD="${GPU_PIN_THRESHOLD:-50}"
OUTPUT="${OUTPUT:-planning/integration-bisect-$(date -u +%Y%m%d-%H%M%S).md}"

# Selector — either explicit args or every IntegrationTests file.
#
# NOTE on word splitting: zsh does NOT split unquoted parameter
# expansions like bash does, so `script.sh $SOMEVAR` from a zsh shell
# passes a single concatenated argument when SOMEVAR contains spaces.
# Callers that want to pass many suite names from a zsh shell should
# either (a) quote / expand explicitly (`script.sh "${ARR[@]}"`), or
# (b) pipe via `xargs script.sh < list.txt`. Inside this script we
# split each incoming arg on whitespace into the SUITES array so the
# "one big string" case still works.
if [[ $# -gt 0 ]]; then
  SUITES=()
  for arg in "$@"; do
    # Split any space-delimited blob into individual suite names.
    for tok in $arg; do
      SUITES+=("${tok%IntegrationTests}IntegrationTests")
    done
  done
else
  SUITES=()
  while IFS= read -r line; do
    SUITES+=("$line")
    # Phase A.3 reorg moved suites into Tests/ModelTests/{Text,Vision,Audio}/...
    # subfolders; recurse with `find` so every nested IntegrationTests.swift
    # is picked up, and de-dupe in case macOS APFS surfaces case-insensitive
    # path twins.
  done < <(find Tests/ModelTests -name '*IntegrationTests.swift' -type f | sed 's|.*/||;s|\.swift$||' | sort -u)
fi

# Use sudo -n (no password) to test whether powermetrics is callable.
# If not, GPU sampling is disabled but the rest still runs.
GPU_AVAILABLE=0
if sudo -n true 2>/dev/null; then
  if command -v powermetrics &>/dev/null; then
    GPU_AVAILABLE=1
  fi
fi

# Sample GPU active-residency once. Returns an integer percentage.
# Output is "%" or "?" if unavailable.
sample_gpu_pct() {
  if [[ $GPU_AVAILABLE -eq 0 ]]; then
    echo "?"
    return
  fi
  # powermetrics --samplers gpu_power -i 500 -n 1 emits a GPU power
  # block; the "GPU Active residency" line gives a percentage 0..100.
  local pct
  pct=$(sudo -n powermetrics --samplers gpu_power -i 500 -n 1 2>/dev/null \
        | awk '/GPU [aA]ctive residency/{
            for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+%$/) { print int($i); exit }
            for(i=1;i<=NF;i++) if($i ~ /^[0-9]+%$/) { print int($i); exit }
          }')
  echo "${pct:-?}"
}

# Ensure planning/ exists.
mkdir -p "$(dirname "$OUTPUT")"

# Header.
{
  echo "# Integration bisect — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "Per-suite serial integration run with GPU pinning probe."
  echo
  echo "- Per-test timeout: ${PER_TEST_TIMEOUT}s"
  echo "- GPU settle: ${GPU_SETTLE_SEC}s"
  echo "- GPU pin threshold: ${GPU_PIN_THRESHOLD}%"
  echo "- GPU sampling: $([[ $GPU_AVAILABLE -eq 1 ]] && echo "powermetrics (sudo OK)" || echo "DISABLED (sudo unavailable)")"
  echo
  echo "| Suite | Status | Duration | GPU post-test | Notes |"
  echo "|---|---|---|---|---|"
} | tee "$OUTPUT"

PASS=0; FAIL=0; TIMEOUT=0; PINNED=0

for suite in "${SUITES[@]}"; do
  printf "  ▶ %-46s" "$suite"
  start=$SECONDS
  log=$(mktemp)
  status="PASS"
  note=""

  # `gtimeout` (coreutils) if present, else `timeout`. macOS ships
  # neither by default; install via `brew install coreutils`. Fall
  # back to no timeout (acceptable for the per-suite run; the user
  # can ^C if a suite truly hangs).
  TIMER=()
  if command -v gtimeout &>/dev/null; then
    TIMER=(gtimeout --kill-after=10s "$PER_TEST_TIMEOUT")
  elif command -v timeout &>/dev/null; then
    TIMER=(timeout --kill-after=10s "$PER_TEST_TIMEOUT")
  fi

  # Run the single suite. `make test-unit`-style flags. The Makefile
  # uses --parallel --num-workers 1 for ModelTests but for a single
  # suite the parallelism flag is moot.
  #
  # `${TIMER[@]+...}` is the canonical bash idiom for expanding a
  # possibly-empty array under `set -u` — without it, an empty TIMER
  # makes `${TIMER[@]}` trip the unbound-variable check. With the `+`
  # form, the whole expansion is skipped (no `timeout` prefix) when
  # the array is empty.
  #
  # `arch -arm64` forces the Apple-Silicon native slice of swift's
  # universal binary. Without this, a Rosetta'd parent bash subshell
  # (Claude Code's Bash tool runs through one in some configurations)
  # picks the x86_64 slice, which sets Float16 availability to
  # "macOS unavailable" and fails the whole build with bogus errors
  # like "argument passed to call that takes no arguments". See the
  # 2026-05-24 bisect-resume autopsy.
  FFAI_MAX_COMMAND_BUFFERS=16 ${TIMER[@]+"${TIMER[@]}"} \
    arch -arm64 swift test --filter "${suite}$" > "$log" 2>&1
  rc=$?
  dur=$(( SECONDS - start ))

  if [[ $rc -eq 124 ]] || [[ $rc -eq 137 ]]; then
    status="TIMEOUT"
    note="exceeded ${PER_TEST_TIMEOUT}s"
    TIMEOUT=$((TIMEOUT + 1))
  elif [[ $rc -ne 0 ]]; then
    status="FAIL"
    # Pull the first recorded issue / assertion for the notes column.
    note=$(grep -m1 -E "recorded an issue|Expectation failed|error:" "$log" \
           | sed 's/[|]/\\|/g' | head -c 100 || true)
    FAIL=$((FAIL + 1))
  else
    PASS=$((PASS + 1))
  fi

  # Let the GPU come to rest (or NOT, that's the signal).
  sleep "$GPU_SETTLE_SEC"
  gpu_pct=$(sample_gpu_pct)
  gpu_marker=""
  if [[ "$gpu_pct" =~ ^[0-9]+$ ]] && [[ $gpu_pct -ge $GPU_PIN_THRESHOLD ]]; then
    gpu_marker="🔴 PINNED"
    PINNED=$((PINNED + 1))
  elif [[ "$gpu_pct" =~ ^[0-9]+$ ]]; then
    gpu_marker="${gpu_pct}%"
  else
    gpu_marker="?"
  fi

  # Per-suite log file kept under planning/ so the user can inspect.
  log_dir="planning/integration-bisect-logs"
  mkdir -p "$log_dir"
  cp "$log" "$log_dir/${suite}.log"
  rm -f "$log"

  printf "  %-7s  %4ds  GPU=%s\n" "$status" "$dur" "$gpu_marker"
  printf "| %s | %s | %ds | %s | %s |\n" \
    "$suite" "$status" "$dur" "$gpu_marker" "${note:-—}" >> "$OUTPUT"
done

{
  echo
  echo "## Summary"
  echo
  echo "- Total: ${#SUITES[@]}"
  echo "- PASS: $PASS"
  echo "- FAIL: $FAIL"
  echo "- TIMEOUT: $TIMEOUT"
  echo "- GPU-pinned after exit: $PINNED"
  echo
  echo "Per-suite swift-test logs: \`planning/integration-bisect-logs/\`"
  echo
  echo "Next step: for each TIMEOUT or PINNED suite, attach Instruments"
  echo "(\`xcrun xctrace record --template 'Metal System Trace' --attach <pid>\`)"
  echo "during a re-run to identify the in-flight kernel + dispatch shape."
} | tee -a "$OUTPUT"

echo
echo "Wrote: $OUTPUT"

# Non-zero exit if anything was abnormal.
if [[ $FAIL -gt 0 ]] || [[ $TIMEOUT -gt 0 ]] || [[ $PINNED -gt 0 ]]; then
  exit 1
fi
