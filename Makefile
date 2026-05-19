# FFAI — Makefile
#
# Common dev-loop targets. See planning/plan.md for the phased
# build-out and scripts/ for the longer-form scripts.

.DEFAULT_GOAL := help

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT := $(shell pwd)
METALTILE_DIR := $(PROJECT_ROOT)/../metaltile
KERNEL_OUT := $(PROJECT_ROOT)/Sources/MetalTileSwift

# ─── Help ─────────────────────────────────────────────────────────────
.PHONY: help
help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────
.PHONY: setup
setup: ## one-time dev environment setup (toolchains, deps, first build)
	./scripts/setup-dev.sh

# ─── Build ────────────────────────────────────────────────────────────
.PHONY: build
build: regenerate-kernels ## swift build (debug)
	swift build

.PHONY: build-release
build-release: regenerate-kernels ## swift build (release)
	swift build -c release

.PHONY: regenerate-kernels
regenerate-kernels: ## run `tile build --emit all` to regenerate metallib + Swift wrappers
	@if [ ! -d "$(METALTILE_DIR)" ]; then \
	  echo "Error: metaltile not found at $(METALTILE_DIR)"; \
	  echo "Clone the sibling metaltile repo at ../metaltile."; \
	  exit 1; \
	fi
	@# Run cargo from the metaltile dir so its rust-toolchain.toml (nightly,
	@# 2024 edition) is honored. Running cargo from FFAI/ would use the
	@# system default toolchain, which lacks edition=2024 support.
	@#
	@# `tile build --emit all` writes:
	@#   $(KERNEL_OUT)/Resources/kernels/<name>.metal     per-kernel MSL
	@#   $(KERNEL_OUT)/Resources/kernels.metallib         compiled metallib
	@#   $(KERNEL_OUT)/Resources/manifest.json            IR descriptor
	@#   $(KERNEL_OUT)/Generated/MetalTileKernels.swift   dispatch wrappers
	cd $(METALTILE_DIR) && cargo run --release \
	  --bin tile -- build --emit all --out $(KERNEL_OUT)

# ─── Test ─────────────────────────────────────────────────────────────
#
# Test execution defaults to serialized (`--parallel --num-workers 1`)
# for **both** the unit and integration suites. Backstory:
#
#   ModelTests download multi-GB HuggingFace snapshots and do real
#   end-to-end inference per suite. CI for `make test-integration` has
#   always serialized those.
#
#   `make test-unit` was originally parallel (matching ci.yml), but a
#   local run on Apple Silicon froze the entire box — GPU pinned, UI
#   unresponsive, hard recovery only. FFAITests + MetalTileSwiftTests
#   each run real Metal dispatches; Swift Testing parallelizes across
#   suites, and ~30 suites × multiple GPU tests each appears to
#   saturate the GPU command queue and starve the WindowServer.
#
#   Until we diagnose root cause (memory leak in our Tensor / buffer
#   pool? PSO compile thundering herd? specific kernel watchdog?
#   ARC release timing under parallel Swift Testing?), we run unit
#   tests **serialized too**. Cost: unit suite is slower locally; the
#   freeze risk is gone. CI is unchanged — ci.yml still runs parallel
#   because the GitHub-hosted M-series runners haven't reproduced the
#   freeze (fewer cores → less effective parallelism → less GPU queue
#   pressure).
#
# Targets:
# - `make test-unit`            — FFAITests + MetalTileSwiftTests,
#                                 serialized (safe default).
# - `make test-unit-parallel`   — same suite but parallel. Opt-in;
#                                 risks the freeze on the affected
#                                 hardware. Useful when triaging.
# - `make test-integration`     — ModelTests, serialized.
# - `make test`                 — both in sequence.

.PHONY: test
test: regenerate-kernels test-unit test-integration ## run unit then integration test suites (serialized)

.PHONY: test-unit
test-unit: regenerate-kernels ## unit + Metal tests, serialized (--num-workers 1) — safe local default
	@# Serialized because parallel default has frozen Apple Silicon
	@# locally on this branch. See header comment + planning notes for
	@# the diagnostic plan. Swift PM rejects `--num-workers` without
	@# `--parallel`, so we use the combo.
	swift test --filter "FFAITests|MetalTileSwiftTests" --parallel --num-workers 1

.PHONY: test-unit-parallel
test-unit-parallel: regenerate-kernels ## OPT-IN unit + Metal tests, parallel — may freeze on some hardware
	@echo "⚠️  Parallel unit tests have frozen the box on this branch."
	@echo "   Use only when diagnosing GPU-queue / buffer-pool contention."
	@echo ""
	swift test --filter "FFAITests|MetalTileSwiftTests"

.PHONY: test-integration
test-integration: regenerate-kernels ## end-to-end model tests, serialized (--num-workers 1); matches release.yml
	@# Swift PM rejects `--num-workers` without `--parallel`. The combo
	@# enables the scheduler but caps it at one worker, which means
	@# suites run effectively serially — only one model resident at a
	@# time. Matches the "Run integration tests (serialized)" step in
	@# release.yml exactly.
	swift test --filter "ModelTests" --parallel --num-workers 1

.PHONY: coverage
coverage: ## swift test with coverage report (unit suite only, matches ci.yml)
	./scripts/coverage.sh

# ─── Lint / format ────────────────────────────────────────────────────
.PHONY: format
format: ## run swift-format on all .swift files
	swift-format format --in-place --configuration .swift-format --recursive .

.PHONY: format-check
format-check: ## check formatting without modifying files
	swift-format lint --configuration .swift-format --recursive . && echo "format OK"

# ─── Docs ─────────────────────────────────────────────────────────────
# User-facing documentation lives at https://ffai.dev
# (source: github.com/thewafflehaus/ffai-website).
#
# The website fetches markdown from this repo at build time, so committing
# changes to documentation/, README.md, planning/architecture.md, or
# planning/roadmap.md on `main` triggers a rebuild via a GitHub Action.

WEBSITE_DIR := $(PROJECT_ROOT)/../ffai-website

.PHONY: docs
docs: ## verify markdown + preview the docs site locally (if ../ffai-website is checked out)
	./scripts/verify-docs.sh
	@if [ -d "$(WEBSITE_DIR)" ]; then \
	  echo ""; \
	  echo "Preview the docs site (Ctrl+C to stop):"; \
	  echo "  cd $(WEBSITE_DIR) && pnpm dev"; \
	  echo ""; \
	  echo "Or to build a one-shot static preview:"; \
	  echo "  cd $(WEBSITE_DIR) && pnpm build && pnpm dlx serve dist"; \
	else \
	  echo ""; \
	  echo "Tip: clone the docs site to preview locally:"; \
	  echo "  git clone https://github.com/thewafflehaus/ffai-website $(WEBSITE_DIR)"; \
	fi

.PHONY: docs-verify
docs-verify: ## swift-docc target-by-target verification only (no website preview)
	./scripts/verify-docs.sh

# ─── Clean ────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## remove .build and generated kernel artifacts
	rm -rf .build
	rm -f Sources/MetalTileSwift/Resources/kernels.metallib
	rm -f Sources/MetalTileSwift/Resources/manifest.json
	rm -rf Sources/MetalTileSwift/Resources/kernels
	rm -rf Sources/MetalTileSwift/Generated/MetalTileKernels.swift
