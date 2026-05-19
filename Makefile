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
# Serialization comes from two independent mechanisms (defense-in-depth):
#
#  1. `.serialized` trait on every @Suite (Tests/{FFAITests,MetalTileSwiftTests,
#     ModelTests}/*.swift). This is the only way to actually disable Swift
#     Testing's in-bundle parallelism — `--parallel/--no-parallel` and
#     `--num-workers` on `swift test` control SwiftPM's bundle-level
#     parallelism, NOT Swift Testing's scheduler. Pre-mitigation experiment:
#     `--parallel --num-workers 1` still spun up ~35 suites concurrently
#     because each @Suite is a Swift Testing scheduling unit, not a SwiftPM
#     bundle. Suite-level `.serialized` is what guarantees one-at-a-time.
#
#  2. `MetalTileLibrary.defaultMaxCommandBufferCount` (Sources/MetalTileSwift/
#     MetalTileLibrary.swift) caps the shared command queue at 16 in-flight
#     command buffers via `makeCommandQueue(maxCommandBufferCount:)`. Belt-
#     and-suspenders: even if anything bypasses suite-level serialization
#     (production code, opt-in parallel tests, agent code, future test
#     framework that forgets the trait), Metal applies backpressure before
#     the queue saturates. Pre-mitigation observation: parallel unit tests
#     piled hundreds of cmdbufs in flight, starved the WindowServer of GPU
#     time, and (twice) crashed WindowServer → system freeze → hard reboot.
#     Override at runtime via `FFAI_MAX_COMMAND_BUFFERS=N` env var.
#
#  3. `ModelLoadLock` (Tests/ModelTests/ModelLoadLock.swift) is a global
#     async mutex around `Model.load(...)`. Necessary because `.serialized`
#     is per-suite — multiple ModelTests suites can still race their
#     model loads concurrently, spiking RAM + GPU memory + download IO.
#     The lock makes Model.load() a global critical section so only one
#     multi-GB checkpoint is loading at a time across the bundle.
#
# Why a `test-unit-parallel` opt-in still exists: to validate the
# mitigations actually work (run the parallel version, confirm no freeze)
# and to triage future regressions. Don't run casually.
#
# Targets:
# - `make test-unit`           — FFAITests + MetalTileSwiftTests. Each
#                                suite is `.serialized`; no flag gymnastics.
# - `make test-unit-parallel`  — bypasses serialization via `--parallel`.
#                                OPT-IN; triage only.
# - `make test-integration`    — ModelTests. Each suite is `.serialized`;
#                                also passes `--parallel --num-workers 1`
#                                to cap SwiftPM bundle workers (matches
#                                release.yml).
# - `make test`                — both in sequence.

.PHONY: test
test: regenerate-kernels test-unit test-integration ## run unit then integration test suites (serialized)

.PHONY: test-unit
test-unit: regenerate-kernels ## unit + Metal tests, serialized via @Suite traits — safe local default
	@# Suite-level `.serialized` traits do the actual serialization (Swift
	@# Testing parallelizes across suites by default — `--num-workers` is
	@# SwiftPM's bundle-level knob, not Swift Testing's scheduler knob).
	@# MetalTileLibrary.defaultMaxCommandBufferCount caps the shared Metal
	@# queue at 32 in-flight cmdbufs as defense-in-depth.
	swift test --filter "FFAITests|MetalTileSwiftTests"

.PHONY: test-unit-parallel
test-unit-parallel: regenerate-kernels ## OPT-IN: triage parallel behavior; --parallel can override .serialized
	@echo "⚠️  Triage mode. Parallel test runs have frozen the box on this hardware."
	@echo "   Only use when validating GPU-queue / buffer-pool mitigations."
	@echo ""
	swift test --filter "FFAITests|MetalTileSwiftTests" --parallel

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
