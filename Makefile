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
# Three layers prevent the parallel-test GPU pile-up that crashed the
# WindowServer pre-mitigation:
#
#  1. **`FFAI_MAX_COMMAND_BUFFERS=1` for `make test-unit`.** Forces the
#     shared MTLCommandQueue's max-in-flight depth to 1, which means
#     Metal blocks the 2nd concurrent `makeCommandBuffer()` caller
#     until the 1st cmdbuf completes. Because every GPU-touching test
#     calls `cmd.waitUntilCompleted()` before returning, this gives us
#     ACTUAL global GPU-access serialization across parallel suites
#     without writing any async-lock plumbing. The cap-of-1 only
#     affects this test invocation; production keeps the default 16.
#
#     Why this works where `.serialized` (per-suite trait) doesn't:
#     `.serialized` only orders tests WITHIN a suite. Swift Testing
#     still runs different @Suite types concurrently, and there's no
#     CLI flag that disables that. The Metal-layer cap doesn't care
#     who the callers are — it sees `makeCommandBuffer()` calls and
#     blocks the surplus.
#
#  2. **Thread-safe shared state.** PSOCache uses single-flight
#     compilation (compileLock) so two parallel suites can't both
#     compile the same PSO and produce a corrupted pipeline. BufferPool
#     uses NSLock. See PSOCache.swift + BufferPool.swift for the
#     specifics.
#
#  3. **ModelLoadLock** (Tests/ModelTests/ModelLoadLock.swift) — global
#     async mutex around `Model.load(...)`. Different concern from
#     GPU access: model load is heavy on RAM + disk-IO + GPU memory
#     allocation BEFORE any cmdbuf exists, so the queue cap doesn't
#     apply. The lock makes Model.load() a global critical section so
#     only one multi-GB checkpoint is loading at a time.
#
# Pure-Swift suites (no GPU) can run in parallel — the queue cap and
# locks only matter when something actually dispatches. The Makefile
# doesn't try to separate them; we let Swift Testing's default
# scheduler do its thing and rely on the layers above to keep
# GPU-touching parallel runs safe.
#
# Targets:
# - `make test-unit`           — FFAITests + MetalTileSwiftTests with
#                                FFAI_MAX_COMMAND_BUFFERS=1.
# - `make test-unit-parallel`  — OPT-IN: drops the cap-of-1 to repro
#                                pre-mitigation behavior for triage.
# - `make test-integration`    — ModelTests with FFAI_MAX_COMMAND_BUFFERS=1
#                                + ModelLoadLock. Matches release.yml.
# - `make test`                — both in sequence.

.PHONY: test
test: regenerate-kernels test-unit test-integration ## run unit then integration test suites

.PHONY: test-unit
test-unit: regenerate-kernels ## unit + Metal tests; queue cap 1 forces serial GPU access
	FFAI_MAX_COMMAND_BUFFERS=1 swift test --filter "FFAITests|MetalTileSwiftTests"

.PHONY: test-unit-parallel
test-unit-parallel: regenerate-kernels ## OPT-IN: triage; drops cap-of-1, hits pre-mitigation races
	@echo "⚠️  Triage mode. Drops the queue-cap-of-1 guard. Reproduces the"
	@echo "   PSOCache compile race / WindowServer starvation pre-mitigation."
	@echo "   Use only when validating that the cap-of-1 in 'make test-unit'"
	@echo "   is still required."
	@echo ""
	swift test --filter "FFAITests|MetalTileSwiftTests"

.PHONY: test-integration
test-integration: regenerate-kernels ## end-to-end model tests; queue cap 1 + ModelLoadLock; matches release.yml
	@# Queue cap of 1 forces serial GPU dispatch across parallel suites.
	@# ModelLoadLock (Tests/ModelTests/ModelLoadLock.swift) separately
	@# serializes Model.load() across suites so multi-GB checkpoints
	@# load one at a time.
	FFAI_MAX_COMMAND_BUFFERS=1 swift test --filter "ModelTests"

.PHONY: coverage
coverage: ## swift test with coverage report (unit suite only, matches ci.yml)
	FFAI_MAX_COMMAND_BUFFERS=1 ./scripts/coverage.sh

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
