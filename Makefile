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
regenerate-kernels: ## run `tile emit` to regenerate metallib + Swift wrappers
	@if [ ! -d "$(METALTILE_DIR)" ]; then \
	  echo "Error: metaltile not found at $(METALTILE_DIR)"; \
	  echo "Clone the sibling metaltile repo at ../metaltile."; \
	  exit 1; \
	fi
	@# Run cargo from the metaltile dir so its rust-toolchain.toml (nightly,
	@# 2024 edition) is honored. Running cargo from FFAI/ would use the
	@# system default toolchain, which lacks edition=2024 support.
	@#
	@# `tile emit` writes:
	@#   $(KERNEL_OUT)/Resources/kernels/<name>.metal     per-kernel MSL
	@#   $(KERNEL_OUT)/Resources/kernels.metallib         compiled metallib
	@#   $(KERNEL_OUT)/Resources/manifest.json            IR descriptor
	@#   $(KERNEL_OUT)/Generated/MetalTileKernels.swift   dispatch wrappers
	cd $(METALTILE_DIR) && cargo run --release \
	  --bin tile -- emit --out $(KERNEL_OUT)

# ─── Test ─────────────────────────────────────────────────────────────
#
# Production-parity defaults. The 2026-05-19 GPU-pin root cause —
# wrong-dispatch-shape Ops wrappers — is fixed at the source (see the
# post-mortem in papers/ and OpsValidation in Sources/FFAI/). Test
# runs now use the same FFAI_MAX_COMMAND_BUFFERS=16 cap as production,
# so anything that passes in CI is proven safe under production load.
#
# Defense in depth still in place:
#
#  1. **OpsValidation** preconditions on every reduction-mode wrapper.
#     Catches degenerate dispatch shapes (wrong head_dim, wrong n,
#     etc.) before the kernel ever launches.
#
#  2. **Thread-safe shared state.** PSOCache uses single-flight
#     compilation (compileLock) so parallel suites can't both compile
#     the same PSO. BufferPool uses NSLock. See PSOCache.swift +
#     BufferPool.swift.
#
#  3. **ModelLoadLock** (Tests/ModelTests/ModelLoadLock.swift) — global
#     async mutex around `Model.load(...)`. Different concern from GPU
#     access: model load is heavy on RAM + disk-IO + GPU memory
#     allocation BEFORE any cmdbuf exists. The lock makes Model.load()
#     a global critical section so only one multi-GB checkpoint is
#     loading at a time.
#
# Targets:
# - `make test-unit`         — FFAITests + MetalTileSwiftTests at the
#                              production cap (FFAI_MAX_COMMAND_BUFFERS=16).
# - `make test-integration`  — ModelTests at production cap + ModelLoadLock
#                              + `--parallel --num-workers 1` (memory
#                              pressure, not GPU). Matches release.yml.
# - `make test`              — both in sequence.
# - `make test-stress`       — canary; both suites at production cap with
#                              integration parallelism uncapped. Run after
#                              touching anything dispatch-related to
#                              confirm production safety holds under
#                              maximal parallel load.

.PHONY: test
test: regenerate-kernels test-unit test-integration ## run unit then integration test suites

.PHONY: test-unit
test-unit: regenerate-kernels ## unit + Metal tests at production cap (FFAI_MAX_COMMAND_BUFFERS=16)
	FFAI_MAX_COMMAND_BUFFERS=16 swift test --filter "FFAITests|MetalTileSwiftTests"

.PHONY: test-integration
test-integration: regenerate-kernels ## end-to-end model tests; production cap + ModelLoadLock; matches release.yml
	@# ModelLoadLock (Tests/ModelTests/ModelLoadLock.swift) serializes
	@# Model.load() across suites so multi-GB checkpoints load one at a
	@# time. --num-workers 1 caps Swift Testing's cross-suite parallelism
	@# to one model resident at a time (memory pressure, not GPU).
	FFAI_MAX_COMMAND_BUFFERS=16 swift test --parallel --num-workers 1 --filter "ModelTests"

.PHONY: test-stress
test-stress: regenerate-kernels ## canary; production cap with uncapped parallelism — run after touching dispatch code
	@echo "Stress mode. Running unit + integration at FFAI_MAX_COMMAND_BUFFERS=16"
	@echo "with no --num-workers cap on integration. If anything regresses our"
	@echo "wrapper-precondition / PSOCache / ModelLoadLock defenses, this is"
	@echo "where it surfaces."
	@echo ""
	FFAI_MAX_COMMAND_BUFFERS=16 swift test --filter "FFAITests|MetalTileSwiftTests"
	FFAI_MAX_COMMAND_BUFFERS=16 swift test --filter "ModelTests"

.PHONY: coverage
coverage: ## swift test with coverage report (unit suite only, matches ci.yml)
	FFAI_MAX_COMMAND_BUFFERS=16 ./scripts/coverage.sh

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
