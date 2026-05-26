#!/bin/bash
#
# Copyright 2026 Eric Kryski (@ekryski)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# setup-dev.sh
# One-time development environment setup for FFAI.
# Verifies toolchains, resolves dependencies, runs first build.
#
#   ./scripts/setup-dev.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
METALTILE_DIR="$PROJECT_ROOT/../metaltile"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}  ✓${NC} $1"; }
warn() { echo -e "${YELLOW}  ⚠${NC}  $1"; }
fail() { echo -e "${RED}  ✗${NC} $1"; exit 1; }

echo ""
echo "Setting up FFAI development environment..."
echo ""

# ─────────────────────────────────────────────
# Prerequisites
# ─────────────────────────────────────────────
echo "Checking prerequisites..."

if ! xcode-select -p &>/dev/null; then
    fail "Xcode Command Line Tools not found. Install with: xcode-select --install"
fi
ok "Xcode CLI tools: $(xcode-select -p)"

if ! xcrun --find metal &>/dev/null 2>&1; then
    fail "xcrun metal not found. Install Xcode (full IDE, not just CLI tools)."
fi
ok "xcrun metal: available"

if ! command -v swift &>/dev/null; then
    fail "Swift not found. Install Xcode or a Swift toolchain."
fi
SWIFT_VERSION=$(swift --version 2>&1 | grep -oE 'Swift version [0-9]+\.[0-9]+' | head -1)
ok "Swift: $SWIFT_VERSION"

if ! command -v cargo &>/dev/null; then
    fail "Cargo not found. Install Rust via https://rustup.rs/"
fi
ok "Cargo: $(cargo --version)"

# ─────────────────────────────────────────────
# Lint / format tooling
# ─────────────────────────────────────────────
# swift-format powers `make format` / `make format-check` (also the
# pre-commit git hook installed by `make install-hooks`). On macOS we
# prefer the toolchain-bundled binary if Xcode ships one; otherwise fall
# back to Homebrew. If neither path resolves, install via brew now so
# the dev loop is unblocked end-to-end on a fresh machine.
echo ""
echo "Checking lint / format tooling..."
if command -v swift-format &>/dev/null; then
    ok "swift-format: $(swift-format --version 2>&1 | head -1)"
elif xcrun --find swift-format &>/dev/null 2>&1; then
    ok "swift-format: $(xcrun --find swift-format) (Xcode toolchain)"
else
    warn "swift-format not found"
    if command -v brew &>/dev/null; then
        echo "  Installing via Homebrew..."
        brew install swift-format
        ok "swift-format: $(swift-format --version 2>&1 | head -1)"
    else
        fail "Install swift-format manually: brew install swift-format"
    fi
fi

# ─────────────────────────────────────────────
# Sibling metaltile checkout
# ─────────────────────────────────────────────
echo ""
echo "Checking sibling metaltile repo..."
if [ ! -d "$METALTILE_DIR" ]; then
    warn "metaltile not found at $METALTILE_DIR"
    echo ""
    echo "Clone it with:"
    echo "  cd $(dirname "$PROJECT_ROOT") && git clone https://github.com/thewafflehaus/metaltile"
    echo ""
    fail "metaltile checkout required for kernel generation"
fi
ok "metaltile: $METALTILE_DIR"

# ─────────────────────────────────────────────
# Resolve packages
# ─────────────────────────────────────────────
echo ""
echo "Resolving Swift package dependencies..."
cd "$PROJECT_ROOT"
swift package resolve
ok "Packages resolved"

# ─────────────────────────────────────────────
# Build (kernels + Swift)
# ─────────────────────────────────────────────
echo ""
echo "Building (make build)..."
make -C "$PROJECT_ROOT" build
ok "Build complete"

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Common targets:"
echo "  make test           # run all tests"
echo "  make coverage       # tests + coverage report"
echo "  make build          # rebuild (regenerates kernels)"
echo "  make format         # swift-format the repo"
echo "  make format-check   # swift-format lint (no writes)"
echo "  make install-hooks  # install pre-commit / commit-msg / pre-push"
echo "  make docs           # verify docs build"
echo ""
