#!/bin/bash
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
# Sibling metaltile checkout
# ─────────────────────────────────────────────
echo ""
echo "Checking sibling metaltile repo..."
if [ ! -d "$METALTILE_DIR" ]; then
    warn "metaltile not found at $METALTILE_DIR"
    echo ""
    echo "Clone it with:"
    echo "  cd $(dirname "$PROJECT_ROOT") && git clone https://github.com/ekryski/metaltile"
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
echo "  make test       # run all tests"
echo "  make coverage   # tests + coverage report"
echo "  make build      # rebuild (regenerates kernels)"
echo "  make format     # swift-format the repo"
echo "  make docs       # verify docs build"
echo ""
