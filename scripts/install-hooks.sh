#!/usr/bin/env bash
#
# Install the checked-in hooks under scripts/hooks/ so git fires them
# on the appropriate events. Uses `git config core.hooksPath` rather
# than symlinks into .git/hooks so the install is per-clone but the
# hooks themselves stay version-controlled.
#
# Wrapped by `make install-hooks` for discoverability.
# Uninstall: `make uninstall-hooks` (or `git config --unset core.hooksPath`).

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$REPO_ROOT/scripts/hooks"

if [ ! -d "$HOOK_DIR" ]; then
    echo "✗ scripts/hooks/ not found — run from repo root"
    exit 1
fi

# Mark each hook executable in case the checkout didn't preserve bits.
chmod +x "$HOOK_DIR"/pre-commit "$HOOK_DIR"/commit-msg "$HOOK_DIR"/pre-push

git -C "$REPO_ROOT" config core.hooksPath "scripts/hooks"

echo "✓ Installed hooks (core.hooksPath = scripts/hooks)"
echo
echo "  pre-commit:  make format-check (~1-3 s)"
echo "  commit-msg:  banned-term + trailer-shape scan (~50ms)"
echo "  pre-push:    make build + make test-unit (~2-3 min)"
echo
echo "  Bypass an individual run with --no-verify."
echo "  Uninstall:   make uninstall-hooks"
