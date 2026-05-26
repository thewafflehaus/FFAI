#!/usr/bin/env bash
#
# Install the checked-in hooks under .github/scripts/hooks/ so git fires them
# on the appropriate events. Uses `git config core.hooksPath` rather
# than symlinks into .git/hooks so the install is per-clone but the
# hooks themselves stay version-controlled.
#
# Uninstall: `git config --unset core.hooksPath`

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$REPO_ROOT/.github/scripts/hooks"

if [ ! -d "$HOOK_DIR" ]; then
    echo "✗ .github/scripts/hooks/ not found — run from repo root"
    exit 1
fi

# Mark each hook executable in case the checkout didn't preserve bits.
chmod +x "$HOOK_DIR"/pre-commit "$HOOK_DIR"/commit-msg "$HOOK_DIR"/pre-push

git -C "$REPO_ROOT" config core.hooksPath ".github/scripts/hooks"

echo "✓ Installed hooks (core.hooksPath = .github/scripts/hooks)"
echo
echo "  pre-commit:  make build + make test-unit (~2-3 min)"
echo "  commit-msg:  banned-term + trailer-shape scan (~50ms)"
echo "  pre-push:    make build + make test-unit (~2-3 min)"
echo
echo "  Bypass an individual run with --no-verify."
echo "  Uninstall:   git config --unset core.hooksPath"
