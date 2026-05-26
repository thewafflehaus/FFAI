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
# verify-docs.sh
# Verify documentation builds without warnings, target by target.
# Adapted from mlx-swift-lm.

set -e

cd "$(dirname "$0")/.."

# Discover library product targets from Package.swift, skipping
# test/macro/executable targets.
TARGETS=$(swift package dump-package | python3 -c "
import json, sys
pkg = json.load(sys.stdin)
targets = set()
for p in pkg['products']:
    if p['type'].get('library') is not None:
        targets.update(p['targets'])
for t in sorted(targets):
    print(t)
")

if [ -z "$TARGETS" ]; then
    echo "No library targets found."
    exit 1
fi

FAILED=0

while IFS= read -r TARGET; do
    echo "Building documentation for $TARGET..."
    if ! swift package generate-documentation --target "$TARGET" --warnings-as-errors; then
        FAILED=1
    fi
    echo ""
done <<< "$TARGETS"

if [ "$FAILED" -ne 0 ]; then
    echo "Documentation build failed with warnings."
    exit 1
fi

echo "All documentation builds passed."
