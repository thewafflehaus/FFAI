#!/bin/bash
# coverage.sh
# Run unit tests with coverage and print a summary.
# Excludes generated code, .build, and Tests themselves from the report.
#
# Scope is intentionally limited to the unit suite (FFAITests +
# MetalTileSwiftTests). ModelIntegrationTests download multi-GB HuggingFace
# snapshots + do real end-to-end inference, which:
#   (a) takes tens of minutes per run
#   (b) hits memory contention if multiple suites parallelize
#   (c) doesn't meaningfully contribute to *unit* coverage anyway
# Matches .github/workflows/ci.yml's coverage step exactly. For
# integration tests, see `make test-integration`.

set -e

cd "$(dirname "$0")/.."

echo "Running unit tests with coverage enabled..."
swift test --enable-code-coverage --filter "FFAITests|MetalTileSwiftTests"

BIN_PATH=$(swift build --show-bin-path)
PROF_DATA="$BIN_PATH/codecov/default.profdata"

if [ ! -f "$PROF_DATA" ]; then
    echo "No coverage data at $PROF_DATA. Did the tests run?"
    exit 1
fi

# Find the test bundle. SPM names it <Package>PackageTests.xctest
TEST_BUNDLE=$(find "$BIN_PATH" -name "*PackageTests.xctest" -type d | head -1)
if [ -z "$TEST_BUNDLE" ]; then
    echo "No test bundle found in $BIN_PATH"
    exit 1
fi
TEST_BIN="$TEST_BUNDLE/Contents/MacOS/$(basename "$TEST_BUNDLE" .xctest)"

echo ""
echo "Coverage report (excluding .build, Tests, Generated):"
echo ""
xcrun llvm-cov report \
    "$TEST_BIN" \
    -instr-profile="$PROF_DATA" \
    -ignore-filename-regex='(\.build|Tests|Generated)'
