#!/bin/bash
# Lint and format check script using ruff
# This script checks and auto-fixes ruff issues in the codebase

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Running Ruff Lint Check"
echo "========================================"

# Check ruff version
ruff --version

echo ""
echo "========================================"
echo "Running Ruff Format Check"
echo "========================================"

# Check if files need formatting (will fail if not formatted)
if ! ruff format --check . 2>&1; then
    echo ""
    echo "Formatting issues detected. Running ruff format..."
    ruff format .
    echo "Formatting applied."
else
    echo "Code is properly formatted."
fi

echo ""
echo "========================================"
echo "Running Ruff Lint Check"
echo "========================================"

# Run ruff check with auto-fix
if ruff check . --fix 2>&1; then
    echo ""
    echo "No lint issues found."
else
    echo ""
    echo "Lint issues found and fixed where possible."
    echo "Please review any remaining issues above."
fi

echo ""
echo "========================================"
echo "Skipping Pyright Type Check"
echo "========================================"
echo "Note: Pyright strict type checking for pandas/numpy often produces false positives."
echo "      Use 'pyright' directly if you need strict type checking."

echo ""
echo "========================================"
echo "Checking for remaining issues..."
echo "========================================"

# Final check - show any remaining issues (non-fatal)
REMAINING_ISSUES=$(ruff check . 2>&1 || true)
if [ -n "$REMAINING_ISSUES" ]; then
    echo "$REMAINING_ISSUES"
    echo ""
    echo "NOTE: Some issues could not be auto-fixed."
else
    echo "All ruff checks passed!"
fi

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
