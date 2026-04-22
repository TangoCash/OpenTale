#!/bin/bash

# Script to update Python packages and regenerate requirements.txt
# Ensures pip is installed with ensurepip before upgrading

# Exit on any error
set -e

# Source shared package utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/packages_utils.sh"

# Override Python version for this project
PYTHON_VERSION="3.12"

# ============================================================================
# EDIT THESE LISTS TO ADD/REMOVE PACKAGES
# ============================================================================
# Core packages - runtime dependencies needed for the application
CORE_PACKAGES=(
    "flask"
    "python-dotenv"
    "openai"
)

# Dev packages - development/testing dependencies
DEV_PACKAGES=(
    "ruff"
    "pytest"
)
# ============================================================================

# Build combined list for installation and requirements.txt
ALL_PACKAGES=("${CORE_PACKAGES[@]}" "${DEV_PACKAGES[@]}")

# Function to update all packages
update_packages() {
    ensure_pip
    
    echo "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip wheel setuptools
    
    echo ""
    echo "Step 1: Pinning protected packages..."
    pin_protected_packages
             
    echo ""
    echo "Step 2: Installing/upgrading core packages..."
    pip install "${CORE_PACKAGES[@]}"

    echo ""
    echo "Step 3: Installing/upgrading dev packages..."
    pip install "${DEV_PACKAGES[@]}"

    echo ""
    echo "Step 4: Verifying protected packages are still correct..."
    pin_protected_packages

    echo ""
    echo "Step 5: Verifying no dependency conflicts exist..."
    # 'pip check' will explicitly fail the script if any installed package has a broken dependency
    pip check

    echo ""
    echo "Step 6: Regenerating requirements.txt..."
    # Build grep pattern from ALL package names (case-insensitive)
    local grep_pattern=""
    for pkg in "${ALL_PACKAGES[@]}"; do
        if [ -z "$grep_pattern" ]; then
            grep_pattern="^${pkg}=="
        else
            grep_pattern="${grep_pattern}|^${pkg}=="
        fi
    done
    pip freeze --all | grep -iE "$grep_pattern" > requirements.txt
    echo "requirements.txt has been regenerated."
    
    echo ""
    echo "Package update completed!"
}

# Main execution
echo "Starting package update process for AI Book Writer..."

# Navigate to the project root directory
cd "$SCRIPT_DIR/.."

# Check for Python version and set up virtual environment
check_python
setup_venv

# Perform the updates
update_packages

echo ""
echo "========================================================"
echo "  Update Completed Successfully!"
echo "  requirements.txt has been regenerated with current versions"
echo "========================================================"
