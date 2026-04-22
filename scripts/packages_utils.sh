#!/bin/bash

# Shared Python utility functions for install_packages.sh and update_packages.sh
# This file contains common functions for Python version management and virtual environment setup

# Default virtual environment directory
VENV_DIR=".venv"

# Python version to use
PYTHON_VERSION="3.12"

# Function to check if required Python version is available
check_python() {
    # Check if python${PYTHON_VERSION} is available via Homebrew
    if command -v /opt/homebrew/bin/python${PYTHON_VERSION} &> /dev/null; then
        PYTHON_CMD="/opt/homebrew/bin/python${PYTHON_VERSION}"
        echo "Using Python $PYTHON_VERSION from Homebrew"
    elif command -v /usr/local/bin/python${PYTHON_VERSION} &> /dev/null; then
        PYTHON_CMD="/usr/local/bin/python${PYTHON_VERSION}"
        echo "Using Python $PYTHON_VERSION from Homebrew"
    elif command -v python${PYTHON_VERSION} &> /dev/null; then
        PYTHON_CMD="python${PYTHON_VERSION}"
        echo "Using Python $PYTHON_VERSION"
    else
        echo "Python $PYTHON_VERSION not found."
        echo "Installing Python $PYTHON_VERSION via Homebrew..."
        brew install python@${PYTHON_VERSION}
        
        # Add to PATH for this session
        if [ -f "/opt/homebrew/bin/python${PYTHON_VERSION}" ]; then
            PYTHON_CMD="/opt/homebrew/bin/python${PYTHON_VERSION}"
        elif [ -f "/usr/local/bin/python${PYTHON_VERSION}" ]; then
            PYTHON_CMD="/usr/local/bin/python${PYTHON_VERSION}"
        else
            echo "Error: Failed to install Python $PYTHON_VERSION"
            exit 1
        fi
        echo "Successfully installed Python $PYTHON_VERSION"
    fi
}

# Function to create or activate virtual environment
setup_venv() {
    # Check if virtual environment already exists
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment in $VENV_DIR using $PYTHON_CMD..."
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    else
        echo "Virtual environment already exists in $VENV_DIR"
    fi

    # Activate the virtual environment for this script execution
    echo "Activating virtual environment..."
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    
    # Verify we are using the venv python
    CURRENT_PYTHON=$(which python)
    if [[ "$CURRENT_PYTHON" != *"$VENV_DIR"* ]]; then
        echo "Error: Failed to activate virtual environment."
        exit 1
    fi
}

# Function to ensure pip is available and upgraded
# This function must be called AFTER the venv is activated
ensure_pip() {
    echo "Ensuring pip is installed in the virtual environment..."
    # Use the activated venv's python -m pip (not the system Python)
    python -m ensurepip --upgrade || true
    
    echo "Upgrading pip, setuptools, and wheel..."
    python -m pip install --upgrade pip wheel setuptools
}

# Protected packages array
# These packages must be pinned to specific versions because:
# 1. Flask and Werkzeug must match Airflow's constraints (2.2.5 and 2.2.3 for Airflow 3.1.8)
# 2. cryptography is pinned to match mlflow's requirements (46.0.6)
PROTECTED_PACKAGES=(
    # package==version number
)

# Function to pin protected packages
pin_protected_packages() {
    if [ ${#PROTECTED_PACKAGES[@]} -eq 0 ]; then
        echo "No protected packages to pin."
        return 0
    fi
    
    echo "Pinning protected packages..."
    for pkg in "${PROTECTED_PACKAGES[@]}"; do
        echo "  Pinning $pkg..."
        # Use --force-reinstall to ensure the specific version is installed
        # even if a newer version was already present.
        pip install "$pkg" --force-reinstall --quiet
    done
}
