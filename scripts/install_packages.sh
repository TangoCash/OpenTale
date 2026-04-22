#!/bin/bash

# Script to ensure a virtual environment exists and install packages from requirements.txt

# Exit on any error
set -e

# Default virtual environment directory
VENV_DIR=".venv"

# Function to check if python3 is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 is not installed or not found in PATH"
        echo "On macOS, you can install it with: brew reinstall python3"
        exit 1
    fi
}

# Function to create or activate virtual environment
setup_venv() {
    # Check if virtual environment already exists
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment in $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    else
        echo "Virtual environment already exists in $VENV_DIR"
    fi

    # Activate the virtual environment
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
}

# Function to install packages
install_packages() {
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        echo "Error: requirements.txt not found in the current directory"
        exit 1
    fi

    echo "Ensuring pip is installed in the virtual environment..."
    python3 -m ensurepip --upgrade
    
    echo "Upgrading pip, setuptools, and wheel..."
    python3 -m pip install --upgrade pip setuptools wheel
    
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
}

# Main execution
echo "Starting package installation process..."

# Check for python3
check_python

# Set up and activate virtual environment
setup_venv

# Install packages
install_packages

echo "Package installation completed!"
echo "Current package list:"
pip list

echo "Virtual environment is active. To deactivate, run: deactivate"