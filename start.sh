#!/bin/bash
#
# Start script for OpenTale (Linux/macOS)
# ========================================
# Creates a virtual environment if it doesn't exist,
# installs dependencies, then runs the web app.
#

set -e

VENV_DIR=".venv"

# Check if the virtual environment exists, create if not.
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Install/update dependencies.
echo "Installing dependencies..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r requirements.txt

# Activate the virtual environment and run the app.
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Running web_app.py..."
python web_app.py