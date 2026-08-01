#!/usr/bin/env bash
set -e

echo "=== Smart AI Router: Environment Setup ==="
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete! Activate environment with:"
echo "  source .venv/bin/activate"
