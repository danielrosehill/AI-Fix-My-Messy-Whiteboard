#!/usr/bin/env bash
#
# Whiteboard Processor Runner
# Handles venv setup and runs the CLI tool
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR/app"
VENV_DIR="$APP_DIR/.venv"
REQUIREMENTS="$APP_DIR/requirements.txt"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    uv venv "$VENV_DIR"
    echo "Installing dependencies..."
    uv pip install --python "$VENV_DIR/bin/python" -r "$REQUIREMENTS"
    echo ""
fi

# Check if dependencies need updating (requirements.txt newer than venv)
if [ "$REQUIREMENTS" -nt "$VENV_DIR" ]; then
    echo "Updating dependencies..."
    uv pip install --python "$VENV_DIR/bin/python" -r "$REQUIREMENTS"
    touch "$VENV_DIR"
    echo ""
fi

# Run the processor
exec "$VENV_DIR/bin/python" "$APP_DIR/process_whiteboard.py" "$@"
