#!/bin/bash
# If you encounter "Permission denied", run: chmod +x run_tests.sh
set -e

# Define path to venv python
VENV_PYTHON="./.venv/bin/python"

# Check if venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at ./.venv"
    exit 1
fi

# Run pytest using the venv's python
echo "Running tests with $VENV_PYTHON -m pytest..."
"$VENV_PYTHON" -m pytest tests/ "$@"
