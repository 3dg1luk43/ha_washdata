#!/bin/bash
# If you encounter "Permission denied", run: chmod +x run_tests.sh
#
# Test runner with categories:
#   ./run_tests.sh           Fast suite (default, skips slow + benchmark)
#   ./run_tests.sh --slow    Only slow tests (real-data replays, stress sims)
#   ./run_tests.sh --bench   Only benchmark tests
#   ./run_tests.sh --all     Everything
#   ./run_tests.sh <pytest-args>  Pass through any other args
#
# Categories live in pytest.ini under `markers` and the default `-m` filter.
set -e

VENV_PYTHON="./.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at ./.venv"
    exit 1
fi

# First arg may select a category; remaining args pass through to pytest.
mode="${1:-fast}"

case "$mode" in
    --fast|fast)
        shift
        echo "Running FAST tests (skipping slow + benchmark)..."
        exec "$VENV_PYTHON" -m pytest tests/ "$@"
        ;;
    --slow|slow)
        shift
        echo "Running SLOW tests only..."
        exec "$VENV_PYTHON" -m pytest tests/ -m slow "$@"
        ;;
    --bench|--benchmark|bench)
        shift
        echo "Running BENCHMARK tests only..."
        exec "$VENV_PYTHON" -m pytest tests/ -m benchmark "$@"
        ;;
    --all|all)
        shift
        echo "Running ALL tests (fast + slow + benchmark)..."
        # Override the pytest.ini default -m filter.
        exec "$VENV_PYTHON" -m pytest tests/ -m "" "$@"
        ;;
    -h|--help)
        sed -n '2,11p' "$0"
        exit 0
        ;;
    *)
        # No mode keyword -> default fast suite, pass all args through.
        echo "Running FAST tests (skipping slow + benchmark)..."
        exec "$VENV_PYTHON" -m pytest tests/ "$@"
        ;;
esac
