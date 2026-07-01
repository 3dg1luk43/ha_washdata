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

# Panel JS guard: syntax + a headless render smoke test that instantiates the
# panel and calls every tab/modal renderer, catching load-time (TDZ) and
# template ReferenceErrors that `node --check` alone cannot. Skipped if node is
# unavailable; fatal on failure.
js_check() {
    local panel="custom_components/ha_washdata/www/ha-washdata-panel.js"
    if command -v node >/dev/null 2>&1; then
        echo "Checking panel JS (syntax + render smoke)..."
        node --check "$panel" || exit 1
        [ -f devtools/panel_smoke.js ] && { node devtools/panel_smoke.js || exit 1; }
    fi
}

# First arg may select a category; remaining args pass through to pytest.
mode="${1:-fast}"

case "$mode" in
    --fast|fast)
        shift
        js_check
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
        js_check
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
        js_check
        echo "Running FAST tests (skipping slow + benchmark)..."
        exec "$VENV_PYTHON" -m pytest tests/ "$@"
        ;;
esac
