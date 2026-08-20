#!/bin/bash
# If you encounter "Permission denied", run: chmod +x run_tests.sh
#
# Test runner with categories:
#   ./run_tests.sh           Fast suite (default, skips slow + benchmark)
#   ./run_tests.sh --slow    Only slow tests (real-data replays, stress sims)
#   ./run_tests.sh --bench   Only benchmark tests
#   ./run_tests.sh --e2e     Only Playwright E2E browser tests (requires Node + npx)
#   ./run_tests.sh --e2e-min Same E2E suite, but against the minified build artifacts
#   ./run_tests.sh --all     Everything (fast + slow + benchmark + E2E readable + E2E min)
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

# Playwright E2E runner: 336 tests across chromium + mobile-chrome.
# Skipped if npx is unavailable; fatal on failure when available.
e2e_check() {
    local e2e_dir="playwright-tests"
    if ! command -v npx >/dev/null 2>&1; then
        echo "Warning: npx not found, skipping E2E tests."
        return 0
    fi
    if [ ! -d "$e2e_dir/node_modules" ]; then
        echo "Installing Playwright dependencies..."
        (cd "$e2e_dir" && npm ci --silent) || exit 1
    fi
    echo "Running E2E tests (Playwright, 336 tests across chromium + mobile-chrome)..."
    (cd "$e2e_dir" && npx playwright test "$@") || exit 1
}

# Same E2E suite, but served from the minified *.min.js artifacts instead of the
# readable sources -- i.e. the exact bytes users download. This is what makes a
# broken minified bundle impossible to release: minification is a real transform
# and `node --check` on the source proves nothing about its output.
#
# The build must be current first. A stale artifact would silently test the
# previous release's code and report a meaningless pass, so a failed --check is
# fatal here rather than triggering an implicit rebuild (an auto-rebuild would
# mask the fact that someone forgot, which is precisely what the gate exists to
# catch). serve.mjs picks the artifacts up via PANEL_BUILD=min, and
# playwright.config.ts shifts to port 4568 so reuseExistingServer can never hand
# this run a server that is still serving the readable sources.
e2e_min_check() {
    local e2e_dir="playwright-tests"
    if ! command -v npx >/dev/null 2>&1; then
        echo "Warning: npx not found, skipping minified E2E tests."
        return 0
    fi
    if ! command -v node >/dev/null 2>&1; then
        echo "Warning: node not found, skipping minified E2E tests."
        return 0
    fi
    echo "Verifying panel build is current..."
    node devtools/build_panel.mjs --check || {
        echo "Refusing to run the minified E2E suite against a stale build."
        exit 1
    }
    if [ ! -d "$e2e_dir/node_modules" ]; then
        echo "Installing Playwright dependencies..."
        (cd "$e2e_dir" && npm ci --silent) || exit 1
    fi
    echo "Running E2E tests against MINIFIED build (336 tests)..."
    (cd "$e2e_dir" && PANEL_BUILD=min npx playwright test "$@") || exit 1
}

# First arg may select a category; remaining args pass through to pytest.
mode="${1:-fast}"

case "$mode" in
    --fast|fast)
        [ "$#" -gt 0 ] && shift
        js_check
        echo "Running FAST tests (skipping slow + benchmark)..."
        exec "$VENV_PYTHON" -m pytest tests/ "$@"
        ;;
    --slow|slow)
        [ "$#" -gt 0 ] && shift
        echo "Running SLOW tests only..."
        exec "$VENV_PYTHON" -m pytest tests/ -m slow "$@"
        ;;
    --bench|--benchmark|bench)
        [ "$#" -gt 0 ] && shift
        echo "Running BENCHMARK tests only..."
        exec "$VENV_PYTHON" -m pytest tests/ -m benchmark "$@"
        ;;
    --e2e|e2e)
        [ "$#" -gt 0 ] && shift
        e2e_check "$@"
        ;;
    --e2e-min|e2e-min)
        [ "$#" -gt 0 ] && shift
        e2e_min_check "$@"
        ;;
    --all|all)
        [ "$#" -gt 0 ] && shift
        js_check
        echo "Running ALL tests (fast + slow + benchmark + E2E readable + E2E min)..."
        # Explicit halt on pytest failure so E2E success can never mask a Python
        # failure (belt-and-suspenders on top of `set -e`, since this branch does
        # not `exec` and continues to e2e_check).
        "$VENV_PYTHON" -m pytest tests/ -m "" "$@" || exit 1
        e2e_check
        # Then the same suite against the shipped bytes. Runs last because it is
        # the narrower gate: a failure here with the readable run green means the
        # minifier changed behaviour, not that the panel logic is wrong.
        e2e_min_check
        ;;
    -h|--help)
        sed -n '2,13p' "$0"
        exit 0
        ;;
    *)
        # No mode keyword -> default fast suite, pass all args through.
        js_check
        echo "Running FAST tests (skipping slow + benchmark)..."
        exec "$VENV_PYTHON" -m pytest tests/ "$@"
        ;;
esac
