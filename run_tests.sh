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
#   ./run_tests.sh [mode] --serial  One process (no pytest-xdist)
#
# Parallel by default: the pytest modes run on pytest-xdist (installed with
# pytest-homeassistant-custom-component), one worker per core, at most 8
# (PYTEST_XDIST_AUTO_NUM_WORKERS overrides the count). --serial opts out; an
# explicit -n / --numprocesses is passed through as given. The fast suite uses
# --dist load (tests go out in small batches as workers free up; worksteal measured
# no faster, 28 vs 30 s on 4 workers). The slow tier uses --dist loadgroup: the
# suggestion-loop corpus run is one xdist_group per device, so each device runs once
# on one worker, and tests marked `heavy` go out first (tests/conftest.py). Measured
# 2026-10-05 on 4 cores: slow 384 s serial, 302 s load, 269 s loadgroup. A run naming
# test files or node ids stays serial: worker start-up costs more there.
# For the whole verification (static checks, both suites, both E2E runs, the
# harness gates) on one shared core budget, see devtools/verify.sh.
#
# Categories live in pytest.ini under `markers` and the default `-m` filter.
# Per-test timeouts: 60 s (pytest.ini), 1800 s for tests marked slow or benchmark
# (tests/conftest.py), whichever mode runs them. Skips print with their reason,
# and skips of the gitignored cycle_data/ corpus are counted on one line.
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

# Playwright E2E runner: every spec on chromium + mobile-chrome.
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
    echo "Running E2E tests (Playwright, chromium + mobile-chrome)..."
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
    # build_panel.mjs needs its own dependencies (esbuild); without them the
    # --check below fails on a fresh clone before a single test has run.
    if [ ! -d "devtools/node_modules" ]; then
        echo "Installing panel build dependencies..."
        npm ci --prefix devtools --silent || exit 1
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
    echo "Running E2E tests against the MINIFIED build..."
    (cd "$e2e_dir" && PANEL_BUILD=min npx playwright test "$@") || exit 1
}

# pytest-xdist arguments for one run: XDIST=(...) for the given --dist mode, or
# nothing when --serial was given, the caller chose -n itself, xdist is missing,
# or the run targets specific test files / node ids. Reads and rewrites ARGS.
XDIST=()
xdist_args() {
    local dist="$1" a prev="" serial=0 nset="" distset=0 targeted=0 noplugin=0 kept=()
    for a in "${ARGS[@]}"; do
        case "$a" in
            --serial) serial=1; continue ;;
            -n|--numprocesses) nset="?" ;;
            -n*) nset="${a#-n}" ;;
            --numprocesses=*) nset="${a#*=}" ;;
            --dist|--dist=*) distset=1 ;;
            no:xdist|-pno:xdist) noplugin=1 ;;
            -*) ;;
            *)
                if [ "$prev" = "-n" ] || [ "$prev" = "--numprocesses" ]; then nset="$a"
                elif [[ "$a" == *::* || -e "${a%%::*}" ]]; then targeted=1; fi ;;
        esac
        prev="$a"
        kept+=("$a")
    done
    ARGS=("${kept[@]}")
    XDIST=()
    [ "$noplugin" -eq 1 ] && return 0
    "$VENV_PYTHON" -c "import xdist" 2>/dev/null || return 0
    if [ "$serial" -eq 1 ] || { [ "$targeted" -eq 1 ] && [ -z "$nset" ]; }; then
        XDIST=(-n 0)
        return 0
    fi
    # An explicit worker count keeps this tier's --dist unless that was given too.
    if [ -n "$nset" ]; then
        [ "$distset" -eq 0 ] && [ "$nset" != "0" ] && XDIST=(--dist "$dist")
        return 0
    fi
    [ "$distset" -eq 1 ] && return 0
    local n="${PYTEST_XDIST_AUTO_NUM_WORKERS:-$(nproc 2>/dev/null || echo 1)}"
    [ "$n" -gt 8 ] && n=8
    if [ "$n" -le 1 ]; then XDIST=(-n 0); else XDIST=(-n "$n" --dist "$dist"); fi
}

# First arg may select a category; remaining args pass through to pytest.
mode="${1:-fast}"

case "$mode" in
    --fast|fast)
        [ "$#" -gt 0 ] && shift
        ARGS=("$@"); xdist_args load
        js_check
        echo "Running FAST tests (skipping slow + benchmark)..."
        exec "$VENV_PYTHON" -m pytest tests/ "${XDIST[@]}" "${ARGS[@]}"
        ;;
    --slow|slow)
        [ "$#" -gt 0 ] && shift
        ARGS=("$@"); xdist_args loadgroup
        echo "Running SLOW tests only..."
        exec "$VENV_PYTHON" -m pytest tests/ -m slow "${XDIST[@]}" "${ARGS[@]}"
        ;;
    --bench|--benchmark|bench)
        [ "$#" -gt 0 ] && shift
        echo "Running BENCHMARK tests only..."
        rc=0
        "$VENV_PYTHON" -m pytest tests/ -m benchmark "$@" || rc=$?
        # Exit 5 = nothing collected: the timing benchmarks were replaced by
        # deterministic work budgets that run in the fast suite (audit PERF-08).
        if [ "$rc" -eq 5 ]; then
            echo "No benchmark tests; work budgets run in the fast suite (tests/test_perf_budgets.py)."
            exit 0
        fi
        exit "$rc"
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
        ARGS=("$@"); xdist_args loadgroup
        "$VENV_PYTHON" -m pytest tests/ -m "" "${XDIST[@]}" "${ARGS[@]}" || exit 1
        e2e_check
        # Then the same suite against the shipped bytes. Runs last because it is
        # the narrower gate: a failure here with the readable run green means the
        # minifier changed behaviour, not that the panel logic is wrong.
        e2e_min_check
        ;;
    -h|--help)
        sed -n '2,30p' "$0"
        exit 0
        ;;
    *)
        # No mode keyword -> default fast suite, pass all args through.
        ARGS=("$@"); xdist_args load
        js_check
        echo "Running FAST tests (skipping slow + benchmark)..."
        exec "$VENV_PYTHON" -m pytest tests/ "${XDIST[@]}" "${ARGS[@]}"
        ;;
esac
