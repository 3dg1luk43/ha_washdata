#!/usr/bin/env bash
# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Final verification on one shared core budget (the scheduler is devtools/verify.py).
#
#   devtools/verify.sh                 quick: generated files, docs_check, fast suite, E2E
#   devtools/verify.sh full            + slow suite, E2E (minified), release_check
#                                      --skip-tests, end_gate_eval --check, eval.py gate
#   devtools/verify.sh full --box      + the real-HA test box (never by default)
#   devtools/verify.sh full --cores 4  share 4 cores instead of all of them
#   devtools/verify.sh --dry-run full  the stages, their work estimates, first allocation
#   devtools/verify.sh --list          every stage
#
# Exit status 1 when a stage failed. Logs: one temp dir per run, printed first.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="./.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"
exec "$PY" devtools/verify.py "$@"
