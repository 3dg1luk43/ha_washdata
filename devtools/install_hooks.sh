#!/bin/sh
# Point git at the repo's tracked hooks (devtools/hooks).
#
# `core.hooksPath` rather than copying into .git/hooks: the hooks stay versioned, so a
# fix reaches everyone on the next pull instead of living in one clone. Local setting,
# so it never affects anyone who does not run this.
#
#   ./devtools/install_hooks.sh          install
#   git config --unset core.hooksPath    uninstall
set -eu

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

chmod +x devtools/hooks/* 2>/dev/null || true
git config core.hooksPath devtools/hooks

echo "Installed git hooks from devtools/hooks:"
for h in devtools/hooks/*; do
  [ -f "$h" ] && echo "  - $(basename "$h")"
done
echo
echo "pre-commit refuses a commit whose www/*.min.js were not rebuilt from the"
echo "sources being committed. Bypass a single commit with: git commit --no-verify"
