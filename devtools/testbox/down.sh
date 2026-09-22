#!/usr/bin/env bash
# Stop the test box.
#
#   ./down.sh          stop the container, keep config/ (profiles, token, logs)
#   ./down.sh --wipe   stop and delete config/ as well
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

docker compose down --remove-orphans

if [ "${1:-}" = "--wipe" ]; then
  rm -rf config
  echo "config/ removed - the next ./up.sh starts from an empty Home Assistant"
fi
