#!/usr/bin/env bash
# Bring up the WashData test box and leave it ready to drive.
#
#   ./up.sh            start (keeps any existing config/)
#   ./up.sh --fresh    wipe config/ first, so storage migrations run from zero
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV_PY="$REPO/.venv/bin/python3"
FRESH=0
[ "${1:-}" = "--fresh" ] && FRESH=1

cd "$HERE"

if [ "$FRESH" = 1 ] && [ -d config ]; then
  echo "== wiping config/"
  docker compose down --remove-orphans >/dev/null 2>&1 || true
  rm -rf config
fi

mkdir -p config
# Never overwrite a config the user has been editing in place.
[ -f config/configuration.yaml ] || cp support/configuration.yaml config/configuration.yaml
: > config/notify_capture.jsonl

echo "== starting Home Assistant (this takes ~30 s on a cold config)"
docker compose up -d

"$VENV_PY" "$HERE/hactl.py" wait --timeout 240
"$VENV_PY" "$HERE/hactl.py" onboard

echo
echo "== WashData loaded?"
if docker compose logs ha 2>&1 | grep -q "Setup of domain ha_washdata"; then
  docker compose logs ha 2>&1 | grep -E "ha_washdata" | tail -5
else
  echo "   (no config entry yet - run ./smoke.sh or hactl.py setup-device)"
fi

cat <<EOF

box:    http://127.0.0.1:8321   (user testbox / testbox-pw-0123)
token:  $HERE/config/.testbox_token
drive:  $VENV_PY $HERE/hactl.py --help
EOF
