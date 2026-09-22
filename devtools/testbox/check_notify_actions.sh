#!/usr/bin/env bash
# Prove the notification-ACTION delivery path, end to end.
#
# WashData has two ways to deliver a notification: the notify services
# (`_send_notification_service`, covered by smoke.sh and by the schema guard in
# tests/conftest.py) and user-supplied *actions* - a YAML script run through
# `script_helper.Script` with the notification's variables bound. Every unit test
# that touches the action path mocks that Script, so nothing anywhere proved an
# action actually delivers. This does, in about a minute: it configures a real
# action, starts a cycle with a power spike, and looks for the record only an
# action could have produced.
#
#   ./check_notify_actions.sh          reuses the running box
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python3"
HACTL="$PY $HERE/hactl.py"
SENSOR="sensor.washdata_power"

cd "$HERE"
if ! curl -fsS -o /dev/null "http://127.0.0.1:8321/manifest.json" 2>/dev/null; then
  ./up.sh
fi

ENTRY=$($HACTL setup-device --name "Test Washer" --type washing_machine | tail -1)
echo "entry_id: $ENTRY"

echo
echo "== configure an action, and NO notify services for the start event"
# With notify_start_services empty, anything that arrives can only have come
# through the action. The marker message is what the assertion looks for.
$HACTL set-options "$ENTRY" \
  power_sensor="$SENSOR" \
  notify_start_services='[]' \
  notify_actions='[{"action": "notify.plain_testbox", "data": {"message": "ACTION PATH {{ device }}: {{ message }}", "title": "via action"}}]' \
  start_duration_threshold=1 \
  sampling_interval=1 \
  watchdog_interval=5 \
  off_delay=10 \
  min_off_gap=10 > /dev/null
echo "options written"

BEFORE=$(wc -l < config/notify_capture.jsonl)

echo
echo "== start a cycle"
$HACTL set "$SENSOR" 0
sleep 2
for w in 250 300 320 310; do $HACTL set "$SENSOR" "$w"; sleep 3; done

echo
echo "== wait for the capture to settle"
PREV=-1
for _ in $(seq 1 10); do
  NOW=$(wc -l < config/notify_capture.jsonl)
  [ "$NOW" = "$PREV" ] && break
  PREV="$NOW"
  sleep 3
done

echo
echo "== result"
$PY - "$BEFORE" <<'EOF'
import json, sys
from pathlib import Path

before = int(sys.argv[1])
lines = Path("config/notify_capture.jsonl").read_text().splitlines()[before:]
records = [json.loads(line) for line in lines if line.strip()]
for r in records:
    print(f"  {r['service']:<19} {r.get('title')!r:<14} {r['message'][:60]}")
hits = [r for r in records if r["message"].startswith("ACTION PATH")]
if hits:
    print(f"\nPASS: the action delivered {len(hits)} notification(s) with the "
          "variables bound")
    print(f"      {hits[0]['message']!r} title={hits[0].get('title')!r}")
else:
    print("\nFAIL: nothing arrived from the action path. Either the script did "
          "not run or it raised; check config/home-assistant.log for "
          "'Error executing script' around now.")
    sys.exit(1)
EOF

echo
echo "== restore the cycle to idle"
$HACTL set "$SENSOR" 0 > /dev/null
