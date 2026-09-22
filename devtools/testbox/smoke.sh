#!/usr/bin/env bash
# One end-to-end cycle on a real Home Assistant, start to finish.
#
#   ./smoke.sh                 reuse a running box
#   ./smoke.sh --fresh         wipe and rebuild the box first
#   ./smoke.sh --speedup 120   replay faster (see the note on timings below)
#
# What it proves that the unit suite cannot: the config flow creates a working
# entry, the entities appear, an export imports through the real storage
# migration, a replayed cycle is detected and stored, and every notification
# WashData sends - including the dismissals - is accepted by Home Assistant's
# service bus and reaches a notify platform.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python3"
HACTL="$PY $HERE/hactl.py"

# A washing machine on purpose. Dishwashers carry a hardcoded 30 min floor
# (DISHWASHER_MIN_CYCLE_DURATION_S, not an option), so a compressed dishwasher
# cycle can never reach its end gate: at 60x the whole cycle is 97 s of wall
# clock. To smoke a dishwasher, pass --type dishwasher with --speedup 4 or less
# and expect a ~35 min run. See README.md, "Time compression".
NAME="Test Washer"
SLUG="test_washer"
TYPE="washing_machine"
SPEEDUP=60
EXPORT="$REPO/cycle_data/me/washdata_export_01KXGA3C.json"
CYCLE=0
FRESH=0

while [ $# -gt 0 ]; do
  case "$1" in
    --fresh) FRESH=1; shift ;;
    --type) TYPE="$2"; NAME="Test $2"; SLUG="test_$2"; shift 2 ;;
    --name) NAME="$2"; SLUG="$(echo "$2" | tr '[:upper:] ' '[:lower:]_')"; shift 2 ;;
    --speedup) SPEEDUP="$2"; shift 2 ;;
    --export) EXPORT="$2"; shift 2 ;;
    --cycle) CYCLE="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$HERE"
if [ "$FRESH" = 1 ]; then
  ./up.sh --fresh
elif ! curl -fsS -o /dev/null "http://127.0.0.1:8321/manifest.json" 2>/dev/null; then
  ./up.sh
fi

echo
echo "== 1. config entry (real config flow)"
ENTRY=$($HACTL setup-device --name "$NAME" --type "$TYPE" | tail -1)
echo "entry_id: $ENTRY"

echo
echo "== 2. seed profiles (real import + storage migration)"
# The export's clock is rescaled by the same factor as the replay. Without it
# the matcher compares a 76 s replay against a 1720 s learned profile, reads the
# cycle as 7% complete and refuses to end it - which is correct behaviour on a
# wrong premise. Shapes are untouched; only the clock moves. (hactl.py
# compress-export)
COMPRESSED="$HERE/config/.compressed_export.json"
$HACTL compress-export "$EXPORT" "$COMPRESSED" --speedup "$SPEEDUP"
$HACTL import "$ENTRY" "$COMPRESSED" | head -20

echo
echo "== 3. options: notify targets + timings compressed by ${SPEEDUP}x"
# power_sensor is re-asserted here because an import legitimately may not carry
# it (item 317: an export names the exporter's entities, so the import drops
# local bindings). Setting it explicitly keeps the smoke run honest either way.
# The detector's gates are wall-clock, so a cycle replayed ${SPEEDUP}x faster
# needs its gates divided by the same factor or it would never close inside the
# run. Everything else is left at the shipped default on purpose - the point is
# to exercise production behaviour, not a bespoke configuration.
#
# notify_live_interval_seconds is floored at 30 in the manager, so at 60x one
# live update covers ~30 min of appliance time: a 76 min cycle yields ~2.
#
# interrupted_min_seconds is marked "internal use only" in const.py but has to be
# compressed too: a cycle shorter than it is filed as "interrupted" and never
# stored, which reads as a detection failure rather than a unit mismatch.
#
# profile_match_interval likewise: at its 300 s default the matcher never fires
# inside a 76 s replay, so every live update stays on the "no profile matched
# yet" branch and the matched payload (progress, chronometer, when) is never
# exercised - which is exactly the payload users see for most of a cycle.
$HACTL set-options "$ENTRY" \
  power_sensor=sensor.washdata_power \
  notify_start_services='["notify.mobile_app_testbox","notify.plain_testbox"]' \
  notify_finish_services='["notify.mobile_app_testbox","notify.plain_testbox"]' \
  notify_live_services='["notify.mobile_app_testbox","notify.plain_testbox"]' \
  notify_live_interval_seconds=30 \
  notify_live_chronometer=true \
  notify_before_end_minutes=20 \
  profile_match_interval=10 \
  sampling_interval=1 \
  watchdog_interval=5 \
  start_duration_threshold=1 \
  off_delay=10 \
  min_off_gap=10 \
  completion_min_seconds=5 \
  interrupted_min_seconds=5 > /dev/null
echo "options written"

echo
echo "== 4. baseline"
BASELINE="$HERE/config/.baseline.json"
CYCLES_BEFORE=$($HACTL ws ha_washdata/get_device_cycles entry_id="$ENTRY" \
  | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d.get("total", len(d.get("cycles", []))))')
$PY - "$BASELINE" "$CYCLES_BEFORE" <<'EOF'
import json, sys
from datetime import datetime, timezone
path, count = sys.argv[1], int(sys.argv[2])
json.dump({"since": datetime.now(timezone.utc).isoformat(), "cycle_count": count}, open(path, "w"))
print(f"cycles before: {count}")
EOF

echo
echo "== 5. replay"
$HACTL replay "$EXPORT" --cycle "$CYCLE" --speedup "$SPEEDUP" --tail 45

echo
echo "== 6. let the cycle close"
# 0 W repeated does not fire a state_changed event (Home Assistant drops
# unchanged states), which is exactly the report-on-change plug that #424/#427
# were about: from here on the watchdog is the only thing driving the detector.
for _ in $(seq 1 150); do
  STATE=$($HACTL state "sensor.${SLUG}_state" | $PY -c 'import json,sys; print(json.load(sys.stdin)["state"])')
  echo "   state: $STATE"
  case "$STATE" in
    finished|clean|off) break ;;
  esac
  sleep 5
done

echo
echo "== 6b. settle"
# The cycle-end notifications are dispatched a moment AFTER the state flips
# (measured: 0.46 s), and the assertions read the capture file, so asserting the
# instant the state changes is a race that loses the finish alert and the
# live-activity clear. Wait for the capture to stop growing instead.
SETTLE_PREV=-1
for _ in $(seq 1 12); do
  SETTLE_NOW=$(wc -l < "$HERE/config/notify_capture.jsonl")
  if [ "$SETTLE_NOW" = "$SETTLE_PREV" ]; then
    echo "   capture stable at $SETTLE_NOW notifications"
    break
  fi
  SETTLE_PREV="$SETTLE_NOW"
  sleep 3
done

echo
echo "== 7. acceptance"
cd "$HERE" && $PY assert_run.py "$ENTRY" --slug "$SLUG" --baseline "$BASELINE"
