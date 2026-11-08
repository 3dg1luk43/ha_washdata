#!/usr/bin/env bash
# Prove the Clean state survives a Home Assistant restart (#500), end to end.
#
# Why the box: a real restart does NOT unload the config entry, so the only
# write a restart can rely on is the one made while HA runs (write-through) or
# at EVENT_HOMEASSISTANT_STOP. The unit tier simulates a restart by unloading,
# which a real HA never does; here the container is restarted for real, once
# gracefully (SIGTERM: the stop path) and once killed (SIGKILL: no stop at all,
# so only the write-through can carry the state).
#
# Two devices on one power sensor, both reaching Clean together:
#   "Clean Restart"  no door sensor, unload_track_without_door, a 1 min reminder
#   "Clean Door"     a door sensor, no reminder
#
#   ./check_clean_restart.sh          rebuild the box first (default)
#   ./check_clean_restart.sh --reuse  reuse a running box
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python3"
HACTL="$PY $HERE/hactl.py"
SENSOR="sensor.washdata_power"
DOOR="binary_sensor.testbox_door"
CAPTURE="$HERE/config/notify_capture.jsonl"
NAG="CLEANNAG"

REUSE=0
[ "${1:-}" = "--reuse" ] && REUSE=1

cd "$HERE"
if [ "$REUSE" = 0 ]; then
  ./up.sh --fresh
elif ! curl -fsS -o /dev/null "http://127.0.0.1:8321/manifest.json" 2>/dev/null; then
  ./up.sh
fi

# Time compression, as in check_unload_confirm.sh (README, "Time compression").
common_options() {
  $HACTL set-options "$1" \
    power_sensor="$SENSOR" \
    start_duration_threshold=1 \
    watchdog_interval=5 \
    off_delay=10 \
    min_off_gap=10 \
    interrupted_min_seconds=5 \
    completion_min_seconds=5 \
    progress_reset_delay=3600 > /dev/null
}

state_of() {
  $HACTL state "$1" | $PY -c 'import json,sys; print(json.load(sys.stdin)["state"])'
}

clean_record() {  # clean_record <entry> <field>: from the device's active file
  $PY - "$HERE/config/.storage/ha_washdata.$1.active" "$2" <<'PYR'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text()).get("data", {}) if path.exists() else {}
record = data.get("clean_state")
print("none" if record is None else record.get(sys.argv[2]))
PYR
}

nag_count() {
  [ -f "$CAPTURE" ] || { echo 0; return; }
  grep -c "$NAG" "$CAPTURE" || true
}

# --reuse keeps the capture file, and an earlier run's reminders with it: count
# only the ones sent after this run's baseline (taken before the cycle).
NAG0=0
nag_sent() { echo $(( $(nag_count) - NAG0 )); }

FAIL=0
check() {  # check <label> <got> <expected>
  if [ "$2" = "$3" ]; then
    echo "  PASS  $1 ($2)"
  else
    echo "  FAIL  $1 (got $2, expected $3)"
    FAIL=1
  fi
}

echo "== two devices on one power sensor"
ENTRY_A=$($HACTL setup-device --name "Clean Restart" --type washing_machine | tail -1)
ENTRY_B=$($HACTL setup-device --name "Clean Door" --type washing_machine | tail -1)
common_options "$ENTRY_A"
$HACTL set-options "$ENTRY_A" \
  unload_track_without_door=true \
  notify_finish_services='["notify.plain_testbox"]' \
  notify_unload_delay_minutes=1 \
  notify_unload_message="$NAG {device} {duration}" > /dev/null
common_options "$ENTRY_B"
$HACTL set-options "$ENTRY_B" \
  door_sensor_entity="$DOOR" \
  notify_unload_delay_minutes=0 > /dev/null
$HACTL restart > /dev/null
# A state pushed through the REST API is not restored across a restart, so the
# door is seeded after it (closed: only a closed door enters Clean).
$HACTL set-state "$DOOR" off device_class=door > /dev/null

NAG0=$(nag_count)

echo
echo "== one cycle through both"
$HACTL set "$SENSOR" 0
sleep 2
for w in 250 300 320 310 300 280; do $HACTL set "$SENSOR" "$w"; sleep 3; done
$HACTL set "$SENSOR" 0
for _ in $(seq 1 24); do
  SA=$(state_of sensor.clean_restart_state)
  SB=$(state_of sensor.clean_door_state)
  echo "  $SA / $SB"
  [ "$SA" = "clean" ] && [ "$SB" = "clean" ] && break
  sleep 5
done
check "no door sensor: Clean" "$(state_of sensor.clean_restart_state)" clean
check "door sensor: Clean" "$(state_of sensor.clean_door_state)" clean

echo
echo "== the reminder goes out once (the expiry tick runs every 60 s)"
for _ in $(seq 1 30); do
  [ "$(nag_sent)" -ge 1 ] && break
  sleep 5
done
check "reminder sent" "$(nag_sent)" 1
START_A=$(clean_record "$ENTRY_A" start)
check "written while HA runs (no unload on a restart)" \
  "$(clean_record "$ENTRY_A" notified)" True

echo
echo "== graceful restart (SIGTERM, the stop path)"
$HACTL restart > /dev/null
sleep 5
check "no door sensor: still Clean" "$(state_of sensor.clean_restart_state)" clean
check "door sensor: still Clean" "$(state_of sensor.clean_door_state)" clean
# Not unavailable: `unknown` on a fresh box, an earlier run's press time on --reuse.
BTN=$(state_of button.clean_restart_mark_unloaded)
check "Mark Unloaded still available" \
  "$([ "$BTN" != unavailable ] && echo available || echo "$BTN")" available
check "original Clean start kept" "$(clean_record "$ENTRY_A" start)" "$START_A"

echo
echo "== crash (SIGKILL, nothing written at stop)"
docker compose kill -s SIGKILL ha > /dev/null
docker compose start ha > /dev/null
$HACTL wait > /dev/null
sleep 5
check "no door sensor: Clean after a crash" "$(state_of sensor.clean_restart_state)" clean
check "original Clean start kept" "$(clean_record "$ENTRY_A" start)" "$START_A"
echo "  [ ..  ] waiting 150 s: a resent reminder would arrive in that time"
sleep 150
check "reminder not sent again" "$(nag_sent)" 1

echo
echo "== the unload is still confirmable after the restarts"
# The door entity is absent after the restart (REST state), so this is a
# None -> on change: the door listener reads it as opened.
$HACTL set-state "$DOOR" on device_class=door > /dev/null
sleep 3
check "door open cleared Clean" "$(state_of sensor.clean_door_state)" finished
$HACTL call button.press entity_id=button.clean_restart_mark_unloaded > /dev/null
sleep 3
check "Mark Unloaded cleared Clean" "$(state_of sensor.clean_restart_state)" finished
check "record dropped" "$(clean_record "$ENTRY_A" start)" none

echo
echo "== a restart after the unload stays unloaded"
$HACTL restart > /dev/null
sleep 5
GOT=$(state_of sensor.clean_restart_state)
[ "$GOT" != "clean" ] && echo "  PASS  not Clean again ($GOT)" \
  || { echo "  FAIL  Clean came back after the unload"; FAIL=1; }

echo
echo "== nothing rejected or raised"
if $HACTL errors | grep -iE "clean_state|Clean state|MultipleInvalid" ; then
  echo "  FAIL  an error mentions the Clean state"
  FAIL=1
else
  echo "  PASS  no Clean-state error in the log"
fi

$HACTL set "$SENSOR" 0 > /dev/null
echo
[ "$FAIL" = 0 ] && echo "ALL CHECKS PASSED" || { echo "CHECKS FAILED"; exit 1; }
