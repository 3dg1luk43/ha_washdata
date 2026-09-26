#!/usr/bin/env bash
# Prove unload confirmation without a door sensor (#451), end to end.
#
# Three things here need a real Home Assistant and cannot be judged under a
# MagicMock: the new `button.*_mark_unloaded` entity has to be wired into the
# platform and to follow its availability, `ha_washdata.mark_unloaded` has to
# pass the real service schema (register item 316 is what happens when it does
# not), and the confirmation entity is subscribed through
# `async_track_state_change_event` against the real state machine.
#
# One cycle, two devices, both fed by the same power sensor so both reach the
# Clean state together: one clears through a Zigbee-style button entity, the
# other through the service. The button entity is checked on the second device
# after the service has already cleared it - unavailable is the assertion there.
#
#   ./check_unload_confirm.sh          rebuild the box first (default)
#   ./check_unload_confirm.sh --reuse  reuse a running box
#
# Fresh by default, unlike check_notify_actions.sh: the assertions here are on
# exact states, and a box that already has an active-cycle snapshot restores it
# on the restart below and folds this run's power into the previous run's cycle.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python3"
HACTL="$PY $HERE/hactl.py"
SENSOR="sensor.washdata_power"
BUTTON="event.laundry_button"

REUSE=0
[ "${1:-}" = "--reuse" ] && REUSE=1

cd "$HERE"
if [ "$REUSE" = 0 ]; then
  ./up.sh --fresh
elif ! curl -fsS -o /dev/null "http://127.0.0.1:8321/manifest.json" 2>/dev/null; then
  ./up.sh
fi

# Shared by both devices. The cycle is deliberately tiny, so the gates have to
# come down with it: `completion_min_seconds` (600 s by default) is what decides
# completed vs interrupted, and only a completed cycle reaches the Clean state -
# `check_state()` surfaces the overlay for OFF and FINISHED only. See README,
# "Time compression".
#
# The restart below is no longer required for `min_off_gap` - this check is what
# found register item 351, where the options-reload path never re-applied it to
# the detector, and that is fixed. It is kept because a restart is also the only
# way to prove the confirmation-entity listener survives one.
common_options() {
  $HACTL set-options "$1" \
    power_sensor="$SENSOR" \
    start_duration_threshold=1 \
    sampling_interval=1 \
    watchdog_interval=5 \
    off_delay=10 \
    min_off_gap=10 \
    interrupted_min_seconds=5 \
    completion_min_seconds=5 \
    progress_reset_delay=3600 \
    notify_unload_delay_minutes=0 > /dev/null
}

echo "== two devices on one power sensor"
ENTRY_A=$($HACTL setup-device --name "Unload Entity" --type washing_machine | tail -1)
ENTRY_B=$($HACTL setup-device --name "Unload Service" --type washing_machine | tail -1)
echo "entity route: $ENTRY_A"
echo "service route: $ENTRY_B"

common_options "$ENTRY_A"
$HACTL set-options "$ENTRY_A" unload_confirm_entity="$BUTTON" > /dev/null
common_options "$ENTRY_B"
$HACTL set-options "$ENTRY_B" unload_track_without_door=true > /dev/null

DEVICE_B=$($HACTL device-id --entry "$ENTRY_B")
echo "device id (service route): $DEVICE_B"

echo
echo "== restart (the listener has to come back with it)"
$HACTL restart > /dev/null
# The restart clears `hass.data`, which re-anchors the unload-confirm replay
# window (register item 367) - deliberately, since an HA restart IS the moment a
# retained MQTT value can replay. The first-press check at the end of this script
# waits that window out explicitly rather than hoping the cycle work covers it.

# Seed the button AFTER the restart, and before the cycle. Two reasons, and the
# first one cost a run: a state pushed through the REST API is not restored
# across a restart, so seeding earlier leaves the entity absent and the press
# below arrives with no old state. Which is exactly what the handler ignores,
# deliberately - a button entity that restores its last-press timestamp on HA
# start must not read as somebody standing at the machine.
$HACTL set-state "$BUTTON" "2026-09-24T09:00:00+00:00" device_class=button > /dev/null

echo
echo "== run one cycle through both"
$HACTL set "$SENSOR" 0
sleep 2
for w in 250 300 320 310 300 280; do $HACTL set "$SENSOR" "$w"; sleep 3; done
$HACTL set "$SENSOR" 0

echo
echo "== wait for both to finish"
for _ in $(seq 1 24); do
  SA=$($HACTL state sensor.unload_entity_state | $PY -c 'import json,sys; print(json.load(sys.stdin)["state"])')
  SB=$($HACTL state sensor.unload_service_state | $PY -c 'import json,sys; print(json.load(sys.stdin)["state"])')
  echo "  $SA / $SB"
  [ "$SA" = "clean" ] && [ "$SB" = "clean" ] && break
  sleep 5
done

FAIL=0
check() {  # check <label> <entity> <expected>
  local got
  got=$($HACTL state "$2" | $PY -c 'import json,sys; print(json.load(sys.stdin)["state"])')
  if [ "$got" = "$3" ]; then
    echo "  PASS  $1 ($2 = $got)"
  else
    echo "  FAIL  $1 ($2 = $got, expected $3)"
    FAIL=1
  fi
}

echo
echo "== both entered the Clean state with no door sensor"
check "confirmation entity enables Clean" sensor.unload_entity_state clean
check "manual flag enables Clean" sensor.unload_service_state clean
check "Mark Unloaded is available while waiting" \
  button.unload_service_mark_unloaded unknown

echo
echo "== clear one with a button press, the other with the service"
$HACTL set-state "$BUTTON" "2026-09-24T10:30:00+00:00" device_class=button > /dev/null
sleep 3
$HACTL call ha_washdata.mark_unloaded device_id="$DEVICE_B" > /dev/null
sleep 3

check "a button press cleared Clean" sensor.unload_entity_state finished
check "the service cleared Clean" sensor.unload_service_state finished
check "Mark Unloaded goes unavailable once nothing is waiting" \
  button.unload_service_mark_unloaded unavailable

# ── register item 367: the FIRST EVER press of a fresh button ────────────────
# A fresh event/button/input_button entity sits at `unknown` until it is pressed,
# and the handler used to exclude `unknown -> value` outright, swallowing that
# press. It is accepted once we are past UNLOAD_CONFIRM_REPLAY_GRACE_S from the
# subscription, which a full cycle replay has long exceeded by this point. Real
# HA is the only tier that can show this: the transition depends on how HA itself
# restores and writes entity state, which a MagicMock cannot reproduce.
echo
echo "== the first ever press of a fresh button (unknown -> timestamp)"
REPLAY_WINDOW=$($PY - <<'PYW'
import pathlib, re
src = pathlib.Path("../../custom_components/ha_washdata/const.py").read_text()
m = re.search(r"^UNLOAD_CONFIRM_REPLAY_GRACE_S\s*=\s*([0-9.]+)", src, re.M)
print(int(float(m.group(1))) + 5 if m else 125)
PYW
)
$HACTL set "$SENSOR" 0 > /dev/null
# Sending the button to `unknown` is itself a drop-out, and a drop-out RE-ARMS the
# window (register item 368) - so the wait below is measured from HERE, not from
# the restart. Measuring it from the restart is what the box caught: the cycle in
# between was not long enough, the press landed inside the re-armed window, and it
# was correctly ignored.
$HACTL set-state "$BUTTON" unknown device_class=button > /dev/null
UNKNOWN_AT=$(date +%s)
sleep 2
# Re-enter Clean: run another short cycle on device A.
for w in 250 300 320 300; do $HACTL set "$SENSOR" "$w"; sleep 3; done
$HACTL set "$SENSOR" 0
for _ in $(seq 1 40); do
  SA=$($HACTL state sensor.unload_entity_state | $PY -c 'import json,sys; print(json.load(sys.stdin)["state"])')
  [ "$SA" = "clean" ] && break
  sleep 3
done
if [ "$SA" = "clean" ]; then
  REMAIN=$(( REPLAY_WINDOW - ( $(date +%s) - UNKNOWN_AT ) ))
  if [ "$REMAIN" -gt 0 ]; then
    echo "  [ ..  ] waiting ${REMAIN}s for the re-armed replay window to close"
    sleep "$REMAIN"
  fi
  $HACTL set-state "$BUTTON" "2026-09-24T12:00:00+00:00" device_class=button > /dev/null
  sleep 3
  check "a first-ever press from unknown clears Clean" \
    sensor.unload_entity_state finished
else
  echo "  [ ..  ] could not re-enter Clean in time; first-press check skipped"
fi

echo
echo "== the service was accepted by the bus (item 316 class)"
if $HACTL errors | grep -iE "mark_unloaded|MultipleInvalid" ; then
  echo "  FAIL  the service call was rejected or logged an error"
  FAIL=1
else
  echo "  PASS  no rejection in the log"
fi

$HACTL set "$SENSOR" 0 > /dev/null
echo
[ "$FAIL" = 0 ] && echo "ALL CHECKS PASSED" || { echo "CHECKS FAILED"; exit 1; }
