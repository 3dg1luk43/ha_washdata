#!/usr/bin/env bash
# Prove the finish notification's review answers (audit item 1, UX 3), end to end.
#
# When a cycle end raises a review request, the finish notification carries two
# mobile actions named after the request's top two ranked programmes, and a tap
# (a `mobile_app_notification_action` event) resolves the request the way the
# panel's resolve_feedback does. What only a real Home Assistant can show: the
# payload with `actions` passes the real notify schema and reaches the mobile
# target only, the listener is wired into the real bus, it is re-registered by
# the setup that follows a restart, and a tap on an answered request is a no-op.
#
#   ./check_review_actions.sh          rebuild the box first (default)
#   ./check_review_actions.sh --reuse  reuse a running box
#
# Four 75 s cycles, ~10 min. Alpha and Bravo are made from the first two (near
# twins), the third is Alpha-like: both programmes top its ranking and the label
# gate refuses it on the margin, so the request is raised. Round two asks about
# another Alpha-like cycle, then restarts the box between the request and the
# tap. No mobile device is registered here, so the tap is fired on the bus
# directly, with the action id the notification carried.
#
# 75 s, not shorter: a cycle of 60 s or less never shapes an envelope
# (profile_store.async_rebuild_envelope), so the complete-cycle match has no
# candidates, scores 0 and nothing is ever asked.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python3"
HACTL="$PY $HERE/hactl.py"
SENSOR="sensor.washdata_power"
CAPTURE="$HERE/config/notify_capture.jsonl"
EVENT="mobile_app_notification_action"

REUSE=0
[ "${1:-}" = "--reuse" ] && REUSE=1

cd "$HERE"
if [ "$REUSE" = 0 ]; then
  ./up.sh --fresh
elif ! curl -fsS -o /dev/null "http://127.0.0.1:8321/manifest.json" 2>/dev/null; then
  ./up.sh
fi

# A new device every run, so a reused box's profiles and cycles cannot reach
# this device's rankings, and the capture can be filtered on its name.
RUN=$(date +%H%M%S)
NAME="Review $RUN"
SLUG="review_$RUN"

# The log position at the start: only lines after it are judged at the end.
LOG_MARK=$($PY - "$HERE/config" <<'EOF'
import pathlib, sys
log = pathlib.Path(sys.argv[1]) / "home-assistant.log"
mark = ""
if log.exists():
    for line in log.read_text(errors="replace").splitlines():
        if len(line) >= 23 and line[:4].isdigit() and line[4] == "-":
            mark = line[:23]
print(mark)
EOF
)

now_utc() { $PY -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())'; }

FAIL=0
pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1"; FAIL=1; }

echo "== device: $NAME"
ENTRY=$($HACTL setup-device --name "$NAME" --type washing_machine | tail -1)
PREFIX="REVIEW_WD_$(echo "${ENTRY:0:8}" | tr '[:lower:]' '[:upper:]')_"
echo "entry_id: $ENTRY (action prefix $PREFIX)"

# smoke.sh's timing options, for the same reason (README, "Time compression"),
# with the finish notification on a mobile AND a plain target. Start and live
# notifications are off so the capture holds only what this check reads.
# match_persistence=1: a cycle the live matcher never committed is not asked
# about (learning._maybe_request_feedback has no expected duration then), and at
# the default of 3 a 75 s cycle can end before its third winning tick.
# learning_confidence=0.3: a synthetic square-wave cycle scores lower against
# its own programme than a real trace; under the floor no request is raised.
$HACTL set-options "$ENTRY" \
  learning_confidence=0.3 \
  auto_label_confidence=0.95 \
  power_sensor="$SENSOR" \
  notify_start_services='[]' \
  notify_live_services='[]' \
  notify_finish_services='["notify.mobile_app_testbox","notify.plain_testbox"]' \
  profile_match_interval=10 \
  match_persistence=1 \
  watchdog_interval=5 \
  start_duration_threshold=1 \
  off_delay=10 \
  min_off_gap=10 \
  completion_min_seconds=5 \
  interrupted_min_seconds=5 > /dev/null
echo "options written"

# ── helpers ──────────────────────────────────────────────────────────────────
newest_cycle_id() {
  $HACTL ws ha_washdata/get_device_cycles entry_id="$ENTRY" limit=1 | $PY -c '
import json, sys
cycles = json.load(sys.stdin).get("cycles") or []
print(cycles[0]["id"] if cycles else "")'
}

cycle_label() {  # cycle_label <cycle_id> -> "<profile_name>|<label_source>"
  $HACTL ws ha_washdata/get_device_cycles entry_id="$ENTRY" limit=50 | $PY -c '
import json, sys
cid = sys.argv[1]
cyc = next((c for c in json.load(sys.stdin).get("cycles") or [] if c.get("id") == cid), {})
print("%s|%s" % (cyc.get("profile_name"), cyc.get("label_source")))' "$1"
}

pending_for() {  # pending_for <cycle_id> -> the waiting request as JSON, or null
  $HACTL ws ha_washdata/get_feedbacks entry_id="$ENTRY" | $PY -c '
import json, sys
cid = sys.argv[1]
reqs = json.load(sys.stdin).get("feedbacks") or []
print(json.dumps(next((r for r in reqs if r.get("cycle_id") == cid), None)))' "$1"
}

pending_total() {
  $HACTL ws ha_washdata/get_feedbacks entry_id="$ENTRY" \
    | $PY -c 'import json, sys; print(len(json.load(sys.stdin).get("feedbacks") or []))'
}

fire() {  # fire <action id>: what the companion app sends when a button is tapped
  $HACTL ws fire_event event_type="$EVENT" event_data="{\"action\": \"$1\"}" > /dev/null
}

drive() {  # drive <watts> <seconds>: a reading every ~2 s, jittered so each is a state change
  local end=$(( $(date +%s) + $2 )) i=0
  while [ "$(date +%s)" -lt "$end" ]; do
    $HACTL set "$SENSOR" $(( $1 + (i % 2) * 7 )) > /dev/null
    i=$((i + 1))
    sleep 2
  done
}

# run_cycle <label> <watts:seconds>...: one cycle, waits until it is stored and
# its notifications have settled, and leaves its id in $CID.
run_cycle() {
  local label=$1 prev step fin_prev
  shift
  echo
  echo "== cycle $label"
  sleep 12  # past min_off_gap, so it is not merged into the previous one
  prev=$(newest_cycle_id)
  fin_prev=$(finish_count)
  for step in "$@"; do drive "${step%%:*}" "${step##*:}"; done
  $HACTL set "$SENSOR" 0 > /dev/null
  CID=""
  for _ in $(seq 1 60); do
    CID=$(newest_cycle_id)
    [ -n "$CID" ] && [ "$CID" != "$prev" ] && break
    CID=""
    sleep 3
  done
  if [ -z "$CID" ]; then
    echo "  FAIL  cycle $label was never stored; nothing below can be judged"
    exit 1
  fi
  # The finish notification goes out a moment after the store (smoke.sh 6b):
  # wait for this cycle's, on both targets, rather than for the capture to stop
  # growing (a delivery slower than one poll read as settled).
  for _ in $(seq 1 20); do
    [ "$(finish_count)" -ge $((fin_prev + 2)) ] && break
    sleep 3
  done
  echo "  stored as $CID ($(cycle_label "$CID"))"
}

finish_count() {  # finish notifications delivered so far, every target
  $PY - "$CAPTURE" "$NAME" <<'EOF'
import json, sys
path, name = sys.argv[1:3]
n = 0
for line in open(path, encoding="utf-8", errors="replace"):
    try:
        rec = json.loads(line)
    except ValueError:
        continue
    if str(rec.get("message", "")).startswith(f"{name} finished"):
        n += 1
print(n)
EOF
}

wait_request() {  # wait_request <cycle_id>: the request is queued just after the store
  local req
  for _ in $(seq 1 10); do
    req=$(pending_for "$1")
    [ "$req" != "null" ] && { echo "$req"; return 0; }
    sleep 2
  done
  echo "null"
}

new_programme() {  # new_programme <cycle_id> <name>: the panel's "label as new programme"
  $HACTL ws ha_washdata/label_cycle entry_id="$ENTRY" cycle_id="$1" \
    profile_name=__create_new__ new_profile_name="$2" > /dev/null
  echo "  $1 -> new programme $2 ($(cycle_label "$1"))"
}

# check_actions <since> <cycle_id> <request-json> <expected detected> <allowed runners-up>
# The finish notification of <cycle_id>: C_ and R1_ ids for this device and
# cycle, titled after the request's detected programme and its ranking[1], on
# the mobile target only.
check_actions() {
  $PY - "$CAPTURE" "$1" "$NAME" "$PREFIX" "$2" "$3" "$4" "$5" <<'EOF'
import json, sys
path, since, name, prefix, cid, req_raw, want_detected, runners = sys.argv[1:9]
req = json.loads(req_raw)
ok = True

def check(cond, text):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + text)
    ok = ok and cond

recs = []
for line in open(path, encoding="utf-8", errors="replace"):
    try:
        rec = json.loads(line)
    except ValueError:
        continue
    if rec.get("ts", "") >= since and str(rec.get("message", "")).startswith(f"{name} finished"):
        recs.append(rec)
mobile = [r for r in recs if r.get("service") == "mobile_app_testbox"]
plain = [r for r in recs if r.get("service") == "plain_testbox"]
check(bool(mobile) and bool(plain), f"the finish notification reached both targets "
      f"({len(mobile)} mobile, {len(plain)} plain)")
ranking = [c.get("name") for c in req.get("ranking") or []]
detected = req.get("detected_profile")
check(detected in want_detected.split(",") and len(ranking) > 1 and ranking[0] == detected
      and ranking[1] in runners.split(",") and ranking[1] != detected,
      f"the request names {detected!r}, ranking {ranking[:2]} "
      f"(expected {want_detected!r} then one of {runners})")
actions = (mobile[-1].get("data") or {}).get("actions") if mobile else None
want = [
    {"action": f"{prefix}C_{cid}", "title": f"It was {detected}"},
    {"action": f"{prefix}R1_{cid}", "title": f"It was {ranking[1] if len(ranking) > 1 else '?'}"},
]
check(actions == want, f"two answers on the mobile finish notification: {actions}")
check(not any("actions" in (r.get("data") or {}) for r in plain),
      "no actions on the plain target")
sys.exit(0 if ok else 1)
EOF
}

# answer_round <cycle_id> <request-json>: tap the correction, then tap again.
answer_round() {
  local cid=$1 want got
  want=$(echo "$2" | $PY -c 'import json, sys; print(json.load(sys.stdin)["ranking"][1]["name"])')
  fire "${PREFIX}R1_${cid}"
  for _ in $(seq 1 10); do
    got=$(cycle_label "$cid")
    [ "$got" = "$want|manual" ] && break
    sleep 2
  done
  [ "$got" = "$want|manual" ] \
    && pass "the R1 tap labelled the cycle $want, label_source manual" \
    || fail "the R1 tap left the cycle at $got (expected $want|manual)"
  [ "$(pending_for "$cid")" = "null" ] && [ "$(pending_total)" = 0 ] \
    && pass "get_feedbacks is empty" \
    || fail "a request is still waiting: $($HACTL ws ha_washdata/get_feedbacks entry_id="$ENTRY")"

  # Both buttons again: the request is answered, so nothing may change.
  fire "${PREFIX}R1_${cid}"
  fire "${PREFIX}C_${cid}"
  sleep 4
  got=$(cycle_label "$cid")
  [ "$got" = "$want|manual" ] && [ "$(pending_total)" = 0 ] \
    && pass "a second tap on an answered request changes nothing" \
    || fail "a second tap changed the cycle to $got"
}

# ── round one: a waiting request, answered from the notification ─────────────
$HACTL set "$SENSOR" 0 > /dev/null
sleep 2

# Bravo is a near-twin of Alpha, so an Alpha-like cycle matches both closely and
# the label gate refuses it on the margin: the request this check needs.
run_cycle "1 (Alpha shape)" 1800:25 250:50
new_programme "$CID" Alpha
run_cycle "2 (Bravo shape, Alpha's twin)" 1750:25 270:50
new_programme "$CID" Bravo
# Labelling resolves a cycle's own request (#331); anything else left over would
# make "get_feedbacks is empty" below a statement about an older cycle.
$HACTL ws ha_washdata/dismiss_all_feedbacks entry_id="$ENTRY" > /dev/null

SINCE=$(now_utc)
run_cycle "3 (Alpha-like)" 1800:25 250:50
REQ=$(wait_request "$CID")
echo
echo "== round one: the finish notification of $CID"
if [ "$REQ" = "null" ]; then
  fail "cycle 3 raised no review request; the rest of round one is skipped"
else
  check_actions "$SINCE" "$CID" "$REQ" Alpha,Bravo Alpha,Bravo || FAIL=1
  answer_round "$CID" "$REQ"
fi

# ── round two: the request is raised, HA restarts, then the tap arrives ──────
# Round one's answer was a correction, which does not spend the programme's review
# budget, so another Alpha-like cycle is asked about again.
SINCE=$(now_utc)
run_cycle "4 (Alpha-like again)" 1800:25 250:50
REQ=$(wait_request "$CID")
echo
echo "== round two: the finish notification of $CID"
if [ "$REQ" = "null" ]; then
  fail "cycle 4 raised no review request; the restart round is skipped"
else
  check_actions "$SINCE" "$CID" "$REQ" Alpha,Bravo Alpha,Bravo || FAIL=1
  echo
  echo "== restart with the request waiting (the setup must re-register the listener)"
  $HACTL restart > /dev/null
  # A REST-pushed state is not restored across a restart (check_unload_confirm.sh).
  $HACTL set "$SENSOR" 0 > /dev/null
  for _ in $(seq 1 30); do
    STATE=$($HACTL state "sensor.${SLUG}_state" 2> /dev/null \
      | $PY -c 'import json, sys; print(json.load(sys.stdin)["state"])' 2> /dev/null || echo unavailable)
    [ "$STATE" != "unavailable" ] && [ "$STATE" != "unknown" ] && break
    sleep 2
  done
  echo "  device state after the restart: $STATE"
  answer_round "$CID" "$REQ"
fi

# ── the log ──────────────────────────────────────────────────────────────────
# assert_run.py's three log checks, on this run's lines only (the restart rolls
# the log over to home-assistant.log.1, so both files are read). WARNING lines
# are not judged: every restart logs the custom-integration warning.
echo
echo "== the log since the start of this run"
if $PY - "$HERE/config" "$LOG_MARK" <<'EOF'
import pathlib, sys
cfg, mark = pathlib.Path(sys.argv[1]), sys.argv[2]
bad = []
for name in ("home-assistant.log.1", "home-assistant.log"):
    path = cfg / name
    if not path.exists():
        continue
    stamp = ""
    for line in path.read_text(errors="replace").splitlines():
        if len(line) >= 23 and line[:4].isdigit() and line[4] == "-":
            stamp = line[:23]
        if stamp <= mark:
            continue
        if ("MultipleInvalid" in line or "not a valid value" in line
                or ("ERROR" in line and "ha_washdata" in line)
                or "Traceback (most recent call last)" in line):
            bad.append(line)
print("\n".join(bad[-12:]))
sys.exit(1 if bad else 0)
EOF
then
  pass "no schema rejection, ha_washdata error or traceback"
else
  fail "the log has errors since the start of this run (above)"
fi

$HACTL set "$SENSOR" 0 > /dev/null
echo
[ "$FAIL" = 0 ] && echo "ALL CHECKS PASSED" || { echo "CHECKS FAILED"; exit 1; }
