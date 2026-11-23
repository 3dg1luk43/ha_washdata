#!/usr/bin/env bash
# Prove that Trim can move a cycle's end past its trace from the recorder (#536).
#
# What only a real Home Assistant can show: the readings come from the recorder
# (`history.state_changes_during_period`, the state in force at the window's start
# included), the edit runs as a task-registry job under the entry's write lock,
# and the WS contract the panel reads (`get_cycle_context.next_start_s`) holds.
#
# One cycle, then a few readings under the Start Threshold after it ended (a
# dishwasher's drying beep, #531/#487): the cycle's end is moved over them and
# back, and a second cycle bounds how far it may go.
#
#   ./check_extend_end.sh          rebuild the box first (default)
#   ./check_extend_end.sh --reuse  reuse a running box
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python3"
HACTL="$PY $HERE/hactl.py"
SENSOR="sensor.washdata_power"

REUSE=0
[ "${1:-}" = "--reuse" ] && REUSE=1

cd "$HERE"
if [ "$REUSE" = 0 ]; then
  ./up.sh --fresh
elif ! curl -fsS -o /dev/null "http://127.0.0.1:8321/manifest.json" 2>/dev/null; then
  ./up.sh
fi
SINCE=$(date -u +%Y-%m-%dT%H:%M:%S)

echo "== one device, compressed timing (README, \"Time compression\")"
ENTRY=$($HACTL setup-device --name "Extend Check" --type washing_machine | tail -1)
echo "entry: $ENTRY"
$HACTL set-options "$ENTRY" \
  power_sensor="$SENSOR" \
  start_threshold_w=20 \
  stop_threshold_w=5 \
  start_duration_threshold=1 \
  watchdog_interval=5 \
  off_delay=10 \
  min_off_gap=10 \
  interrupted_min_seconds=5 \
  completion_min_seconds=5 > /dev/null

state() { $HACTL state sensor.extend_check_state | $PY -c 'import json,sys; print(json.load(sys.stdin)["state"])'; }

run_cycle() {
  $HACTL set "$SENSOR" 0; sleep 2
  for w in 250 300 320 310 300 280; do $HACTL set "$SENSOR" "$w"; sleep 3; done
  $HACTL set "$SENSOR" 0
  for _ in $(seq 1 20); do
    s=$(state); [ "$s" != "running" ] && [ "$s" != "ending" ] && [ "$s" != "paused" ] && [ "$s" != "starting" ] && break
    sleep 3
  done
  echo "  state: $(state)"
}

echo
echo "== a cycle, then readings under the Start Threshold after it ended"
run_cycle
sleep 3
for w in 3 12 3; do $HACTL set "$SENSOR" "$w"; sleep 4; done
$HACTL set "$SENSOR" 0
sleep 2

echo
echo "== extend, trim back, and bound by the next cycle"
FAIL=0
ENTRY="$ENTRY" HERE="$HERE" "$PY" - <<'PYEOF' || FAIL=1
import asyncio, os, sys
sys.path.insert(0, os.environ["HERE"])
import aiohttp
import hactl

ENTRY = os.environ["ENTRY"]
fails = 0


def check(label, ok, detail=""):
    global fails
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' (' + detail + ')') if detail else ''}")
    fails += 0 if ok else 1


async def run_task(ws, **msg):
    tid = (await ws.cmd("ha_washdata/trim_cycle", entry_id=ENTRY, **msg))["task_id"]
    for _ in range(60):
        snap = await ws.cmd("ha_washdata/get_task_result", task_id=tid)
        if snap["state"] in ("done", "error", "cancelled"):
            return snap
        await asyncio.sleep(0.5)
    return {"state": "timeout"}


def wh(samples):
    total = 0.0
    for (t0, p0), (t1, _p1) in zip(samples, samples[1:]):
        total += p0 * (t1 - t0) / 3600.0
    return total


async def main():
    async with aiohttp.ClientSession() as session, hactl.WS(session) as ws:
        cycles = (await ws.cmd("ha_washdata/get_device_cycles", entry_id=ENTRY))["cycles"]
        check("one cycle stored", len(cycles) == 1, f"{len(cycles)}")
        cid = cycles[0]["id"]
        before = await ws.cmd("ha_washdata/get_cycle_power_data", entry_id=ENTRY, cycle_id=cid)
        d0 = before["full_duration_s"]
        ctx = await ws.cmd("ha_washdata/get_cycle_context", entry_id=ENTRY, cycle_id=cid,
                           before_s=0, after_s=14400)
        after = [p for p in ctx["after"] if p[1] is not None and p[0] > d0]
        # The final 0 W may not be committed yet (the recorder commits every few s).
        check("the recorder holds the readings after the end",
              [p[1] for p in after if p[1] > 0] == [3.0, 12.0, 3.0], str(after))
        check("no next run yet", ctx["next_start_s"] is None)
        target = [p[0] for p in after if p[1] == 3.0][-1]

        snap = await run_task(ws, cycle_id=cid, start_s=0, end_s=target)
        check("the extension ran", snap["state"] == "done", str(snap.get("error")))
        ext = await ws.cmd("ha_washdata/get_cycle_power_data", entry_id=ENTRY, cycle_id=cid)
        check("the end moved onto the last reading", abs(ext["full_duration_s"] - target) < 0.6,
              f"{ext['full_duration_s']} vs {target}")
        values = [p[1] for p in ext["samples"] if p[0] > d0]
        check("the recorded readings were added", 12.0 in values and values[-1] == 3.0, str(values))
        check("energy grew by the added stretch", wh(ext["samples"]) > wh(before["samples"]))

        snap = await run_task(ws, cycle_id=cid, start_s=0, end_s=d0)
        check("trimmed back", snap["state"] == "done", str(snap.get("error")))
        back = await ws.cmd("ha_washdata/get_cycle_power_data", entry_id=ENTRY, cycle_id=cid)
        check("the round trip restores the trace", back["samples"] == before["samples"],
              f"{len(back['samples'])} vs {len(before['samples'])} samples")
        rows = {c["id"]: c for c in (await ws.cmd("ha_washdata/get_device_cycles", entry_id=ENTRY))["cycles"]}
        check("and its stored energy", abs(float(rows[cid].get("energy_wh") or 0) - float(cycles[0].get("energy_wh") or 0)) < 0.01,
              f"{rows[cid].get('energy_wh')} vs {cycles[0].get('energy_wh')}")
        return cid, d0


cid, d0 = asyncio.run(main())
with open(os.path.join(os.environ["HERE"], ".extend_check_state"), "w") as fh:
    fh.write(f"{cid} {d0}\n")
sys.exit(1 if fails else 0)
PYEOF

echo
echo "== a second cycle bounds the first one's end"
run_cycle
ENTRY="$ENTRY" HERE="$HERE" "$PY" - <<'PYEOF' || FAIL=1
import asyncio, os, sys
sys.path.insert(0, os.environ["HERE"])
import aiohttp
import hactl

ENTRY = os.environ["ENTRY"]
cid, d0 = open(os.path.join(os.environ["HERE"], ".extend_check_state")).read().split()
d0 = float(d0)


async def main():
    async with aiohttp.ClientSession() as session, hactl.WS(session) as ws:
        ctx = await ws.cmd("ha_washdata/get_cycle_context", entry_id=ENTRY, cycle_id=cid,
                           before_s=0, after_s=14400)
        nxt = ctx["next_start_s"]
        ok = nxt is not None and nxt > d0
        print(f"  {'PASS' if ok else 'FAIL'}  the context names the next run's start ({nxt})")
        tid = (await ws.cmd("ha_washdata/trim_cycle", entry_id=ENTRY, cycle_id=cid,
                            start_s=0, end_s=(nxt or d0) + 5))["task_id"]
        for _ in range(60):
            snap = await ws.cmd("ha_washdata/get_task_result", task_id=tid)
            if snap["state"] in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.5)
        refused = snap["state"] == "error" and snap.get("error") == "extend_overlaps"
        print(f"  {'PASS' if refused else 'FAIL'}  an end past it is refused ({snap['state']} {snap.get('error')})")
        data = await ws.cmd("ha_washdata/get_cycle_power_data", entry_id=ENTRY, cycle_id=cid)
        kept = abs(data["full_duration_s"] - d0) < 0.6
        print(f"  {'PASS' if kept else 'FAIL'}  and the cycle is untouched ({data['full_duration_s']})")
        return ok and refused and kept


sys.exit(0 if asyncio.run(main()) else 1)
PYEOF
rm -f "$HERE/.extend_check_state"

echo
echo "== Home Assistant log"
ERR=$($HACTL errors --since "$SINCE" 2>/dev/null | grep -i -E "ha_washdata|Traceback" || true)
if [ -z "$ERR" ]; then echo "  PASS  no ha_washdata errors"; else echo "  FAIL  $ERR"; FAIL=1; fi

echo
[ "$FAIL" = 0 ] && echo "extend-end checks passed" || { echo "extend-end checks FAILED"; exit 1; }
