#!/usr/bin/env bash
# Fail when the box's Home Assistant log shows a WashData problem a check's own
# assertions can pass over: an ERROR from ha_washdata, a traceback, or a
# deprecated-API report ("Detected that custom integration 'ha_washdata' ..."),
# which HA logs as a WARNING and is how #532 reached users. Run it right after a
# check: the next `up.sh --fresh` wipes the log.
#
#   ./log_check.sh            check config/home-assistant.log
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/config/home-assistant.log"
if [ ! -f "$LOG" ]; then
  echo "  FAIL  no log at $LOG"
  exit 1
fi

FAIL=0
report() {  # report <label> <lines>
  if [ -z "$2" ]; then
    echo "  PASS  $1"
  else
    echo "  FAIL  $1"
    printf '%s\n' "$2" | tail -8 | sed 's/^/        /'
    FAIL=1
  fi
}

report "no deprecated Home Assistant API reported" \
  "$(grep -E "custom integration 'ha_washdata'" "$LOG" || true)"
report "no ha_washdata errors" "$(grep -E "ERROR.*ha_washdata" "$LOG" || true)"
report "no tracebacks" "$(grep -n "Traceback (most recent call last)" "$LOG" || true)"
exit "$FAIL"
