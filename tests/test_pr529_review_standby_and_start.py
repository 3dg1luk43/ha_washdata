"""Review of PR #529: the standby advisory's evidence and the steady-load Start fix.

* A legacy trace stamped with ISO times weighed its kept tail at None, so a hand
  stop read its last sample alone (a 6 W pulse over a settled 3 W standby).
* A Stop Threshold row recorded ``old=None`` while Stop was unset and followed Min
  Power, so the cycles that ended under that effective Stop counted as ending
  below a lowered one and outvoted the hand stops showing the standby.
* On a kW sensor the steady-load correction rounded a 0.08 running level to 0.1
  and proposed Start 0.1, above the load it was meant to start on.
* The device-list poll reads the settled-tail cache the manager fills in the
  executor at setup, so a restart does not sort the tails on the event loop.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ha_washdata import suggestion_engine as se
from custom_components.ha_washdata import ws_api
from custom_components.ha_washdata.const import (
    CONF_MIN_POWER,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DOMAIN,
    STORAGE_KEY,
    STORAGE_VERSION,
)

T0 = datetime(2026, 11, 1, 8, tzinfo=timezone.utc)


def _hand_stop(day: int, *, iso: bool, hours: float = 0.5) -> dict[str, Any]:
    """Ten minutes at 2 kW, then ``hours`` of a 3 W standby, stopped by hand on a 6 W pulse."""
    start = T0 + timedelta(days=day)
    pts: list[list[Any]] = [[10.0 * i, 2000.0] for i in range(60)]
    pts += [[600.0 + 10.0 * i, 3.0] for i in range(int(hours * 360))]
    pts.append([pts[-1][0] + 10.0, 6.0])
    if iso:
        pts = [[(start + timedelta(seconds=t)).isoformat(), w] for t, w in pts]
        end_s = 600.0 + 10.0 * int(hours * 360)
    else:
        end_s = pts[-1][0]
    return {
        "id": f"h{day}", "status": "completed", "termination_reason": "user",
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=end_s)).isoformat(),
        "duration": end_s, "power_data": pts, "signature": {"max_power": 2000.0},
    }


@pytest.mark.parametrize("iso", [False, True])
def test_a_legacy_iso_trace_weighs_its_tail_like_an_offset_one(iso: bool) -> None:
    se._TAIL_SETTLED_CACHE.clear()  # noqa: SLF001
    cycles = [_hand_stop(0, iso=iso), _hand_stop(1, iso=iso)]
    assert se._idle_evidence_w(cycles[0], 5.0) == 3.0  # noqa: SLF001
    # Settled at 3 W under a 5 W Stop: below it, so no "still drawing" advisory.
    assert se.detect_standby_above_stop(cycles, 5.0) is None
