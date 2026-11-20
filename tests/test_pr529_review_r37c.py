"""Review of PR #529, round 37c: the Minor findings, each reproduced.

* A hand-stopped trace ending on a synthetic 0 W row (a watchdog keepalive while
  the sensor was unavailable) read as idling below the Stop Threshold: the
  settled tail skipped the row, the final-sample check did not.
"""
from __future__ import annotations

from typing import Any

from custom_components.ha_washdata.suggestion_engine import (
    _idle_evidence_w,
    detect_standby_above_stop,
)

# ── A keepalive row is not the last sample ───────────────────────────────────


def _hand_stopped(tail: list[float], synthetic_last: int = 0) -> dict[str, Any]:
    """Ten minutes at 500 W, then ``tail`` at 10 s steps; stopped by hand. The last
    ``synthetic_last`` rows are marked as keepalives WashData wrote."""
    pts = [[float(t), 500.0] for t in range(0, 600, 10)]
    pts += [[600.0 + 10.0 * i, p] for i, p in enumerate(tail)]
    cycle: dict[str, Any] = {
        "status": "completed", "termination_reason": "user", "power_data": pts,
        "signature": {"max_power": 500.0},
    }
    if synthetic_last:
        cycle["synthetic_offsets"] = [p[0] for p in pts[-synthetic_last:]]
    return cycle


def test_a_trailing_keepalive_row_does_not_hide_a_standby_above_stop() -> None:
    above = _hand_stopped([3.4] * 30)
    ka = _hand_stopped([3.4] * 30 + [0.0], synthetic_last=1)
    assert _idle_evidence_w(ka, 2.56) == 3.4
    found = detect_standby_above_stop([above, above, ka, ka], 2.56)
    assert found is not None and found["cycles_above"] == 4
    # Several keepalives in a row: the last real reading is the one before them.
    assert _idle_evidence_w(_hand_stopped([3.4] * 30 + [0.0] * 3, synthetic_last=3), 2.56) == 3.4
    # Only keepalives after the work: nothing real to read as the idle level.
    assert _idle_evidence_w(_hand_stopped([0.0] * 3, synthetic_last=3), 2.56) is None


def test_a_real_final_zero_stays_evidence_below_stop() -> None:
    above = _hand_stopped([3.4] * 30)
    # The plug itself reported 0 W at the end: unmarked, still a stop below.
    real_zero = _hand_stopped([3.4] * 30 + [0.0])
    assert _idle_evidence_w(real_zero, 2.56) == 0.0
    # A keepalive earlier in the tail does not make the real final 0 W synthetic.
    early_ka = _hand_stopped([3.4] * 15 + [0.0] + [3.4] * 14 + [0.0])
    early_ka["synthetic_offsets"] = [600.0 + 10.0 * 15]
    assert _idle_evidence_w(early_ka, 2.56) == 0.0
    assert detect_standby_above_stop([above, above, real_zero, real_zero], 2.56) is None


def test_a_legacy_iso_stamped_trace_reads_its_last_real_row() -> None:
    """ISO-stamped rows have no numeric offset: the normalised trace names them."""
    from datetime import datetime, timedelta, timezone  # noqa: PLC0415

    start = datetime(2026, 10, 1, 8, tzinfo=timezone.utc)
    cycle = _hand_stopped([3.4] * 30 + [0.0], synthetic_last=1)
    cycle["start_time"] = start.isoformat()
    cycle["power_data"] = [
        [(start + timedelta(seconds=t)).isoformat(), p] for t, p in cycle["power_data"]
    ]
    assert _idle_evidence_w(cycle, 2.56) == 3.4
