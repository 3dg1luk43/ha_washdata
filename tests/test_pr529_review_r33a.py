"""Review of PR #529, round 33a: the findings, each reproduced.

* The batch pass's cycle list dropped every cycle with no trace before the batch
  kept its history for the Min Off Gap merge ceiling: a labelled run that ended
  100 s before the next load went with it, and the batch proposed a value over
  that recorded turnaround.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

from custom_components.ha_washdata.const import CONF_MIN_OFF_GAP
from custom_components.ha_washdata.learning import LearningManager
from tests.test_suggestion_stage2 import _timed_cycles


def _lm(hass: Any, past: list[dict[str, Any]]) -> LearningManager:
    hass.config_entries.async_get_entry.return_value = None
    store = MagicMock()
    store.get_past_cycles.return_value = past
    store.get_profiles.return_value = {}
    store.get_suggestions.return_value = {}
    return LearningManager(hass, "entry1", store, device_type="dishwasher")


def _untraced_before(cycle: dict[str, Any], turnaround_s: float) -> dict[str, Any]:
    """A labelled, completed run with no trace, ending ``turnaround_s`` before ``cycle``."""
    end = datetime.fromisoformat(cycle["start_time"]) - timedelta(seconds=turnaround_s)
    start = end - timedelta(seconds=3600)
    return {
        "id": "untraced", "profile_name": "Cotton", "status": "completed",
        "termination_reason": "timeout", "duration": 3600.0, "power_data": [],
        "start_time": start.isoformat(), "end_time": end.isoformat(),
    }


# ── The batch's merge ceiling keeps untraced runs ────────────────────────────


def test_the_batch_keeps_an_untraced_turnaround_for_the_merge_ceiling(mock_hass: Any) -> None:
    traced = _timed_cycles(6, gap_s=6 * 3600)   # ~2400 s bridge, hours of headroom
    alone = _lm(mock_hass, traced)
    sized = alone.suggestion_engine.for_job().run_batch_simulation(alone._batch_cycles())  # noqa: SLF001
    assert sized[CONF_MIN_OFF_GAP]["value"] > 100   # what the bridge alone asks for

    past = traced[:3] + [_untraced_before(traced[3], 100.0)] + traced[3:]
    lm = _lm(mock_hass, past)
    out = lm.suggestion_engine.for_job().run_batch_simulation(lm._batch_cycles())  # noqa: SLF001
    # No value bridges the cycle and stays under the recorded 100 s turnaround.
    assert CONF_MIN_OFF_GAP not in out


def test_an_untraced_run_is_no_new_evidence_for_the_batch_cadence(mock_hass: Any) -> None:
    """Kept for the ceiling, but four traced runs still do not start a batch."""
    traced = _timed_cycles(4, gap_s=6 * 3600)
    past = [_untraced_before(traced[0], 3600.0 * (k + 1)) for k in range(3)] + traced
    lm = _lm(mock_hass, past)
    lm._spawn = MagicMock(side_effect=lambda coro: coro.close())  # noqa: SLF001
    lm._maybe_run_batch_simulation()  # noqa: SLF001
    lm._spawn.assert_not_called()  # noqa: SLF001
    lm.profile_store.get_past_cycles.return_value = past + _timed_cycles(1, gap_s=0)
    lm._maybe_run_batch_simulation()  # noqa: SLF001
    lm._spawn.assert_called_once()  # noqa: SLF001
