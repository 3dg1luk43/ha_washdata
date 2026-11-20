"""Review of PR #529, round 40b, each reproduced.

* The end-gate guard read a wait of 0 (off_delay and min_off_gap both 0) as no
  gate, so Apply accepted it; live, the first quiet reading in ENDING completes
  the cycle, splitting it at any pause that gets there. An off_delay of 0 is a
  one-reading energy window, which nothing holds.
* The kept standby tail stopped at a synthetic reading at working power, so a
  real 3 W tail behind a 20 W keepalive was cut and the 6 W after it read as the
  idle level: two such cycles raised "still drawing" at a 5 W stop.
* The joint end-gate check withdrew a pending Start with every withdrawn Stop,
  though a 50 W Start correction is valid against the 4 W Stop in force.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

from custom_components.ha_washdata.const import (
    CONF_DEVICE_TYPE,
    DOMAIN,
    CONF_END_ENERGY_THRESHOLD,
    CONF_MIN_OFF_GAP,
    CONF_OFF_DELAY,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.learning import LearningManager
from custom_components.ha_washdata.suggestion_engine import (
    SuggestionEngine,
    detect_standby_above_stop,
    end_gate_would_split,
)
from custom_components.ha_washdata.time_utils import SYNTHETIC_OFFSETS_KEY

from .test_pr529_review_r35a import _Store

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _paused_trace(pause_w: float, pause_s: int, step: int = 10) -> list[tuple[float, float]]:
    """300 W for 20 min, a pause at ``pause_w``, 300 W for 20 min, then 0 W."""
    pts: list[tuple[float, float]] = []
    t = 0
    for _ in range(1200 // step):
        pts.append((float(t), 300.0))
        t += step
    for _ in range(pause_s // step):
        pts.append((float(t), pause_w))
        t += step
    for _ in range(1200 // step):
        pts.append((float(t), 300.0))
        t += step
    for _ in range(2000 // step):
        pts.append((float(t), 0.0))
        t += step
    return pts


def _replay(options: dict[str, Any], pts: list[tuple[float, float]], device_type: str) -> list[Any]:
    """The cycles the shipped detector stores for ``pts`` under ``options``."""
    cfg = build_detector_config(options, {}, device_type)
    ends: list[Any] = []
    det = CycleDetector(cfg, lambda *_a: None, ends.append)
    for offset, watts in pts:
        det.process_reading(watts, T0 + timedelta(seconds=offset))
    return ends


# ── A wait of 0 is a gate ────────────────────────────────────────────────────


_ZERO_OPTS = {
    CONF_STOP_THRESHOLD_W: 2.0, CONF_START_THRESHOLD_W: 5.0, "min_power": 3.0,
    CONF_END_ENERGY_THRESHOLD: 0.05,
}


def test_a_zero_wait_splits_live_and_the_guard_says_so() -> None:
    """off_delay 0 and min_off_gap 0: the detector stores two cycles for one with a
    5 min pause at 1 W, and the guard reports it. 600 s each bridges it, in both."""
    pts = _paused_trace(1.0, 300)
    zero = {**_ZERO_OPTS, CONF_OFF_DELAY: 0, CONF_MIN_OFF_GAP: 0}
    assert len(_replay(zero, pts, "washing_machine")) == 2
    assert end_gate_would_split(pts, 2.0, 0, 0, 0.05)
    bridged = {**_ZERO_OPTS, CONF_OFF_DELAY: 600, CONF_MIN_OFF_GAP: 600}
    assert len(_replay(bridged, pts, "washing_machine")) == 1
    assert not end_gate_would_split(pts, 2.0, 600, 600, 0.05)


def test_a_zero_off_delay_holds_no_energy() -> None:
    """off_delay 0 under a 60 s gap: the one-reading window passes the energy gate,
    so the 300 s pause splits live; under a 600 s gap it does not, in both."""
    pts = _paused_trace(1.0, 300)
    short = {**_ZERO_OPTS, CONF_OFF_DELAY: 0, CONF_MIN_OFF_GAP: 60}
    assert len(_replay(short, pts, "washing_machine")) == 2
    assert end_gate_would_split(pts, 2.0, 0, 60, 0.05)
    long_gap = {**_ZERO_OPTS, CONF_OFF_DELAY: 0, CONF_MIN_OFF_GAP: 600}
    assert len(_replay(long_gap, pts, "washing_machine")) == 1
    assert not end_gate_would_split(pts, 2.0, 0, 600, 0.05)


def test_a_zero_wait_without_a_pause_splits_nothing() -> None:
    """No pause, only the trailing quiet: a wait of 0 splits nothing."""
    pts = _paused_trace(1.0, 0)
    assert len(_replay({**_ZERO_OPTS, CONF_OFF_DELAY: 0, CONF_MIN_OFF_GAP: 0}, pts, "washing_machine")) == 1
    assert not end_gate_would_split(pts, 2.0, 0, 0, 0.05)


# ── A synthetic reading does not cut the kept tail ───────────────────────────


def _stopped_by_hand(cid: str) -> dict[str, Any]:
    """200 W for 10 min, 30 min at 3 W, a 20 W keepalive WashData wrote, then
    two minutes at 6 W until the user stopped it."""
    rows = [[float(t), 200.0] for t in range(0, 600, 10)]
    rows += [[float(t), 3.0] for t in range(600, 2400, 10)]
    rows.append([2400.0, 20.0])
    rows += [[float(t), 6.0] for t in range(2410, 2530, 10)]
    return {
        "id": cid, "status": "interrupted", "termination_reason": "user",
        "power_data": rows, SYNTHETIC_OFFSETS_KEY: [2400.0],
    }


def test_a_synthetic_reading_does_not_cut_the_standby_tail() -> None:
    """At a 5 W stop the tail settled at 3 W, below it; cut at the keepalive it
    read 6 W, and two such cycles raised the advisory."""
    cycles = [_stopped_by_hand("a"), _stopped_by_hand("b")]
    assert detect_standby_above_stop(cycles, 5.0) is None
    # A real 20 W reading still ends the tail: the 6 W after it is the evidence.
    for cycle in cycles:
        cycle[SYNTHETIC_OFFSETS_KEY] = []
    found = detect_standby_above_stop(cycles, 5.0)
    assert found is not None and found["idle_w"] == 6.0


# ── A withdrawn Stop leaves a Start valid against the Stop in force ──────────


_PAIR_OPTS = {
    CONF_DEVICE_TYPE: "washing_machine", CONF_STOP_THRESHOLD_W: 4.0,
    CONF_START_THRESHOLD_W: 120.0, CONF_OFF_DELAY: 180, CONF_MIN_OFF_GAP: 300,
    CONF_END_ENERGY_THRESHOLD: 0.25,
}


def _pair_engine(pause_w: float, pause_s: int, opts: dict[str, Any]) -> tuple[SuggestionEngine, _Store]:
    pts = _paused_trace(pause_w, pause_s)
    cycle = {
        "id": "w1", "status": "completed", "termination_reason": "timeout",
        "power_data": [[t, p] for t, p in pts],
    }
    store = _Store(cycles=[cycle])
    eng = SuggestionEngine(MagicMock(), "e", store, device_type="washing_machine")
    eng._entry_options = lambda: dict(opts)  # noqa: SLF001
    return eng, store


def test_a_withdrawn_stop_keeps_a_start_valid_against_the_stop_in_force() -> None:
    """A 5 W Stop makes the 400 s pause at 4.5 W quiet and it splits (the 4 W in
    force bridges it, live too); the 50 W Start correction stays above 4 W."""
    pts = _paused_trace(4.5, 400)
    assert len(_replay(_PAIR_OPTS, pts, "washing_machine")) == 1
    assert len(_replay({**_PAIR_OPTS, CONF_STOP_THRESHOLD_W: 5.0}, pts, "washing_machine")) == 2
    eng, _store = _pair_engine(4.5, 400, _PAIR_OPTS)
    stored = {
        CONF_STOP_THRESHOLD_W: {"value": 5.0, "reason": "r", "corrective": True},
        CONF_START_THRESHOLD_W: {"value": 50.0, "reason": "r", "corrective": True},
    }
    assert eng.joint_end_gate_drops(stored) == [CONF_STOP_THRESHOLD_W]
    assert eng.joint_end_gate_drops(stored, strict=True) == [CONF_STOP_THRESHOLD_W]


def test_a_start_derived_from_the_withdrawn_stop_goes_with_it() -> None:
    """A Start cascade exists only to sit above the Stop it follows."""
    eng, _store = _pair_engine(4.5, 400, _PAIR_OPTS)
    stored = {
        CONF_STOP_THRESHOLD_W: {"value": 5.0, "reason": "r", "corrective": True},
        CONF_START_THRESHOLD_W: {"value": 125.0, "reason": "r", "cascade": True},
    }
    assert eng.joint_end_gate_drops(stored) == [CONF_STOP_THRESHOLD_W, CONF_START_THRESHOLD_W]


def test_a_start_the_stop_in_force_inverts_goes_with_the_stop() -> None:
    """A 2 W Stop makes a 1200 s pause at 1.8 W near the stop, so the flat-standby
    bypass splits it; the 3 W Start beside it would sit under the 4 W in force."""
    opts = {**_PAIR_OPTS, CONF_END_ENERGY_THRESHOLD: 0.05}
    eng, _store = _pair_engine(1.8, 1200, opts)
    stored = {
        CONF_STOP_THRESHOLD_W: {"value": 2.0, "reason": "r"},
        CONF_START_THRESHOLD_W: {"value": 3.0, "reason": "r"},
    }
    assert eng.joint_end_gate_drops(stored) == [CONF_STOP_THRESHOLD_W, CONF_START_THRESHOLD_W]


async def test_the_pass_check_leaves_the_start_stored(mock_hass: Any) -> None:
    """After every pass: the Stop is withdrawn, the reconcile after it keeps the
    Start."""
    store = _Store(cycles=[{
        "id": "w1", "status": "completed", "termination_reason": "timeout",
        "power_data": [[t, p] for t, p in _paused_trace(4.5, 400)],
    }])
    entry = MagicMock()
    entry.data = {}
    entry.options = dict(_PAIR_OPTS)
    mock_hass.config_entries.async_get_entry.return_value = entry
    mock_hass.async_create_task = lambda coro, *_a: coro.close()
    lm = LearningManager(mock_hass, "e", store, device_type="washing_machine")
    store.set_suggestion(CONF_STOP_THRESHOLD_W, 5.0, reason="r", flags={"corrective": True})
    store.set_suggestion(CONF_START_THRESHOLD_W, 50.0, reason="r", flags={"corrective": True})
    await lm._enforce_joint_end_gate(strict=True)  # noqa: SLF001
    assert CONF_STOP_THRESHOLD_W not in store.suggestions
    assert store.suggestions[CONF_START_THRESHOLD_W]["value"] == 50.0


async def test_apply_all_writes_the_start_its_withdrawn_stop_leaves_valid(
    hass: Any, setup_washdata_entry: Any, hass_ws_client: Any
) -> None:
    """Apply all on a real entry: the 5 W Stop is withdrawn, the 50 W Start is
    written, and the pair check passes against the 4 W Stop kept."""
    opts = {k: v for k, v in _PAIR_OPTS.items() if k != CONF_DEVICE_TYPE}
    entry = await setup_washdata_entry(options=opts)
    store = hass.data[DOMAIN][entry.entry_id].profile_store
    store.set_suggestion(CONF_STOP_THRESHOLD_W, 5.0, reason="r", flags={"corrective": True})
    store.set_suggestion(CONF_START_THRESHOLD_W, 50.0, reason="r", flags={"corrective": True})
    client = await hass_ws_client(hass)
    with patch.object(
        SuggestionEngine, "_end_gate_evidence", return_value=[_paused_trace(4.5, 400)]
    ):
        await client.send_json_auto_id({
            "type": f"{DOMAIN}/apply_suggestions", "entry_id": entry.entry_id,
            "keys": [CONF_STOP_THRESHOLD_W, CONF_START_THRESHOLD_W],
        })
        res = await client.receive_json()
    assert res["success"], res
    assert res["result"]["applied"] == [CONF_START_THRESHOLD_W]
    await hass.async_block_till_done()
    options = hass.config_entries.async_get_entry(entry.entry_id).options
    assert options[CONF_START_THRESHOLD_W] == 50.0
    assert options[CONF_STOP_THRESHOLD_W] == 4.0
