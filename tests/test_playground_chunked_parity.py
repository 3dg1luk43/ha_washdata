"""A Playground replay run in chunks must equal the same replay run in one go.

The background task (`ws_api`) drives `_DetailSim.step` across many small executor
jobs so the event loop breathes on long cycles (#311); the one-shot path calls
`step(0, n)` once. `_simulate_cycle_detail_inner` claimed a golden test for this,
but none existed (register item 462).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata import playground, ws_api
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig
from custom_components.ha_washdata.profile_store import ProfileStore

BASE = datetime(2026, 9, 1, 8, 0, 0, tzinfo=timezone.utc)


def _wash(scale: float = 1.0) -> list[list[float]]:
    """Heat, a quiet soak, agitation, a spin: enough structure to match and pause."""
    pts: list[list[float]] = []
    t = 0.0
    for watts, secs in ((2000 * scale, 900), (3, 600), (300, 1200), (600, 300), (0, 60)):
        for _ in range(int(secs // 30)):
            pts.append([t, float(watts)])
            t += 30.0
    return pts


def _store() -> ProfileStore:
    store = ProfileStore(MagicMock(), "pg-chunked")
    store._data = {
        "profiles": {
            "Cotton": {"avg_duration": 3060.0, "sample_cycle_id": "a"},
            "Synthetics": {"avg_duration": 3060.0, "sample_cycle_id": "b"},
        },
        "past_cycles": [
            {"id": "a", "profile_name": "Cotton", "status": "completed", "duration": 3060.0,
             "start_time": BASE.isoformat(), "power_data": _wash(1.0)},
            {"id": "b", "profile_name": "Synthetics", "status": "completed", "duration": 3060.0,
             "start_time": BASE.isoformat(), "power_data": _wash(0.4)},
        ],
        "envelopes": {},
    }
    return store


def _run(chunk: int | None) -> dict:
    cycle = {"id": "c1", "duration": 3060.0, "status": "completed", "profile_name": "Cotton",
             "start_time": BASE.isoformat(), "power_data": _wash(0.95)}
    cfg = CycleDetectorConfig(min_power=10.0, off_delay=180, stop_threshold_w=5.0,
                              completion_min_seconds=60)
    sim = playground._DetailSim(cycle, cfg, None, _store(), {}, 0.3)
    assert sim.ready
    if chunk is None:
        sim.step(0, sim.n_readings)
    else:
        for i in range(0, sim.n_readings, chunk):
            sim.step(i, i + chunk)
    sim.run_tail()
    return json.loads(json.dumps(sim.finalize(), default=str))


@pytest.mark.parametrize("chunk", [1, 7, ws_api._PG_DETAIL_CHUNK])  # noqa: SLF001
def test_a_chunked_replay_equals_the_one_shot_replay(chunk: int) -> None:
    one_shot = _run(None)
    assert one_shot["outcome"]["detected"] is True
    assert one_shot["series"] and one_shot["events"]
    assert _run(chunk) == one_shot
