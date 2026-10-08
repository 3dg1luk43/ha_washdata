# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Audit PLAYGROUND-11 residue (register item 490): two one-job steps were left.

After the CSV parser was stepped, the scan still ran ``build_scan`` (block finder,
gates, quiet-gap densification; ~0.7 s on a desktop at the 500k-row cap) and the
recorder's ``samples_from_readings`` (~0.4-0.7 s) as one executor job each. Both are
resumable objects now (``ScanBuilder``, ``RecorderReadings``) stepped a slice per job
like ``HistoryCsvParser``; the one-shot functions drive the same objects, so the
output is the same by construction, and these tests compare it anyway.
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from custom_components.ha_washdata import history_import as hi
from custom_components.ha_washdata import task_registry, ws_api
from custom_components.ha_washdata.cycle_detector import CycleDetectorConfig

sys.path.insert(0, str(Path(__file__).parent))
from test_ws_history_import import (  # noqa: E402
    _conn, _csv, _entry, _hass, _manager, _scan, _upload,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parent / "fixtures" / "history_import_sample.csv"


def _ts(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def _configs():
    yield CycleDetectorConfig(
        min_power=2.0, off_delay=180, device_type="washing_machine",
        stop_threshold_w=2.0, start_threshold_w=5.0, min_off_gap=600,
        completion_min_seconds=600,
    )
    yield CycleDetectorConfig(
        min_power=2.0, off_delay=300, device_type="dishwasher",
        stop_threshold_w=1.5, start_threshold_w=5.0, min_off_gap=3600,
        completion_min_seconds=1800,
    )
    yield CycleDetectorConfig(  # stop 0: the 1 W quiet fallback
        min_power=2.0, off_delay=60, device_type="washing_machine",
        stop_threshold_w=0.0, start_threshold_w=0.0, min_off_gap=60,
        completion_min_seconds=300,
    )


_VARIANTS = {
    "fixture": hi.parse_history_csv(FIXTURE.read_text(encoding="utf-8")).samples,
    "empty": [],
    "one": [(_ts(0), 5.0)],
    "all_breaks": [(_ts(i), None) for i in range(10)],
    "idle_only": [(_ts(5 * i), 0.5) for i in range(500)],
    "breaks": [
        (_ts(5 * i), None if i % 37 == 0 else (2000.0 if (i // 300) % 2 else 0.2))
        for i in range(5000)
    ],
    "long_hole": [(_ts(5 * i), 1500.0) for i in range(400)] + [(_ts(2000), None)]
    + [(_ts(90000 + 5 * i), 1500.0) for i in range(400)],
    "too_long": [(_ts(5 * i), 300.0) for i in range(12000)],
    "sparse": [(_ts(3600 * i), 800.0 if i % 2 else 0.0) for i in range(100)],
    "change_based": [(_ts(0), 0.0)] + [(_ts(600 + 5 * i), 1800.0) for i in range(600)]
    + [(_ts(4000), 0.0), (_ts(9000), 2000.0)]
    + [(_ts(9005 + 5 * i), 1700.0) for i in range(500)]
    + [(_ts(12000), 0.0), (_ts(80000), 0.0)],
}


def _view(result):
    """Everything a runner carries, then everything it produces."""
    if isinstance(result, dict):
        return result
    state = (result._streams, result.skipped, result.parse_report, result.total)
    while not result.finished:
        result.step(1000)
    return state, result.finalize()


@pytest.mark.parametrize("name", sorted(_VARIANTS))
def test_the_stepped_build_is_the_one_shot_build(name: str) -> None:
    samples = _VARIANTS[name]
    rng = random.Random(name)
    for config in _configs():
        for interval in (None, 5.0):
            one_shot = hi.build_scan(
                samples, config, sampling_interval_s=interval, parse_report={"p": 1}
            )
            builder = hi.ScanBuilder(
                samples, config, sampling_interval_s=interval, parse_report={"p": 1}
            )
            while not builder.finished:
                builder.step(rng.choice([1, 13, 500]))
            assert _view(builder.result()) == _view(one_shot)


def test_the_error_markers_are_unchanged() -> None:
    config = next(_configs())
    assert hi.ScanBuilder([], config).result() == {"error": "no_readings"}
    assert hi.ScanBuilder([(_ts(0), None)], config).result() == {"error": "no_readings"}
    idle = hi.ScanBuilder(_VARIANTS["idle_only"], config).result()
    assert idle["error"] == "no_usable_blocks" and idle["skipped"]
    # Junk never raises.
    assert hi.ScanBuilder([("junk",)], config).result() == {"error": "scan_failed"}


def test_the_build_takes_several_jobs_at_a_small_budget() -> None:
    samples = _VARIANTS["breaks"]
    builder = hi.ScanBuilder(samples, next(_configs()))
    jobs = 0
    while not builder.finished:
        spent = builder.step(1000)
        assert spent <= 1000 + 600  # a budget, overrun by at most one block
        jobs += 1
    assert jobs >= len(samples) // 1000


@pytest.mark.parametrize("step", [1, 7, 1000])
def test_stepped_recorder_conversion_is_the_one_shot_one(step: int) -> None:
    rows = [(T0.timestamp() + 5 * (i % 97) + i // 97, float(i % 300)) for i in range(2000)]
    rows += [(None, 1.0), (1.0, "x"), (2.0, float("nan")), (3.0, -4.0)]
    conv = hi.RecorderReadings(rows)
    jobs = 0
    while not conv.finished:
        conv.step(step)
        jobs += 1
    assert conv.result() == hi.samples_from_readings(rows)
    assert jobs == -(-len(rows) // step)


@pytest.mark.asyncio
async def test_the_scan_builds_a_slice_per_executor_job() -> None:
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    token = await _upload(hass, conn, manager, _csv())
    steps: list[int] = []
    real_step = hi.ScanBuilder.step

    def _counted(self, n=hi.SCAN_BUILD_STEP_SAMPLES):
        steps.append(n)
        return real_step(self, n)

    with patch.object(hi, "SCAN_BUILD_STEP_SAMPLES", 200), \
         patch.object(hi.ScanBuilder, "step", _counted):
        task = await _scan(hass, conn, manager, entry, token)
    assert task.state == task_registry.STATE_DONE
    assert task.result["found"] == 2
    # ~960 samples at 200 a job: several build jobs, not one.
    assert len(steps) >= 5 and set(steps) == {200}


@pytest.mark.asyncio
async def test_a_recorder_scan_converts_a_slice_per_executor_job() -> None:
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()

    async def _fake_recorder(_hass, _entity, start_dt, *, end_dt=None):
        base = start_dt.timestamp()
        return [(base + 5 * i, 1800.0 if i % 400 < 300 else 0.0) for i in range(1200)]

    with patch.object(ws_api, "_get_manager", return_value=manager), \
         patch.object(ws_api, "_recorder_power", side_effect=_fake_recorder):
        await ws_api.ws_history_import_recorder.__wrapped__(
            hass, conn, {"id": 1, "entry_id": "e", "days": 2}
        )
    token = conn.send_result.call_args.args[1]["token"]
    steps: list[int] = []
    real_step = hi.RecorderReadings.step

    def _counted(self, n=hi.PARSE_STEP_ROWS):
        steps.append(n)
        return real_step(self, n)

    with patch.object(hi, "PARSE_STEP_ROWS", 300), \
         patch.object(hi.RecorderReadings, "step", _counted):
        task = await _scan(hass, conn, manager, entry, token)
    assert task.state == task_registry.STATE_DONE
    assert task.result["parse"]["source"] == "recorder"
    # 2400 rows at 300 a job.
    assert len(steps) == 8 and set(steps) == {300}


@pytest.mark.asyncio
async def test_a_cancel_during_the_build_stops_the_scan() -> None:
    hass, conn = _hass(), _conn()
    manager, entry = _manager(hass), _entry()
    token = await _upload(hass, conn, manager, _csv())
    real_step = hi.ScanBuilder.step
    reg = task_registry.get_registry(hass)

    def _cancel_after_one(self, n=hi.SCAN_BUILD_STEP_SAMPLES):
        for snap in reg.snapshot("e"):  # the user presses Cancel mid-build
            reg.cancel(snap["id"])
        return real_step(self, n)

    with patch.object(hi, "SCAN_BUILD_STEP_SAMPLES", 50), \
         patch.object(hi.ScanBuilder, "step", _cancel_after_one):
        task = await _scan(hass, conn, manager, entry, token)
    assert task.state == task_registry.STATE_CANCELLED
    assert "segments" not in (task.result or {})
