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
"""Stage 4 tests: on-device NumPy-only training (training_task).

Covers the training pipeline (total_energy only since 0.5.8). The logistic spec
scoring and the engine's on-device classifier preference went with the on-device
classifier heads (register item 518).
"""
from __future__ import annotations

from custom_components.ha_washdata.ml.training_task import train_from_cycles


# ---------------------------------------------------------------------------
# training_task.py - label derivation + gating
# ---------------------------------------------------------------------------


def _trace(peak=1000.0, dur=3600.0, n=180, pause_frac=0.2, pause_len=120.0, flat_off=True):
    step = dur / (n - 1)
    ps = pause_frac * dur
    pts = []
    for i in range(n):
        t = i * step
        frac = i / (n - 1)
        if ps <= t <= ps + pause_len:
            p = 0.0
        elif frac < 0.1:
            p = peak * (frac / 0.1)
        elif frac > 0.85:
            p = peak * max(0.0, (0.9 - frac) / 0.05)
        else:
            p = peak
        pts.append([round(t, 1), round(max(p, 0.0), 1)])
    if flat_off:
        for k in range(1, 7):
            pts.append([round(dur + k * 20, 1), 0.0])
    return pts


def _completed(i):
    return {
        "id": f"c{i}", "status": "completed", "profile_name": "Cotton", "duration": 3600.0,
        "energy_wh": 800.0, "max_power": 1000.0, "match_confidence": 0.85,
        "power_data": _trace(pause_frac=0.15 + 0.01 * (i % 5)),
        "start_time": "2026-01-01T10:00:00+00:00",
    }


def _force_stopped(i):
    return {
        "id": f"f{i}", "status": "force_stopped", "profile_name": "Cotton", "duration": 1800.0,
        "energy_wh": 400.0, "max_power": 1000.0, "match_confidence": 0.5,
        "power_data": _trace(dur=1800.0, flat_off=False),
        "start_time": "2026-01-01T10:00:00+00:00",
    }


def test_only_the_energy_head_is_trained() -> None:
    """Since 0.5.8 only total_energy is trained on-device. The quality and
    live_match heads went with their consumers (audit ML-02/06/10); end and
    remaining_time stopped training because their consumers are frozen off and
    the classifier gate promoted worse models (ML-11). Data that used to promote
    an `end` model (it did, on exactly these cycles) yields no record for it."""
    cycles = [_completed(i) for i in range(30)] + [_force_stopped(i) for i in range(25)]
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-01T02:00:00+00:00")
    assert {r["capability"] for r in summary["results"]} == {"total_energy"}
    assert set(summary["promoted"]) <= {"total_energy"}


def test_training_skips_when_too_few_rows() -> None:
    # Four completed cycles: too few prefix rows for the energy head.
    cycles = [_completed(i) for i in range(4)]
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-01T02:00:00+00:00")
    (record,) = summary["results"]
    assert record["promoted"] is False
    assert "insufficient" in record["reason"]
    assert summary["promoted"] == {}


def test_training_result_shape() -> None:
    cycles = [_completed(i) for i in range(30)] + [_force_stopped(i) for i in range(25)]
    summary = train_from_cycles(cycles, "washing_machine", 2.0, "2026-07-01T02:00:00+00:00")
    assert set(summary) == {"results", "promoted"}
    for r in summary["results"]:
        assert "capability" in r and "promoted" in r
