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
"""Register item 469(a): the terminal signature took a heating block for the event.

Shape of the contributed dishwasher 01KGM619's "Eco": heating blocks 120-160 s
apart until ~6300 s, ~4850 s of passive drying, then a ~25 W pump-out. Most stored
cycles were closed before the pump-out, so their last run above the signature's
threshold is the last HEATING block, 130 s after the previous one, which
``TERMINAL_QUIET_MIN_S`` (120 s) let through as the terminal event. The median of
4850 s pump-out quiets and 130 s heating gaps read 2500 s (element 11), and those
truncated cycles also counted as cycles that "did not show the event".

Now a last run above ``TERMINAL_EVENT_MAX_PEAK_FRAC`` of the cycle's peak is main
activity, and a cycle with no event is censored unless it stayed quiet at least as
long as the measured quiet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore

PUMP_OUT_QUIET = 4850.0


@pytest.fixture
def store() -> ProfileStore:
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(MagicMock(), "e")
    ps._data["profiles"] = {"Eco": {"avg_duration": 11100.0}}
    return ps


def _eco(cid: str, *, tail_s: float | None, pump_out: bool) -> dict[str, Any]:
    """Two heating blocks 130 s apart, drying at 0.8 W, then a pump-out or not.

    ``tail_s`` is how long the stored trace runs past the last heating block when
    there is no pump-out (a cycle closed early).
    """
    pts: list[list[float]] = []
    # Heating 0-3000 s (2200 W with 60 W circulation), a 130 s gap, heating again.
    for t in range(0, 3000, 30):
        pts.append([float(t), 2200.0 if (t // 300) % 2 == 0 else 60.0])
    for t in range(3000, 3130, 30):
        pts.append([float(t), 0.8])
    for t in range(3130, 6300, 30):
        pts.append([float(t), 2200.0 if (t // 300) % 2 == 0 else 60.0])
    last_heat = pts[-1][0]
    end = last_heat + (PUMP_OUT_QUIET if pump_out else float(tail_s or 0.0))
    t = last_heat + 30.0
    while t < end:
        pts.append([t, 0.8])
        t += 30.0
    if pump_out:
        pts += [[end, 28.5], [end + 30.0, 20.3], [end + 60.0, 0.8]]
    else:
        pts.append([end, 0.8])
    return {
        "id": cid,
        "profile_name": "Eco",
        "status": "completed",
        "duration": pts[-1][0],
        "power_data": pts,
    }


def test_a_cycle_closed_before_its_pump_out_offers_no_event(store: ProfileStore) -> None:
    """Three pump-out cycles and three closed after 4500 s of drying.

    Before the fix the closed ones contributed their 130 s heating gap: median
    quiet ~2490 s, a 2200 W "event", consistency 6/6 in name only.
    """
    store._data["past_cycles"] = [
        _eco(f"p{i}", tail_s=None, pump_out=True) for i in range(3)
    ] + [_eco(f"c{i}", tail_s=4500.0, pump_out=False) for i in range(3)]

    sig = store.compute_profile_terminal_signature("Eco")

    assert sig is not None
    assert sig["quiet_before_s"] == pytest.approx(PUMP_OUT_QUIET, abs=60.0)
    assert sig["event_watts"] < 50.0, "the heating block was taken for the event"
    assert sig["seen_in"] == 3
    # Closed 350 s before the pump-out was due: censored, not an absence.
    assert sig["measured"] == 3 and sig["censored"] == 3
    assert sig["consistency"] == 1.0
    assert store.profile_terminal_quiet_seconds("Eco") == pytest.approx(
        PUMP_OUT_QUIET, abs=60.0
    )


def test_a_cycle_quiet_for_the_whole_wait_without_an_event_is_an_absence(
    store: ProfileStore,
) -> None:
    """The other half of the rule: a trace that waited the measured quiet and saw
    nothing is the case ``consistency`` exists to report, so it stays counted."""
    store._data["past_cycles"] = [
        _eco(f"p{i}", tail_s=None, pump_out=True) for i in range(3)
    ] + [_eco("long", tail_s=PUMP_OUT_QUIET + 600.0, pump_out=False)]

    sig = store.compute_profile_terminal_signature("Eco")

    assert sig is not None
    assert sig["seen_in"] == 3 and sig["measured"] == 4 and sig["censored"] == 0
    assert sig["consistency"] == pytest.approx(0.75)


def test_most_cycles_closed_early_no_longer_hide_the_measurement(
    store: ProfileStore,
) -> None:
    """01KGM619's proportions: 6 pump-outs, 11 closed early, 3 quiet long enough.

    Counting every closed cycle as an absence read 6/20 = 0.3, under the 0.6 the
    accessor needs, so element 11 would vanish. Censored, it reads 6/9.
    """
    store._data["past_cycles"] = (
        [_eco(f"p{i}", tail_s=None, pump_out=True) for i in range(6)]
        + [_eco(f"c{i}", tail_s=3500.0 + 100.0 * i, pump_out=False) for i in range(11)]
        + [_eco(f"l{i}", tail_s=PUMP_OUT_QUIET + 200.0, pump_out=False) for i in range(3)]
    )

    sig = store.compute_profile_terminal_signature("Eco")

    assert sig is not None
    assert (sig["seen_in"], sig["measured"], sig["censored"]) == (6, 9, 11)
    assert store.profile_terminal_quiet_seconds("Eco") == pytest.approx(
        PUMP_OUT_QUIET, abs=60.0
    )


def test_a_high_power_last_run_is_main_activity_whatever_the_gap(
    store: ProfileStore,
) -> None:
    """A full-power block after a long pause is the wash resuming, not a pump-out."""
    def _resumed(cid: str) -> dict[str, Any]:
        pts = [[float(t), 2000.0] for t in range(0, 600, 30)]
        pts += [[float(t), 0.4] for t in range(600, 1500, 30)]
        pts += [[float(t), 1900.0] for t in range(1500, 1800, 30)]
        pts.append([1800.0, 0.0])
        return {"id": cid, "profile_name": "Eco", "status": "completed",
                "duration": 1800.0, "power_data": pts}

    store._data["past_cycles"] = [_resumed(f"r{i}") for i in range(4)]

    sig = store.compute_profile_terminal_signature("Eco")

    assert sig is not None
    assert sig["seen_in"] == 0 and sig["quiet_before_s"] is None


_REPO = Path(__file__).resolve().parent.parent
_HAYES = "config_entry-ha_washdata-01KGM619VSVQEKXW6V5B8FE8BG"


@pytest.mark.slow
def test_the_contributed_eco_reads_its_pump_out_quiet() -> None:
    """01KGM619 itself: 2500.1 s (12/20) before, the 4840-4860 s pump-out quiet now."""
    hits = [p for p in (_REPO / "cycle_data").rglob("*.json") if _HAYES in p.name]
    if not hits:
        pytest.skip("contributor export not in cycle_data/")
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "wd_end_gate_eval_469a", _REPO / "devtools" / "end_gate_eval.py"
    )
    ege = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ege
    spec.loader.exec_module(ege)
    doc = ege._load_doc(hits[0], True)  # noqa: SLF001
    _cfg, store, _opts = ege._production(doc, doc["data"])  # noqa: SLF001

    sig = store.compute_profile_terminal_signature("Eco")

    assert sig is not None
    assert (sig["seen_in"], sig["measured"]) == (6, 9)
    assert 4800.0 <= sig["quiet_before_s"] <= 4900.0
    assert sig["event_watts"] < 40.0


def test_a_run_closed_without_its_pump_out_keeps_the_measured_drying() -> None:
    """The keep-tail cap, the consumer the wrong reading fed (item 465: ~8.1k s
    stored). With 4850 s measured it was still cut at TERMINAL_QUIET_CAP_S, 1800 s,
    so a run that dried without a pump-out stored ~8.1k s of an ~11.1k s programme."""
    from datetime import datetime, timedelta, timezone

    from custom_components.ha_washdata.cycle_detector import (
        CycleDetector,
        CycleDetectorConfig,
    )

    t0 = datetime(2026, 2, 26, 23, 30, tzinfo=timezone.utc)
    det = CycleDetector(
        CycleDetectorConfig(min_power=2.0, off_delay=1800, device_type="dishwasher"),
        lambda a, b: None,
        lambda p: None,
    )
    det._current_cycle_start = t0
    det._expected_duration = 10690.0
    det._matched_terminal_quiet_s = det._sanitize_terminal_quiet(PUMP_OUT_QUIET)
    det._end_spike_seen = False
    det._end_spike_duration = 0.0
    det._last_active_time = t0 + timedelta(seconds=6270.0)

    cap = det._keep_tail_cap(t0)

    assert cap is not None
    assert (cap - t0).total_seconds() == pytest.approx(6270.0 + PUMP_OUT_QUIET)
