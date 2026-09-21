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
"""Register item 297: Smart Termination stored its confirmation delay as cycle time.

`_keep_tail_cap` capped a kept tail at the matched profile's EXPECTED end - the
mean of these same stored durations - so a banked tail raised ``avg_duration``,
the higher average allowed a longer tail, and the reported end drifted later every
run. Measured over 375 cycles from 16 devices: smart-terminated cycles banked a
median 12.6 min (washing machines 22.7 min, p90 40.4 min) against ~0 for every
other termination path, and profiles carried a mean +5.3% duration inflation.

Covers both halves: the live cap, and the one-time repair of already-stored data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata.const import (
    BANKED_TAIL_REPAIR_KEY,
    BANKED_TAIL_REPAIR_MIN_S,
    STORAGE_KEY,
    STORAGE_VERSION,
    TERMINAL_QUIET_CAP_S,
)
from custom_components.ha_washdata.cycle_detector import (
    CycleDetector,
    CycleDetectorConfig,
)
from custom_components.ha_washdata.profile_store import ProfileStore, WashDataStore

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# The live cap
# ---------------------------------------------------------------------------


def _det(device_type="washing_machine", quiet=None, spike=False, last_active=3000.0):
    cfg = CycleDetectorConfig(min_power=2.0, off_delay=180, device_type=device_type)
    d = CycleDetector(cfg, lambda a, b: None, lambda p: None)
    d._current_cycle_start = T0
    d._expected_duration = 3600.0
    d._matched_terminal_quiet_s = quiet
    d._end_spike_seen = spike
    d._last_active_time = T0 + timedelta(seconds=last_active) if last_active else None
    return d


def _cap_offset(det) -> float | None:
    cap = det._keep_tail_cap(T0)
    return None if cap is None else (cap - T0).total_seconds()


def test_a_washer_ends_at_its_last_activity() -> None:
    """No device type except a dishwasher has a passive terminal phase, so there
    is nothing legitimate to bank after the last activity. This is the 22.7 min
    washing-machine median."""
    assert _cap_offset(_det("washing_machine")) == pytest.approx(3000.0)
    assert _cap_offset(_det("dryer")) == pytest.approx(3000.0)
    assert _cap_offset(_det("washer_dryer")) == pytest.approx(3000.0)


def test_a_dishwasher_that_pumped_out_ends_there_too() -> None:
    """The pump-out IS the end; _last_active_time already sits on it."""
    d = _det("dishwasher", quiet=600.0, spike=True)
    assert _cap_offset(d) == pytest.approx(3000.0)


def test_a_dishwasher_without_its_pump_out_keeps_the_measured_drying() -> None:
    """Then the drying phase is the tail, and it is real cycle content."""
    d = _det("dishwasher", quiet=600.0, spike=False)
    assert _cap_offset(d) == pytest.approx(3600.0)


def test_an_unmeasured_dishwasher_keeps_the_old_behaviour() -> None:
    """Truncating a drying phase on no evidence is the worse error: the pump-out
    is measurably absent in a substantial minority of runs on some machines."""
    d = _det("dishwasher", quiet=None, spike=False)
    assert _cap_offset(d) == pytest.approx(3600.0)  # the expected end, as before
    d2 = _det("dishwasher", quiet=None, spike=False, last_active=5000.0)
    assert _cap_offset(d2) == pytest.approx(5000.0)  # ...or later activity


def test_a_corrupt_quiet_span_cannot_license_an_unbounded_tail() -> None:
    d = _det("dishwasher", quiet=99999.0, spike=False)
    assert _cap_offset(d) == pytest.approx(3000.0 + TERMINAL_QUIET_CAP_S)


def test_an_unmatched_cycle_is_untouched() -> None:
    d = _det("washing_machine")
    d._expected_duration = 0.0
    assert d._keep_tail_cap(T0) is None


# ---------------------------------------------------------------------------
# The trust gate on the measurement
# ---------------------------------------------------------------------------


class _Store(ProfileStore):
    def __init__(self, data):  # pylint: disable=super-init-not-called
        self._data = data
        self._logger = MagicMock()

    def iter_evidence_cycles(self):
        yield from (self._data.get("past_cycles") or [])

    async def async_save(self):
        return None

    async def async_rebuild_envelope(self, profile_name):  # noqa: ARG002
        return True


def _sig(store, **over):
    base = {
        "quiet_before_s": 600.0,
        "seen_in": 10,
        "measured": 10,
        "consistency": 1.0,
    }
    base.update(over)
    store.compute_profile_terminal_signature = MagicMock(return_value=base)
    return store


def test_a_span_seen_once_is_not_a_measurement() -> None:
    """Real corpus: washing-machine profiles produced values from a SINGLE cycle
    out of 4-12, one of them 2400 s - which as an allowance would have banked 40
    min, worse than the bug it fixes."""
    st = _sig(_Store({}), quiet_before_s=2400.0, seen_in=1, measured=4, consistency=0.25)
    assert st.profile_terminal_quiet_seconds("p") is None


def test_a_consistently_measured_span_is_trusted() -> None:
    """Real corpus: both dishwashers measured their drying phase in 20/20 and
    17/17 cycles."""
    st = _sig(_Store({}), quiet_before_s=1810.0, seen_in=20, measured=20, consistency=1.0)
    assert st.profile_terminal_quiet_seconds("p") == pytest.approx(1810.0)


def test_no_event_ever_means_no_opinion() -> None:
    st = _sig(_Store({}), quiet_before_s=None, seen_in=0, measured=11, consistency=0.0)
    assert st.profile_terminal_quiet_seconds("p") is None


def test_a_failing_statistic_never_breaks_matching() -> None:
    st = _Store({})
    st.compute_profile_terminal_signature = MagicMock(side_effect=RuntimeError("boom"))
    assert st.profile_terminal_quiet_seconds("p") is None


# ---------------------------------------------------------------------------
# The storage migration + one-time repair
# ---------------------------------------------------------------------------


def _hass():
    h = MagicMock()
    h.config.config_dir = "/tmp"
    return h


@pytest.mark.asyncio
async def test_v12_to_v13_sets_the_repair_marker() -> None:
    store = WashDataStore(_hass(), STORAGE_VERSION, f"{STORAGE_KEY}.test")
    result = await store._async_migrate_func(12, 1, {"past_cycles": []})
    assert result[BANKED_TAIL_REPAIR_KEY] is True


@pytest.mark.asyncio
async def test_the_migration_does_not_touch_cycles() -> None:
    """It is marker-only: deciding where activity ended needs stop_threshold_w,
    which lives in entry.options and is not visible here."""
    cycles = [{"id": "a", "duration": 3600.0, "power_data": [[0, 100.0], [10, 0.0]]}]
    store = WashDataStore(_hass(), STORAGE_VERSION, f"{STORAGE_KEY}.test")
    result = await store._async_migrate_func(12, 1, {"past_cycles": cycles})
    assert result["past_cycles"][0]["duration"] == 3600.0
    assert result["past_cycles"][0]["power_data"] == [[0, 100.0], [10, 0.0]]


def _cycle(cid, run_s, tail_s, profile="P", step=30.0):
    """Runs at 100 W for run_s, then sits at 0 W for tail_s; stored as the whole span."""
    pts = [[float(t), 100.0] for t in range(0, int(run_s), int(step))]
    pts += [[float(t), 0.0] for t in range(int(run_s), int(run_s + tail_s), int(step))]
    return {
        "id": cid,
        "profile_name": profile,
        "start_time": T0.isoformat(),
        "duration": float(run_s + tail_s),
        "termination_reason": "smart",
        "power_data": pts,
    }


@pytest.mark.asyncio
async def test_repair_trims_a_banked_washer_cycle() -> None:
    data = {
        "past_cycles": [_cycle("a", 3000, 1200)],
        BANKED_TAIL_REPAIR_KEY: True,
    }
    st = _Store(data)
    assert st.banked_tail_repair_pending()

    res = await st.async_repair_banked_tails(2.0, "washing_machine")

    assert res["repaired"] == 1
    c = data["past_cycles"][0]
    # Ends at the last sample above the threshold, not 1200 s later.
    assert c["duration"] == pytest.approx(2970.0, abs=31.0)
    assert max(p for _t, p in c["power_data"]) == 100.0
    assert c["power_data"][-1][1] == 100.0, "the dead tail must be trimmed off"
    assert c["end_time"].startswith("2026-01-01")
    assert not st.banked_tail_repair_pending()


@pytest.mark.asyncio
async def test_repair_is_idempotent() -> None:
    data = {"past_cycles": [_cycle("a", 3000, 1200)], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)
    await st.async_repair_banked_tails(2.0, "washing_machine")
    first = data["past_cycles"][0]["duration"]

    data[BANKED_TAIL_REPAIR_KEY] = True
    res2 = await st.async_repair_banked_tails(2.0, "washing_machine")

    assert res2["repaired"] == 0
    assert data["past_cycles"][0]["duration"] == first


@pytest.mark.asyncio
async def test_repair_leaves_an_unmeasured_dishwasher_alone() -> None:
    """Its tail may be a real drying phase; truncating it on no evidence is worse
    than leaving the banking in place."""
    data = {"past_cycles": [_cycle("a", 3000, 1200)], BANKED_TAIL_REPAIR_KEY: True}
    st = _sig(_Store(data), quiet_before_s=None, seen_in=0, measured=9, consistency=0.0)

    res = await st.async_repair_banked_tails(2.0, "dishwasher")

    assert res["repaired"] == 0
    assert data["past_cycles"][0]["duration"] == pytest.approx(4200.0)


@pytest.mark.asyncio
async def test_repair_keeps_a_measured_drying_phase() -> None:
    data = {"past_cycles": [_cycle("a", 3000, 1200)], BANKED_TAIL_REPAIR_KEY: True}
    st = _sig(_Store(data), quiet_before_s=600.0, seen_in=20, measured=20, consistency=1.0)

    res = await st.async_repair_banked_tails(2.0, "dishwasher")

    assert res["repaired"] == 1
    # last activity (2970) + the measured 600 s drying, not the full 1200 s tail.
    assert data["past_cycles"][0]["duration"] == pytest.approx(3570.0, abs=31.0)


@pytest.mark.asyncio
async def test_repair_skips_trivial_tails() -> None:
    """Don't churn a whole history for a few seconds; the median banking this
    exists to remove was 12.6 min."""
    c = _cycle("a", 3000, 30)
    # Last sample above threshold is t=2970, so pin the stored duration just under
    # the floor rather than relying on the sampling grid to land there.
    c["duration"] = 2970.0 + (BANKED_TAIL_REPAIR_MIN_S - 1.0)
    data = {"past_cycles": [c], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)
    res = await st.async_repair_banked_tails(2.0, "washing_machine")
    assert res["repaired"] == 0
    assert data["past_cycles"][0]["duration"] == pytest.approx(
        2970.0 + BANKED_TAIL_REPAIR_MIN_S - 1.0
    )


@pytest.mark.asyncio
async def test_repair_takes_a_tail_at_the_floor() -> None:
    """...and does act once it reaches the floor, so the boundary is pinned from
    both sides."""
    c = _cycle("a", 3000, 30)
    c["duration"] = 2970.0 + BANKED_TAIL_REPAIR_MIN_S
    data = {"past_cycles": [c], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)
    res = await st.async_repair_banked_tails(2.0, "washing_machine")
    assert res["repaired"] == 1
    assert data["past_cycles"][0]["duration"] == pytest.approx(2970.0)


@pytest.mark.asyncio
async def test_repair_ignores_reference_and_backfill_cycles() -> None:
    """Community templates were never recorded here, and backfilled cycles never
    went through Smart Termination, so neither can carry a banked tail."""
    ref = [_cycle("r", 3000, 1200)]
    back = [_cycle("b", 3000, 1200)]
    data = {
        "past_cycles": [],
        "reference_cycles": ref,
        "backfill_cycles": back,
        BANKED_TAIL_REPAIR_KEY: True,
    }
    st = _Store(data)
    await st.async_repair_banked_tails(2.0, "washing_machine")
    assert data["reference_cycles"][0]["duration"] == pytest.approx(4200.0)
    assert data["backfill_cycles"][0]["duration"] == pytest.approx(4200.0)


@pytest.mark.asyncio
async def test_a_failed_repair_costs_the_user_nothing() -> None:
    data = {"past_cycles": [_cycle("a", 3000, 1200)], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)
    st.async_rebuild_envelope = MagicMock(side_effect=RuntimeError("boom"))
    # Must not raise, and must not leave a half-written history.
    res = await st.async_repair_banked_tails(2.0, "washing_machine")
    assert isinstance(res, dict)
    assert data["past_cycles"][0]["duration"] > 0


@pytest.mark.asyncio
async def test_a_cycle_that_never_rose_above_the_threshold_is_skipped() -> None:
    data = {
        "past_cycles": [
            {
                "id": "z",
                "profile_name": "P",
                "start_time": T0.isoformat(),
                "duration": 4200.0,
                "termination_reason": "smart",
                "power_data": [[float(t), 0.5] for t in range(0, 4200, 30)],
            }
        ],
        BANKED_TAIL_REPAIR_KEY: True,
    }
    st = _Store(data)
    res = await st.async_repair_banked_tails(2.0, "washing_machine")
    assert res["repaired"] == 0
    assert data["past_cycles"][0]["duration"] == pytest.approx(4200.0)
