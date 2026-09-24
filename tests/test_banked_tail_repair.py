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
    # WHEN the spike happened matters, not just that it did: only a spike at
    # >= 90% of expected is the terminal pump-out, the same bar
    # `_resolve_smart_ratio` uses. Below that it can be the pre-final-rinse
    # drain with a passive Dry phase still to come.
    d._end_spike_duration = last_active if spike else 0.0
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
    """The pump-out IS the end; _last_active_time already sits on it.

    Late enough to be terminal: 3400 s of a 3600 s programme is 94%, past the
    0.90 bar `_resolve_smart_ratio` uses for the same judgement.
    """
    d = _det("dishwasher", quiet=600.0, spike=True, last_active=3400.0)
    assert _cap_offset(d) == pytest.approx(3400.0)


def test_a_pre_rinse_drain_does_not_cut_the_drying_phase() -> None:
    """Found in the PR #448 round-10 review.

    `_end_spike_seen` is set from DISHWASHER_END_SPIKE_MIN_PROGRESS (0.85), but
    a spike at 83% of expected is the pre-final-rinse drain, with the Dry phase
    still to come. Capping at the drain cuts that drying off, which lowers
    avg_duration, which makes the NEXT Smart Termination fire earlier - the
    error compounds toward split cycles. Falls through to the measured quiet
    span instead.
    """
    d = _det("dishwasher", quiet=600.0, spike=True, last_active=3000.0)  # 83%
    assert _cap_offset(d) == pytest.approx(3600.0)  # 3000 + 600 measured quiet


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


@pytest.mark.asyncio
async def test_repair_rebuilds_the_signature_from_the_kept_samples() -> None:
    """The trace is trimmed, so the signature derived from it has to be rebuilt.

    Left stale it would describe a duration and a power distribution taken from
    samples the cycle no longer has, and the signature feeds candidate rejection.
    The sibling trim and the merge path both already recompute it; this one did
    not (found in the PR #448 review).
    """
    cyc = _cycle("a", 3000, 1200)
    stale = {
        "duration": 4200.0,
        "total_energy": 999.0,
        "max_power": 100.0,
        "p05": 0.0,
    }
    cyc["signature"] = dict(stale)
    data = {"past_cycles": [cyc], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)

    res = await st.async_repair_banked_tails(2.0, "washing_machine")

    assert res["repaired"] == 1
    sig = data["past_cycles"][0]["signature"]
    assert sig is not None and sig != stale, "the signature still describes the old trace"
    # It now agrees with the duration the repair wrote.
    assert sig["duration"] == pytest.approx(
        data["past_cycles"][0]["duration"], abs=31.0
    )
    # The dead tail is gone, so the power distribution no longer contains zeros.
    assert sig["p05"] == pytest.approx(100.0)
    assert sig["max_power"] == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_a_cycle_the_repair_skips_keeps_its_signature() -> None:
    """Only a trimmed trace needs a new signature; an untouched one must not move."""
    cyc = _cycle("a", 3000, 10)  # tail below BANKED_TAIL_REPAIR_MIN_S
    cyc["signature"] = {"duration": 3010.0, "max_power": 100.0}
    data = {"past_cycles": [cyc], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)

    res = await st.async_repair_banked_tails(2.0, "washing_machine")

    assert res["repaired"] == 0
    assert data["past_cycles"][0]["signature"] == {"duration": 3010.0, "max_power": 100.0}


@pytest.mark.asyncio
async def test_a_dishwasher_repair_leaves_the_trace_covering_its_duration() -> None:
    """Found in the PR #448 round-6 review: the drying allowance was undone later.

    A dishwasher keeps `allowance` seconds of measured drying past its last
    ACTIVE sample, and a change-only plug reports nothing across it - so without
    a terminal point the trace ends before the stored duration.
    `_reprocess_all_data_sync` reads that gap as drift and snaps duration and
    end_time back down to the trace end (tolerance `max(5, 2 * sampling_interval)`
    against an allowance of up to TERMINAL_QUIET_CAP_S), silently removing the
    drying the repair exists to keep - and dragging the profile average with it.
    """
    # A change-only plug: it reports the terminal drop and then NOTHING across
    # the drying phase, so the trace simply stops. This is the shape that
    # reproduces the bug - a trace padded with 0 W samples hides it, because the
    # kept tail then already reaches new_duration.
    pts = [[float(t), 100.0] for t in range(0, 3000, 30)]
    pts.append([3000.0, 0.0])
    cyc = {
        "id": "a",
        "profile_name": "Eco",
        "start_time": T0.isoformat(),
        "duration": 6000.0,  # banked: the confirmation wait counted as cycle time
        "termination_reason": "smart",
        "sampling_interval": 30.0,
        "power_data": pts,
    }
    data = {"past_cycles": [cyc], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)
    st.profile_terminal_quiet_seconds = lambda _n: 600.0  # type: ignore[assignment]

    res = await st.async_repair_banked_tails(2.0, "dishwasher")

    assert res["repaired"] == 1
    out = data["past_cycles"][0]
    trace_end = float(out["power_data"][-1][0])
    stored = float(out["duration"])
    si = float(out.get("sampling_interval", 30.0) or 30.0)
    # The invariant _reprocess_all_data_sync relies on, and the exact tolerance
    # it applies: a reprocess must find nothing to snap.
    assert abs(stored - trace_end) <= max(5.0, 2.0 * si), (
        f"duration {stored} vs trace end {trace_end}: a reprocess would snap the "
        "measured drying allowance back off"
    )
    # The allowance really is still there (last activity 2970s + 600s).
    assert stored == pytest.approx(3570.0, abs=31.0)
    assert trace_end == pytest.approx(3570.0, abs=31.0)


@pytest.mark.asyncio
async def test_a_washer_repair_adds_no_spurious_terminal_sample() -> None:
    """With no allowance the trace already ends at the duration; nothing to add."""
    data = {"past_cycles": [_cycle("a", 3000, 1200)], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)

    before = len([p for p in data["past_cycles"][0]["power_data"] if p[0] <= 2970.0])
    res = await st.async_repair_banked_tails(2.0, "washing_machine")

    assert res["repaired"] == 1
    out = data["past_cycles"][0]
    assert len(out["power_data"]) == before, "no terminal point should have been added"
    assert out["power_data"][-1][1] == 100.0


@pytest.mark.asyncio
async def test_a_user_stopped_cycle_is_never_repaired() -> None:
    """`user_stop` finishes with keep_tail=True and DELIBERATELY no tail_cap.

    "User implies Done Now" (`cycle_detector.user_stop`), so its tail is the tail
    the user asked to keep - not a banked Smart Termination confirmation delay.
    Repairing it truncates real recorded history and rewrites the profile
    statistics with the shortened value. Found in the PR #448 round-7 review.
    """
    cyc = _cycle("a", 3000, 1200)
    cyc["termination_reason"] = "user"
    data = {"past_cycles": [cyc], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)

    res = await st.async_repair_banked_tails(2.0, "washing_machine")

    assert res["repaired"] == 0
    assert data["past_cycles"][0]["duration"] == pytest.approx(4200.0)
    assert not st.banked_tail_repair_pending(), "the marker must still be cleared"


@pytest.mark.asyncio
async def test_a_cycle_with_no_recorded_reason_is_left_alone() -> None:
    """A one-time upgrade rewrite must not guess about history it cannot verify."""
    cyc = _cycle("a", 3000, 1200)
    cyc.pop("termination_reason", None)
    data = {"past_cycles": [cyc], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)

    res = await st.async_repair_banked_tails(2.0, "washing_machine")

    assert res["repaired"] == 0
    assert data["past_cycles"][0]["duration"] == pytest.approx(4200.0)


@pytest.mark.asyncio
async def test_the_repair_still_examines_every_cycle_but_repairs_only_smart() -> None:
    """Mixed history: only the Smart-terminated cycle moves."""
    smart = _cycle("smart", 3000, 1200)
    user = _cycle("user", 3000, 1200)
    user["termination_reason"] = "user"
    timeout = _cycle("timeout", 3000, 1200)
    timeout["termination_reason"] = "timeout"
    data = {"past_cycles": [smart, user, timeout], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)

    res = await st.async_repair_banked_tails(2.0, "washing_machine")

    assert res["examined"] == 3
    assert res["repaired"] == 1
    by_id = {c["id"]: c for c in data["past_cycles"]}
    assert by_id["smart"]["duration"] < 4200.0
    assert by_id["user"]["duration"] == pytest.approx(4200.0)
    assert by_id["timeout"]["duration"] == pytest.approx(4200.0)


@pytest.mark.asyncio
async def test_a_terminal_pump_out_gets_no_drying_allowance() -> None:
    """Found in the PR #448 round-11 review.

    `quiet_before_s` is the quiet measured BEFORE the terminal event. When the
    final above-threshold sample IS that pump-out, the drying already happened
    before it, so adding the allowance on top counts the same quiet twice - and
    the inflated `new_duration` can shrink the reclaim below
    BANKED_TAIL_REPAIR_MIN_S, leaving the banked tail unrepaired.
    """
    # wash -> 600 s of drying -> a 60 s pump-out -> banked dead tail
    pts = [[float(t), 100.0] for t in range(0, 2400, 30)]
    pts += [[float(t), 0.0] for t in range(2400, 3000, 30)]      # drying
    pts += [[float(t), 80.0] for t in range(3000, 3060, 30)]     # pump-out
    pts += [[float(t), 0.0] for t in range(3060, 4800, 30)]      # banked tail
    cyc = {
        "id": "a", "profile_name": "Eco", "start_time": T0.isoformat(),
        "duration": 4800.0, "termination_reason": "smart",
        "sampling_interval": 30.0, "power_data": pts,
    }
    data = {"past_cycles": [cyc], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)
    st.profile_terminal_quiet_seconds = lambda _n: 600.0  # type: ignore[assignment]

    res = await st.async_repair_banked_tails(2.0, "dishwasher")

    assert res["repaired"] == 1
    # Ends at the pump-out (3030 s, its last sample), NOT 3030 + 600.
    assert float(data["past_cycles"][0]["duration"]) == pytest.approx(3030.0, abs=31.0)


@pytest.mark.asyncio
async def test_a_cycle_that_ends_in_drying_still_gets_its_allowance() -> None:
    """The other half: no terminal event, so the tail IS the drying."""
    pts = [[float(t), 100.0] for t in range(0, 3000, 30)]
    pts.append([3000.0, 0.0])                                    # plug goes quiet
    cyc = {
        "id": "b", "profile_name": "Eco", "start_time": T0.isoformat(),
        "duration": 6000.0, "termination_reason": "smart",
        "sampling_interval": 30.0, "power_data": pts,
    }
    data = {"past_cycles": [cyc], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)
    st.profile_terminal_quiet_seconds = lambda _n: 600.0  # type: ignore[assignment]

    res = await st.async_repair_banked_tails(2.0, "dishwasher")

    assert res["repaired"] == 1
    # last activity 2970 + 600 measured drying.
    assert float(data["past_cycles"][0]["duration"]) == pytest.approx(3570.0, abs=31.0)


@pytest.mark.asyncio
async def test_the_terminal_point_never_carries_active_power() -> None:
    """Found in the PR #448 round-16 review.

    The drying allowance pushes `new_duration` past the last ACTIVE sample. If
    the plug reports the drop LATE - nothing at all between the last activity and
    `new_duration` - then `kept[-1]` is that active sample, and copying its power
    into the appended terminal point makes interpolation read the entire drying
    span as full draw. That goes straight into the recomputed signature, the
    rebuilt envelope, and every conformance/artifact check taken against it.
    """
    pts = [[float(t), 100.0] for t in range(0, 3000, 30)]
    pts.append([5000.0, 0.0])  # the drop, reported well after new_duration
    cyc = {
        "id": "a", "profile_name": "Eco", "start_time": T0.isoformat(),
        "duration": 6000.0, "termination_reason": "smart",
        "sampling_interval": 30.0, "power_data": pts,
    }
    data = {"past_cycles": [cyc], BANKED_TAIL_REPAIR_KEY: True}
    st = _Store(data)
    st.profile_terminal_quiet_seconds = lambda _n: 600.0  # type: ignore[assignment]

    res = await st.async_repair_banked_tails(2.0, "dishwasher")

    assert res["repaired"] == 1
    out = data["past_cycles"][0]
    # last activity 2970 + the 600 s measured drying.
    assert float(out["duration"]) == pytest.approx(3570.0, abs=31.0)
    terminal = out["power_data"][-1]
    assert float(terminal[0]) == pytest.approx(3570.0, abs=31.0)
    assert float(terminal[1]) == pytest.approx(0.0), (
        "the terminal point took the last ACTIVE sample's power, so the whole "
        "drying phase now interpolates as full draw"
    )
    # ...and the signature rebuilt from that trace agrees.
    assert float(out["signature"]["total_energy"]) < 100.0
