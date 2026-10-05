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
"""Register item 498: a verified pause a revoked match orphaned.

A revoke (divergence revert, or every candidate rejected) drops the detector's
match and expected duration but used to leave the envelope's verified pause on.
Nothing but high power could clear it then (the 95%-of-span release needs the
match, the #375 release the expected duration), so a finished appliance sat until
the force stop: 8 h on a plug that keeps reporting 0 W, the watchdog's 4.5 h
silence limit on one that only gets keepalives.

Maintainer decision (option C): keep the pause on a revoke, release it after a
bounded quiet: 1.25 x the longest pause the revoked programme's recorded cycles
resumed from, never before max(off_delay, min_off_gap, 600 s), never after 3 h
above that floor; never a user pause.

Real manager, real ProfileStore and detector, real watchdog; only the matcher and
the envelope alignment are scripted, because the shipped matcher never revokes a
match with a pause behind it on the corpus (0 of 117 revokes).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed_exact

from custom_components.ha_washdata import match_rules
from custom_components.ha_washdata.const import (
    END_GATE_HAZARD_MARGIN,
    ENDING_HARD_FINALIZE_MIN_QUIET_S,
    ORPHANED_PAUSE_MAX_WAIT_S,
    STATE_ENDING,
)
from custom_components.ha_washdata.cycle_detector import CycleDetector, MatchContext
from custom_components.ha_washdata.detector_config import build_detector_config
from custom_components.ha_washdata.profile_store import MatchResult

from .real_manager import POWER, boot, make_entry, record_notify

PROFILE_S = 7200.0
SOAK_START, SOAK_S = 0.40, 1200.0  # every recorded run has a 20 min soak at 40%
OFF_DELAY, MIN_OFF_GAP = 300, 900  # floor 900 s < 1.25 x 1200 s = 1500 s
WAIT_S = END_GATE_HAZARD_MARGIN * SOAK_S
ACTIVE_W = 500.0
QUIET_AT = 1800  # the live run goes quiet 30 min in, long before its expected end


# --- the rule ---------------------------------------------------------------------


def test_the_wait_is_the_revoked_programmes_longest_pause_with_margin() -> None:
    wait = match_rules.orphaned_pause_wait_s(
        off_delay=OFF_DELAY, min_off_gap=MIN_OFF_GAP, longest_pause_s=SOAK_S
    )
    assert wait == WAIT_S == 1500.0


def test_the_floor_is_what_the_unmatched_fallback_waits_anyway() -> None:
    assert match_rules.orphaned_pause_wait_s(
        off_delay=300, min_off_gap=900, longest_pause_s=0.0
    ) == 900.0
    # Never below the #375 release's own quiet floor.
    assert match_rules.orphaned_pause_wait_s(
        off_delay=60, min_off_gap=60, longest_pause_s=None
    ) == ENDING_HARD_FINALIZE_MIN_QUIET_S
    # Unusable input is "no evidence", not a raise.
    assert match_rules.orphaned_pause_wait_s(
        off_delay="x", min_off_gap=None, longest_pause_s=float("nan")
    ) == ENDING_HARD_FINALIZE_MIN_QUIET_S


def test_the_cap_bounds_the_evidence_but_not_the_floor() -> None:
    assert match_rules.orphaned_pause_wait_s(
        off_delay=300, min_off_gap=900, longest_pause_s=50_000.0
    ) == ORPHANED_PAUSE_MAX_WAIT_S
    # A user's own min_off_gap above the cap still wins: the release never ends a
    # cycle sooner than the unmatched fallback would.
    assert match_rules.orphaned_pause_wait_s(
        off_delay=300, min_off_gap=14400, longest_pause_s=50_000.0
    ) == 14400.0


def _decide(**kw: Any) -> bool:
    args: dict[str, Any] = {
        "verified_pause": True, "user_paused": False, "current_matched": None,
        "time_below": WAIT_S, "wait_s": WAIT_S, "longest_pause_s": SOAK_S,
    }
    args.update(kw)
    return bool(match_rules.decide_orphaned_pause_release(**args).verified_pause)


def test_only_an_orphaned_automatic_pause_is_released() -> None:
    assert _decide() is False
    assert _decide(time_below=WAIT_S - 1) is True
    assert _decide(user_paused=True) is True
    assert _decide(current_matched="Cotton") is True
    assert _decide(verified_pause=False) is False


# --- the detector -----------------------------------------------------------------

T0 = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)
CATALOGUE = (3, ((SOAK_START, SOAK_S), (0.9, 60.0)) * 3)


def _ctx(name: str | None, *, revoke: bool = False, catalogue: Any = CATALOGUE) -> MatchContext:
    return MatchContext(
        profile_name=name, confidence=0.8 if name else 0.0,
        expected_duration=PROFILE_S if name else 0.0, is_confident_mismatch=revoke,
        pause_catalogue=catalogue if name else None,
    )


def _orphan(*, catalogue: Any = CATALOGUE, user_paused: bool = False) -> tuple[CycleDetector, datetime]:
    """A detector in ENDING whose verified pause outlived a revoke."""
    det = CycleDetector(
        build_detector_config(
            {"off_delay": OFF_DELAY, "min_off_gap": MIN_OFF_GAP, "min_power": 2.0,
             "stop_threshold_w": 2.0, "start_threshold_w": 5.0},
            {}, "washing_machine",
        ),
        MagicMock(), MagicMock(),
    )
    t = T0
    for _ in range(QUIET_AT // 30):
        det.process_reading(ACTIVE_W, t)
        t += timedelta(seconds=30)
    # Quiet is credited from the last active reading (the interval it closes).
    det._quiet_from = t - timedelta(seconds=30)  # type: ignore[attr-defined]  # noqa: SLF001
    det.update_match(_ctx("Cotton", catalogue=catalogue))
    for _ in range(10):
        det.process_reading(0.0, t)
        t += timedelta(seconds=30)
    assert det.state == STATE_ENDING
    det.set_verified_pause(True)
    det.set_user_paused(user_paused)
    det.update_match(_ctx(None, revoke=True))
    assert det.matched_profile is None and det.expected_duration_seconds == 0.0
    assert det._verified_pause  # noqa: SLF001 - the orphaned state under test
    return det, t


def _quiet_until_end(det: CycleDetector, t: datetime, *, synthetic: bool, hours: float = 8.5) -> float | None:
    """Quiet readings (0 W, or watchdog-style keepalives); quiet seconds at the end."""
    quiet_from = det._quiet_from  # type: ignore[attr-defined]  # noqa: SLF001
    end = t + timedelta(hours=hours)
    while t < end:
        det.process_reading(0.0, t, synthetic=synthetic, observed=True)
        if det.state != STATE_ENDING:
            return (t - quiet_from).total_seconds()
        t += timedelta(seconds=30)
    return None


@pytest.mark.parametrize("synthetic", [False, True], ids=["reporting", "keepalives"])
def test_the_detector_releases_the_orphan_after_the_bounded_wait(synthetic: bool) -> None:
    det, t = _orphan()
    assert det._orphaned_pause_evidence_s == SOAK_S  # noqa: SLF001
    quiet = _quiet_until_end(det, t, synthetic=synthetic)
    assert quiet is not None, "held to the force stop"
    assert WAIT_S <= quiet <= WAIT_S + 60


def test_without_evidence_the_floor_applies() -> None:
    det, t = _orphan(catalogue=None)
    quiet = _quiet_until_end(det, t, synthetic=False)
    assert quiet is not None
    assert MIN_OFF_GAP <= quiet <= MIN_OFF_GAP + 60


def test_the_detector_never_releases_a_user_pause() -> None:
    det, t = _orphan(user_paused=True)
    assert _quiet_until_end(det, t, synthetic=False, hours=4.0) is None
    assert det._verified_pause  # noqa: SLF001


def test_a_new_match_ends_the_orphan_and_a_reset_clears_the_evidence() -> None:
    det, t = _orphan()
    det.update_match(_ctx("Cotton"))
    # Matched again: the orphan rule stands down (the normal releases own it).
    det._time_below_threshold_gapfree = 10 * WAIT_S  # noqa: SLF001
    det._release_orphaned_pause()  # noqa: SLF001
    assert det._verified_pause  # noqa: SLF001
    det.reset()
    assert det._orphaned_pause_evidence_s == 0.0  # noqa: SLF001


# --- the real manager, both plug behaviours ---------------------------------------


def _profile_trace(total_s: float) -> list[list[float]]:
    soak_from = SOAK_START * total_s
    out = []
    for t in range(0, int(total_s), 60):
        watts = 0.0 if soak_from <= t < soak_from + SOAK_S else ACTIVE_W
        out.append([float(t), watts])
    return [*out, [float(total_s), 0.0]]


def _cycle(i: int) -> dict[str, Any]:
    start = dt_util.parse_datetime(f"2026-01-{1 + i:02d}T08:00:00+00:00")
    assert start is not None
    return {
        "id": f"cotton-{i}",
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=PROFILE_S)).isoformat(),
        "duration": PROFILE_S,
        "status": "completed",
        "power_data": _profile_trace(PROFILE_S),
        "profile_name": "Cotton",
        "label_source": "manual",
    }


def _script_matcher(mgr: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Cotton while active; from the first quiet tick on, every candidate rejected.

    On that tick the manager first runs the envelope alignment against the
    programme the detector still holds (confirmed: the pause engages), then hands
    the detector the revoke, so the pause is orphaned whatever the device (a
    dishwasher's match freeze would stop any later tick).
    """
    store = mgr.profile_store
    state = {"revoked": False}

    async def _match(readings: Any, _duration: Any, **_kw: Any) -> MatchResult:
        if readings and float(readings[-1][1]) < 2.0:
            state["revoked"] = True
        if state["revoked"]:
            return MatchResult(
                None, 0.0, 0.0, None, [], False, 0.0,
                is_confident_mismatch=True, mismatch_reason="all_rejected",
            )
        return MatchResult(
            "Cotton", 0.8, PROFILE_S, None, [{"name": "Cotton", "score": 0.8}], False, 0.8,
        )

    async def _confirm(_name: Any, _formatted: Any) -> tuple[bool, float, Any]:
        return True, 900.0, None

    monkeypatch.setattr(store, "async_match_profile", _match)
    monkeypatch.setattr(store, "async_verify_alignment", _confirm)
    monkeypatch.setattr(store, "envelope_time_span", lambda _name: PROFILE_S)
    return state


async def _boot(hass: Any, device_type: str) -> Any:
    record_notify(hass)
    mgr = await boot(hass, make_entry(hass, {
        "device_type": device_type, "min_power": 2.0, "stop_threshold_w": 2.0,
        "start_threshold_w": 5.0, "off_delay": OFF_DELAY, "min_off_gap": MIN_OFF_GAP,
        "watchdog_interval": 30,
    }))
    store = mgr.profile_store
    store._data["past_cycles"].extend(_cycle(i) for i in range(5))  # noqa: SLF001
    await store.create_profile("Cotton", "cotton-0")
    return mgr


async def _drive(hass: Any, freezer: Any, mgr: Any, plug: str, schedule: Any,
                 *, until_s: float, stop_at_first: bool = False,
                 on_step: Any = None) -> dict[str, Any]:
    """Run ``schedule(t) -> watts`` (0 = quiet). "reporting" re-reports 0 W every
    minute; "silent" publishes each drop once, so only keepalives reach the detector."""
    store, det = mgr.profile_store, mgr.detector
    n_before = len(store.get_past_cycles())
    t = 0
    last_published: float | None = None
    orphaned = False
    first_end: int | None = None
    while t < until_s:
        step = 60 if schedule(t) == 0.0 else 30
        freezer.tick(timedelta(seconds=step))
        t += step
        watts = schedule(t)
        if watts > 0 or plug == "reporting" or last_published != 0.0:
            hass.states.async_set(POWER, str(watts), {"unit_of_measurement": "W"}, force_update=True)
            last_published = watts
        async_fire_time_changed_exact(hass, dt_util.utcnow())
        await hass.async_block_till_done()
        if det._verified_pause and det.matched_profile is None and det.state == STATE_ENDING:  # noqa: SLF001
            orphaned = True
        if on_step is not None:
            await on_step(t)
        if first_end is None and len(store.get_past_cycles()) > n_before:
            first_end = t
            if stop_at_first:
                break
    return {"first_end": first_end, "cycles": store.get_past_cycles()[n_before:],
            "orphaned": orphaned}


def _finished_at(t: float) -> float:
    return ACTIVE_W if t < QUIET_AT else 0.0


@pytest.mark.parametrize("plug", ["reporting", "silent"])
@pytest.mark.parametrize("device_type", ["washing_machine", "dishwasher"])
async def test_a_finished_cycle_ends_after_the_bounded_wait_not_the_force_stop(
    hass: Any, freezer: Any, monkeypatch: pytest.MonkeyPatch, device_type: str, plug: str
) -> None:
    mgr = await _boot(hass, device_type)
    state = _script_matcher(mgr, monkeypatch)
    out = await _drive(
        hass, freezer, mgr, plug, _finished_at, until_s=8.5 * 3600, stop_at_first=True
    )
    assert state["revoked"] and out["orphaned"], "the orphaned pause never formed"
    assert len(out["cycles"]) == 1, out
    cycle = out["cycles"][0]
    # Before item 498: force_stopped at 8 h (reporting) or by the 4.5 h staleness
    # limit (silent).
    assert cycle["status"] == "completed", (
        cycle["status"], cycle.get("termination_reason"), out["first_end"] / 3600
    )
    quiet = out["first_end"] - QUIET_AT
    assert WAIT_S <= quiet <= WAIT_S + 180, quiet


@pytest.mark.parametrize("plug", ["reporting", "silent"])
async def test_a_recorded_soak_right_after_the_revoke_is_not_split(
    hass: Any, freezer: Any, monkeypatch: pytest.MonkeyPatch, plug: str
) -> None:
    """The soak (1300 s) outlasts the floor (900 s, where clearing the pause on the
    revoke would have ended the cycle) but not 1.25 x the recorded 1200 s soak."""
    mgr = await _boot(hass, "washing_machine")
    state = _script_matcher(mgr, monkeypatch)
    resume_at = QUIET_AT + 1300

    def _soak(t: float) -> float:
        if t < QUIET_AT or resume_at <= t < resume_at + 1200:
            return ACTIVE_W
        return 0.0

    out = await _drive(hass, freezer, mgr, plug, _soak, until_s=resume_at + 1200 + 3 * 3600)
    assert state["revoked"] and out["orphaned"]
    assert len(out["cycles"]) == 1, [(c["status"], c["duration"]) for c in out["cycles"]]
    assert out["cycles"][0]["status"] == "completed"
    assert out["cycles"][0]["duration"] >= resume_at + 1200 - 60


async def test_a_user_pause_after_the_revoke_is_never_released(
    hass: Any, freezer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = await _boot(hass, "washing_machine")
    _script_matcher(mgr, monkeypatch)
    paused: dict[str, bool] = {}

    async def _pause(_t: float) -> None:
        det = mgr.detector
        if not paused and det._verified_pause and det.matched_profile is None:  # noqa: SLF001
            paused["done"] = await mgr.async_pause_cycle()

    out = await _drive(
        hass, freezer, mgr, "reporting", _finished_at,
        until_s=QUIET_AT + 4 * WAIT_S, on_step=_pause,
    )
    assert paused.get("done") is True
    assert out["cycles"] == []
    assert mgr.detector.state == STATE_ENDING
    assert mgr.detector._verified_pause  # noqa: SLF001
